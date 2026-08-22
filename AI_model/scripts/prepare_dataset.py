"""Preprocessing dataset: konversi, split, validasi, dan penulisan.

    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --dry-run

Seluruh angka hasil pengukuran ditulis ke reports/metrics/dataset_stats.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionqc_ai.data import convert_goodsad, convert_mvtec, convert_visa  # noqa: E402
from visionqc_ai.data.records import ConversionResult, SampleRecord  # noqa: E402
from visionqc_ai.data.sources import GOODSAD, MVTEC, get_category  # noqa: E402
from visionqc_ai.data.split import (  # noqa: E402
    SPLIT_NAMES,
    SplitAssignment,
    SplitRatios,
    stratified_split,
)
from visionqc_ai.data.taxonomy import (  # noqa: E402
    CLASS_IDS,
    CLASS_SEVERITY,
    DEFECT_CLASSES,
)
from visionqc_ai.data.validate import validate_dataset  # noqa: E402
from visionqc_ai.data.visa_codes import VisaCodeMap  # noqa: E402
from visionqc_ai.data.writer import (  # noqa: E402
    DETECT,
    SEGMENT,
    write_anomaly_dataset,
    write_yolo_dataset,
)

log = logging.getLogger("prepare_dataset")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def convert_all(
    categories: list[str], raw_root: Path, preprocess: dict[str, Any]
) -> tuple[dict[str, ConversionResult], dict[str, VisaCodeMap]]:
    """Konversi setiap kategori sesuai dataset asalnya."""
    kwargs = {
        "imgsz": preprocess["imgsz"],
        "min_component_area_px": preprocess["min_component_area_px"],
        "min_scaled_side_px": preprocess["min_scaled_side_px"],
        "polygon_epsilon_ratio": preprocess["polygon_epsilon_ratio"],
        "polygon_max_points": preprocess["polygon_max_points"],
    }
    results: dict[str, ConversionResult] = {}
    code_maps: dict[str, VisaCodeMap] = {}

    for name in categories:
        spec = get_category(name)
        if spec.source == MVTEC:
            result = convert_mvtec.convert_category(spec, raw_root, **kwargs)
        elif spec.source == GOODSAD:
            result = convert_goodsad.convert_category(spec, raw_root, **kwargs)
        else:
            result, code_map = convert_visa.convert_category(spec, raw_root, **kwargs)
            code_maps[name] = code_map
            log.info("  %s", code_map.summary())
        results[name] = result
        log.info(
            "  %-12s cacat=%-4d normal=%-4d instance=%-5d "
            "dibuang(kecil=%d, tak-terpetakan=%d)",
            name,
            len(result.defects),
            len(result.normals),
            result.stats.kept,
            result.stats.dropped_small_area + result.stats.dropped_small_side,
            result.stats.dropped_unmapped,
        )
    return results, code_maps


def split_category(
    result: ConversionResult, *, seed: int, ratios: SplitRatios
) -> tuple[SplitAssignment, SplitAssignment]:
    """Bagi record cacat dan record normal dengan seed yang sama.

    Dipisah agar dataset deteksi dan anomali menarik dari pembagian identik,
    sehingga gambar latar pada split uji deteksi tidak muncul sebagai data
    latih anomali.
    """
    defects = stratified_split(result.defects, seed=seed, ratios=ratios)
    normals = stratified_split(result.normals, seed=seed, ratios=ratios)
    return defects, normals


def build_detect_assignment(
    per_category: dict[str, tuple[SplitAssignment, SplitAssignment]],
    categories: list[str],
    *,
    seed: int,
    background_ratio: float,
) -> SplitAssignment:
    """Gabungkan kategori menjadi satu dataset deteksi, plus gambar latar."""
    buckets: dict[str, list[SampleRecord]] = {name: [] for name in SPLIT_NAMES}
    for category in categories:
        defects, normals = per_category[category]
        for split in SPLIT_NAMES:
            defect_records = getattr(defects, split)
            normal_records = getattr(normals, split)
            quota = min(
                len(normal_records), round(len(defect_records) * background_ratio)
            )
            chosen = random.Random(f"{seed}:bg:{category}:{split}").sample(
                sorted(normal_records, key=lambda r: r.sample_id), quota
            )
            buckets[split].extend(defect_records)
            buckets[split].extend(chosen)
    return SplitAssignment(
        train=buckets["train"], val=buckets["val"], test=buckets["test"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocessing dataset VisionQC")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/dataset.yaml"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="hanya konversi, split, dan validasi tanpa menulis gambar",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    config = load_config(args.config)

    raw_root = PROJECT_ROOT / config["paths"]["raw"]
    processed_root = PROJECT_ROOT / config["paths"]["processed"]
    reports_root = PROJECT_ROOT / config["paths"]["reports"]
    preprocess = config["preprocess"]
    split_cfg = config["split"]
    validation_cfg = config["validation"]

    detect_categories: list[str] = config["categories"]["detect"]
    anomaly_categories: list[str] = config["categories"]["anomaly"]
    all_categories = sorted(set(detect_categories) | set(anomaly_categories))
    ratios = SplitRatios(split_cfg["train"], split_cfg["val"], split_cfg["test"])
    seed = int(split_cfg["seed"])

    log.info("VisionQC - Step 2: preprocessing dataset")
    log.info("konfigurasi : %s", args.config.name)
    log.info(
        "kategori    : deteksi=%s | anomali=%s", detect_categories, anomaly_categories
    )
    log.info("split       : %s (seed %d, dikunci)", ratios.as_tuple(), seed)
    log.info("")
    log.info("[1/4] Konversi dataset mentah")
    conversions, code_maps = convert_all(all_categories, raw_root, preprocess)

    log.info("")
    log.info("[2/4] Pembagian split terstratifikasi")
    per_category = {
        name: split_category(result, seed=seed, ratios=ratios)
        for name, result in conversions.items()
    }
    detect_assignment = build_detect_assignment(
        per_category,
        detect_categories,
        seed=seed,
        background_ratio=preprocess["background_ratio"],
    )
    for split in SPLIT_NAMES:
        records = getattr(detect_assignment, split)
        defect_count = sum(1 for r in records if not r.is_normal)
        log.info(
            "  %-5s %4d gambar (%d cacat + %d latar), %d instance",
            split,
            len(records),
            defect_count,
            len(records) - defect_count,
            sum(len(r.instances) for r in records),
        )

    log.info("")
    log.info("[3/4] Validasi statistik")
    validation = validate_dataset(
        detect_assignment,
        DEFECT_CLASSES,
        chi2_alpha=validation_cfg["chi2_alpha"],
        max_imbalance_ratio=validation_cfg["max_imbalance_ratio"],
        min_normalized_entropy=validation_cfg["min_normalized_entropy"],
        z=validation_cfg["sample_size"]["z"],
        expected_recall=validation_cfg["sample_size"]["expected_recall"],
        margin_of_error=validation_cfg["sample_size"]["margin_of_error"],
    )
    chi = validation.chi_square
    log.info(
        "  khi-kuadrat  : chi2=%.3f  p=%.4f  df=%d  -> %s %s",
        chi.statistic or 0.0,
        chi.p_value or 0.0,
        chi.degrees_of_freedom or 0,
        "LULUS" if chi.passed else "TIDAK LULUS",
        f"({chi.note})" if chi.note else "",
    )
    log.info("  jumlah instance per kelas: %s", validation.balance.counts)
    log.info(
        "  rasio ketimpangan: %.2f (target <= %.1f) -> %s",
        validation.balance.imbalance_ratio or 0.0,
        validation_cfg["max_imbalance_ratio"],
        "LULUS" if validation.balance.ir_passed else "TIDAK LULUS",
    )
    log.info(
        "  entropi ternormalisasi: %.4f (target >= %.2f) -> %s",
        validation.balance.normalized_entropy or 0.0,
        validation_cfg["min_normalized_entropy"],
        "LULUS" if validation.balance.entropy_passed else "TIDAK LULUS",
    )
    size = validation.sample_size
    log.info(
        "  ukuran test  : %d instance cacat pada %d gambar "
        "(minimum %d untuk galat +-%.0f%%) -> %s",
        size.actual_instances,
        size.actual_defect_images,
        size.required_instances,
        size.parameters["E"] * 100,
        "LULUS" if size.passed else "TIDAK LULUS",
    )
    log.info(
        "                 galat estimasi recall yang tercapai: +-%.1f%% (95%% CI)",
        (size.achieved_margin_of_error or 0.0) * 100,
    )

    log.info("")
    log.info("[4/4] Penulisan dataset")
    written: dict[str, Any] = {}
    if args.dry_run:
        log.info("  --dry-run: penulisan gambar dilewati")
    else:
        detect_root = processed_root / DETECT
        written[DETECT] = write_yolo_dataset(
            detect_assignment,
            detect_root,
            DEFECT_CLASSES,
            CLASS_IDS,
            task=DETECT,
            imgsz=preprocess["imgsz"],
            jpeg_quality=preprocess["jpeg_quality"],
            notes=f"kategori: {', '.join(detect_categories)}",
        )
        log.info("  detect : %s", written[DETECT])

        written[SEGMENT] = write_yolo_dataset(
            detect_assignment,
            processed_root / SEGMENT,
            DEFECT_CLASSES,
            CLASS_IDS,
            task=SEGMENT,
            imgsz=preprocess["imgsz"],
            jpeg_quality=preprocess["jpeg_quality"],
            notes=f"kategori: {', '.join(detect_categories)}",
            mirror_from=detect_root,
        )
        log.info("  seg    : %s", written[SEGMENT])

        anomaly_counts: dict[str, dict[str, int]] = {}
        for category in anomaly_categories:
            defects, normals = per_category[category]
            anomaly_counts[category] = write_anomaly_dataset(
                processed_root / "anomaly",
                category,
                normals=normals.as_dict(),
                defects=defects.as_dict(),
                imgsz=preprocess["imgsz"],
                jpeg_quality=preprocess["jpeg_quality"],
            )
            log.info("  anomaly/%-11s %s", category, anomaly_counts[category])
        written["anomaly"] = anomaly_counts

    stats: dict[str, Any] = {
        "generated_on": date.today().isoformat(),
        "step": "2 - akuisisi & preprocessing dataset",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "config": {
            "seed": seed,
            "ratios": {"train": ratios.train, "val": ratios.val, "test": ratios.test},
            "preprocess": preprocess,
            "categories": config["categories"],
        },
        "taxonomy": {
            "classes": list(DEFECT_CLASSES),
            "class_ids": CLASS_IDS,
            "severity_weight": CLASS_SEVERITY,
        },
        "visa_code_reconstruction": {
            name: {
                "code_to_label": code_map.code_to_label,
                "code_to_class": code_map.code_to_class,
                "class_merged_codes": {
                    str(k): list(v) for k, v in code_map.ambiguous.items()
                },
                "learned_codes": code_map.learned_codes,
                "inferred_codes": code_map.inferred_codes,
                "validated_multilabel_images": code_map.validated_images,
            }
            for name, code_map in code_maps.items()
        },
        "per_category": {
            name: {
                "defect_images": len(result.defects),
                "normal_images": len(result.normals),
                "instances_kept": result.stats.kept,
                "dropped_small_area": result.stats.dropped_small_area,
                "dropped_small_side": result.stats.dropped_small_side,
                "dropped_unmapped": result.stats.dropped_unmapped,
                "dropped_no_polygon": result.stats.dropped_no_polygon,
                "missing_masks": result.missing_masks,
                "learnable_at_640_pct": get_category(name).learnable_at_640_pct,
            }
            for name, result in conversions.items()
        },
        "validation": validation.to_dict(),
        "written": written,
    }
    reports_root.mkdir(parents=True, exist_ok=True)
    output = reports_root / "dataset_stats.json"
    output.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("")
    log.info("Statistik ditulis ke %s", output.relative_to(PROJECT_ROOT))
    log.info(
        "Status validasi keseluruhan: %s",
        "SEMUA LULUS"
        if validation.all_passed
        else "ADA YANG BELUM LULUS (lihat di atas)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
