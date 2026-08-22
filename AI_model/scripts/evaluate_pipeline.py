"""Evaluasi ujung-ke-ujung keputusan PASS/REJECT lewat pipeline sungguhan.

    python scripts/evaluate_pipeline.py

Berbeda dari compare_detection.py yang mengukur kualitas kotak pembatas,
skrip ini mengukur hal yang benar-benar dilihat pengguna: apakah satu gambar
produk diputuskan PASS atau REJECT. Jalannya melewati InspectionPipeline,
yaitu ONNX beserta lapisan privasi, anomali, dan decision engine, bukan bobot
PyTorch. Angka dari sini yang sah dipakai menjawab "seberapa benar sistem
memisahkan produk cacat dari produk bagus".

Gambar normal tambahan diambil dengan fungsi dan seed yang sama dengan
calibrate_decision.py, sehingga tidak ada gambar latih yang bocor ke sini.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from calibrate_decision import detect_categories, gather_extra_normals  # noqa: E402
from visionqc_ai import load_pipeline  # noqa: E402

log = logging.getLogger("evaluate_pipeline")

REJECT = "REJECT"
GROUPS = {
    "lama (MVTec + VisA)": ("mvtec_", "visa_"),
    "baru (PKU-GoodsAD)": ("goodsad_",),
}


def group_of(image_id: str) -> str:
    """Kelompokkan gambar menurut dataset asalnya."""
    for name, prefixes in GROUPS.items():
        if image_id.startswith(prefixes):
            return name
    return "lainnya"


def labelled_images(dataset_root: Path, split: str) -> list[tuple[Path, bool]]:
    """Gambar satu split beserta label cacat, dibaca dari berkas label YOLO."""
    items: list[tuple[Path, bool]] = []
    for image in sorted((dataset_root / "images" / split).glob("*.jpg")):
        label = dataset_root / "labels" / split / f"{image.stem}.txt"
        is_defect = label.is_file() and bool(label.read_text(encoding="utf-8").strip())
        items.append((image, is_defect))
    return items


def _fmt(value: float | None) -> str:
    """Tampilkan angka, atau n/a bila pembaginya nol.

    Menampilkan 0,0000 untuk kasus tanpa pembagi akan terbaca sebagai kinerja
    terburuk, padahal artinya tidak ada data untuk dinilai.
    """
    return "n/a" if value is None else f"{value:.4f}"


def summarise(rows: list[tuple[str, bool, str]]) -> dict[str, object]:
    """Hitung matriks kebingungan biner dari daftar hasil."""
    tp = sum(1 for _, truth, verdict in rows if truth and verdict == REJECT)
    fn = sum(1 for _, truth, verdict in rows if truth and verdict != REJECT)
    fp = sum(1 for _, truth, verdict in rows if not truth and verdict == REJECT)
    tn = sum(1 for _, truth, verdict in rows if not truth and verdict != REJECT)
    total = tp + fn + fp + tn
    return {
        "n": total,
        "defects": tp + fn,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "accuracy": (tp + tn) / total if total else None,
        "verdicts": dict(Counter(verdict for _, _, verdict in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluasi ujung-ke-ujung pipeline")
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data/processed/detect"
    )
    parser.add_argument(
        "--dataset-config", type=Path, default=PROJECT_ROOT / "configs/dataset.yaml"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("val", "test"),
        help="val adalah set kalibrasi; keputusan konfigurasi wajib diambil di sana",
    )
    parser.add_argument("--extra-normals", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--limit", type=int, default=0, help="batasi jumlah gambar untuk uji cepat"
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    pipeline = load_pipeline()
    if not pipeline.ready:
        log.error("pipeline tidak siap; jalankan scripts/export_onnx.py lebih dulu")
        return 1
    log.info("Evaluasi ujung-ke-ujung pipeline")
    log.info("split            : %s", args.split)
    log.info("anomali aktif    : %s", pipeline.anomaly_available)
    log.info("peredaman wajah  : %s", pipeline.face_blur_available)

    items = labelled_images(args.dataset, args.split)
    calib_normals, test_normals = gather_extra_normals(
        PROJECT_ROOT,
        args.dataset,
        per_side=args.extra_normals,
        seed=args.seed,
        categories=detect_categories(args.dataset_config),
    )
    # Sisi normal tambahan harus mengikuti split yang dinilai, kalau tidak
    # gambar yang sama dipakai untuk memilih konfigurasi dan untuk melaporkan.
    extra_normals = calib_normals if args.split == "val" else test_normals
    items += [(path, False) for path in extra_normals]
    if args.limit:
        items = items[: args.limit]
    log.info(
        "gambar dinilai   : %d (%d cacat)",
        len(items),
        sum(1 for _, truth in items if truth),
    )

    rows: list[tuple[str, bool, str]] = []
    for index, (path, truth) in enumerate(items, start=1):
        result = pipeline.inspect(path.read_bytes())
        rows.append((path.stem, truth, str(result.verdict)))
        if index % 100 == 0:
            log.info("  %d/%d", index, len(items))

    overall = summarise(rows)
    per_group = {
        name: summarise([r for r in rows if group_of(r[0]) == name])
        for name in GROUPS
        if any(group_of(r[0]) == name for r in rows)
    }

    log.info("")
    header = f"{'kelompok':<22}{'n':>6}{'cacat':>7}{'recall':>9}{'specificity':>13}"
    log.info(header)
    log.info("-" * len(header))
    for name, stats in per_group.items():
        log.info(
            "%-22s%6d%7d%9s%13s",
            name,
            stats["n"],
            stats["defects"],
            _fmt(stats["recall"]),
            _fmt(stats["specificity"]),
        )
    log.info("-" * len(header))
    log.info(
        "%-22s%6d%7d%9s%13s",
        "GABUNGAN",
        overall["n"],
        overall["defects"],
        _fmt(overall["recall"]),
        _fmt(overall["specificity"]),
    )
    log.info("")
    log.info("akurasi          : %s", _fmt(overall["accuracy"]))
    log.info("sebaran keputusan: %s", overall["verdicts"])

    suffix = "" if args.split == "test" else f"_{args.split}"
    target = PROJECT_ROOT / "reports" / "metrics" / f"pipeline_evaluation{suffix}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_on": date.today().isoformat(),
        "split": args.split,
        "extra_normals_per_side": args.extra_normals,
        "seed": args.seed,
        "overall": overall,
        "per_group": per_group,
    }
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    log.info("Hasil ditulis ke %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
