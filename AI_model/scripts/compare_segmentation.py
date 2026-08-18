"""Bandingkan model segmentasi sebelum dan sesudah fine-tuning.

    python scripts/compare_segmentation.py

Menghasilkan reports/metrics/segmentation_comparison.json beserta gambar
pendukung. Selain metrik pencocokan, dilaporkan pula galat estimasi luas cacat
karena angka itulah yang dipakai decision engine.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from visionqc_ai.data.taxonomy import DEFECT_CLASSES  # noqa: E402
from visionqc_ai.evaluation.bootstrap import bca_interval  # noqa: E402
from visionqc_ai.evaluation.metrics import (  # noqa: E402
    detection_flags,
    image_level_metrics,
    instance_metrics,
    per_class_recall,
    per_class_support,
)
from visionqc_ai.evaluation.segmentation_eval import (  # noqa: E402
    SegmentationRun,
    evaluate_segmentation,
)
from visionqc_ai.evaluation.significance import (  # noqa: E402
    cohens_h,
    interpret_cohens_h,
    mcnemar,
)

log = logging.getLogger("compare_segmentation")


def summarise(run: SegmentationRun, *, seed: int, n_resamples: int) -> dict[str, Any]:
    instance = instance_metrics(run.matches)
    image_level = image_level_metrics(run.matches)
    support = per_class_support(run.matches, len(DEFECT_CLASSES))
    recalls = per_class_recall(run.matches, len(DEFECT_CLASSES))

    def metric_ci(name: str):
        def statistic(indices: np.ndarray) -> float:
            subset = [run.matches[int(i)] for i in indices]
            result = instance_metrics(subset)
            return {
                "recall": result.recall,
                "precision": result.precision,
                "f2": result.f2,
            }[name]

        return bca_interval(
            np.arange(len(run.matches), dtype=float),
            statistic,
            n_resamples=n_resamples,
            seed=seed,
        ).to_dict()

    errors = [a.absolute_error for a in run.areas]
    mask_iou_mean = float(np.mean(run.mask_ious)) if run.mask_ious else 0.0
    area_ci = (
        bca_interval(
            errors, lambda d: float(np.mean(d)), n_resamples=n_resamples, seed=seed
        ).to_dict()
        if errors
        else None
    )

    return {
        **run.to_dict(),
        "instance": instance.to_dict(),
        "image_level": image_level.to_dict(),
        "confidence_intervals_bca": {
            n: metric_ci(n) for n in ("recall", "precision", "f2")
        },
        "mask_iou": {
            "mean": mask_iou_mean,
            "median": float(np.median(run.mask_ious)) if run.mask_ious else 0.0,
            "matched_count": len(run.mask_ious),
        },
        "area_error_pct": {
            "mean_absolute": float(np.mean(errors)) if errors else 0.0,
            "median_absolute": float(np.median(errors)) if errors else 0.0,
            "p95_absolute": float(np.percentile(errors, 95)) if errors else 0.0,
            "bca_mean_absolute": area_ci,
        },
        "per_class": {
            DEFECT_CLASSES[i]: {"support": support[i], "recall": recalls[i]}
            for i in range(len(DEFECT_CLASSES))
        },
    }


def plot_segmentation(
    before: dict[str, Any], after: dict[str, Any], run: SegmentationRun, target: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    metrics = ["recall", "precision", "f2"]
    positions = np.arange(len(metrics))
    width = 0.36
    for offset, (payload, name, colour) in enumerate(
        (
            (before, "Sebelum fine-tuning", "#b58900"),
            (after, "Sesudah fine-tuning", "#268bd2"),
        )
    ):
        values = [payload["confidence_intervals_bca"][m]["estimate"] for m in metrics]
        lower = [
            max(0.0, v - payload["confidence_intervals_bca"][m]["lower"])
            for v, m in zip(values, metrics, strict=True)
        ]
        upper = [
            max(0.0, payload["confidence_intervals_bca"][m]["upper"] - v)
            for v, m in zip(values, metrics, strict=True)
        ]
        axes[0].bar(
            positions + (offset - 0.5) * width,
            values,
            width,
            label=name,
            color=colour,
            yerr=[lower, upper],
            capsize=4,
        )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(["Recall", "Precision", "F2"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Metrik mask (galat = 95% BCa)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    if run.mask_ious:
        axes[1].hist(run.mask_ious, bins=16, range=(0.5, 1.0), color="#268bd2")
        axes[1].axvline(
            float(np.mean(run.mask_ious)),
            color="#dc322f",
            linestyle="--",
            label=f"rerata {np.mean(run.mask_ious):.3f}",
        )
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("IoU mask pada pasangan yang cocok")
    axes[1].set_ylabel("jumlah")
    axes[1].set_title("Sebaran IoU mask")
    axes[1].grid(alpha=0.3)

    truth = [a.truth_pct for a in run.areas]
    predicted = [a.predicted_pct for a in run.areas]
    limit = max(truth + predicted + [1.0]) * 1.05
    axes[2].scatter(truth, predicted, s=18, alpha=0.7, color="#268bd2")
    axes[2].plot([0, limit], [0, limit], color="#586e75", linewidth=1, linestyle="--")
    axes[2].set_xlim(0, limit)
    axes[2].set_ylim(0, limit)
    axes[2].set_xlabel("luas cacat sebenarnya (%)")
    axes[2].set_ylabel("luas cacat dugaan (%)")
    axes[2].set_title("Ketepatan estimasi luas cacat")
    axes[2].grid(alpha=0.3)

    figure.suptitle("Bukti fine-tuning segmentasi cacat VisionQC", fontsize=12)
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=120, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Perbandingan segmentasi sebelum dan sesudah fine-tuning"
    )
    parser.add_argument(
        "--baseline", type=Path, default=PROJECT_ROOT / "yolo11n-seg.pt"
    )
    parser.add_argument(
        "--finetuned",
        type=Path,
        default=PROJECT_ROOT / "models/finetuned/seg/weights/best.pt",
    )
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data/processed/seg"
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=PROJECT_ROOT / "configs/inference.yaml",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--iou-match", type=float, default=0.5)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if not args.finetuned.exists():
        log.error("bobot hasil fine-tuning belum ada: %s", args.finetuned)
        return 1

    with args.inference_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)["models"]
    detection = config["detection"]
    segmentation = config["segmentation"]

    common = {
        "dataset_root": args.dataset,
        "split": args.split,
        "imgsz": int(segmentation["imgsz"]),
        "conf": float(detection["conf_threshold"]),
        "iou_nms": float(detection["iou_threshold"]),
        "iou_match": args.iou_match,
        "max_detections": int(detection["max_detections"]),
    }

    log.info(
        "Evaluasi segmentasi pada split %s, IoU mask %.2f", args.split, args.iou_match
    )
    log.info("[1/3] Baseline pra-latih: %s", args.baseline.name)
    baseline_run = evaluate_segmentation(args.baseline, tag="baseline", **common)
    log.info("[2/3] Hasil fine-tuning : %s", args.finetuned)
    finetuned_run = evaluate_segmentation(args.finetuned, tag="finetuned", **common)

    log.info("[3/3] Statistik pembanding")
    before = summarise(baseline_run, seed=args.seed, n_resamples=args.resamples)
    after = summarise(finetuned_run, seed=args.seed, n_resamples=args.resamples)

    test = mcnemar(
        detection_flags(baseline_run.matches), detection_flags(finetuned_run.matches)
    )
    effect = cohens_h(after["instance"]["recall"], before["instance"]["recall"])

    figures = PROJECT_ROOT / "reports/figures"
    plot_segmentation(
        before, after, finetuned_run, figures / "segmentation_comparison.png"
    )
    copied = []
    for source_name, target_name in (
        ("MaskPR_curve.png", "segmentation_pr_curve.png"),
        ("confusion_matrix_normalized.png", "segmentation_confusion_matrix.png"),
    ):
        source = args.finetuned.parent.parent / source_name
        if source.exists():
            shutil.copy2(source, figures / target_name)
            copied.append(target_name)

    payload = {
        "generated_on": date.today().isoformat(),
        "split": args.split,
        "iou_match": args.iou_match,
        "prediction_settings": common | {"dataset_root": str(args.dataset)},
        "before": before,
        "after": after,
        "mcnemar": test.to_dict(),
        "cohens_h": {"value": effect, "interpretation": interpret_cohens_h(effect)},
        "figures": ["segmentation_comparison.png", *copied],
    }
    target = PROJECT_ROOT / "reports/metrics/segmentation_comparison.json"
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("")
    log.info("%-26s %12s %12s", "metrik mask", "sebelum", "sesudah")
    for key in ("recall", "precision", "f2"):
        log.info(
            "%-26s %12.4f %12.4f", key, before["instance"][key], after["instance"][key]
        )
    log.info(
        "%-26s %12.4f %12.4f",
        "IoU mask rerata",
        before["mask_iou"]["mean"],
        after["mask_iou"]["mean"],
    )
    log.info(
        "%-26s %12.4f %12.4f",
        "galat luas rerata (%)",
        before["area_error_pct"]["mean_absolute"],
        after["area_error_pct"]["mean_absolute"],
    )
    log.info("")
    log.info(
        "McNemar: b=%d c=%d, %s, p=%.3g -> %s",
        test.only_first_correct,
        test.only_second_correct,
        test.method,
        test.p_value,
        "signifikan" if test.significant else "tidak signifikan",
    )
    log.info("Cohen's h = %.3f (%s)", effect, interpret_cohens_h(effect))
    log.info("Hasil lengkap: %s", target.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
