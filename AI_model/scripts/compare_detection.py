"""Bandingkan model deteksi sebelum dan sesudah fine-tuning.

    python scripts/compare_detection.py
    python scripts/compare_detection.py --split test --iou-match 0.5

Menghasilkan reports/metrics/detection_comparison.json beserta gambar
pendukung di reports/figures. Seluruh angka berasal dari run yang benar-benar
dijalankan; tidak ada nilai yang diisi manual.
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
from visionqc_ai.evaluation.bootstrap import (  # noqa: E402
    bca_interval,
    proportion_interval,
)
from visionqc_ai.evaluation.detection_eval import (  # noqa: E402
    EvaluationRun,
    evaluate_detection,
)
from visionqc_ai.evaluation.matching import ImageMatch  # noqa: E402
from visionqc_ai.evaluation.metrics import (  # noqa: E402
    detection_flags,
    image_level_metrics,
    instance_metrics,
    per_class_recall,
    per_class_support,
)
from visionqc_ai.evaluation.significance import (  # noqa: E402
    cohens_h,
    interpret_cohens_h,
    mcnemar,
)

log = logging.getLogger("compare_detection")


def bootstrap_over_images(
    matches: list[ImageMatch], metric: str, *, seed: int, n_resamples: int
):
    """Bootstrap dengan gambar sebagai unit resampling.

    Instance cacat di dalam satu gambar tidak saling bebas, jadi yang diambil
    ulang adalah gambarnya dan bukan instance-nya. Mengabaikan hal ini membuat
    selang kepercayaan tampak lebih sempit daripada yang sebenarnya.
    """

    def statistic(indices: np.ndarray) -> float:
        subset = [matches[int(i)] for i in indices]
        result = instance_metrics(subset)
        return {
            "recall": result.recall,
            "precision": result.precision,
            "f1": result.f1,
            "f2": result.f2,
        }[metric]

    return bca_interval(
        np.arange(len(matches), dtype=float),
        statistic,
        n_resamples=n_resamples,
        seed=seed,
    )


def summarise(run: EvaluationRun, *, seed: int, n_resamples: int) -> dict[str, Any]:
    instance = instance_metrics(run.matches)
    agnostic = instance_metrics(run.matches_class_agnostic)
    image_level = image_level_metrics(run.matches)
    support = per_class_support(run.matches, len(DEFECT_CLASSES))
    recalls = per_class_recall(run.matches, len(DEFECT_CLASSES))

    intervals = {
        metric: bootstrap_over_images(
            run.matches, metric, seed=seed, n_resamples=n_resamples
        ).to_dict()
        for metric in ("recall", "precision", "f1", "f2")
    }
    wilson = proportion_interval(
        instance.true_positives, instance.true_positives + instance.false_negatives
    )

    return {
        **run.to_dict(),
        "instance": instance.to_dict(),
        "instance_class_agnostic": agnostic.to_dict(),
        "image_level": image_level.to_dict(),
        "confidence_intervals_bca": intervals,
        "recall_wilson": wilson.to_dict(),
        "per_class": {
            DEFECT_CLASSES[i]: {"support": support[i], "recall": recalls[i]}
            for i in range(len(DEFECT_CLASSES))
        },
    }


def plot_comparison(
    before: dict[str, Any], after: dict[str, Any], target: Path
) -> None:
    metrics = ["recall", "precision", "f1", "f2"]
    labels = ["Recall", "Precision", "F1", "F2"]
    positions = np.arange(len(metrics))
    width = 0.36

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axis = axes[0]
    for offset, (payload, name, colour) in enumerate(
        (
            (before, "Sebelum fine-tuning", "#b58900"),
            (after, "Sesudah fine-tuning", "#268bd2"),
        )
    ):
        values = [payload["confidence_intervals_bca"][m]["estimate"] for m in metrics]
        lower = [
            v - payload["confidence_intervals_bca"][m]["lower"]
            for v, m in zip(values, metrics, strict=True)
        ]
        upper = [
            payload["confidence_intervals_bca"][m]["upper"] - v
            for v, m in zip(values, metrics, strict=True)
        ]
        axis.bar(
            positions + (offset - 0.5) * width,
            values,
            width,
            label=name,
            color=colour,
            yerr=[np.clip(lower, 0, None), np.clip(upper, 0, None)],
            capsize=4,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("nilai")
    axis.set_title("Metrik tingkat instance (galat = 95% BCa)")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)

    axis = axes[1]
    names = list(after["per_class"].keys())
    supports = [after["per_class"][n]["support"] for n in names]
    before_recall = [before["per_class"][n]["recall"] for n in names]
    after_recall = [after["per_class"][n]["recall"] for n in names]
    positions = np.arange(len(names))
    axis.barh(positions + 0.18, before_recall, 0.36, color="#b58900", label="Sebelum")
    axis.barh(positions - 0.18, after_recall, 0.36, color="#268bd2", label="Sesudah")
    axis.set_yticks(positions)
    axis.set_yticklabels(
        [f"{n}\n(n={s})" for n, s in zip(names, supports, strict=True)], fontsize=8
    )
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("recall")
    axis.set_title("Recall per kelas pada test set")
    axis.legend(fontsize=8)
    axis.grid(axis="x", alpha=0.3)

    figure.suptitle("Bukti fine-tuning deteksi cacat VisionQC", fontsize=12)
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=120, bbox_inches="tight")
    plt.close(figure)


def plot_training_curves(results_csv: Path, target: Path) -> bool:
    if not results_csv.exists():
        return False
    rows = [
        r.strip()
        for r in results_csv.read_text(encoding="utf-8").splitlines()
        if r.strip()
    ]
    header = [h.strip() for h in rows[0].split(",")]
    data = np.array([[float(v) for v in r.split(",")] for r in rows[1:]])
    if data.size == 0:
        return False

    def column(name: str) -> np.ndarray | None:
        return data[:, header.index(name)] if name in header else None

    epochs = column("epoch")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, label in (
        ("train/box_loss", "box"),
        ("train/cls_loss", "cls"),
        ("train/dfl_loss", "dfl"),
    ):
        values = column(name)
        if values is not None:
            axes[0].plot(epochs, values, label=label)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Kurva galat pelatihan")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for name, label in (
        ("metrics/mAP50(B)", "mAP50"),
        ("metrics/mAP50-95(B)", "mAP50-95"),
        ("metrics/recall(B)", "recall"),
    ):
        values = column(name)
        if values is not None:
            axes[1].plot(epochs, values, label=label)
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Metrik pada split validasi")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Perbandingan deteksi sebelum vs sesudah"
    )
    parser.add_argument("--baseline", type=Path, default=PROJECT_ROOT / "yolo11n.pt")
    parser.add_argument(
        "--finetuned",
        type=Path,
        default=PROJECT_ROOT / "models/finetuned/detect_goodsad/weights/best.pt",
    )
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data/processed/detect"
    )
    parser.add_argument(
        "--inference-config", type=Path, default=PROJECT_ROOT / "configs/inference.yaml"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--iou-match", type=float, default=0.5)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=0,
        help="0 untuk GPU pertama, cpu untuk memaksa CPU",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if not args.finetuned.exists():
        log.error("bobot hasil fine-tuning belum ada: %s", args.finetuned)
        return 1

    with args.inference_config.open(encoding="utf-8") as handle:
        inference = yaml.safe_load(handle)["models"]["detection"]

    common = {
        "device": args.device,
        "dataset_root": args.dataset,
        "split": args.split,
        "imgsz": int(inference["imgsz"]),
        "conf": float(inference["conf_threshold"]),
        "iou_nms": float(inference["iou_threshold"]),
        "iou_match": args.iou_match,
        "max_detections": int(inference["max_detections"]),
    }

    log.info(
        "Evaluasi pada split %s, ambang IoU pencocokan %.2f", args.split, args.iou_match
    )
    log.info("[1/3] Baseline pra-latih: %s", args.baseline.name)
    baseline_run = evaluate_detection(args.baseline, tag="baseline", **common)
    log.info("[2/3] Hasil fine-tuning : %s", args.finetuned)
    finetuned_run = evaluate_detection(args.finetuned, tag="finetuned", **common)

    log.info("[3/3] Statistik pembanding")
    before = summarise(baseline_run, seed=args.seed, n_resamples=args.resamples)
    after = summarise(finetuned_run, seed=args.seed, n_resamples=args.resamples)

    flags_before = detection_flags(baseline_run.matches)
    flags_after = detection_flags(finetuned_run.matches)
    test = mcnemar(flags_before, flags_after)
    effect = cohens_h(after["instance"]["recall"], before["instance"]["recall"])

    figures = PROJECT_ROOT / "reports/figures"
    plot_comparison(before, after, figures / "detection_comparison.png")
    curves = plot_training_curves(
        args.finetuned.parent.parent / "results.csv", figures / "training_curves.png"
    )
    copied = []
    for source_name, target_name in (
        ("confusion_matrix_normalized.png", "detection_confusion_matrix.png"),
        ("BoxPR_curve.png", "detection_pr_curve.png"),
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
        "figures": ["detection_comparison.png"]
        + (["training_curves.png"] if curves else [])
        + copied,
    }
    target = PROJECT_ROOT / "reports/metrics/detection_comparison.json"
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("")
    log.info("%-26s %12s %12s", "metrik (instance)", "sebelum", "sesudah")
    for key in ("recall", "precision", "f1", "f2"):
        log.info(
            "%-26s %12.4f %12.4f",
            key,
            before["instance"][key],
            after["instance"][key],
        )
    log.info(
        "%-26s %12.4f %12.4f",
        "MCC (tingkat gambar)",
        before["image_level"]["mcc"],
        after["image_level"]["mcc"],
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
