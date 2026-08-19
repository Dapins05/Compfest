"""Kalibrasi keyakinan, conformal prediction, dan ambang sensitif biaya.

    python scripts/calibrate_decision.py

Split val dipakai sebagai set kalibrasi dan split test hanya dipakai untuk
menguji. Seluruh nilai yang dihasilkan bersifat statis dan disalin ke
configs/inference.yaml, tidak pernah dihitung ulang saat sistem berjalan.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from visionqc_ai.evaluation.detection_eval import image_level_scores  # noqa: E402
from visionqc_ai.statistics.calibration import (  # noqa: E402
    apply_platt,
    apply_temperature,
    calibration_report,
    fit_platt,
    fit_temperature,
    reliability_bins,
)
from visionqc_ai.statistics.conformal import (  # noqa: E402
    calibrate,
    evaluate_coverage,
)
from visionqc_ai.statistics.cost_sensitive import (  # noqa: E402
    CostModel,
    cost_curve,
    evaluate_threshold,
    optimal_threshold,
    savings_versus,
)

log = logging.getLogger("calibrate_decision")


def gather_extra_normals(
    project_root: Path, dataset_root: Path, *, per_side: int, seed: int
) -> tuple[list[Path], list[Path]]:
    """Kumpulkan gambar normal tambahan untuk kalibrasi dan pengujian.

    Split deteksi dibangun dari gambar cacat dan hanya menyertakan sedikit
    gambar latar, sehingga contoh normalnya terlalu sedikit untuk menghitung
    kuantil conformal maupun kalibrasi yang dapat dipercaya.

    Gambar yang sudah dipakai split deteksi mana pun dikecualikan, sehingga
    yang tersisa dijamin tidak pernah dilihat model saat pelatihan dan tidak
    tumpang tindih dengan data uji yang sudah dilaporkan. Kategori diambil dari
    yang memang dilatih detektor; menyertakan kategori lain berarti menguji
    model pada produk yang tidak pernah dipelajarinya.
    """
    used = {
        path.stem
        for split in ("train", "val", "test")
        for path in (dataset_root / "images" / split).glob("*.jpg")
    }
    categories = ("bottle", "chewinggum", "cashew", "pipe_fryum")

    pool: list[Path] = []
    anomaly_root = project_root / "data" / "processed" / "anomaly"
    for category in categories:
        for subset in ("train/good", "test/good"):
            pool += [
                path
                for path in sorted((anomaly_root / category / subset).glob("*.jpg"))
                if path.stem not in used
            ]

    random.Random(seed).shuffle(pool)
    chosen = pool[: per_side * 2]
    return chosen[:per_side], chosen[per_side : per_side * 2]


def plot_reliability(before, after, target: Path) -> None:
    """Diagram keandalan sebelum dan sesudah temperature scaling."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for axis, (bins, title) in zip(
        axes,
        ((before, "Sebelum kalibrasi"), (after, "Sesudah kalibrasi")),
        strict=True,
    ):
        centres = [(b.lower + b.upper) / 2 for b in bins]
        accuracy = [b.empirical_accuracy for b in bins]
        counts = [b.count for b in bins]
        axis.plot([0, 1], [0, 1], color="#586e75", linestyle="--", linewidth=1)
        axis.bar(
            centres,
            accuracy,
            width=0.09,
            color="#268bd2",
            edgecolor="#073642",
            linewidth=0.5,
        )
        for centre, count in zip(centres, counts, strict=True):
            if count:
                axis.text(centre, 0.02, str(count), ha="center", fontsize=7)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.05)
        axis.set_xlabel("keyakinan model")
        axis.set_ylabel("ketepatan sebenarnya")
        axis.set_title(title)
        axis.grid(alpha=0.3)
    figure.suptitle("Diagram keandalan keputusan cacat", fontsize=12)
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=120, bbox_inches="tight")
    plt.close(figure)


def plot_cost(curve, bayes: float, chosen: float, target: Path) -> None:
    """Kurva biaya total terhadap ambang keputusan."""
    thresholds = [p.threshold for p in curve]
    costs = [p.cost_per_item for p in curve]
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(thresholds, costs, color="#268bd2", linewidth=1.8)
    axis.axvline(
        bayes, color="#859900", linestyle=":", linewidth=1.6, label=f"Bayes {bayes:.4f}"
    )
    axis.axvline(
        chosen,
        color="#dc322f",
        linestyle="--",
        linewidth=1.6,
        label=f"minimum empiris {chosen:.4f}",
    )
    axis.set_xlabel("ambang keputusan")
    axis.set_ylabel("biaya per unit produk (Rp)")
    axis.set_title("Biaya total terhadap ambang keputusan")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=120, bbox_inches="tight")
    plt.close(figure)


def plot_coverage(report, target: Path) -> None:
    """Cakupan conformal beserta komposisi ukuran himpunan prediksi."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["keseluruhan"] + [f"kelas {k}" for k in sorted(report.per_class_coverage)]
    values = [report.empirical_coverage] + [
        report.per_class_coverage[k] for k in sorted(report.per_class_coverage)
    ]
    axes[0].bar(labels, values, color="#268bd2")
    axes[0].axhline(
        report.target_coverage,
        color="#dc322f",
        linestyle="--",
        linewidth=1.6,
        label=f"jaminan {report.target_coverage:.2f}",
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("cakupan empiris")
    axes[0].set_title("Cakupan pada data uji")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(
        ["keputusan diambil", "diserahkan ke manusia"],
        [report.singleton_rate, report.review_rate],
        color=["#268bd2", "#b58900"],
    )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("proporsi gambar")
    axes[1].set_title(
        "Komposisi keputusan (rerata ukuran himpunan "
        + f"{report.average_set_size:.2f})".replace(".", ",")
    )
    axes[1].grid(axis="y", alpha=0.3)

    figure.suptitle("Conformal prediction sebagai landasan kelas REVIEW", fontsize=12)
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=120, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Kalibrasi dan lapisan statistik")
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "models/finetuned/detect/weights/best.pt",
    )
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data/processed/detect"
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=PROJECT_ROOT / "configs/inference.yaml",
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--review-budget",
        type=float,
        default=0.15,
        help="proporsi maksimum gambar yang boleh diserahkan ke manusia",
    )
    parser.add_argument(
        "--extra-normals",
        type=int,
        default=300,
        help="gambar normal tambahan per sisi; nol untuk memakai split apa adanya",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if not args.weights.exists():
        log.error("bobot deteksi tidak ditemukan: %s", args.weights)
        return 1

    with args.inference_config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    decision = config["decision"]
    detection = config["models"]["detection"]
    alpha = float(decision["conformal"]["alpha"])
    prevalence = float(decision["cost"].get("assumed_defect_prevalence", 0.03))
    costs = CostModel(
        false_reject=float(decision["cost"]["c_false_reject"]),
        missed_defect=float(decision["cost"]["c_missed_defect"]),
        review=float(decision["cost"].get("c_review", 0.0)),
    )

    log.info("Kalibrasi lapisan keputusan")
    log.info("bobot   : %s", args.weights.name)
    log.info("alpha   : %.2f (jaminan cakupan %.0f persen)", alpha, (1 - alpha) * 100)
    log.info(
        "biaya   : salah tolak Rp %.0f | cacat lolos Rp %.0f",
        costs.false_reject,
        costs.missed_defect,
    )
    log.info("prevalensi cacat diasumsikan: %.1f persen", prevalence * 100)

    common = {
        "dataset_root": args.dataset,
        "imgsz": int(detection["imgsz"]),
        "iou_nms": float(detection["iou_threshold"]),
        "max_detections": int(detection["max_detections"]),
    }
    log.info("")
    log.info("[1/4] Skor tingkat gambar")
    extra_calib, extra_test = (
        gather_extra_normals(
            PROJECT_ROOT, args.dataset, per_side=args.extra_normals, seed=42
        )
        if args.extra_normals
        else ([], [])
    )
    if extra_calib or extra_test:
        log.info(
            "  normal tambahan: %d kalibrasi, %d uji (tak pernah dilihat model)",
            len(extra_calib),
            len(extra_test),
        )
    calib = image_level_scores(
        args.weights, split="val", extra_normals=extra_calib, **common
    )
    test = image_level_scores(
        args.weights, split="test", extra_normals=extra_test, **common
    )
    calib_probs = [s.score for s in calib]
    calib_labels = [int(s.is_defect) for s in calib]
    test_probs = [s.score for s in test]
    test_labels = [int(s.is_defect) for s in test]
    log.info("  kalibrasi (val) : %d gambar, %d cacat", len(calib), sum(calib_labels))
    log.info("  uji (test)      : %d gambar, %d cacat", len(test), sum(test_labels))

    log.info("")
    log.info("[2/4] Kalibrasi keyakinan")
    temperature = fit_temperature(calib_probs, calib_labels)
    platt = fit_platt(calib_probs, calib_labels)

    candidates = {
        "mentah": (calib_probs, test_probs),
        "temperature": (
            apply_temperature(calib_probs, temperature),
            apply_temperature(test_probs, temperature),
        ),
        "platt": (
            apply_platt(calib_probs, platt),
            apply_platt(test_probs, platt),
        ),
    }
    on_calibration = {
        name: calibration_report(pair[0], calib_labels, n_bins=args.bins)
        for name, pair in candidates.items()
    }
    on_test = {
        name: calibration_report(pair[1], test_labels, n_bins=args.bins)
        for name, pair in candidates.items()
    }
    chosen = min(on_calibration, key=lambda name: on_calibration[name].ece)
    calib_calibrated, test_calibrated = candidates[chosen]

    log.info("  suhu temperature : %.4f", temperature)
    log.info(
        "  platt            : kemiringan %.4f, intersep %.4f",
        platt.slope,
        platt.intercept,
    )
    log.info("  %-12s %10s %10s", "metode", "ECE kalib", "ECE uji")
    for name in candidates:
        log.info(
            "  %-12s %10.4f %10.4f%s",
            name,
            on_calibration[name].ece,
            on_test[name].ece,
            "  <- dipilih" if name == chosen else "",
        )
    before = on_test["mentah"]
    after = on_test[chosen]
    log.info("  MCE   %.4f -> %.4f", before.mce, after.mce)
    log.info("  Brier %.4f -> %.4f", before.brier, after.brier)

    log.info("")
    log.info("[3/4] Conformal prediction")

    candidates_alpha = (0.01, 0.05, 0.10, 0.15, 0.20)
    sweep_calibration = []
    for candidate in candidates_alpha:
        q = calibrate(calib_calibrated, calib_labels, alpha=candidate, mode="mondrian")
        c = evaluate_coverage(calib_calibrated, calib_labels, q)
        sweep_calibration.append(
            {
                "alpha": candidate,
                "coverage": c.empirical_coverage,
                "review_rate": c.review_rate,
                "normal_coverage": c.per_class_coverage.get(0, 0.0),
            }
        )

    # Alpha dipilih pada set KALIBRASI, tidak pernah pada set uji. Aturannya:
    # ambil jaminan sekuat mungkin, yaitu alpha sekecil mungkin, yang masih
    # menyisakan cukup banyak keputusan untuk diambil sistem sendiri. Sistem QC
    # yang menyerahkan mayoritas produk ke manusia tidak menyelesaikan apa pun.
    affordable = [
        row for row in sweep_calibration if row["review_rate"] <= args.review_budget
    ]
    selected = min(affordable, key=lambda row: row["alpha"]) if affordable else None
    if selected is None:
        log.warning(
            "  tidak ada alpha yang memenuhi batas menahan %.0f persen; memakai config",
            args.review_budget * 100,
        )
    else:
        alpha = selected["alpha"]

    log.info("  sapuan alpha pada SET KALIBRASI (dasar pemilihan):")
    log.info("  %6s %10s %10s %12s", "alpha", "cakupan", "ditahan", "cakupan normal")
    for row in sweep_calibration:
        log.info(
            "  %6.2f %10.4f %10.4f %12.4f%s",
            row["alpha"],
            row["coverage"],
            row["review_rate"],
            row["normal_coverage"],
            "  <- dipilih" if selected and row["alpha"] == alpha else "",
        )

    quantiles = calibrate(calib_calibrated, calib_labels, alpha=alpha, mode="mondrian")
    coverage = evaluate_coverage(test_calibrated, test_labels, quantiles)
    log.info("")
    log.info(
        "  alpha terpilih   : %.2f (batas menahan %.0f persen)",
        alpha,
        args.review_budget * 100,
    )
    log.info(
        "  kuantil per kelas: %s",
        {k: round(v, 4) for k, v in quantiles.quantiles.items()},
    )
    log.info(
        "  cakupan pada UJI : %.4f (jaminan %.2f) -> %s",
        coverage.empirical_coverage,
        coverage.target_coverage,
        "tercapai" if coverage.guarantee_met else "TIDAK tercapai",
    )
    log.info(
        "  cakupan per kelas: %s",
        {k: round(v, 4) for k, v in coverage.per_class_coverage.items()},
    )
    log.info("  ukuran himpunan  : rerata %.3f", coverage.average_set_size)
    log.info(
        "  keputusan diambil: %.1f persen | diserahkan: %.1f persen",
        coverage.singleton_rate * 100,
        coverage.review_rate * 100,
    )

    alpha_sweep = []
    for candidate in candidates_alpha:
        q = calibrate(calib_calibrated, calib_labels, alpha=candidate, mode="mondrian")
        c = evaluate_coverage(test_calibrated, test_labels, q)
        alpha_sweep.append(
            {
                "alpha": candidate,
                "coverage": c.empirical_coverage,
                "review_rate": c.review_rate,
                "normal_coverage": c.per_class_coverage.get(0, 0.0),
                "quantiles": {str(k): v for k, v in q.quantiles.items()},
            }
        )

    log.info("")
    log.info("[4/4] Ambang sensitif biaya")
    best = optimal_threshold(
        calib_calibrated, calib_labels, costs, prevalence=prevalence
    )
    curve = cost_curve(test_calibrated, test_labels, costs, prevalence=prevalence)
    comparison = savings_versus(
        test_calibrated,
        test_labels,
        costs,
        baseline_threshold=0.5,
        chosen_threshold=best.threshold,
        prevalence=prevalence,
    )
    log.info("  ambang Bayes     : %.4f", costs.bayes_threshold)
    log.info(
        "  minimum empiris  : %.4f (biaya Rp %.0f/unit pada kalibrasi)",
        best.threshold,
        best.cost_per_item,
    )
    log.info(
        "  pada data uji    : recall %.4f, biaya Rp %.0f/unit",
        comparison["optimal"]["recall"],
        comparison["optimal"]["cost_per_item"],
    )
    log.info(
        "  pembanding 0,50  : recall %.4f, biaya Rp %.0f/unit",
        comparison["baseline"]["recall"],
        comparison["baseline"]["cost_per_item"],
    )
    log.info(
        "  penghematan      : %.1f persen", comparison["cost_reduction_ratio"] * 100
    )
    naive = evaluate_threshold(
        test_calibrated, test_labels, 0.0, costs, prevalence=prevalence
    )
    log.info(
        "  tolak semua      : recall 1,0000, biaya Rp %.0f/unit", naive.cost_per_item
    )

    figures = PROJECT_ROOT / "reports/figures"
    plot_reliability(
        reliability_bins(test_probs, test_labels, n_bins=args.bins),
        reliability_bins(test_calibrated, test_labels, n_bins=args.bins),
        figures / "reliability_diagram.png",
    )
    plot_cost(curve, costs.bayes_threshold, best.threshold, figures / "cost_curve.png")
    plot_coverage(coverage, figures / "conformal_coverage.png")

    payload: dict[str, Any] = {
        "generated_on": date.today().isoformat(),
        "weights": str(args.weights),
        "extra_normals_per_side": args.extra_normals,
        "counts": {
            "calibration": len(calib),
            "calibration_defects": int(sum(calib_labels)),
            "test": len(test),
            "test_defects": int(sum(test_labels)),
        },
        "temperature": temperature,
        "platt": platt.to_dict(),
        "chosen_method": chosen,
        "calibration_on_calibration_set": {
            k: v.to_dict() for k, v in on_calibration.items()
        },
        "calibration_on_test_set": {k: v.to_dict() for k, v in on_test.items()},
        "calibration_before": before.to_dict(),
        "calibration_after": after.to_dict(),
        "conformal": quantiles.to_dict()
        | {
            "coverage": coverage.to_dict(),
            "alpha_sweep_test": alpha_sweep,
            "alpha_sweep_calibration": sweep_calibration,
            "review_budget": args.review_budget,
            "alpha_selected_on": "calibration",
        },
        "cost": costs.to_dict()
        | comparison
        | {"assumed_defect_prevalence": prevalence},
        "figures": [
            "reliability_diagram.png",
            "cost_curve.png",
            "conformal_coverage.png",
        ],
    }
    target = PROJECT_ROOT / "reports/metrics/calibration_results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("")
    log.info("Hasil ditulis ke %s", target.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
