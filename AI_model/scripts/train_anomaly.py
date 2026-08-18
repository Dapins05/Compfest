"""Latih model anomali dan turunkan ambangnya dengan teori nilai ekstrem.

    python scripts/train_anomaly.py
    python scripts/train_anomaly.py --categories bottle --model padim

Ambang tidak dipilih tangan melainkan dihitung dari ekor sebaran skor pada
sampel normal, menargetkan laju alarm palsu yang ditetapkan di config.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
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

from visionqc_ai.statistics.evt import (  # noqa: E402
    compute_evt_threshold,
    quantile_threshold,
)
from visionqc_ai.training.train_anomaly import (  # noqa: E402
    AnomalyScores,
    build_datamodule,
    build_model,
    cap_training_images,
    split_scores,
)

log = logging.getLogger("train_anomaly")

RESULTS_PATH = PROJECT_ROOT / "reports/metrics/anomaly_results.json"


def save_results(
    per_category: dict[str, Any], failures: dict[str, str], meta: dict[str, Any]
) -> None:
    """Tulis hasil segera setelah tiap kategori selesai.

    Menulis hanya di akhir berarti satu kategori yang gagal akan menghapus
    kerja kategori sebelumnya. Hasil digabung dengan isi berkas yang sudah ada
    supaya menjalankan satu kategori tidak menghapus kategori lain.
    """
    existing: dict[str, Any] = {}
    if RESULTS_PATH.exists():
        try:
            existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    merged_results = dict(existing.get("per_category", {})) | per_category
    merged_failures = {
        k: v
        for k, v in (dict(existing.get("failures", {})) | failures).items()
        if k not in merged_results
    }
    payload = meta | {
        "generated_on": date.today().isoformat(),
        "per_category": merged_results,
        "failures": merged_failures,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_category(
    category: str,
    *,
    anomaly_config: dict[str, Any],
    seed: int,
) -> AnomalyScores:
    """Latih satu kategori lalu kumpulkan skor kalibrasi dan skor uji."""
    from anomalib.engine import Engine
    from lightning.pytorch import seed_everything

    seed_everything(seed, workers=True)
    category_root = PROJECT_ROOT / "data" / "processed" / "anomaly" / category
    if not category_root.is_dir():
        raise FileNotFoundError(f"dataset anomali tidak ditemukan: {category_root}")

    model_name = str(anomaly_config.get("model", "padim")).lower()
    model = build_model(anomaly_config, PROJECT_ROOT)
    datamodule = build_datamodule(
        category_root,
        category,
        batch_size=int(anomaly_config.get("batch", 1)),
    )

    engine = Engine(
        default_root_dir=str(PROJECT_ROOT / "models" / "finetuned" / "anomaly"),
        max_epochs=int(anomaly_config.get("max_epochs", 1)),
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=True,
    )
    datamodule.setup()
    used = cap_training_images(
        datamodule, limit=int(anomaly_config.get("max_train_images", 0)), seed=seed
    )
    log.info("  gambar latih    : %d dipakai", used)
    engine.fit(model=model, datamodule=datamodule)

    calibration, _ = split_scores(
        engine.predict(model=model, dataloaders=datamodule.val_dataloader()) or []
    )
    test_normal, test_defect = split_scores(
        engine.predict(model=model, dataloaders=datamodule.test_dataloader()) or []
    )
    return AnomalyScores(
        category=category,
        model_name=model_name,
        calibration_normal=calibration,
        test_normal=test_normal,
        test_defect=test_defect,
    )


def plot_category(
    scores: AnomalyScores, evt: Any, quantile: float, target: Path
) -> None:
    """Gambar sebaran skor, ambang, dan kecocokan GPD pada ekornya."""
    from scipy import stats as scipy_stats

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3))

    everything = scores.test_normal + scores.test_defect + scores.calibration_normal
    bins = np.linspace(min(everything), max(everything), 40)
    axes[0].hist(
        scores.test_normal, bins=bins, alpha=0.75, label="normal (uji)", color="#268bd2"
    )
    axes[0].hist(
        scores.test_defect, bins=bins, alpha=0.75, label="cacat (uji)", color="#dc322f"
    )
    axes[0].axvline(
        evt.threshold,
        color="#002b36",
        linestyle="--",
        linewidth=1.6,
        label=f"ambang EVT {evt.threshold:.4f}",
    )
    axes[0].axvline(
        quantile,
        color="#859900",
        linestyle=":",
        linewidth=1.6,
        label=f"kuantil empiris {quantile:.4f}",
    )
    axes[0].set_xlabel("skor anomali")
    axes[0].set_ylabel("jumlah gambar")
    axes[0].set_title(f"Sebaran skor - {scores.category}")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    normal = np.asarray(scores.calibration_normal, dtype=float)
    exceedances = np.sort(
        normal[normal > evt.fit.initial_threshold] - evt.fit.initial_threshold
    )
    if exceedances.size:
        empirical = (np.arange(1, exceedances.size + 1) - 0.5) / exceedances.size
        theoretical = scipy_stats.genpareto.cdf(
            exceedances, evt.fit.shape_xi, loc=0.0, scale=evt.fit.scale_sigma
        )
        axes[1].plot([0, 1], [0, 1], color="#586e75", linestyle="--", linewidth=1)
        axes[1].scatter(theoretical, empirical, s=20, color="#268bd2")
        axes[1].set_xlabel("peluang kumulatif model GPD")
        axes[1].set_ylabel("peluang kumulatif empiris")
        axes[1].set_title(
            f"Kecocokan GPD pada ekor (KS p = {evt.fit.ks_p_value:.3f})".replace(
                ".", ","
            )
        )
        axes[1].grid(alpha=0.3)

    figure.suptitle(
        f"Ambang anomali berbasis teori nilai ekstrem - {scores.category}", fontsize=12
    )
    figure.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=120, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pelatihan anomali VisionQC")
    parser.add_argument(
        "--training-config", type=Path, default=PROJECT_ROOT / "configs/training.yaml"
    )
    parser.add_argument(
        "--dataset-config", type=Path, default=PROJECT_ROOT / "configs/dataset.yaml"
    )
    parser.add_argument(
        "--inference-config", type=Path, default=PROJECT_ROOT / "configs/inference.yaml"
    )
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--model", default=None, help="menimpa pilihan model di config")
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    warnings.filterwarnings("ignore")

    with args.training_config.open(encoding="utf-8") as handle:
        training = yaml.safe_load(handle)
    with args.dataset_config.open(encoding="utf-8") as handle:
        dataset = yaml.safe_load(handle)
    with args.inference_config.open(encoding="utf-8") as handle:
        inference = yaml.safe_load(handle)

    anomaly_config = dict(training["anomaly"])
    if args.model:
        anomaly_config["model"] = args.model
    if args.max_epochs is not None:
        anomaly_config["max_epochs"] = args.max_epochs

    categories = args.categories or dataset["categories"]["anomaly"]
    target_far = float(inference["decision"]["anomaly_target_far"])
    seed = int(dataset["split"]["seed"])

    log.info("Pelatihan anomali VisionQC")
    log.info("model     : %s", anomaly_config.get("model"))
    log.info("kategori  : %s", categories)
    log.info("target laju alarm palsu: %.1f%%", target_far * 100)

    meta = {
        "model": anomaly_config.get("model"),
        "variant": anomaly_config.get("variant"),
        "max_epochs": anomaly_config.get("max_epochs"),
        "max_train_images": anomaly_config.get("max_train_images"),
        "target_false_alarm_rate": target_far,
        "seed": seed,
    }
    results: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for category in categories:
        log.info("")
        log.info("=== %s ===", category)
        try:
            scores = run_category(
                category,
                anomaly_config=anomaly_config,
                seed=seed,
            )
        except Exception as error:
            log.error("  gagal: %s", error)
            failures[category] = str(error)
            save_results(results, failures, meta)
            continue
        try:
            evt = compute_evt_threshold(
                scores.calibration_normal, target_false_alarm_rate=target_far
            )
        except ValueError as error:
            log.error("  ambang EVT tidak dapat dihitung: %s", error)
            failures[category] = str(error)
            save_results(results, failures, meta)
            continue
        quantile = quantile_threshold(
            scores.calibration_normal, target_false_alarm_rate=target_far
        )
        auroc = scores.image_auroc()
        recall = float(np.mean(np.asarray(scores.test_defect) > evt.threshold))
        held_out_far = float(np.mean(np.asarray(scores.test_normal) > evt.threshold))

        log.info("  gambar          : %s", scores.counts)
        log.info("  AUROC gambar    : %.4f", auroc)
        log.info(
            "  ambang EVT      : %.4f (kuantil empiris %.4f)", evt.threshold, quantile
        )
        log.info(
            "  GPD             : xi=%+.4f sigma=%.4f u=%.4f (q%.0f) N_u=%d KS p=%.3f",
            evt.fit.shape_xi,
            evt.fit.scale_sigma,
            evt.fit.initial_threshold,
            evt.fit.used_quantile * 100,
            evt.fit.n_exceedances,
            evt.fit.ks_p_value,
        )
        log.info("  %s", evt.fit.tail_type)
        log.info(
            "  alarm palsu     : %.3f kalibrasi, %.3f uji terpisah (target %.3f)",
            evt.empirical_false_alarm_rate,
            held_out_far,
            target_far,
        )
        log.info("  recall cacat    : %.4f pada ambang EVT", recall)

        plot_category(
            scores,
            evt,
            quantile,
            PROJECT_ROOT / "reports/figures" / f"anomaly_{category}.png",
        )
        results[category] = {
            "model": scores.model_name,
            "counts": scores.counts,
            "image_auroc": auroc,
            "defect_recall_at_evt": recall,
            "held_out_false_alarm_rate": held_out_far,
            "evt": evt.to_dict(),
            "empirical_quantile_threshold": quantile,
            "statement_id": evt.describe_id(),
        }
        save_results(results, failures, meta)

    log.info("")
    log.info("Hasil ada di %s", RESULTS_PATH.relative_to(PROJECT_ROOT))
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
