"""Fine-tuning model deteksi cacat.

    python scripts/train_detection.py
    python scripts/train_detection.py --epochs 120 --name detect_cepat

Bobot hasil training disimpan di models/finetuned dan tidak masuk git.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionqc_ai.training.train_detect import fine_tune  # noqa: E402

log = logging.getLogger("train_detection")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tuning deteksi cacat VisionQC")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/training.yaml"
    )
    parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "data/processed/detect/data.yaml"
    )
    parser.add_argument("--name", default="detect")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--section", default="detection")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if not args.data.exists():
        log.error("data.yaml tidak ditemukan: %s", args.data)
        log.error("Jalankan scripts/prepare_dataset.py lebih dulu.")
        return 1

    project = PROJECT_ROOT / "models" / "finetuned"
    log.info("Fine-tuning %s", args.section)
    log.info("data    : %s", args.data)
    log.info("keluaran: %s", project / args.name)

    outcome = fine_tune(
        config_path=args.config,
        data_yaml=args.data,
        project=project,
        name=args.name,
        section=args.section,
        epochs_override=args.epochs,
    )

    summary = outcome.to_dict()
    summary["finished_on"] = date.today().isoformat()
    summary["section"] = args.section
    target = PROJECT_ROOT / "reports/metrics" / f"training_{args.name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("")
    log.info(
        "Selesai pada epoch %d dari %d",
        outcome.epochs_completed,
        outcome.epochs_requested,
    )
    log.info("Bobot terbaik: %s", outcome.weights_best)
    log.info("Ringkasan    : %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
