"""Lanjutkan fine-tuning deteksi yang terhenti di tengah jalan.

    python scripts/resume_detection.py --name detect_balanced

Berbeda dengan `train_detection.py` yang memulai run baru, skrip ini menyambung
run yang sudah ada dari `weights/last.pt`. Ultralytics membaca kembali
`args.yaml` di dalam folder run itu, sehingga jumlah epoch, dataset,
hyperparameter, dan keadaan optimizer persis melanjutkan yang sebelumnya - bukan
memulai ulang dengan setelan yang kebetulan mirip.

Run `detect_balanced` terhenti di epoch 124 dari 150 ketika mesin dimatikan.
Menyambungnya lebih murah daripada mengulang dari nol, dan hasilnya tetap sah
dibandingkan karena datanya sama.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

log = logging.getLogger("resume_detection")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sambung fine-tuning deteksi VisionQC")
    parser.add_argument(
        "--name",
        default="detect_balanced",
        help="nama run di models/finetuned yang akan disambung",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    run_dir = PROJECT_ROOT / "models" / "finetuned" / args.name
    last = run_dir / "weights" / "last.pt"
    if not last.is_file():
        log.error("bobot terakhir tidak ditemukan: %s", last)
        log.error("Run itu belum pernah dimulai, atau folder bobotnya sudah dibuang.")
        return 1

    from ultralytics import YOLO

    log.info("menyambung run %s dari %s", args.name, last)
    model = YOLO(str(last))
    # `resume=True` menuntut bobotnya berupa checkpoint run yang belum tuntas;
    # bila run-nya sudah mencapai epoch terakhir, Ultralytics berhenti sendiri
    # dan tidak melatih apa pun.
    model.train(resume=True)

    log.info("selesai; bobot terbaik ada di %s", run_dir / "weights" / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
