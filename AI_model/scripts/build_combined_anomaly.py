"""Gabungkan dataset anomali per kategori menjadi satu kategori `combined`.

    python scripts/build_combined_anomaly.py

Model anomali yang benar-benar dipakai saat penyajian adalah model gabungan,
bukan model per kategori. Alasannya sederhana: bagi model yang hanya mengenal
botol, kacang mete pun terbaca sebagai anomali. Sistem menerima foto produk
apa saja, jadi model normalitasnya harus mengenal seluruh kategori yang
didukung.

Sebelumnya folder `combined` dibuat manual di luar repo, sehingga tidak dapat
direproduksi dan sempat tertinggal memuat empat kategori lama saja setelah
PKU-GoodsAD masuk. Skrip ini menjadikannya langkah yang tercatat.

Gambar disalin sebagai hard link bila didukung, jadi tidak menggandakan
pemakaian disk.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

log = logging.getLogger("build_combined_anomaly")

COMBINED = "combined"
SUBSETS = ("train/good", "test/good", "test/defect")


def link_or_copy(source: Path, target: Path) -> None:
    """Hard link bila memungkinkan, salin bila tidak."""
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def build(
    anomaly_root: Path,
    categories: list[str],
    *,
    per_category_train: int,
    seed: int,
) -> dict[str, int]:
    """Bangun ulang folder gabungan dari kategori yang diberikan.

    Gambar latih dibatasi per kategori, bukan diambil acak dari kumpulan
    gabungan. Dua alasan:

    1. **Memori.** PaDiM menahan embedding seluruh gambar latih sebelum
       mencocokkan Gaussian. Dengan 5.383 gambar, prosesnya meminta 6,57 GiB
       pada GPU 4 GB dan mati. Batas `max_train_images` di configs tidak dapat
       diandalkan untuk ini: `engine.fit()` menjalankan ulang `setup()`
       sehingga pembatasan di memori terbuang. Membatasi jumlah berkasnya
       secara fisik adalah satu-satunya cara yang tidak bisa dibatalkan
       anomalib.
    2. **Keseimbangan.** Mengambil acak dari kumpulan gabungan akan didominasi
       kategori terbesar; `food_bottle` menyumbang 1.069 gambar sementara
       `bottle` hanya 195. Model normalitas yang dihasilkan akan mengenal botol
       makanan jauh lebih baik daripada kategori lain, padahal sistem menerima
       foto produk apa saja.

    Pengambilan sampelnya deterministik terhadap seed.
    """
    combined_root = anomaly_root / COMBINED
    if combined_root.exists():
        shutil.rmtree(combined_root)

    counts: dict[str, int] = {}
    for subset in SUBSETS:
        target_dir = combined_root / subset
        target_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for category in categories:
            source_dir = anomaly_root / category / subset
            if not source_dir.is_dir():
                continue
            images = sorted(source_dir.glob("*.jpg"))
            if subset == "train/good" and 0 < per_category_train < len(images):
                rng = random.Random(f"{seed}:{category}")
                images = sorted(rng.sample(images, per_category_train))
            for image in images:
                link_or_copy(image, target_dir / image.name)
                written += 1
        counts[subset] = written
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Bangun dataset anomali gabungan")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/dataset.yaml"
    )
    parser.add_argument(
        "--anomaly-root", type=Path, default=PROJECT_ROOT / "data/processed/anomaly"
    )
    parser.add_argument(
        "--per-category-train",
        type=int,
        default=60,
        help="gambar latih maksimum per kategori; 0 berarti tanpa batas",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    categories = [c for c in config["categories"]["anomaly"] if c != COMBINED]

    log.info("Membangun dataset anomali gabungan")
    log.info("kategori : %s", categories)
    log.info("batas latih per kategori : %d", args.per_category_train)
    counts = build(
        args.anomaly_root,
        categories,
        per_category_train=args.per_category_train,
        seed=args.seed,
    )
    for subset, total in counts.items():
        log.info("  %-12s %d gambar", subset, total)
    log.info("Selesai: %s", args.anomaly_root / COMBINED)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
