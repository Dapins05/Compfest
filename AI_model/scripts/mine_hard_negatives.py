"""Tambahkan gambar produk bagus yang paling sering salah dituduh cacat.

    python scripts/mine_hard_negatives.py

Masalah yang diselesaikan. Pada set kalibrasi, seluruh 39 salah-tolak berasal
dari PKU-GoodsAD; kelompok MVTec dan VisA tidak menyumbang satu pun. Detektor
menuduh produk bagus dengan `terbuka` 15 kali dan `gores` 14 kali. Menggeser
ambang keputusan hanya memindahkan posisi pada kurva yang sama - specificity
naik dengan menukar recall. Yang menggeser kurvanya adalah memberi model contoh
tentang seperti apa produk bagus itu.

Gambar latar tanpa label memang cara yang disediakan YOLO untuk itu. Set latih
sekarang memuat 13 persen gambar latar; menambahnya menurunkan kecenderungan
model menemukan cacat pada permukaan bersih.

Kenapa dipilih, bukan diambil acak. Gambar bagus yang sudah dilewatkan model
dengan benar tidak mengajarkan apa pun yang belum diketahuinya. Yang berguna
adalah gambar yang saat ini ditolak keliru, jadi kandidat diberi skor lebih
dulu memakai detektor yang ada dan yang paling meyakinkan salahnya diambil
duluan.

Ini langkah penyiapan data sekali jalan pada masa pengembangan, bukan loop
umpan balik saat penyajian. Parameter inferensi tetap statis saat demo.

Kandidat hanya diambil dari stem yang belum dipakai split deteksi mana pun,
sehingga tidak ada gambar val atau test yang berpindah ke set latih.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

log = logging.getLogger("mine_hard_negatives")

PREFIX = "hardneg_"
SYNTH_PREFIX = "synth_kotor_"


def used_stems(detect_root: Path) -> set[str]:
    """Stem yang sudah dipakai split deteksi mana pun, termasuk turunannya."""
    stems: set[str] = set()
    for split in ("train", "val", "test"):
        for path in (detect_root / "images" / split).glob("*.jpg"):
            stem = path.stem
            for prefix in (PREFIX, SYNTH_PREFIX):
                if stem.startswith(prefix):
                    stem = stem[len(prefix) :]
            stems.add(stem)
    return stems


def main() -> int:
    parser = argparse.ArgumentParser(description="Tambang negatif keras")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/dataset.yaml"
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=PROJECT_ROOT / "configs/inference.yaml",
    )
    parser.add_argument(
        "--detect-root", type=Path, default=PROJECT_ROOT / "data/processed/detect"
    )
    parser.add_argument(
        "--anomaly-root", type=Path, default=PROJECT_ROOT / "data/processed/anomaly"
    )
    parser.add_argument(
        "--count", type=int, default=300, help="jumlah negatif keras yang ditambahkan"
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=900,
        help="jumlah kandidat yang diberi skor sebelum diambil yang tersulit",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--clean", action="store_true", help="hapus negatif keras sebelumnya"
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if args.clean:
        removed = 0
        for kind in ("images", "labels"):
            for path in (args.detect_root / kind / "train").glob(f"{PREFIX}*"):
                path.unlink()
                removed += 1
        log.info("negatif keras lama dihapus: %d berkas", removed)

    with args.config.open(encoding="utf-8") as handle:
        categories = yaml.safe_load(handle)["categories"]["detect"]
    with args.inference_config.open(encoding="utf-8") as handle:
        inference = yaml.safe_load(handle)
    detection = inference["models"]["detection"]

    used = used_stems(args.detect_root)
    pool: list[Path] = []
    for category in categories:
        for subset in ("train/good", "test/good"):
            pool += [
                path
                for path in sorted(
                    (args.anomaly_root / category / subset).glob("*.jpg")
                )
                if path.stem not in used
            ]
    if not pool:
        log.error("tidak ada kandidat di luar split deteksi")
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    candidates = pool[: args.candidates]

    log.info("Penambangan negatif keras")
    log.info("kandidat tersedia : %d", len(pool))
    log.info("kandidat dinilai  : %d", len(candidates))
    log.info("akan ditambahkan  : %d", args.count)

    from ultralytics import YOLO

    model = YOLO(str(PROJECT_ROOT / detection["path"]), task="detect")
    imgsz = int(detection["imgsz"])
    conf = float(detection["conf_threshold"])

    scored: list[tuple[float, Path]] = []
    for index, path in enumerate(candidates, start=1):
        prediction = model.predict(
            str(path), imgsz=imgsz, conf=conf, verbose=False, device="cpu"
        )[0]
        boxes = prediction.boxes
        # Skor kesulitan adalah keyakinan tertinggi pada gambar yang seharusnya
        # tidak memuat cacat sama sekali. Nol berarti model sudah benar.
        worst = float(boxes.conf.max()) if boxes is not None and len(boxes) else 0.0
        scored.append((worst, path))
        if index % 100 == 0:
            log.info("  dinilai %d/%d", index, len(candidates))

    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = scored[: args.count]
    flagged = sum(1 for score, _ in chosen if score > 0.0)

    images_dir = args.detect_root / "images" / "train"
    labels_dir = args.detect_root / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    added = 0
    for _score, path in chosen:
        stem = f"{PREFIX}{path.stem}"
        shutil.copy2(path, images_dir / f"{stem}.jpg")
        # Berkas label kosong adalah cara YOLO menyatakan gambar latar: tidak
        # ada objek di sini. Berkas yang tidak dibuat sama sekali dibaca sebagai
        # label hilang, dan Ultralytics memperlakukannya berbeda.
        (labels_dir / f"{stem}.txt").write_text("", encoding="utf-8")
        added += 1

    log.info("")
    log.info("negatif keras ditambahkan : %d", added)
    log.info("di antaranya salah dituduh : %d", flagged)
    if chosen:
        log.info("keyakinan salah tertinggi  : %.4f", chosen[0][0])
        log.info("keyakinan salah terendah   : %.4f", chosen[-1][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
