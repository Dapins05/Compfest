"""Hasilkan contoh cacat `kotor` sintetik untuk menyeimbangkan kelas.

    python scripts/generate_synthetic_kotor.py

Kenapa hanya kelas ini. Pada set latih, `kotor` punya 16 instance melawan 548
milik `gores` - rasio 34 kali. Lebih buruk lagi, seluruh 16 instance itu
berasal dari satu kategori produk saja, yaitu botol MVTec, sehingga model tidak
pernah melihat kontaminasi pada kemasan jenis lain. Padahal `kotor` adalah
satu-satunya anggota `critical_classes`: ia menolak produk berapa pun luasnya,
karena menyangkut keamanan konsumsi. Kelas dengan wewenang sebesar itu tidak
boleh menjadi kelas dengan bukti paling sedikit.

Data sintetik diizinkan R7.6 sejajar dengan sumber publik. Yang tidak diizinkan
adalah menyamarkannya, jadi setiap berkas keluaran diberi awalan `synth_kotor_`
agar selalu dapat dikenali dan dihitung terpisah.

Batas yang dijaga ketat:

* Keluaran HANYA masuk split latih. Menaruh gambar sintetik di val atau test
  akan membuat angka yang dilaporkan mengukur kemampuan meniru generator ini,
  bukan kemampuan menemukan kontaminasi sungguhan.
* Gambar dasar diambil hanya dari stem yang belum dipakai split deteksi mana
  pun, sehingga tidak ada gambar val atau test yang bocor lewat pintu belakang.

Kontaminasi digambar dengan pencampuran perkalian, bukan penimpaan. Kotoran
sungguhan menggelapkan permukaan tetapi teksturnya masih tembus; menimpa piksel
akan menghasilkan tambalan rata yang dapat dikenali model dari keratanya saja,
bukan dari bentuknya.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

log = logging.getLogger("generate_synthetic_kotor")

KOTOR_CLASS_ID = 3
PREFIX = "synth_kotor_"
# Awalan yang dipakai scripts/mine_hard_negatives.py. Dibutuhkan di sini
# untuk melucuti nama turunan kembali ke stem gambar dasarnya.
HARD_NEGATIVE_PREFIX = "hardneg_"

# Luas kontaminasi sebagai pecahan luas produk. Batas bawah menjaga cacat tetap
# terlihat pada 640 piksel; batas atas menjaganya tetap berupa kontaminasi,
# bukan produk yang tertutup kotoran seluruhnya.
#
# Rentang semula (0,004 sampai 0,045) terbukti terlalu sempit dan terlalu kecil.
# Diukur terhadap 22 instance kontaminasi asli, luas relatifnya 0,0011 melawan
# 0,1317 - beda seratus kali lipat. Rentangnya diperlebar supaya mencakup
# bercak besar, bukan hanya taburan halus.
#
# Rentang ini TIDAK disetel sampai uji KS lolos. Ke-22 instance asli seluruhnya
# berasal dari satu produk pada satu sudut pandang, yaitu mulut botol MVTec
# difoto dari atas, dan menyamakan sebaran dengannya justru akan menyempitkan
# data sintetik ke satu tampilan yang tidak mewakili produk lain.
AREA_FRACTION = (0.012, 0.16)

# Bobot pemilihan bentuk. Gerombolan remah menghasilkan banyak sisi tajam:
# kerapatan tepi terukur 0,2215 melawan 0,0298 pada kontaminasi asli. Noda
# bertepi lunak diberi bobot lebih besar untuk mengimbanginya.
BLOB_WEIGHTS = (2, 5, 2)


@dataclass(frozen=True)
class Sample:
    """Satu gambar sintetik beserta maskernya."""

    image: np.ndarray
    mask: np.ndarray


def product_mask(image: np.ndarray) -> np.ndarray:
    """Perkirakan wilayah produk, memisahkannya dari latar rak.

    Cara pertama yang dicoba - memperkirakan warna latar dari piksel tepi lalu
    mengambil piksel yang jauh darinya - GAGAL pada dataset ini, dan gagalnya
    berbahaya. Foto PKU-GoodsAD memuat DUA latar sekaligus: papan berlubang di
    belakang dan permukaan rak di bawah, dengan warna yang berbeda. Median
    tepinya menjadi campuran yang tidak menyerupai keduanya, sehingga rak ikut
    terbaca sebagai produk. Kotoran lalu ditempel di rak, dan model akan
    belajar bahwa kontaminasi memang muncul di latar - persis kebalikan dari
    yang dibutuhkan.

    GrabCut dipakai sebagai gantinya. Ia dimulai dari dugaan bahwa produk
    berada di kotak tengah bingkai, lalu memisahkan latar depan dan latar
    belakang dengan model campuran Gauss atas warna. Dugaan awal itu sahih di
    sini karena seluruh dataset difoto dengan produk di tengah.
    """
    height, width = image.shape[:2]
    margin_x, margin_y = int(width * 0.10), int(height * 0.10)
    rect = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)

    grab = np.zeros((height, width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            image,
            grab,
            rect,
            background_model,
            foreground_model,
            4,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return _centre_box(height, width)

    mask = np.where((grab == cv2.GC_FGD) | (grab == cv2.GC_PR_FGD), 255, 0).astype(
        np.uint8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return _centre_box(height, width)

    centre = np.array([width / 2, height / 2])
    best, best_score = 1, -1.0
    for index in range(1, count):
        area = stats[index, cv2.CC_STAT_AREA]
        cx = stats[index, cv2.CC_STAT_LEFT] + stats[index, cv2.CC_STAT_WIDTH] / 2
        cy = stats[index, cv2.CC_STAT_TOP] + stats[index, cv2.CC_STAT_HEIGHT] / 2
        distance_to_centre = float(np.linalg.norm(np.array([cx, cy]) - centre))
        score = area / (1.0 + distance_to_centre)
        if score > best_score:
            best, best_score = index, score
    chosen = ((labels == best).astype(np.uint8)) * 255

    # Erosi menjauhkan kotoran dari siluet produk. Tanpa ini, sebagian noda
    # mendarat tepat di tepi dan menutupi garis batas produk terhadap latar,
    # yang lebih menyerupai cacat bentuk daripada kontaminasi permukaan.
    inset = max(3, min(height, width) // 60)
    eroded = cv2.erode(
        chosen, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inset, inset))
    )
    return eroded if int(np.count_nonzero(eroded)) > 400 else chosen


def _centre_box(height: int, width: int) -> np.ndarray:
    """Kotak tengah, dipakai bila pemisahan latar depan tidak dapat dipercaya."""
    fallback = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(
        fallback,
        (width // 4, height // 4),
        (width * 3 // 4, height * 3 // 4),
        255,
        -1,
    )
    return fallback


def _blob_serpihan(size: int, rng: random.Random) -> np.ndarray:
    """Gerombolan remah: banyak partikel kecil yang berdekatan."""
    canvas = np.zeros((size, size), dtype=np.float32)
    for _ in range(rng.randint(12, 40)):
        radius = rng.randint(max(1, size // 40), max(2, size // 12))
        cx = rng.randint(radius, size - radius)
        cy = rng.randint(radius, size - radius)
        cv2.circle(canvas, (cx, cy), radius, rng.uniform(0.55, 1.0), -1)
    return canvas


def _blob_noda(size: int, rng: random.Random) -> np.ndarray:
    """Noda basah: satu bentuk tak beraturan bertepi lunak."""
    canvas = np.zeros((size, size), dtype=np.float32)
    centre = size // 2
    points = []
    for step in range(rng.randint(7, 12)):
        angle = 2 * np.pi * step / 9
        radius = centre * rng.uniform(0.45, 0.95)
        points.append(
            [centre + radius * np.cos(angle), centre + radius * np.sin(angle)]
        )
    cv2.fillPoly(canvas, [np.array(points, dtype=np.int32)], 1.0)
    return cv2.GaussianBlur(canvas, (0, 0), size * 0.06)


def _blob_serat(size: int, rng: random.Random) -> np.ndarray:
    """Serat atau rambut: garis tipis melengkung."""
    canvas = np.zeros((size, size), dtype=np.float32)
    x, y = rng.randint(0, size - 1), rng.randint(0, size - 1)
    angle = rng.uniform(0, 2 * np.pi)
    thickness = max(1, size // 45)
    for _ in range(rng.randint(18, 32)):
        angle += rng.uniform(-0.45, 0.45)
        nx = int(np.clip(x + np.cos(angle) * size * 0.07, 0, size - 1))
        ny = int(np.clip(y + np.sin(angle) * size * 0.07, 0, size - 1))
        cv2.line(canvas, (x, y), (nx, ny), 1.0, thickness)
        x, y = nx, ny
    return canvas


_BLOBS = (_blob_serpihan, _blob_noda, _blob_serat)


def contaminate(image: np.ndarray, rng: random.Random) -> Sample | None:
    """Tempelkan kontaminasi pada wilayah produk dan kembalikan maskernya."""
    region = product_mask(image)
    area = int(np.count_nonzero(region))
    height, width = image.shape[:2]
    if area < 0.02 * height * width:
        return None

    ys, xs = np.nonzero(region)
    result = image.astype(np.float32)
    total = np.zeros((height, width), dtype=np.float32)

    for _ in range(rng.randint(1, 3)):
        fraction = rng.uniform(*AREA_FRACTION)
        size = int(np.sqrt(area * fraction))
        size = int(np.clip(size, 12, max(13, min(height, width) // 3)))
        maker = rng.choices(_BLOBS, weights=BLOB_WEIGHTS, k=1)[0]
        blob = maker(size, rng)
        blob = cv2.warpAffine(
            blob,
            cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(0, 360), 1.0),
            (size, size),
        )
        blob = cv2.GaussianBlur(blob, (0, 0), max(0.8, size * 0.045))
        blob = np.clip(blob, 0.0, 1.0)

        # Titik tempel diambil dari piksel produk yang sesungguhnya, bukan dari
        # kotak pembatasnya, supaya kotoran tidak mendarat di sudut kosong.
        pick = rng.randrange(len(xs))
        left = int(np.clip(int(xs[pick]) - size // 2, 0, max(0, width - size)))
        top = int(np.clip(int(ys[pick]) - size // 2, 0, max(0, height - size)))

        window = region[top : top + size, left : left + size] > 0
        if window.shape != blob.shape:
            continue
        blob = blob * window
        if float(blob.sum()) < 0.015 * size * size:
            continue

        darkness = rng.uniform(0.30, 0.68)
        tint = np.array(
            [rng.uniform(0.75, 1.0), rng.uniform(0.80, 1.0), rng.uniform(0.85, 1.05)],
            dtype=np.float32,
        )
        patch = result[top : top + size, left : left + size]
        alpha = blob[..., None]
        # Perkalian, bukan penimpaan: tekstur di bawah kotoran tetap tembus.
        patch *= 1.0 - alpha * (1.0 - darkness * tint)
        # Kotoran sungguhan tidak rata warnanya; derau halus mencegah tambalan
        # tampak seperti bentuk geometri yang bersih.
        spread = rng.uniform(0.03, 0.09)
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
            0.0, spread, patch.shape
        )
        patch *= 1.0 + alpha * noise.astype(np.float32)
        result[top : top + size, left : left + size] = patch
        total[top : top + size, left : left + size] = np.maximum(
            total[top : top + size, left : left + size], blob
        )

    mask = (total > 0.30).astype(np.uint8) * 255
    if int(np.count_nonzero(mask)) < 40:
        return None
    return Sample(image=np.clip(result, 0, 255).astype(np.uint8), mask=mask)


def labels_from_mask(mask: np.ndarray) -> tuple[list[str], list[str]]:
    """Ubah masker menjadi baris label YOLO deteksi dan segmentasi."""
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detect_rows: list[str] = []
    seg_rows: list[str] = []
    for contour in contours:
        if cv2.contourArea(contour) < 25:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        detect_rows.append(
            f"{KOTOR_CLASS_ID} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} "
            f"{w / width:.6f} {h / height:.6f}"
        )
        simplified = cv2.approxPolyDP(
            contour, 0.006 * cv2.arcLength(contour, True), True
        )
        if len(simplified) < 3:
            continue
        coords = " ".join(
            f"{point[0][0] / width:.6f} {point[0][1] / height:.6f}"
            for point in simplified
        )
        seg_rows.append(f"{KOTOR_CLASS_ID} {coords}")
    return detect_rows, seg_rows


def used_stems(detect_root: Path) -> set[str]:
    """Stem gambar dasar yang sudah dipakai split deteksi mana pun.

    Awalan turunan dilucuti lebih dulu. Gambar yang masuk set latih sebagai
    negatif keras tersimpan bernama `hardneg_<stem>`; tanpa pelucutan ini,
    `<stem>` yang sama masih terbaca sebagai belum terpakai dan dapat terpilih
    lagi sebagai dasar gambar sintetik. Model sudah pernah melihat versi
    bersihnya saat latihan, sehingga memakainya untuk penilaian akan
    melaporkan angka yang lebih baik daripada yang sebenarnya.
    """
    stems: set[str] = set()
    for split in ("train", "val", "test"):
        for path in (detect_root / "images" / split).glob("*.jpg"):
            stem = path.stem
            for prefix in (PREFIX, HARD_NEGATIVE_PREFIX):
                if stem.startswith(prefix):
                    stem = stem[len(prefix) :]
            stems.add(stem)
    return stems


def main() -> int:
    parser = argparse.ArgumentParser(description="Hasilkan cacat kotor sintetik")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/dataset.yaml"
    )
    parser.add_argument(
        "--detect-root", type=Path, default=PROJECT_ROOT / "data/processed/detect"
    )
    parser.add_argument(
        "--seg-root", type=Path, default=PROJECT_ROOT / "data/processed/seg"
    )
    parser.add_argument(
        "--anomaly-root", type=Path, default=PROJECT_ROOT / "data/processed/anomaly"
    )
    parser.add_argument("--count", type=int, default=220, help="jumlah gambar sintetik")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--clean", action="store_true", help="hapus keluaran sintetik sebelumnya"
    )
    parser.add_argument(
        "--exclude-from",
        type=Path,
        default=None,
        help=(
            "akar deteksi resmi yang stem-nya dipantang; dipakai saat keluaran "
            "ditulis ke folder lain, misalnya set evaluasi sintetik"
        ),
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if args.clean:
        removed = 0
        for root in (args.detect_root, args.seg_root):
            for kind in ("images", "labels"):
                for path in (root / kind / "train").glob(f"{PREFIX}*"):
                    path.unlink()
                    removed += 1
        log.info("keluaran sintetik lama dihapus: %d berkas", removed)

    with args.config.open(encoding="utf-8") as handle:
        categories = yaml.safe_load(handle)["categories"]["detect"]

    official = args.exclude_from or args.detect_root
    used = used_stems(official)
    # Gambar dasar yang sudah terpakai membuat versi sintetik di split resmi
    # ikut dipantang. Tanpa ini, set evaluasi sintetik dapat memakai dasar yang
    # sama dengan set latih sintetik, dan angkanya akan mengukur hafalan.
    already = {
        path.stem[len(PREFIX) :]
        for split in ("train", "val", "test")
        for path in (official / "images" / split).glob(f"{PREFIX}*.jpg")
    }
    pool: list[Path] = []
    for category in categories:
        for subset in ("train/good", "test/good"):
            pool += [
                path
                for path in sorted(
                    (args.anomaly_root / category / subset).glob("*.jpg")
                )
                if path.stem not in used and path.stem not in already
            ]
    if not pool:
        log.error("tidak ada gambar dasar yang tersedia di luar split deteksi")
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    log.info("Generator cacat kotor sintetik")
    log.info("gambar dasar tersedia : %d", len(pool))
    log.info("target                : %d gambar, hanya split latih", args.count)

    for root in (args.detect_root, args.seg_root):
        (root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "train").mkdir(parents=True, exist_ok=True)

    made = skipped = instances = 0
    per_category: dict[str, int] = {}
    for source in pool:
        if made >= args.count:
            break
        image = cv2.imread(str(source))
        if image is None:
            skipped += 1
            continue
        sample = contaminate(image, rng)
        if sample is None:
            skipped += 1
            continue
        detect_rows, seg_rows = labels_from_mask(sample.mask)
        if not detect_rows:
            skipped += 1
            continue

        stem = f"{PREFIX}{source.stem}"
        cv2.imwrite(
            str(args.detect_root / "images" / "train" / f"{stem}.jpg"), sample.image
        )
        (args.detect_root / "labels" / "train" / f"{stem}.txt").write_text(
            "\n".join(detect_rows) + "\n", encoding="utf-8"
        )
        if seg_rows:
            cv2.imwrite(
                str(args.seg_root / "images" / "train" / f"{stem}.jpg"), sample.image
            )
            (args.seg_root / "labels" / "train" / f"{stem}.txt").write_text(
                "\n".join(seg_rows) + "\n", encoding="utf-8"
            )
        made += 1
        instances += len(detect_rows)
        key = source.parent.parent.parent.name
        per_category[key] = per_category.get(key, 0) + 1

    log.info("")
    log.info("gambar dibuat   : %d", made)
    log.info("instance kotor  : %d", instances)
    log.info("dilewati        : %d", skipped)
    for key in sorted(per_category):
        log.info("  %-16s %d", key, per_category[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
