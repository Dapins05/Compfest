"""Peredaman wajah sebelum gambar masuk ke model.

Wajah adalah data biometrik. UU PDP No. 27 Tahun 2022 Pasal 4 ayat 2
menggolongkannya sebagai data pribadi bersifat spesifik yang perlindungannya
lebih ketat daripada data pribadi umum.

Pada inspeksi kualitas, wajah tidak pernah menjadi bagian dari yang diperiksa.
Ia hanya muncul tanpa sengaja ketika operator ikut terpotret. Karena itu wajah
diburamkan **sebelum** gambar mencapai model, bukan sesudahnya: model tidak
perlu pernah melihatnya sama sekali.

Pendeteksinya YuNet, jaringan ringan bawaan OpenCV berukuran sekitar 230 KB.
Ia dipilih menggantikan Haar cascade karena OpenCV 5 tidak lagi menyediakan
cascade pada modul utamanya, dan karena YuNet lebih tahan terhadap wajah yang
menyamping. Bila modelnya tidak tersedia, peredaman dilaporkan tidak aktif
alih-alih diam-diam dilewati; klaim privasi yang tidak berjalan lebih berbahaya
daripada klaim yang dinyatakan gagal.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

MODEL_FILENAME = "face_detection_yunet.onnx"
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "onnx" / MODEL_FILENAME
)


@dataclass(frozen=True)
class BlurResult:
    """Hasil peredaman wajah beserta keterangan apakah lapisan ini aktif."""

    image: np.ndarray
    faces_blurred: int
    detector_available: bool
    note: str = ""


_DETECTOR_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _detector(model_path: str) -> object | None:
    """Muat pendeteksi wajah sekali lalu pakai ulang.

    Mengembalikan ``None`` bila model atau dukungan OpenCV tidak tersedia,
    supaya pemanggil dapat melaporkannya alih-alih gagal total.
    """
    if not Path(model_path).exists() or not hasattr(cv2, "FaceDetectorYN"):
        return None
    try:
        return cv2.FaceDetectorYN.create(model_path, "", (320, 320))
    except cv2.error:
        return None


def detect_faces(
    image: np.ndarray, *, model_path: Path | None = None, threshold: float = 0.7
) -> tuple[list[tuple[int, int, int, int]], bool]:
    """Cari wajah dan kembalikan kotaknya beserta status ketersediaan pendeteksi."""
    detector = _detector(str(model_path or DEFAULT_MODEL_PATH))
    if detector is None:
        return [], False

    height, width = image.shape[:2]
    # Pendeteksi ini dipakai ulang lintas permintaan, sedangkan menyetel ukuran
    # masukan lalu menjalankan deteksi adalah dua langkah terpisah pada objek
    # yang sama. Tanpa kunci, permintaan lain dapat menyisipkan ukuran berbeda
    # di antara keduanya dan OpenCV menggugurkan proses karena bentuk buffer
    # tidak lagi cocok.
    with _DETECTOR_LOCK:
        detector.setInputSize((width, height))
        detector.setScoreThreshold(threshold)
        _, faces = detector.detect(image)
    if faces is None:
        return [], True
    return [(int(row[0]), int(row[1]), int(row[2]), int(row[3])) for row in faces], True


def count_faces(image: np.ndarray, *, model_path: Path | None = None) -> int:
    """Jumlah wajah yang terdeteksi."""
    return len(detect_faces(image, model_path=model_path)[0])


def blur_faces(
    image: np.ndarray,
    *,
    kernel: int = 45,
    margin: float = 0.15,
    model_path: Path | None = None,
) -> BlurResult:
    """Buramkan setiap wajah pada salinan gambar.

    Kotak wajah diperlebar sedikit karena pendeteksi cenderung memotong dahi
    dan dagu, sehingga tepi wajah dapat tersisa jelas bila diburamkan pas pada
    kotaknya. Ukuran kernel diperbesar mengikuti ukuran wajah agar wajah besar
    tidak tetap terbaca setelah diburamkan.
    """
    faces, available = detect_faces(image, model_path=model_path)
    if not available:
        return BlurResult(
            image=image,
            faces_blurred=0,
            detector_available=False,
            note=f"{MODEL_FILENAME} tidak tersedia; peredaman wajah tidak aktif",
        )
    if not faces:
        return BlurResult(image=image, faces_blurred=0, detector_available=True)

    canvas = image.copy()
    height, width = canvas.shape[:2]
    for x, y, w, h in faces:
        pad_x, pad_y = int(w * margin), int(h * margin)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(width, x + w + pad_x), min(height, y + h + pad_y)
        region = canvas[y1:y2, x1:x2]
        if region.size == 0:
            continue
        size = max(kernel, (max(x2 - x1, y2 - y1) // 4) | 1)
        size = size if size % 2 else size + 1
        canvas[y1:y2, x1:x2] = cv2.GaussianBlur(region, (size, size), 0)
    return BlurResult(image=canvas, faces_blurred=len(faces), detector_available=True)
