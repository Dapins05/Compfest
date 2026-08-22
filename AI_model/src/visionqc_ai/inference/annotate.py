"""Penggambaran anotasi di atas gambar asli.

Keluaran sistem harus dapat dipahami tanpa membaca angka. Kotak pembatas
memberi tahu di mana cacatnya, arsiran mask memberi tahu bentuk persisnya, dan
pita ringkas di bagian atas memberi tahu keputusan akhirnya.

Warna kelas diambil dari satu tempat agar konsisten antara lembar kontak
dataset, gambar hasil evaluasi, dan tampilan yang dilihat pengguna.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence

import cv2
import numpy as np

from visionqc_ai.data.taxonomy import CLASS_LABELS_ID, DEFECT_CLASSES

#: Warna per kelas dalam urutan BGR, mengikuti urutan DEFECT_CLASSES.
#: Jumlahnya WAJIB sama dengan DEFECT_CLASSES; dijaga oleh tests/test_annotate.py.
CLASS_COLORS_BGR: tuple[tuple[int, int, int], ...] = (
    (47, 50, 220),
    (210, 139, 38),
    (0, 137, 181),
    (130, 54, 211),
    (0, 153, 133),
    (30, 200, 230),
)

#: Warna cadangan bila sebuah kelas belum punya warna sendiri.
FALLBACK_COLOR_BGR: tuple[int, int, int] = (128, 128, 128)

VERDICT_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "PASS": (60, 160, 60),
    "REJECT": (47, 50, 220),
    "REVIEW": (0, 165, 235),
}


def class_color(class_name: str) -> tuple[int, int, int]:
    """Warna BGR untuk sebuah kelas cacat.

    IndexError ikut ditangkap, bukan hanya ValueError. Ketika kelas keenam
    ditambahkan, daftar warnanya sempat tertinggal dan seluruh penggambaran
    anotasi gagal dengan IndexError, padahal modelnya sendiri bekerja normal.
    Kekurangan warna tidak boleh menjatuhkan hasil inspeksi.
    """
    try:
        return CLASS_COLORS_BGR[DEFECT_CLASSES.index(class_name)]
    except (ValueError, IndexError):
        return FALLBACK_COLOR_BGR


def draw_defects(
    image: np.ndarray,
    boxes: Sequence[tuple[str, tuple[int, int, int, int], float]],
    polygons: Sequence[tuple[str, np.ndarray]] = (),
    *,
    mask_alpha: float = 0.35,
) -> np.ndarray:
    """Gambar mask dan kotak pembatas di atas salinan gambar.

    Mask digambar lebih dulu sebagai arsiran tembus pandang, baru kotak di
    atasnya, supaya garis kotak tidak tertutup arsiran.
    """
    canvas = image.copy()

    if polygons:
        overlay = canvas.copy()
        for class_name, points in polygons:
            if len(points) >= 3:
                cv2.fillPoly(
                    overlay, [points.astype(np.int32)], class_color(class_name)
                )
        canvas = cv2.addWeighted(overlay, mask_alpha, canvas, 1.0 - mask_alpha, 0)

    for class_name, (x, y, w, h), confidence in boxes:
        colour = class_color(class_name)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 2)
        caption = f"{CLASS_LABELS_ID.get(class_name, class_name)} {confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(text_h + 6, y)
        cv2.rectangle(
            canvas, (x, top - text_h - 6), (x + text_w + 6, top), colour, cv2.FILLED
        )
        cv2.putText(
            canvas,
            caption,
            (x + 3, top - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def draw_verdict_banner(image: np.ndarray, verdict: str, reason: str) -> np.ndarray:
    """Tambahkan pita keputusan di bagian atas gambar.

    Alasan dipotong bila terlalu panjang; teks utuhnya tetap dikirim ke Backend
    lewat medan ``reason`` sehingga tidak ada informasi yang hilang.
    """
    height, width = image.shape[:2]
    banner_height = 46
    canvas = np.zeros((height + banner_height, width, 3), dtype=np.uint8)
    canvas[banner_height:] = image
    canvas[:banner_height] = VERDICT_COLORS_BGR.get(verdict, (90, 90, 90))

    cv2.putText(
        canvas,
        verdict,
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    limit = max(0, (width - 130) // 8)
    text = reason if len(reason) <= limit else reason[: max(0, limit - 3)] + "..."
    cv2.putText(
        canvas,
        text,
        (120, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def to_base64_jpeg(image: np.ndarray, *, quality: int = 90) -> str:
    """Kodekan gambar menjadi JPEG base64 siap disematkan di halaman web."""
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("gagal mengodekan gambar hasil anotasi")
    payload = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"
