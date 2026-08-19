"""Pembersihan metadata gambar.

Berkas foto membawa jauh lebih banyak daripada pikselnya: koordinat GPS, merek
dan nomor seri perangkat, waktu pengambilan, bahkan kadang nama pemilik. Untuk
inspeksi kualitas, tidak satu pun dari itu dibutuhkan, sementara semuanya dapat
dipakai mengidentifikasi orang atau lokasi pabrik.

Pendekatan yang dipakai adalah daftar-tolak menyeluruh, bukan daftar-izin.
Gambar diurai menjadi piksel lalu dikodekan ulang dari nol, sehingga apa pun
yang bukan piksel tidak ikut terbawa. Menghapus medan metadata satu per satu
selalu menyisakan risiko ada medan yang terlewat, terutama pada format yang
menyimpan data di tempat tak terduga.
"""

from __future__ import annotations

import cv2
import numpy as np


def strip_metadata(image_bytes: bytes, *, quality: int = 95) -> bytes:
    """Kembalikan gambar yang sama tanpa metadata apa pun.

    Melempar :class:`ValueError` bila berkas tidak dapat diurai sebagai gambar.
    """
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("berkas tidak dapat dibaca sebagai gambar")
    ok, encoded = cv2.imencode(
        ".jpg", decoded, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )
    if not ok:
        raise ValueError("gagal mengodekan ulang gambar")
    return encoded.tobytes()


def has_metadata(image_bytes: bytes) -> bool:
    """Perkiraan kasar apakah berkas JPEG masih membawa segmen metadata.

    Memeriksa keberadaan penanda APP1 yang dipakai EXIF. Ini pemeriksaan
    ringan untuk pengujian dan pelaporan, bukan pengurai metadata lengkap.
    """
    return b"\xff\xe1" in image_bytes[:4096] or b"Exif" in image_bytes[:4096]
