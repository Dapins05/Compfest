"""Pemeriksaan berkas masukan sebelum gambar diproses.

Batas ukuran, format, dan dimensi sudah dinyatakan pada `configs/inference.yaml`
sejak awal, tetapi sebelumnya hanya dimensi terkecil yang benar-benar diperiksa.
Batas yang tertulis tetapi tidak ditegakkan lebih berbahaya daripada batas yang
tidak pernah ada, karena pemanggil akan mengandalkannya.

Galat masukan dipisahkan dari galat internal lewat :class:`InvalidImageError`
supaya Backend dapat membedakan kesalahan pengguna, yang pantas dijawab 400,
dari kegagalan sistem yang pantas dijawab 500.
"""

from __future__ import annotations

from typing import Any


class InvalidImageError(ValueError):
    """Berkas yang dikirim pemanggil tidak memenuhi syarat.

    Ini kesalahan pada masukan, bukan pada sistem.
    """


_SIGNATURES: tuple[tuple[str, bytes], ...] = (
    ("jpg", b"\xff\xd8\xff"),
    ("png", b"\x89PNG\r\n\x1a\n"),
    ("gif", b"GIF8"),
    ("bmp", b"BM"),
    ("tiff", b"II*\x00"),
    ("tiff", b"MM\x00*"),
)

_ALIASES = {"jpeg": "jpg"}


def sniff_format(data: bytes) -> str:
    """Kenali format gambar dari sidik awal berkasnya.

    Ekstensi nama berkas tidak dipercaya karena berasal dari pemanggil dan
    dapat menyesatkan. Yang diperiksa adalah isi berkasnya sendiri.
    """
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    for name, signature in _SIGNATURES:
        if data.startswith(signature):
            return name
    return "unknown"


def _allowed(limits: dict[str, Any]) -> set[str]:
    raw = limits.get("allowed_formats") or []
    return {_ALIASES.get(str(f).lower(), str(f).lower()) for f in raw}


def validate_bytes(data: bytes, limits: dict[str, Any]) -> str:
    """Periksa ukuran dan format berkas. Kembalikan format yang terdeteksi."""
    if not data:
        raise InvalidImageError("berkas kosong")

    max_mb = limits.get("max_file_size_mb")
    if max_mb is not None:
        size_mb = len(data) / (1024 * 1024)
        if size_mb > float(max_mb):
            raise InvalidImageError(
                f"ukuran berkas {size_mb:.1f} MB melebihi batas {max_mb} MB"
            )

    detected = sniff_format(data)
    allowed = _allowed(limits)
    if allowed and detected not in allowed:
        raise InvalidImageError(
            f"format {detected} tidak diizinkan; "
            f"yang diterima {', '.join(sorted(allowed))}"
        )
    return detected


def validate_dimensions(height: int, width: int, limits: dict[str, Any]) -> None:
    """Periksa dimensi gambar setelah berhasil diurai."""
    minimum = limits.get("min_dimension")
    if minimum is not None and min(height, width) < int(minimum):
        raise InvalidImageError(
            f"dimensi terkecil {min(height, width)} piksel, minimum {minimum}"
        )

    maximum = limits.get("max_dimension")
    if maximum is not None and max(height, width) > int(maximum):
        raise InvalidImageError(
            f"dimensi terbesar {max(height, width)} piksel, maksimum {maximum}"
        )
