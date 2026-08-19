"""Pencatatan yang tidak pernah menyimpan gambar.

Sistem tetap perlu dapat menjawab pertanyaan seperti "apakah gambar ini pernah
diperiksa" dan "berapa lama pemeriksaannya". Menjawabnya tidak memerlukan
gambarnya, cukup sidik jarinya.

Karena itu catatan hanya memuat SHA-256 dari gambar. Sidik jari itu cukup untuk
membuktikan bahwa dua gambar sama, tetapi tidak dapat dikembalikan menjadi
gambar. Yang tersimpan tidak dapat membocorkan apa pun tentang isinya.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def hash_image(image_bytes: bytes) -> str:
    """Sidik jari SHA-256 sebuah gambar."""
    return hashlib.sha256(image_bytes).hexdigest()


@dataclass(frozen=True)
class AuditRecord:
    """Satu baris catatan pemeriksaan, tanpa gambar sama sekali."""

    image_sha256: str
    verdict: str
    latency_ms: int
    faces_blurred: int = 0
    metadata_stripped: bool = True
    ocr_fields_discarded: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        """Satu baris ringkas untuk log."""
        return (
            f"{self.timestamp} sha256={self.image_sha256[:16]}... "
            f"verdict={self.verdict} latensi={self.latency_ms}ms "
            f"wajah_diburamkan={self.faces_blurred}"
        )
