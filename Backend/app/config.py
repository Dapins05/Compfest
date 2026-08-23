"""Konfigurasi layanan.

Ambang model TIDAK disimpan di sini lagi. Sebelumnya berkas ini memuat
salinannya sendiri, dan salinan itu sudah bertentangan dengan nilai yang
benar-benar dipakai modul AI:

    T_ANOMALY = 0.75    nilai sesungguhnya 83.2714
    T_CONF    = 0.60    nilai sesungguhnya 0.35 (binary_threshold)

Angka di Backend tidak pernah dipakai menghitung apa pun, jadi tidak ada yang
rusak karenanya. Bahayanya lain: keduanya terbaca seperti parameter resmi, dan
siapa pun yang memeriksa kepatuhan R7.4 akan menemukan dua sumber yang
berselisih. Ambang sekarang hanya ada di `AI_model/configs/inference.yaml`.

Batas unggahan pun dibaca dari config yang sama, supaya Backend tidak menolak
gambar yang sebenarnya diterima modul AI, atau sebaliknya meneruskan gambar
yang pasti ditolaknya.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

# Akar modul AI. Dapat ditimpa lewat VISIONQC_ROOT bila Backend dijalankan
# dari tata letak folder yang berbeda, misalnya di dalam image Docker.
DEFAULT_AI_ROOT = Path(__file__).resolve().parents[2] / "AI_model"

# Tipe MIME yang diterima, diturunkan dari format yang didukung modul AI.
_MIME_BY_FORMAT = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}


class Settings(BaseModel):
    """Pengaturan layanan yang dibaca sekali saat startup."""

    ai_root: Path
    max_file_size_mb: int
    allowed_content_types: tuple[str, ...]
    samples_dir: Path


def _ai_root() -> Path:
    import os

    override = os.environ.get("VISIONQC_ROOT")
    return Path(override).resolve() if override else DEFAULT_AI_ROOT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Baca konfigurasi sekali lalu pakai ulang."""
    root = _ai_root()
    config_path = root / "configs" / "inference.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)

    limits = config.get("input", {})
    formats = limits.get("allowed_formats", ["jpg", "jpeg", "png"])
    mimes = tuple(
        dict.fromkeys(_MIME_BY_FORMAT[f] for f in formats if f in _MIME_BY_FORMAT)
    )

    return Settings(
        ai_root=root,
        max_file_size_mb=int(limits.get("max_file_size_mb", 10)),
        allowed_content_types=mimes,
        samples_dir=Path(__file__).resolve().parents[1] / "samples",
    )
