"""Registri kategori dataset dan lokasinya di data/raw.

Struktur hasil ekstraksi berbeda dari asumsi awal: MVTec terekstrak dengan
folder ganda (bottle/bottle/...) dan VisA memakai tata letak
<kategori>/Data/Images/. Hanya modul ini yang perlu tahu soal itu.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MVTEC = "mvtec"
VISA = "visa"

VISA_ROOT_DIRNAME = "VisA_20220922"


@dataclass(frozen=True)
class CategorySpec:
    """Satu kategori produk beserta lokasi dan perannya."""

    name: str
    source: str
    product_id: str
    learnable_at_640_pct: float

    def raw_dir(self, raw_root: Path) -> Path:
        """Folder mentah kategori ini di dalam data/raw."""
        if self.source == MVTEC:
            return raw_root / self.name / self.name
        return raw_root / VISA_ROOT_DIRNAME / self.name


CATEGORIES: dict[str, CategorySpec] = {
    "bottle": CategorySpec("bottle", MVTEC, "Botol minuman kaca", 100.0),
    "chewinggum": CategorySpec("chewinggum", VISA, "Permen karet kemasan", 96.3),
    "cashew": CategorySpec("cashew", VISA, "Kacang mete kemasan", 67.6),
    "pipe_fryum": CategorySpec("pipe_fryum", VISA, "Snack goreng (pipe fryum)", 63.9),
    "fryum": CategorySpec("fryum", VISA, "Snack goreng (fryum)", 29.8),
    "macaroni1": CategorySpec("macaroni1", VISA, "Pasta kering (makaroni A)", 37.8),
    "macaroni2": CategorySpec("macaroni2", VISA, "Pasta kering (makaroni B)", 31.8),
}


def get_category(name: str) -> CategorySpec:
    """Ambil spesifikasi kategori, dengan pesan jelas bila salah ketik."""
    try:
        return CATEGORIES[name]
    except KeyError as exc:
        raise KeyError(
            f"kategori {name!r} tidak dikenal. Pilihan: {sorted(CATEGORIES)}"
        ) from exc
