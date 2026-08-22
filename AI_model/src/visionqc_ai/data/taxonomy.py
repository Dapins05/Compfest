"""Taksonomi cacat dan pemetaan label dari dataset sumber.

MVTec AD dan PKU-GoodsAD menamai cacat lewat nama folder, sedangkan VisA
memakai frasa bebas di image_anno.csv dengan ejaan Britania dan Amerika yang
bercampur. Seluruh pemetaan ke enam kelas VisionQC ditulis eksplisit di modul
ini supaya bisa diperiksa ulang.
"""

from __future__ import annotations

import re

DEFECT_CLASSES: tuple[str, ...] = (
    "pecah",
    "gores",
    "noda",
    "kotor",
    "deformasi",
    "terbuka",
)

CLASS_IDS: dict[str, int] = {name: i for i, name in enumerate(DEFECT_CLASSES)}

CLASS_LABELS_ID: dict[str, str] = {
    "pecah": "Pecah / Retak",
    "gores": "Gores",
    "noda": "Noda Warna",
    "kotor": "Kontaminasi",
    "deformasi": "Deformasi",
    "terbuka": "Segel Terbuka",
}

CLASS_SEVERITY: dict[str, float] = {
    "pecah": 0.9,
    "gores": 0.4,
    "noda": 0.6,
    "kotor": 1.0,
    "deformasi": 0.5,
    "terbuka": 0.9,
}


_RAW_TO_CLASS: dict[str, str] = {
    "broken_large": "pecah",
    "broken_small": "pecah",
    "chunk of gum missing": "pecah",
    "corner missing": "pecah",
    "small cracks": "pecah",
    "corner or edge breakage": "pecah",
    "corner and edge breakage": "pecah",
    "middle breakage": "pecah",
    "breakage down the middle": "pecah",
    "chip around edge and corner": "pecah",
    "small chip around edge": "pecah",
    "small holes": "pecah",
    "scratch": "gores",
    "scratches": "gores",
    "small scratches": "gores",
    "similar color spot": "noda",
    "different color spot": "noda",
    "same color spot": "noda",
    "color spot similar to the object": "noda",
    "discolor": "noda",
    "burnt": "noda",
    "contamination": "kotor",
    "foreign particals on candle": "kotor",
    "stuck together": "deformasi",
    "fryum stuck together": "deformasi",
    "bent": "deformasi",
    "misshape": "deformasi",
    "bubble": "deformasi",
    # PKU-GoodsAD. Nama folder memakai garis bawah dan tidak beririsan dengan
    # label MVTec maupun VisA, jadi tabelnya cukup disambung di sini.
    "broken": "pecah",
    "surface_damage": "gores",
    "surface_anomaly": "noda",
    "deformation": "deformasi",
    # Empat label berikut sama-sama berarti kemasan sudah tidak tersegel.
    # Dijadikan satu kelas karena akibatnya di lini produksi identik: produk
    # tidak boleh dikirim, berapa pun luas bagian yang terbuka.
    "cap_open": "terbuka",
    "cap_half_open": "terbuka",
    "opened": "terbuka",
    "straw_missing": "terbuka",
}

IGNORED_RAW_LABELS: frozenset[str] = frozenset({"other", "normal", "good"})

_WHITESPACE = re.compile(r"\s+")


def normalize_raw_label(label: str) -> str:
    """Rapikan label mentah agar bisa dicocokkan dengan tabel pemetaan.

    Menyeragamkan huruf besar/kecil, spasi, dan ejaan Britania → Amerika
    (`colour` → `color`), karena VisA memakai keduanya secara bergantian.
    """
    text = _WHITESPACE.sub(" ", label.strip().lower())
    return text.replace("colour", "color")


def map_raw_label(label: str) -> str | None:
    """Petakan satu label mentah ke nama kelas VisionQC.

    Mengembalikan None bila label sengaja diabaikan, dan melempar KeyError
    bila labelnya tidak dikenal sama sekali. Gagal keras lebih baik daripada
    diam-diam membuang data.
    """
    key = normalize_raw_label(label)
    if key in IGNORED_RAW_LABELS:
        return None
    try:
        return _RAW_TO_CLASS[key]
    except KeyError as exc:
        raise KeyError(
            f"label mentah tidak dikenal: {label!r} (ternormalisasi: {key!r}). "
            "Tambahkan ke _RAW_TO_CLASS di taxonomy.py."
        ) from exc


def class_id(label: str) -> int | None:
    """Petakan label mentah langsung ke id kelas YOLO."""
    name = map_raw_label(label)
    return None if name is None else CLASS_IDS[name]
