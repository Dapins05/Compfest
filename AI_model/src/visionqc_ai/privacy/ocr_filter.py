"""Penyaring keluaran OCR memakai daftar-izin.

OCR membaca apa pun yang terlihat, termasuk nama pada seragam, papan nama, dan
tulisan tangan yang kebetulan masuk bingkai. Yang dibutuhkan sistem ini hanya
kode batch.

Karena itu keluaran OCR disaring dengan **daftar-izin**: hanya teks yang cocok
pola kode batch yang dipertahankan, sisanya dibuang. Daftar-tolak akan selalu
tertinggal dari bentuk teks tak terduga yang mungkin muncul.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


def compile_patterns(patterns: Sequence[str]) -> list[re.Pattern[str]]:
    """Susun pola daftar-izin dari config."""
    return [re.compile(pattern) for pattern in patterns]


def filter_text(
    candidates: Sequence[str], patterns: Sequence[str]
) -> tuple[list[str], int]:
    """Pertahankan hanya teks yang cocok salah satu pola daftar-izin.

    Mengembalikan teks yang lolos beserta jumlah yang dibuang, sehingga
    banyaknya teks yang disaring dapat ikut dilaporkan pada audit privasi.
    """
    compiled = compile_patterns(patterns)
    kept = [
        text
        for text in (c.strip() for c in candidates)
        if text and any(pattern.match(text) for pattern in compiled)
    ]
    return kept, len(candidates) - len(kept)
