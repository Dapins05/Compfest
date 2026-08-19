"""Pemrosesan gambar yang tidak meninggalkan jejak.

Gambar produk dapat memuat wajah operator, tata letak lini produksi, atau kode
internal. Cara paling sederhana untuk menjaganya adalah tidak pernah
menyimpannya: gambar hidup di memori selama satu permintaan, lalu buffernya
ditimpa nol sebelum dilepas.

Menimpa dengan nol tidak menjamin apa pun terhadap penyerang yang menguasai
mesin, karena Python dapat menyalin data di balik layar. Yang dijamin adalah
tidak ada gambar yang sengaja disimpan, dan buffer yang masih dipegang program
tidak lagi berisi gambar.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np


@contextmanager
def ephemeral_buffer(image: np.ndarray) -> Iterator[np.ndarray]:
    """Pinjamkan larik gambar lalu timpa isinya dengan nol setelah selesai.

    with ephemeral_buffer(image) as working:
        hasil = periksa(working)
    """
    try:
        yield image
    finally:
        image.fill(0)


def zero_out(*arrays: np.ndarray) -> None:
    """Timpa beberapa larik sekaligus dengan nol."""
    for array in arrays:
        array.fill(0)
