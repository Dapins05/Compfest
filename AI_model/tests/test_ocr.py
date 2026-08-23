"""Pengujian pembaca kode batch.

Mesin OCR sungguhan tidak dipanggil di sini. Yang diuji adalah perilaku di
sekelilingnya: penyaringan daftar-izin, pemilihan kandidat, dan yang paling
penting, bahwa mesin yang tidak tersedia dinyatakan alih-alih menjatuhkan
inspeksi.
"""

from __future__ import annotations

import numpy as np
import pytest

from visionqc_ai.inference.ocr import BatchCodeReader

PATTERNS = (
    r"^[A-Z]{1,3}\d{4,10}(?:[-/]\d{2,5})?$",
    r"^\d{2}[/.]\d{2}[/.]\d{2,4}$",
    r"^(?:LOT|BN|EXP)[-: ]?[A-Z0-9]{4,12}$",
)


class _FakeEngine:
    """Menirukan bentuk keluaran PaddleOCR: [halaman][baris][kotak, (teks, skor)]."""

    def __init__(self, lines: list[tuple[str, float]]) -> None:
        self.lines = lines

    def ocr(self, image: np.ndarray, cls: bool = True) -> list:
        return [
            [
                [[[0, 0], [1, 0], [1, 1], [0, 1]], (text, score)]
                for text, score in self.lines
            ]
        ]


def _reader(lines: list[tuple[str, float]], **kwargs: object) -> BatchCodeReader:
    reader = BatchCodeReader(patterns=PATTERNS, **kwargs)  # type: ignore[arg-type]
    reader._engine = _FakeEngine(lines)
    reader._loaded = True
    return reader


@pytest.fixture
def image() -> np.ndarray:
    return np.zeros((32, 32, 3), dtype=np.uint8)


def test_mesin_tak_tersedia_tidak_menjatuhkan_inspeksi(image: np.ndarray) -> None:
    """Ini yang melindungi Backend: OCR mati harus jadi null, bukan pengecualian."""
    reader = BatchCodeReader(patterns=PATTERNS)
    reader._loaded = True
    reader._engine = None
    reader._note = "mesin OCR tidak tersedia"

    result = reader.read(image)

    assert result.batch_code is None
    assert result.available is False
    assert result.note


def test_kode_batch_terbaca_dari_teks_yang_cocok(image: np.ndarray) -> None:
    result = _reader([("B240815-021", 0.98)]).read(image)

    assert result.batch_code == "B240815-021"
    assert result.available is True


def test_nama_orang_dibuang_oleh_daftar_izin(image: np.ndarray) -> None:
    """Nama operator yang ikut terpotret tidak boleh keluar dari modul ini."""
    result = _reader([("B240815-021", 0.97), ("Budi Santoso", 0.96)]).read(image)

    assert result.batch_code == "B240815-021"
    assert "Budi Santoso" not in result.candidates
    assert result.discarded == 1


def test_teks_berkeyakinan_rendah_diabaikan(image: np.ndarray) -> None:
    result = _reader([("B240815-021", 0.11)]).read(image)

    assert result.batch_code is None


def test_kandidat_terpanjang_yang_dipilih(image: np.ndarray) -> None:
    """Kemasan memuat tanggal dan nomor lot sekaligus; nomor lot yang dicari."""
    result = _reader([("01/02/2026", 0.95), ("LOT-A1B2C3D4", 0.95)]).read(image)

    assert result.batch_code == "LOT-A1B2C3D4"


def test_tidak_ada_teks_yang_cocok_menghasilkan_null(image: np.ndarray) -> None:
    result = _reader([("Isi Bersih 500 ml", 0.99)]).read(image)

    assert result.batch_code is None
    assert result.available is True
    assert result.discarded == 1
