"""Uji lapisan privasi.

Seluruh gambar uji dibuat sintetik. Memakai foto orang sungguhan untuk menguji
fitur privasi bertentangan dengan tujuan fitur itu sendiri.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from visionqc_ai.privacy import (
    AuditRecord,
    blur_faces,
    ephemeral_buffer,
    filter_text,
    hash_image,
    strip_metadata,
)
from visionqc_ai.privacy.exif import has_metadata

BATCH_PATTERNS = (
    r"^[A-Z]{1,3}\d{4,10}(?:[-/]\d{2,5})?$",
    r"^\d{2}[/.]\d{2}[/.]\d{2,4}$",
    r"^(?:LOT|BN|EXP)[-: ]?[A-Z0-9]{4,12}$",
)


@pytest.fixture
def gambar() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)


def encode(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def test_metadata_dibersihkan(gambar: np.ndarray) -> None:
    hasil = strip_metadata(encode(gambar))
    assert not has_metadata(hasil)


def test_pembersihan_mempertahankan_ukuran(gambar: np.ndarray) -> None:
    """Piksel boleh berubah karena dikodekan ulang, dimensinya tidak boleh."""
    hasil = strip_metadata(encode(gambar))
    decoded = cv2.imdecode(np.frombuffer(hasil, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == gambar.shape


def test_berkas_rusak_ditolak() -> None:
    with pytest.raises(ValueError):
        strip_metadata(b"ini bukan gambar")


def test_gambar_tanpa_wajah_tidak_berubah(gambar: np.ndarray) -> None:
    hasil = blur_faces(gambar)
    assert hasil.faces_blurred == 0


def test_peredaman_melaporkan_ketersediaan(gambar: np.ndarray) -> None:
    """Bila model tidak ada, lapisan ini harus menyatakannya, bukan diam."""
    from pathlib import Path

    hasil = blur_faces(gambar, model_path=Path("tidak/ada/model.onnx"))
    assert hasil.detector_available is False
    assert "tidak aktif" in hasil.note


def test_ocr_hanya_meloloskan_kode_batch() -> None:
    """Format B240815-021 berasal dari contoh pada kontrak API."""
    kandidat = ["B240815-021", "Budi Santoso", "15/08/26", "PT Contoh", "x"]
    lolos, dibuang = filter_text(kandidat, BATCH_PATTERNS)
    assert lolos == ["B240815-021", "15/08/26"]
    assert dibuang == 3


def test_ocr_membuang_alamat_dan_nama_perusahaan() -> None:
    kandidat = ["Jl. Merdeka 17", "PT Contoh Sejahtera", "LOT-4F2A91"]
    lolos, _ = filter_text(kandidat, BATCH_PATTERNS)
    assert lolos == ["LOT-4F2A91"]


def test_ocr_membuang_nama_orang() -> None:
    lolos, _ = filter_text(["Siti Aminah", "Andi Wijaya"], BATCH_PATTERNS)
    assert lolos == []


def test_buffer_dinolkan_setelah_dipakai() -> None:
    array = np.full((8, 8, 3), 255, dtype=np.uint8)
    with ephemeral_buffer(array) as working:
        assert working.max() == 255
    assert array.max() == 0


def test_buffer_dinolkan_walau_terjadi_galat() -> None:
    """Kegagalan di tengah pemrosesan tidak boleh menyisakan gambar di memori."""
    array = np.full((8, 8, 3), 255, dtype=np.uint8)
    with pytest.raises(RuntimeError), ephemeral_buffer(array):
        raise RuntimeError("gagal di tengah jalan")
    assert array.max() == 0


def test_audit_tidak_memuat_gambar(gambar: np.ndarray) -> None:
    raw = encode(gambar)
    record = AuditRecord(
        image_sha256=hash_image(raw), verdict="PASS", latency_ms=140, faces_blurred=0
    )
    payload = record.to_dict()
    assert len(payload["image_sha256"]) == 64
    assert not any(isinstance(v, bytes) for v in payload.values())


def test_hash_berbeda_untuk_gambar_berbeda(gambar: np.ndarray) -> None:
    lain = np.roll(gambar, 5, axis=0)
    assert hash_image(encode(gambar)) != hash_image(encode(lain))
