"""Uji titik sambung antara modul AI dan Backend.

Berkas ini menjaga hal-hal yang baru terlihat ketika modul dipakai sungguhan
oleh sebuah layanan: berkas masukan yang tidak wajar, dua permintaan yang
datang bersamaan, dan penelusuran akar proyek ketika paket dipasang.
"""

from __future__ import annotations

import concurrent.futures as futures

import cv2
import numpy as np
import pytest

from visionqc_ai.inference.validation import (
    InvalidImageError,
    sniff_format,
    validate_bytes,
    validate_dimensions,
)

LIMITS = {
    "max_file_size_mb": 10,
    "allowed_formats": ["jpg", "jpeg", "png", "webp"],
    "min_dimension": 224,
    "max_dimension": 4096,
}


def _encode(height: int, width: int, extension: str = ".jpg") -> bytes:
    image = np.random.default_rng(0).integers(
        0, 255, (height, width, 3), dtype=np.uint8
    )
    return cv2.imencode(extension, image)[1].tobytes()


def test_format_dikenali_dari_isi_bukan_nama() -> None:
    assert sniff_format(_encode(256, 256, ".jpg")) == "jpg"
    assert sniff_format(_encode(256, 256, ".png")) == "png"
    assert sniff_format(_encode(256, 256, ".bmp")) == "bmp"
    assert sniff_format(b"bukan gambar sama sekali") == "unknown"


def test_format_di_luar_daftar_izin_ditolak() -> None:
    with pytest.raises(InvalidImageError, match="tidak diizinkan"):
        validate_bytes(_encode(256, 256, ".bmp"), LIMITS)


def test_format_yang_diizinkan_diterima() -> None:
    assert validate_bytes(_encode(256, 256, ".png"), LIMITS) == "png"


def test_berkas_kosong_ditolak() -> None:
    with pytest.raises(InvalidImageError, match="kosong"):
        validate_bytes(b"", LIMITS)


def test_berkas_terlalu_besar_ditolak() -> None:
    oversized = b"\xff\xd8\xff" + b"\x00" * (11 * 1024 * 1024)
    with pytest.raises(InvalidImageError, match="melebihi batas"):
        validate_bytes(oversized, LIMITS)


def test_dimensi_melebihi_batas_atas_ditolak() -> None:
    with pytest.raises(InvalidImageError, match="maksimum"):
        validate_dimensions(5000, 5000, LIMITS)


def test_dimensi_di_bawah_batas_bawah_ditolak() -> None:
    with pytest.raises(InvalidImageError, match="minimum"):
        validate_dimensions(100, 100, LIMITS)


def test_dimensi_dalam_rentang_diterima() -> None:
    validate_dimensions(640, 480, LIMITS)


def test_galat_masukan_dapat_dipisahkan_dari_galat_sistem() -> None:
    """Backend memetakan InvalidImageError ke 400 dan sisanya ke 500."""
    assert issubclass(InvalidImageError, ValueError)


def test_pendeteksi_wajah_aman_dipakai_banyak_thread() -> None:
    """Pendeteksi wajah dipakai ulang lintas permintaan.

    Menyetel ukuran masukan lalu menjalankan deteksi adalah dua langkah pada
    objek yang sama. Tanpa kunci, permintaan dengan ukuran gambar berbeda yang
    datang bersamaan menggugurkan proses di dalam OpenCV.
    """
    from visionqc_ai.privacy.face_blur import detect_faces

    sizes = [(240, 320), (480, 640), (720, 1280), (256, 256)] * 4
    images = [np.zeros((h, w, 3), dtype=np.uint8) for h, w in sizes]
    errors: list[BaseException] = []

    def run(image: np.ndarray) -> None:
        try:
            detect_faces(image)
        except BaseException as exc:
            errors.append(exc)

    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, images))

    assert not errors, f"deteksi wajah gagal saat bersamaan: {errors[0]}"


def test_inspeksi_bersamaan_memberi_hasil_yang_sama() -> None:
    """Dua permintaan bersamaan tidak boleh saling mengubah hasil.

    Backend FastAPI melayani endpoint sinkron di atas threadpool, sehingga
    keadaan ini pasti terjadi begitu ada dua pengguna. Uji dilewati bila bobot
    model belum diunduh, karena tanpa model tidak ada yang bisa dibandingkan.
    """
    from visionqc_ai.inference.pipeline import InspectionPipeline, default_project_root

    root = default_project_root()
    if not (root / "models" / "onnx" / "yolo11n-defect.onnx").is_file():
        pytest.skip("bobot model belum diunduh; jalankan scripts/download_models.py")

    pipeline = InspectionPipeline(project_root=root)
    payloads = [
        _encode(h, w) for h, w in ((320, 320), (480, 640), (256, 256), (400, 300))
    ]

    def summarise(result: object) -> tuple[str, int]:
        return (result.verdict, len(result.defects))

    expected = [summarise(pipeline.inspect(p)) for p in payloads]
    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        actual = [summarise(r) for r in pool.map(pipeline.inspect, payloads)]

    assert actual == expected
