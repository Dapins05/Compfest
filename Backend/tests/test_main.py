"""Pengujian layanan API terhadap pipeline AI yang sungguhan.

Pipeline tidak ditiru di sini. Tiruan hanya akan membuktikan bahwa tiruannya
bekerja, sedangkan yang perlu dijaga justru sambungan antara Backend dan modul
AI: bahwa kontraknya cocok, bahwa berkas keliru dibalas 400 dan bukan 500, dan
bahwa keputusan yang keluar hanya PASS atau REJECT.

Gambar uji memakai berkas contoh 900x900 di `samples/`. Berkas PNG 1x1 yang
dipakai sebelumnya tidak lagi memadai: modul AI menolak gambar di bawah 224
piksel, jadi berkas itu tidak pernah sampai ke model dan pengujiannya tidak
membuktikan apa pun tentang jalur yang sesungguhnya dipakai.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
VALID_IMAGE = SAMPLES / "bottle_good_01.png"


@pytest.fixture(scope="module")
def image_bytes() -> bytes:
    return VALID_IMAGE.read_bytes()


def test_health_check_melaporkan_komponen() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    # Status boleh `degraded` bila bobot belum diunduh; yang tidak boleh adalah
    # melaporkan `ok` tanpa memeriksa apa pun.
    assert body["status"] in {"ok", "degraded"}
    assert "detection" in body["components"]


def test_inspect_menolak_berkas_bukan_gambar() -> None:
    response = client.post(
        "/api/v1/inspect",
        files={"file": ("test.txt", b"bukan gambar", "text/plain")},
    )
    assert response.status_code == 400
    assert "tidak didukung" in response.json()["detail"]


def test_inspect_menolak_berkas_kosong() -> None:
    response = client.post(
        "/api/v1/inspect", files={"file": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 400


def test_inspect_menolak_gambar_rusak_dengan_400_bukan_500() -> None:
    """Berkas rusak adalah kesalahan pengguna, bukan kegagalan server.

    Ini yang dijaga: penangan galat global sempat menangkap HTTPException juga,
    sehingga setiap penolakan 400 berubah menjadi 500.
    """
    response = client.post(
        "/api/v1/inspect",
        files={"file": ("fake.png", b"ini bukan data gambar asli", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]


def test_inspect_mengembalikan_kontrak_lengkap(image_bytes: bytes) -> None:
    response = client.post(
        "/api/v1/inspect",
        files={"file": ("bottle_good_01.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    for field in (
        "verdict",
        "reason",
        "confidence",
        "batch_code",
        "defects",
        "defect_area_pct",
        "anomaly",
        "decision",
        "annotated_image_base64",
        "model_version",
        "latency_ms",
    ):
        assert field in body, f"medan {field} hilang dari respons"


def test_keputusan_hanya_pass_atau_reject(image_bytes: bytes) -> None:
    """Mesin keputusan berjalan pada mode biner dan tidak pernah menahan.

    REVIEW pernah menjadi nilai yang sah dan masih tercantum pada skema lama
    Backend. Pengujian ini mengunci perilaku yang berlaku sekarang, supaya
    Frontend tidak perlu menyiapkan cabang yang tidak akan pernah terjadi.
    """
    response = client.post(
        "/api/v1/inspect",
        files={"file": ("bottle_good_01.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["verdict"] in {"PASS", "REJECT"}


def test_cacat_membawa_label_tampilan(image_bytes: bytes) -> None:
    """Setiap cacat harus membawa nama berbahasa Indonesia untuk ditampilkan."""
    response = client.post(
        "/api/v1/inspect",
        files={"file": ("bottle_broken_01.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    for defect in response.json()["defects"]:
        assert defect["label"], "cacat tanpa label tampilan"
        assert defect["type"]


def test_model_info_bukan_lagi_nilai_tiruan() -> None:
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "mock" not in body["version"].lower()
    assert "components" in body


def test_samples_mengembalikan_daftar() -> None:
    response = client.get("/api/v1/samples")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert all("name" in item and "url" in item for item in body)


def test_gambar_rusak_bertanda_tangan_sah_dibalas_400_bukan_500() -> None:
    """Berkas yang tanda tangannya sah tetapi isinya rusak tetap 400.

    Pemeriksaan format modul AI membaca tanda tangan berkas, sehingga PNG
    dengan header utuh dan isi rusak melewatinya dan baru gagal di lapisan
    privasi. Kegagalan di sana sempat keluar sebagai `ValueError` polos, yang
    dibaca layanan sebagai kegagalan tak terduga dan dibalas 500 - padahal
    penyebabnya berkas kiriman, bukan server.
    """
    rusak = b"\x89PNG\r\n\x1a\n" + bytes(4096)
    response = client.post(
        "/api/v1/inspect",
        files={"file": ("rusak.png", rusak, "image/png")},
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"]
