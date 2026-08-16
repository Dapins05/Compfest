from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_inspect_rejects_non_image_file():
    response = client.post(
        "/api/v1/inspect",
        files={"file": ("test.txt", b"bukan gambar", "text/plain")},
    )
    assert response.status_code == 400
    assert "tidak didukung" in response.json()["detail"]


def test_inspect_accepts_valid_image():
    # gambar PNG 1x1 pixel minimal, cukup buat lolos validasi tipe & ukuran
    fake_png = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
        "de0000000c4944415478da6360000000020001e221bc330000000049454e44ae426082"
    )
    response = client.post(
        "/api/v1/inspect",
        files={"file": ("test.png", fake_png, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] in ["PASS", "REJECT", "REVIEW"]


def test_model_info_returns_valid_schema():
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "model_name" in body
    assert "version" in body


def test_samples_returns_list():
    response = client.get("/api/v1/samples")
    assert response.status_code == 200
    assert isinstance(response.json(), list)