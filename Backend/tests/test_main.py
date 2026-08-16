from fastapi.testclient import TestClient
from main import app
import base64

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
    # gambar PNG 1x1 pixel valid (base64), cukup buat lolos validasi
    fake_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+A8AAQUBAScY42YAAAAASUVORK5CYII="
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

def test_inspect_rejects_empty_file():
    response = client.post(
        "/api/v1/inspect",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400


def test_inspect_rejects_corrupted_image():
    response = client.post(
        "/api/v1/inspect",
        files={"file": ("fake.png", b"ini bukan data gambar asli", "image/png")},
    )
    assert response.status_code == 400
    assert "rusak" in response.json()["detail"] or "korup" in response.json()["detail"]