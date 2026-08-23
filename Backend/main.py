"""Layanan API VisionQC.

Pemrosesan bersifat sinkron sepenuhnya: satu permintaan membawa satu gambar,
model berjalan di dalam permintaan itu, dan satu respons lengkap dikembalikan.
Tidak ada antrean, pekerjaan latar, maupun basis data.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.config import get_settings
from app.routers import inspect, meta
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from visionqc_ai import load_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("visionqc.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Muat model sekali saat startup.

    Tanpa ini, bobot baru dimuat pada permintaan pertama dan pengguna pertama
    menunggu sekitar 4,8 detik alih-alih 0,25 detik. Kegagalan pemuatan juga
    akan baru ketahuan ketika ada pengguna yang gagal dilayani, bukan saat
    layanan dinyalakan.

    Layanan tetap dinyalakan meskipun model tidak tersedia. Endpoint kesehatan
    melaporkan keadaan itu apa adanya, dan `/inspect` membalas 503 dengan
    keterangan cara memperbaikinya. Mematikan seluruh layanan hanya karena
    bobot belum diunduh akan membuat penyebabnya lebih sulit ditemukan.
    """
    try:
        pipeline = load_pipeline()
        log.info(
            "model dimuat - deteksi %s, segmentasi %s, anomali %s, wajah %s, ocr %s",
            pipeline.ready,
            pipeline.segmentation_available,
            pipeline.anomaly_available,
            pipeline.face_blur_available,
            pipeline.ocr_available,
        )
        if not pipeline.ready:
            log.warning(
                "model deteksi TIDAK tersedia; jalankan "
                "AI_model/scripts/download_models.py"
            )
    except Exception as error:  # noqa: BLE001 - dilaporkan, tidak disembunyikan
        log.error("gagal memuat pipeline saat startup: %s", error)
    yield


app = FastAPI(
    title="VisionQC API",
    description="Inspeksi kualitas kemasan pangan dan minuman berbasis computer vision.",
    version="1.1.0",
    lifespan=lifespan,
)

# Frontend berjalan pada origin yang berbeda saat pengembangan. Daftarnya
# sengaja dibatasi ke localhost alih-alih memakai tanda bintang.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(inspect.router, prefix="/api/v1", tags=["inspect"])
app.include_router(meta.router, prefix="/api/v1", tags=["meta"])

_samples = get_settings().samples_dir
if _samples.is_dir():
    app.mount("/samples", StaticFiles(directory=str(_samples)), name="samples")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Teruskan HTTPException dengan kode aslinya.

    Penangan sebelumnya memasang `@app.exception_handler(Exception)` saja.
    Pada Starlette, penangan itu ikut menangkap `HTTPException`, sehingga
    setiap penolakan 400 yang sudah ditulis dengan cermat di router berubah
    menjadi 500 dengan pesan seragam. Akibatnya klien tidak dapat membedakan
    berkas yang salah dari layanan yang rusak.
    """
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Jaring pengaman terakhir; galatnya dicatat, bukan ditelan diam-diam."""
    log.exception("galat tak tertangani pada %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Terjadi kesalahan tak terduga di server. Silakan coba lagi."
        },
    )


@app.get("/healthz")
def health_check() -> dict[str, object]:
    """Kesiapan layanan beserta lapisan mana yang aktif.

    `status` menjadi `degraded`, bukan `ok`, ketika model deteksi tidak
    tersedia. Melaporkan `ok` pada layanan yang tidak dapat memeriksa apa pun
    akan menyembunyikan kegagalan dari siapa pun yang memantau.
    """
    try:
        pipeline = load_pipeline()
    except Exception as error:  # noqa: BLE001 - kesiapan harus tetap terjawab
        log.error("pipeline tidak dapat dimuat: %s", error)
        return {"status": "degraded", "detail": "pipeline tidak dapat dimuat"}

    return {
        "status": "ok" if pipeline.ready else "degraded",
        "components": {
            "detection": pipeline.ready,
            "segmentation": pipeline.segmentation_available,
            "anomaly": pipeline.anomaly_available,
            "face_blur": pipeline.face_blur_available,
            "ocr": pipeline.ocr_available,
        },
    }
