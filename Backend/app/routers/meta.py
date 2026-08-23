"""Endpoint keterangan: gambar contoh dan informasi model.

`/model-info` sebelumnya mengembalikan nilai tiruan `mock-v0`. Nilai tiruan
pada endpoint yang justru dipakai memeriksa versi model adalah hal yang
berbahaya: ia terlihat seperti jawaban sah, sehingga tidak ada yang menyadari
bahwa yang dilaporkan bukan model yang benar-benar melayani. Sekarang isinya
dibaca dari pipeline dan daftar model yang sungguhan.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter
from visionqc_ai import load_pipeline
from visionqc_ai.inference.pipeline import MODEL_VERSION

from app.config import get_settings
from app.schemas import ModelInfo, SampleImage

log = logging.getLogger(__name__)

router = APIRouter()

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


@router.get("/samples", response_model=list[SampleImage])
def get_samples() -> list[SampleImage]:
    """Gambar contoh yang disediakan supaya sistem dapat dicoba tanpa unggahan."""
    samples_dir = get_settings().samples_dir
    if not samples_dir.is_dir():
        return []
    return [
        SampleImage(name=path.name, url=f"/samples/{path.name}")
        for path in sorted(samples_dir.iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]


@router.get("/model-info", response_model=ModelInfo)
def get_model_info() -> ModelInfo:
    """Keterangan model yang sedang dilayani, dibaca dari pipeline yang hidup."""
    pipeline = load_pipeline()
    settings = get_settings()

    release = "tidak diketahui"
    manifest = settings.ai_root / "models" / "models.json"
    if manifest.is_file():
        try:
            release = json.loads(manifest.read_text(encoding="utf-8"))["release_tag"]
        except (json.JSONDecodeError, KeyError) as error:
            # Daftar model yang tidak terbaca tidak boleh menjatuhkan endpoint
            # ini; keterangan versi berguna, tetapi bukan syarat melayani.
            log.warning("models.json tidak terbaca: %s", error)

    return ModelInfo(
        model_name="yolo11n-defect + yolo11n-seg-defect + padim-combined",
        version=f"{MODEL_VERSION} ({release})",
        dataset="MVTec AD, VisA, PKU-GoodsAD - taksonomi 6 kelas",
        trained_at="2026-08-23",
        # Sengaja dikosongkan. Metrik yang dilaporkan lewat API harus berasal
        # dari pengukuran yang tercatat, dan tempatnya di AI_model/EXPERIMENTS.md
        # bukan disalin tangan ke sini, karena salinan akan basi tanpa ketahuan.
        metrics={},
        components={
            "detection": pipeline.ready,
            "segmentation": pipeline.segmentation_available,
            "anomaly": pipeline.anomaly_available,
            "face_blur": pipeline.face_blur_available,
            "ocr": pipeline.ocr_available,
        },
    )
