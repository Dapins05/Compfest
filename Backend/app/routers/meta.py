import os
from fastapi import APIRouter

from app.schemas import SampleImage, ModelInfo

router = APIRouter()

SAMPLES_DIR = "samples"


@router.get("/samples", response_model=list[SampleImage])
def get_samples():
    files = [
        f for f in os.listdir(SAMPLES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return [
        SampleImage(name=f, url=f"/samples/{f}")
        for f in files
    ]


@router.get("/model-info", response_model=ModelInfo)
def get_model_info():
    # ⚠️ MOCK — akan diisi data asli setelah Anggota 1 selesai fine-tuning (lihat ml/EXPERIMENTS.md)
    return ModelInfo(
        model_name="yolo11n-defect",
        version="mock-v0",
        dataset="MVTec AD (belum final)",
        trained_at=None,
        metrics={},
    )