import io
import random
import time
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError

from app.schemas import InspectionResult, Defect, BBox, AnomalyResult
from app.config import settings

router = APIRouter()


@router.post("/inspect", response_model=InspectionResult)
async def inspect_image(file: UploadFile = File(...)):
    # Validasi tipe file
    if file.content_type not in settings.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipe file {file.content_type} tidak didukung. Gunakan JPG atau PNG.",
        )

    # Validasi ukuran file
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"Ukuran file {size_mb:.1f}MB melebihi batas {settings.MAX_FILE_SIZE_MB}MB.",
        )

    # Validasi file kosong
    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="File gambar kosong.",
        )

    # Validasi gambar bisa dibuka (bukan file korup/rusak)
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="File tidak bisa dibaca sebagai gambar. Kemungkinan file rusak atau korup.",
        )

    start = time.time()

    # ⚠️ MOCK — hasil acak, akan diganti pipeline AI asli oleh Anggota 1 nanti
    verdict = random.choice(["PASS", "REJECT", "REVIEW"])

    defects = []
    if verdict != "PASS":
        defects.append(
            Defect(
                type="gores",
                bbox=BBox(x=120, y=88, w=64, h=40),
                confidence=0.87,
                area_pct=3.4,
            )
        )

    result = InspectionResult(
        verdict=verdict,
        reason="hasil mock — belum terhubung ke model AI asli",
        confidence=0.9 if verdict != "REVIEW" else 0.55,
        batch_code="B240815-021",
        defects=defects,
        anomaly=AnomalyResult(score=0.3, threshold=settings.T_ANOMALY, heatmap_base64=None),
        annotated_image_base64="data:image/jpeg;base64,mock",
        model_version="mock-v0",
        latency_ms=int((time.time() - start) * 1000),
    )
    return result