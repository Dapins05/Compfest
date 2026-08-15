import random
import time
from fastapi import APIRouter, UploadFile, File

from app.schemas import InspectionResult, Defect, BBox, AnomalyResult

router = APIRouter()


@router.post("/inspect", response_model=InspectionResult)
async def inspect_image(file: UploadFile = File(...)):
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
        anomaly=AnomalyResult(score=0.3, threshold=0.75, heatmap_base64=None),
        annotated_image_base64="data:image/jpeg;base64,mock",
        model_version="mock-v0",
        latency_ms=int((time.time() - start) * 1000),
    )
    return result