from typing import Literal
from pydantic import BaseModel


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class Defect(BaseModel):
    type: str
    bbox: BBox
    confidence: float
    area_pct: float | None = None


class AnomalyResult(BaseModel):
    score: float
    threshold: float
    heatmap_base64: str | None = None


class InspectionResult(BaseModel):
    verdict: Literal["PASS", "REJECT", "REVIEW"]
    reason: str
    confidence: float | None
    batch_code: str | None
    defects: list[Defect]
    anomaly: AnomalyResult | None
    annotated_image_base64: str
    model_version: str
    latency_ms: int

class SampleImage(BaseModel):
    name: str
    url: str


class ModelInfo(BaseModel):
    model_name: str
    version: str
    dataset: str
    trained_at: str | None = None
    metrics: dict[str, float] = {}