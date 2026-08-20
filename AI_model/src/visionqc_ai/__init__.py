"""Modul AI VisionQC untuk inspeksi kualitas kemasan pangan dan minuman.

Backend cukup memakai dua nama dari sini:

    from visionqc_ai import load_pipeline, run_inspection, InspectionResult

    load_pipeline()                       saat startup, agar model siap
    result: InspectionResult = run_inspection(image_bytes)

Galat masukan dilaporkan sebagai InvalidImageError dan pantas dijawab 400;
galat lain berarti kegagalan sistem.
"""

from visionqc_ai.inference.pipeline import (
    InspectionPipeline,
    load_pipeline,
    run_inspection,
)
from visionqc_ai.inference.validation import InvalidImageError
from visionqc_ai.schemas import (
    AnomalyResult,
    BBox,
    DecisionDetail,
    Defect,
    InspectionResult,
)

__version__ = "1.0.0"

__all__ = [
    "AnomalyResult",
    "BBox",
    "DecisionDetail",
    "Defect",
    "InspectionPipeline",
    "InspectionResult",
    "InvalidImageError",
    "load_pipeline",
    "run_inspection",
]
