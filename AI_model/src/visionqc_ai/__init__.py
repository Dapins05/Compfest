"""Modul AI VisionQC untuk inspeksi kualitas kemasan pangan dan minuman.

Backend cukup memakai dua nama dari sini:

    from visionqc_ai import run_inspection, InspectionResult

    result: InspectionResult = run_inspection(image_bytes)
"""

from visionqc_ai.inference.pipeline import InspectionPipeline, run_inspection
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
    "run_inspection",
]
