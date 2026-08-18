"""Menjalankan model deteksi pada sebuah split dan mengumpulkan hasilnya.

Ground truth dibaca langsung dari berkas label YOLO, bukan dari cache
Ultralytics, supaya evaluasi tetap sah walau dijalankan pada model yang sama
sekali tidak mengenal dataset ini, misalnya bobot pra-latih COCO.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visionqc_ai.evaluation.matching import Box, ImageMatch, match_image


@dataclass(frozen=True)
class EvaluationRun:
    """Hasil evaluasi satu model pada satu split."""

    tag: str
    weights: str
    split: str
    matches: list[ImageMatch]
    matches_class_agnostic: list[ImageMatch]
    image_count: int
    speed_ms: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "weights": self.weights,
            "split": self.split,
            "image_count": self.image_count,
            "speed_ms": self.speed_ms,
        }


def read_ground_truth(label_path: Path) -> list[Box]:
    """Baca label YOLO dan ubah dari xywh ternormalisasi menjadi xyxy."""
    if not label_path.exists():
        return []
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        xc, yc, w, h = (float(v) for v in parts[1:5])
        boxes.append(
            Box(
                class_id=class_id,
                x1=xc - w / 2,
                y1=yc - h / 2,
                x2=xc + w / 2,
                y2=yc + h / 2,
            )
        )
    return boxes


def _predictions_from_result(result: Any) -> list[Box]:
    boxes: list[Box] = []
    if result.boxes is None:
        return boxes
    for xyxyn, class_id, confidence in zip(
        result.boxes.xyxyn.tolist(),
        result.boxes.cls.tolist(),
        result.boxes.conf.tolist(),
        strict=True,
    ):
        boxes.append(
            Box(
                class_id=int(class_id),
                x1=float(xyxyn[0]),
                y1=float(xyxyn[1]),
                x2=float(xyxyn[2]),
                y2=float(xyxyn[3]),
                confidence=float(confidence),
            )
        )
    return boxes


def evaluate_detection(
    weights: Path,
    dataset_root: Path,
    *,
    tag: str,
    split: str = "test",
    imgsz: int = 640,
    conf: float = 0.25,
    iou_nms: float = 0.45,
    iou_match: float = 0.5,
    max_detections: int = 30,
    device: int | str = 0,
) -> EvaluationRun:
    """Jalankan model pada satu split lalu cocokkan hasilnya dengan ground truth."""
    from ultralytics import YOLO

    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"tidak ada gambar pada {image_dir}")

    model = YOLO(str(weights))
    matches: list[ImageMatch] = []
    agnostic: list[ImageMatch] = []
    speed_totals: dict[str, float] = {}

    for path in images:
        result = model.predict(
            source=str(path),
            imgsz=imgsz,
            conf=conf,
            iou=iou_nms,
            max_det=max_detections,
            device=device,
            verbose=False,
        )[0]
        predictions = _predictions_from_result(result)
        truth = read_ground_truth(label_dir / f"{path.stem}.txt")

        matches.append(
            match_image(path.stem, predictions, truth, iou_threshold=iou_match)
        )
        agnostic.append(
            match_image(
                path.stem,
                predictions,
                truth,
                iou_threshold=iou_match,
                class_agnostic=True,
            )
        )
        for key, value in (result.speed or {}).items():
            speed_totals[key] = speed_totals.get(key, 0.0) + float(value)

    count = len(images)
    return EvaluationRun(
        tag=tag,
        weights=str(weights),
        split=split,
        matches=matches,
        matches_class_agnostic=agnostic,
        image_count=count,
        speed_ms={k: round(v / count, 3) for k, v in speed_totals.items()},
    )
