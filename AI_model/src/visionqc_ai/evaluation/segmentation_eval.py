"""Evaluasi model segmentasi beserta estimasi luas cacat.

Berbeda dari deteksi, pencocokan di sini memakai IoU **mask** dan bukan IoU
kotak pembatas. Kotak pembatas terlalu longgar untuk menilai segmentasi: dua
mask yang bentuknya sangat berbeda bisa saja punya kotak yang nyaris sama.

Modul ini juga menghitung luas cacat relatif terhadap luas gambar, yang menjadi
masukan bagi decision engine. Yang dilaporkan bukan hanya nilai dugaannya,
melainkan juga galat terhadap luas sebenarnya, sehingga sistem dapat menyatakan
luas cacat beserta ketidakpastiannya.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionqc_ai.evaluation.matching import ImageMatch


@dataclass(frozen=True)
class Polygon:
    """Poligon ternormalisasi terhadap ukuran gambar."""

    class_id: int
    points: tuple[tuple[float, float], ...]
    confidence: float = 1.0

    def to_mask(self, width: int, height: int) -> np.ndarray:
        canvas = np.zeros((height, width), dtype=np.uint8)
        pixels = np.array(
            [(round(x * width), round(y * height)) for x, y in self.points],
            dtype=np.int32,
        )
        if len(pixels) >= 3:
            cv2.fillPoly(canvas, [pixels], 1)
        return canvas


@dataclass
class AreaComparison:
    """Perbandingan luas cacat dugaan dengan luas sebenarnya."""

    image_id: str
    predicted_pct: float
    truth_pct: float

    @property
    def absolute_error(self) -> float:
        return abs(self.predicted_pct - self.truth_pct)


@dataclass
class SegmentationRun:
    """Hasil evaluasi satu model segmentasi pada satu split."""

    tag: str
    weights: str
    split: str
    matches: list[ImageMatch] = field(default_factory=list)
    mask_ious: list[float] = field(default_factory=list)
    areas: list[AreaComparison] = field(default_factory=list)
    image_count: int = 0
    speed_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "weights": self.weights,
            "split": self.split,
            "image_count": self.image_count,
            "speed_ms": self.speed_ms,
        }


def read_ground_truth_polygons(label_path: Path) -> list[Polygon]:
    """Baca label segmentasi YOLO berupa poligon ternormalisasi."""
    if not label_path.exists():
        return []
    polygons: list[Polygon] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        values = [float(v) for v in parts[1:]]
        points = tuple((values[i], values[i + 1]) for i in range(0, len(values) - 1, 2))
        polygons.append(Polygon(class_id=int(parts[0]), points=points))
    return polygons


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU dua mask biner."""
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return intersection / union if union > 0 else 0.0


def area_percentage(polygons: list[Polygon], width: int, height: int) -> float:
    """Luas gabungan seluruh poligon sebagai persentase luas gambar.

    Digabung lebih dulu, bukan dijumlahkan satu per satu, supaya cacat yang
    saling bertumpang tindih tidak terhitung dua kali.
    """
    if not polygons:
        return 0.0
    union = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        union |= polygon.to_mask(width, height)
    return 100.0 * float(np.count_nonzero(union)) / (width * height)


def match_masks(
    image_id: str,
    predictions: list[Polygon],
    ground_truth: list[Polygon],
    *,
    width: int,
    height: int,
    iou_threshold: float = 0.5,
) -> tuple[ImageMatch, list[float]]:
    """Cocokkan poligon prediksi dengan ground truth memakai IoU mask."""
    pred_masks = [p.to_mask(width, height) for p in predictions]
    truth_masks = [p.to_mask(width, height) for p in ground_truth]

    order = sorted(range(len(predictions)), key=lambda i: -predictions[i].confidence)
    used: set[int] = set()
    correct = [False] * len(predictions)
    detected = [False] * len(ground_truth)
    matched_ious: list[float] = []

    for pred_index in order:
        best_index = -1
        best_iou = iou_threshold
        for gt_index, truth in enumerate(ground_truth):
            if gt_index in used or truth.class_id != predictions[pred_index].class_id:
                continue
            score = mask_iou(pred_masks[pred_index], truth_masks[gt_index])
            if score >= best_iou:
                best_iou = score
                best_index = gt_index
        if best_index >= 0:
            used.add(best_index)
            correct[pred_index] = True
            detected[best_index] = True
            matched_ious.append(best_iou)

    match = ImageMatch(
        image_id=image_id,
        gt_detected=detected,
        gt_class_ids=[p.class_id for p in ground_truth],
        prediction_is_correct=correct,
        prediction_class_ids=[p.class_id for p in predictions],
    )
    return match, matched_ious


def _predictions_from_result(result: Any) -> list[Polygon]:
    polygons: list[Polygon] = []
    if result.masks is None or result.boxes is None:
        return polygons
    for xyn, class_id, confidence in zip(
        result.masks.xyn,
        result.boxes.cls.tolist(),
        result.boxes.conf.tolist(),
        strict=True,
    ):
        points = tuple((float(x), float(y)) for x, y in xyn)
        if len(points) >= 3:
            polygons.append(
                Polygon(
                    class_id=int(class_id),
                    points=points,
                    confidence=float(confidence),
                )
            )
    return polygons


def evaluate_segmentation(
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
) -> SegmentationRun:
    """Jalankan model segmentasi pada satu split dan kumpulkan hasilnya."""
    from ultralytics import YOLO

    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"tidak ada gambar pada {image_dir}")

    model = YOLO(str(weights))
    run = SegmentationRun(tag=tag, weights=str(weights), split=split)
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
        height, width = result.orig_shape

        predictions = _predictions_from_result(result)
        truth = read_ground_truth_polygons(label_dir / f"{path.stem}.txt")
        match, ious = match_masks(
            path.stem,
            predictions,
            truth,
            width=width,
            height=height,
            iou_threshold=iou_match,
        )
        run.matches.append(match)
        run.mask_ious.extend(ious)
        run.areas.append(
            AreaComparison(
                image_id=path.stem,
                predicted_pct=area_percentage(predictions, width, height),
                truth_pct=area_percentage(truth, width, height),
            )
        )
        for key, value in (result.speed or {}).items():
            speed_totals[key] = speed_totals.get(key, 0.0) + float(value)

    run.image_count = len(images)
    run.speed_ms = {k: round(v / len(images), 3) for k, v in speed_totals.items()}
    return run
