"""Pencocokan kotak prediksi dengan kotak ground truth.

Pencocokan dilakukan serakah menurut urutan confidence menurun: prediksi
berkeyakinan tertinggi memilih lebih dulu, dan satu kotak ground truth hanya
boleh dipakai sekali. Ini konvensi yang sama dengan perhitungan mAP, sehingga
hasilnya sebanding dengan angka yang dilaporkan Ultralytics.

Keluaran utamanya bukan sekadar jumlah, melainkan status per instance
ground truth: tertangkap atau tidak. Status per instance itulah yang membuat
uji McNemar berpasangan antara dua model menjadi mungkin.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Box:
    """Kotak dalam koordinat xyxy ternormalisasi terhadap ukuran gambar."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class ImageMatch:
    """Hasil pencocokan pada satu gambar."""

    image_id: str
    gt_detected: list[bool] = field(default_factory=list)
    gt_class_ids: list[int] = field(default_factory=list)
    prediction_is_correct: list[bool] = field(default_factory=list)
    prediction_class_ids: list[int] = field(default_factory=list)

    @property
    def true_positives(self) -> int:
        return sum(self.prediction_is_correct)

    @property
    def false_positives(self) -> int:
        return len(self.prediction_is_correct) - self.true_positives

    @property
    def false_negatives(self) -> int:
        return len(self.gt_detected) - sum(self.gt_detected)


def iou(a: Box, b: Box) -> float:
    """Intersection over union dua kotak."""
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def match_image(
    image_id: str,
    predictions: list[Box],
    ground_truth: list[Box],
    *,
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> ImageMatch:
    """Cocokkan prediksi dengan ground truth pada satu gambar.

    Bila ``class_agnostic`` bernilai benar, kelas diabaikan dan yang dinilai
    hanya apakah lokasi cacat ditemukan. Mode itu berguna untuk memisahkan
    kesalahan lokalisasi dari kesalahan penamaan kelas.
    """
    order = sorted(range(len(predictions)), key=lambda i: -predictions[i].confidence)
    used: set[int] = set()
    correct = [False] * len(predictions)
    detected = [False] * len(ground_truth)

    for pred_index in order:
        prediction = predictions[pred_index]
        best_index = -1
        best_iou = iou_threshold
        for gt_index, truth in enumerate(ground_truth):
            if gt_index in used:
                continue
            if not class_agnostic and truth.class_id != prediction.class_id:
                continue
            score = iou(prediction, truth)
            if score >= best_iou:
                best_iou = score
                best_index = gt_index
        if best_index >= 0:
            used.add(best_index)
            correct[pred_index] = True
            detected[best_index] = True

    return ImageMatch(
        image_id=image_id,
        gt_detected=detected,
        gt_class_ids=[b.class_id for b in ground_truth],
        prediction_is_correct=correct,
        prediction_class_ids=[b.class_id for b in predictions],
    )
