"""Metrik evaluasi untuk deteksi cacat.

Dua tingkat pengukuran dipakai karena keduanya menjawab pertanyaan berbeda:

Tingkat instance menjawab "berapa banyak cacat yang berhasil ditemukan", dan
di sinilah recall menjadi metrik utama karena cacat yang lolos jauh lebih mahal
daripada produk bagus yang salah ditolak. Karena itu dilaporkan F2, bukan F1.

Tingkat gambar menjawab "apakah produk ini benar ditandai bermasalah". Di
tingkat ini gambar tanpa cacat ikut dihitung, sehingga true negative terdefinisi
dan MCC dapat dipakai. MCC lebih jujur daripada akurasi pada data yang timpang.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from visionqc_ai.evaluation.matching import ImageMatch


@dataclass(frozen=True)
class InstanceMetrics:
    """Metrik pada tingkat instance cacat."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    f2: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageLevelMetrics:
    """Metrik pada tingkat gambar: bercacat atau tidak."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    specificity: float
    accuracy: float
    mcc: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def f_beta(precision: float, recall: float, beta: float) -> float:
    """F-score dengan bobot beta pada recall."""
    b2 = beta * beta
    denominator = b2 * precision + recall
    return _safe_divide((1 + b2) * precision * recall, denominator)


def instance_metrics(matches: list[ImageMatch]) -> InstanceMetrics:
    """Hitung metrik agregat pada tingkat instance."""
    tp = sum(m.true_positives for m in matches)
    fp = sum(m.false_positives for m in matches)
    fn = sum(m.false_negatives for m in matches)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return InstanceMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f_beta(precision, recall, 1.0),
        f2=f_beta(precision, recall, 2.0),
    )


def per_class_recall(matches: list[ImageMatch], class_count: int) -> dict[int, float]:
    """Recall untuk setiap kelas cacat."""
    found = [0] * class_count
    total = [0] * class_count
    for match in matches:
        for detected, class_id in zip(
            match.gt_detected, match.gt_class_ids, strict=True
        ):
            if 0 <= class_id < class_count:
                total[class_id] += 1
                found[class_id] += int(detected)
    return {i: _safe_divide(found[i], total[i]) for i in range(class_count)}


def per_class_support(matches: list[ImageMatch], class_count: int) -> dict[int, int]:
    """Jumlah instance ground truth per kelas."""
    total = [0] * class_count
    for match in matches:
        for class_id in match.gt_class_ids:
            if 0 <= class_id < class_count:
                total[class_id] += 1
    return dict(enumerate(total))


def image_level_metrics(matches: list[ImageMatch]) -> ImageLevelMetrics:
    """Hitung metrik keputusan pada tingkat gambar."""
    tp = fp = tn = fn = 0
    for match in matches:
        has_defect = len(match.gt_class_ids) > 0
        flagged = len(match.prediction_class_ids) > 0
        if has_defect and flagged:
            tp += 1
        elif has_defect and not flagged:
            fn += 1
        elif not has_defect and flagged:
            fp += 1
        else:
            tn += 1

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    accuracy = _safe_divide(tp + tn, tp + tn + fp + fn)

    numerator = tp * tn - fp * fn
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = numerator / denominator if denominator > 0 else 0.0

    return ImageLevelMetrics(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        specificity=specificity,
        accuracy=accuracy,
        mcc=mcc,
    )


def detection_flags(matches: list[ImageMatch]) -> list[int]:
    """Status per instance ground truth: 1 bila tertangkap, 0 bila lolos.

    Urutannya stabil terhadap urutan gambar dan urutan instance di dalamnya,
    sehingga vektor dari dua model dapat disandingkan pasangan demi pasangan.
    """
    flags: list[int] = []
    for match in sorted(matches, key=lambda m: m.image_id):
        flags.extend(int(x) for x in match.gt_detected)
    return flags
