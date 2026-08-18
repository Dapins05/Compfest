"""Ambang keputusan berdasarkan biaya kesalahan, bukan kebiasaan.

Ambang yang lazim dipakai, entah 0,5 atau laju alarm palsu 1 persen, diam-diam
mengandaikan kedua jenis kesalahan sama mahalnya. Pada inspeksi produk pangan
andaian itu keliru jauh: cacat yang lolos ke konsumen berbiaya jauh lebih besar
daripada produk bagus yang salah ditolak.

Bila biayanya diketahui, ambang optimalnya tidak perlu dicari-cari. Keputusan
menolak lebih murah daripada meloloskan ketika

    p * C_lolos > (1 - p) * C_salah_tolak

sehingga ambang Bayes-nya adalah

    tau* = C_salah_tolak / (C_salah_tolak + C_lolos)

Selain ambang teoretis itu, modul ini juga mencari ambang yang benar-benar
meminimalkan biaya pada data kalibrasi. Keduanya jarang sama persis, dan
selisihnya justru memberi tahu seberapa jauh model menyimpang dari kalibrasi
sempurna.

Satu koreksi penting: test set dibangun dari gambar cacat sehingga proporsi
cacatnya jauh lebih tinggi daripada lini produksi sungguhan. Tanpa koreksi,
biaya yang dihitung akan menyimpulkan bahwa menolak semua produk adalah
pilihan termurah, dan kesimpulan itu hanya berlaku pada data yang memang
didominasi cacat. Parameter ``prevalence`` menimbang ulang kedua kelas agar
biaya yang dilaporkan mencerminkan proporsi cacat yang diharapkan di produksi.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CostModel:
    """Biaya tiap jenis kesalahan, dalam rupiah per unit produk."""

    false_reject: float
    missed_defect: float
    review: float = 0.0

    @property
    def bayes_threshold(self) -> float:
        """Ambang optimal bila skor model sudah terkalibrasi sebagai peluang."""
        total = self.false_reject + self.missed_defect
        return self.false_reject / total if total > 0 else 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"bayes_threshold": self.bayes_threshold}


@dataclass(frozen=True)
class CostPoint:
    """Rincian biaya pada satu ambang."""

    threshold: float
    false_rejects: int
    missed_defects: int
    total_cost: float
    cost_per_item: float
    recall: float
    specificity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_threshold(
    probabilities: Sequence[float],
    labels: Sequence[int],
    threshold: float,
    costs: CostModel,
    *,
    prevalence: float | None = None,
) -> CostPoint:
    """Hitung biaya yang timbul bila ambang tertentu dipakai.

    Bila ``prevalence`` diberikan, kedua kelas ditimbang ulang agar proporsi
    cacatnya sesuai nilai itu. Recall dan specificity tidak ikut ditimbang
    karena keduanya sudah bersyarat pada kelas masing-masing.
    """
    probs = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels, dtype=int)
    flagged = probs >= threshold

    false_rejects = int(np.sum(flagged & (truth == 0)))
    missed = int(np.sum(~flagged & (truth == 1)))
    positives = int(np.sum(truth == 1))
    negatives = int(np.sum(truth == 0))

    weight_positive, weight_negative = 1.0, 1.0
    if prevalence is not None and positives and negatives:
        observed = positives / truth.size
        weight_positive = prevalence / observed
        weight_negative = (1.0 - prevalence) / (1.0 - observed)

    total = (
        false_rejects * weight_negative * costs.false_reject
        + missed * weight_positive * costs.missed_defect
    )
    effective_n = positives * weight_positive + negatives * weight_negative
    return CostPoint(
        threshold=float(threshold),
        false_rejects=false_rejects,
        missed_defects=missed,
        total_cost=float(total),
        cost_per_item=float(total / effective_n) if effective_n else 0.0,
        recall=float((positives - missed) / positives) if positives else 0.0,
        specificity=float((negatives - false_rejects) / negatives)
        if negatives
        else 0.0,
    )


def cost_curve(
    probabilities: Sequence[float],
    labels: Sequence[int],
    costs: CostModel,
    *,
    n_points: int = 101,
    prevalence: float | None = None,
) -> list[CostPoint]:
    """Susun kurva biaya pada rentang ambang nol sampai satu."""
    grid = np.linspace(0.0, 1.0, n_points)
    return [
        evaluate_threshold(
            probabilities, labels, float(t), costs, prevalence=prevalence
        )
        for t in grid
    ]


def optimal_threshold(
    probabilities: Sequence[float],
    labels: Sequence[int],
    costs: CostModel,
    *,
    n_points: int = 101,
    prevalence: float | None = None,
) -> CostPoint:
    """Cari ambang berbiaya terendah pada data kalibrasi.

    Bila beberapa ambang menghasilkan biaya sama, dipilih yang terendah karena
    lebih memihak penangkapan cacat, dan itulah sisi yang lebih murah salahnya
    dalam konteks pangan.
    """
    curve = cost_curve(
        probabilities, labels, costs, n_points=n_points, prevalence=prevalence
    )
    return min(curve, key=lambda point: (point.total_cost, point.threshold))


def savings_versus(
    probabilities: Sequence[float],
    labels: Sequence[int],
    costs: CostModel,
    *,
    baseline_threshold: float,
    chosen_threshold: float,
    prevalence: float | None = None,
) -> dict[str, Any]:
    """Bandingkan dua ambang pada data yang sama.

    Keduanya dievaluasi ulang pada data yang diberikan. Ambang pilihan boleh
    saja berasal dari set kalibrasi, tetapi biayanya harus dihitung pada data
    yang sedang dinilai; mengembalikan biaya set kalibrasi apa adanya akan
    melaporkan angka kalibrasi seolah-olah angka uji.
    """
    baseline = evaluate_threshold(
        probabilities, labels, baseline_threshold, costs, prevalence=prevalence
    )
    optimal = evaluate_threshold(
        probabilities, labels, chosen_threshold, costs, prevalence=prevalence
    )
    reduction = baseline.total_cost - optimal.total_cost
    ratio = reduction / baseline.total_cost if baseline.total_cost > 0 else 0.0
    return {
        "baseline": baseline.to_dict(),
        "optimal": optimal.to_dict(),
        "cost_reduction": float(reduction),
        "cost_reduction_ratio": float(ratio),
    }
