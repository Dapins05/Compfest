"""Conformal prediction sebagai landasan kelas REVIEW.

Kebanyakan sistem menentukan kapan harus ragu lewat ambang yang dipilih
tangan. Conformal prediction membalik urutannya: tetapkan lebih dulu jaminan
yang diinginkan, misalnya label sebenarnya tercakup pada minimal 95 persen
kasus, lalu ambangnya mengikuti sebagai konsekuensi.

Cara kerjanya sederhana. Pada set kalibrasi dihitung skor ketidaksesuaian
untuk label yang benar, yaitu satu dikurangi peluang yang diberikan model
kepada label itu. Kuantil ke-(1-alpha) dari skor tersebut menjadi ambang.
Untuk gambar baru, sebuah label dimasukkan ke himpunan prediksi bila skor
ketidaksesuaiannya tidak melampaui ambang.

Yang membuatnya berguna untuk QC adalah **ukuran himpunannya**:

* satu label   - sistem yakin, keputusan diambil
* dua label    - kedua kemungkinan masih masuk akal, serahkan ke manusia
* nol label    - gambar tidak menyerupai apa pun di set kalibrasi

Jadi kelas REVIEW bukan lagi tebakan, melainkan akibat langsung dari jaminan
cakupan yang dipilih. Jaminan itu bersifat bebas distribusi dan hanya
mensyaratkan data kalibrasi dan data uji dapat dipertukarkan.

Ragam Mondrian menghitung kuantil terpisah untuk tiap kelas. Pada data yang
timpang, kuantil tunggal akan didominasi kelas mayoritas sehingga kelas langka
memperoleh jaminan yang jauh lebih lemah daripada yang dijanjikan.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

NORMAL = 0
DEFECT = 1


@dataclass(frozen=True)
class ConformalQuantiles:
    """Ambang ketidaksesuaian hasil kalibrasi."""

    alpha: float
    mode: str
    #: Kuantil per kelas; kunci 0 untuk normal dan 1 untuk cacat.
    quantiles: dict[int, float]
    calibration_counts: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "mode": self.mode,
            "quantiles": {str(k): v for k, v in self.quantiles.items()},
            "calibration_counts": {
                str(k): v for k, v in self.calibration_counts.items()
            },
        }


@dataclass(frozen=True)
class CoverageReport:
    """Hasil pemeriksaan jaminan cakupan pada data yang tidak dipakai kalibrasi."""

    alpha: float
    empirical_coverage: float
    target_coverage: float
    average_set_size: float
    singleton_rate: float
    review_rate: float
    empty_rate: float
    per_class_coverage: dict[int, float] = field(default_factory=dict)
    n: int = 0

    @property
    def guarantee_met(self) -> bool:
        return self.empirical_coverage >= self.target_coverage

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["per_class_coverage"] = {
            str(k): v for k, v in self.per_class_coverage.items()
        }
        payload["guarantee_met"] = self.guarantee_met
        return payload


def nonconformity(probability_defect: float, label: int) -> float:
    """Skor ketidaksesuaian: satu dikurangi peluang yang diberikan ke label itu."""
    probability = probability_defect if label == DEFECT else 1.0 - probability_defect
    return 1.0 - probability


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Kuantil conformal dengan koreksi sampel hingga.

    Tingkat kuantilnya adalah ceil((n+1)(1-alpha))/n, bukan sekadar (1-alpha).
    Koreksi ini yang membuat jaminan berlaku pada sampel berukuran terbatas,
    bukan hanya secara asimtotik. Bila n terlalu kecil sehingga tingkatnya
    melampaui satu, jaminan tidak dapat diberikan dan ambang dipatok pada nilai
    maksimum yang teramati.
    """
    values = np.sort(np.asarray(scores, dtype=float))
    n = values.size
    if n == 0:
        return 1.0
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return float(values[-1])
    return float(np.quantile(values, level, method="higher"))


def calibrate(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    alpha: float = 0.05,
    mode: str = "mondrian",
) -> ConformalQuantiles:
    """Hitung ambang ketidaksesuaian dari set kalibrasi."""
    if mode not in ("split", "mondrian"):
        raise ValueError(f"mode {mode!r} tidak dikenal; pilihan: split, mondrian")

    probs = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels, dtype=int)
    scores = np.array(
        [nonconformity(p, y) for p, y in zip(probs, truth, strict=True)], dtype=float
    )

    if mode == "split":
        shared = conformal_quantile(scores, alpha)
        quantiles = {NORMAL: shared, DEFECT: shared}
    else:
        quantiles = {
            label: conformal_quantile(scores[truth == label], alpha)
            for label in (NORMAL, DEFECT)
        }

    return ConformalQuantiles(
        alpha=alpha,
        mode=mode,
        quantiles=quantiles,
        calibration_counts={
            NORMAL: int((truth == NORMAL).sum()),
            DEFECT: int((truth == DEFECT).sum()),
        },
    )


def prediction_set(
    probability_defect: float, quantiles: ConformalQuantiles
) -> set[int]:
    """Susun himpunan prediksi untuk satu gambar."""
    return {
        label
        for label in (NORMAL, DEFECT)
        if nonconformity(probability_defect, label) <= quantiles.quantiles[label]
    }


def evaluate_coverage(
    probabilities: Sequence[float],
    labels: Sequence[int],
    quantiles: ConformalQuantiles,
) -> CoverageReport:
    """Uji apakah jaminan cakupan benar-benar tercapai pada data terpisah.

    Jaminan conformal bersifat teoretis; pemeriksaan empiris inilah yang
    menunjukkan apakah asumsi keterpertukaran memang berlaku pada data ini.
    """
    probs = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels, dtype=int)
    if truth.size == 0:
        return CoverageReport(
            quantiles.alpha, 0.0, 1 - quantiles.alpha, 0.0, 0.0, 0.0, 0.0
        )

    sets = [prediction_set(float(p), quantiles) for p in probs]
    covered = [int(y in s) for y, s in zip(truth, sets, strict=True)]
    sizes = [len(s) for s in sets]

    per_class = {
        label: float(
            np.mean([c for c, y in zip(covered, truth, strict=True) if y == label])
        )
        for label in (NORMAL, DEFECT)
        if int((truth == label).sum()) > 0
    }

    return CoverageReport(
        alpha=quantiles.alpha,
        empirical_coverage=float(np.mean(covered)),
        target_coverage=1.0 - quantiles.alpha,
        average_set_size=float(np.mean(sizes)),
        singleton_rate=float(np.mean([s == 1 for s in sizes])),
        review_rate=float(np.mean([s != 1 for s in sizes])),
        empty_rate=float(np.mean([s == 0 for s in sizes])),
        per_class_coverage=per_class,
        n=int(truth.size),
    )
