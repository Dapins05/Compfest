"""Selang kepercayaan bootstrap dengan koreksi bias dan akselerasi.

Metrik tunggal tanpa selang kepercayaan menyembunyikan ketidakpastian. Pada
test set berukuran 75 instance, perbedaan recall beberapa persen bisa saja
hanya derau, dan selang kepercayaanlah yang menunjukkan hal itu.

Metode yang dipakai adalah BCa: kuantil bootstrap digeser oleh koreksi bias
z0 dan akselerasi a yang diperoleh dari jackknife. Persentil polos cenderung
terlalu sempit ketika distribusi statistiknya menceng, dan recall pada data
timpang memang menceng.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class ConfidenceInterval:
    """Estimasi titik beserta selang kepercayaannya."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    method: str
    n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_id(self, digits: int = 3) -> str:
        """Tulis sebagai teks Indonesia, mis. 0,842 [0,761; 0,908]."""

        def fmt(value: float) -> str:
            return f"{value:.{digits}f}".replace(".", ",")

        return f"{fmt(self.estimate)} [{fmt(self.lower)}; {fmt(self.upper)}]"


def bca_interval(
    sample: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> ConfidenceInterval:
    """Selang kepercayaan BCa untuk sebuah statistik.

    Bila seluruh nilai sampel identik, distribusi bootstrap menjadi degenerate
    sehingga koreksi bias tidak terdefinisi. Kasus itu ditangani dengan
    mengembalikan selang selebar nol, bukan melempar galat.
    """
    data = np.asarray(sample, dtype=float)
    n = data.size
    if n == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, confidence, "bca", 0)

    observed = float(statistic(data))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    replicates = np.array([statistic(data[idx]) for idx in indices], dtype=float)

    if np.allclose(replicates, observed):
        return ConfidenceInterval(observed, observed, observed, confidence, "bca", n)

    proportion = float(np.mean(replicates < observed))
    proportion = min(max(proportion, 1.0 / n_resamples), 1.0 - 1.0 / n_resamples)
    z0 = scipy_stats.norm.ppf(proportion)

    jackknife = np.array([statistic(np.delete(data, i)) for i in range(n)], dtype=float)
    deviation = jackknife.mean() - jackknife
    denominator = 6.0 * (np.sum(deviation**2) ** 1.5)
    acceleration = float(np.sum(deviation**3) / denominator) if denominator > 0 else 0.0

    alpha = 1.0 - confidence
    quantiles = []
    for z_alpha in (
        scipy_stats.norm.ppf(alpha / 2),
        scipy_stats.norm.ppf(1 - alpha / 2),
    ):
        adjusted = z0 + (z0 + z_alpha) / (1 - acceleration * (z0 + z_alpha))
        quantiles.append(float(scipy_stats.norm.cdf(adjusted)))

    lower = float(np.percentile(replicates, 100 * quantiles[0]))
    upper = float(np.percentile(replicates, 100 * quantiles[1]))
    return ConfidenceInterval(observed, lower, upper, confidence, "bca", n)


def proportion_interval(
    successes: int, total: int, *, confidence: float = 0.95
) -> ConfidenceInterval:
    """Selang Wilson untuk sebuah proporsi.

    Dipakai untuk recall dan precision karena selang Wald menjadi tidak sahih
    saat proporsinya mendekati nol atau satu, dan kelas langka pada dataset ini
    memang berada di dekat kedua ujung itu.
    """
    if total == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, confidence, "wilson", 0)

    z = float(scipy_stats.norm.ppf(1 - (1 - confidence) / 2))
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denominator
    return ConfidenceInterval(
        estimate=p,
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
        confidence=confidence,
        method="wilson",
        n=total,
    )
