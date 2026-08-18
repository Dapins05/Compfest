"""Kalibrasi skor keyakinan model.

Model deteksi mengeluarkan angka keyakinan, tetapi angka itu belum tentu
berarti peluang. Model modern lazimnya terlalu percaya diri: kumpulan prediksi
berkeyakinan 0,9 sering ternyata benar jauh di bawah 90 persen. Selama
keyakinan belum berarti peluang, ambang keputusan apa pun yang dibangun di
atasnya tidak dapat ditafsirkan.

Kalibrasi diukur dengan Expected Calibration Error, yaitu selisih rata-rata
antara keyakinan dan ketepatan sebenarnya pada sejumlah keranjang skor.
Dua metode perbaikan disediakan, dan keduanya tidak mengubah urutan skor
sehingga AUROC tetap sama persis:

* **Temperature scaling** memakai satu parameter yang melembutkan atau
  menajamkan skor. Karena tidak punya intersep, metode ini tidak dapat
  menggeser skor ke arah proporsi kelas yang sebenarnya.
* **Platt scaling** memakai kemiringan dan intersep. Intersep itulah yang
  memungkinkan penyesuaian terhadap proporsi kelas yang timpang.

Pemilihan di antara keduanya dilakukan pada set kalibrasi, tidak pernah pada
set uji.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import optimize

EPSILON = 1e-6


@dataclass(frozen=True)
class CalibrationReport:
    """Ukuran kualitas kalibrasi pada satu kumpulan prediksi."""

    ece: float
    mce: float
    brier: float
    n: int
    n_bins: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReliabilityBin:
    """Satu keranjang pada diagram keandalan."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    empirical_accuracy: float


def _clip(values: np.ndarray) -> np.ndarray:
    return np.clip(values, EPSILON, 1.0 - EPSILON)


def reliability_bins(
    probabilities: Sequence[float], labels: Sequence[int], *, n_bins: int = 10
) -> list[ReliabilityBin]:
    """Bagi prediksi ke keranjang skor dan bandingkan keyakinan dengan ketepatan."""
    probs = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    bins: list[ReliabilityBin] = []
    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        mask = (
            (probs > low) & (probs <= high)
            if index
            else (probs >= low) & (probs <= high)
        )
        count = int(mask.sum())
        bins.append(
            ReliabilityBin(
                lower=float(low),
                upper=float(high),
                count=count,
                mean_confidence=float(probs[mask].mean()) if count else 0.0,
                empirical_accuracy=float(truth[mask].mean()) if count else 0.0,
            )
        )
    return bins


def calibration_report(
    probabilities: Sequence[float], labels: Sequence[int], *, n_bins: int = 10
) -> CalibrationReport:
    """Hitung ECE, MCE, dan skor Brier.

    ECE adalah rata-rata selisih berbobot antara keyakinan dan ketepatan; MCE
    mengambil selisih terburuk. Skor Brier menilai ketepatan dan kalibrasi
    sekaligus sehingga berguna sebagai pemeriksa silang.
    """
    probs = _clip(np.asarray(probabilities, dtype=float))
    truth = np.asarray(labels, dtype=int)
    total = truth.size
    if total == 0:
        return CalibrationReport(0.0, 0.0, 0.0, 0, n_bins)

    gaps = [
        (b.count, abs(b.mean_confidence - b.empirical_accuracy))
        for b in reliability_bins(probs, truth, n_bins=n_bins)
        if b.count
    ]
    ece = sum(count * gap for count, gap in gaps) / total if gaps else 0.0
    mce = max((gap for _, gap in gaps), default=0.0)
    brier = float(np.mean((probs - truth) ** 2))
    return CalibrationReport(
        ece=float(ece), mce=float(mce), brier=brier, n=int(total), n_bins=n_bins
    )


def _to_logit(probabilities: np.ndarray) -> np.ndarray:
    probs = _clip(probabilities)
    return np.log(probs / (1.0 - probs))


def apply_temperature(
    probabilities: Sequence[float], temperature: float
) -> list[float]:
    """Terapkan temperature scaling pada peluang biner.

    Suhu di atas satu melembutkan skor sehingga model menjadi kurang yakin;
    di bawah satu menajamkannya. Urutan skor tidak pernah berubah.
    """
    if temperature <= 0:
        raise ValueError("suhu harus bernilai positif")
    logits = _to_logit(np.asarray(probabilities, dtype=float)) / temperature
    return [float(1.0 / (1.0 + math.exp(-z))) for z in logits]


def fit_temperature(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> float:
    """Cari suhu yang meminimalkan negative log likelihood pada set kalibrasi.

    Optimasi dilakukan pada set kalibrasi, bukan set uji. Menyetelnya pada set
    uji akan membuat angka kalibrasi yang dilaporkan tidak lagi sah.
    """
    probs = _clip(np.asarray(probabilities, dtype=float))
    truth = np.asarray(labels, dtype=int)
    logits = _to_logit(probs)

    def negative_log_likelihood(temperature: float) -> float:
        scaled = logits / max(temperature, EPSILON)
        calibrated = _clip(1.0 / (1.0 + np.exp(-scaled)))
        return float(
            -np.mean(truth * np.log(calibrated) + (1 - truth) * np.log(1 - calibrated))
        )

    result = optimize.minimize_scalar(
        negative_log_likelihood, bounds=bounds, method="bounded"
    )
    return float(result.x)


@dataclass(frozen=True)
class PlattParameters:
    """Kemiringan dan intersep hasil Platt scaling."""

    slope: float
    intercept: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_platt(
    probabilities: Sequence[float], parameters: PlattParameters
) -> list[float]:
    """Terapkan Platt scaling pada peluang biner."""
    logits = _to_logit(np.asarray(probabilities, dtype=float))
    scaled = parameters.slope * logits + parameters.intercept
    return [float(1.0 / (1.0 + math.exp(-z))) for z in np.clip(scaled, -30, 30)]


def fit_platt(probabilities: Sequence[float], labels: Sequence[int]) -> PlattParameters:
    """Cari kemiringan dan intersep yang meminimalkan negative log likelihood.

    Berbeda dari temperature scaling, intersep di sini memungkinkan model
    menyesuaikan diri terhadap proporsi kelas yang timpang. Pada dataset yang
    didominasi satu kelas, kemampuan menggeser itulah yang menentukan.
    """
    probs = _clip(np.asarray(probabilities, dtype=float))
    truth = np.asarray(labels, dtype=int)
    logits = _to_logit(probs)

    def negative_log_likelihood(params: np.ndarray) -> float:
        scaled = np.clip(params[0] * logits + params[1], -30, 30)
        calibrated = _clip(1.0 / (1.0 + np.exp(-scaled)))
        return float(
            -np.mean(truth * np.log(calibrated) + (1 - truth) * np.log(1 - calibrated))
        )

    result = optimize.minimize(
        negative_log_likelihood, x0=np.array([1.0, 0.0]), method="Nelder-Mead"
    )
    return PlattParameters(slope=float(result.x[0]), intercept=float(result.x[1]))
