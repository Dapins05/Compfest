"""Penentuan ambang anomali memakai teori nilai ekstrem.

Pertanyaan pertama yang akan diajukan penilai terhadap ambang anomali adalah
"kenapa angkanya sekian". Menjawabnya dengan "karena terlihat bagus" tidak
dapat dipertahankan. Modul ini menurunkan ambang itu dari data.

Gagasannya: yang menentukan alarm palsu bukan keseluruhan sebaran skor anomali
pada produk normal, melainkan **ekornya** saja. Teorema Pickands-Balkema-de Haan
menyatakan bahwa nilai yang melampaui suatu ambang awal yang cukup tinggi akan
mendekati sebaran Pareto Tergeneralisasi, berapa pun sebaran aslinya. Karena
itu ekor tersebut dimodelkan dengan GPD, lalu ambang akhir dihitung sebagai
kuantil yang menjamin laju alarm palsu yang diinginkan.

Prosedurnya dikenal sebagai Peaks Over Threshold:

1. Pilih ambang awal u, biasanya kuantil ke-90 dari skor sampel normal.
2. Cocokkan GPD pada kelebihan nilai di atas u, yaitu (x - u) untuk x > u.
3. Hitung ambang akhir z_q untuk laju alarm palsu q.

Rumus kuantilnya, untuk parameter bentuk xi tidak nol:

    z_q = u + (sigma / xi) * ((q * n / N_u) ** (-xi) - 1)

dengan n banyaknya sampel normal dan N_u banyaknya yang melampaui u.
Untuk xi mendekati nol dipakai bentuk limit eksponensialnya.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class GPDFit:
    """Hasil pencocokan sebaran Pareto Tergeneralisasi pada ekor."""

    initial_threshold: float
    #: Kuantil yang benar-benar dipakai; bisa lebih rendah dari yang diminta
    #: bila ekornya terlalu sedikit terisi.
    used_quantile: float
    shape_xi: float
    scale_sigma: float
    n_total: int
    n_exceedances: int
    ks_statistic: float
    ks_p_value: float

    @property
    def tail_type(self) -> str:
        """Tafsiran parameter bentuk terhadap perilaku ekor."""
        if self.shape_xi > 0.05:
            return "ekor berat, nilai ekstrem masih mungkin jauh lebih besar"
        if self.shape_xi < -0.05:
            return "ekor terbatas, ada batas atas yang tak terlampaui"
        return "ekor eksponensial, meluruh dengan laju tetap"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"tail_type": self.tail_type}


@dataclass(frozen=True)
class EvtThreshold:
    """Ambang anomali beserta seluruh besaran yang membentuknya."""

    threshold: float
    target_false_alarm_rate: float
    empirical_false_alarm_rate: float
    fit: GPDFit

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "target_false_alarm_rate": self.target_false_alarm_rate,
            "empirical_false_alarm_rate": self.empirical_false_alarm_rate,
            "gpd_fit": self.fit.to_dict(),
        }

    def describe_id(self) -> str:
        """Kalimat siap kutip untuk laporan dan proposal."""
        return (
            f"Ambang {self.threshold:.4f} diperoleh dengan memodelkan ekor sebaran "
            f"skor anomali sampel normal memakai GPD "
            f"(xi = {self.fit.shape_xi:.4f}; sigma = {self.fit.scale_sigma:.4f}; "
            f"u = {self.fit.initial_threshold:.4f}; "
            f"N_u/n = {self.fit.n_exceedances}/{self.fit.n_total}), "
            f"menargetkan laju alarm palsu {self.target_false_alarm_rate:.1%} "
            f"dan mencapai {self.empirical_false_alarm_rate:.1%} pada data yang sama."
        ).replace(".", ",")


def fit_gpd_tail(
    normal_scores: Sequence[float],
    *,
    initial_quantile: float = 0.90,
    min_exceedances: int = 10,
) -> GPDFit:
    """Cocokkan GPD pada ekor atas skor sampel normal.

    Ambang awal u diturunkan bertahap bila kuantil yang diminta menyisakan
    terlalu sedikit nilai ekor. Ini kompromi yang tak terhindarkan pada sampel
    kecil: u yang tinggi membuat pendekatan GPD lebih sahih tetapi datanya
    tidak cukup, sedangkan u yang rendah memberi data lebih banyak dengan
    imbalan pendekatannya melemah. Kuantil yang benar-benar dipakai dicatat
    agar kompromi itu terlihat, bukan tersembunyi.

    Uji Kolmogorov-Smirnov ikut dilaporkan sebagai diagnostik kecocokan. Nilai p
    yang besar berarti tidak ada bukti bahwa GPD keliru dipakai, dan itulah yang
    diharapkan.
    """
    scores = np.asarray(normal_scores, dtype=float)
    if scores.size < 20:
        raise ValueError(
            f"butuh minimal 20 skor normal untuk memodelkan ekor, ada {scores.size}"
        )

    candidates = [initial_quantile]
    candidates += [q for q in (0.85, 0.80, 0.75, 0.70) if q < initial_quantile]

    u = None
    for quantile in candidates:
        candidate = float(np.quantile(scores, quantile))
        if int((scores > candidate).sum()) >= min_exceedances:
            u = candidate
            break

    if u is None:
        # Tangga kuantil bisa gagal ketika banyak skor bernilai kembar, karena
        # kuantil yang berbeda menghasilkan nilai yang sama. Jalan yang selalu
        # bekerja adalah memilih langsung dari statistik urutan: turuni nilai
        # unik dari yang terbesar sampai jumlah kelebihan mencukupi.
        for candidate in np.unique(scores)[::-1]:
            if int((scores > candidate).sum()) >= min_exceedances:
                u = float(candidate)
                break

    if u is None:
        raise ValueError(
            f"sampel terlalu kecil: {scores.size} skor tidak menyisakan "
            f"{min_exceedances} nilai ekor yang berbeda"
        )
    chosen_quantile = float(np.mean(scores <= u))

    exceedances = scores[scores > u] - u
    shape, _, scale = scipy_stats.genpareto.fit(exceedances, floc=0.0)
    ks = scipy_stats.kstest(exceedances, "genpareto", args=(shape, 0.0, scale))

    return GPDFit(
        initial_threshold=u,
        used_quantile=float(chosen_quantile),
        shape_xi=float(shape),
        scale_sigma=float(scale),
        n_total=int(scores.size),
        n_exceedances=int(exceedances.size),
        ks_statistic=float(ks.statistic),
        ks_p_value=float(ks.pvalue),
    )


def threshold_from_fit(fit: GPDFit, *, target_false_alarm_rate: float) -> float:
    """Hitung ambang z_q dari hasil pencocokan GPD."""
    if not 0.0 < target_false_alarm_rate < 1.0:
        raise ValueError("laju alarm palsu harus berada di antara 0 dan 1")

    ratio = target_false_alarm_rate * fit.n_total / fit.n_exceedances
    if ratio >= 1.0:
        return fit.initial_threshold

    xi, sigma, u = fit.shape_xi, fit.scale_sigma, fit.initial_threshold
    if abs(xi) < 1e-8:
        return u + sigma * (-math.log(ratio))
    return u + (sigma / xi) * (ratio ** (-xi) - 1.0)


def compute_evt_threshold(
    normal_scores: Sequence[float],
    *,
    target_false_alarm_rate: float = 0.01,
    initial_quantile: float = 0.90,
    min_exceedances: int = 10,
) -> EvtThreshold:
    """Turunkan ambang anomali dari skor sampel normal.

    Laju alarm palsu empiris ikut dilaporkan agar terlihat seberapa dekat hasil
    nyata dengan target. Keduanya jarang persis sama, dan menyembunyikan selisih
    itu akan menyesatkan.
    """
    fit = fit_gpd_tail(
        normal_scores,
        initial_quantile=initial_quantile,
        min_exceedances=min_exceedances,
    )
    threshold = threshold_from_fit(fit, target_false_alarm_rate=target_false_alarm_rate)
    scores = np.asarray(normal_scores, dtype=float)
    empirical = float(np.mean(scores > threshold))
    return EvtThreshold(
        threshold=float(threshold),
        target_false_alarm_rate=target_false_alarm_rate,
        empirical_false_alarm_rate=empirical,
        fit=fit,
    )


def quantile_threshold(
    normal_scores: Sequence[float], *, target_false_alarm_rate: float = 0.01
) -> float:
    """Ambang kuantil empiris polos, dipakai sebagai pembanding.

    Ambang ini hanya membaca data yang ada dan tidak dapat menyimpulkan apa pun
    di luar nilai terbesar yang pernah teramati. Perbandingan dengannya
    memperlihatkan apa yang sebenarnya ditambahkan oleh pendekatan EVT.
    """
    scores = np.asarray(normal_scores, dtype=float)
    return float(np.quantile(scores, 1.0 - target_false_alarm_rate))
