"""Uji signifikansi untuk membandingkan dua model pada data yang sama.

Dua model dievaluasi pada test set yang identik, sehingga hasilnya berpasangan
dan bukan dua sampel bebas. Uji McNemar memanfaatkan itu: yang diperiksa hanya
kasus diskordan, yaitu instance yang tertangkap satu model tetapi lolos dari
model lainnya. Instance yang sama-sama tertangkap atau sama-sama lolos tidak
membawa informasi tentang perbedaan keduanya.

Nilai p menjawab apakah perbedaannya nyata. Cohen's h menjawab seberapa besar
perbedaannya. Keduanya perlu dilaporkan, karena perbedaan yang signifikan
belum tentu berarti besar.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from scipy import stats as scipy_stats


@dataclass(frozen=True)
class McNemarResult:
    """Hasil uji McNemar berpasangan."""

    both_correct: int
    only_first_correct: int
    only_second_correct: int
    both_wrong: int
    statistic: float | None
    p_value: float
    method: str
    n_pairs: int
    significant: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def discordant(self) -> int:
        return self.only_first_correct + self.only_second_correct


def mcnemar(
    first: Sequence[int],
    second: Sequence[int],
    *,
    alpha: float = 0.05,
    exact_below: int = 25,
) -> McNemarResult:
    """Uji McNemar untuk dua vektor hasil berpasangan berisi nol dan satu.

    Bila jumlah kasus diskordan sedikit, pendekatan khi-kuadrat tidak lagi
    tepat sehingga dipakai uji binomial eksak. Ambang peralihannya adalah
    ``exact_below``.
    """
    if len(first) != len(second):
        raise ValueError(
            f"panjang kedua vektor harus sama: {len(first)} dan {len(second)}"
        )

    both_correct = only_first = only_second = both_wrong = 0
    for a, b in zip(first, second, strict=True):
        if a and b:
            both_correct += 1
        elif a and not b:
            only_first += 1
        elif b and not a:
            only_second += 1
        else:
            both_wrong += 1

    discordant = only_first + only_second
    if discordant == 0:
        return McNemarResult(
            both_correct=both_correct,
            only_first_correct=only_first,
            only_second_correct=only_second,
            both_wrong=both_wrong,
            statistic=None,
            p_value=1.0,
            method="tidak ada kasus diskordan",
            n_pairs=len(first),
            significant=False,
        )

    if discordant < exact_below:
        p_value = float(
            scipy_stats.binomtest(min(only_first, only_second), discordant, 0.5).pvalue
        )
        statistic = None
        method = "binomial eksak"
    else:
        statistic = float((abs(only_first - only_second) - 1) ** 2 / discordant)
        p_value = float(scipy_stats.chi2.sf(statistic, df=1))
        method = "khi-kuadrat dengan koreksi kontinuitas"

    return McNemarResult(
        both_correct=both_correct,
        only_first_correct=only_first,
        only_second_correct=only_second,
        both_wrong=both_wrong,
        statistic=statistic,
        p_value=p_value,
        method=method,
        n_pairs=len(first),
        significant=bool(p_value < alpha),
    )


def cohens_h(p1: float, p2: float) -> float:
    """Besar efek untuk selisih dua proporsi.

    Panduan tafsiran yang lazim: 0,2 kecil, 0,5 sedang, 0,8 besar.
    """
    phi1 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
    phi2 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    return phi1 - phi2


def interpret_cohens_h(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.2:
        return "dapat diabaikan"
    if magnitude < 0.5:
        return "kecil"
    if magnitude < 0.8:
        return "sedang"
    return "besar"
