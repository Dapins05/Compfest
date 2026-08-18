"""Pembagian dataset train, val, dan test yang terstratifikasi.

Seed dan rasio dibaca dari configs/dataset.yaml dan tidak diubah setelah
training berjalan. Stratifikasi per kategori dan kelas dominan mencegah kelas
langka jatuh seluruhnya ke satu split.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from visionqc_ai.data.records import SampleRecord

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


@dataclass(frozen=True)
class SplitRatios:
    """Proporsi tiap split. Harus berjumlah 1,0."""

    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"rasio split harus berjumlah 1,0 (sekarang {total})")

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.train, self.val, self.test)


@dataclass(frozen=True)
class SplitAssignment:
    """Hasil pembagian, satu daftar record per split."""

    train: list[SampleRecord]
    val: list[SampleRecord]
    test: list[SampleRecord]

    def as_dict(self) -> dict[str, list[SampleRecord]]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


def _allocate(count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Bagi count item ke tiga split memakai metode sisa terbesar.

    Bila jumlahnya minimal tiga, val dan test dijamin kebagian satu. Stratum
    yang seluruhnya masuk train membuat kelas itu tak pernah terevaluasi.
    """
    if count == 0:
        return (0, 0, 0)

    exact = [count * r for r in ratios]
    base = [int(value) for value in exact]
    remainder = count - sum(base)
    order = sorted(range(3), key=lambda i: exact[i] - base[i], reverse=True)
    for i in range(remainder):
        base[order[i % 3]] += 1

    if count >= 3:
        for index in (1, 2):
            if base[index] == 0:
                donor = max(range(3), key=lambda i: base[i])
                if base[donor] > 1:
                    base[donor] -= 1
                    base[index] += 1
    return (base[0], base[1], base[2])


def stratified_split(
    records: list[SampleRecord], *, seed: int, ratios: SplitRatios
) -> SplitAssignment:
    """Bagi record menjadi train, val, dan test secara terstratifikasi.

    Strata diproses berurutan dan tiap stratum diacak dengan RNG ber-seed
    sendiri, sehingga menambah kategori baru tidak mengubah pembagian
    kategori yang sudah ada.
    """
    strata: defaultdict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        strata[f"{record.category}|{record.dominant_class()}"].append(record)

    buckets: dict[str, list[SampleRecord]] = {name: [] for name in SPLIT_NAMES}
    for key in sorted(strata):
        items = sorted(strata[key], key=lambda r: r.sample_id)
        random.Random(f"{seed}:{key}").shuffle(items)

        n_train, n_val, _ = _allocate(len(items), ratios.as_tuple())
        buckets["train"].extend(items[:n_train])
        buckets["val"].extend(items[n_train : n_train + n_val])
        buckets["test"].extend(items[n_train + n_val :])

    return SplitAssignment(
        train=buckets["train"], val=buckets["val"], test=buckets["test"]
    )
