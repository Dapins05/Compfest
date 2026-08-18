"""Validasi statistik dataset hasil preprocessing.

Memeriksa keselarasan distribusi kelas antar split, tingkat ketimpangan kelas,
dan kecukupan ukuran test set untuk mengestimasi recall. Modul ini hanya
mengukur dan melaporkan; tidak ada nilai yang disetel otomatis dari hasilnya.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from visionqc_ai.data.records import SampleRecord
from visionqc_ai.data.split import SplitAssignment


@dataclass
class ChiSquareResult:
    """Hasil uji khi-kuadrat keselarasan distribusi kelas antar split."""

    statistic: float | None
    p_value: float | None
    degrees_of_freedom: int | None
    passed: bool | None
    note: str = ""


@dataclass
class BalanceResult:
    """Ukuran ketimpangan kelas pada keseluruhan dataset."""

    counts: dict[str, int]
    imbalance_ratio: float | None
    normalized_entropy: float | None
    ir_passed: bool | None
    entropy_passed: bool | None


@dataclass
class SampleSizeResult:
    """Kecukupan ukuran test set untuk mengestimasi recall."""

    required_instances: int
    actual_instances: int
    actual_defect_images: int
    achieved_margin_of_error: float | None
    passed: bool
    formula: str
    parameters: dict[str, float]


@dataclass
class DatasetValidation:
    """Gabungan seluruh pemeriksaan, siap diserialisasi ke JSON."""

    chi_square: ChiSquareResult
    balance: BalanceResult
    sample_size: SampleSizeResult
    per_split_images: dict[str, int] = field(default_factory=dict)
    per_split_instances: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        checks = [
            self.chi_square.passed,
            self.balance.ir_passed,
            self.balance.entropy_passed,
            self.sample_size.passed,
        ]
        return all(check is True for check in checks)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _instance_counts(records: list[SampleRecord]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(record.class_names)
    return counter


def chi_square_split_homogeneity(
    assignment: SplitAssignment, classes: tuple[str, ...], *, alpha: float
) -> ChiSquareResult:
    """Uji apakah distribusi kelas konsisten di seluruh split.

    Hipotesis nol: proporsi tiap kelas sama pada train, val, dan test. Nilai p
    di atas alpha adalah hasil yang diinginkan.
    """
    table = np.array(
        [
            [_instance_counts(records)[name] for name in classes]
            for records in assignment.as_dict().values()
        ],
        dtype=float,
    )
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2 or table.shape[0] < 2:
        return ChiSquareResult(None, None, None, None, "kelas efektif kurang dari dua")

    expected_min = (
        table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True)
    ).min() / table.sum()
    statistic, p_value, dof, _ = scipy_stats.chi2_contingency(table)
    note = (
        "frekuensi harapan minimum < 5; nilai p hanya indikatif"
        if expected_min < 5
        else ""
    )
    return ChiSquareResult(
        statistic=float(statistic),
        p_value=float(p_value),
        degrees_of_freedom=int(dof),
        passed=bool(p_value > alpha),
        note=note,
    )


def class_balance(
    records: list[SampleRecord],
    classes: tuple[str, ...],
    *,
    max_imbalance_ratio: float,
    min_normalized_entropy: float,
) -> BalanceResult:
    """Hitung rasio ketimpangan dan entropi Shannon ternormalisasi."""
    counter = _instance_counts(records)
    counts = {name: counter[name] for name in classes}
    present = [value for value in counts.values() if value > 0]
    if not present:
        return BalanceResult(counts, None, None, None, None)

    imbalance = max(present) / min(present)
    total = sum(present)
    entropy = -sum((c / total) * math.log(c / total) for c in present)
    max_entropy = math.log(len(classes))
    normalized = entropy / max_entropy if max_entropy > 0 else 1.0

    return BalanceResult(
        counts=counts,
        imbalance_ratio=float(imbalance),
        normalized_entropy=float(normalized),
        ir_passed=bool(imbalance <= max_imbalance_ratio),
        entropy_passed=bool(normalized >= min_normalized_entropy),
    )


def minimum_test_size(
    test_records: list[SampleRecord],
    *,
    z: float,
    expected_recall: float,
    margin_of_error: float,
) -> SampleSizeResult:
    """Ukuran sampel minimum test set dengan n = z^2 * p(1-p) / E^2.

    Hitungan dilakukan pada tingkat instance cacat, bukan gambar, karena recall
    deteksi dihitung per objek ground-truth. Selain lulus atau tidak, galat
    yang benar-benar tercapai ikut dilaporkan.
    """
    required = math.ceil(
        (z**2) * expected_recall * (1 - expected_recall) / (margin_of_error**2)
    )
    defect_images = sum(1 for record in test_records if not record.is_normal)
    instances = sum(len(record.instances) for record in test_records)
    achieved = (
        z * math.sqrt(expected_recall * (1 - expected_recall) / instances)
        if instances > 0
        else None
    )
    return SampleSizeResult(
        required_instances=int(required),
        actual_instances=int(instances),
        actual_defect_images=int(defect_images),
        achieved_margin_of_error=float(achieved) if achieved is not None else None,
        passed=bool(instances >= required),
        formula="n = z^2 * p(1-p) / E^2",
        parameters={"z": z, "p": expected_recall, "E": margin_of_error},
    )


def validate_dataset(
    assignment: SplitAssignment,
    classes: tuple[str, ...],
    *,
    chi2_alpha: float,
    max_imbalance_ratio: float,
    min_normalized_entropy: float,
    z: float,
    expected_recall: float,
    margin_of_error: float,
) -> DatasetValidation:
    """Jalankan seluruh pemeriksaan dan kumpulkan hasilnya."""
    all_records = assignment.train + assignment.val + assignment.test
    per_split_instances = {
        name: dict(_instance_counts(records))
        for name, records in assignment.as_dict().items()
    }
    return DatasetValidation(
        chi_square=chi_square_split_homogeneity(assignment, classes, alpha=chi2_alpha),
        balance=class_balance(
            all_records,
            classes,
            max_imbalance_ratio=max_imbalance_ratio,
            min_normalized_entropy=min_normalized_entropy,
        ),
        sample_size=minimum_test_size(
            assignment.test,
            z=z,
            expected_recall=expected_recall,
            margin_of_error=margin_of_error,
        ),
        per_split_images={
            name: len(records) for name, records in assignment.as_dict().items()
        },
        per_split_instances=per_split_instances,
    )
