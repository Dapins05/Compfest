"""Rekonstruksi kode jenis cacat pada mask VisA.

Nilai piksel mask VisA adalah kode jenis cacat, bukan nilai biner. Pemetaan
kode ke jenis tidak ikut didistribusikan bersama dataset dan urutannya tidak
alfabetis, jadi pemetaan itu dipulihkan dari gambar berlabel tunggal,
dilengkapi lewat eliminasi, lalu diuji ulang ke seluruh gambar multi-label.

Tanpa langkah ini VisA hanya dapat dipakai sebagai data biner normal atau
anomali. Dengan langkah ini VisA memberi label jenis cacat per instance.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

LabelMapper = Callable[[str], str | None]

_NORMAL = "normal"


@dataclass(frozen=True)
class VisaAnnotation:
    """Satu baris image_anno.csv yang sudah dirapikan."""

    image_path: Path
    mask_path: Path | None
    raw_labels: tuple[str, ...]

    @property
    def is_normal(self) -> bool:
        return self.raw_labels == (_NORMAL,)


@dataclass(frozen=True)
class VisaCodeMap:
    """Hasil rekonstruksi kode mask untuk satu kategori VisA."""

    category: str
    code_to_label: dict[int, str]
    code_to_class: dict[int, str | None]
    ambiguous: dict[int, tuple[str, ...]] = field(default_factory=dict)
    learned_codes: int = 0
    inferred_codes: int = 0
    validated_images: int = 0

    def summary(self) -> str:
        """Ringkasan satu baris untuk log dan laporan."""
        return (
            f"{self.category}: {len(self.code_to_class)} kode "
            f"(belajar {self.learned_codes}, eliminasi {self.inferred_codes}, "
            f"lebur-kelas {len(self.ambiguous)}), "
            f"validasi multi-label {self.validated_images} gambar"
        )


def _split_labels(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def read_annotations(category_dir: Path, visa_root: Path) -> list[VisaAnnotation]:
    """Baca image_anno.csv sebuah kategori VisA.

    Jalur di dalam CSV relatif terhadap akar VisA, bukan folder kategori.
    """
    csv_path = category_dir / "image_anno.csv"
    rows: list[VisaAnnotation] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mask = row.get("mask") or ""
            rows.append(
                VisaAnnotation(
                    image_path=visa_root / row["image"],
                    mask_path=(visa_root / mask) if mask else None,
                    raw_labels=_split_labels(row["label"]),
                )
            )
    return rows


def mask_codes(mask_path: Path) -> list[int]:
    """Daftar kode cacat yang muncul pada sebuah mask; nol berarti latar."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"mask tidak terbaca: {mask_path}")
    return sorted(int(v) for v in np.unique(mask) if v > 0)


def build_code_map(
    category: str,
    annotations: list[VisaAnnotation],
    *,
    to_class: LabelMapper,
) -> VisaCodeMap:
    """Pulihkan pemetaan kode mask untuk satu kategori VisA.

    Melempar ValueError bila rekonstruksi tidak lolos validasi. Lebih baik
    berhenti di sini daripada melatih model di atas label yang salah.
    """
    defects = [a for a in annotations if not a.is_normal and a.mask_path is not None]
    codes_cache: dict[Path, list[int]] = {}

    def codes_of(ann: VisaAnnotation) -> list[int]:
        assert ann.mask_path is not None
        cached = codes_cache.get(ann.mask_path)
        if cached is None:
            cached = mask_codes(ann.mask_path)
            codes_cache[ann.mask_path] = cached
        return cached

    def labels_of(ann: VisaAnnotation) -> list[str]:
        return [lab for lab in ann.raw_labels if lab != _NORMAL]

    observed: defaultdict[int, Counter[str]] = defaultdict(Counter)
    for ann in defects:
        labels, codes = labels_of(ann), codes_of(ann)
        if len(labels) == 1 and len(codes) == 1:
            observed[codes[0]][labels[0]] += 1

    conflicts = {code: dict(c) for code, c in observed.items() if len(c) > 1}
    if conflicts:
        raise ValueError(
            f"[{category}] satu kode mask memetakan ke beberapa label: {conflicts}"
        )
    code_to_label: dict[int, str] = {
        code: counter.most_common(1)[0][0] for code, counter in observed.items()
    }
    learned = len(code_to_label)

    vocabulary = {label for ann in defects for label in labels_of(ann)}
    for _ in range(len(vocabulary) + 1):
        progressed = False
        for ann in defects:
            known = set(code_to_label.values())
            unknown = [c for c in codes_of(ann) if c not in code_to_label]
            remaining = [lab for lab in labels_of(ann) if lab not in known]
            if len(unknown) == 1 and len(remaining) == 1:
                code_to_label[unknown[0]] = remaining[0]
                progressed = True
        if not progressed:
            break
    inferred = len(code_to_label) - learned

    candidates: defaultdict[int, set[str]] = defaultdict(set)
    for ann in defects:
        known = set(code_to_label.values())
        unknown = [c for c in codes_of(ann) if c not in code_to_label]
        remaining = {lab for lab in labels_of(ann) if lab not in known}
        for code in unknown:
            candidates[code] |= remaining

    code_to_class: dict[int, str | None] = {
        code: to_class(label) for code, label in code_to_label.items()
    }
    ambiguous: dict[int, tuple[str, ...]] = {}
    for code, options in candidates.items():
        classes = {to_class(label) for label in options}
        if len(classes) != 1:
            raise ValueError(
                f"[{category}] kode mask {code} tidak terpecahkan: kandidat "
                f"{sorted(options)} bermuara ke kelas berbeda "
                f"{sorted(map(str, classes))}. Perlu penanganan manual."
            )
        code_to_class[code] = classes.pop()
        ambiguous[code] = tuple(sorted(options))

    checked = 0
    for ann in defects:
        labels = labels_of(ann)
        if len(labels) < 2:
            continue
        codes = codes_of(ann)
        missing = [c for c in codes if c not in code_to_class]
        if missing:
            raise ValueError(
                f"[{category}] kode mask {missing} tidak dapat dipulihkan "
                f"(gambar {ann.image_path.name})"
            )
        got = sorted(str(code_to_class[c]) for c in codes)
        want = sorted(str(to_class(lab)) for lab in labels)
        if got != want:
            raise ValueError(
                f"[{category}] validasi gagal pada {ann.image_path.name}: "
                f"kode {codes} menghasilkan kelas {got}, seharusnya {want}"
            )
        checked += 1

    return VisaCodeMap(
        category=category,
        code_to_label=dict(sorted(code_to_label.items())),
        code_to_class=dict(sorted(code_to_class.items())),
        ambiguous=dict(sorted(ambiguous.items())),
        learned_codes=learned,
        inferred_codes=inferred,
        validated_images=checked,
    )
