"""Representasi antara antara konverter dataset dan penulis format YOLO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from visionqc_ai.data.mask_utils import DefectInstance, ExtractionStats


@dataclass(frozen=True)
class SampleRecord:
    """Satu gambar beserta seluruh cacat yang ada padanya."""

    sample_id: str
    category: str
    source: str
    source_image: Path
    image_size: tuple[int, int]
    instances: tuple[DefectInstance, ...]

    @property
    def is_normal(self) -> bool:
        return not self.instances

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(inst.class_name for inst in self.instances)

    def dominant_class(self) -> str:
        """Kelas dengan total luas terbesar, dipakai sebagai kunci stratifikasi.

        Gambar normal memakai penanda __normal__ agar ikut terstratifikasi.
        """
        if not self.instances:
            return "__normal__"
        totals: dict[str, int] = {}
        for inst in self.instances:
            totals[inst.class_name] = totals.get(inst.class_name, 0) + inst.area_px
        return max(totals.items(), key=lambda kv: kv[1])[0]


@dataclass(frozen=True)
class ConversionResult:
    """Hasil konversi satu kategori dataset."""

    defects: list[SampleRecord]
    normals: list[SampleRecord]
    stats: ExtractionStats
    missing_masks: int = 0
