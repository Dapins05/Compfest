"""Konversi VisA menjadi SampleRecord.

Jenis cacat tersimpan sebagai kode piksel di dalam mask, bukan pada nama
folder, sehingga pemetaannya dipulihkan lebih dulu oleh visa_codes.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from visionqc_ai.data.mask_utils import (
    ExtractionStats,
    extract_instances,
    merge_stats,
)
from visionqc_ai.data.records import ConversionResult, SampleRecord
from visionqc_ai.data.sources import VISA_ROOT_DIRNAME, CategorySpec
from visionqc_ai.data.taxonomy import map_raw_label
from visionqc_ai.data.visa_codes import VisaCodeMap, build_code_map, read_annotations


def convert_category(
    spec: CategorySpec,
    raw_root: Path,
    *,
    imgsz: int,
    min_component_area_px: int,
    min_scaled_side_px: int,
    polygon_epsilon_ratio: float,
    polygon_max_points: int,
) -> tuple[ConversionResult, VisaCodeMap]:
    """Ubah satu kategori VisA menjadi record cacat dan record normal.

    Mengembalikan hasil konversi beserta pemetaan kode mask yang dipulihkan,
    supaya rekonstruksinya bisa dicatat di laporan dan diaudit ulang.
    """
    visa_root = raw_root / VISA_ROOT_DIRNAME
    category_dir = spec.raw_dir(raw_root)
    if not category_dir.is_dir():
        raise FileNotFoundError(f"folder VisA tidak ditemukan: {category_dir}")

    annotations = read_annotations(category_dir, visa_root)
    code_map = build_code_map(spec.name, annotations, to_class=map_raw_label)

    defects: list[SampleRecord] = []
    normals: list[SampleRecord] = []
    stats = ExtractionStats()
    missing_masks = 0

    for ann in annotations:
        stem = ann.image_path.stem
        if ann.is_normal:
            image = cv2.imread(str(ann.image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            normals.append(
                SampleRecord(
                    sample_id=f"visa_{spec.name}_normal_{stem}",
                    category=spec.name,
                    source=spec.source,
                    source_image=ann.image_path,
                    image_size=(width, height),
                    instances=(),
                )
            )
            continue

        if ann.mask_path is None or not ann.mask_path.exists():
            missing_masks += 1
            continue
        mask = cv2.imread(str(ann.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            missing_masks += 1
            continue

        height, width = mask.shape[:2]
        scale = imgsz / max(height, width)
        instances, sub_stats = extract_instances(
            mask,
            code_map.code_to_class.get,
            min_component_area_px=min_component_area_px,
            min_scaled_side_px=min_scaled_side_px,
            scale=min(scale, 1.0),
            polygon_epsilon_ratio=polygon_epsilon_ratio,
            polygon_max_points=polygon_max_points,
        )
        stats = merge_stats(stats, sub_stats)
        if not instances:
            continue

        defects.append(
            SampleRecord(
                sample_id=f"visa_{spec.name}_anomaly_{stem}",
                category=spec.name,
                source=spec.source,
                source_image=ann.image_path,
                image_size=(width, height),
                instances=tuple(instances),
            )
        )

    return ConversionResult(defects, normals, stats, missing_masks), code_map
