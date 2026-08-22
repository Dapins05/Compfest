"""Konversi PKU-GoodsAD menjadi SampleRecord.

Tata letaknya mengikuti gaya MVTec, tetapi dengan dua perbedaan yang membuat
konverter MVTec tidak bisa dipakai ulang apa adanya: gambarnya berformat JPEG
dan berkas mask memakai nama batang yang sama persis dengan gambarnya, tanpa
akhiran _mask.

Jenis cacat diambil dari nama folder, sehingga satu gambar hanya memiliki satu
jenis cacat.
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
from visionqc_ai.data.sources import CategorySpec
from visionqc_ai.data.taxonomy import map_raw_label

GOOD_DIRNAME = "good"
IMAGE_SUFFIX = "*.jpg"
MASK_SUFFIX = ".png"


def _image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"gambar tidak terbaca: {path}")
    height, width = image.shape[:2]
    return width, height


def convert_category(
    spec: CategorySpec,
    raw_root: Path,
    *,
    imgsz: int,
    min_component_area_px: int,
    min_scaled_side_px: int,
    polygon_epsilon_ratio: float,
    polygon_max_points: int,
) -> ConversionResult:
    """Ubah satu kategori GoodsAD menjadi record cacat dan record normal."""
    root = spec.raw_dir(raw_root)
    if not root.is_dir():
        raise FileNotFoundError(f"folder GoodsAD tidak ditemukan: {root}")

    defects: list[SampleRecord] = []
    normals: list[SampleRecord] = []
    stats = ExtractionStats()
    missing_masks = 0

    for subset in ("train", "test"):
        good_dir = root / subset / GOOD_DIRNAME
        if not good_dir.is_dir():
            continue
        for image_path in sorted(good_dir.glob(IMAGE_SUFFIX)):
            normals.append(
                SampleRecord(
                    sample_id=f"goodsad_{spec.name}_{subset}_good_{image_path.stem}",
                    category=spec.name,
                    source=spec.source,
                    source_image=image_path,
                    image_size=_image_size(image_path),
                    instances=(),
                )
            )

    for defect_dir in sorted((root / "test").iterdir()):
        if not defect_dir.is_dir() or defect_dir.name == GOOD_DIRNAME:
            continue
        class_name = map_raw_label(defect_dir.name)
        if class_name is None:
            continue

        for image_path in sorted(defect_dir.glob(IMAGE_SUFFIX)):
            mask_path = (
                root
                / "ground_truth"
                / defect_dir.name
                / f"{image_path.stem}{MASK_SUFFIX}"
            )
            if not mask_path.exists():
                missing_masks += 1
                continue

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                missing_masks += 1
                continue

            height, width = mask.shape[:2]
            scale = imgsz / max(height, width)
            instances, sub_stats = extract_instances(
                mask,
                lambda _code, name=class_name: name,
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
                    sample_id=(
                        f"goodsad_{spec.name}_{defect_dir.name}_{image_path.stem}"
                    ),
                    category=spec.name,
                    source=spec.source,
                    source_image=image_path,
                    image_size=(width, height),
                    instances=tuple(instances),
                )
            )

    return ConversionResult(defects, normals, stats, missing_masks)
