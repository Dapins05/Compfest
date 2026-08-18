"""Penulisan dataset hasil preprocessing ke disk.

Menghasilkan tata letak detect, seg, dan anomaly dari pembagian split yang
sama, sehingga tidak ada gambar uji yang bocor antar model. Gambar detect dan
seg identik, jadi salinan kedua dibuat sebagai hard link bila didukung.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2

from visionqc_ai.data.mask_utils import resize_longest, to_detect_line, to_segment_line
from visionqc_ai.data.records import SampleRecord
from visionqc_ai.data.split import SPLIT_NAMES, SplitAssignment

DETECT = "detect"
SEGMENT = "seg"


def _ensure_empty(directory: Path) -> None:
    """Siapkan folder keluaran dalam keadaan bersih.

    Sisa berkas dari run sebelumnya membuat hitungan tidak bisa dipercaya.
    """
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def write_images(
    assignment: SplitAssignment, out_root: Path, *, imgsz: int, jpeg_quality: int
) -> int:
    """Tulis gambar yang sudah diperkecil ke out_root/images/<split>."""
    written = 0
    for split, records in assignment.as_dict().items():
        target_dir = out_root / "images" / split
        target_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            image = cv2.imread(str(record.source_image))
            if image is None:
                raise FileNotFoundError(f"gambar tidak terbaca: {record.source_image}")
            resized, _ = resize_longest(image, imgsz)
            cv2.imwrite(
                str(target_dir / f"{record.sample_id}.jpg"),
                resized,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            written += 1
    return written


def mirror_images(source_root: Path, target_root: Path) -> int:
    """Duplikasi folder gambar ke dataset lain tanpa membaca ulang gambar."""
    mirrored = 0
    for split in SPLIT_NAMES:
        source_dir = source_root / "images" / split
        target_dir = target_root / "images" / split
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(source_dir.glob("*.jpg")):
            _link_or_copy(path, target_dir / path.name)
            mirrored += 1
    return mirrored


def write_labels(
    assignment: SplitAssignment,
    out_root: Path,
    class_ids: dict[str, int],
    *,
    task: str,
) -> int:
    """Tulis berkas label YOLO, satu berkas per gambar.

    Gambar normal tetap mendapat berkas label kosong, karena begitulah YOLO
    mengenali gambar latar belakang yang menekan alarm palsu.
    """
    to_line = to_detect_line if task == DETECT else to_segment_line
    written = 0
    for split, records in assignment.as_dict().items():
        target_dir = out_root / "labels" / split
        target_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            width, height = record.image_size
            lines = [
                to_line(instance, class_ids[instance.class_name], width, height)
                for instance in record.instances
            ]
            path = target_dir / f"{record.sample_id}.txt"
            path.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            written += 1
    return written


def write_data_yaml(
    out_root: Path, classes: tuple[str, ...], *, task: str, notes: str
) -> Path:
    """Tulis data.yaml yang dibaca Ultralytics."""
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(classes))
    content = (
        f"# data.yaml - dataset VisionQC ({task})\n"
        f"# Dihasilkan otomatis oleh scripts/prepare_dataset.py.\n"
        f"# Jangan diedit tangan; ubah configs/dataset.yaml lalu jalankan ulang.\n"
        f"#\n"
        f"# {notes}\n\n"
        f"path: {out_root.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n\n"
        f"nc: {len(classes)}\n"
        f"names:\n{names}\n"
    )
    path = out_root / "data.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def write_yolo_dataset(
    assignment: SplitAssignment,
    out_root: Path,
    classes: tuple[str, ...],
    class_ids: dict[str, int],
    *,
    task: str,
    imgsz: int,
    jpeg_quality: int,
    notes: str,
    mirror_from: Path | None = None,
) -> dict[str, int]:
    """Tulis satu dataset YOLO lengkap: gambar, label, dan data.yaml."""
    _ensure_empty(out_root)
    if mirror_from is None:
        images = write_images(
            assignment, out_root, imgsz=imgsz, jpeg_quality=jpeg_quality
        )
    else:
        images = mirror_images(mirror_from, out_root)
    labels = write_labels(assignment, out_root, class_ids, task=task)
    write_data_yaml(out_root, classes, task=task, notes=notes)
    return {"images": images, "labels": labels}


def write_anomaly_dataset(
    out_root: Path,
    category: str,
    *,
    normals: dict[str, list[SampleRecord]],
    defects: dict[str, list[SampleRecord]],
    imgsz: int,
    jpeg_quality: int,
) -> dict[str, int]:
    """Tulis tata letak bergaya MVTec untuk anomalib.

    EfficientAD hanya belajar dari gambar normal, jadi train tidak berisi satu
    pun gambar cacat. Split-nya sama dengan dataset deteksi supaya tidak ada
    gambar uji yang bocor ke pelatihan.
    """
    root = out_root / category
    _ensure_empty(root)

    def emit(records: list[SampleRecord], target: Path) -> int:
        target.mkdir(parents=True, exist_ok=True)
        for record in records:
            image = cv2.imread(str(record.source_image))
            if image is None:
                continue
            resized, _ = resize_longest(image, imgsz)
            cv2.imwrite(
                str(target / f"{record.sample_id}.jpg"),
                resized,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
        return len(records)

    train_normals = normals.get("train", []) + normals.get("val", [])
    counts = {
        "train_good": emit(train_normals, root / "train" / "good"),
        "test_good": emit(normals.get("test", []), root / "test" / "good"),
        "test_defect": emit(defects.get("test", []), root / "test" / "defect"),
    }
    return counts
