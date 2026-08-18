"""Lembar kontak untuk memeriksa anotasi hasil preprocessing.

Label YOLO digambar ulang di atas gambar olahan lalu disusun menjadi satu
lembar. Validasi statistik bisa lulus sementara koordinatnya tergeser, dan
pemeriksaan mata menutup celah itu.

    python scripts/preview_dataset.py --split test --per-class 4
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionqc_ai.data.taxonomy import CLASS_LABELS_ID, DEFECT_CLASSES  # noqa: E402

log = logging.getLogger("preview_dataset")

CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
    (220, 50, 47),
    (38, 139, 210),
    (181, 137, 0),
    (211, 54, 130),
    (133, 153, 0),
)


def _read_label(path: Path) -> list[tuple[int, list[float]]]:
    rows: list[tuple[int, list[float]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        rows.append((int(parts[0]), [float(v) for v in parts[1:]]))
    return rows


def render(image_path: Path, detect_label: Path, seg_label: Path):
    """Gambar kotak deteksi dan poligon segmentasi di atas satu gambar."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    overlay = image.copy()

    if seg_label.exists():
        for class_id, values in _read_label(seg_label):
            points = np.array(
                [
                    (round(values[i] * width), round(values[i + 1] * height))
                    for i in range(0, len(values) - 1, 2)
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(overlay, [points], CLASS_COLORS[class_id][::-1])
    image = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)

    for class_id, (xc, yc, bw, bh) in _read_label(detect_label):  # type: ignore[misc]
        x1 = round((xc - bw / 2) * width)
        y1 = round((yc - bh / 2) * height)
        x2 = round((xc + bw / 2) * width)
        y2 = round((yc + bh / 2) * height)
        colour = CLASS_COLORS[class_id][::-1]
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            image,
            DEFECT_CLASSES[class_id],
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> int:
    parser = argparse.ArgumentParser(description="Lembar kontak dataset VisionQC")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    processed = PROJECT_ROOT / "data/processed"
    detect_labels = processed / "detect/labels" / args.split
    seg_labels = processed / "seg/labels" / args.split
    images = processed / "detect/images" / args.split

    by_class: dict[int, list[Path]] = {i: [] for i in range(len(DEFECT_CLASSES))}
    for label_path in sorted(detect_labels.glob("*.txt")):
        for class_id, _ in _read_label(label_path):
            by_class[class_id].append(label_path)

    chosen: list[tuple[int, Path]] = []
    for class_id, paths in by_class.items():
        unique = sorted({p for p in paths})
        picked = random.Random(f"{args.seed}:{class_id}").sample(
            unique, min(args.per_class, len(unique))
        )
        chosen.extend((class_id, p) for p in picked)

    if not chosen:
        log.error("tidak ada label pada split %s", args.split)
        return 1

    columns = args.per_class
    rows = (len(chosen) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.4 * rows))
    axes = axes.reshape(-1) if rows * columns > 1 else [axes]

    for axis in axes:
        axis.axis("off")
    for axis, (class_id, label_path) in zip(axes, chosen, strict=False):
        image = render(
            images / f"{label_path.stem}.jpg", label_path, seg_labels / label_path.name
        )
        axis.imshow(image)
        axis.set_title(
            f"{CLASS_LABELS_ID[DEFECT_CLASSES[class_id]]} - {label_path.stem[:34]}",
            fontsize=8,
        )

    figure.suptitle(
        f"Verifikasi anotasi hasil preprocessing - split {args.split} "
        f"(kotak = deteksi, arsiran = segmentasi)",
        fontsize=11,
    )
    figure.tight_layout()

    output = PROJECT_ROOT / f"reports/figures/dataset_samples_{args.split}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=110, bbox_inches="tight")
    plt.close(figure)
    log.info("Lembar kontak %d gambar ditulis ke %s", len(chosen), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
