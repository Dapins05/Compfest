"""Konversi mask piksel menjadi anotasi format YOLO.

Mask dipecah per kode jenis cacat lalu per komponen terhubung. Kotak pembatas
dan poligon diturunkan dari sumber yang sama sehingga keduanya tidak pernah
saling bertentangan.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

CodeMapper = Callable[[int], str | None]


@dataclass(frozen=True)
class DefectInstance:
    """Satu cacat pada satu gambar, dalam koordinat piksel."""

    class_name: str
    bbox: tuple[int, int, int, int]
    polygon: tuple[tuple[int, int], ...]
    area_px: int

    @property
    def longest_side(self) -> int:
        return max(self.bbox[2], self.bbox[3])


@dataclass(frozen=True)
class ExtractionStats:
    """Berapa instance yang diambil dan berapa yang dibuang, beserta alasannya."""

    kept: int = 0
    dropped_small_area: int = 0
    dropped_small_side: int = 0
    dropped_unmapped: int = 0
    dropped_no_polygon: int = 0


def merge_stats(left: ExtractionStats, right: ExtractionStats) -> ExtractionStats:
    """Jumlahkan dua statistik ekstraksi."""
    return ExtractionStats(
        kept=left.kept + right.kept,
        dropped_small_area=left.dropped_small_area + right.dropped_small_area,
        dropped_small_side=left.dropped_small_side + right.dropped_small_side,
        dropped_unmapped=left.dropped_unmapped + right.dropped_unmapped,
        dropped_no_polygon=left.dropped_no_polygon + right.dropped_no_polygon,
    )


def resize_longest(image: np.ndarray, imgsz: int) -> tuple[np.ndarray, float]:
    """Perkecil gambar hingga sisi terpanjangnya imgsz, rasio dipertahankan.

    Mengembalikan gambar hasil beserta faktor skala yang dipakai.
    """
    height, width = image.shape[:2]
    scale = imgsz / max(height, width)
    if scale >= 1.0:
        return image, 1.0
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _contour_to_polygon(
    contour: np.ndarray, *, epsilon_ratio: float, max_points: int
) -> tuple[tuple[int, int], ...]:
    """Sederhanakan kontur menjadi poligon ringkas yang tetap setia bentuknya."""
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    while len(approx) > max_points and epsilon < cv2.arcLength(contour, True):
        epsilon *= 1.5
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return tuple((int(x), int(y)) for x, y in approx)


def extract_instances(
    mask: np.ndarray,
    code_to_class: CodeMapper,
    *,
    min_component_area_px: int,
    min_scaled_side_px: int,
    scale: float,
    polygon_epsilon_ratio: float,
    polygon_max_points: int,
) -> tuple[list[DefectInstance], ExtractionStats]:
    """Pecah mask menjadi daftar DefectInstance.

    Tiap kode piksel diproses terpisah lalu dipecah per komponen terhubung,
    sehingga dua cacat sejenis yang terpisah letaknya tetap menjadi dua
    instance dan bukan satu kotak besar yang meliputi keduanya.

    Penyaring min_scaled_side_px diuji terhadap ukuran setelah resize, karena
    itu yang dilihat model.
    """
    instances: list[DefectInstance] = []
    kept = small_area = small_side = unmapped = no_polygon = 0

    for code in (int(v) for v in np.unique(mask) if v > 0):
        class_name = code_to_class(code)
        component = (mask == code).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(component)

        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if class_name is None:
                unmapped += 1
                continue
            if area < min_component_area_px:
                small_area += 1
                continue

            x = int(stats[index, cv2.CC_STAT_LEFT])
            y = int(stats[index, cv2.CC_STAT_TOP])
            w = int(stats[index, cv2.CC_STAT_WIDTH])
            h = int(stats[index, cv2.CC_STAT_HEIGHT])
            if max(w, h) * scale < min_scaled_side_px:
                small_side += 1
                continue

            contours, _ = cv2.findContours(
                (labels == index).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                no_polygon += 1
                continue
            polygon = _contour_to_polygon(
                max(contours, key=cv2.contourArea),
                epsilon_ratio=polygon_epsilon_ratio,
                max_points=polygon_max_points,
            )
            if len(polygon) < 3:
                no_polygon += 1
                continue

            instances.append(
                DefectInstance(
                    class_name=class_name,
                    bbox=(x, y, w, h),
                    polygon=polygon,
                    area_px=area,
                )
            )
            kept += 1

    stats_out = ExtractionStats(
        kept=kept,
        dropped_small_area=small_area,
        dropped_small_side=small_side,
        dropped_unmapped=unmapped,
        dropped_no_polygon=no_polygon,
    )
    return instances, stats_out


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def to_detect_line(
    instance: DefectInstance, class_id: int, width: int, height: int
) -> str:
    """Baris label deteksi YOLO: cls xc yc w h, ternormalisasi."""
    x, y, w, h = instance.bbox
    xc = _clamp01((x + w / 2) / width)
    yc = _clamp01((y + h / 2) / height)
    nw = _clamp01(w / width)
    nh = _clamp01(h / height)
    return f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"


def to_segment_line(
    instance: DefectInstance, class_id: int, width: int, height: int
) -> str:
    """Baris label segmentasi YOLO: cls x1 y1 x2 y2 dan seterusnya, ternormalisasi."""
    coords: list[str] = []
    for px, py in instance.polygon:
        coords.append(f"{_clamp01(px / width):.6f}")
        coords.append(f"{_clamp01(py / height):.6f}")
    return f"{class_id} " + " ".join(coords)
