"""Orkestrasi inferensi: dari byte gambar sampai InspectionResult.

Ini satu-satunya pintu masuk dari Backend ke modul AI. Backend tidak perlu tahu
apa pun tentang model, ambang, atau pustaka yang dipakai di dalamnya.

Model dimuat sekali lewat :class:`InspectionPipeline` dan dipakai berulang.
Memuat model pada setiap permintaan akan menambah beberapa detik pada tiap
inspeksi dan membuat permintaan pertama tampak jauh lebih lambat.

Inferensi berjalan di atas berkas ONNX melalui Ultralytics, yang menangani
pascapemrosesan seperti NMS dan penguraian mask. Menulis ulang pascapemrosesan
itu sendiri berarti menduplikasi kode yang sudah teruji tanpa manfaat yang
sepadan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from visionqc_ai.data.taxonomy import CLASS_LABELS_ID, DEFECT_CLASSES
from visionqc_ai.inference.annotate import (
    draw_defects,
    draw_verdict_banner,
    to_base64_jpeg,
)
from visionqc_ai.inference.anomaly import AnomalyScorer
from visionqc_ai.inference.decision import DecisionConfig, DetectedDefect, decide
from visionqc_ai.privacy import AuditRecord, blur_faces, hash_image, strip_metadata
from visionqc_ai.schemas import (
    AnomalyResult,
    BBox,
    DecisionDetail,
    Defect,
    InspectionResult,
)

MODEL_VERSION = "visionqc-detect-seg-v1"


@dataclass
class _Prediction:
    """Keluaran mentah satu model untuk satu gambar."""

    boxes: list[tuple[str, tuple[int, int, int, int], float]]
    polygons: list[tuple[str, np.ndarray]]


class InspectionPipeline:
    """Memuat model sekali lalu melayani banyak permintaan inspeksi."""

    def __init__(
        self,
        *,
        project_root: Path,
        config_path: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.project_root = project_root
        path = config_path or project_root / "configs" / "inference.yaml"
        with path.open(encoding="utf-8") as handle:
            self.config: dict[str, Any] = yaml.safe_load(handle)

        self.decision_config = DecisionConfig.from_yaml(self.config)
        self.device = device
        self.input_limits = self.config.get("input", {})
        self.privacy = self.config.get("privacy", {})
        self.face_blur_available = True
        self.last_audit: AuditRecord | None = None
        self._detector = self._load(self.config["models"]["detection"]["path"])
        self._segmenter = self._load(self.config["models"]["segmentation"]["path"])
        anomaly_config = self.config["models"]["anomaly"]
        self._anomaly = AnomalyScorer(
            project_root / anomaly_config["path"],
            imgsz=int(anomaly_config.get("imgsz", 256)),
            threads=int(self.config.get("runtime", {}).get("intra_op_threads", 4)),
        )
        self.anomaly_available = self._anomaly.available

    def _load(self, relative: str) -> Any:
        """Muat sebuah model bila berkasnya ada; kembalikan None bila tidak.

        Sistem tetap berjalan tanpa salah satu model, dengan kemampuan yang
        berkurang. Gagal total hanya karena satu berkas tidak ada akan membuat
        keseluruhan sistem tidak dapat didemokan.
        """
        path = self.project_root / relative
        if not path.exists():
            return None
        from ultralytics import YOLO

        return YOLO(str(path), task="segment" if "seg" in path.name else "detect")

    @property
    def ready(self) -> bool:
        """Benar bila setidaknya model deteksi tersedia."""
        return self._detector is not None

    def _predict(self, model: Any, image: np.ndarray, imgsz: int) -> _Prediction:
        detection = self.config["models"]["detection"]
        result = model.predict(
            source=image,
            imgsz=imgsz,
            conf=float(detection["conf_threshold"]),
            iou=float(detection["iou_threshold"]),
            max_det=int(detection["max_detections"]),
            device=self.device,
            verbose=False,
        )[0]

        boxes: list[tuple[str, tuple[int, int, int, int], float]] = []
        if result.boxes is not None:
            for xyxy, class_id, confidence in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.cls.tolist(),
                result.boxes.conf.tolist(),
                strict=True,
            ):
                index = int(class_id)
                if not 0 <= index < len(DEFECT_CLASSES):
                    continue
                x1, y1, x2, y2 = (round(v) for v in xyxy)
                boxes.append(
                    (
                        DEFECT_CLASSES[index],
                        (x1, y1, max(0, x2 - x1), max(0, y2 - y1)),
                        float(confidence),
                    )
                )

        polygons: list[tuple[str, np.ndarray]] = []
        if getattr(result, "masks", None) is not None and result.boxes is not None:
            for points, class_id in zip(
                result.masks.xy, result.boxes.cls.tolist(), strict=True
            ):
                index = int(class_id)
                if 0 <= index < len(DEFECT_CLASSES) and len(points) >= 3:
                    polygons.append((DEFECT_CLASSES[index], np.asarray(points)))
        return _Prediction(boxes=boxes, polygons=polygons)

    @staticmethod
    def _area_percentage(
        polygons: list[tuple[str, np.ndarray]], width: int, height: int
    ) -> float:
        """Luas gabungan seluruh mask terhadap luas gambar, dalam persen.

        Digabung lebih dulu supaya cacat yang saling bertumpang tindih tidak
        terhitung dua kali.
        """
        if not polygons:
            return 0.0
        union = np.zeros((height, width), dtype=np.uint8)
        for _, points in polygons:
            cv2.fillPoly(union, [points.astype(np.int32)], 1)
        return 100.0 * float(np.count_nonzero(union)) / (width * height)

    def inspect(
        self, image_bytes: bytes, *, anomaly_score: float | None = None
    ) -> InspectionResult:
        """Periksa satu gambar dan kembalikan hasil lengkapnya.

        Skor anomali dihitung sendiri dari model anomali bila tersedia.
        ``anomaly_score`` hanya perlu diisi bila pemanggil ingin memakai skor
        dari sumber lain, misalnya model khusus untuk kategori produk yang
        berbeda dari model bawaan.
        """
        started = time.perf_counter()
        digest = hash_image(image_bytes)

        # Lapisan privasi dijalankan sebelum apa pun menyentuh model. Metadata
        # dibuang lebih dulu, lalu wajah diburamkan, sehingga model tidak
        # pernah melihat data yang bukan urusannya.
        stripped = self.privacy.get("strip_exif", True)
        if stripped:
            image_bytes = strip_metadata(image_bytes)

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("berkas tidak dapat dibaca sebagai gambar")

        faces = 0
        if self.privacy.get("blur_faces", True):
            blurred = blur_faces(
                image, kernel=int(self.privacy.get("face_blur_kernel", 45))
            )
            image, faces = blurred.image, blurred.faces_blurred
            self.face_blur_available = blurred.detector_available
        height, width = image.shape[:2]
        minimum = int(self.input_limits.get("min_dimension", 0))
        if min(height, width) < minimum:
            raise ValueError(
                f"dimensi terkecil {min(height, width)} piksel, minimum {minimum}"
            )
        if not self.ready:
            raise RuntimeError(
                "model deteksi belum tersedia; jalankan scripts/export_onnx.py"
            )

        imgsz = int(self.config["models"]["detection"]["imgsz"])
        detected = self._predict(self._detector, image, imgsz)
        segmented = (
            self._predict(self._segmenter, image, imgsz)
            if self._segmenter is not None
            else _Prediction([], [])
        )

        if anomaly_score is None:
            measured = self._anomaly.score(image)
            anomaly_score = measured.score
            self.anomaly_available = measured.available
        else:
            self.anomaly_available = True

        area_pct = self._area_percentage(segmented.polygons, width, height)
        verdict = decide(
            [DetectedDefect(name, conf) for name, _, conf in detected.boxes],
            defect_area_pct=area_pct,
            anomaly_score=anomaly_score,
            config=self.decision_config,
        )

        annotated = draw_defects(image, detected.boxes, segmented.polygons)
        annotated = draw_verdict_banner(annotated, verdict.label, verdict.reason)

        threshold = self.decision_config.anomaly_threshold
        latency = round((time.perf_counter() - started) * 1000)
        self.last_audit = AuditRecord(
            image_sha256=digest,
            verdict=verdict.label,
            latency_ms=latency,
            faces_blurred=faces,
            metadata_stripped=stripped,
        )
        return InspectionResult(
            verdict=verdict.label,
            reason=verdict.reason,
            confidence=verdict.calibrated_probability,
            defects=[
                Defect(
                    type=name,
                    label=CLASS_LABELS_ID.get(name, name),
                    bbox=BBox(x=box[0], y=box[1], w=box[2], h=box[3]),
                    confidence=conf,
                )
                for name, box, conf in detected.boxes
            ],
            defect_area_pct=round(area_pct, 4),
            anomaly=AnomalyResult(
                score=anomaly_score,
                threshold=threshold,
                exceeded=anomaly_score > threshold,
            ),
            decision=DecisionDetail(
                calibrated_probability=verdict.calibrated_probability,
                prediction_set=list(verdict.prediction_set),
                severity=verdict.severity,
                conformal_alpha=self.decision_config.conformal.alpha,
            ),
            annotated_image_base64=to_base64_jpeg(annotated),
            model_version=MODEL_VERSION,
            latency_ms=latency,
        )


_PIPELINE: InspectionPipeline | None = None


def run_inspection(
    image_bytes: bytes,
    *,
    project_root: Path | None = None,
    anomaly_score: float | None = None,
) -> InspectionResult:
    """Pintu masuk tunggal dari Backend ke modul AI.

    Pipeline disimpan pada tingkat modul supaya model hanya dimuat sekali
    selama proses hidup. Backend memanggilnya sekali saat startup agar
    permintaan pertama tidak menanggung biaya pemuatan.
    """
    global _PIPELINE
    if _PIPELINE is None:
        root = project_root or Path(__file__).resolve().parents[3]
        _PIPELINE = InspectionPipeline(project_root=root)
    return _PIPELINE.inspect(image_bytes, anomaly_score=anomaly_score)
