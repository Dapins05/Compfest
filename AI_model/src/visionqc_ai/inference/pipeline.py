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

import os
import threading
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
from visionqc_ai.inference.ocr import BatchCodeReader
from visionqc_ai.inference.validation import (
    InvalidImageError,
    validate_bytes,
    validate_dimensions,
)
from visionqc_ai.privacy import (
    AuditRecord,
    blur_faces,
    ephemeral_buffer,
    hash_image,
    strip_metadata,
)
from visionqc_ai.privacy.face_blur import MODEL_FILENAME as FACE_MODEL_FILENAME
from visionqc_ai.schemas import (
    AnomalyResult,
    BBox,
    DecisionDetail,
    Defect,
    InspectionResult,
)

# Menandai bobot yang benar-benar dipakai, bukan sekadar nama pipeline.
# Angkanya mengikuti release_tag di models/models.json; keduanya harus
# bergerak bersama, karena taksonomi enam kelas membuat bobot v1.0.0 tidak
# lagi sepadan dengan configs/inference.yaml.
MODEL_VERSION = "visionqc-models-v1.2.0-6class"

ROOT_ENV_VAR = "VISIONQC_ROOT"


def default_project_root() -> Path:
    """Cari akar proyek yang memuat configs/inference.yaml.

    Menghitung akar dari kedalaman folder sumber hanya benar selama paket
    dijalankan dari pohon sumbernya. Begitu paket dipasang ke tempat lain,
    hitungan itu menunjuk ke folder yang keliru dan kegagalannya baru terlihat
    saat model gagal dimuat. Karena itu akar ditelusuri dari penanda yang
    nyata, dan dapat ditimpa lewat variabel lingkungan bila Backend menaruh
    berkas config di tempat lain.
    """
    override = os.environ.get(ROOT_ENV_VAR)
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "configs" / "inference.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "configs/inference.yaml tidak ditemukan dari "
        f"{here}; setel {ROOT_ENV_VAR} atau berikan project_root secara eksplisit"
    )


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
        # Inferensi dijalankan satu per satu. Backend FastAPI melayani endpoint
        # sinkron di atas threadpool, sehingga dua permintaan dapat memasuki
        # pipeline yang sama pada saat bersamaan, sementara objek model dan
        # pendeteksi wajah menyimpan status di dalam dirinya.
        self._lock = threading.Lock()
        self._detector = self._load(self.config["models"]["detection"]["path"])
        self._segmenter = self._load(self.config["models"]["segmentation"]["path"])
        anomaly_config = self.config["models"]["anomaly"]
        self._anomaly = AnomalyScorer(
            project_root / anomaly_config["path"],
            imgsz=int(anomaly_config.get("imgsz", 256)),
            threads=int(self.config.get("runtime", {}).get("intra_op_threads", 4)),
        )
        self.anomaly_available = self._anomaly.available
        ocr_config = self.config["models"].get("ocr", {})
        self.ocr_enabled = bool(ocr_config.get("enabled", False))
        self.ocr_available = False
        self._ocr = BatchCodeReader(
            lang=str(ocr_config.get("lang", "en")),
            use_gpu=bool(ocr_config.get("use_gpu", False)),
            patterns=tuple(self.privacy.get("ocr_patterns", ())),
            allowlist_only=bool(self.privacy.get("ocr_allowlist_only", True)),
        )
        self.face_model_path = project_root / "models" / "onnx" / FACE_MODEL_FILENAME
        if self.config.get("runtime", {}).get("warmup_on_startup", False):
            self.warmup()

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

    @property
    def segmentation_available(self) -> bool:
        """Benar bila model segmentasi tersedia.

        Tanpa segmentasi, luas cacat dihitung dari kotak pembatas dan aturan
        luas menjadi lebih longgar. Backend melaporkannya lewat endpoint
        keterangan supaya keadaan itu terlihat dari luar.
        """
        return self._segmenter is not None

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
    def _union_mask(
        polygons: list[tuple[str, np.ndarray]], width: int, height: int
    ) -> np.ndarray | None:
        """Gabungkan seluruh poligon menjadi satu mask biner.

        Digabung lebih dulu supaya cacat yang saling bertumpang tindih tidak
        terhitung dua kali.
        """
        if not polygons:
            return None
        union = np.zeros((height, width), dtype=np.uint8)
        for _, points in polygons:
            cv2.fillPoly(union, [points.astype(np.int32)], 1)
        return union

    @staticmethod
    def _area_percentage(mask: np.ndarray | None, width: int, height: int) -> float:
        """Luas mask terhadap luas gambar, dalam persen."""
        if mask is None:
            return 0.0
        return 100.0 * float(np.count_nonzero(mask)) / (width * height)

    @classmethod
    def _box_area_percentage(
        cls,
        mask: np.ndarray | None,
        box: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> float | None:
        """Luas cacat di dalam satu kotak, terhadap luas gambar, dalam persen.

        Dihitung dari irisan mask segmentasi dengan wilayah kotak, bukan dari
        pemasangan poligon ke kotak. Irisan tidak memerlukan tebakan pasangan
        mana yang cocok, sehingga angkanya tidak bergantung pada heuristik.
        """
        if mask is None:
            return None
        x, y, w, h = box
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inside = int(np.count_nonzero(mask[y1:y2, x1:x2]))
        return round(100.0 * inside / (width * height), 4)

    def warmup(self) -> None:
        """Jalankan satu inferensi tiruan supaya permintaan pertama tidak lambat.

        Pemuatan bobot dan penyiapan grafik ONNX baru terjadi pada inferensi
        pertama. Tanpa langkah ini, pengguna pertama menanggung beberapa detik
        tambahan yang tidak ada hubungannya dengan gambarnya.
        """
        if not self.ready:
            return
        size = max(int(self.input_limits.get("min_dimension", 224)), 224)
        blank = np.zeros((size, size, 3), dtype=np.uint8)
        imgsz = int(self.config["models"]["detection"]["imgsz"])
        self._predict(self._detector, blank, imgsz)
        if self._segmenter is not None:
            self._predict(self._segmenter, blank, imgsz)
        if self._anomaly.available:
            self._anomaly.score(blank)
        if self.ocr_enabled:
            # Mesin OCR mengambil bobotnya sendiri saat pemuatan pertama.
            # Memanggilnya di sini memastikan hal itu terjadi saat startup,
            # bukan di tengah permintaan pengguna; klaim tanpa panggilan
            # jaringan pada privacy berlaku untuk waktu inferensi.
            self.ocr_available = self._ocr.available

    def inspect(
        self, image_bytes: bytes, *, anomaly_score: float | None = None
    ) -> InspectionResult:
        """Periksa satu gambar dan kembalikan hasil lengkapnya.

        Skor anomali dihitung sendiri dari model anomali bila tersedia.
        ``anomaly_score`` hanya perlu diisi bila pemanggil ingin memakai skor
        dari sumber lain, misalnya model khusus untuk kategori produk yang
        berbeda dari model bawaan.

        Melempar :class:`InvalidImageError` bila berkasnya sendiri yang tidak
        memenuhi syarat, sehingga pemanggil dapat memisahkannya dari kegagalan
        sistem.
        """
        with self._lock:
            return self._inspect(image_bytes, anomaly_score=anomaly_score)

    def _inspect(
        self, image_bytes: bytes, *, anomaly_score: float | None = None
    ) -> InspectionResult:
        """Badan inspeksi yang berjalan di dalam kunci."""
        started = time.perf_counter()
        digest = hash_image(image_bytes)
        validate_bytes(image_bytes, self.input_limits)

        # Lapisan privasi dijalankan sebelum apa pun menyentuh model. Metadata
        # dibuang lebih dulu, lalu wajah diburamkan, sehingga model tidak
        # pernah melihat data yang bukan urusannya.
        stripped = self.privacy.get("strip_exif", True)
        if stripped:
            try:
                image_bytes = strip_metadata(image_bytes)
            except ValueError as error:
                # `strip_metadata` melempar ValueError polos, sedangkan pemanggil
                # membedakan kesalahan masukan dari kegagalan sistem lewat
                # InvalidImageError. Tanpa penerjemahan ini, berkas rusak yang
                # tanda tangannya masih sah - PNG terpotong, misalnya - lolos dari
                # `validate_bytes`, gagal di sini, lalu dilaporkan sebagai 500.
                # Cabang InvalidImageError beberapa baris di bawah tidak pernah
                # tercapai untuk berkas semacam itu karena lapisan privasi sudah
                # menyentuh pikselnya lebih dulu.
                raise InvalidImageError(str(error)) from error

        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError("berkas tidak dapat dibaca sebagai gambar")
        validate_dimensions(image.shape[0], image.shape[1], self.input_limits)

        faces = 0
        if self.privacy.get("blur_faces", True):
            blurred = blur_faces(
                image,
                kernel=int(self.privacy.get("face_blur_kernel", 45)),
                model_path=self.face_model_path,
            )
            image, faces = blurred.image, blurred.faces_blurred
            self.face_blur_available = blurred.detector_available
        height, width = image.shape[:2]
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

        # Kode batch dibaca dari gambar yang SUDAH melewati lapisan privasi,
        # sehingga wajah yang kebetulan terpotret tidak pernah sampai ke mesin
        # OCR. Hasilnya masih disaring lagi dengan daftar-izin di dalam
        # BatchCodeReader sebelum meninggalkan modul ini.
        batch_code: str | None = None
        if self.ocr_enabled:
            reading = self._ocr.read(image)
            self.ocr_available = reading.available
            batch_code = reading.batch_code

        union = self._union_mask(segmented.polygons, width, height)
        area_pct = self._area_percentage(union, width, height)
        verdict = decide(
            [DetectedDefect(name, conf) for name, _, conf in detected.boxes],
            defect_area_pct=area_pct,
            anomaly_score=anomaly_score,
            config=self.decision_config,
        )

        annotated = draw_defects(image, detected.boxes, segmented.polygons)
        annotated = draw_verdict_banner(annotated, verdict.label, verdict.reason)
        encoded = to_base64_jpeg(annotated)
        if self.privacy.get("ephemeral_buffers", True):
            # Piksel gambar sudah tidak dibutuhkan setelah keluaran terbentuk.
            # Menimpanya menutup jendela ketika larik masih dipegang program
            # padahal isinya bukan lagi urusan sistem.
            with ephemeral_buffer(image):
                pass
            with ephemeral_buffer(annotated):
                pass

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
            batch_code=batch_code,
            defects=[
                Defect(
                    type=name,
                    label=CLASS_LABELS_ID.get(name, name),
                    bbox=BBox(x=box[0], y=box[1], w=box[2], h=box[3]),
                    confidence=conf,
                    area_pct=self._box_area_percentage(union, box, width, height),
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
            annotated_image_base64=encoded,
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
    return load_pipeline(project_root=project_root).inspect(
        image_bytes, anomaly_score=anomaly_score
    )


def load_pipeline(*, project_root: Path | None = None) -> InspectionPipeline:
    """Siapkan pipeline sekali lalu pakai ulang.

    Backend memanggilnya saat startup supaya bobot model dan warmup selesai
    sebelum permintaan pertama masuk, dan supaya kesiapan model dapat
    dilaporkan lewat endpoint kesehatan alih-alih baru ketahuan ketika ada
    pengguna yang gagal dilayani.
    """
    global _PIPELINE
    if _PIPELINE is None:
        root = project_root or default_project_root()
        _PIPELINE = InspectionPipeline(project_root=root)
    return _PIPELINE
