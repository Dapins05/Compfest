"""Penilaian anomali di atas berkas ONNX.

Detektor cacat hanya mengenali jenis yang pernah dilabeli. Model anomali
dilatih tanpa satu pun contoh cacat: ia mempelajari seperti apa produk normal
lalu menandai apa pun yang menyimpang, sehingga cacat jenis baru tetap
terjaring.

Skornya harus berada pada skala yang sama persis dengan ambang di
`configs/inference.yaml`, yang diturunkan dengan teori nilai ekstrem pada
Step 6. Karena itu prapemrosesan di sini menyalin persis apa yang dipakai saat
ambang tersebut dihitung: resize ke 256 piksel, urutan kanal RGB, lalu
normalisasi memakai rerata dan simpangan baku ImageNet. Menyimpang sedikit saja
membuat skor tidak lagi sebanding dengan ambangnya.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class AnomalyScore:
    """Skor anomali sebuah gambar beserta status ketersediaan model."""

    score: float
    available: bool
    note: str = ""


class AnomalyScorer:
    """Memuat model anomali sekali lalu menilai banyak gambar."""

    def __init__(self, model_path: Path, *, imgsz: int = 256, threads: int = 4) -> None:
        self.model_path = model_path
        self.imgsz = imgsz
        self._session: Any | None = None
        self._input_name = ""

        if not model_path.exists():
            return
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    @property
    def available(self) -> bool:
        return self._session is not None

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Siapkan gambar BGR menjadi tensor masukan model."""
        resized = cv2.resize(image, (self.imgsz, self.imgsz))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalised = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return normalised.transpose(2, 0, 1)[None]

    def score(self, image: np.ndarray) -> AnomalyScore:
        """Nilai satu gambar BGR.

        Skornya adalah nilai tertinggi pada peta anomali, sebagaimana yang
        dipakai ketika ambang dihitung. Bila model tidak tersedia, hal itu
        dinyatakan alih-alih mengembalikan nol diam-diam; nol akan terbaca
        sebagai "tidak ada anomali" dan menyembunyikan bahwa lapisan ini mati.
        """
        if self._session is None:
            return AnomalyScore(
                score=0.0,
                available=False,
                note=(
                    f"{self.model_path.name} tidak tersedia; "
                    "penilaian anomali tidak aktif"
                ),
            )
        outputs = self._session.run(None, {self._input_name: self.preprocess(image)})
        highest = max(float(np.asarray(output).max()) for output in outputs)
        return AnomalyScore(score=highest, available=True)
