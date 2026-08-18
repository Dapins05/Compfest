"""Ekspor bobot hasil fine-tuning ke ONNX dan ukur latensinya di CPU.

Target penyajian adalah CPU, bukan GPU. Panitia menjalankan sistem di komputer
mereka sendiri dan tidak dapat diandalkan memiliki kartu grafis, sehingga
angka latensi yang bermakna adalah angka CPU, bukan angka GPU yang tercatat
saat pelatihan.

Latensi dilaporkan sebagai median dan persentil ke-95 dari sejumlah pengulangan.
Satu pengukuran tunggal tidak bermakna karena sangat dipengaruhi keadaan mesin
saat itu; persentil ke-95 memperlihatkan seberapa buruk kasus terburuknya.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExportResult:
    """Hasil ekspor satu model."""

    name: str
    source: str
    onnx_path: str
    size_mb: float
    imgsz: int
    opset: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatencyResult:
    """Ringkasan latensi satu model pada satu perangkat."""

    name: str
    device: str
    runs: int
    median_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def export_to_onnx(
    weights: Path, *, imgsz: int = 640, opset: int = 12, simplify: bool = True
) -> ExportResult:
    """Ekspor bobot Ultralytics ke ONNX.

    Berkas hasil ekspor diletakkan Ultralytics di samping bobot asalnya, lalu
    dipindahkan pemanggil ke ``models/onnx/`` agar seluruh artefak penyajian
    berkumpul di satu tempat.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    exported = model.export(
        format="onnx", imgsz=imgsz, opset=opset, simplify=simplify, dynamic=False
    )
    path = Path(exported)
    return ExportResult(
        name=weights.parent.parent.name,
        source=str(weights),
        onnx_path=str(path),
        size_mb=round(path.stat().st_size / 1024 / 1024, 2),
        imgsz=imgsz,
        opset=opset,
    )


def benchmark_onnx(
    onnx_path: Path,
    *,
    name: str,
    imgsz: int = 640,
    runs: int = 100,
    warmup: int = 10,
    threads: int = 4,
) -> LatencyResult:
    """Ukur latensi inferensi ONNX di CPU.

    Beberapa pengulangan pertama dibuang sebagai pemanasan karena pemanggilan
    awal masih menanggung biaya alokasi memori dan pemuatan kernel yang tidak
    akan berulang pada permintaan berikutnya.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(
        str(onnx_path), options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    dummy = np.random.default_rng(42).random((1, 3, imgsz, imgsz), dtype=np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: dummy})

    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        timings.append((time.perf_counter() - start) * 1000.0)

    values = np.asarray(timings, dtype=float)
    return LatencyResult(
        name=name,
        device="cpu",
        runs=runs,
        median_ms=round(float(np.median(values)), 3),
        p95_ms=round(float(np.percentile(values, 95)), 3),
        mean_ms=round(float(values.mean()), 3),
        min_ms=round(float(values.min()), 3),
        max_ms=round(float(values.max()), 3),
    )
