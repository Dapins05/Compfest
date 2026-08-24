"""Ekspor model deteksi dan segmentasi ke ONNX lalu ukur latensinya di CPU.

    python scripts/export_onnx.py
    python scripts/export_onnx.py --runs 200

Berkas ONNX tidak masuk git karena ukurannya; yang ikut tercatat adalah
metriknya di reports/metrics/latency_benchmark.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionqc_ai.export.onnx_export import benchmark_onnx, export_to_onnx  # noqa: E402

log = logging.getLogger("export_onnx")

# Run yang diekspor.
#
# Deteksi diambil dari `detect_balanced` (24 Agustus 2026), yaitu run yang
# dilatih dengan tambahan 110 gambar cacat `kotor` sintetik dan 300 gambar
# latar sebagai negatif keras. Run itu sempat terhenti di epoch 124 dan
# disambung sampai 150 lewat `scripts/resume_detection.py`.
#
# Peringatan pada versi berkas ini sebelumnya menyuruh JANGAN mengarah ke sana,
# dan pada saat itu peringatannya benar: run-nya belum tuntas. Setelah tuntas
# dan diukur, gambarannya berubah - pada set kalibrasi biaya harapannya 9,1
# persen lebih rendah, dan pada set eval `kotor` yang ditahan recall-nya 0,4171
# melawan 0,0160 milik run sebelumnya. Rinciannya di EXPERIMENTS.md bagian 3.
#
# Segmentasi tetap dari `seg_goodsad`; pasangan `seg_balanced` tidak pernah
# dilatih, dan jalur segmentasi tidak terpengaruh penambahan data deteksi.
#
# Run lama detect dan seg sengaja tidak dihapus: keduanya menjadi rujukan
# keadaan sebelum dataset diperbesar.
MODELS = (
    (
        "detect",
        "models/finetuned/detect_balanced/weights/best.pt",
        "yolo11n-defect.onnx",
    ),
    (
        "seg",
        "models/finetuned/seg_goodsad/weights/best.pt",
        "yolo11n-seg-defect.onnx",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ekspor ONNX dan tolok ukur latensi")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=PROJECT_ROOT / "configs/inference.yaml",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    with args.inference_config.open(encoding="utf-8") as handle:
        threads = int(yaml.safe_load(handle)["runtime"]["intra_op_threads"])

    target_dir = PROJECT_ROOT / "models" / "onnx"
    target_dir.mkdir(parents=True, exist_ok=True)

    log.info("Ekspor ONNX dan tolok ukur latensi CPU")
    log.info(
        "imgsz %d | opset %d | %d pengulangan | %d utas",
        args.imgsz,
        args.opset,
        args.runs,
        threads,
    )

    exports: list[dict[str, Any]] = []
    latencies: list[dict[str, Any]] = []
    for name, relative, filename in MODELS:
        weights = PROJECT_ROOT / relative
        if not weights.exists():
            log.warning("  %s: bobot tidak ditemukan, dilewati", name)
            continue

        log.info("")
        log.info("=== %s ===", name)
        result = export_to_onnx(weights, imgsz=args.imgsz, opset=args.opset)
        final = target_dir / filename
        shutil.move(result.onnx_path, final)
        log.info("  ONNX     : %s (%.2f MB)", final.name, result.size_mb)

        latency = benchmark_onnx(
            final,
            name=name,
            imgsz=args.imgsz,
            runs=args.runs,
            warmup=args.warmup,
            threads=threads,
        )
        log.info(
            "  latensi  : median %.1f ms | p95 %.1f ms | rentang %.1f-%.1f ms",
            latency.median_ms,
            latency.p95_ms,
            latency.min_ms,
            latency.max_ms,
        )
        exports.append(result.to_dict() | {"onnx_path": str(final)})
        latencies.append(latency.to_dict())

    if not exports:
        log.error("tidak ada model yang berhasil diekspor")
        return 1

    payload = {
        "generated_on": date.today().isoformat(),
        "imgsz": args.imgsz,
        "opset": args.opset,
        "runs": args.runs,
        "warmup": args.warmup,
        "intra_op_threads": threads,
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "exports": exports,
        "latency_cpu": latencies,
    }
    target = PROJECT_ROOT / "reports/metrics/latency_benchmark.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total = sum(item["median_ms"] for item in latencies)
    log.info("")
    log.info("Total median inferensi kedua model: %.1f ms", total)
    log.info("Hasil ditulis ke %s", target.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
