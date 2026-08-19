"""Ekspor model anomali ke ONNX dan verifikasi kesetaraannya dengan PyTorch.

    python scripts/export_anomaly.py --category combined

Ekspor tidak dianggap berhasil hanya karena berkasnya terbentuk. Skor ONNX
dibandingkan langsung dengan skor PyTorch pada sejumlah gambar sungguhan,
karena model yang menyimpang sedikit saja membuat skornya tidak lagi sebanding
dengan ambang yang sudah ditetapkan di config.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from visionqc_ai.inference.anomaly import AnomalyScorer  # noqa: E402

log = logging.getLogger("export_anomaly")


def newest_checkpoint(category: str) -> Path:
    """Checkpoint terbaru untuk sebuah kategori."""
    base = PROJECT_ROOT / "models" / "finetuned" / "anomaly" / "Padim" / category
    checkpoints = sorted(
        base.glob("v*/weights/lightning/model.ckpt"), key=lambda p: p.stat().st_mtime
    )
    if not checkpoints:
        raise FileNotFoundError(f"tidak ada checkpoint untuk kategori {category}")
    return checkpoints[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ekspor model anomali ke ONNX")
    parser.add_argument("--category", default="combined")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    warnings.filterwarnings("ignore")

    import cv2
    import torch
    from anomalib.models import Padim

    checkpoint = newest_checkpoint(args.category)
    log.info("checkpoint : %s", checkpoint.relative_to(PROJECT_ROOT))

    model = Padim.load_from_checkpoint(
        str(checkpoint), post_processor=False, map_location="cpu"
    )
    model = model.cpu().eval()

    target = PROJECT_ROOT / "models" / "onnx" / f"padim-{args.category}.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.rand(1, 3, args.imgsz, args.imgsz)
    torch.onnx.export(
        model.model,
        dummy,
        str(target),
        opset_version=args.opset,
        input_names=["image"],
        output_names=["anomaly_map"],
        dynamo=False,
    )
    size_mb = target.stat().st_size / 1024 / 1024
    log.info("ONNX       : %s (%.1f MB)", target.name, size_mb)

    scorer = AnomalyScorer(target, imgsz=args.imgsz)
    root = PROJECT_ROOT / "data" / "processed" / "anomaly" / args.category
    images = (
        sorted((root / "test" / "good").glob("*.jpg"))[: args.samples]
        + sorted((root / "test" / "defect").glob("*.jpg"))[: args.samples]
    )

    differences: list[float] = []
    for path in images:
        image = cv2.imread(str(path))
        tensor = torch.from_numpy(scorer.preprocess(image))
        with torch.no_grad():
            reference = model.model(tensor)
        expected = float(np.asarray(reference.pred_score.cpu()).ravel()[0])
        differences.append(abs(expected - scorer.score(image).score))

    worst = max(differences) if differences else 0.0
    log.info("verifikasi : %d gambar, selisih maksimum %.6f", len(images), worst)
    if worst > 1e-3:
        log.error("ONNX menyimpang dari PyTorch; ekspor tidak dapat dipakai")
        return 1

    payload = {
        "generated_on": date.today().isoformat(),
        "category": args.category,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "onnx_path": str(target.relative_to(PROJECT_ROOT)),
        "size_mb": round(size_mb, 2),
        "imgsz": args.imgsz,
        "opset": args.opset,
        "max_abs_difference_vs_torch": worst,
        "verified_on_images": len(images),
    }
    report = PROJECT_ROOT / "reports/metrics/anomaly_export.json"
    report.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("hasil      : %s", report.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
