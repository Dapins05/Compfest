"""Pilih ambang keputusan biner pada set kalibrasi.

Mode biner tidak pernah menahan keputusan: setiap gambar dinyatakan PASS atau
REJECT. Yang menentukan pembagian itu satu ambang keyakinan, dan ambang itu
harus berasal dari suatu tempat yang dapat dipertanggungjawabkan.

Di sini ambang dipilih dengan meminimumkan biaya yang diharapkan pada model
biaya di `configs/inference.yaml`, dihitung **hanya pada set kalibrasi**. Set
uji dipakai sesudahnya semata-mata untuk melaporkan hasil. Memilih ambang dari
set uji akan membuat angka yang dilaporkan tidak lagi berarti apa-apa, karena
ambangnya sudah menyesuaikan diri pada jawaban yang seharusnya dirahasiakan.

Prevalensi cacat ikut ditimbang ulang. Tanpa itu, perhitungan biaya memakai
proporsi cacat pada set evaluasi yang jauh lebih tinggi daripada lini produksi
sungguhan, dan hasilnya condong menolak terlalu banyak produk.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import yaml  # noqa: E402

from calibrate_decision import gather_extra_normals  # noqa: E402
from visionqc_ai.evaluation.detection_eval import image_level_scores  # noqa: E402
from visionqc_ai.statistics.cost_sensitive import (  # noqa: E402
    CostModel,
    cost_curve,
    evaluate_threshold,
)

log = logging.getLogger("select_binary_threshold")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pemilihan ambang keputusan biner")
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "models/finetuned/detect/weights/best.pt",
    )
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "data/processed/detect"
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=PROJECT_ROOT / "configs/inference.yaml",
    )
    parser.add_argument("--extra-normals", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--points", type=int, default=201)
    parser.add_argument("--device", default=0)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if not args.weights.exists():
        log.error("bobot deteksi tidak ditemukan: %s", args.weights)
        return 1

    config = yaml.safe_load(args.inference_config.read_text(encoding="utf-8"))
    cost_config = config["decision"]["cost"]
    costs = CostModel(
        false_reject=float(cost_config["c_false_reject"]),
        missed_defect=float(cost_config["c_missed_defect"]),
        review=float(cost_config["c_review"]),
    )
    prevalence = float(cost_config["assumed_defect_prevalence"])

    extra_calibration, extra_test = gather_extra_normals(
        PROJECT_ROOT, args.dataset, per_side=args.extra_normals, seed=args.seed
    )
    common = {"dataset_root": args.dataset, "device": args.device}
    calibration = image_level_scores(
        args.weights, split="val", extra_normals=extra_calibration, **common
    )
    test = image_level_scores(
        args.weights, split="test", extra_normals=extra_test, **common
    )

    calibration_scores = [s.score for s in calibration]
    calibration_labels = [int(s.is_defect) for s in calibration]
    test_scores = [s.score for s in test]
    test_labels = [int(s.is_defect) for s in test]

    curve = cost_curve(
        calibration_scores,
        calibration_labels,
        costs,
        n_points=args.points,
        prevalence=prevalence,
    )
    chosen = min(curve, key=lambda point: (point.total_cost, point.threshold))

    log.info("kalibrasi n=%d cacat=%d", len(calibration), sum(calibration_labels))
    log.info("ambang terpilih pada kalibrasi: %.4f", chosen.threshold)
    log.info(
        "  recall %.4f  specificity %.4f  biaya per unit Rp%.1f",
        chosen.recall,
        chosen.specificity,
        chosen.cost_per_item,
    )

    on_test = evaluate_threshold(
        test_scores, test_labels, chosen.threshold, costs, prevalence=prevalence
    )
    previous = evaluate_threshold(
        test_scores,
        test_labels,
        float(config["decision"]["min_confidence"]),
        costs,
        prevalence=prevalence,
    )
    log.info("pada set uji, dilaporkan tanpa penyetelan lebih lanjut:")
    log.info(
        "  terpilih %.2f  recall %.4f  specificity %.4f  Rp%.1f",
        chosen.threshold,
        on_test.recall,
        on_test.specificity,
        on_test.cost_per_item,
    )
    log.info(
        "  ambang lama %.2f  recall %.4f  specificity %.4f  Rp%.1f",
        float(config["decision"]["min_confidence"]),
        previous.recall,
        previous.specificity,
        previous.cost_per_item,
    )

    target = PROJECT_ROOT / "reports/metrics/binary_threshold.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "selected_on": "calibration",
                "threshold": chosen.threshold,
                "prevalence": prevalence,
                "costs": costs.to_dict(),
                "calibration": chosen.to_dict(),
                "calibration_n": len(calibration),
                "calibration_defects": sum(calibration_labels),
                "test": on_test.to_dict(),
                "test_n": len(test),
                "test_defects": sum(test_labels),
                "previous_threshold": previous.to_dict(),
                "curve": [point.to_dict() for point in curve],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("hasil ditulis ke %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
