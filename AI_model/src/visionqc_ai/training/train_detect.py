"""Fine-tuning YOLO11 untuk deteksi cacat kemasan.

Hyperparameter dibaca dari configs/training.yaml supaya nilai yang dipakai
tercatat di satu tempat dan dapat direproduksi. Seed dikunci dan test set tidak
pernah disentuh selama training; Ultralytics hanya melihat split train dan val.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingOutcome:
    """Ringkasan satu run training."""

    weights_best: Path
    weights_last: Path
    run_dir: Path
    epochs_requested: int
    epochs_completed: int
    results_csv: Path | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights_best": str(self.weights_best),
            "weights_last": str(self.weights_last),
            "run_dir": str(self.run_dir),
            "epochs_requested": self.epochs_requested,
            "epochs_completed": self.epochs_completed,
            "results_csv": str(self.results_csv) if self.results_csv else None,
        }


def load_training_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_train_kwargs(
    config: dict[str, Any], section: str, *, data_yaml: Path, project: Path, name: str
) -> dict[str, Any]:
    """Susun argumen Ultralytics dari configs/training.yaml.

    Hanya kunci yang benar-benar dikenal Ultralytics yang diteruskan; sisanya
    seperti nama model dan catatan internal ditangani terpisah.
    """
    block = dict(config[section])
    hardware = config.get("hardware", {})
    block.pop("model", None)

    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "seed": config.get("seed", 0),
        "deterministic": config.get("deterministic", True),
        "device": hardware.get("device", 0),
        "amp": hardware.get("amp", True),
        "workers": hardware.get("workers", 4),
        "cache": hardware.get("cache", False),
        "val": True,
        "plots": True,
        "verbose": True,
    }
    kwargs.update(block)
    return kwargs


def count_completed_epochs(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    with results_csv.open(encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def fine_tune(
    *,
    config_path: Path,
    data_yaml: Path,
    project: Path,
    name: str,
    section: str = "detection",
    epochs_override: int | None = None,
) -> TrainingOutcome:
    """Jalankan fine-tuning dan kembalikan lokasi bobot hasilnya."""
    from ultralytics import YOLO

    config = load_training_config(config_path)
    base_model = config[section]["model"]
    kwargs = build_train_kwargs(
        config, section, data_yaml=data_yaml, project=project, name=name
    )
    if epochs_override is not None:
        kwargs["epochs"] = epochs_override

    model = YOLO(base_model)
    model.train(**kwargs)

    run_dir = project / name
    results_csv = run_dir / "results.csv"
    return TrainingOutcome(
        weights_best=run_dir / "weights" / "best.pt",
        weights_last=run_dir / "weights" / "last.pt",
        run_dir=run_dir,
        epochs_requested=int(kwargs["epochs"]),
        epochs_completed=count_completed_epochs(results_csv),
        results_csv=results_csv if results_csv.exists() else None,
    )
