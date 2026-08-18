"""Pelatihan model anomali dari gambar normal saja.

Detektor cacat berbasis label hanya mengenali jenis cacat yang pernah
dilabeli. Model anomali dilatih **tanpa satu pun contoh cacat**: ia mempelajari
seperti apa produk normal, lalu menandai apa pun yang menyimpang. Karena itu
jenis cacat baru, yang belum pernah muncul di data latih, tetap terjaring.

Data dibagi tiga dan pembagian itu penting. Model belajar dari split latih,
ambang dihitung dari split kalibrasi yang tidak pernah ikut dilatih, lalu laju
alarm palsu diukur pada split uji yang terpisah dari keduanya. Tanpa pemisahan
ini, ambang akan dikalibrasi dan dinilai pada data yang sama, dan angka laju
alarm palsunya menjadi melingkar.

Pilihan arsitektur dibaca dari `configs/training.yaml`. EfficientAD menjadi
pilihan utama dengan PaDiM dan PatchCore sebagai cadangan, sehingga pergantian
model cukup mengubah config dan tidak menyentuh kode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_MODELS = ("efficient_ad", "padim", "patchcore")


@dataclass(frozen=True)
class AnomalyScores:
    """Skor anomali satu kategori, dipisah menurut peran splitnya."""

    category: str
    model_name: str
    #: Skor pada gambar normal yang tidak pernah ikut dilatih; dasar ambang.
    calibration_normal: list[float] = field(default_factory=list)
    #: Skor pada gambar normal split uji; dasar laju alarm palsu.
    test_normal: list[float] = field(default_factory=list)
    #: Skor pada gambar cacat split uji; dasar recall.
    test_defect: list[float] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "kalibrasi_normal": len(self.calibration_normal),
            "uji_normal": len(self.test_normal),
            "uji_cacat": len(self.test_defect),
        }

    def image_auroc(self) -> float:
        """AUROC tingkat gambar pada split uji, dihitung dari peringkat skor.

        Dihitung sendiri, bukan diambil dari metrik anomalib, supaya angkanya
        berasal dari skor yang sama persis dengan yang dipakai menurunkan ambang.
        """
        if not self.test_normal or not self.test_defect:
            return float("nan")
        normal = np.asarray(self.test_normal, dtype=float)
        defect = np.asarray(self.test_defect, dtype=float)
        ranks = np.argsort(np.argsort(np.concatenate([normal, defect]))) + 1
        n_pos, n_neg = defect.size, normal.size
        rank_sum = float(ranks[n_neg:].sum())
        return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def build_model(config: dict[str, Any], project_root: Path) -> Any:
    """Bangun model anomali sesuai config.

    Post-processor bawaan anomalib sengaja dimatikan. Post-processor itu
    menormalisasi skor terhadap ambang internalnya sendiri dan pada praktiknya
    mengkuantisasi keluaran menjadi segelintir nilai saja, sehingga ekor
    sebarannya hilang dan tidak ada lagi yang dapat dimodelkan dengan GPD.
    Skor mentah dibutuhkan justru karena ambangnya akan dihitung sendiri.

    Melempar :class:`ValueError` untuk nama model yang tidak dikenal, karena
    salah ketik pada config lebih baik ketahuan sekarang daripada setelah
    berjam-jam training.
    """
    name = str(config.get("model", "padim")).lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"model {name!r} tidak dikenal; pilihan: {SUPPORTED_MODELS}")

    if name == "efficient_ad":
        from anomalib.models import EfficientAd

        return EfficientAd(
            model_size=str(config.get("variant", "small")),
            imagenet_dir=str(project_root / "data" / "imagenette"),
            post_processor=False,
        )
    if name == "patchcore":
        from anomalib.models import Patchcore

        return Patchcore(post_processor=False)

    from anomalib.models import Padim

    return Padim(post_processor=False)


def build_datamodule(
    category_root: Path,
    category: str,
    *,
    batch_size: int,
    calibration_ratio: float = 0.2,
) -> Any:
    """Susun datamodule dari tata letak bergaya MVTec hasil Step 2.

    Split validasi sengaja diambil dari **train**, bukan dari test. Bawaan
    anomalib memotong separuh test menjadi validasi, dan itu akan mengecilkan
    split uji sekaligus membuat ambang dikalibrasi pada data uji.

    Worker dataloader dipaksa nol. Di Windows setiap worker adalah proses
    Python terpisah yang memuat ulang torch dan menghabiskan ratusan megabyte,
    padahal pemuatan data bukan hambatan pada model ini. Pada mesin dengan RAM
    terbatas, biaya itu justru yang menghentikan proses.
    """
    from anomalib.data import Folder

    return Folder(
        name=category,
        root=category_root,
        normal_dir="train/good",
        abnormal_dir="test/defect",
        normal_test_dir="test/good",
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=0,
        val_split_mode="from_train",
        val_split_ratio=calibration_ratio,
    )


def cap_training_images(datamodule: Any, *, limit: int, seed: int) -> int:
    """Batasi jumlah gambar latih dan kembalikan jumlah yang akhirnya dipakai.

    Kebutuhan memori PaDiM tumbuh linier terhadap jumlah gambar latih karena
    seluruh embedding ditahan sebelum Gaussian dicocokkan. Tanpa batas ini,
    kategori dengan 428 gambar menghabiskan memori mesin dan proses terhenti.

    Pengambilan sampel memakai seed sehingga subset yang terpilih sama pada
    setiap pengulangan.
    """
    samples = datamodule.train_data.samples
    if limit <= 0 or len(samples) <= limit:
        return len(samples)
    datamodule.train_data.samples = (
        samples.sample(n=limit, random_state=seed).sort_index().reset_index(drop=True)
    )
    return limit


def split_scores(predictions: list[Any]) -> tuple[list[float], list[float]]:
    """Pisahkan skor sebuah dataloader menjadi kelompok normal dan cacat."""
    normal: list[float] = []
    defect: list[float] = []
    for batch in predictions:
        scores = np.atleast_1d(np.asarray(batch.pred_score.cpu()).ravel())
        labels = np.atleast_1d(np.asarray(batch.gt_label.cpu()).ravel())
        for score, label in zip(scores, labels, strict=True):
            (defect if int(label) == 1 else normal).append(float(score))
    return normal, defect
