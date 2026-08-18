"""Kontrak data antara modul AI dan Backend.

Bentuk di sini mengikuti kontrak yang disepakati pada PROJECT.md bagian 12.5.
Menambah field opsional bersifat aman; mengganti nama, menghapus, atau mengubah
tipe adalah perubahan yang merusak dan harus dibicarakan lebih dulu karena
Backend dan Frontend membangun di atasnya.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VerdictLabel = Literal["PASS", "REJECT", "REVIEW"]


class BBox(BaseModel):
    """Kotak pembatas dalam piksel pada gambar asli."""

    x: int = Field(ge=0, description="tepi kiri")
    y: int = Field(ge=0, description="tepi atas")
    w: int = Field(ge=0, description="lebar")
    h: int = Field(ge=0, description="tinggi")


class Defect(BaseModel):
    """Satu cacat yang terdeteksi."""

    type: str = Field(description="nama kelas cacat")
    label: str = Field(description="nama kelas untuk ditampilkan ke pengguna")
    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)
    area_pct: float | None = Field(
        default=None, ge=0.0, description="luas cacat terhadap luas gambar, persen"
    )


class AnomalyResult(BaseModel):
    """Skor anomali beserta ambang yang dipakai membandingkannya."""

    score: float
    threshold: float
    exceeded: bool = Field(description="benar bila skor melampaui ambang")
    heatmap_base64: str | None = None


class DecisionDetail(BaseModel):
    """Angka yang mendasari keputusan, supaya hasilnya dapat ditelusuri."""

    calibrated_probability: float = Field(ge=0.0, le=1.0)
    prediction_set: list[str] = Field(
        description="himpunan prediksi conformal; lebih dari satu berarti sistem ragu"
    )
    severity: float = Field(ge=0.0)
    conformal_alpha: float


class InspectionResult(BaseModel):
    """Hasil lengkap satu kali inspeksi. Ini yang dikembalikan ke Backend."""

    # Pydantic mencadangkan awalan "model_" untuk keperluannya sendiri. Nama
    # model_version sudah menjadi bagian kontrak dengan Backend, sehingga
    # pencadangan itu dinonaktifkan alih-alih mengganti nama medan.
    model_config = ConfigDict(protected_namespaces=())

    verdict: VerdictLabel
    reason: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    batch_code: str | None = None
    defects: list[Defect] = Field(default_factory=list)
    defect_area_pct: float = Field(default=0.0, ge=0.0)
    anomaly: AnomalyResult | None = None
    decision: DecisionDetail | None = None
    annotated_image_base64: str = ""
    model_version: str = ""
    latency_ms: int = Field(default=0, ge=0)
