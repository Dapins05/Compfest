"""Skema API.

Kontrak inspeksi TIDAK didefinisikan ulang di sini. Sumber kebenarannya
`visionqc_ai.schemas`, sebagaimana ditetapkan di PROJECT.md bagian 9.1, dan
modul ini hanya mengekspornya kembali supaya router cukup mengimpor dari satu
tempat.

Alasannya bukan kerapian belaka. Salinan yang ada sebelumnya sudah menyimpang
dari kontrak yang sesungguhnya dikembalikan modul AI:

* `Defect` kehilangan `label`, yaitu nama kelas berbahasa Indonesia yang
  ditampilkan Frontend ke pengguna
* `InspectionResult` kehilangan `defect_area_pct` dan `decision`, sehingga
  keputusan tidak dapat ditelusuri dari respons
* `verdict` masih memuat `REVIEW`, padahal mesin keputusan sudah berjalan pada
  mode biner dan tidak pernah mengembalikannya

Selisih seperti itu tidak memunculkan galat apa pun; ia hanya membuat respons
kehilangan medan diam-diam. Dengan mengimpor langsung, ketidakcocokan muncul
sebagai galat impor pada saat startup.
"""

from pydantic import BaseModel, ConfigDict
from visionqc_ai.schemas import (
    AnomalyResult,
    BBox,
    DecisionDetail,
    Defect,
    InspectionResult,
    VerdictLabel,
)

__all__ = [
    "AnomalyResult",
    "BBox",
    "DecisionDetail",
    "Defect",
    "InspectionResult",
    "ModelInfo",
    "SampleImage",
    "VerdictLabel",
]


class SampleImage(BaseModel):
    """Gambar contoh yang disediakan layanan untuk mencoba sistem."""

    name: str
    url: str


class ModelInfo(BaseModel):
    """Keterangan model yang sedang dilayani.

    `components` melaporkan lapisan mana yang benar-benar aktif. Sistem tetap
    berjalan ketika sebuah model tidak tersedia, dengan kemampuan berkurang,
    dan keadaan itu harus dapat dilihat dari luar alih-alih hanya tampak
    sebagai hasil yang diam-diam berbeda.
    """

    # `model_name` bertabrakan dengan ruang nama `model_` milik Pydantic dan
    # memunculkan peringatan pada setiap impor. Medannya sudah dipakai kontrak
    # yang berjalan, jadi peringatannya dimatikan alih-alih medannya diganti.
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    version: str
    dataset: str
    trained_at: str | None = None
    metrics: dict[str, float] = {}
    components: dict[str, bool] = {}
