"""Mesin keputusan tiga kelas: PASS, REJECT, dan REVIEW.

Bagian ini menyatukan seluruh hasil langkah sebelumnya menjadi satu keputusan.
Yang membedakannya dari rangkaian pernyataan bersyarat biasa adalah asal-usul
setiap ambang yang dipakai:

* Peluang cacat berasal dari skor deteksi yang **sudah dikalibrasi** Platt,
  sehingga angkanya benar-benar dapat dibaca sebagai peluang.
* Kelas REVIEW ditentukan **himpunan prediksi conformal**, bukan ambang
  keyakinan pilihan tangan. Sistem menahan keputusan tepat ketika dua label
  masih sama-sama masuk akal pada tingkat jaminan yang dipilih.
* Ambang anomali berasal dari pemodelan **ekor sebaran** dengan teori nilai
  ekstrem, menargetkan laju alarm palsu yang ditetapkan.
* Ambang luas cacat dan bobot keparahan dibaca dari config.

Seluruh nilai bersifat statis: dibaca sekali saat startup dan tidak pernah
berubah saat sistem berjalan. Tidak ada penyetelan otomatis di sini.

Urutan aturan disengaja. Ketidakpastian diperiksa lebih dulu, karena sistem
yang ragu tidak boleh terlanjur memberi keputusan tegas. Sesudah itu barulah
bukti cacat dinilai, dan anomali tak dikenal diperiksa paling akhir sebagai
jaring pengaman untuk cacat yang belum pernah dilabeli.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from visionqc_ai.statistics.calibration import PlattParameters, apply_platt
from visionqc_ai.statistics.conformal import (
    DEFECT,
    NORMAL,
    ConformalQuantiles,
    prediction_set,
)

VerdictLabel = Literal["PASS", "REJECT", "REVIEW"]

PASS = "PASS"
REJECT = "REJECT"
REVIEW = "REVIEW"


@dataclass(frozen=True)
class DetectedDefect:
    """Satu cacat yang terdeteksi pada gambar."""

    class_name: str
    confidence: float
    area_pct: float = 0.0


@dataclass(frozen=True)
class DecisionConfig:
    """Seluruh ambang keputusan, dibaca sekali dari configs/inference.yaml."""

    area_pct_threshold: float
    anomaly_threshold: float
    min_confidence: float
    operating_threshold: float
    platt: PlattParameters
    conformal: ConformalQuantiles
    #: "binary" tidak pernah mengembalikan REVIEW; "three_class" boleh menahan.
    mode: str = "binary"
    #: Ambang keyakinan pemisah PASS dan REJECT pada mode biner. Dipilih pada
    #: set kalibrasi dengan meminimumkan biaya, bukan dari hasil set uji.
    binary_threshold: float = 0.22
    severity_weights: dict[str, float] = field(default_factory=dict)
    #: Kelas yang langsung ditolak berapa pun luasnya, karena berdampak pada
    #: keamanan konsumsi dan bukan sekadar cacat penampilan.
    critical_classes: tuple[str, ...] = ("kotor",)

    @classmethod
    def from_yaml(cls, config: dict[str, Any]) -> DecisionConfig:
        """Susun dari isi configs/inference.yaml yang sudah dimuat."""
        decision = config["decision"]
        calibration = decision["calibration"]
        conformal = decision["conformal"]
        quantiles = conformal["quantiles"]
        return cls(
            area_pct_threshold=float(decision["area_pct_threshold"]),
            anomaly_threshold=float(decision["anomaly_threshold"]),
            min_confidence=float(decision["min_confidence"]),
            operating_threshold=float(decision["cost"]["operating_threshold"]),
            mode=str(decision.get("mode", "binary")),
            binary_threshold=float(decision.get("binary_threshold", 0.22)),
            platt=PlattParameters(
                slope=float(calibration["platt_slope"]),
                intercept=float(calibration["platt_intercept"]),
            ),
            conformal=ConformalQuantiles(
                alpha=float(conformal["alpha"]),
                mode=str(conformal["mode"]),
                quantiles={
                    NORMAL: float(quantiles["normal"]),
                    DEFECT: float(quantiles["defect"]),
                },
                calibration_counts={},
            ),
            severity_weights={
                str(c["name"]): float(c.get("severity_weight", 0.5))
                for c in config.get("classes", [])
            },
        )


@dataclass(frozen=True)
class Verdict:
    """Keputusan akhir beserta alasan dan seluruh angka pendukungnya."""

    label: VerdictLabel
    reason: str
    calibrated_probability: float
    prediction_set: tuple[str, ...]
    severity: float
    defect_area_pct: float
    anomaly_score: float
    top_defect: str | None
    top_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _severity(
    defects: Sequence[DetectedDefect], weights: dict[str, float]
) -> tuple[float, DetectedDefect | None]:
    """Keparahan tertinggi di antara cacat yang terdeteksi.

    Keparahan menimbang keyakinan dengan bobot kelas, sehingga kontaminasi yang
    terdeteksi lemah tetap dinilai lebih berat daripada goresan yang terdeteksi
    kuat. Untuk produk pangan, urutan itulah yang benar.
    """
    if not defects:
        return 0.0, None
    scored = [(d.confidence * weights.get(d.class_name, 0.5), d) for d in defects]
    best_score, best = max(scored, key=lambda pair: pair[0])
    return float(best_score), best


def decide(
    defects: Sequence[DetectedDefect],
    *,
    defect_area_pct: float,
    anomaly_score: float,
    config: DecisionConfig,
) -> Verdict:
    """Tentukan PASS, REJECT, atau REVIEW dari keluaran seluruh model.

    Mengembalikan :class:`Verdict` yang memuat alasan dalam bahasa manusia
    beserta angka yang mendasarinya, supaya keputusan dapat ditelusuri dan
    tidak tampil sebagai kotak hitam.
    """
    raw_confidence = max((d.confidence for d in defects), default=0.0)
    probability = apply_platt([raw_confidence], config.platt)[0]
    labels = prediction_set(probability, config.conformal)
    severity, top = _severity(defects, config.severity_weights)
    names = tuple("cacat" if label == DEFECT else "normal" for label in sorted(labels))

    def build(label: VerdictLabel, reason: str) -> Verdict:
        return Verdict(
            label=label,
            reason=reason,
            calibrated_probability=float(probability),
            prediction_set=names,
            severity=severity,
            defect_area_pct=float(defect_area_pct),
            anomaly_score=float(anomaly_score),
            top_defect=top.class_name if top else None,
            top_confidence=float(raw_confidence),
        )

    if config.mode == "binary":
        return _decide_binary(
            defects, raw_confidence, defect_area_pct, anomaly_score, config, build
        )

    if len(labels) != 1:
        reason = (
            "sistem tidak dapat memastikan kondisi produk pada tingkat jaminan "
            f"{(1 - config.conformal.alpha):.0%}"
            if labels
            else "produk tidak menyerupai contoh mana pun yang pernah dipelajari"
        )
        return build(REVIEW, reason)

    if labels == {DEFECT}:
        critical = next(
            (d for d in defects if d.class_name in config.critical_classes), None
        )
        if critical is not None:
            return build(
                REJECT,
                f"terdeteksi {critical.class_name}, yang ditolak berapa pun luasnya "
                "karena berdampak pada keamanan konsumsi",
            )
        if defect_area_pct > config.area_pct_threshold:
            return build(
                REJECT,
                f"luas cacat {defect_area_pct:.2f} persen melampaui batas "
                f"{config.area_pct_threshold:.2f} persen",
            )
        if raw_confidence < config.min_confidence:
            return build(
                REVIEW,
                f"cacat terdeteksi tetapi keyakinannya {raw_confidence:.2f}, "
                f"di bawah batas {config.min_confidence:.2f}",
            )
        return build(
            REJECT,
            f"cacat {top.class_name if top else 'tidak dikenal'} terdeteksi "
            "dengan keyakinan memadai",
        )

    if anomaly_score > config.anomaly_threshold:
        return build(
            REVIEW,
            f"tidak ada cacat yang dikenali, tetapi skor anomali "
            f"{anomaly_score:.2f} melampaui ambang {config.anomaly_threshold:.2f}; "
            "kemungkinan cacat jenis baru",
        )

    return build(PASS, "tidak ditemukan cacat maupun penyimpangan dari normal")


def _decide_binary(
    defects: Sequence[DetectedDefect],
    raw_confidence: float,
    defect_area_pct: float,
    anomaly_score: float,
    config: DecisionConfig,
    build: Any,
) -> Verdict:
    """Putuskan PASS atau REJECT tanpa pernah menahan keputusan.

    Ambang keyakinan di sini menggantikan peran ``min_confidence`` pada mode
    tiga kelas. Pada mode tiga kelas, cacat yang terdeteksi lemah diserahkan ke
    manusia; di sini cacat itu ditolak, karena pada produk pangan melewatkan
    cacat jauh lebih mahal daripada menolak produk yang sebenarnya baik.
    Perbandingan biayanya bukan pendapat: 50.000 berbanding 2.000 rupiah pada
    model biaya di config.

    Himpunan prediksi conformal tetap dihitung dan tetap dilaporkan supaya
    keputusan dapat ditelusuri, tetapi tidak lagi dipakai menahan keputusan.
    """
    critical = next(
        (d for d in defects if d.class_name in config.critical_classes), None
    )
    if critical is not None:
        return build(
            REJECT,
            f"terdeteksi {critical.class_name}, yang ditolak berapa pun luasnya "
            "karena berdampak pada keamanan konsumsi",
        )

    if defect_area_pct > config.area_pct_threshold:
        return build(
            REJECT,
            f"luas cacat {defect_area_pct:.2f} persen melampaui batas "
            f"{config.area_pct_threshold:.2f} persen",
        )

    if raw_confidence >= config.binary_threshold:
        return build(
            REJECT,
            f"cacat terdeteksi dengan keyakinan {raw_confidence:.2f}, "
            f"melampaui ambang keputusan {config.binary_threshold:.2f}",
        )

    if anomaly_score > config.anomaly_threshold:
        return build(
            REJECT,
            f"tidak ada cacat yang dikenali, tetapi skor anomali "
            f"{anomaly_score:.2f} melampaui ambang {config.anomaly_threshold:.2f}; "
            "produk menyimpang dari contoh normal",
        )

    return build(PASS, "tidak ditemukan cacat maupun penyimpangan dari normal")
