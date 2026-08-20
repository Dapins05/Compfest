"""Uji mesin keputusan.

Mesin keputusan diuji terpisah dari model karena ia murni logika: keluaran
model apa pun dapat disimulasikan, dan justru kasus batas yang jarang muncul
pada data nyata yang paling perlu dijaga perilakunya.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from visionqc_ai.inference.decision import (
    PASS,
    REJECT,
    REVIEW,
    DecisionConfig,
    DetectedDefect,
    decide,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "inference.yaml"


@pytest.fixture(scope="module")
def config() -> DecisionConfig:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return DecisionConfig.from_yaml(yaml.safe_load(handle))


@pytest.fixture(scope="module")
def three_class(config: DecisionConfig) -> DecisionConfig:
    """Konfigurasi mode tiga kelas, yang masih boleh menahan keputusan.

    Mode ini tidak dipakai saat penyajian tetapi tetap dijaga perilakunya,
    karena tahap Final berencana memakainya kembali.
    """
    return replace(config, mode="three_class")


def run(config: DecisionConfig, defects, area=0.0, anomaly=0.0):
    return decide(defects, defect_area_pct=area, anomaly_score=anomaly, config=config)


def test_kontaminasi_ditolak_walau_luasnya_kecil(config: DecisionConfig) -> None:
    """Kontaminasi menyangkut keamanan konsumsi, bukan penampilan."""
    verdict = run(config, [DetectedDefect("kotor", 0.7)], area=0.1)
    assert verdict.label == REJECT
    assert "keamanan konsumsi" in verdict.reason


def test_luas_melampaui_ambang_ditolak(config: DecisionConfig) -> None:
    verdict = run(config, [DetectedDefect("pecah", 0.9)], area=5.0)
    assert verdict.label == REJECT
    assert "melampaui batas" in verdict.reason


def test_mode_bawaan_adalah_biner(config: DecisionConfig) -> None:
    """Config yang dikirim ke Backend tidak boleh menahan keputusan."""
    assert config.mode == "binary"


def test_keyakinan_rendah_diserahkan_ke_manusia(three_class: DecisionConfig) -> None:
    """Pada mode tiga kelas, model yang tidak yakin tidak memutuskan sendiri."""
    verdict = run(three_class, [DetectedDefect("noda", 0.30)], area=0.2)
    assert verdict.label == REVIEW


def test_biner_menolak_cacat_yang_terdeteksi_lemah(config: DecisionConfig) -> None:
    """Cacat lemah ditolak, bukan diserahkan ke manusia.

    Pada produk pangan, melewatkan cacat jauh lebih mahal daripada menolak
    produk yang sebenarnya baik.
    """
    verdict = run(config, [DetectedDefect("noda", 0.30)], area=0.2)
    assert verdict.label == REJECT


def test_biner_meloloskan_di_bawah_ambang_keputusan(config: DecisionConfig) -> None:
    weak = config.binary_threshold - 0.05
    verdict = run(config, [DetectedDefect("gores", weak)], area=0.1)
    assert verdict.label == PASS


def test_anomali_tinggi_tanpa_cacat_dikenali(three_class: DecisionConfig) -> None:
    """Jaring pengaman untuk cacat jenis baru yang belum pernah dilabeli."""
    high = three_class.anomaly_threshold + 10.0
    verdict = decide([], defect_area_pct=0.0, anomaly_score=high, config=three_class)
    assert verdict.label == REVIEW
    assert verdict.anomaly_score == pytest.approx(high)


def test_biner_menolak_anomali_tinggi(config: DecisionConfig) -> None:
    high = config.anomaly_threshold + 10.0
    verdict = decide([], defect_area_pct=0.0, anomaly_score=high, config=config)
    assert verdict.label == REJECT


def test_mode_biner_tidak_pernah_menahan_keputusan(config: DecisionConfig) -> None:
    """Tidak ada kombinasi masukan yang menghasilkan REVIEW pada mode biner."""
    grid = [
        run(config, defects, area=area, anomaly=anomaly)
        for defects in (
            [],
            [DetectedDefect("gores", 0.05)],
            [DetectedDefect("noda", 0.30)],
            [DetectedDefect("kotor", 0.10)],
            [DetectedDefect("pecah", 0.99)],
            [DetectedDefect("deformasi", 0.21), DetectedDefect("gores", 0.23)],
        )
        for area in (0.0, 0.5, 2.0, 25.0)
        for anomaly in (0.0, 39.0, 40.0, 500.0)
    ]
    assert grid
    assert all(v.label in (PASS, REJECT) for v in grid)


def test_ambang_bersifat_statis(config: DecisionConfig) -> None:
    """Memanggil berkali-kali tidak boleh menggeser ambang mana pun."""
    before = (config.area_pct_threshold, config.anomaly_threshold)
    for _ in range(20):
        run(config, [DetectedDefect("gores", 0.8)], area=1.0)
    assert (config.area_pct_threshold, config.anomaly_threshold) == before


def test_keparahan_menimbang_bobot_kelas(config: DecisionConfig) -> None:
    """Kontaminasi lemah harus dinilai lebih berat daripada goresan kuat."""
    kotor = run(config, [DetectedDefect("kotor", 0.5)], area=0.1)
    gores = run(config, [DetectedDefect("gores", 0.8)], area=0.1)
    assert kotor.severity > gores.severity


def test_hasil_memuat_angka_pendukung(config: DecisionConfig) -> None:
    """Keputusan harus dapat ditelusuri, bukan tampil sebagai kotak hitam."""
    verdict = run(config, [DetectedDefect("pecah", 0.85)], area=3.0)
    payload = verdict.to_dict()
    for key in (
        "calibrated_probability",
        "prediction_set",
        "severity",
        "defect_area_pct",
        "anomaly_score",
    ):
        assert key in payload


def test_gambar_bersih_menghasilkan_pass(config: DecisionConfig) -> None:
    """Gambar tanpa deteksi dan tanpa anomali harus diloloskan.

    Uji ini sebelumnya mengunci perilaku yang salah: gambar bersih menghasilkan
    REVIEW karena himpunan conformal memuat kedua label. Penyebabnya set
    kalibrasi hanya berisi tujuh gambar normal. Setelah set itu diperluas
    dengan gambar normal yang tidak pernah dilihat model, perilakunya benar.
    Sistem QC yang tidak pernah dapat meloloskan produk tidak berguna.
    """
    verdict = run(config, [], area=0.0, anomaly=0.0)
    assert verdict.label == PASS
    assert verdict.prediction_set == ("normal",)
