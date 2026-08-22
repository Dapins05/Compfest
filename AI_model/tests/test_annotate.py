"""Uji penggambaran anotasi.

Ditulis setelah penambahan kelas keenam membuat seluruh penggambaran gagal
dengan IndexError, karena daftar warna tertinggal di lima entri sementara
taksonomi sudah enam. Modelnya sendiri baik-baik saja; yang jatuh justru
lapisan yang menampilkan hasilnya.
"""

from __future__ import annotations

from visionqc_ai.data.taxonomy import CLASS_LABELS_ID, DEFECT_CLASSES
from visionqc_ai.inference.annotate import (
    CLASS_COLORS_BGR,
    FALLBACK_COLOR_BGR,
    class_color,
)


def test_setiap_kelas_punya_warna_sendiri() -> None:
    """Jumlah warna harus mengikuti taksonomi, bukan tertinggal di belakangnya."""
    assert len(CLASS_COLORS_BGR) == len(DEFECT_CLASSES)


def test_semua_kelas_dapat_digambar_tanpa_warna_cadangan() -> None:
    for name in DEFECT_CLASSES:
        assert class_color(name) != FALLBACK_COLOR_BGR, (
            f"kelas {name!r} jatuh ke warna cadangan; tambahkan warnanya"
        )


def test_kelas_tak_dikenal_tidak_menjatuhkan_penggambaran() -> None:
    """Nama kelas asing harus menghasilkan warna cadangan, bukan pengecualian."""
    assert class_color("kelas_yang_tidak_ada") == FALLBACK_COLOR_BGR


def test_setiap_kelas_punya_label_tampilan() -> None:
    for name in DEFECT_CLASSES:
        assert name in CLASS_LABELS_ID


def test_warna_tidak_kembar() -> None:
    """Warna kembar membuat dua kelas berbeda tampak sama pada hasil inspeksi."""
    assert len(set(CLASS_COLORS_BGR)) == len(CLASS_COLORS_BGR)
