"""Uji kontrak data dengan Backend."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from visionqc_ai.schemas import BBox, Defect, InspectionResult


def test_hasil_minimal_sah() -> None:
    result = InspectionResult(verdict="PASS", reason="tidak ditemukan cacat")
    assert result.defects == []
    assert result.latency_ms == 0


def test_verdict_di_luar_tiga_kelas_ditolak() -> None:
    with pytest.raises(ValidationError):
        InspectionResult(verdict="MUNGKIN", reason="tidak sah")


def test_keyakinan_di_luar_rentang_ditolak() -> None:
    with pytest.raises(ValidationError):
        InspectionResult(verdict="PASS", reason="uji", confidence=1.5)


def test_defect_dapat_diserialisasi() -> None:
    defect = Defect(
        type="pecah",
        label="Pecah / Retak",
        bbox=BBox(x=10, y=20, w=30, h=40),
        confidence=0.9,
        area_pct=1.5,
    )
    payload = defect.model_dump()
    assert payload["bbox"]["w"] == 30
    assert payload["label"] == "Pecah / Retak"
