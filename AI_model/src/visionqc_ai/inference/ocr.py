"""Pembacaan kode batch dari gambar produk.

Kode batch adalah satu-satunya teks yang berguna bagi sistem ini. Ia menjawab
pertanyaan yang tidak bisa dijawab model cacat: bila satu produk ditolak,
kumpulan produksi mana yang perlu ditelusuri.

Mesinnya PaddleOCR yang dipakai apa adanya tanpa pelatihan ulang. Pilihan itu
disengaja dan tercatat: OCR di sini komponen pelengkap, bukan model inovasi
inti, sehingga kewajiban fine-tuning tidak berlaku padanya. Yang wajib dilatih
sendiri adalah detektor cacat.

Keluaran OCR **tidak pernah** dikembalikan mentah. PaddleOCR membaca apa pun
yang terlihat, termasuk nama pada seragam operator dan tulisan yang kebetulan
masuk bingkai. Setiap hasil disaring lebih dulu dengan daftar-izin di
:mod:`visionqc_ai.privacy.ocr_filter`, sehingga hanya teks berbentuk kode batch
yang lolos dan sisanya dibuang sebelum meninggalkan modul ini.

Bila PaddleOCR tidak terpasang, keadaannya dinyatakan lewat ``available``
alih-alih diam-diam mengembalikan kode kosong. Kode batch yang kosong karena
pustaka tidak ada dan kode batch yang kosong karena memang tidak terbaca adalah
dua hal berbeda, dan hanya salah satunya perlu diperbaiki.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from visionqc_ai.privacy.ocr_filter import filter_text

log = logging.getLogger(__name__)

# Di bawah panjang ini teks nyaris selalu potongan kata, bukan kode batch.
# Pola daftar-izin sudah menyaring bentuknya; ambang ini hanya memangkas
# pekerjaan pencocokan pada derau OCR yang jelas tidak relevan.
MIN_TEXT_LENGTH = 4

# Keyakinan minimum per kotak teks. PaddleOCR mengembalikan banyak kotak
# berkeyakinan rendah pada permukaan kemasan yang mengilap; menyaringnya di
# sini mencegah kode batch palsu yang kebetulan lolos pola.
MIN_TEXT_CONFIDENCE = 0.60


@dataclass(frozen=True)
class BatchCodeResult:
    """Kode batch yang terbaca beserta jejak penyaringan privasinya."""

    batch_code: str | None
    available: bool
    candidates: list[str] = field(default_factory=list)
    discarded: int = 0
    note: str = ""


class BatchCodeReader:
    """Memuat mesin OCR sekali lalu membaca banyak gambar.

    Pemuatan ditunda sampai pembacaan pertama. PaddleOCR menarik paddlepaddle
    dan bobotnya sendiri, yang memakan beberapa detik dan ratusan megabita;
    menanggung biaya itu saat impor akan memperlambat setiap proses yang
    sekadar mengimpor paket ini, termasuk rangkaian pengujian.
    """

    def __init__(
        self,
        *,
        lang: str = "en",
        use_gpu: bool = False,
        patterns: tuple[str, ...] = (),
        allowlist_only: bool = True,
    ) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self.patterns = tuple(patterns)
        self.allowlist_only = allowlist_only
        self._engine: Any | None = None
        self._loaded = False
        self._note = ""
        # PaddleOCR menyimpan keadaan di dalam objeknya dan tidak dirancang
        # untuk dipanggil dari beberapa thread sekaligus. Backend melayani
        # permintaan di threadpool, jadi pembacaan diserialkan di sini.
        self._lock = threading.Lock()

    def _load(self) -> Any | None:
        """Muat mesin OCR sekali; ``None`` bila pustakanya tidak tersedia."""
        if self._loaded:
            return self._engine
        self._loaded = True
        try:
            from paddleocr import PaddleOCR
        except Exception as error:
            # Sengaja menangkap Exception, bukan ImportError saja. Mengimpor
            # paddleocr menjalankan impor paddlepaddle beserta modul protobuf
            # dan numpy-nya, dan ketidakcocokan versi di sana muncul sebagai
            # TypeError atau AttributeError, bukan ImportError. Menangkap
            # ImportError saja membuat kegagalan itu menjatuhkan seluruh
            # inspeksi, padahal kode batch hanya pelengkap.
            self._note = f"mesin OCR tidak tersedia: {type(error).__name__}: {error}"
            log.warning(self._note)
            return None
        try:
            self._engine = PaddleOCR(
                use_angle_cls=True, lang=self.lang, use_gpu=self.use_gpu, show_log=False
            )
        except Exception as error:  # pragma: no cover - bergantung lingkungan
            # Kegagalan di sini berarti unduhan bobot atau backend paddle
            # bermasalah. Itu tidak boleh menjatuhkan seluruh inspeksi, karena
            # kode batch hanya pelengkap dari keputusan cacat.
            self._note = f"mesin OCR gagal dimuat: {error}"
            log.warning(self._note)
            self._engine = None
        return self._engine

    @property
    def available(self) -> bool:
        """Benar bila mesin OCR siap dipakai."""
        return self._load() is not None

    def _extract(self, raw: Any) -> list[str]:
        """Ambil teks berkeyakinan cukup dari keluaran PaddleOCR.

        Bentuk keluarannya bersarang dan pernah berubah antar versi, jadi
        pembacaannya dibuat bertahan terhadap baris yang tidak berbentuk
        seperti yang diharapkan alih-alih mengandalkan satu bentuk persis.
        """
        texts: list[str] = []
        for page in raw or []:
            for line in page or []:
                try:
                    text, confidence = line[1][0], float(line[1][1])
                except (IndexError, TypeError, ValueError):
                    continue
                if confidence >= MIN_TEXT_CONFIDENCE and len(text.strip()) >= (
                    MIN_TEXT_LENGTH
                ):
                    texts.append(text.strip())
        return texts

    def read(self, image: np.ndarray) -> BatchCodeResult:
        """Baca kode batch dari satu gambar BGR.

        Kode batch yang dikembalikan adalah kandidat **terpanjang** yang lolos
        daftar-izin. Kemasan sering memuat lebih dari satu teks berbentuk kode,
        misalnya tanggal kedaluwarsa di samping nomor lot, dan yang terpanjang
        adalah yang paling mungkin membawa nomor lot lengkap.
        """
        engine = self._load()
        if engine is None:
            return BatchCodeResult(batch_code=None, available=False, note=self._note)

        with self._lock:
            try:
                raw = engine.ocr(image, cls=True)
            except Exception as error:  # pragma: no cover - bergantung lingkungan
                note = f"pembacaan OCR gagal: {error}"
                log.warning(note)
                return BatchCodeResult(batch_code=None, available=True, note=note)

        texts = self._extract(raw)
        if self.allowlist_only:
            kept, discarded = filter_text(texts, self.patterns)
        else:
            # Jalur ini hanya sah untuk penelusuran saat pengembangan. Pada
            # penyajian, privacy.ocr_allowlist_only wajib menyala.
            kept, discarded = texts, 0

        batch_code = max(kept, key=len) if kept else None
        return BatchCodeResult(
            batch_code=batch_code,
            available=True,
            candidates=kept,
            discarded=discarded,
        )
