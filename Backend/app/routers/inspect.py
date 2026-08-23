"""Endpoint inspeksi.

Sebelumnya endpoint ini mengembalikan hasil acak lewat `random.choice`, sebagai
tiruan yang disengaja supaya Frontend dapat dikerjakan lebih dulu. Sekarang
tiruan itu diganti pipeline AI yang sungguhan.

Pemrosesan berjalan SINKRON di dalam permintaan: satu gambar masuk, satu hasil
lengkap keluar, tanpa antrean maupun pekerjaan latar. Itu batasan ruang lingkup
penyisihan, sekaligus yang membuat `docker compose up` cukup menyalakan dua
proses tanpa infrastruktur lain.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from visionqc_ai import InspectionResult, InvalidImageError, run_inspection

from app.config import get_settings

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/inspect", response_model=InspectionResult)
async def inspect_image(
    file: Annotated[UploadFile, File()],
) -> InspectionResult:
    """Periksa satu gambar produk dan kembalikan keputusannya.

    Pemeriksaan berlapis dua, dan itu disengaja. Lapisan di sini menolak
    kiriman yang jelas keliru sebelum menyentuh model - tipe MIME salah, berkas
    kosong, ukuran berlebih - sehingga pengguna mendapat pesan yang menjelaskan
    persoalannya. Lapisan kedua ada di dalam modul AI, yang memeriksa isi
    berkasnya sendiri; header `Content-Type` dikirim oleh klien dan tidak dapat
    dipercaya sebagai bukti bahwa isinya benar-benar gambar.
    """
    settings = get_settings()

    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Tipe file {file.content_type} tidak didukung. "
                f"Gunakan {', '.join(settings.allowed_content_types)}."
            ),
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="File gambar kosong.")

    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Ukuran file {size_mb:.1f}MB melebihi batas "
                f"{settings.max_file_size_mb}MB."
            ),
        )

    try:
        return run_inspection(contents)
    except InvalidImageError as error:
        # Kesalahan pada berkas kiriman pengguna, bukan kegagalan sistem.
        # Modul AI sengaja memisahkan keduanya supaya Backend dapat membalas
        # 400 dan bukan 500; membalas 500 akan menyalahkan server atas berkas
        # yang memang tidak memenuhi syarat.
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        # Dilempar ketika bobot model belum tersedia. Ini kegagalan penyiapan
        # layanan, dan pesannya menyebut cara memperbaikinya alih-alih sekadar
        # menyatakan galat.
        log.error("pipeline tidak siap: %s", error)
        raise HTTPException(
            status_code=503,
            detail=(
                "Model belum siap melayani. Jalankan "
                "AI_model/scripts/download_models.py lebih dulu."
            ),
        ) from error
