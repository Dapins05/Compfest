# Image layanan API VisionQC.
#
# Konteks build adalah AKAR REPO, bukan folder Backend. Layanan ini mengimpor
# `visionqc_ai` dari AI_model/, jadi image harus memuat keduanya. Dockerfile
# sebelumnya dibangun dengan konteks ./Backend, sehingga modul AI tidak pernah
# ikut dan container gagal pada impor pertama.
#
#     docker compose up --build

FROM python:3.11-slim

# libGL dan libglib dibutuhkan OpenCV. Varian headless memang tidak menarik
# seluruh Qt, tetapi tetap bertumpu pada pustaka sistem ini, dan tanpanya
# `import cv2` gagal dengan galat pustaka bersama yang tidak menyebut OpenCV.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VISIONQC_ROOT=/app/AI_model

# Torch dipasang lebih dulu dari indeks CPU. Ultralytics menuntut torch, dan
# roda bawaan PyPI membawa seluruh runtime CUDA yang berukuran beberapa
# gigabyte. Layanan ini berjalan di CPU - panitia tidak diwajibkan punya GPU -
# jadi roda CUDA hanya menggelembungkan image tanpa dipakai sama sekali.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 torchvision==0.23.0

# Modul AI dipasang dari sumbernya. Versi numpy, opencv, onnxruntime, dan
# ultralytics karenanya hanya ditulis di AI_model/pyproject.toml.
COPY AI_model/pyproject.toml /app/AI_model/pyproject.toml
COPY AI_model/src /app/AI_model/src
RUN pip install --no-cache-dir /app/AI_model

COPY Backend/requirements.txt /app/Backend/requirements.txt
RUN pip install --no-cache-dir -r /app/Backend/requirements.txt

# Config dan daftar model disalin sebelum unduhan, karena skrip unduhan
# membaca models.json untuk mengetahui berkas apa dan sidik jari mana yang
# diharapkan.
COPY AI_model/configs /app/AI_model/configs
COPY AI_model/models/models.json /app/AI_model/models/models.json
COPY AI_model/scripts/download_models.py /app/AI_model/scripts/download_models.py

# Bobot diunduh saat build, bukan saat container menyala. Mengunduhnya saat
# menyala membuat layanan tampak hidup lalu gagal pada permintaan pertama bila
# jaringan bermasalah; saat build, kegagalannya terlihat langsung dan lapisan
# hasilnya di-cache sehingga penyalaan berikutnya tidak mengunduh ulang.
RUN python /app/AI_model/scripts/download_models.py \
    && python /app/AI_model/scripts/download_models.py --check

COPY Backend /app/Backend

WORKDIR /app/Backend

EXPOSE 8000

# Pemeriksaan kesehatan memakai endpoint layanan sendiri, sehingga container
# baru dinyatakan sehat setelah model benar-benar termuat.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=5).status == 200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
