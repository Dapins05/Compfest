# AI_model - Modul AI VisionQC

Modul inspeksi kualitas berbasis computer vision untuk **makanan & minuman kemasan** .
Keluaran akhir modul ini adalah satu fungsi yang dipanggil Backend.

```python
from visionqc_ai import InspectionResult, run_inspection

result: InspectionResult = run_inspection(image_bytes)
```

---

## Titik Sambung ke Backend

Backend cukup memakai empat nama. Tidak ada yang lain dari modul ini yang perlu
diimpor.

```python
from visionqc_ai import (
    InspectionResult,   # model Pydantic, langsung dapat dikembalikan FastAPI
    InvalidImageError,  # kesalahan pada berkas kiriman pengguna
    load_pipeline,      # dipanggil sekali saat startup
    run_inspection,     # dipanggil per permintaan
)
```

**Saat startup.** Panggil `load_pipeline()` satu kali. Bobot dimuat dan warmup
dijalankan di sana, sehingga permintaan pertama tidak menanggung biaya
pemuatan. Selisihnya nyata: tanpa warmup panggilan pertama memakan sekitar 4,8
detik, dengan warmup sekitar 0,25 detik.

```python
pipeline = load_pipeline()
if not pipeline.ready:
    ...  # bobot belum diunduh; jalankan scripts/download_models.py
```

`pipeline.ready`, `pipeline.anomaly_available`, dan `pipeline.face_blur_available`
dapat dipakai mengisi endpoint kesehatan. Ketiganya melaporkan keadaan apa
adanya: bila sebuah model tidak ada, sistem tetap berjalan dengan kemampuan
berkurang dan mengatakannya, alih-alih diam-diam melewatinya.

**Per permintaan.**

```python
try:
    result = run_inspection(image_bytes)
except InvalidImageError as exc:      # berkas pengguna yang tidak memenuhi syarat
    raise HTTPException(status_code=400, detail=str(exc)) from exc
```

`InvalidImageError` adalah turunan `ValueError` dan hanya dilempar untuk
kesalahan pada masukan: berkas kosong, format di luar daftar izin, ukuran
melebihi 10 MB, dimensi di luar rentang 224 sampai 4096 piksel, atau berkas
yang tidak dapat diurai sebagai gambar. Galat jenis lain berarti kegagalan
sistem dan pantas dijawab 500.

**Yang perlu diketahui:**

| Hal | Keterangan |
|---|---|
| Letak config dan model | Ditelusuri dari penanda `configs/inference.yaml`. Bila Backend menaruhnya di tempat lain, setel variabel lingkungan `VISIONQC_ROOT` |
| Permintaan bersamaan | Aman. Inspeksi dijalankan satu per satu di dalam kunci, karena objek model menyimpan status di dalam dirinya |
| Latensi | Sekitar 230 milidetik per gambar pada CPU setelah warmup |
| Jaringan | Tidak ada panggilan keluar saat inferensi |
| `pipeline.last_audit` | Catatan berbasis SHA-256 untuk permintaan terakhir. Karena satu pipeline dipakai bersama, pakailah nilai kembalian `inspect` bila catatan per permintaan dibutuhkan |

---

## Dokumentasi

| Dokumen | Isi |
|---|---|
| [AI_MODEL_PLAN.md](./AI_MODEL_PLAN.md) | **Mulai dari sini** - rencana 10 langkah & kebutuhan |
| [DATASET_REQUIREMENTS.md](./DATASET_REQUIREMENTS.md) | Dataset yang dibutuhkan + sumber valid |
| [STATISTICS.md](./STATISTICS.md) | Rumus statistik - **diferensiator utama** |
| [PRIVACY.md](./PRIVACY.md) | Desain privasi & kepatuhan UU PDP No. 27/2022 |
| [EXPERIMENTS.md](./EXPERIMENTS.md) | Bukti fine-tuning: angka dan seberapa dapat dipercaya |
| [OUTPUTS.md](./OUTPUTS.md) | Peta keluaran tiap langkah: berkas apa, di mana |

---


## Yang Membedakan Modul Ini

| # | Lapisan | Isi |
|---|---|---|
| 1 | **Statistical inference** | Conformal prediction (jaminan cakupan bebas distribusi), kalibrasi kepercayaan, ambang anomali via Extreme Value Theory - bukan ambang batas pilihan tangan |
| 2 | **Bukti fine-tuning statistik** | Uji McNemar + selang kepercayaan bootstrap, bukan sekadar menampilkan dua angka |
| 3 | **Privacy by design** | Pemrosesan lokal, tanpa penyimpanan gambar, EXIF dibersihkan, wajah diburamkan, log hanya hash |

---


## Arsitektur Pipeline

```
gambar masuk
   |
[ Lapisan privasi ]   hapus EXIF, buramkan wajah, buffer sekali pakai
   |
[ Prapemrosesan ]     validasi, resize 640x640, normalisasi
   |
[ Inferensi ]         YOLO11-detect
                      YOLO11-seg        --> [ Lapisan statistik ]
                      EfficientAD           conformal + kalibrasi + ambang EVT
                      PaddleOCR
   |
[ Decision engine ]   PASS / REJECT + alasan
   |
InspectionResult --> Backend
```

---

## Struktur Folder

```
AI_model/
|-- configs/
|   |-- dataset.yaml         # kategori, preprocessing, split terkunci
|   |-- training.yaml        # hyperparameter (disetel untuk RTX 3050)
|   |-- inference.yaml       # ambang keputusan, statis saat runtime
|-- src/visionqc_ai/
|   |-- data/                # unduh, konversi, split, sintetik
|   |-- training/            # skrip fine-tuning
|   |-- evaluation/          # metrik, bootstrap, uji signifikansi
|   |-- statistics/          # conformal, kalibrasi, EVT, SPC
|   |-- privacy/             # EXIF, face blur, ephemeral, audit
|   |-- inference/           # pipeline, decision engine, anotasi
|-- scripts/                 # entrypoint yang dijalankan manual
|-- data/                    # tidak masuk git
|-- models/                  # tidak masuk git
|-- reports/                 # metrik & gambar hasil evaluasi
|-- notebooks/               # eksplorasi
```

---

## Menjalankan Preprocessing (Step 2)

```bash
python scripts/prepare_dataset.py               # konversi, split, validasi, tulis dataset
python scripts/preview_dataset.py --split train # lembar kontak verifikasi anotasi
```

## Menjalankan Fine-Tuning Deteksi (Step 4)

```bash
python scripts/train_detection.py               # fine-tune YOLO11n
python scripts/compare_detection.py             # baseline vs hasil, uji McNemar
```

## Menjalankan Fine-Tuning Segmentasi (Step 5)

```bash
python scripts/train_detection.py --data data/processed/seg/data.yaml \
    --name seg --section segmentation
python scripts/compare_segmentation.py
```

## Melatih Model Anomali (Step 6)

```bash
python scripts/train_anomaly.py
python scripts/train_anomaly.py --categories bottle --model padim
```

Model dilatih hanya dari gambar normal. Ambangnya dihitung dari ekor sebaran
skor memakai teori nilai ekstrem, bukan dipilih tangan.

## Kalibrasi Lapisan Keputusan (Step 7)

```bash
python scripts/calibrate_decision.py
```

Mengukur kalibrasi kepercayaan, menghitung kuantil conformal, dan menilai
ambang berdasarkan biaya kesalahan.

Kuantil conformal tetap dihitung dan dilaporkan sebagai keterlacakan, tetapi
sejak mode keputusan menjadi biner (20 Agustus 2026) kuantil itu **tidak lagi
menentukan apa pun**. Sistem hanya mengembalikan PASS atau REJECT; tidak ada
kelas REVIEW. Mode tiga kelas masih ada di kode dan masih diuji, disiapkan
untuk tahap Final.

## Ekspor ONNX dan Tolok Ukur Latensi (Step 9)

```bash
python scripts/export_onnx.py
```

Menghasilkan `models/onnx/` dan mengukur latensi di CPU sebagai median beserta
persentil ke-95 dari 100 pengulangan.

## Menjalankan Uji

```bash
pip install -e .
pytest
```

Keluaran: `models/finetuned/detect/`, `reports/metrics/detection_comparison.json`,
`reports/figures/detection_comparison.png`.
```

Keluaran: `data/processed/{detect,seg,anomaly}/`,
`reports/metrics/dataset_stats.json`, `reports/figures/dataset_samples_*.png`.

---

## Mengunduh Bobot Model

Bobot model tidak disimpan di git karena ukurannya. Unduh sekali setelah
mengkloning repo, sebelum menjalankan apa pun:

```bash
cd AI_model
python scripts/download_models.py
```

Skrip ini hanya memakai pustaka bawaan Python sehingga dapat dijalankan
sebelum `pip install`. Setiap berkas diverifikasi dengan SHA-256 terhadap
daftar di `models/models.json`; unduhan yang rusak dibuang alih-alih dipakai,
karena model cacat tetap berjalan dan memberi hasil keliru tanpa gejala.

Memeriksa tanpa mengunduh:

```bash
python scripts/download_models.py --check
```

Sumbernya adalah GitHub Release `models-v1.0.0` pada repo ini. Unduhan
anonim hanya berhasil bila repo bersifat publik.

---

## Setup

```bash
# 1. PyTorch dengan CUDA - HARUS lebih dulu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Sisanya
pip install -r requirements.txt

# 3. Verifikasi GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Status

| Step | Nama | Status |
|---|---|---|
| 1 | Fondasi & Perencanaan | selesai |
| 2 | Akuisisi & Preprocessing Dataset | selesai |
| 3 | Generator Cacat Sintetik | ditunda |
| 4 | Baseline + Fine-Tune Detection | selesai |
| 5 | Fine-Tune Segmentation | selesai |
| 6 | Anomaly Detection + Ambang EVT | selesai |
| 7 | Lapisan Statistik & Kalibrasi | selesai |
| 8 | Lapisan Privasi | selesai |
| 9 | Decision Engine + Ekspor ONNX | selesai |
| 10 | Integrasi & Laporan Evaluasi | selesai |

Rincian tiap step ada di [AI_MODEL_PLAN.md](./AI_MODEL_PLAN.md) bagian 4.

---

## Prinsip Kerja

- Angka yang belum diukur ditulis `belum diukur`, tidak pernah dikarang.
- Fine-tuning wajib dan buktinya harus ada di repo: skrip, config, log, metrik.
- Ambang keputusan bersifat statis dan tidak berubah saat runtime.
- Test set tidak pernah dilihat saat penyetelan hyperparameter.
- Tidak ada auto-tuning maupun loop umpan balik otomatis.
