# AI_model - Modul AI VisionQC

Modul inspeksi kualitas berbasis computer vision untuk **makanan & minuman kemasan** .
Keluaran akhir modul ini adalah satu fungsi yang dipanggil Backend.

```python
from visionqc_ai import run_inspection, InspectionResult

result: InspectionResult = run_inspection(image_bytes, config)
```

---

## Dokumentasi

| Dokumen | Isi |
|---|---|
| [AI_MODEL_PLAN.md](./AI_MODEL_PLAN.md) | **Mulai dari sini** - rencana 10 langkah & kebutuhan |
| [DATASET_REQUIREMENTS.md](./DATASET_REQUIREMENTS.md) | Dataset yang dibutuhkan + sumber valid |
| [STATISTICS.md](./STATISTICS.md) | Rumus statistik - **diferensiator utama** |
| [PRIVACY.md](./PRIVACY.md) | Desain privasi & kepatuhan UU PDP No. 27/2022 |
| [EXPERIMENTS.md](./EXPERIMENTS.md) | Bukti fine-tuning (wajib panitia) |

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
[ Decision engine ]   PASS / REJECT / REVIEW + alasan
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

Keluaran: `models/finetuned/detect/`, `reports/metrics/detection_comparison.json`,
`reports/figures/detection_comparison.png`.
```

Keluaran: `data/processed/{detect,seg,anomaly}/`,
`reports/metrics/dataset_stats.json`, `reports/figures/dataset_samples_*.png`.

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
| 5 | Fine-Tune Segmentation | belum |
| 6 | Anomaly Detection + Ambang EVT | belum |
| 7 | Lapisan Statistik & Kalibrasi | belum |
| 8 | Lapisan Privasi | belum |
| 9 | Decision Engine + Ekspor ONNX | belum |
| 10 | Integrasi & Laporan Evaluasi | belum |

Rincian tiap step ada di [AI_MODEL_PLAN.md](./AI_MODEL_PLAN.md) bagian 4.

---

## Prinsip Kerja

- Angka yang belum diukur ditulis `belum diukur`, tidak pernah dikarang.
- Fine-tuning wajib dan buktinya harus ada di repo: skrip, config, log, metrik.
- Ambang keputusan bersifat statis dan tidak berubah saat runtime.
- Test set tidak pernah dilihat saat penyetelan hyperparameter.
- Tidak ada auto-tuning maupun loop umpan balik otomatis.
