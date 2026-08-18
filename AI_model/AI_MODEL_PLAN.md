# AI_MODEL_PLAN.md - Rencana Kerja Modul AI

**VisionQC**· COMPFEST 18 AIC · Penyisihan
**Domain:** inspeksi kualitas **makanan & minuman kemasan** (botol, kaleng, pouch, sachet)
**Pemilik modul:** Anggota 1 (AI + Frontend)
**Hardware:** NVIDIA RTX 3050 Laptop
**Deadline:** 25 Agustus 2026, 23.55 WIB

Dokumen ini adalah **peta kerja modul AI dari nol sampai bisa dipanggil Backend.** | Dokumen pendamping | Isi |---|---|
| [DATASET_REQUIREMENTS.md](./DATASET_REQUIREMENTS.md) | Dataset apa saja yang dibutuhkan + sumber valid | [STATISTICS.md](./STATISTICS.md) | Rumus statistik - **diferensiator utama** |
| [PRIVACY.md](./PRIVACY.md) | Desain privasi & kepatuhan UU PDP | [EXPERIMENTS.md](./EXPERIMENTS.md) | Bukti fine-tuning (wajib panitia) |

---

## 1. Sasaran Modul

Modul AI ini **satu-satunya** hal yang perlu dibangun Anggota 1 di sisi model. Keluarannya:

```python
# Ini kontrak final ke Backend. Semua pekerjaan di bawah bermuara ke sini.
from visionqc_ai import run_inspection, InspectionResult

result: InspectionResult = run_inspection(image_bytes, config)
```

**Definisi selesai untuk seluruh modul AI:** 1.  `run_inspection()` bisa dipanggil Backend dan mengembalikan hasil lengkap
2.  Bobot model **hasil fine-tuning** tersedia dalam format ONNX
3.  Ada bukti fine-tuning: baseline vs sesudah, dengan **uji signifikansi statistik** 4.  Threshold bersifat **statis** , dibaca dari `configs/inference.yaml`
5.  Lapisan privasi aktif: EXIF dibersihkan, wajah diburamkan, tidak ada penyimpanan gambar
6.  Latensi terukur dan wajar untuk demo
7.  `EXPERIMENTS.md` terisi angka hasil run sungguhan (bukan karangan)

---

## 2. Diferensiator - Kenapa Modul Ini Berbeda

Peserta lain kemungkinan besar: latih YOLO  laporkan akurasi  selesai.
Modul ini menambahkan **tiga lapisan yang jarang dipakai peserta lomba** :

| # | Lapisan | Apa bedanya |
|---|---|---|
| **1** | **Statistical Inference Layer** | Keputusan tidak memakai ambang batas asal-asalan. Memakai **conformal prediction** (jaminan cakupan bebas distribusi), **kalibrasi kepercayaan** , dan **Extreme Value Theory** untuk ambang anomali. Lihat [STATISTICS.md](./STATISTICS.md) |
| **2** | **Bukti statistik untuk fine-tuning** | Bukan sekadar "mAP naik dari 0,72 ke 0,89", tapi **uji McNemar**+ **selang kepercayaan bootstrap**"peningkatan signifikan secara statistik, p < 0,001". Langsung menjawab kewajiban fine-tuning dari panitia |
| **3** | **Privacy-by-design** | Pemrosesan sepenuhnya lokal, tanpa penyimpanan gambar, EXIF dibersihkan, wajah operator diburamkan, log hanya berisi hash. Selaras **UU PDP No. 27/2022** . Lihat [PRIVACY.md](./PRIVACY.md) |

Ketiganya **gratis secara komputasi** (tidak butuh GPU tambahan) tapi memberi kedalaman yang sulit

ditandingi dalam waktu 10 hari.

---

## 3. Peta 10 Langkah

Tiap langkah dirancang agar selesai sebagai satu unit kerja yang berdiri sendiri.

| Step | Nama | Keluaran utama | Est. waktu | Hari |
|---|---|---|---|---|
| **1** | **Fondasi & Perencanaan** | Struktur folder, 5 dokumen, config, requirements | 1 jam | H-10 |
| **2** | **Akuisisi & Preprocessing Dataset** | Dataset terunduh, format YOLO, split terkunci + validasi statistik | 3 jam | H-7  |
| **3** | Generator Cacat Sintetik | Penambah data + penyeimbang kelas (diizinkan panitia) | 3 jam | H-9 |
| **4** | **Baseline + Fine-Tune Detection** | Model detect terlatih + **uji McNemar** vs baseline | 4 jam | H-7 selesai |
| **5** | **Fine-Tune Segmentation** | Model seg + perhitungan luas cacat % | 3 jam | H-7 selesai |
| **6** | Anomaly Detection + Ambang EVT | EfficientAD + ambang berbasis Extreme Value Theory | 4 jam¹ | H-8 |
| **7** | Lapisan Statistik & Kalibrasi | Conformal prediction, temperature scaling, ambang sensitif biaya | 3 jam | H-7 |
| **8** | Lapisan Privasi | EXIF scrub, face blur, ephemeral buffer, hash audit | 2 jam | H-7 |
| **9** | Decision Engine + Ekspor ONNX | Mesin keputusan 3 kelas + ONNX + tolok ukur latensi | 3 jam | H-6 |
| **10** | Integrasi & Laporan Evaluasi | `run_inspection()` siap Backend + EXPERIMENTS.md terisi | 3 jam | H-6 |

¹ *termasuk waktu tunggu training di RTX 3050 - bisa dijalankan sambil mengerjakan Frontend.*

**Total ± 29 jam kerja aktif** tersebar di 5 hari (H-10 s/d H-6), menyisakan H-5 s/d H-1 untuk
Frontend, video, dan perbaikan.

---

## 4. Rincian Tiap Step

### Step 1 - Fondasi & Perencanaan `[SELESAI]`

**Dibuat:** - Struktur folder `AI_model/`
- `AI_MODEL_PLAN.md` (dokumen ini)
- `DATASET_REQUIREMENTS.md` - dataset + sumber valid
- `STATISTICS.md` - seluruh rumus statistik
- `PRIVACY.md` - desain privasi
- `EXPERIMENTS.md` - kerangka tabel bukti fine-tuning
- `configs/training.yaml` - hyperparameter khusus RTX 3050
- `configs/inference.yaml` - threshold **statis** - `requirements.txt`

**Belum ada kode** - ini memang langkah perencanaan.

---

### Step 2 - Akuisisi & Preprocessing Dataset `[SELESAI 18 Agu 2026]`

**Dibuat:** ```
src/visionqc_ai/data/
|-- taxonomy.py          # 5 kelas + pemetaan 30+ label mentah (dapat diaudit)
|-- sources.py           # registri kategori + tata letak folder mentah yang sebenarnya
|-- visa_codes.py        # rekonstruksi kode jenis cacat pada mask VisA
|-- mask_utils.py        # mask  bbox + poligon YOLO
|-- records.py           # representasi antara
|-- convert_mvtec.py     # MVTec (mask biner + nama folder)  record
|-- convert_visa.py      # VisA (mask ber-kode + CSV)  record
|-- split.py             # stratified split, seed 42 terkunci
|-- validate.py          # χ², entropi, rasio ketimpangan, ukuran sampel
|-- writer.py            # penulisan detect / seg / anomaly
scripts/prepare_dataset.py     scripts/preview_dataset.py
configs/dataset.yaml
reports/metrics/dataset_stats.json
reports/figures/dataset_samples_{train,test}.png
```

**Temuan terpenting - mask VisA menyimpan jenis cacat.** Nilai piksel mask VisA bukan biner melainkan **kode jenis cacat** , tetapi
pemetaan kodejenis tidak ikut didistribusikan. Kode itu dipulihkan lewat
belajar dari gambar berlabel tunggal  eliminasi  peleburan kelas, lalu diuji
ulang: **120/120 gambar multi-label cocok** . Tanpa langkah ini VisA hanya
berguna sebagai data biner normal/anomali; dengan langkah ini VisA memberi
label jenis cacat **per instance** .

**Keputusan berbasis pengukuran, bukan selera.** Kategori dipilih dari ukuran
cacat setelah resize ke 640 px: `bottle` 100 %, `chewinggum` 96,3 %,
`cashew` 67,6 %, `pipe_fryum` 63,9 % objek ≥ 12 px  masuk deteksi.
`fryum` (29,8 %), `macaroni1` (37,8 %), `macaroni2` (31,8 %) terlalu kecil
dialihkan ke jalur anomali, tidak dibuang.

**Hasil:** 414 gambar (291/64/59), 643 instance cacat.

| Uji | Nilai | Lulus? |
|---|---|---|
| χ² keselarasan split | p = 0,5194 | |
| Rasio ketimpangan | 20,31 (target < 3,0) | tugas Step 3 |
| Entropi ternormalisasi | 0,7556 (target > 0,85) | tugas Step 3 |
| Instance cacat di test | 75 (minimum 139) | galat recall ±6,8 %, dilaporkan apa adanya |

Rincian lengkap: [EXPERIMENTS.md bagian 2](./EXPERIMENTS.md).

---

### Step 3 - Generator Cacat Sintetik

**Tujuan:** cacat itu langka. Sintetik menyelesaikan ketimpangan kelas **dan** menjadi diferensiator
(panitia mengizinkan data sintetik secara eksplisit).

**Yang dibuat:** ```
src/visionqc_ai/data/
|-- synthetic.py         # injeksi cacat: penyok, gores, sobek, tumpah, label miring, segel rusak
|-- augment.py           # pipeline Albumentations
scripts/generate_synthetic.py
reports/figures/synthetic_samples.png
```

**Jenis cacat yang dibuat** (sesuai domain kemasan makanan/minuman):

| Cacat | Cara sintesis | Relevansi |
|---|---|---|
| `penyok` (dent) | Deformasi lokal + perubahan shading | Kaleng minuman penyok saat distribusi |
| `gores` (scratch) | Goresan Perlin noise + blending | Botol/kaleng tergores |
| `sobek` (tear) | Robekan tepi tidak beraturan | Kemasan pouch/sachet |
| `segel_rusak` (seal) | Distorsi area segel | Segel tutup botol tidak sempurna |
| `label_miring` (misalign) | Rotasi + translasi label | Label tertempel miring |
| `kotor` (contamination) | Bercak/noda acak | Kontaminasi permukaan |

**Kontrol kualitas:** setiap gambar sintetik divalidasi agar distribusinya tidak terlalu jauh dari
gambar asli - memakai **jarak Wasserstein** pada histogram intensitas (STATISTICS.md bagian 2.4). Sintetik
yang terlalu "palsu" justru merusak model.

**Sasaran terukur dari Step 2** (bukan lagi perkiraan):

| Kelas | Instance sekarang | Kebutuhan agar IR < 3,0 | Tambahan |
|---|---|---|---|
| `pecah` | 325 | - | - |
| `noda` | 141 | - | - |
| `gores` | 139 | - | - |
| **`kotor`** | **22** | ≥ 109 | **+87** |
| **`deformasi`** | **16** | ≥ 109 | **+93** |

**Kriteria selesai:** rasio ketimpangan turun di bawah 3,0 dan entropi

ternormalisasi naik di atas 0,85 - keduanya diukur ulang oleh
`scripts/prepare_dataset.py`. Sampel sintetik hanya masuk **train** ; test set
wajib tetap murni data nyata.

---

### Step 4 - Baseline + Fine-Tune Detection `[SELESAI 18 Agu 2026]`

**Dibuat:**
```
src/visionqc_ai/training/train_detect.py
src/visionqc_ai/evaluation/
|-- matching.py         # pencocokan prediksi dengan ground truth
|-- metrics.py          # precision, recall, F2, MCC tingkat gambar
|-- bootstrap.py        # selang kepercayaan BCa dan Wilson
|-- significance.py     # uji McNemar dan Cohen's h
|-- detection_eval.py   # menjalankan model pada satu split
scripts/train_detection.py    scripts/compare_detection.py
models/finetuned/detect/      reports/metrics/detection_comparison.json
reports/figures/{detection_comparison,training_curves,detection_confusion_matrix,detection_pr_curve}.png
```

**Hasil pada test set** (75 instance, ambang IoU 0,5):

| Metrik | Baseline pra-latih | Sesudah fine-tune |
|---|---|---|
| Recall | 0,0000 | **0,6933** [0,5793 ; 0,7989] |
| Precision | 0,0000 | **0,7429** [0,6195 ; 0,8382] |
| F2 | 0,0000 | 0,7027 [0,5968 ; 0,7977] |
| MCC tingkat gambar | 0,090 | **0,719** |

Uji McNemar: 52 instance membaik, 0 memburuk, khi-kuadrat 50,019,
p = 1,52 x 10^-12. Cohen's h = 1,968 yang tergolong besar.

Training berhenti awal di epoch 161 dari 300 dengan bobot terbaik pada epoch
137, memakan waktu sekitar 22 menit pada RTX 3050 Laptop 4 GB.

Rincian lengkap beserta keterbatasannya ada di
[EXPERIMENTS.md bagian 3](./EXPERIMENTS.md).

---

### Step 5 - Fine-Tune Segmentation `[SELESAI 18 Agu 2026]`

**Dibuat:**
```
src/visionqc_ai/evaluation/segmentation_eval.py
scripts/compare_segmentation.py
models/finetuned/seg/    reports/metrics/segmentation_comparison.json
reports/figures/{segmentation_comparison,segmentation_pr_curve,segmentation_confusion_matrix}.png
```

Training memakai ulang `scripts/train_detection.py` dengan bagian
`segmentation`, sehingga tidak ada kode training yang digandakan.

**Hasil pada test set** (75 instance, ambang IoU mask 0,5):

| Metrik | Baseline pra-latih | Sesudah fine-tune |
|---|---|---|
| Recall mask | 0,0000 | **0,6000** [0,4681 ; 0,7101] |
| IoU mask rerata | 0,0000 | **0,7847** |
| MCC tingkat gambar | 0,097 | **0,838** |
| Galat luas cacat | 15,37 poin persen | **0,37 poin persen** |

Uji McNemar: 45 instance membaik, 0 memburuk, khi-kuadrat 43,022,
p = 5,41 x 10^-11. Cohen's h = 1,772 yang tergolong besar.

Berjalan penuh 250 epoch tanpa berhenti awal, bobot terbaik epoch 203,
memakan waktu 66,8 menit.

Rincian lengkap di [EXPERIMENTS.md bagian 4](./EXPERIMENTS.md).

---

### Step 6 - Anomaly Detection + Ambang EVT

**Tujuan:** menangkap cacat **jenis baru** yang tidak ada di data latih.

**Yang dibuat:** ```
src/visionqc_ai/training/train_anomaly.py    # EfficientAD via anomalib
src/visionqc_ai/inference/anomaly.py
src/visionqc_ai/statistics/evt.py            # POT + Generalized Pareto
scripts/train_anomaly.py
reports/figures/{anomaly_score_dist,evt_fit}.png
```

**Diferensiator statistik:** ambang anomali **tidak dipilih asal** . Memakai
**Peaks-Over-Threshold + Generalized Pareto Distribution** (STATISTICS.md bagian 7):

> *"Ambang 0,7412 dipilih dengan memodelkan ekor distribusi skor anomali sampel normal memakai GPD
> (ξ̂ = 0,18; σ̂ = 0,042), menjamin laju alarm palsu ≤ 1%."*

Ini pernyataan yang bisa dipertahankan di depan juri. "Kami pakai 0,75 karena terlihat bagus" tidak.

**Rencana cadangan:** kalau EfficientAD terlalu berat untuk RTX 3050 dalam waktu tersedia,
turun ke **PaDiM** atau **PatchCore** (jauh lebih ringan). Dicatat jujur di EXPERIMENTS.md.

---

### Step 7 - Lapisan Statistik & Kalibrasi

**Tujuan:** ini inti diferensiator. Mengubah keluaran model mentah menjadi keputusan yang
**terjamin secara statistik** .

**Yang dibuat:** ```
src/visionqc_ai/statistics/
|-- conformal.py         # split & Mondrian conformal prediction
|-- calibration.py       # temperature scaling, ECE, reliability diagram
|-- cost_sensitive.py    # ambang optimal berbasis biaya asimetris
|-- spc.py               # kartu kendali p-chart, CUSUM, EWMA (untuk laporan)
reports/figures/{reliability_diagram,conformal_coverage,cost_curve}.png
```

**Tiga hasil konkret:** 1. **Conformal prediction** - kelas `REVIEW` bukan lagi tebakan, tapi konsekuensi matematis:
   > *"Dengan α = 0,05, sistem menjamin label sebenarnya tercakup dalam himpunan prediksi
   > minimal 95% dari waktu. Bila himpunan berisi lebih dari satu label, sistem menahan
   > keputusan (REVIEW)."*

2. **Kalibrasi** - skor kepercayaan jadi bermakna. Sebelum kalibrasi ECE biasanya 0,1-0,2
   (artinya "confidence 0,9" sebenarnya cuma benar ~75%). Setelah temperature scaling turun < 0,05.

3. **Ambang sensitif biaya** - memakai biaya nyata:
   > *cacat lolos ke konsumen ≈ Rp 50.000 · salah tolak produk bagus ≈ Rp 2.000*
   >  ambang optimal Bayes = 2.000/52.000 ≈ 0,038

---

### Step 8 - Lapisan Privasi

**Tujuan:** memenuhi permintaan "sistem privasi maksimal" dengan implementasi nyata, bukan klaim.

**Yang dibuat:** ```
src/visionqc_ai/privacy/
|-- exif.py              # hapus SELURUH metadata (GPS, perangkat, waktu)
|-- face_blur.py         # deteksi & buramkan wajah sebelum inferensi
|-- ephemeral.py         # context manager: buffer di-nolkan setelah dipakai
|-- ocr_filter.py        # allowlist regex - hanya pola kode batch yang dipertahankan
|-- audit.py             # log hanya SHA-256, tidak pernah gambar mentah
reports/privacy_audit.md
```

Rincian ancaman & mitigasi ada di [PRIVACY.md](./PRIVACY.md).

---

### Step 9 - Decision Engine + Ekspor ONNX

**Yang dibuat:** ```
src/visionqc_ai/inference/
|-- decision.py          # PASS / REJECT / REVIEW - menggabungkan semua sinyal
|-- annotate.py          # gambar bbox + mask + heatmap di atas gambar asli
|-- pipeline.py          # orkestrasi urutan
src/visionqc_ai/export/onnx_export.py
scripts/benchmark_latency.py
models/onnx/
reports/metrics/latency_benchmark.json
```

**Decision engine** menggabungkan: himpunan conformal + luas cacat + skor anomali (ambang EVT) +
kepercayaan terkalibrasi + ambang sensitif biaya.

**Tolok ukur latensi** dilaporkan sebagai **median dan persentil ke-95** dari 100 kali jalan,
bukan satu angka - karena satu pengukuran tidak berarti apa-apa.

---

### Step 10 - Integrasi & Laporan Evaluasi

**Yang dibuat:** ```
src/visionqc_ai/__init__.py        # ekspor run_inspection() + InspectionResult
src/visionqc_ai/schemas.py         # Pydantic - sinkron dengan Backend
pyproject.toml                     # agar Backend bisa: pip install -e ../AI_model
tests/                             # unit test secukupnya (bukan bulk testing)
EXPERIMENTS.md                     # TERISI PENUH
reports/evaluation_report.md       # bahan langsung untuk proposal
```

**Serah terima ke Backend:** ```python
from visionqc_ai import run_inspection
result = run_inspection(image_bytes, config)   # InspectionResult
```

Setelah step ini, modul AI **selesai** dan Backend tinggal memanggil.

---

## 5. Prasyarat Sebelum Tiap Tahap

Daftar ini mencegah ada langkah yang tertahan di tengah jalan.

### Wajib sebelum Step 2 (blokir kalau tidak ada)

| # | Kebutuhan | Kenapa | Cara |
|---|---|---|---|
| 1 | ~~Dataset terunduh~~ | **BERES** - MVTec bottle/capsule/pill + VisA 12 kategori | - |
| 2 | ~~Python 3.11 + CUDA~~ | **BERES** - Python 3.11.1, torch 2.5.1+cu121, CUDA aktif | - |
| 3 | Ruang disk ± 15 GB | tersisa 27 GB - cukup tapi tipis | pantau saat training |
| 4 | ~~Konfirmasi VRAM~~ | **BERES** - 4096 MiB (4 GB). `training.yaml` sudah sesuai | - |

### Dibutuhkan sebelum Step 4 (training)

| # | Kebutuhan | Kenapa |
|---|---|---|
| 5 | **Laptop bisa menyala lama** | Fine-tuning 1-3 jam per model. Sebaiknya dijalankan malam hari |
| 6 | ~~Konfirmasi kategori~~ | **BERES** - bottle, chewinggum, cashew, pipe_fryum (deteksi) + fryum (anomali) |

### Dibutuhkan sebelum Step 7 (opsional tapi memperkuat)

| # | Kebutuhan | Kenapa |
|---|---|---|
| 7 | **Angka biaya nyata** | Untuk ambang sensitif biaya. Perkiraan kasar sudah cukup: berapa rugi kalau produk cacat lolos vs salah tolak produk bagus? |
| 8 | **Target laju alarm palsu** | Standar industri 1%, dipakai sebagai nilai awal |

### Dibutuhkan sebelum Step 10 (integrasi)

| # | Kebutuhan | Kenapa |
|---|---|---|
| 9 | **Skema `InspectionResult` dari Anggota 2** | Harus sinkron persis dengan Backend, kalau tidak integrasi gagal |
| 10 | **Kesepakatan cara distribusi bobot model** | GitHub Releases (rekomendasi) atau Hugging Face |

### Opsional - sangat menaikkan kualitas demo

| # | Kebutuhan | Nilai tambah |
|---|---|---|
| 11 | **30-60 foto produk kemasan nyata** (Teh Botol, Aqua, Chitato, Indomie) pakai HP | Video demo dengan produk lokal yang dikenal juri jauh lebih meyakinkan daripada dataset asing. Bukan untuk latih - untuk gambar contoh & demo |

---


## 6. Aturan Kerja Modul Ini

| Aturan | Isi |
|---|---|
| | Jangan pernah mengarang angka. Kolom kosong ditulis `belum diukur` |
| | Fine-tuning wajib, dan buktinya harus ada di repo |
| | Threshold statis di `configs/inference.yaml`, tidak berubah saat runtime |
| | Test set tidak pernah dilihat saat tuning. Seed terkunci di config |
| | Dilarang auto-tuning, bulk testing script, loop umpan balik otomatis |
| | Bobot model & dataset tidak masuk git |
| | Commit conventional tanpa scope: `feat: ...`, `fix: ...`, `refactor: ...` |

---
