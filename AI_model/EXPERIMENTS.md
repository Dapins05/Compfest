# EXPERIMENTS.md - Catatan Eksperimen & Bukti Fine-Tuning

Dokumen ini adalah bukti bahwa model benar-benar di-fine-tune. Panitia
mewajibkan: *"Model wajib di fine tune sesuai dengan inovasi fitur per tim."*

Angka yang belum diukur ditulis `belum diukur` dan tidak pernah dikarang.
Panitia berhak meminta demo langsung dan klarifikasi saat penjurian.

**Status:** Step 2 dan Step 4 selesai. Deteksi sudah di-fine-tune dan diuji
pada test set. Segmentasi, anomali, dan kalibrasi masih `belum diukur`.

---

## 1. Lingkungan

Diukur langsung pada mesin yang dipakai, 18 Agustus 2026.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop - **VRAM 4096 MiB (4 GB)**, terkonfirmasi `nvidia-smi` |
| Driver / CUDA | 592.00 / CUDA 13.1 (runtime torch: cu121) |
| Python | 3.11.1 |
| torch | 2.5.1+cu121 - `torch.cuda.is_available() = True` |
| opencv-python | 4.13.0 |
| numpy / scipy / scikit-learn | 2.4.6 / 1.17.1 / 1.5.0 |
| ultralytics | 8.4.121 |
| Seed | 42 (dikunci di `configs/dataset.yaml`) |

## 2. Dataset

Dihasilkan `scripts/prepare_dataset.py`; angka lengkap ada di
`reports/metrics/dataset_stats.json`.

### 2.1 Sumber dan pemilihan kategori

Kategori dipilih berdasarkan **ukuran cacat setelah resize ke 640 px**, diukur
langsung dari mask. Objek di bawah ~12 px praktis tak dapat dipelajari detektor.

| Kategori | Sumber | Lisensi | Gambar cacat | Gambar normal | Instance | Objek >=12px @640 | Peran |
|---|---|---|---|---|---|---|---|
| `bottle` | MVTec AD | CC BY-NC-SA 4.0 | 63 | 229 | 68 | **100,0 %** | deteksi + segmentasi + anomali |
| `chewinggum` | VisA | CC BY 4.0 | 100 | 503 | 214 | **96,3 %** | deteksi + segmentasi + anomali |
| `cashew` | VisA | CC BY 4.0 | 100 | 500 | 141 | **67,6 %** | deteksi + segmentasi + anomali |
| `pipe_fryum` | VisA | CC BY 4.0 | 100 | 500 | 220 | **63,9 %** | deteksi + segmentasi + anomali |
| `fryum` | VisA | CC BY 4.0 | 100 | 500 | 254 | 29,8 % | **anomali saja** - cacat terlalu kecil untuk bbox |
| `macaroni1` | VisA | CC BY 4.0 | - | - | - | 37,8 % | tidak diproses (cadangan) |
| `macaroni2` | VisA | CC BY 4.0 | - | - | - | 31,8 % | tidak diproses (cadangan) |
| Sintetik (Step 3) | dibuat sendiri | - | belum dibuat | - | - | - | penyeimbang kelas |

Kategori berdefek sangat kecil **tidak dibuang**, melainkan dialihkan ke jalur
anomaly detection yang tidak memerlukan kotak pembatas. Ini sekaligus menjadi
bukti empiris kenapa sistem memerlukan dua pendekatan sekaligus.

### 2.2 Rekonstruksi label VisA

Mask VisA menyimpan **kode jenis cacat** pada nilai pikselnya, tetapi pemetaan
kode ke jenis tidak ikut didistribusikan. Pemetaan itu dipulihkan secara empiris
lalu diuji ulang; tanpa langkah ini VisA hanya berguna sebagai data biner.

| Kategori | Kode dipelajari | Kode dari eliminasi | Kode dilebur ke kelas | Validasi multi-label |
|---|---|---|---|---|
| `chewinggum` | 4 | 1 | 0 | **43/43 cocok** |
| `cashew` | 8 | 0 | 0 | **10/10 cocok** |
| `pipe_fryum` | 6 | 0 | 2 | **12/12 cocok** |
| `fryum` | 7 | 1 | 0 | **13/13 cocok** |
| `macaroni1` | 5 | 1 | 0 | **20/20 cocok** |
| `macaroni2` | 6 | 1 | 0 | **22/22 cocok** |
| **Total** | | | | **120/120 (100 %)** |

Pada `pipe_fryum`, dua kode tidak terpisahkan karena *middle breakage* dan
*small cracks* tidak pernah muncul sendirian. Keduanya bermuara ke kelas
`pecah` sehingga tetap dapat dipakai, dan hal itu dicatat apa adanya.

### 2.3 Taksonomi kelas

30+ label mentah dua dataset disatukan menjadi 5 kelas. Pemetaan lengkap dan
dapat diaudit ada di `src/visionqc_ai/data/taxonomy.py`.

| id | Kelas | Contoh label mentah yang dipetakan |
|---|---|---|
| 0 | `pecah` | `broken_large`, `broken_small`, `chunk of gum missing`, `corner missing`, `middle breakage`, `small holes`, `small cracks` |
| 1 | `gores` | `scratches`, `small scratches` |
| 2 | `noda` | `similar colour spot`, `different color spot`, `same colour spot`, `burnt` |
| 3 | `kotor` | `contamination` |
| 4 | `deformasi` | `stuck together`, `fryum stuck together` |

> `burnt` sengaja dilebur ke `noda`: keduanya penyimpangan warna permukaan,
> dan memisahkannya hanya menyisakan sekitar 34 contoh yang terlalu tipis
> untuk dilatih. Dapat dipisah kembali di tahap Final.

### 2.4 Pembagian split (seed 42, dikunci)

| Split | Gambar | dari itu latar belakang | Instance | `pecah` | `gores` | `noda` | `kotor` | `deformasi` |
|---|---|---|---|---|---|---|---|---|
| train | 291 | 37 | 471 | 231 | 110 | 103 | 16 | 11 |
| val | 64 | 7 | 97 | 54 | 20 | 17 | 3 | 3 |
| test | 59 | 7 | 75 | 40 | 9 | 21 | 3 | 2 |
| **total** | **414** | **51** | **643** | **325** | **139** | **141** | **22** | **16** |

### 2.5 Validasi statistik (STATISTICS.md bagian 2)

| Uji | Nilai | Kriteria | Lulus? |
|---|---|---|---|
| Khi-kuadrat keselarasan distribusi kelas antar split | 7,161 - df 8 - **p = 0,5194** | p > 0,05 | **lulus** |
| Rasio ketimpangan (IR) | **20,31** | < 3,0 | belum lulus |
| Entropi Shannon ternormalisasi | **0,7556** | > 0,85 | belum lulus |
| Instance cacat di test | **75** (pada 52 gambar) | minimal 139 | belum lulus |

**Tafsiran jujur atas tiga yang belum lulus:**

1. **IR 20,31 dan entropi 0,7556** - akibat langsung kelangkaan `kotor` (22
   instance) dan `deformasi` (16). Inilah tugas Step 3: generator cacat
   sintetik kini punya **sasaran terukur**, bukan sekadar "tambah data".
   Untuk menurunkan IR ke bawah 3,0 dibutuhkan sekitar 108 instance tambahan
   pada kedua kelas langka tersebut.
2. **Ukuran test set** - dengan 75 instance, recall dapat diestimasi pada galat
   **6,8 % (95 % CI)**, bukan 5 % seperti yang ditargetkan. Angka 6,8 %
   inilah yang wajib ditulis di proposal. Menambah data sintetik **tidak
   boleh** dipakai memperbesar test set, karena test set harus tetap mewakili
   data nyata.
3. Catatan uji khi-kuadrat: frekuensi harapan minimum di bawah 5 pada sel
   kelas langka,
   sehingga nilai p bersifat indikatif. Akan dihitung ulang setelah Step 3.

### 2.6 Instance yang dibuang saat konversi

Komponen mask di bawah 25 px² dianggap derau pelabelan.

| Kategori | Dibuang (luas) | Dibuang (sisi < 3px) | Label tak dikenal | Mask hilang |
|---|---|---|---|---|
| `bottle` | 0 | 0 | 0 | 0 |
| `chewinggum` | 42 | 0 | 0 | 0 |
| `cashew` | 11 | 0 | 0 | 0 |
| `pipe_fryum` | 94 | 0 | 0 | 0 |
| `fryum` | 280 | 19 | 1 (`other`) | 0 |

Angka `fryum` yang mencolok (280 dari 553 komponen) memperkuat keputusan
mengeluarkannya dari dataset deteksi.

**Verifikasi visual:** `reports/figures/dataset_samples_train.png` dan
`dataset_samples_test.png` - kotak dan poligon digambar ulang dari berkas label
yang sesungguhnya, lalu diperiksa mata. Validasi statistik bisa lulus sementara
koordinatnya tergeser; lembar kontak inilah yang menutup celah itu.

## 3. BUKTI FINE-TUNING - Deteksi

Dihasilkan `scripts/compare_detection.py`; angka lengkap ada di
`reports/metrics/detection_comparison.json`. Evaluasi dijalankan pada split
**test** yang tidak pernah dilihat selama training maupun pemilihan epoch.

### 3.1 Ringkas run

| | |
|---|---|
| Model dasar | YOLO11n pra-latih COCO (`yolo11n.pt`, 2.583.127 parameter) |
| Dataset | 291 train / 64 val / 59 test, 5 kelas cacat |
| Epoch diminta | 300 |
| Epoch dijalankan | **161** (berhenti awal, `patience` 50) |
| Epoch terbaik | **137** |
| Waktu tempuh | sekitar 22 menit pada RTX 3050 Laptop 4 GB |
| Metrik val terbaik | precision 0,7319 - recall 0,7543 - mAP50 **0,7967** - mAP50-95 0,4033 |

### 3.2 Perbandingan metrik pada test set

Pencocokan memakai ambang IoU 0,5. Selang kepercayaan BCa diperoleh dari 2.000
resampling dengan **gambar** sebagai unit resampling, bukan instance, karena
cacat di dalam satu gambar tidak saling bebas.

| Metrik (tingkat instance) | Baseline pra-latih | Sesudah fine-tune | 95% BCa (sesudah) |
|---|---|---|---|
| Recall | 0,0000 | **0,6933** | [0,5793 ; 0,7989] |
| Precision | 0,0000 | **0,7429** | [0,6195 ; 0,8382] |
| F1 | 0,0000 | 0,7172 | [0,6166 ; 0,8030] |
| **F2** (recall diutamakan) | 0,0000 | **0,7027** | [0,5968 ; 0,7977] |
| True positive | 0 | 52 | dari 75 instance |
| False negative | 75 | 23 | |

Selang Wilson untuk recall: 0,6933 [0,5817 ; 0,7861], konsisten dengan BCa.

| Metrik (tingkat gambar) | Baseline | Sesudah fine-tune |
|---|---|---|
| TP / FP / TN / FN | 22 / 2 / 5 / 30 | 49 / 1 / 6 / 3 |
| Recall | 0,423 | **0,942** |
| Specificity | 0,714 | 0,857 |
| **MCC** | **0,090** | **0,719** |

### 3.3 Uji signifikansi McNemar

Kedua model dievaluasi pada test set yang sama, sehingga hasilnya berpasangan
per instance ground truth.

| | Fine-tuned menangkap | Fine-tuned gagal |
|---|---|---|
| **Baseline menangkap** | a = 0 | b = 0 |
| **Baseline gagal** | c = 52 | d = 23 |

| | Nilai |
|---|---|
| Metode | khi-kuadrat dengan koreksi kontinuitas |
| Khi-kuadrat McNemar | **50,019** |
| Nilai p | **1,52 x 10^-12** |
| Pasangan (n) | 75 |
| Cohen's h | **1,968** (besar) |
| Kesimpulan | perbedaan signifikan secara statistik |

**Kalimat untuk proposal:**

> Fine-tuning memperbaiki 52 instance cacat yang sebelumnya lolos dan tidak
> merusak satu pun instance yang sebelumnya tertangkap. Uji McNemar
> menunjukkan perbedaan yang signifikan secara statistik
> (khi-kuadrat = 50,019; p = 1,52 x 10^-12; n = 75), dengan besar efek
> Cohen's h = 1,968 yang tergolong besar. Recall kelas cacat naik dari 0,0000
> menjadi 0,6933 [0,5793 ; 0,7989].

### 3.4 Kenapa baseline bernilai nol, dan kenapa itu tetap bermakna

Recall baseline nol bukan kebetulan: kosakata kelas COCO tidak memuat satu pun
jenis cacat, sehingga secara konstruksi model itu tidak mungkin menamai cacat
dengan benar. Karena angka nol mudah disalahartikan sebagai kesalahan
pengukuran, evaluasi diulang dalam mode **kelas-abai**, yang hanya menilai
apakah lokasi cacat ditemukan tanpa mempedulikan namanya:

| Mode kelas-abai | Baseline | Sesudah fine-tune |
|---|---|---|
| Recall | 0,0133 | 0,7467 |
| Precision | 0,0357 | 0,8000 |

Baseline hanya menemukan 1 dari 75 lokasi cacat. Jadi model pra-latih bukan
sekadar salah menamai, melainkan memang tidak melihat cacatnya sama sekali.

Hal serupa terlihat pada tingkat gambar: baseline mencapai recall 0,423 karena
kerap menandai gambar sebagai bermasalah, tetapi MCC-nya hanya 0,090. MCC
membongkar bahwa penandaan itu praktis acak, dan inilah alasan akurasi tidak
dipakai sebagai metrik utama pada data yang timpang.

### 3.5 Recall per kelas

| Kelas | n (test) | Baseline | Sesudah fine-tune |
|---|---|---|---|
| `pecah` | 40 | 0,000 | 0,725 |
| `noda` | 21 | 0,000 | 0,619 |
| `gores` | 9 | 0,000 | 0,556 |
| `kotor` | 3 | 0,000 | 1,000 |
| `deformasi` | 2 | 0,000 | 1,000 |

> **Peringatan penting.** Recall 1,000 pada `kotor` dan `deformasi` **tidak
> boleh dibaca sebagai model sempurna**. Dukungannya hanya 3 dan 2 instance,
> sehingga selang kepercayaan Wilson-nya masing-masing membentang dari sekitar
> 0,44 dan 0,34 sampai 1,00. Angka itu tidak informatif dan hanya dicantumkan
> demi kelengkapan. Kelangkaan kedua kelas inilah yang menjadi sasaran Step 3.

### 3.6 Latensi

Diukur pada RTX 3050 Laptop, rata-rata 59 gambar test, resolusi 640 px.

| Tahap | ms/gambar |
|---|---|
| Prapemrosesan | 2,3 |
| Inferensi | 11,3 |
| Pascapemrosesan | 1,4 |
| **Total** | **15,0** |

Angka ini untuk PyTorch di GPU. Latensi ONNX di CPU, yang menjadi target
penyajian, diukur pada Step 9 dan belum tersedia.

### 3.7 Hyperparameter

Sumber: `configs/training.yaml` bagian `detection`. Tidak ada perubahan manual
selama run. Nilai kunci: AdamW, lr0 0,001 dengan cosine decay, batch 8,
imgsz 640, mosaic dimatikan pada 15 epoch terakhir, seed 42.

### 3.8 Keterbatasan yang perlu dinyatakan

1. Test set hanya memuat 75 instance cacat, sehingga selang kepercayaan lebar
   (recall +-0,11). Menambah data sintetik tidak boleh dipakai memperbesar
   test set karena test harus tetap mewakili data nyata.
2. Recall `gores` 0,556 adalah yang terendah di antara kelas berdukungan
   memadai. Objek gores memang paling kecil ukurannya.
3. Ketimpangan kelas belum ditangani; run ini dilatih di atas data apa adanya
   dengan rasio ketimpangan 20,31.

**Gambar pendukung:** `reports/figures/detection_comparison.png`,
`training_curves.png`, `detection_confusion_matrix.png`,
`detection_pr_curve.png`.

## 4. BUKTI FINE-TUNING - Segmentasi

| Metrik | Baseline | Fine-tuned | Δ | 95% CI |
|---|---|---|---|---|
| Mask mAP@50 | belum diukur | belum diukur | - | - |
| Mean IoU | belum diukur | belum diukur | - | - |
| Galat estimasi luas cacat (%) | belum diukur | belum diukur | - | - |

Uji McNemar Nilai

|---|---| khi-kuadrat / p | belum diukur | ---

## 5. Anomaly Detection

| Metrik | Nilai | Catatan |
|---|---|---|
| Image AUROC | belum diukur | |
| Pixel AUROC | belum diukur | |
| Model yang dipakai | belum diputuskan | EfficientAD, atau cadangan PaDiM/PatchCore |

**Ambang via Extreme Value Theory** (STATISTICS.md bagian 7):


| Parameter | Nilai |
|---|---|
| ambang POT awal (u) | belum diukur |
| ξ̂ (bentuk GPD) | belum diukur |
| σ̂ (skala GPD) | belum diukur |
| $N_u$ / n | belum diukur |
| **Ambang akhir $z_q$** | belum diukur |
| Laju alarm palsu tercapai | belum diukur (target ≤ 1%) |

---

## 6. Kalibrasi & Conformal

| Metrik | Sebelum | Sesudah | Target |
|---|---|---|---|
| ECE | belum diukur | belum diukur | < 0,05 |
| MCE | belum diukur | belum diukur | - |
| Skor Brier | belum diukur | belum diukur | - |
| Temperature $T^*$ | - | belum diukur | - |

| Conformal (α = 0,05) | Nilai |
|---|---|
| Cakupan empiris | belum diukur (target >= 0,95) |
| Ukuran himpunan rata-rata | belum diukur |
| Cakupan per kelas (Mondrian) | belum diukur |

---


## 7. Latensi & Ukuran Model

| Tahap | Median (ms) | p95 (ms) | Ukuran |
|---|---|---|---|
| PyTorch (GPU) | belum diukur | belum diukur | - |
| ONNX (CPU) | belum diukur | belum diukur | - |
| **Pipeline penuh (CPU)** | belum diukur | belum diukur | - |

Dilaporkan sebagai median dan persentil ke-95 dari **100 kali jalan** - satu pengukuran tunggal

tidak bermakna.

---

## 8. Riwayat Run

| # | Tgl | Model | Perubahan | Hasil | Keputusan |
|---|---|---|---|---|---|
| 1 | 2026-08-18 | YOLO11n detect | fine-tune pertama dari bobot COCO, config apa adanya | berhenti awal di epoch 161, terbaik epoch 137, mAP50 val 0,7967 | diterima sebagai model deteksi tahap penyisihan |

> **Aturan:** satu perubahan per run. Kalau augmentasi, ukuran model, dan learning rate diubah

> sekaligus lalu hasilnya membaik, penyebabnya tidak dapat diketahui.

---

## 9. Kegagalan & Pelajaran

*(Isi bagian ini dengan jujur - kegagalan yang tercatat justru memperkuat kredibilitas proposal
dan menunjukkan proses kerja yang nyata.)*

| Tgl | Yang gagal | Penyebab | Tindakan |
|---|---|---|---|
| 2026-08-18 | Peringatan ketidakcocokan ABI numpy dan scipy | pemasangan ultralytics menaikkan numpy ke 2.4.6 sementara scipy 1.13 menuntut di bawah 2.3 | scipy dinaikkan ke 1.17.1, seluruh angka statistik dihitung ulang di atas kombinasi yang sah |
| 2026-08-18 | Recall `gores` hanya 0,556 | objek gores paling kecil ukurannya dan dukungannya hanya 9 instance di test | dicatat sebagai keterbatasan; penambahan sampel menjadi bahan Step 3 |
