# EXPERIMENTS.md - Catatan Eksperimen & Bukti Fine-Tuning

>  **Dokumen ini adalah bukti kepatuhan .**Panitia mewajibkan: *"Model wajib di fine tune
> sesuai dengan inovasi fitur per tim."* Tabel di bawah adalah cara kita membuktikannya.
>
>  **dilarang mengarang angka.**Sel yang belum diukur ditulis `belum diukur`.
> Panitia berhak meminta demo langsung dan klarifikasi saat penjurian.

**Status:**Step 2 selesai (dataset siap & tervalidasi). Belum ada run training -
seluruh sel metrik model masih `belum diukur`.

---

## 1. Lingkungan

Diukur langsung pada mesin yang dipakai, 18 Agustus 2026.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop - **VRAM 4096 MiB (4 GB)** , terkonfirmasi `nvidia-smi` |
| Driver / CUDA | 592.00 / CUDA 13.1 (runtime torch: cu121) |
| Python | 3.11.1 |
| torch | 2.5.1+cu121 - `torch.cuda.is_available() = True` |
| opencv-python | 4.13.0 |
| numpy / scipy / scikit-learn | 1.26.4 / 1.13.0 / 1.5.0 |
| ultralytics | `belum terpasang` (dibutuhkan mulai Step 4) |
| Seed | 42 (dikunci di `configs/dataset.yaml`, ) |

## 2. Dataset

Dihasilkan `scripts/prepare_dataset.py`; angka lengkap ada di
`reports/metrics/dataset_stats.json`.

### 2.1 Sumber dan pemilihan kategori

Kategori dipilih berdasarkan **ukuran cacat setelah resize ke 640 px** , diukur
langsung dari mask. Objek di bawah ~12 px praktis tak dapat dipelajari detektor.

| Kategori | Sumber | Lisensi | Gambar cacat | Gambar normal | Instance | Objek ≥12px @640 | Peran |
|---|---|---|---|---|---|---|---|
| `bottle` | MVTec AD | CC BY-NC-SA 4.0 | 63 | 229 | 68 | **100,0 %** | deteksi + segmentasi + anomali |
| `chewinggum` | VisA | CC BY 4.0 | 100 | 503 | 214 | **96,3 %** | deteksi + segmentasi + anomali |
| `cashew` | VisA | CC BY 4.0 | 100 | 500 | 141 | **67,6 %** | deteksi + segmentasi + anomali |
| `pipe_fryum` | VisA | CC BY 4.0 | 100 | 500 | 220 | **63,9 %** | deteksi + segmentasi + anomali |
| `fryum` | VisA | CC BY 4.0 | 100 | 500 | 254 | 29,8 % | **anomali saja** - cacat terlalu kecil untuk bbox |
| `macaroni1` | VisA | CC BY 4.0 | - | - | - | 37,8 % | tidak diproses (cadangan) |
| `macaroni2` | VisA | CC BY 4.0 | - | - | - | 31,8 % | tidak diproses (cadangan) |
| Sintetik (Step 3) | dibuat sendiri | - | belum dibuat | - | - | - | penyeimbang kelas |

Kategori berdefek sangat kecil **tidak dibuang** , melainkan dialihkan ke jalur
anomaly detection yang tidak memerlukan kotak pembatas. Ini sekaligus menjadi
bukti empiris kenapa sistem memerlukan dua pendekatan sekaligus.

### 2.2 Rekonstruksi label VisA

Mask VisA menyimpan **kode jenis cacat** pada nilai pikselnya, tetapi pemetaan
kodejenis tidak ikut didistribusikan. Pemetaan itu dipulihkan secara empiris
lalu diuji ulang; tanpa langkah ini VisA hanya berguna sebagai data biner.

| Kategori | Kode dipelajari | Kode dari eliminasi | Kode dilebur ke kelas | Validasi multi-label |
|---|---|---|---|---|
| `chewinggum` | 4 | 1 | 0 | **43/43 cocok** |
| `cashew` | 8 | 0 | 0 | **10/10 cocok** |
| `pipe_fryum` | 6 | 0 | 2 | **12/12 cocok** |
| `fryum` | 7 | 1 | 0 | **13/13 cocok** |
| `macaroni1` | 5 | 1 | 0 | **20/20 cocok** |
| `macaroni2` | 6 | 1 | 0 | **22/22 cocok** |
| **Total** | | | **120/120 (100 %)** | Pada `pipe_fryum`, dua kode tak terpisahkan karena *middle breakage* dan *small
cracks* tidak pernah muncul sendirian. Keduanya bermuara ke kelas `pecah`
sehingga tetap dapat dipakai - dicatat apa adanya, bukan disembunyikan.

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

> `burnt` sengaja dilebur ke `noda`: keduanya penyimpangan warna permukaan, dan

> memisahkannya hanya menyisakan ±34 contoh - terlalu tipis untuk dilatih.
> Dapat dipisah kembali di tahap Final.

### 2.4 Pembagian split (seed 42, dikunci - )

| Split | Gambar | dari itu latar belakang | Instance | `pecah` | `gores` | `noda` | `kotor` | `deformasi` |
|---|---|---|---|---|---|---|---|---|
| train | 291 | 37 | 471 | 231 | 110 | 103 | 16 | 11 |
| val | 64 | 7 | 97 | 54 | 20 | 17 | 3 | 3 |
| test | 59 | 7 | 75 | 40 | 9 | 21 | 3 | 2 |
| **total** | **414** | **51** | **643** | **325** | **139** | **141** | **22** | **16** |

### 2.5 Validasi statistik (STATISTICS.md bagian 2)

| Uji | Nilai | Kriteria | Lulus? |
|---|---|---|---|
| χ² keselarasan distribusi kelas antar split | χ² = 7,161 · df = 8 · **p = 0,5194** | p > 0,05 | |
| Rasio ketimpangan (IR) | **20,31** | < 3,0 | |
| Entropi Shannon ternormalisasi | **0,7556** | > 0,85 | |
| Instance cacat di test | **75** (pada 52 gambar) | ≥ 139 | |

**Tafsiran jujur atas tiga yang belum lulus:**1. **IR 20,31 dan entropi 0,7556** - akibat langsung kelangkaan `kotor` (22
   instance) dan `deformasi` (16). Inilah tugas Step 3: generator cacat
   sintetik kini punya **sasaran terukur** , bukan sekadar "tambah data".
   Untuk menurunkan IR ke bawah 3,0 dibutuhkan ±108 instance `pecah`-ekuivalen
   pada kedua kelas langka tersebut.
2. **Ukuran test set** - dengan 75 instance, recall dapat diestimasi pada galat
   **±6,8 % (95 % CI)** , bukan ±5 % seperti yang ditargetkan. Angka ±6,8 %
   inilah yang wajib ditulis di proposal. Menambah data sintetik **tidak boleh** dipakai memperbesar test set: test set harus tetap mewakili data nyata.
3. Catatan uji χ²: frekuensi harapan minimum < 5 pada sel kelas langka,
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

**Verifikasi visual:**`reports/figures/dataset_samples_train.png` dan
`dataset_samples_test.png` - kotak dan poligon digambar ulang dari berkas label
yang sesungguhnya, lalu diperiksa mata. Validasi statistik bisa lulus sementara
koordinatnya tergeser; lembar kontak inilah yang menutup celah itu.

## 3.  BUKTI FINE-TUNING - Deteksi

### 3.1 Perbandingan metrik

| Metrik | Baseline (pre-trained) | Fine-tuned | Δ | 95% BCa CI (Δ) |
|---|---|---|---|---|
| mAP@50 | belum diukur | belum diukur | - | - |
| mAP@50-95 | belum diukur | belum diukur | - | - |
| **Recall (cacat)** | belum diukur | belum diukur | - | - |
| Precision (cacat) | belum diukur | belum diukur | - | - |
| F2-score | belum diukur | belum diukur | - | - |
| MCC | belum diukur | belum diukur | - | - |

### 3.2 Uji signifikansi - McNemar (STATISTICS.md bagian 4)

| | Fine-tuned benar | Fine-tuned salah |
|---|---|---|
| **Baseline benar** | a = - | b = - |
| **Baseline salah** | c = - | d = - |

| | Nilai |
|---|---|
| χ² McNemar | belum diukur |
| nilai p | belum diukur |
| Cohen's h (besar efek) | belum diukur |
| **Kesimpulan** | *belum diuji* |

**Kalimat target untuk proposal***(isi setelah run sungguhan)*:
> Fine-tuning memperbaiki __ kasus yang sebelumnya salah dan merusak __ kasus yang sebelumnya benar.
> Uji McNemar menunjukkan perbedaan signifikan secara statistik (χ² = __, p = __, n = __).

### 3.3 Hyperparameter yang dipakai
Sumber: `configs/training.yaml` bagian detection · Perubahan dari default: *belum ada run*

---

## 4.  BUKTI FINE-TUNING - Segmentasi

| Metrik | Baseline | Fine-tuned | Δ | 95% CI |
|---|---|---|---|---|
| Mask mAP@50 | belum diukur | belum diukur | - | - |
| Mean IoU | belum diukur | belum diukur | - | - |
| Galat estimasi luas cacat (%) | belum diukur | belum diukur | - | - |

Uji McNemar Nilai

|---|---| χ² / p | belum diukur | ---

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
| Cakupan empiris | belum diukur (target ≥ 0,95) |
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
| - | - | - | *belum ada run* | - | - |

> **Aturan:**satu perubahan per run. Kalau augmentasi, ukuran model, dan learning rate diubah

> sekaligus lalu hasilnya membaik, penyebabnya tidak dapat diketahui.

---

## 9. Kegagalan & Pelajaran

*(Isi bagian ini dengan jujur - kegagalan yang tercatat justru memperkuat kredibilitas proposal
dan menunjukkan proses kerja yang nyata.)*

| Tgl | Yang gagal | Penyebab | Tindakan |
|---|---|---|---|
| - | - | - | - |
