# EXPERIMENTS.md - Catatan Eksperimen & Bukti Fine-Tuning

Dokumen ini adalah bukti bahwa model benar-benar di-fine-tune. Panitia
mewajibkan: *"Model wajib di fine tune sesuai dengan inovasi fitur per tim."*

Angka yang belum diukur ditulis `belum diukur` dan tidak pernah dikarang.
Panitia berhak meminta demo langsung dan klarifikasi saat penjurian.

**Status:** Step 2, 4, 5, dan 6 selesai. Deteksi, segmentasi, dan anomali sudah
dilatih serta diuji. Kalibrasi dan conformal masih `belum diukur`.

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

Dihasilkan `scripts/compare_segmentation.py`; angka lengkap di
`reports/metrics/segmentation_comparison.json`. Pencocokan memakai **IoU mask**,
bukan IoU kotak pembatas, karena dua mask yang bentuknya sangat berbeda bisa
saja punya kotak yang nyaris sama.

### 4.1 Ringkas run

| | |
|---|---|
| Model dasar | YOLO11n-seg pra-latih COCO |
| Dataset | `data/processed/seg`, split identik dengan deteksi |
| Epoch diminta | 250 |
| Epoch dijalankan | **250** (tidak berhenti awal) |
| Epoch terbaik | **203** |
| Waktu tempuh | **66,8 menit** pada RTX 3050 Laptop 4 GB |
| Metrik val terbaik (mask) | precision 0,8817 - recall 0,7253 - mAP50 **0,7899** - mAP50-95 0,5218 |
| Metrik val terbaik (kotak) | mAP50 0,8090 - mAP50-95 0,5895 |

Berbeda dengan deteksi yang berhenti awal di epoch 161, run ini berjalan penuh
sampai batas 250. Bobot terbaik jatuh di epoch 203, artinya hanya 47 epoch
terakhir yang tanpa perbaikan sementara `patience` bernilai 40. Model **mungkin
masih bisa membaik** bila diberi epoch lebih banyak, dan hal itu dinyatakan apa
adanya alih-alih diklaim sudah konvergen.

### 4.2 Perbandingan metrik pada test set

| Metrik (tingkat instance, IoU mask 0,5) | Baseline pra-latih | Sesudah fine-tune | 95% BCa |
|---|---|---|---|
| Recall | 0,0000 | **0,6000** | [0,4681 ; 0,7101] |
| Precision | 0,0000 | **0,6000** | [0,4731 ; 0,7083] |
| F2 | 0,0000 | 0,6000 | [0,4724 ; 0,7049] |
| True positive | 0 | 45 | dari 75 instance |
| False positive | 40 | 30 | |
| False negative | 75 | 30 | |

| Kualitas mask pada pasangan yang cocok | Baseline | Sesudah fine-tune |
|---|---|---|
| IoU rerata | 0,0000 | **0,7847** |
| IoU median | 0,0000 | 0,8065 |
| Jumlah pasangan | 0 | 45 |

| Metrik (tingkat gambar) | Baseline | Sesudah fine-tune |
|---|---|---|
| TP / FP / TN / FN | 30 / 3 / 4 / 22 | 51 / 1 / 6 / 1 |
| Recall | 0,577 | **0,981** |
| **MCC** | **0,097** | **0,838** |

MCC tingkat gambar model segmentasi (0,838) justru **lebih tinggi** daripada
model deteksi (0,719). Untuk keputusan lolos atau tolak, segmentasi terbukti
lebih dapat diandalkan meskipun recall per instance-nya lebih rendah.

### 4.3 Estimasi luas cacat

Ini keluaran yang benar-benar dipakai decision engine, dan karena itu galatnya
diukur tersendiri. Satuannya **poin persen** terhadap luas gambar.

| Galat mutlak luas cacat | Baseline | Sesudah fine-tune |
|---|---|---|
| Rerata | 15,3698 | **0,3714** |
| 95% BCa rerata | [11,1014 ; 21,2449] | **[0,1662 ; 0,9968]** |
| Median | 10,6940 | **0,0381** |
| Persentil ke-95 | 56,3491 | **1,2943** |

Baseline meleset rata-rata lebih dari 15 poin persen karena model COCO
menyegmentasi **seluruh objeknya**, misalnya botol utuh, bukan area cacatnya.
Sesudah fine-tuning, galat rerata turun menjadi 0,37 poin persen.

**Kalimat untuk proposal:**

> Sistem melaporkan luas cacat dengan galat mutlak rerata 0,37 poin persen
> [0,17 ; 1,00] pada test set. Karena ambang keputusan luas ditetapkan pada
> 2,0 persen, galat sebesar itu tidak mengubah keputusan pada mayoritas kasus.

Sebaran titik dugaan terhadap luas sebenarnya dapat dilihat pada panel kanan
`reports/figures/segmentation_comparison.png`; titik-titiknya menempel rapat
pada garis identitas.

### 4.4 Uji signifikansi McNemar

| | Fine-tuned menangkap | Fine-tuned gagal |
|---|---|---|
| **Baseline menangkap** | a = 0 | b = 0 |
| **Baseline gagal** | c = 45 | d = 30 |

| | Nilai |
|---|---|
| Metode | khi-kuadrat dengan koreksi kontinuitas |
| Khi-kuadrat McNemar | **43,022** |
| Nilai p | **5,41 x 10^-11** |
| Pasangan (n) | 75 |
| Cohen's h | **1,772** (besar) |
| Kesimpulan | perbedaan signifikan secara statistik |

### 4.5 Recall per kelas

| Kelas | n (test) | Baseline | Sesudah fine-tune |
|---|---|---|---|
| `noda` | 21 | 0,000 | 0,714 |
| `kotor` | 3 | 0,000 | 0,667 |
| `pecah` | 40 | 0,000 | 0,575 |
| `gores` | 9 | 0,000 | 0,333 |
| `deformasi` | 2 | 0,000 | 1,000 |

`gores` menjadi yang terlemah dengan recall 0,333. Penyebabnya masuk akal:
goresan berbentuk garis tipis, dan IoU mask menghukum bentuk tipis jauh lebih
keras daripada IoU kotak. Pada model deteksi, kelas yang sama mencapai 0,556.

### 4.6 Latensi

| Tahap | ms/gambar |
|---|---|
| Prapemrosesan | 6,0 |
| Inferensi | 40,7 |
| Pascapemrosesan | 10,9 |
| **Total** | **57,6** |

Segmentasi sekitar empat kali lebih lambat daripada deteksi yang 15,0 ms.
Angka ini PyTorch di GPU; ONNX di CPU diukur pada Step 9.

### 4.7 Keterbatasan

1. Recall segmentasi (0,600) lebih rendah daripada deteksi (0,693) karena IoU
   mask pada ambang 0,5 jauh lebih ketat daripada IoU kotak.
2. Run berjalan sampai batas epoch, bukan berhenti karena konvergen. Menaikkan
   `epochs` berpotensi memperbaiki hasil bila waktu memungkinkan.
3. Dukungan `kotor` dan `deformasi` masih sangat tipis, sehingga recall-nya
   tidak informatif.

**Gambar pendukung:** `reports/figures/segmentation_comparison.png`,
`segmentation_pr_curve.png`, `segmentation_confusion_matrix.png`.

## 5. Anomaly Detection dan Ambang EVT

Dihasilkan `scripts/train_anomaly.py`; angka lengkap di
`reports/metrics/anomaly_results.json`. Model dilatih **hanya dari gambar
normal**, tanpa satu pun contoh cacat, sehingga jenis cacat yang belum pernah
dilabeli tetap dapat terjaring.

### 5.1 Model yang dipakai, dan kenapa bukan pilihan utama

| | |
|---|---|
| Model | **PaDiM** (cadangan), bukan EfficientAD |
| Backbone | ResNet18 pra-latih, 2,8 juta parameter |
| Gambar latih | dibatasi 200 per kategori |
| Kategori | 5, masing-masing satu model |
| Waktu | sekitar 2 menit per kategori |

`configs/training.yaml` menetapkan EfficientAD sebagai pilihan utama dengan
PaDiM sebagai cadangan, dan **cadangan itulah yang dipakai**. Alasannya waktu:
EfficientAD memerlukan unduhan dataset ImageNette sekitar 1,5 GB dan pelatihan
ribuan langkah per kategori, sementara sisa waktu lomba masih harus menampung
kalibrasi, decision engine, dan integrasi. Hal ini dinyatakan terbuka, bukan
disembunyikan, dan EfficientAD tetap terbuka untuk dicoba di tahap Final.

### 5.2 Pembagian data

Ambang dikalibrasi pada data yang **tidak dipakai mengukurnya**. Tanpa
pemisahan ini, laju alarm palsu yang dilaporkan menjadi melingkar.

| Split | Isi | Perannya |
|---|---|---|
| latih | gambar normal | model belajar seperti apa produk normal |
| kalibrasi | gambar normal, diambil dari latih, tidak ikut dilatih | dasar perhitungan ambang |
| uji | gambar normal + cacat | dasar laju alarm palsu dan recall |

| Kategori | Kalibrasi | Uji normal | Uji cacat |
|---|---|---|---|
| `bottle` | 39 | 34 | 9 |
| `chewinggum` | 85 | 75 | 15 |
| `cashew` | 85 | 75 | 14 |
| `pipe_fryum` | 85 | 75 | 14 |
| `fryum` | 85 | 75 | 14 |

### 5.3 Hasil

| Kategori | AUROC gambar | Ambang EVT | Kuantil empiris | Alarm palsu (uji) | Recall cacat |
|---|---|---|---|---|---|
| `bottle` | **0,9967** | 45,861 | 44,431 | 0,029 | **1,0000** |
| `chewinggum` | 0,9307 | 37,234 | 33,630 | 0,067 | 0,8667 |
| `fryum` | 0,8867 | 71,157 | 76,264 | 0,000 | 0,1429 |
| `pipe_fryum` | 0,8571 | 53,243 | 53,041 | 0,000 | 0,2143 |
| `cashew` | 0,8476 | 56,550 | 46,076 | 0,013 | 0,2857 |

Target laju alarm palsu 1 persen, seed 42.

### 5.4 Pencocokan GPD pada ekor

| Kategori | u | Kuantil dipakai | N_u / n | xi | sigma | KS p | Tafsiran ekor |
|---|---|---|---|---|---|---|---|
| `bottle` | 21,389 | 74 | 10/39 | +0,667 | 2,119 | 0,966 | berat |
| `chewinggum` | 23,692 | 85 | 13/85 | +0,312 | 3,150 | 0,940 | berat |
| `cashew` | 25,589 | 85 | 13/85 | +0,454 | 5,737 | 0,593 | berat |
| `pipe_fryum` | 29,897 | 85 | 13/85 | -0,340 | 13,138 | 0,898 | terbatas |
| `fryum` | 31,366 | 85 | 13/85 | +0,790 | 4,124 | 0,978 | berat |

Nilai p uji Kolmogorov-Smirnov berkisar 0,593 sampai 0,978 pada seluruh
kategori. Tidak ada satu pun yang menolak GPD, sehingga pemodelan ekor ini
memang didukung data dan bukan asumsi yang dipaksakan.

Tanda parameter bentuk pun bervariasi secara masuk akal. `pipe_fryum`
terdeteksi berekor terbatas, artinya ada batas atas skor yang tak terlampaui,
sedangkan empat kategori lain berekor berat sehingga nilai ekstrem yang jauh
lebih besar masih mungkin muncul. Perbedaan itu tidak akan terlihat bila
ambang ditetapkan sebagai kuantil empiris begitu saja.

**Kalimat untuk proposal:**

> Ambang 45,8613 pada kategori botol diperoleh dengan memodelkan ekor sebaran
> skor anomali sampel normal memakai sebaran Pareto Tergeneralisasi
> (xi = 0,6669; sigma = 2,1188; u = 21,3890; N_u/n = 10/39), menargetkan laju
> alarm palsu 1 persen. Uji Kolmogorov-Smirnov tidak menolak kecocokan model
> tersebut (p = 0,966).

### 5.5 Temuan penting: AUROC tinggi tetapi recall rendah

Tiga kategori mencatat AUROC di atas 0,84 namun recall hanya 0,14 sampai 0,29
pada ambang EVT. Keduanya tidak bertentangan dan perbedaannya perlu dipahami:

AUROC mengukur kemampuan **mengurutkan**, yaitu peluang skor sebuah gambar
cacat lebih tinggi daripada gambar normal yang diambil acak. Recall mengukur
kinerja pada **satu titik operasi tertentu**. Menargetkan laju alarm palsu 1
persen menempatkan ambang begitu tinggi sehingga mayoritas cacat jatuh di
bawahnya.

Untuk inspeksi produk pangan, 1 persen adalah titik operasi yang keliru.
`configs/inference.yaml` menaksir biaya cacat lolos ke konsumen Rp 50.000
berbanding Rp 2.000 untuk produk bagus yang salah ditolak. Dengan asimetri
biaya sebesar itu, menahan alarm palsu di 1 persen justru memperbesar total
kerugian.

Hasil ini menjadi bukti empiris yang mendasari Step 7: angka 1 persen diwarisi
dari kebiasaan industri, dan data di sini memperlihatkan harganya.

### 5.6 Keterbatasan

1. `chewinggum` mencatat laju alarm palsu 6,7 persen pada split uji padahal
   targetnya 1 persen. Kalibrasi pada 85 sampel ternyata belum cukup mewakili
   ekor sebaran yang sebenarnya.
2. `bottle` hanya memiliki 39 sampel kalibrasi, sehingga ambang awal terpaksa
   turun ke kuantil ke-74 agar ekornya cukup terisi. Semakin rendah kuantil
   awal, semakin lemah landasan teoretis pendekatan GPD.
3. Model yang dipakai adalah cadangan, bukan pilihan utama.
4. Setiap kategori memiliki model dan ambang sendiri. Sistem pada tahap
   penyisihan memakai ambang kategori `bottle` sebagai nilai statis karena
   botol adalah produk demo utama.

**Gambar pendukung:** `reports/figures/anomaly_{bottle,chewinggum,cashew,pipe_fryum,fryum}.png`,
masing-masing memuat sebaran skor beserta ambangnya dan diagram kuantil-kuantil
kecocokan GPD.

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

Diukur pada RTX 3050 Laptop, rata-rata 59 gambar test, resolusi 640 px.

| Model | Tahap | Median (ms) | Ukuran bobot |
|---|---|---|---|
| YOLO11n detect | PyTorch GPU | **15,0** | 5,5 MB |
| YOLO11n-seg | PyTorch GPU | **57,6** | 6,0 MB |
| YOLO11n detect | ONNX CPU | belum diukur | belum diukur |
| YOLO11n-seg | ONNX CPU | belum diukur | belum diukur |
| Pipeline penuh | ONNX CPU | belum diukur | - |

Rincian per tahap ada di bagian 3.6 dan 4.6. Angka ONNX di CPU adalah target
penyajian yang sebenarnya dan baru diukur pada Step 9, dilaporkan sebagai
median dan persentil ke-95 dari 100 kali jalan.

## 8. Riwayat Run

| # | Tgl | Model | Perubahan | Hasil | Keputusan |
|---|---|---|---|---|---|
| 1 | 2026-08-18 | YOLO11n detect | fine-tune pertama dari bobot COCO, config apa adanya | berhenti awal di epoch 161, terbaik epoch 137, mAP50 val 0,7967 | diterima sebagai model deteksi tahap penyisihan |
| 3 | 2026-08-18 | PaDiM anomali | 5 kategori, 200 gambar latih, worker dataloader nol | AUROC 0,848 sampai 0,997; ambang EVT lolos uji KS di semua kategori | diterima; titik operasi 1 persen dinilai terlalu ketat dan diserahkan ke Step 7 |
| 2 | 2026-08-18 | YOLO11n-seg | fine-tune dari bobot COCO, config apa adanya | berjalan penuh 250 epoch, terbaik epoch 203, mask mAP50 val 0,7899 | diterima; galat luas cacat 0,37 poin persen sudah memadai |

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
| 2026-08-18 | Step 6 gagal empat kali berturut-turut | dirinci di bawah | seluruhnya diperbaiki; run bersih kelima berhasil |
| 2026-08-18 | Skor anomali hanya bernilai tiga macam | post-processor anomalib menormalisasi skor terhadap ambang internalnya dan praktis mengkuantisasi keluaran, sehingga ekor sebaran hilang dan tidak ada yang dapat dimodelkan GPD | post-processor dimatikan; skor mentah kembali kontinu dengan 39 nilai unik |
| 2026-08-18 | Hasil kategori pertama hilang saat kategori kedua gagal | JSON hanya ditulis di akhir setelah semua kategori selesai | penulisan dijadikan inkremental dan menggabung isi berkas yang sudah ada |
| 2026-08-18 | Proses mati kehabisan memori | worker dataloader disetel 4; di Windows tiap worker adalah proses Python penuh yang memuat ulang torch, sekitar 690 MB masing-masing | worker disetel nol; pemuatan data bukan hambatan pada model ini |
| 2026-08-18 | Blok penulis JSON lama tidak ikut terhapus dan menimpa hasil gabungan | penyuntingan berkas yang tidak menyeluruh | blok lama dihapus; hasil diverifikasi ulang terhadap log, 20 dari 20 angka cocok |
| 2026-08-18 | Laporan sempat menyebut proses sudah berhenti padahal masih berjalan | `pgrep` pada Git Bash tidak melihat proses Windows | pemeriksaan diganti memakai `Get-CimInstance` |
| 2026-08-18 | Segmentasi tidak berhenti awal | 250 epoch habis sebelum `patience` 40 tercapai, bobot terbaik di epoch 203 | dicatat apa adanya; model belum tentu konvergen dan dapat dilatih lebih lama bila waktu memungkinkan |
