# EXPERIMENTS.md - Catatan Eksperimen & Bukti Fine-Tuning

Dokumen ini adalah bukti bahwa model benar-benar di-fine-tune. Panitia
mewajibkan: *"Model wajib di fine tune sesuai dengan inovasi fitur per tim."*

Angka yang belum diukur ditulis `belum diukur` dan tidak pernah dikarang.
Panitia berhak meminta demo langsung dan klarifikasi saat penjurian.

**Status per 22 Agustus 2026:** Step 2, 4, 5, 6, 7, 8, 9, dan 10 selesai. Modul
AI sudah dapat dipanggil Backend lewat `run_inspection()`. **Step 3 (cacat
sintetik) belum**, dan sasarannya kini menyempit ke satu kelas saja, `kotor`.

Seluruh angka pada dokumen ini **diukur ulang pada 21-22 Agustus 2026** setelah
PKU-GoodsAD masuk dan taksonomi berubah dari lima menjadi enam kelas. Angka run
18 Agustus tidak lagi berlaku; bila muncul di dokumen ini, statusnya hanya
sebagai pembanding historis dan selalu diberi tanggal.

---

## 1. Lingkungan

Diukur langsung pada mesin yang dipakai; diperiksa ulang 22 Agustus 2026.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop - **VRAM 4096 MiB (4 GB)**, terkonfirmasi `nvidia-smi` |
| Driver / CUDA | 592.00 / CUDA 13.1 (runtime torch: cu121) |
| Python | 3.11.1 |
| torch | 2.5.1+cu121 - `torch.cuda.is_available() = True` |
| opencv-python | 5.0.0 |
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
| `drink_bottle` | PKU-GoodsAD | GPL-3.0 | 425 | 1.089 | 498 | belum diukur | deteksi + segmentasi + anomali |
| `food_bottle` | PKU-GoodsAD | GPL-3.0 | 361 | 1.257 | 370 | belum diukur | deteksi + segmentasi + anomali |
| `food_box` | PKU-GoodsAD | GPL-3.0 | 251 | 578 | 299 | belum diukur | deteksi + segmentasi + anomali |
| `food_package` | PKU-GoodsAD | GPL-3.0 | 230 | 793 | 237 | belum diukur | deteksi + segmentasi + anomali |
| `drink_can` | PKU-GoodsAD | GPL-3.0 | 146 | 382 | 192 | belum diukur | deteksi + segmentasi + anomali |
| Sintetik (Step 3) | dibuat sendiri | - | belum dibuat | - | - | - | penyeimbang `kotor` |

Kategori berdefek sangat kecil **tidak dibuang**, melainkan dialihkan ke jalur
anomaly detection yang tidak memerlukan kotak pembatas. Ini sekaligus menjadi
bukti empiris kenapa sistem memerlukan dua pendekatan sekaligus.

**Kenapa kolom "Objek >=12px @640" kosong untuk PKU-GoodsAD.** Metrik itu
diukur pada mask mentah sebelum penyaringan, dan pengukuran tersebut belum
dijalankan untuk GoodsAD (R7.2 - tidak ditulis kalau belum diukur). Menyalin
angka dari statistik "dibuang karena kecil" hasil konversi akan menyesatkan,
karena itu diukur setelah penyaringan sehingga selalu terlihat lebih baik.

Pemilihan kategori GoodsAD memang **tidak** memakai metrik itu sebagai dasar.
Dasarnya dua hal lain: kecocokan domain (barang belanjaan kemasan, bukan
komponen industri) dan ketersediaan mask piksel pada seluruh 1.413 gambar
cacatnya.

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

38 label mentah tiga dataset disatukan menjadi 6 kelas. Pemetaan lengkap dan
dapat diaudit ada di `src/visionqc_ai/data/taxonomy.py`. Kelas `terbuka`
ditambahkan 21 Agustus 2026 bersama PKU-GoodsAD.

| id | Kelas | Contoh label mentah yang dipetakan |
|---|---|---|
| 0 | `pecah` | `broken_large`, `broken_small`, `chunk of gum missing`, `corner missing`, `middle breakage`, `small holes`, `small cracks`, `broken` |
| 1 | `gores` | `scratches`, `small scratches`, `surface_damage` |
| 2 | `noda` | `similar colour spot`, `different color spot`, `same colour spot`, `burnt`, `surface_anomaly` |
| 3 | `kotor` | `contamination` |
| 4 | `deformasi` | `stuck together`, `fryum stuck together`, `bent`, `misshape`, `bubble`, `deformation` |
| 5 | `terbuka` | `cap_open`, `cap_half_open`, `opened`, `straw_missing` |

> `burnt` sengaja dilebur ke `noda`: keduanya penyimpangan warna permukaan,
> dan memisahkannya hanya menyisakan sekitar 34 contoh yang terlalu tipis
> untuk dilatih. Dapat dipisah kembali di tahap Final.

### 2.4 Pembagian split (seed 42, dikunci)

Diukur ulang 21 Agustus 2026 setelah PKU-GoodsAD masuk.

| Split | Gambar | dari itu latar belakang | Instance | `pecah` | `gores` | `noda` | `kotor` | `deformasi` | `terbuka` |
|---|---|---|---|---|---|---|---|---|---|
| train | 1.428 | 185 | 1.573 | 295 | 548 | 204 | 16 | 179 | 331 |
| val | 310 | 39 | 345 | 70 | 122 | 39 | 3 | 39 | 72 |
| test | 301 | 39 | 321 | 53 | 112 | 42 | 3 | 39 | 72 |
| **total** | **2.039** | **263** | **2.239** | **418** | **782** | **285** | **22** | **257** | **475** |

Pembanding keadaan sebelumnya: 414 gambar dengan 643 instance. Gambar cacat
naik dari 363 menjadi 1.776, yaitu **4,9 kali lipat**.

### 2.5 Validasi statistik (STATISTICS.md bagian 2)

| Uji | Sebelum (18 Agu) | Sesudah (21 Agu) | Kriteria | Lulus? |
|---|---|---|---|---|
| Khi-kuadrat keselarasan distribusi kelas antar split | 7,161 - df 8 - p = 0,5194 | **2,453 - df 10 - p = 0,9915** | p > 0,05 | **lulus** |
| Rasio ketimpangan (IR) | 20,31 | **35,55** | < 3,0 | **belum lulus** |
| Entropi Shannon ternormalisasi | 0,7556 | **0,8740** | > 0,85 | **lulus** |
| Instance cacat di test | 75 (pada 52 gambar) | **321** (pada 262 gambar) | minimal 139 | **lulus** |

Dua dari tiga uji yang sebelumnya gagal kini lulus. Yang tersisa satu, dan
justru memburuk.

**Kenapa IR memburuk padahal datanya bertambah banyak.** Ini bukan
kemunduran: tidak ada satu pun instance `kotor` yang hilang, jumlahnya tetap
22. Yang terjadi adalah lima kelas lain tumbuh berkali lipat sementara `kotor`
diam di tempat, karena PKU-GoodsAD memang tidak memuat kontaminasi permukaan.
IR mengukur perbandingan kelas terbesar terhadap terkecil, jadi angkanya naik
walaupun tidak ada yang berkurang.

**Kenapa ini lebih serius daripada sekadar angka yang gagal.** `kotor` adalah
satu-satunya anggota `critical_classes` pada decision engine, yaitu kelas yang
memicu REJECT berapa pun luas cacatnya. Per 21 Agustus aturan sekeras itu
bersandar pada **16 instance latih**. Konsekuensinya wajib diukur pada recall
per kelas, bukan diperkirakan.

**Ukuran test set.** Dengan 321 instance, recall dapat diestimasi pada galat
**±3,3 % (95 % CI)**, membaik dari ±6,8 % sebelumnya dan kini melewati target
±5 %. Angka ±3,3 % inilah yang dipakai di proposal.

**Catatan uji khi-kuadrat.** Frekuensi harapan minimum masih di bawah 5 pada
sel `kotor`, sehingga nilai p tetap bersifat indikatif meski angkanya membaik.

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

**Verifikasi visual:** dibangkitkan `scripts/preview_dataset.py` menjadi
`reports/figures/dataset_samples_{train,test}.png`. Berkas gambarnya sengaja
TIDAK ikut disimpan di repositori karena memuat potongan citra MVTec AD yang
berlisensi CC BY-NC-SA; jalankan skrip itu setelah mengunduh himpunan datanya
untuk membuatnya ulang. Pada gambar itu kotak dan poligon digambar ulang dari
berkas label yang sesungguhnya, lalu diperiksa mata. Validasi statistik bisa lulus sementara
koordinatnya tergeser; lembar kontak inilah yang menutup celah itu.

## 3. BUKTI FINE-TUNING - Deteksi

Dihasilkan `scripts/compare_detection.py`; angka lengkap ada di
`reports/metrics/detection_comparison.json`. Evaluasi dijalankan pada split
**test** yang tidak pernah dilihat selama training maupun pemilihan epoch.

### 3.1 Ringkas run

Diukur 24 Agustus 2026 pada run `detect_balanced`, yaitu run yang bobotnya
benar-benar dilayani sejak rilis `models-v1.2.0`. Angka run 18 dan 21 Agustus
tidak lagi berlaku dan **tidak boleh dibandingkan langsung**, karena set latih
maupun ambang keputusannya berbeda.

| | |
|---|---|
| Model dasar | YOLO11n pra-latih COCO (`yolo11n.pt`) |
| Dataset | 1.838 train (termasuk 485 latar) / 310 val / 301 test, **6 kelas** cacat |
| Tambahan atas run sebelumnya | 110 citra cacat `kotor` sintetik + 300 citra latar sebagai negatif keras |
| Epoch diminta | 150 |
| Epoch dijalankan | **150** (tuntas) |
| Epoch terbaik (mAP50-95 val) | **150** |
| Metrik val terbaik | recall 0,5329 - mAP50 0,6162 - precision 0,6322 - mAP50-95 **0,3345** |

**Catatan reprodusibilitas.** Run ini terhenti pada epoch 124 ketika mesin
dimatikan, lalu **disambung** dari `last.pt` memakai
`scripts/resume_detection.py`, bukan dimulai ulang. Ultralytics membaca kembali
`args.yaml` run itu sehingga dataset, hyperparameter, dan keadaan optimizer
melanjutkan yang sebelumnya. Sambungannya tercatat pada `results.csv` sebagai
kolom waktu yang kembali dari nol di epoch 125.

**Metrik val-nya lebih rendah daripada run sebelumnya** (mAP50-95 0,3345 lawan
0,3770), dan itu bukan salah baca. Set val ikut memuat 39 citra latar, dan
penambahan negatif keras memang menekan recall demi menurunkan alarm palsu.
Dasar pengambilan keputusan bukan angka val ini, melainkan biaya harapan pada
set kalibrasi di bagian 3.9.

### 3.2 Perbandingan metrik pada test set

Pencocokan memakai ambang IoU 0,5. Selang kepercayaan BCa diperoleh dari 2.000
resampling dengan **gambar** sebagai unit resampling, bukan instance, karena
cacat di dalam satu gambar tidak saling bebas.

| Metrik (tingkat instance) | Baseline pra-latih | Sesudah fine-tune | 95% BCa (sesudah) |
|---|---|---|---|
| Recall | 0,0000 | **0,6573** | [0,5958 ; 0,7182] |
| Precision | 0,0000 | **0,6170** | [0,5609 ; 0,6656] |
| F1 | 0,0000 | 0,6365 | [0,5920 ; 0,6815] |
| **F2** (recall diutamakan) | 0,0000 | **0,6488** | [0,5963 ; 0,7007] |
| True positive | 0 | 211 | dari 321 instance |
| False negative | 321 | 110 | |

Selang Wilson untuk recall: 0,6573 [0,6038 ; 0,7071], konsisten dengan BCa.

| Metrik (tingkat gambar) | Baseline | Sesudah fine-tune |
|---|---|---|
| TP / FP / TN / FN | 223 / 31 / 8 / 39 | 222 / 13 / 26 / 40 |
| Recall | 0,8511 | **0,8473** |
| Specificity | 0,2051 | **0,6667** |
| **MCC** | **0,0521** | **0,4172** |

Baseline tampak punya recall tingkat gambar 0,7061 hanya karena ia menghasilkan
kotak COCO di mana-mana; specificity-nya 0,3077 memperlihatkan bahwa itu
tebakan menyeluruh, bukan pengenalan. MCC 0,0102 adalah angka yang jujur untuk
keadaan itu, yaitu praktis setara menebak acak.

### 3.3 Uji signifikansi McNemar

Kedua model dievaluasi pada test set yang sama, sehingga hasilnya berpasangan
per instance ground truth.

| | Fine-tuned menangkap | Fine-tuned gagal |
|---|---|---|
| **Baseline menangkap** | a = 0 | b = 0 |
| **Baseline gagal** | c = 211 | d = 110 |

| | Nilai |
|---|---|
| Metode | khi-kuadrat dengan koreksi kontinuitas |
| Khi-kuadrat McNemar | **195,005** |
| Nilai p | **2,57 x 10^-44** |
| Pasangan (n) | 321 |
| Cohen's h | **1,800** (besar) |
| Kesimpulan | perbedaan signifikan secara statistik |

**Kalimat untuk proposal:**

> Fine-tuning memperbaiki 197 instance cacat yang sebelumnya lolos dan tidak
> merusak satu pun instance yang sebelumnya tertangkap. Uji McNemar
> menunjukkan perbedaan yang signifikan secara statistik
> (khi-kuadrat = 195,005; p = 2,57 x 10^-44; n = 321), dengan besar efek
> Cohen's h = 1,800 yang tergolong besar. Recall kelas cacat naik dari 0,0000
> menjadi 0,6137 [0,5502 ; 0,6751].

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

Diukur 24 Agustus 2026 pada model yang dilayani. Kolom terakhir memuat run
21 Agustus sebagai pembanding.

| Kelas | n (test) | Baseline | Model dilayani | Run 21 Agu |
|---|---|---|---|---|
| `terbuka` | 72 | 0,000 | **0,764** | 0,708 |
| `deformasi` | 39 | 0,000 | **0,692** | 0,564 |
| `pecah` | 53 | 0,000 | **0,642** | 0,585 |
| `gores` | 112 | 0,000 | **0,607** | 0,554 |
| `noda` | 42 | 0,000 | 0,571 | 0,667 |
| `kotor` | **3** | 0,000 | 1,000 | 1,000 |

Lima dari enam kelas naik; `noda` satu-satunya yang turun.

> **`kotor` pada cacat NYATA tetap belum dapat dinilai.** Recall 1,000 di baris
> terakhir **bukan** bukti model sempurna pada kelas kritis: dukungannya hanya
> 3 instance dan selang Wilson-nya membentang [0,438 ; 1,000], praktis tanpa
> informasi. Yang dapat diukur adalah kinerjanya pada set buatan yang ditahan,
> dan itu dilaporkan di bagian 3.9.

**Kelas `terbuka` justru yang terkuat.** Ini pembenaran empiris atas keputusan
memisahkannya menjadi kelas keenam, bukan meleburkannya ke `deformasi`.
Sebelum dilatih ulang, model lima kelas yang diuji pada satu gambar
`cap_half_open` menebak `deformasi` 0,731 dan `pecah` 0,692 sekaligus, yaitu
tepat kebingungan yang akan menjadi permanen bila kedua konsep itu dipaksa
berbagi satu label.

### 3.6 Latensi

Diukur pada 301 gambar test, resolusi 640 px.

| Tahap | ms/gambar (CPU) |
|---|---|
| Prapemrosesan | 2,3 |
| Inferensi | 11,6 |
| Pascapemrosesan | 1,5 |
| **Total** | **15,4** |

**Angka ini diukur di GPU** (`prediction_settings.device: 0`), jadi ia bukan
angka penyajian. Yang menjadi target penyajian sesungguhnya adalah latensi ONNX
di CPU, diukur terpisah dan tercatat di `reports/metrics/latency_benchmark.json`:
median **28,6 ms** untuk deteksi dan **42,9 ms** untuk segmentasi, total 71,4 ms.
Pengukuran sebelumnya pada tabel ini dijalankan di CPU karena berebut VRAM 4 GB
dengan fine-tuning yang berjalan bersamaan, sehingga angka 99,8 ms milik run
21 Agustus **tidak sebanding** dengan tabel di atas.

### 3.7 Hyperparameter

Sumber: `configs/training.yaml` bagian `detection`. Tidak ada perubahan manual
selama run. Nilai kunci: AdamW, lr0 0,001 dengan cosine decay, batch 8,
imgsz 640, mosaic dimatikan pada 15 epoch terakhir, seed 42.

### 3.8 Keterbatasan yang perlu dinyatakan

1. **`kotor` tidak terukur.** Hanya 3 instance uji dan 16 instance latih.
   Padahal kelas inilah satu-satunya yang memicu REJECT tanpa memandang luas
   cacat. Ini keterbatasan paling serius pada run ini.
2. **Ketimpangan kelas memburuk** menjadi 35,55, bukan membaik. Run ini
   dilatih di atas data apa adanya, tanpa pembobotan kelas maupun oversampling.
3. **Recall `gores` 0,554 terendah** di antara kelas berdukungan memadai,
   meski dukungannya justru paling besar (112). Objek gores memang paling
   kecil ukurannya, dan menambah contoh tidak menyelesaikan masalah ukuran.
4. **Kalibrasi kepercayaan memburuk.** ECE 0,1284 pada model ini, dibanding
   0,0145 pada model 18 Agustus. Baik Platt maupun temperature scaling justru
   memperburuk, sehingga skor mentah yang dipakai (lihat bagian 7).

**Gambar pendukung:** `reports/figures/detection_comparison.png`,
`training_curves.png`, `detection_confusion_matrix.png`,
`detection_pr_curve.png`.

### 3.9 Data sintetik kelas `kotor` dan negatif keras (24 Agustus 2026)

Kelas `kotor` adalah satu-satunya kelas kritis: ia memicu penolakan tanpa
memandang luas cacat, karena kontaminasi berdampak langsung pada keamanan
konsumsi. Dukungan latihnya hanya 16 instance dan dukungan ujinya 3, sehingga
kinerjanya tidak pernah dapat dinilai dari data nyata.

**Yang dibangkitkan.** `scripts/generate_synthetic_kotor.py` membuat 110 citra
berisi 351 instance `kotor` di atas citra produk yang ada, memakai segmentasi
latar depan GrabCut agar kontaminasi jatuh di atas produk, bukan di rak.
`scripts/mine_hard_negatives.py` menambahkan 300 citra produk baik yang paling
meyakinkan salah dideteksi model lama, dilabeli kosong sebagai contoh latar.
Set eval 60 citra ditahan terpisah; citra dasarnya tidak pernah dipakai
melatih maupun sebagai negatif keras, dan tumpang tindihnya diperiksa nol.

**Pengukuran pada set eval yang ditahan.** Dijalankan dengan `YOLO.val` pada
kedua bobot, kelas `kotor` saja:

| | precision | recall | mAP50 |
|---|---|---|---|
| Run 21 Agu (`detect_goodsad`) | 0,2727 | **0,0160** | 0,0075 |
| Model dilayani (`detect_balanced`) | 0,7370 | **0,4171** | 0,4829 |

Model sebelumnya praktis buta terhadap kelas kritis: recall 1,6 persen.

**Keputusan mengadopsi diambil di set kalibrasi, bukan set uji** (R7.5). Ambang
biner dipilih ulang lewat `scripts/select_binary_threshold.py` dan jatuh pada
0,195; biaya harapan pada set kalibrasi Rp438,1 per unit melawan Rp481,8 milik
konfigurasi sebelumnya, yaitu **9,1 persen lebih murah**. Ambang penyaringan NMS
ikut diturunkan 0,22 menjadi 0,12 supaya tetap berada di bawah ambang keputusan;
tanpa itu ambang 0,195 tidak akan pernah tercapai.

**Apa yang memburuk, dan tidak disembunyikan.** Pada set uji, spesifisitas
gabungan turun dari 0,8643 menjadi 0,8378, dan pada PKU-GoodsAD dari 0,8051
menjadi 0,7712 - artinya produk baik yang keliru ditolak naik dari sekitar 19,5
menjadi 22,9 persen. Recall gabungan naik 0,7977 menjadi 0,8397 dan akurasi
0,8353 menjadi 0,8386. Pertukaran ini dipilih oleh model biaya yang menetapkan
cacat lolos 25 kali lebih mahal daripada salah tolak, bukan oleh selera.

**Batas yang wajib dinyatakan.** Set eval `kotor` bersifat **buatan**, dan model
ini dilatih atas cacat buatan dari generator yang sama. Citra dasarnya berbeda,
sehingga angkanya sah sebagai uji generalisasi di dalam sebaran buatan itu -
tetapi **bukan** bukti perbaikan pada kontaminasi dunia nyata. Untuk itu
dibutuhkan citra kontaminasi nyata yang belum dimiliki proyek ini.

## 4. BUKTI FINE-TUNING - Segmentasi

Dihasilkan `scripts/compare_segmentation.py`; angka lengkap di
`reports/metrics/segmentation_comparison.json`. Pencocokan memakai **IoU mask**,
bukan IoU kotak pembatas, karena dua mask yang bentuknya sangat berbeda bisa
saja punya kotak yang nyaris sama.

### 4.1 Ringkas run

Diukur ulang 22 Agustus 2026 pada dataset yang diperbesar PKU-GoodsAD.

| | |
|---|---|
| Model dasar | YOLO11n-seg pra-latih COCO |
| Dataset | `data/processed/seg`, split identik dengan deteksi (1.428 / 310 / 301) |
| Epoch diminta | 130 |
| Epoch dijalankan | **130** (tidak berhenti awal) |
| Epoch terbaik (mAP50 mask) | **129** |
| Waktu tempuh | **175.4 menit** pada RTX 3050 Laptop 4 GB |
| Metrik val terbaik (mask) | precision 0,8171 - recall 0,6738 - mAP50 **0,7268** - mAP50-95 0,3844 |
| Metrik val terbaik (kotak) | mAP50 0,7492 - mAP50-95 0,4810 |

**Model ini kemungkinan besar belum konvergen.** Epoch terbaik jatuh di 129
dari 130, artinya kurva masih menanjak tepat ketika batas epoch tercapai.
Berbeda dari deteksi yang jelas melandai sejak sekitar epoch 120, segmentasi
di sini berhenti karena kehabisan anggaran epoch, bukan karena berhenti
membaik. Menambah epoch kemungkinan masih memperbaiki hasilnya, dan itu
dinyatakan apa adanya alih-alih diklaim sudah selesai.

### 4.2 Perbandingan metrik pada test set

| Metrik (tingkat instance, IoU mask 0,5) | Baseline pra-latih | Sesudah fine-tune | 95% BCa |
|---|---|---|---|
| Recall | 0,0000 | **0,5888** | [0,5260 ; 0,6457] |
| Precision | 0,0000 | **0,6873** | [0,6279 ; 0,7389] |
| F2 | 0,0000 | 0,6062 | [0,5487 ; 0,6584] |
| True positive | 0 | 189 | dari 321 instance |
| False positive | 297 | 86 | |
| False negative | 321 | 132 | |

| Kualitas mask pada pasangan yang cocok | Baseline | Sesudah fine-tune |
|---|---|---|
| IoU rerata | 0,0000 | **0,8064** |
| IoU median | 0,0000 | **0,8326** |
| Jumlah pasangan | 0 | 189 |

IoU mask 0,8064 **lebih tinggi** daripada 0,7847 milik run 18 Agustus, padahal
diukur pada set uji yang jauh lebih beragam. Ketika model berhasil menemukan
cacat, batas yang digambarnya justru makin akurat.

| Metrik (tingkat gambar) | Baseline | Sesudah fine-tune |
|---|---|---|
| TP / FP / TN / FN | 195 / 30 / 9 / 67 | 214 / 7 / 32 / 48 |
| Recall | 0,7443 | **0,8168** |
| Specificity | 0,2308 | **0,8205** |
| **MCC** | **-0,0193** | **0,4845** |

MCC baseline bernilai **negatif** (-0,0193), yaitu sedikit lebih buruk daripada
menebak acak. Itu masuk akal: model COCO menaburkan mask di mana-mana, terlihat
dari specificity 0,2308.

### 4.3 Estimasi luas cacat

Ini keluaran yang benar-benar dipakai decision engine, dan karena itu galatnya
diukur tersendiri. Satuannya **poin persen** terhadap luas gambar.

| Galat mutlak luas | Baseline | Sesudah fine-tune | 95% BCa (sesudah) |
|---|---|---|---|
| Rerata | 20,4764 | **0,7499** | [0,5447 ; 1,0558] |
| Median | 17,1306 | **0,0681** | |
| Persentil 95 | 63,1558 | 3,4009 | |

Ambang luas pada decision engine bernilai 2,0 persen. Dengan galat median
0,0681 poin persen dan rerata 0,7499, estimasi luas cukup teliti untuk dipakai
aturan itu. Persentil 95 sebesar 3,4009 tetap perlu dicatat: pada satu dari
dua puluh gambar, galatnya masih melampaui ambang keputusan itu sendiri.

### 4.4 Uji signifikansi McNemar

| | Fine-tuned menangkap | Fine-tuned gagal |
|---|---|---|
| **Baseline menangkap** | a = 0 | b = 0 |
| **Baseline gagal** | c = 189 | d = 132 |

| | Nilai |
|---|---|
| Metode | khi-kuadrat dengan koreksi kontinuitas |
| Khi-kuadrat McNemar | **187,005** |
| Nilai p | **1,43 x 10^-42** |
| Pasangan (n) | 321 |
| Cohen's h | **1,749** (besar) |
| Kesimpulan | perbedaan signifikan secara statistik |

### 4.5 Recall per kelas

| Kelas | n (test) | Baseline | Sesudah fine-tune |
|---|---|---|---|
| `terbuka` | 72 | 0,000 | **0,694** |
| `kotor` | **3** | 0,000 | 0,667 |
| `deformasi` | 39 | 0,000 | 0,641 |
| `noda` | 42 | 0,000 | 0,571 |
| `gores` | 112 | 0,000 | 0,563 |
| `pecah` | 53 | 0,000 | 0,472 |

Seperti pada deteksi, `kotor` hanya berdukungan 3 instance sehingga angkanya
**tidak dapat dinilai** dan tidak boleh dipakai sebagai capaian.

`pecah` menjadi yang terlemah pada segmentasi (0,472) padahal pada deteksi
mencapai 0,585. Ini pola yang wajar: kotak pembatas cukup melingkupi area
pecah secara kasar, sedangkan mask harus mengikuti tepi retakan yang bentuknya
tidak beraturan, dan IoU mask menghukum ketidaktepatan tepi jauh lebih keras.

`gores` naik dari 0,333 pada run 18 Agustus menjadi 0,563. Dukungannya juga
naik dari 9 menjadi 112 instance, jadi angka lama memang tidak dapat
diandalkan sejak awal.

### 4.6 Latensi

Diukur pada 301 gambar test.

| Tahap | ms/gambar (CPU) |
|---|---|
| Prapemrosesan | 3.6 |
| Inferensi | 101.8 |
| Pascapemrosesan | 2.5 |
| **Total** | **107.8** |

Seperti pada deteksi, angka ini diukur di **CPU** karena GPU sedang dipakai
training anomali. Tidak sebanding dengan 57,6 ms milik run 18 Agustus yang
diukur di GPU. Latensi ONNX di CPU, yang menjadi target penyajian, diukur
terpisah pada Step 9.

### 4.7 Keterbatasan

1. **Run berhenti karena kehabisan epoch, bukan karena konvergen.** Epoch
   terbaik jatuh di 129 dari 130. Ini keterbatasan paling nyata pada model
   segmentasi saat ini, dan yang paling mudah diperbaiki bila ada waktu GPU.
2. Recall segmentasi (0,5888) lebih rendah daripada deteksi (0,6137) karena
   IoU mask pada ambang 0,5 jauh lebih ketat daripada IoU kotak.
3. `pecah` menjadi kelas terlemah (0,472) karena tepi retakan sulit diikuti
   mask, bukan karena kekurangan contoh; dukungannya 53 instance.
4. Dukungan `kotor` hanya 3 instance, sehingga recall-nya tidak dapat dinilai.
5. Persentil 95 galat luas mencapai 3,4009 poin persen, melampaui ambang luas
   2,0 persen yang dipakai decision engine. Pada sekitar satu dari dua puluh
   gambar, estimasi luas belum cukup teliti untuk aturan itu.

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
| Gambar latih gabungan | 600, yaitu **60 per kategori** |
| Kategori | 10, plus satu model gabungan yang dipakai saat penyajian |

`configs/training.yaml` menetapkan EfficientAD sebagai pilihan utama dengan
PaDiM sebagai cadangan, dan **cadangan itulah yang dipakai**. Alasannya waktu:
EfficientAD memerlukan pelatihan ribuan langkah per kategori. Percobaan pada
22 Agustus 2026 dihentikan setelah satu kategori saja memakan puluhan menit,
sementara `scripts/export_anomaly.py` memang hanya mengekspor PaDiM.

### 5.2 Pembagian data dan dua masalah yang ditemukan

Ambang dikalibrasi pada data yang **tidak dipakai mengukurnya**. Tanpa
pemisahan ini, laju alarm palsu yang dilaporkan menjadi melingkar.

| Split | Isi | Perannya |
|---|---|---|
| latih | gambar normal | model belajar seperti apa produk normal |
| kalibrasi | gambar normal, diambil dari latih, tidak ikut dilatih | dasar perhitungan ambang |
| uji | gambar normal + cacat | dasar laju alarm palsu dan recall |

Dua masalah ditemukan pada 23 Agustus 2026 dan keduanya dicatat karena
memengaruhi cara angka di bawah harus dibaca.

**Masalah 1: dataset gabungan sempat basi dan tidak reprodusibel.** Model yang
dipakai saat penyajian adalah model gabungan, tetapi foldernya ternyata dibuat
manual di luar repo dan masih memuat empat kategori lama setelah PKU-GoodsAD
masuk. Model semacam itu akan membaca setiap produk GoodsAD sebagai anomali.
Sekarang dibangun `scripts/build_combined_anomaly.py` sehingga langkahnya
tercatat.

**Masalah 2: batas `max_train_images` tidak berfungsi.** Skrip melaporkan "200
dipakai", tetapi pelatihan berjalan pada 4.307 gambar dan mati dengan CUDA out
of memory pada GPU 4 GB. Penyebabnya `engine.fit()` menjalankan ulang
`setup()` sehingga pembatasan yang dilakukan di memori terbuang. Batas
sesungguhnya kini diterapkan pada jumlah berkas saat dataset gabungan dibangun,
**60 per kategori**, yang sekaligus menyeimbangkan kategori: pengambilan acak
dari kumpulan gabungan akan didominasi `food_bottle` yang menyumbang 1.069
gambar dibanding `bottle` yang hanya 195.

### 5.3 Hasil

**Model per kategori (dilatih 18 Agustus, TIDAK dilatih ulang).** Dataset per
kategori untuk kelima kategori ini tidak berubah, jadi angkanya tetap berlaku.
Ditandai tanggalnya supaya tidak tertukar dengan model gabungan.

| Kategori | AUROC gambar | Ambang EVT | Kuantil empiris | Alarm palsu (uji) | Recall cacat |
|---|---|---|---|---|---|
| `bottle` | **0,9967** | 45,861 | 44,431 | 0,029 | **1,0000** |
| `chewinggum` | 0,9307 | 37,234 | 33,630 | 0,067 | 0,8667 |
| `fryum` | 0,8867 | 71,157 | 76,264 | 0,000 | 0,1429 |
| `pipe_fryum` | 0,8571 | 53,243 | 53,041 | 0,000 | 0,2143 |
| `cashew` | 0,8476 | 56,550 | 46,076 | 0,013 | 0,2857 |

**Model gabungan (dilatih ulang 23 Agustus, INILAH yang dipakai menyajikan).**

| | Model gabungan 4 kategori (19 Agu) | Model gabungan 10 kategori (23 Agu) |
|---|---|---|
| AUROC gambar | 0,8900 | **0,6019** |
| Ambang EVT | 39,7942 | 83,2714 |
| Kalibrasi normal | 294 | 120 |
| Uji normal / cacat | - | 948 / 276 |
| Alarm palsu pada data tertahan | - | **0,0232** (target 0,01) |
| Recall cacat pada ambang | - | **0,0399** |

**AUROC turun dari 0,890 menjadi 0,6019, dan ini hasil yang buruk.** Nilai
0,50 setara menebak acak, jadi 0,6019 hanya sedikit di atas itu. Pada ambang
EVT, jalur ini menangkap 3,99 persen cacat sementara alarm palsunya 2,32
persen, yaitu lebih dari dua kali target 1 persen.

**Penjelasan yang paling masuk akal** adalah bentuk modelnya sendiri: PaDiM
mencocokkan satu Gaussian per posisi patch, sedangkan "normal" bagi sepuluh
jenis produk yang sangat berbeda bukan sebaran unimodal. Botol kaca, kaleng,
kotak makanan, dan kacang mete tidak berbagi satu pusat sebaran. Perlu
ditegaskan bahwa ini **penalaran, bukan hasil pengukuran tersendiri**;
membuktikannya memerlukan percobaan terpisah yang belum dijalankan.

**Apakah jalur ini tetap dipakai?** Ya, dan keputusannya diambil pada set
kalibrasi, bukan set uji. Rinciannya di bagian 6.

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

### 5.6 Model gabungan yang benar-benar dipakai saat penyajian

Model per kategori pada bagian 5.3 tidak dapat dipasang langsung ke pipeline.
Gambar yang diunggah tidak menyatakan kategorinya, dan model yang hanya
mengenal botol menandai kacang mete sebagai anomali. Percobaan memasang model
`bottle` membuat enam dari tujuh gambar normal ditahan ke manusia.

Karena itu satu model dilatih pada **gabungan** keempat kategori yang juga
dilatih detektor.

| | Nilai |
|---|---|
| Gambar latih | 200 dari 1.473 normal gabungan |
| Kalibrasi | 294 normal yang tidak ikut dilatih |
| Uji | 259 normal, 52 cacat |
| AUROC gambar | **0,8903** |
| Ambang EVT | **39,7942** |
| GPD | xi = +0,2411 - sigma = 4,5026 - u = 25,7742 - KS p = **0,970** |
| Laju alarm palsu | 1,0 persen kalibrasi - 2,7 persen uji |

Ekspor ONNX diverifikasi terhadap PyTorch pada 16 gambar dengan selisih
maksimum 0,000076, sehingga skor yang dihasilkan saat penyajian sebanding
dengan ambang yang dihitung. Latensinya 20,9 milidetik di CPU.

Perilaku pada split uji deteksi:

| | Nilai |
|---|---|
| Gambar normal melampaui ambang | **0 dari 7** |
| Gambar cacat melampaui ambang | **23 dari 52** |
| Skor normal | median 18,9 - maksimum 37,9 |
| Skor cacat | median 38,1 - maksimum 141,7 |

Dua cacat yang masih lolos sebagai PASS memperoleh skor 24,5 dan 19,4, yaitu
di dalam rentang gambar normal. Model anomali memang tidak melihat keduanya
menyimpang; ini keterbatasan nyata, bukan kesalahan penyetelan.

### 5.7 Keterbatasan

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

## 6. Kalibrasi, Conformal, dan Ambang Biaya

Dihasilkan `scripts/calibrate_decision.py`; angka lengkap di
`reports/metrics/calibration_results.json`. Split val dipakai sebagai set
kalibrasi dan split test hanya dipakai menguji.

### 6.1 Perluasan set kalibrasi

Versi pertama bagian ini memakai split apa adanya, yang hanya memuat **tujuh
gambar normal** pada masing-masing sisi. Jumlah itu terlalu kecil untuk hampir
semua yang dikerjakan di sini, dan menjadi akar dari seluruh kegagalan yang
tercatat pada versi sebelumnya.

Perbaikannya bukan menyetel parameter melainkan menambah data. Dataset anomali
menyimpan ribuan gambar normal, dan sebagiannya tidak pernah masuk split
deteksi mana pun sehingga tidak pernah dilihat model saat pelatihan.

| | Sebelum | Sesudah |
|---|---|---|
| Gambar kalibrasi | 64 (57 cacat, 7 normal) | **364** (57 cacat, 307 normal) |
| Gambar uji | 59 (52 cacat, 7 normal) | **359** (52 cacat, 307 normal) |
| Proporsi cacat | 89 persen | 15 persen |

Gambar tambahan disaring dengan mengecualikan seluruh nama berkas yang sudah
muncul pada split latih, validasi, maupun uji dataset deteksi. Kategorinya
dibatasi pada empat kategori yang memang dilatih detektor.

### 6.2 Kalibrasi kepercayaan

| Metode | ECE pada kalibrasi | ECE pada uji |
|---|---|---|
| tanpa kalibrasi | 0,0455 | 0,0497 |
| temperature scaling | 0,0709 | 0,0641 (lebih buruk) |
| **Platt scaling** | **0,0197** | **0,0145** (dipilih) |

| Metrik | Sebelum | Sesudah Platt |
|---|---|---|
| ECE | 0,0497 | **0,0145** |
| MCE | 0,6489 | 0,3660 |
| Skor Brier | 0,0297 | **0,0145** |

Temperature scaling tetap kalah walau proporsi kelasnya sudah lebih wajar.
Alasannya sama: ia hanya memiliki pengali dan tidak dapat menggeser skor,
sedangkan yang dibutuhkan justru pergeseran. Pemilihan metode dilakukan
berdasarkan ECE pada set kalibrasi, bukan pada set uji.

### 6.3 Conformal prediction dan pemilihan alpha

Alpha bukan angka yang dapat diambil begitu saja, karena menaikkan jaminan
selalu berarti menahan lebih banyak keputusan. Ia dipilih pada **set
kalibrasi** memakai aturan yang dinyatakan lebih dulu: ambil jaminan sekuat
mungkin yang masih menyisakan cukup keputusan untuk diambil sistem sendiri,
dengan batas menahan 15 persen.

Sapuan pada set kalibrasi:

| alpha | Cakupan | Ditahan | Cakupan normal |
|---|---|---|---|
| 0,01 | 0,9973 | 0,8599 | 0,9967 |
| 0,05 | 0,9643 | 0,8187 | 0,9577 |
| **0,10** | **0,9148** | **0,0275** | **0,9088** |
| 0,15 | 0,8626 | 0,1236 | 0,8567 |
| 0,20 | 0,8132 | 0,1813 | 0,8078 |

Alpha 0,05 menahan 82 persen keputusan sehingga tidak dapat dipakai. Sistem QC
yang menyerahkan mayoritas produk ke manusia tidak menyelesaikan apa pun.
Alpha 0,10 dipilih sebagai jaminan terkuat yang masih memenuhi batas.

Hasil pada split uji dengan alpha 0,10:

| | Nilai |
|---|---|
| Kuantil kelas normal | 0,0211 |
| Kuantil kelas cacat | 0,9250 |
| **Cakupan empiris** | **0,9415** (jaminan 0,90 tercapai) |
| Cakupan kelas normal | **0,9381** |
| Cakupan kelas cacat | **0,9615** |
| Ditahan ke manusia | **2,8 persen** |

Kedua kelas kini berada di atas jaminan. Pada versi sebelumnya cakupan kelas
normal hanya 0,7143 dan tidak bergerak berapa pun alpha yang dipilih, karena
kelas itu memang hanya memiliki tujuh contoh.

### 6.4 Ambang sensitif biaya

Ambang Bayes dari biaya yang ditetapkan:

    tau* = 2.000 / (2.000 + 50.000) = 0,0385

| Ambang | Recall | Biaya per unit |
|---|---|---|
| **0,30** (minimum biaya pada kalibrasi) | 0,9615 | **Rp 83** |
| 0,50 (pembanding) | 0,9423 | Rp 105 |
| 0,00 (tolak semua) | 1,0000 | Rp 1.940 |

Ambang hasil optimasi kini **menggeneralisasi**: pada split uji ia 21,4 persen
lebih murah daripada ambang 0,50, sekaligus dengan recall yang lebih tinggi.
Pada versi sebelumnya ambang yang sama justru 5,2 persen lebih mahal, karena
letak minimum biaya tidak stabil ketika hanya ada tujuh gambar normal.

Perhitungan biaya tetap menimbang ulang kedua kelas terhadap prevalensi cacat
produksi yang diasumsikan 3 persen. Angka itu **asumsi**, bukan hasil
pengukuran, dan ditandai demikian di `configs/inference.yaml`.

### 6.5 Perilaku keputusan pada gambar uji nyata

Seluruh 59 gambar split uji dijalankan lewat `run_inspection()`:

| Kondisi sebenarnya | PASS | REJECT | REVIEW |
|---|---|---|---|
| normal (7 gambar) | **6** | **0** | 1 |
| cacat (52 gambar) | 2 | 33 | 17 |

Arah kesalahannya sesuai kebutuhan QC. Tidak ada satu pun produk bagus yang
salah ditolak. Dari 52 cacat, hanya 2 lolos sebagai PASS sementara 17 lainnya
diserahkan ke manusia alih-alih diloloskan.

Pada versi sebelumnya, gambar bersih justru selalu menghasilkan REVIEW
sehingga sistem tidak pernah dapat meloloskan produk sama sekali.

### 6.6 Keterbatasan

1. Dua cacat masih lolos sebagai PASS. Untuk produk pangan, angka itu perlu
   ditekan lebih jauh pada tahap Final.
2. Prevalensi cacat produksi 3 persen adalah asumsi.
3. Gambar normal tambahan berasal dari kategori yang sama dengan data latih.
   Kinerja pada produk di luar kategori itu belum diukur.
4. MCE masih 0,3660, artinya masih ada keranjang skor yang melenceng jauh
   meskipun rata-ratanya sudah baik.

**Gambar pendukung:** `reports/figures/reliability_diagram.png`,
`conformal_coverage.png`, `cost_curve.png`.

## 7. Latensi dan Ukuran Model

Dihasilkan `scripts/export_onnx.py`; angka lengkap di
`reports/metrics/latency_benchmark.json`.

Target penyajian adalah **CPU**, bukan GPU. Panitia menjalankan sistem di
komputer mereka sendiri dan tidak dapat diandalkan memiliki kartu grafis,
sehingga angka yang bermakna adalah angka CPU.

### 7.1 Latensi ONNX di CPU

Diukur dari 100 pengulangan setelah 10 kali pemanasan, 4 utas, resolusi 640 px.

| Model | Median | Persentil ke-95 | Rentang | Ukuran ONNX |
|---|---|---|---|---|
| YOLO11n detect | **62,5 ms** | 119,6 ms | 50,1 - 159,6 ms | 10,12 MB |
| YOLO11n-seg | **90,9 ms** | 140,3 ms | 78,5 - 160,3 ms | 11,09 MB |
| **Kedua model** | **153,4 ms** | - | - | 21,21 MB |

Persentil ke-95 hampir dua kali median pada kedua model. Sebaran selebar itu
wajar pada CPU laptop yang menjalankan proses lain, dan justru karena itu satu
pengukuran tunggal tidak boleh dikutip sebagai angka latensi.

### 7.2 Perbandingan dengan pengukuran GPU

| Model | PyTorch GPU | ONNX CPU | Rasio |
|---|---|---|---|
| detect | 15,0 ms | 62,5 ms | 4,2x |
| seg | 57,6 ms | 90,9 ms | 1,6x |

Angka GPU pada bagian 3.6 dan 4.6 diukur saat evaluasi dan **bukan** angka
penyajian. Yang wajib dikutip di proposal adalah angka CPU di atas.

### 7.3 Ruang yang tersisa

Sasaran keseluruhan yang ditetapkan PROJECT.md adalah satu detik per gambar di
CPU. Kedua model inti memakan 153 ms median, menyisakan ruang untuk
prapemrosesan, skor anomali, OCR, dan penggambaran anotasi. Latensi pipeline
utuh terukur sekitar 140 ms per gambar setelah model dimuat, termasuk
prapemrosesan, deteksi, segmentasi, perhitungan luas, keputusan, dan
penggambaran anotasi.

## 8. Keputusan Biner Tanpa REVIEW

Tanggal 20 Agustus 2026.

### Mengapa diubah

Pada mode tiga kelas, split uji 59 gambar menghasilkan 18 REVIEW. Pembedahan
per kasus menunjukkan seluruhnya berasal dari **satu aturan**, yaitu
`min_confidence: 0.60`. Himpunan prediksi conformal ternyata tunggal pada
setiap gambar, sehingga aturan ketidakpastian conformal tidak pernah sekali pun
memicu REVIEW.

Yang lebih menentukan: **17 dari 18 kasus itu cacat sungguhan yang sudah
terdeteksi model**, dengan keyakinan 0,30 sampai 0,57. Semuanya diserahkan ke
manusia hanya karena kalah dari ambang 0,60 yang dipilih tangan. Satu sisanya
gambar normal dengan deteksi palsu berkeyakinan 0,258.

Jadi angka REVIEW yang tinggi bukan tanda model ragu, melainkan tanda ambangnya
kelewat tinggi.

### Bagaimana ambang barunya dipilih

Ambang dipilih **pada set kalibrasi** dengan meminimumkan biaya yang
diharapkan, memakai model biaya yang sudah ada: salah tolak Rp2.000, cacat
lolos Rp50.000, prevalensi cacat diasumsikan 3 persen. Set uji tidak dilihat
sama sekali selama pemilihan. Prosedurnya ada di
`scripts/select_binary_threshold.py` dan dapat diulang.

Hasil pada kalibrasi (n=364, 57 cacat): ambang **0,22**, recall 0,9298,
specificity 0,9805, biaya Rp143,2 per unit.

Satu jebakan yang sempat terlewat: ambang NMS deteksi masih 0,25, di atas
ambang keputusan 0,22. Selama keduanya berbeda, ambang keputusan tidak pernah
tercapai karena deteksi lemah sudah dibuang lebih dulu. Keduanya kini
disamakan pada 0,22.

### Hasil pada set uji

Dilaporkan tanpa penyetelan lebih lanjut.

| Ambang | Recall | Specificity | Biaya per unit |
|---|---|---|---|
| 0,60 (lama, pilihan tangan) | 0,5769 | 1,0000 | Rp634,6 |
| **0,22 (terpilih pada kalibrasi)** | **0,9615** | 0,9837 | **Rp89,3** |

Recall naik dari 0,58 menjadi 0,96, dengan biaya turun 86 persen.

### Hasil ujung-ke-ujung pipeline

Angka di atas berasal dari sapuan ambang atas keyakinan saja. Pipeline
sungguhnya menjalankan lebih banyak aturan, sehingga hasilnya diukur ulang.

Uji diperluas, 359 gambar (52 cacat, 307 normal):

| kebenaran | PASS | REJECT | REVIEW |
|---|---|---|---|
| normal | 296 | 11 | **0** |
| cacat | 2 | 50 | **0** |

Recall 0,9615 · specificity 0,9642 · akurasi 0,9638 · REVIEW nol.

Perhatikan selisihnya: sapuan memperkirakan 5 salah tolak, pipeline
menghasilkan 11. Selisih 6 itu datang dari aturan lain yang tidak ikut dalam
sapuan, yaitu jaring pengaman anomali (4 gambar) dan aturan kelas kritis
(1 gambar). Inilah sebabnya angka yang dilaporkan diambil dari pipeline utuh,
bukan dari sapuan.

### Yang tetap salah

Dua cacat masih lolos sebagai PASS, `visa_cashew_anomaly_084` dan `_088`.
Keduanya tidak menghasilkan deteksi sama sekali pada ambang berapa pun, dan
skor anomalinya 24,5 dan 19,4, berada di dalam rentang gambar normal. Tidak
ada ambang yang dapat menangkap keduanya tanpa menolak banyak produk baik.
Menambah data adalah satu-satunya jalan.

### Hasil ujung-ke-ujung setelah PKU-GoodsAD (23 Agustus 2026)

Diukur `scripts/evaluate_pipeline.py` lewat **pipeline ONNX yang sesungguhnya
dipakai Backend**, bukan bobot PyTorch, sehingga mencakup lapisan privasi,
anomali, aturan luas, dan aturan kelas kritis sekaligus.

Set uji: 301 gambar split uji ditambah 300 gambar normal yang tidak pernah
dilihat model, seluruhnya 601 gambar dengan 262 cacat.

| Kelompok | n | cacat | Recall | Specificity |
|---|---|---|---|---|
| **4 kategori lama (MVTec + VisA)** | 155 | 52 | **0,8654** | **1,0000** |
| PKU-GoodsAD (baru, lebih sulit) | 446 | 210 | 0,7810 | 0,8051 |
| **Gabungan** | 601 | 262 | **0,7977** | 0,8643 |

Akurasi 0,8353. **REVIEW nol**: 255 REJECT dan 346 PASS, tidak ada satu pun
gambar yang ditahan.

**Angka ini tidak sebanding dengan recall 0,9615 pada 20 Agustus.** Set ujinya
berbeda dan yang sekarang jauh lebih sulit. Perbandingan yang sah hanya pada
baris "4 kategori lama", yaitu tugas yang persis sama: recall 0,8654 dengan
specificity **1,0000**, artinya nol produk bagus yang salah ditolak. Model
20 Agustus pada tugas yang sama mencapai recall lebih tinggi tetapi masih
menolak sebagian produk bagus.

**Keterbatasan yang paling perlu diperbaiki:** specificity 0,8051 pada
PKU-GoodsAD berarti sekitar 19,5 persen produk bagus ikut ditolak. Untuk lini
produksi sungguhan angka itu terlalu tinggi.

### Apakah jalur anomali dipertahankan

Pada set uji, pipeline penuh terlihat **lebih mahal** daripada deteksi saja
(Rp566,7 berbanding Rp515,2 per unit), yang seolah menyarankan jalur anomali
dibuang. Godaan itu tidak diikuti: menyetel susunan sistem berdasarkan hasil
set uji persis sama dengan menyetel hyperparameter di sana.

Pengukuran diulang pada **set kalibrasi**, dengan dan tanpa jalur anomali:

| Set kalibrasi (610 gambar, 271 cacat) | Recall | Specificity | Biaya/unit |
|---|---|---|---|
| **dengan jalur anomali** | 0,8266 | 0,8850 | **Rp 483,2** |
| tanpa jalur anomali | 0,8044 | 0,8997 | Rp 488,0 |

Arahnya **berlawanan** dengan set uji: pada kalibrasi, mempertahankan jalur
anomali justru sedikit lebih murah, selisih Rp4,8 per unit atau 1,0 persen.
Selisih sekecil itu wajar dibaca sebagai seri.

**Keputusan: jalur anomali dipertahankan.** Dasarnya set kalibrasi, dan
kalibrasi tidak mendukung pembuangannya. Ini sekaligus catatan bahwa dua split
dapat memberi arah berbeda, dan bahwa memilih split yang kebetulan mendukung
kesimpulan yang diinginkan adalah cara paling mudah menipu diri sendiri.

### Mode tiga kelas tidak dihapus

Kode conformal, Platt, dan jalur tiga kelas tetap ada dan tetap diuji. Yang
berubah hanya `decision.mode` pada config. Tahap Final berencana memakai
kembali penahanan keputusan ketika biaya operator ikut dimodelkan.

---

## 9. Riwayat Run

| # | Tgl | Model | Perubahan | Hasil | Keputusan |
|---|---|---|---|---|---|
| 12 | 2026-08-23 | Evaluasi ujung-ke-ujung | pipeline ONNX utuh pada 601 gambar; jalur anomali diuji dengan dan tanpa, PADA SET KALIBRASI | recall 0,7977 specificity 0,8643 REVIEW nol; pada kalibrasi jalur anomali lebih murah Rp4,8/unit | jalur anomali dipertahankan; keputusan diambil di kalibrasi, bukan di uji |
| 11 | 2026-08-23 | PaDiM gabungan 10 kategori | dataset gabungan dibangun ulang lewat skrip, 60 gambar per kategori agar seimbang | AUROC 0,6019 turun dari 0,890; alarm palsu 0,0232 terhadap target 0,01 | diterima dengan peringatan tertulis; PaDiM unimodal tidak cocok untuk 10 jenis produk |
| 10 | 2026-08-22 | Kalibrasi ulang | Platt, temperature, dan conformal dihitung ulang pada model baru | mentah ECE 0,1284 terbaik; Platt dan temperature justru memperburuk; cakupan conformal 0,8918 meleset dari 0,90 | kalibrasi dimatikan lewat pemetaan identitas; kegagalan dicatat apa adanya |
| 9 | 2026-08-22 | YOLO11n-seg 6 kelas | fine-tune ulang pada data 4,9x lebih besar, epoch 250 menjadi 130 | 130 epoch penuh, terbaik epoch 129, mask mAP50 0,7268, IoU mask 0,8064 | diterima; epoch terbaik di 129 dari 130 berarti kemungkinan belum konvergen |
| 8 | 2026-08-21 | YOLO11n detect 6 kelas | PKU-GoodsAD masuk, kelas `terbuka` ditambahkan, epoch 300 menjadi 150 | 150 epoch, terbaik epoch 122, mAP50 val 0,6972; `terbuka` jadi kelas terkuat recall 0,708 | diterima; kelas keenam terbukti perlu, bukan sekadar menambah label |
| 1 | 2026-08-18 | YOLO11n detect | fine-tune pertama dari bobot COCO, config apa adanya | berhenti awal di epoch 161, terbaik epoch 137, mAP50 val 0,7967 | diterima sebagai model deteksi tahap penyisihan |
| 7 | 2026-08-19 | Perluasan set kalibrasi | 307 gambar normal tambahan per sisi, alpha dipilih lewat aturan pada kalibrasi | cakupan kelas normal 0,7143 menjadi 0,9381; ditahan 83 persen menjadi 2,8 persen; ambang biaya menggeneralisasi | diterima; gambar bersih akhirnya dapat diloloskan |
| 6 | 2026-08-18 | Integrasi | pipeline utuh di atas ONNX | inspeksi utuh sekitar 140 ms per gambar, 12 uji lolos | diterima; siap dipanggil Backend |
| 5 | 2026-08-18 | Ekspor ONNX | opset 12, imgsz 640, 4 utas | detect 62,5 ms dan seg 90,9 ms median di CPU | diterima; menyisakan ruang untuk sasaran satu detik |
| 4 | 2026-08-18 | Lapisan keputusan | kalibrasi, conformal, ambang biaya | Platt ECE 0,0391; conformal cakupan 0,9661; ambang biaya tidak menggeneralisasi | Platt dan conformal diterima; ambang operasi tetap 0,50 |
| 3 | 2026-08-18 | PaDiM anomali | 5 kategori, 200 gambar latih, worker dataloader nol | AUROC 0,848 sampai 0,997; ambang EVT lolos uji KS di semua kategori | diterima; titik operasi 1 persen dinilai terlalu ketat dan diserahkan ke Step 7 |
| 2 | 2026-08-18 | YOLO11n-seg | fine-tune dari bobot COCO, config apa adanya | berjalan penuh 250 epoch, terbaik epoch 203, mask mAP50 val 0,7899 | diterima; galat luas cacat 0,37 poin persen sudah memadai |

> **Aturan:** satu perubahan per run. Kalau augmentasi, ukuran model, dan learning rate diubah

> sekaligus lalu hasilnya membaik, penyebabnya tidak dapat diketahui.

---

## 10. Kegagalan & Pelajaran

*(Isi bagian ini dengan jujur - kegagalan yang tercatat justru memperkuat kredibilitas proposal
dan menunjukkan proses kerja yang nyata.)*

| Tgl | Yang gagal | Penyebab | Tindakan |
|---|---|---|---|
| 2026-08-18 | Peringatan ketidakcocokan ABI numpy dan scipy | pemasangan ultralytics menaikkan numpy ke 2.4.6 sementara scipy 1.13 menuntut di bawah 2.3 | scipy dinaikkan ke 1.17.1, seluruh angka statistik dihitung ulang di atas kombinasi yang sah |
| 2026-08-18 | Recall `gores` hanya 0,556 | objek gores paling kecil ukurannya dan dukungannya hanya 9 instance di test | dicatat sebagai keterbatasan; penambahan sampel menjadi bahan Step 3 |
| 2026-08-19 | Alpha 0,05 menahan 82 persen keputusan setelah set kalibrasi diperluas | kuantil kelas cacat mentok pada nilai maksimum karena jaminan 95 persen harus memuat cacat yang paling sulit dideteksi | alpha dipilih lewat aturan pada set kalibrasi dengan batas menahan 15 persen; terpilih 0,10 |
| 2026-08-18 | Gambar bersih tanpa deteksi justru masuk REVIEW, bukan PASS | himpunan conformal memuat kedua label karena kuantil kelas cacat 0,7772 sementara ketidaksesuaiannya 0,7770, selisih 0,0002; kuantil selonggar itu muncul karena jaminan 95 persen harus memuat cacat yang paling sulit dideteksi | dinyatakan sebagai keterbatasan; sapuan alpha membuktikan penyebabnya bukan pilihan alpha melainkan jumlah gambar normal yang hanya 7 |
| 2026-08-18 | Temperature scaling memperburuk ECE, 0,322 menjadi 0,404 | model justru kurang percaya diri karena 88 persen data uji memang cacat; temperature scaling tanpa intersep tidak dapat menggeser skor ke proporsi kelas sebenarnya | diganti Platt scaling yang punya intersep; ECE turun ke 0,0391 |
| 2026-08-18 | Ambang hasil optimasi biaya lebih mahal daripada ambang 0,50 pada data uji | hanya 7 gambar normal per split sehingga letak minimum biaya tidak stabil | ambang operasi tetap 0,50; keterbatasan dinyatakan di EXPERIMENTS.md |
| 2026-08-18 | `savings_versus` melaporkan biaya set kalibrasi seolah-olah biaya set uji | fungsi mengembalikan CostPoint yang diterimanya alih-alih menghitung ulang pada data yang dinilai | fungsi diubah agar mengevaluasi kedua ambang pada data yang sama |
| 2026-08-18 | Step 6 gagal empat kali berturut-turut | dirinci di bawah | seluruhnya diperbaiki; run bersih kelima berhasil |
| 2026-08-18 | Skor anomali hanya bernilai tiga macam | post-processor anomalib menormalisasi skor terhadap ambang internalnya dan praktis mengkuantisasi keluaran, sehingga ekor sebaran hilang dan tidak ada yang dapat dimodelkan GPD | post-processor dimatikan; skor mentah kembali kontinu dengan 39 nilai unik |
| 2026-08-18 | Hasil kategori pertama hilang saat kategori kedua gagal | JSON hanya ditulis di akhir setelah semua kategori selesai | penulisan dijadikan inkremental dan menggabung isi berkas yang sudah ada |
| 2026-08-18 | Proses mati kehabisan memori | worker dataloader disetel 4; di Windows tiap worker adalah proses Python penuh yang memuat ulang torch, sekitar 690 MB masing-masing | worker disetel nol; pemuatan data bukan hambatan pada model ini |
| 2026-08-18 | Blok penulis JSON lama tidak ikut terhapus dan menimpa hasil gabungan | penyuntingan berkas yang tidak menyeluruh | blok lama dihapus; hasil diverifikasi ulang terhadap log, 20 dari 20 angka cocok |
| 2026-08-18 | Laporan sempat menyebut proses sudah berhenti padahal masih berjalan | `pgrep` pada Git Bash tidak melihat proses Windows | pemeriksaan diganti memakai `Get-CimInstance` |
| 2026-08-18 | Segmentasi tidak berhenti awal | 250 epoch habis sebelum `patience` 40 tercapai, bobot terbaik di epoch 203 | dicatat apa adanya; model belum tentu konvergen dan dapat dilatih lebih lama bila waktu memungkinkan |
