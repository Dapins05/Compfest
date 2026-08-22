# DATASET_REQUIREMENTS.md - Dataset yang Dibutuhkan

**Domain:** inspeksi kualitas makanan dan minuman kemasan.

**Aturan panitia:** dataset boleh berasal dari sumber publik atau data sintetik.
Preprocessing wajib dikerjakan dan dijelaskan selama periode lomba
(17 Juni - 25 Agustus 2026).

> Alamat unduhan dataset publik dapat berubah sewaktu-waktu. Daftar sumber
> cadangan tersedia di bagian 2.3.

---

## 1. Status - SUDAH TERUNDUH

Diperbarui 21 Agustus 2026, setelah PKU-GoodsAD masuk.

| Tier | Dataset | Status | Terpakai untuk |
|---|---|---|---|
| **1** | **MVTec AD** - `bottle` | terunduh (209 train/good, 83 test, 63 mask) | deteksi + segmentasi + anomali |
| **1** | **MVTec AD** - `capsule`, `pill` | terunduh | tidak dipakai - farmasi, di luar domain pangan |
| **1** | **VisA** - 12 kategori penuh (12.021 gambar) | terunduh | `chewinggum`, `cashew`, `pipe_fryum` dipakai; `fryum` anomali saja |
| **1** | **PKU-GoodsAD** - 5 kategori | terunduh (5.512 gambar, **1.413 cacat bermask**) | deteksi + segmentasi + anomali |
| **1** | **MVTec LOCO** - 5 kategori | terunduh | **tidak dipakai** - alasan di bagian 4.2 |
| **2** | **Sintetik** (Step 3) | belum dibuat | menyeimbangkan `kotor` |
| **3** | **Foto produk lokal** | opsional | gambar contoh & video demo |

### Yang berubah setelah PKU-GoodsAD masuk

Jumlah gambar cacat pada dataset deteksi naik dari 363 menjadi **1.776**,
yaitu **4,9 kali lipat**. Seluruh gambar cacat GoodsAD punya mask piksel,
jadi tambahan itu berlaku untuk deteksi maupun segmentasi sekaligus.

GoodsAD juga memasok kelas keenam, `terbuka`. Empat nama folder aslinya
(`cap_open`, `cap_half_open`, `opened`, `straw_missing`) sama-sama berarti
kemasan tidak lagi tersegel. Memetakannya ke `deformasi` akan menaruh label
yang tidak sesuai isi gambar, jadi dibuatkan kelas sendiri.

| Kategori GoodsAD | train/good | test/good | test cacat | mask |
|---|---|---|---|---|
| `drink_bottle` | 733 | 356 | 425 | 425 |
| `food_bottle` | 1.014 | 243 | 361 | 361 |
| `food_package` | 540 | 253 | 230 | 230 |
| `food_box` | 432 | 146 | 251 | 251 |
| `drink_can` | 235 | 147 | 146 | 146 |


### Kategori final yang dipakai - dan dasarnya

Pemilihan **tidak** berdasarkan selera, melainkan ukuran cacat setelah resize
ke 640 px yang diukur langsung dari mask. Objek di bawah ~12 px tidak dapat
dipelajari detektor.

| Kategori | Produk | Objek ≥12 px @640 | Keputusan |
|---|---|---|---|
| `bottle` | botol minuman kaca | **100,0 %** | deteksi + segmentasi |
| `chewinggum` | permen karet kemasan | **96,3 %** | deteksi + segmentasi |
| `cashew` | kacang mete | **67,6 %** | deteksi + segmentasi |
| `pipe_fryum` | snack goreng | **63,9 %** | deteksi + segmentasi |
| `fryum` | snack goreng | 29,8 % | anomali saja |
| `macaroni1` / `macaroni2` | pasta kering | 37,8 % / 31,8 % | cadangan, tidak diproses |
| `candle`, `pcb1`-`pcb4`, `capsules` | bukan pangan / farmasi | - | di luar domain |

>  **"Perlukah menambah dataset keripik/chips?"** - **Tidak.** Tidak ada

> dataset publik keripik-dalam-kemasan yang beranotasi mask, jadi menambahnya
> berarti melabeli sendiri dari nol - pekerjaan berhari-hari yang tidak muat di
> sisa waktu. Domain "makanan & minuman kemasan" sudah tertutup oleh botol
> minuman, permen karet, kacang, dan snack goreng. Kalau keripik ingin muncul
> di **video demo** , tempuh jalur Tier 3 (foto sendiri, ±45 menit) - itu untuk
> gambar contoh, bukan untuk melatih model.

## 2. Tier 1 - Dataset Publik (WAJIB)

### 2.1 MVTec AD - kategori `bottle`  prioritas tertinggi

| | |
|---|---|
| **URL** | https://www.mvtec.com/company/research/datasets/mvtec-ad |
| **Lisensi** | CC BY-NC-SA 4.0 - bebas untuk riset & kompetisi non-komersial  |
| **Isi kategori bottle** | ~209 gambar normal (train) + ~83 gambar uji (normal & cacat) |
| **Jenis cacat** | `broken_large`, `broken_small`, `contamination` |
| **Anotasi** | **Mask piksel** untuk tiap cacat - bisa dikonversi ke bbox **dan** polygon |
| **Cara unduh** | Isi formulir singkat di situs  dapat tautan unduh |

**Kenapa ini yang utama:** - Kategori `bottle` = botol kaca, persis domain minuman kemasan kita
- Ini **standar emas** anomaly detection industri - dikenal semua reviewer, memperkuat kredibilitas proposal
- Punya mask ground-truth, jadi satu dataset melayani **tiga model sekaligus** (detect, seg, anomaly)
- Struktur `train/good/` saja membuatnya sempurna untuk melatih EfficientAD (yang hanya butuh gambar normal)

**Yang dibutuhkan:** minimal folder `bottle/`. Bila kuota internet cukup, unduh juga
`capsule/` dan `pill/` sebagai cadangan variasi kemasan.

### 2.2 VisA - subset makanan

| | |
|---|---|
| **URL** | https://github.com/amazon-science/spot-diff |
| **Pemilik** | Amazon Science |
| **Lisensi** | CC BY 4.0  |
| **Total** | 12 kategori, ~10.821 gambar |
| **Subset relevan** | `chewinggum` (permen karet kemasan), `fryum` (snack kering), `cashew` (kacang mete), `capsules` (kemasan blister) |

**Kenapa perlu:** MVTec `bottle` menutup sisi minuman, VisA menutup sisi **makanan ringan kemasan** .

Kombinasi keduanya membuat klaim "makanan & minuman kemasan" di proposal benar-benar didukung data,
bukan sekadar narasi.

**Yang dibutuhkan:** minimal `chewinggum` dan `fryum`.

### 2.3 Cadangan bila dua di atas bermasalah

| Dataset | URL | Catatan |
|---|---|---|
| **MVTec LOCO AD** | https://www.mvtec.com/company/research/datasets/mvtec-loco | Cacat logis (komponen kurang/salah posisi) - relevan untuk segel & label |
| **Roboflow Universe** | https://universe.roboflow.com | Cari kata kunci: `bottle defect`, `can dent`, `packaging defect`. Sudah dalam format YOLO - hemat waktu konversi |
| **Kaggle** | https://www.kaggle.com/datasets | Cari: `beverage can defect`, `food packaging defect` |
| **BTAD** | beanTech Anomaly Detection - cari via GitHub/paper | 3 kategori produk industri |
| **Real-IAD** | https://realiad4ad.github.io/Real-IAD/ | 30 kategori, multi-view, dataset 2024 |

---


## 3. Lisensi dan Atribusi

Wajib dicantumkan karena repo bersifat publik dan kedua dataset menuntut
atribusi. Salah satunya juga membatasi pemakaian komersial.

| Dataset | Lisensi | Konsekuensi |
|---|---|---|
| MVTec AD | CC BY-NC-SA 4.0 | Atribusi wajib, **non-komersial**, turunan mengikuti lisensi yang sama |
| VisA | CC BY 4.0 | Atribusi wajib, pemakaian komersial diizinkan |
| PKU-GoodsAD | GPL-3.0 | Atribusi wajib. Catatan penting di bawah. |

**Soal GPL-3.0 pada PKU-GoodsAD.** GPL-3.0 adalah lisensi *perangkat lunak*,
dan repo GoodsAD memakainya untuk seluruh isi repo termasuk tautan datanya.
Apakah bobot model yang dilatih memakai data berlisensi GPL ikut tertular
kewajiban copyleft **belum pernah diuji di pengadilan** dan tidak ada jawaban
yang pasti. Yang dapat dipastikan hanya ini:

- Dataset mentahnya **tidak** ikut masuk repo (sudah diabaikan `.gitignore`),
  jadi repo ini tidak mendistribusikan ulang karya berlisensi GPL.
- Kode VisionQC ditulis sendiri, bukan turunan kode GoodsAD.
- Untuk keperluan lomba yang non-komersial, risikonya kecil.
- **Sebelum dikomersialkan**, status ini wajib ditinjau ulang, dan cara paling
  aman adalah melatih ulang tanpa GoodsAD.

Yang perlu diperhatikan tim: `reports/figures/dataset_samples_train.png` dan
`dataset_samples_test.png` memuat cuplikan gambar dari kedua dataset. Selama
repo ini dipakai untuk lomba dan bukan untuk tujuan komersial, penyertaan itu
masih di dalam ketentuan CC BY-NC-SA sepanjang atribusinya jelas. Bila proyek
ini nanti dikomersialkan, kedua figur tersebut dan seluruh turunan MVTec harus
dikeluarkan lebih dulu.

Sitasi yang dipakai:

- Bergmann, P., Fauser, M., Sattlegger, D., Steger, C. *MVTec AD - A
  Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR
  2019.
- Zou, Y., Jeong, J., Pemula, L., Zhang, D., Dabeer, O. *SPot-the-Difference
  Self-Supervised Pre-training for Anomaly Detection and Segmentation (VisA).*
  ECCV 2022.
- Zhang, J., Ding, R., Ban, M., Yang, G. *PKU-GoodsAD: A Supermarket Goods
  Dataset for Unsupervised Anomaly Detection and Segmentation.* IEEE Robotics
  and Automation Letters, 2024.

---

## 4. Sumber Tambahan untuk Memperbesar Data

Diverifikasi 20 Agustus 2026. Diurutkan menurut manfaatnya bagi proyek ini,
bukan menurut popularitasnya.

### 4.1 PKU-GoodsAD - paling cocok

Satu-satunya dataset publik yang isinya benar-benar barang belanjaan kemasan,
bukan komponen industri. Kategorinya bertepatan langsung dengan domain proyek
ini.

| | |
|---|---|
| Sumber | https://github.com/jianzhang96/GoodsAD |
| Makalah | PKU-GoodsAD, IEEE RA-L 2024 |
| Lisensi | GPL-3.0 |
| Isi | 6.124 gambar, 484 barang berbeda, 6 kategori |
| Anotasi | tingkat gambar **dan** mask piksel |

| Kategori | Latih normal | Uji normal | Uji cacat |
|---|---|---|---|
| `drink_bottle` | 733 | 356 | 425 |
| `food_bottle` | 1.014 | 243 | 361 |
| `food_package` | 540 | 253 | 230 |
| `food_box` | 432 | 146 | 251 |
| `drink_can` | 234 | 147 | 147 |
| `cigarette_box` | 183 | 183 | 246 |

Jenis cacatnya deformasi, kerusakan permukaan, dan kemasan terbuka. Dua yang
pertama persis kelas yang paling kekurangan contoh pada dataset sekarang.

Dua sifat yang membuatnya lebih menantang sekaligus lebih berguna: posisi objek
**tidak disejajarkan**, dan satu kategori memuat banyak barang dengan tampilan
berbeda. Keduanya lebih menyerupai gambar yang akan diunggah pengguna
sungguhan daripada MVTec yang serba terpusat.

Unduhan lewat OneDrive atau Baidu Disk, per kategori 1,1 sampai 3,0 GB.
Ambil `drink_bottle`, `food_bottle`, dan `food_package` lebih dulu; ketiganya
sudah menambah sekitar 1.000 gambar cacat bermask.

**Catatan lisensi:** GPL-3.0 menular pada karya turunan. Perlu diputuskan tim
sebelum dipakai.

### 4.2 MVTec LOCO AD - kategori `juice_bottle`

| | |
|---|---|
| Sumber | https://www.mvtec.com/company/research/datasets/mvtec-loco/downloads |
| Lisensi | CC BY-NC-SA 4.0, non-komersial |
| Ukuran | `juice_bottle` 625 MB, seluruhnya 5,71 GB |

Nilainya bukan pada jumlah, melainkan pada jenis cacatnya: selain cacat
struktural, ada **cacat logis** seperti label tertukar, isi keliru, atau
komponen yang seharusnya ada tetapi tidak ada. Cacat semacam itu tidak ada
sama sekali pada dataset sekarang, dan justru itu yang sering terjadi pada
kemasan minuman sungguhan.

**Status 21 Agustus 2026: sudah terunduh, tetapi TIDAK dipakai.** Tiga alasan,
diurutkan dari yang paling menentukan:

1. **Folder cacatnya campuran.** LOCO hanya memilah `logical_anomalies` dan
   `structural_anomalies`. Keduanya berisi banyak jenis cacat yang berbeda
   dalam satu wadah. Memetakan salah satunya ke satu kelas VisionQC akan
   menaruh label yang tidak sesuai isi gambar - persis alasan yang dipakai
   untuk menolak memetakan `opened` ke `deformasi`.
2. **Tata letak mask-nya berbeda lagi.** Mask LOCO disimpan per gambar di
   subfolder tersendiri, satu berkas untuk tiap instance. Ini bentuk ketiga
   setelah MVTec dan GoodsAD, jadi butuh konverter tersendiri.
3. **Cacat logis butuh rancangan model yang lain.** Makalah LOCO sendiri
   menunjukkan metode berbasis patch embedding seperti PaDiM berkinerja buruk
   pada cacat logis. Menambahkannya ke jalur anomali sekarang tidak akan
   memperbaiki apa pun yang bisa diukur.

Disimpan untuk tahap Final, ketika cacat logis ditangani model tersendiri.

### 4.3 Real-IAD - bila butuh skala

| | |
|---|---|
| Makalah | CVPR 2024, https://arxiv.org/pdf/2403.12580 |
| Isi | 150.000 gambar, 30 objek, lima sudut pandang per objek |

Satu tingkat lebih besar daripada dataset yang ada. Berguna untuk pralatih
model anomali, tetapi objeknya lebih banyak komponen industri daripada
kemasan pangan. Perlu memeriksa syarat aksesnya lebih dulu; tidak semua
bagian dapat diunduh bebas.

### 4.4 Roboflow Universe - pelengkap kecil

Beberapa kumpulan kecil yang sudah berlabel kotak, misalnya
`spark-intelligence-scqhh/bottle-defect-detection` (262 gambar) dan
`teamsawa/beverage-packaging` (152 gambar). Ukurannya terlalu kecil untuk
mengubah hasil training, dan lisensinya berbeda-beda per kumpulan sehingga
harus diperiksa satu per satu. Berguna sebagai gambar demo, bukan sebagai
data latih utama.

### Rekomendasi - SUDAH DIJALANKAN 21 Agustus 2026

Rekomendasi awalnya hanya tiga kategori PKU-GoodsAD (`drink_bottle`,
`food_bottle`, `food_package`). Yang benar-benar diambil **lima kategori**,
karena `drink_can` dan `food_box` ternyata juga membawa mask lengkap dan
memasok `deformasi` - kelas yang sebelumnya paling tipis setelah `kotor`.

Dua prasyarat yang disebut di sini sudah dikerjakan:

- Pemetaan jenis cacat ke taksonomi di `src/visionqc_ai/data/taxonomy.py`,
  termasuk kelas keenam `terbuka` untuk empat label yang tidak punya padanan.
- Konverter tersendiri di `src/visionqc_ai/data/convert_goodsad.py`, karena
  GoodsAD memakai JPEG dan nama mask tanpa akhiran `_mask`.

Konsekuensinya juga sudah berlaku: seluruh angka Step 4 sampai 7 dari model
lama **tidak lagi sah** untuk model baru dan diukur ulang dari nol.

---

## 5. Tier 2 - Data Sintetik (WAJIB, dibuat di Step 3)

Panitia **secara eksplisit mengizinkan data sintetik** . Ini bukan jalan pintas - ini menyelesaikan
masalah nyata dan sekaligus menjadi diferensiator.

> **Ditulis ulang 21 Agustus 2026.** Rencana di bawah dibuat ketika data cacat
> hanya 363 gambar. Setelah PKU-GoodsAD masuk, sebagian besar sasarannya
> ternyata sudah tertutup **data nyata**, yang selalu lebih baik daripada
> sintetik. Yang tersisa justru menyempit ke satu kelas saja.

**Sasaran lama versus keadaan sekarang:**

| Rencana sintetik semula | Ditutup data nyata? | Instance sekarang |
|---|---|---|
| `gores` | ya, `surface_damage` GoodsAD | 782 |
| `segel_rusak` | ya, kelas `terbuka` GoodsAD | 475 |
| `penyok` | ya, `deformation` GoodsAD | 257 |
| `sobek` | sebagian, lewat `pecah` | 418 |
| `label_miring` | tidak, tetapi ini cacat logis - lihat 4.2 | - |
| **`kotor`** | **tidak** | **22** |

**Masalah yang benar-benar tersisa: `kotor`.** Kelas ini tidak ikut naik sama
sekali karena GoodsAD tidak memuat kontaminasi permukaan. Akibatnya rasio
ketimpangan justru **memburuk dari sekitar 4:1 menjadi 35,6:1**, bukan karena
`kotor` berkurang, melainkan karena lima kelas lain tumbuh berkali lipat.

Ini penting melebihi angkanya: `kotor` adalah satu-satunya anggota
`critical_classes` di decision engine, yaitu kelas yang memicu REJECT berapa
pun luas cacatnya. Aturan sekeras itu kini bersandar pada kelas dengan sekitar
15 instance latih.

**Sasaran Step 3 yang diperbarui:** hanya `kotor`, dari sekitar 22 menjadi
sekitar 250 instance. Dibuat dari gambar normal Tier 1 berupa bercak dan
partikel kontaminasi, divalidasi dengan **jarak Wasserstein** dan **uji KS**
(STATISTICS.md bagian 2.4) supaya distribusinya tidak menyimpang jauh dari
kontaminasi asli.

Yang **tidak** boleh dilakukan: menggabung penambahan sintetik ke dalam
perubahan yang sama dengan penambahan data nyata. Keduanya harus diukur
terpisah, kalau tidak, perbaikan yang muncul tidak dapat ditelusuri berasal
dari mana.

---

## 6. Tier 3 - Foto Produk Lokal (Opsional, nilai tinggi)

**Tidak untuk melatih model** - untuk **gambar contoh di aplikasi dan video demo** .

**Kenapa disarankan:** juri Indonesia melihat demo memakai Teh Botol, Aqua, atau Chitato akan
langsung menangkap relevansinya. Dataset MVTec berisi botol laboratorium yang asing bagi
kebanyakan orang. Biayanya hanya ± 45 menit memotret.

Ketentuan pengambilan foto:

```
30-60 foto, kamera HP biasa
|-- 20-40 produk kondisi BAGUS (botol, kaleng, snack pouch)
|-- 10-20 produk kondisi CACAT (sengaja dipenyokkan, digores, segel dibuka)

Ketentuan:
- Latar polos (kertas putih / meja bersih)
- Pencahayaan konsisten, hindari bayangan keras
- Jarak & sudut seragam
-  JANGAN ada wajah orang dalam bingkai (lihat PRIVACY.md)
-  JANGAN ada identitas institusi pendidikan dalam bingkai
```

Simpan di `AI_model/data/raw/local_samples/`.

---

## 7. Struktur Folder Sebagaimana Adanya

Hasil ekstraksi **berbeda** dari yang semula diasumsikan dokumen ini: MVTec
terekstrak dengan folder ganda, dan VisA memakai tata letak `Data/Images/...`.

Berkas mentah **tidak dipindahkan** . Yang menyesuaikan adalah konverternya -
seluruh pengetahuan tentang tata letak ini terkurung di satu berkas,
`src/visionqc_ai/data/sources.py`.

```
AI_model/data/raw/
|-- bottle/bottle/                       folder ganda, memang begitu adanya
|   |-- train/good/                     209 gambar
|   |-- test/{good,broken_large,broken_small,contamination}/
|   |-- ground_truth/<jenis>/<id>_mask.png
|-- capsule/capsule/                    (tidak dipakai)
|-- pill/pill/                          (tidak dipakai)
|-- VisA_20220922/
    |-- <kategori>/Data/Images/{Normal,Anomaly}/*.JPG
    |-- <kategori>/Data/Masks/Anomaly/*.png       mask BER-KODE, bukan biner
    |-- <kategori>/image_anno.csv
    |-- split_csv/                      (split resmi VisA; kita pakai split sendiri)
```

Keluaran Step 2:

```
AI_model/data/processed/
|-- detect/{images,labels}/{train,val,test}/  + data.yaml
|-- seg/{images,labels}/{train,val,test}/     + data.yaml
|-- anomaly/<kategori>/{train/good, test/good, test/defect}/
```

## 8. Verifikasi - SUDAH DILAKUKAN

Dijalankan 18 Agustus 2026:

| Pemeriksaan | Hasil |
|---|---|
| MVTec `bottle` lengkap | 209 train/good + 83 test + 63 mask  |
| VisA terbaca & mask ter-decode | 12.021 gambar, mask ber-kode  |
| GPU terdeteksi | RTX 3050 Laptop, **4096 MiB (4 GB)** , CUDA aktif  |
| Python | 3.11.1  |
| torch | 2.5.1+cu121, `cuda.is_available() = True`  |
| Ruang disk | 27 GB tersisa - cukup, tapi tipis  |

Untuk menjalankan ulang seluruh preprocessing:

```bash
python scripts/prepare_dataset.py              # konversi + split + validasi + tulis
python scripts/prepare_dataset.py --dry-run    # tanpa menulis gambar
python scripts/preview_dataset.py --split test # lembar kontak verifikasi anotasi
```

## 9. Kalau Terkendala

| Kendala | Jalan keluar |
|---|---|
| MVTec butuh isi formulir & lama | Pakai Roboflow Universe dulu (langsung unduh, sudah format YOLO), MVTec menyusul |
| Internet lambat / kuota terbatas | Unduh **hanya** `bottle` dari MVTec (~250 MB). Cukup untuk seluruh 10 langkah |
| Disk penuh | Cukup Tier 1 minimal + sintetik. VisA boleh dilewati, catat sebagai keterbatasan di proposal |
| Tautan mati | Cari sumber pengganti pada daftar cadangan di bagian 2.3 |

>  **Saran soal urutan:** jangan menunggu semua dataset lengkap. Begitu **MVTec `bottle` selesai
> diunduh** , Step 2 sudah bisa dimulai. VisA bisa ditambahkan belakangan tanpa mengubah kode.
