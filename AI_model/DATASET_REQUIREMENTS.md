# DATASET_REQUIREMENTS.md - Dataset yang Dibutuhkan

**Domain:** inspeksi kualitas makanan dan minuman kemasan.

**Aturan panitia:** dataset boleh berasal dari sumber publik atau data sintetik.
Preprocessing wajib dikerjakan dan dijelaskan selama periode lomba
(17 Juni - 25 Agustus 2026).

> Alamat unduhan dataset publik dapat berubah sewaktu-waktu. Daftar sumber
> cadangan tersedia di bagian 2.3.

---

## 1. Status - SUDAH TERUNDUH

Diperbarui 18 Agustus 2026, setelah Step 2 selesai.

| Tier | Dataset | Status | Terpakai untuk |
|---|---|---|---|
| **1** | **MVTec AD** - `bottle` | terunduh (209 train/good, 83 test, 63 mask) | deteksi + segmentasi + anomali |
| **1** | **MVTec AD** - `capsule`, `pill` | terunduh | tidak dipakai - farmasi, di luar domain pangan |
| **1** | **VisA** - 12 kategori penuh (12.021 gambar) | terunduh | `chewinggum`, `cashew`, `pipe_fryum` dipakai; `fryum` anomali saja |
| **2** | **Sintetik** (Step 3) | belum dibuat | menyeimbangkan `kotor` & `deformasi` |
| **3** | **Foto produk lokal** | opsional | gambar contoh & video demo |

**Ukuran di disk:**`data/raw` ≈ 2,6 GB · `data/processed` ≈ 302 MB.


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

>  **"Perlukah menambah dataset keripik/chips?"** - **Tidak.**Tidak ada

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

**Kenapa perlu:**MVTec `bottle` menutup sisi minuman, VisA menutup sisi **makanan ringan kemasan** .

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


## 3. Tier 2 - Data Sintetik (WAJIB, dibuat di Step 3)

Panitia **secara eksplisit mengizinkan data sintetik** . Ini bukan jalan pintas - ini menyelesaikan
masalah nyata dan sekaligus menjadi diferensiator.

**Masalah yang diselesaikan:**MVTec `bottle` hanya punya 3 jenis cacat dengan total di bawah 100
gambar cacat. Terlalu sedikit untuk melatih detektor yang andal, dan rasio ketimpangannya ekstrem.

**Yang akan dihasilkan** (dari gambar normal Tier 1):

| Cacat sintetik | Cara pembuatan | Relevansi domain |
|---|---|---|
| `penyok` | Deformasi lokal + perubahan shading | Kaleng minuman penyok saat distribusi |
| `gores` | Perlin noise berbentuk garis + blending | Botol/kaleng tergores di gudang |
| `sobek` | Robekan tepi tidak beraturan | Kemasan pouch/sachet |
| `segel_rusak` | Distorsi pada area segel | Segel tutup tidak sempurna |
| `label_miring` | Rotasi + translasi area label | Label tertempel miring |
| `kotor` | Bercak/noda acak | Kontaminasi permukaan |

Kualitas tiap gambar sintetik divalidasi dengan **jarak Wasserstein** dan **uji KS** (STATISTICS.md bagian 2.4) agar tidak terlalu menyimpang dari distribusi asli.

**Target:**~500-1.000 gambar sintetik hingga rasio ketimpangan turun di bawah 3:1.

---

## 4. Tier 3 - Foto Produk Lokal (Opsional, nilai tinggi)

**Tidak untuk melatih model** - untuk **gambar contoh di aplikasi dan video demo** .

**Kenapa disarankan:**juri Indonesia melihat demo memakai Teh Botol, Aqua, atau Chitato akan
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

## 5. Struktur Folder Sebagaimana Adanya

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

## 6. Verifikasi - SUDAH DILAKUKAN

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

## 7. Kalau Terkendala

| Kendala | Jalan keluar |
|---|---|
| MVTec butuh isi formulir & lama | Pakai Roboflow Universe dulu (langsung unduh, sudah format YOLO), MVTec menyusul |
| Internet lambat / kuota terbatas | Unduh **hanya** `bottle` dari MVTec (~250 MB). Cukup untuk seluruh 10 langkah |
| Disk penuh | Cukup Tier 1 minimal + sintetik. VisA boleh dilewati, catat sebagai keterbatasan di proposal |
| Tautan mati | Cari sumber pengganti pada daftar cadangan di bagian 2.3 |

>  **Saran soal urutan:**jangan menunggu semua dataset lengkap. Begitu **MVTec `bottle` selesai
> diunduh** , Step 2 sudah bisa dimulai. VisA bisa ditambahkan belakangan tanpa mengubah kode.
