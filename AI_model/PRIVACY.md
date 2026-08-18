# PRIVACY.md - Desain Privasi VisionQC

**Prinsip:***privacy by design* - privasi dirancang sejak awal arsitektur, bukan ditambal belakangan.

**Landasan hukum:UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi (UU PDP)** ,
diselaraskan juga dengan prinsip GDPR Pasal 5.

> **Kenapa ini relevan untuk sistem inspeksi produk?**Pertanyaan yang wajar - yang difoto kan
> produk, bukan orang. Tapi kamera di lini produksi **tidak pernah hanya menangkap produk** :
> tangan dan wajah operator masuk bingkai, metadata EXIF membawa lokasi GPS dan identitas
> perangkat, dan OCR bisa saja membaca teks yang tidak dimaksudkan untuk dibaca. Sistem yang
> menyimpan semuanya "untuk jaga-jaga" adalah kewajiban hukum yang menunggu terjadi.

---

## 1. Empat Prinsip

| # | Prinsip | Penerapan di VisionQC |
|---|---|---|
| 1 | **Minimalisasi data** | Hanya memproses apa yang dibutuhkan untuk memutuskan lolos/tolak. Tidak ada yang dikumpulkan "untuk jaga-jaga" |
| 2 | **Pemrosesan lokal** | Seluruh inferensi berjalan di mesin lokal. Tidak ada API awan, tidak ada telemetri, tidak ada data keluar |
| 3 | **Pemrosesan sekali pakai** | Gambar hidup di memori selama satu permintaan, lalu dihapus. Tidak ditulis ke disk sama sekali |
| 4 | **Transparansi** | Setiap transformasi privasi tercatat dan dapat diaudit |

>  **Kebetulan yang menguntungkan:** arsitektur sinkron tanpa basis data yang diwajibkan panitia
> justru **secara alami** memenuhi prinsip 1 dan 3. Batasan lomba dan desain privasi
> mengarah ke tempat yang sama.

---

## 2. Analisis Ancaman & Mitigasi

| # | Ancaman | Risiko | Mitigasi | Step |
|---|---|---|---|---|
| **A1** | Wajah/tangan operator masuk bingkai | Data biometrik = **data pribadi spesifik** menurut UU PDP Pasal 4 ayat (2) | Deteksi wajah  buramkan **sebelum** inferensi. Wajah tidak pernah sampai ke model | 8 |
| **A2** | Metadata EXIF (GPS, model perangkat, nomor seri, waktu) | Membocorkan lokasi pabrik dan identitas perangkat | Hapus **seluruh** metadata saat gambar masuk. Bukan hanya GPS - semuanya | 8 |
| **A3** | Gambar tersimpan di disk/log | Menciptakan kewajiban penyimpanan & risiko kebocoran | Buffer sekali pakai; gambar tidak pernah menyentuh disk. Buffer dinolkan setelah dipakai | 8 |
| **A4** | OCR membaca teks di luar kode batch | Bisa menangkap nama, nomor telepon, dokumen di latar | Penyaring **daftar-izin regex** - hanya pola kode batch yang dipertahankan, sisanya dibuang | 8 |
| **A5** | Model menghafal data latih (*membership inference*) | Data latih bisa direkonstruksi dari bobot model | Augmentasi kuat + porsi data sintetik besar; opsi DP-SGD untuk tahap Final | 4, 8 |
| **A6** | Log berisi data mentah | Log sering terlupakan padahal berisi data sensitif | **Log hanya SHA-256** dari gambar. Cukup untuk penelusuran, tidak bisa dibalik jadi gambar | 8 |
| **A7** | Bobot model ikut membawa data | Distribusi model membawa jejak data latih | Yang didistribusikan hanya ONNX (bobot), bukan dataset. Provenans dicatat terpisah | 9 |
| **A8** | Gambar terkirim ke layanan pihak ketiga | Data keluar dari kendali | Tidak ada panggilan jaringan sama sekali di jalur inferensi. Diverifikasi lewat pengujian | 8, 10 |

---

## 3. Implementasi Teknis (Step 8)

```
src/visionqc_ai/privacy/
|-- exif.py          # A2 - hapus seluruh metadata
|-- face_blur.py     # A1 - deteksi & buramkan wajah
|-- ephemeral.py     # A3 - context manager, buffer dinolkan
|-- ocr_filter.py    # A4 - daftar-izin regex kode batch
|-- audit.py         # A6 - log hanya hash
```

### 3.1 Pembersihan EXIF (A2)

```
Gambar masuk

Decode piksel saja  buang seluruh blok metadata

Encode ulang tanpa EXIF/XMP/IPTC

Verifikasi: tidak ada satu pun kunci metadata tersisa
```

Pendekatannya **daftar-tolak menyeluruh** , bukan menghapus field satu per satu - karena
format baru bisa memperkenalkan field baru yang tidak kita antisipasi.

### 3.2 Peredaman wajah (A1)

Wajah diburamkan **sebelum** gambar masuk ke model inspeksi. Urutan ini penting: kalau
diburamkan setelah inferensi, wajah tetap pernah diproses model dan berpotensi terekam
dalam aktivasi.

Deteksi wajah memakai model ringan yang berjalan lokal. Bila ragu  tetap buramkan
(memilih aman ketimbang menyesal).

### 3.3 Pemrosesan sekali pakai (A3)

```python
with EphemeralImage(image_bytes) as img:
    result = pipeline.run(img)
# keluar dari blok  buffer ditimpa nol, referensi dilepas
return result   # hanya angka & teks yang keluar, tidak ada piksel
```

Yang keluar dari fungsi hanyalah **hasil keputusan dan gambar beranotasi yang diminta pengguna** .
Tidak ada salinan yang tertinggal.

### 3.4 Penyaring OCR (A4)

Hanya teks yang cocok dengan pola kode batch yang dipertahankan:

```
^[A-Z]{1,3}[-/]?\d{6,10}$        # contoh: B240815021, MFG-20260815
^\d{2}[/.]\d{2}[/.]\d{2,4}$      # tanggal kedaluwarsa
```

Teks lain yang terbaca **langsung dibuang sebelum keluar dari fungsi OCR** . Ini penting:
menyaring di lapisan atas berarti teks sensitif sempat ada di memori lebih lama dari perlunya.

### 3.5 Audit hanya-hash (A6)

```json
{
  "inspection_id": "018f...",
  "image_sha256": "e3b0c44298fc1c149afbf4c8996fb924...",
  "verdict": "REJECT",
  "model_version": "yolo11n-defect-ft-v2",
  "privacy": { "exif_stripped": true, "faces_blurred": 0, "ocr_filtered": 2 },
  "timestamp": "2026-08-15T09:12:44Z"
}
```

Tidak ada piksel. Hash cukup untuk membuktikan gambar yang sama menghasilkan keputusan yang sama,
tanpa menyimpan gambarnya.

---

## 4. Pemetaan ke UU PDP No. 27/2022

| Prinsip UU PDP | Penerapan |
|---|---|
| **Pembatasan tujuan** (Pasal 16) | Gambar hanya dipakai untuk menentukan lolos/tolak. Tidak untuk pemantauan kinerja operator |
| **Minimalisasi** (Pasal 16) | Hanya piksel produk yang diproses; wajah diburamkan lebih dulu |
| **Akurasi** (Pasal 16) | Kalibrasi kepercayaan + conformal prediction memastikan keputusan tidak diambil saat bukti lemah |
| **Keamanan** (Pasal 35) | Pemrosesan lokal, tanpa transmisi jaringan, buffer sekali pakai |
| **Pembatasan retensi** (Pasal 43) | Nol retensi - gambar tidak pernah disimpan |
| **Data pribadi spesifik** (Pasal 4 ayat 2) | Data biometrik (wajah) diburamkan sebelum diproses |
| **Akuntabilitas** (Pasal 20) | Jejak audit hanya-hash yang dapat diverifikasi |

**Untuk proposal (Anggota 3):** ini poin yang layak diberi sub-bab tersendiri. Sangat sedikit

peserta lomba yang memikirkan kepatuhan data, padahal itu justru pertanyaan pertama yang muncul
saat sistem seperti ini hendak dipakai perusahaan sungguhan.

---

## 5. Yang Diparkir ke Tahap Final

| Fitur | Kenapa belum sekarang |
|---|---|
| **DP-SGD** (differentially private training) | Butuh waktu training jauh lebih lama; tidak muat 10 hari |
| **Federated learning** antar pabrik | Butuh arsitektur terdistribusi - dilarang di penyisihan |
| **Enkripsi model saat istirahat** | Perlu manajemen kunci |
| **Kontrol akses berbasis peran** | Otentikasi kompleks dilarang di penyisihan |
| **Pengujian membership inference** | Butuh waktu; disebut sebagai rencana di proposal |

---


## 6. Verifikasi (Step 10)

Klaim privasi harus bisa dibuktikan, bukan sekadar ditulis:

```
[ ] Tidak ada satu pun operasi tulis ke disk pada jalur inferensi (diverifikasi lewat pengujian)
[ ] Tidak ada panggilan jaringan pada jalur inferensi (diverifikasi lewat pengujian)
[ ] Gambar keluaran benar-benar bersih dari EXIF (diperiksa dengan pembaca metadata)
[ ] Wajah pada gambar uji benar-benar diburamkan sebelum inferensi
[ ] Log tidak memuat data gambar dalam bentuk apa pun
[ ] Penyaring OCR membuang teks non-kode-batch
```

Hasilnya ditulis ke `reports/privacy_audit.md` - dan bisa langsung dikutip di proposal serta
ditunjukkan di video proof of work.
