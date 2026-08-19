# Laporan Audit Privasi

Memeriksa apakah klaim privasi pada [PRIVACY.md](../PRIVACY.md) benar-benar
terwujud dalam kode, bukan hanya tertulis. Setiap baris di bawah menunjuk ke
berkas yang mengerjakannya dan uji yang membuktikannya.

Tanggal: 19 Agustus 2026.

---

## 1. Ringkasan

| Ancaman | Mitigasi | Berkas | Status |
|---|---|---|---|
| Metadata membocorkan lokasi dan perangkat | Gambar dikodekan ulang dari piksel | `privacy/exif.py` | aktif |
| Wajah operator ikut terpotret | Diburamkan sebelum masuk model | `privacy/face_blur.py` | aktif |
| Gambar tersimpan di disk | Buffer ditimpa nol setelah dipakai | `privacy/ephemeral.py` | aktif |
| OCR membaca nama dan alamat | Daftar-izin pola kode batch | `privacy/ocr_filter.py` | aktif, OCR belum terpasang |
| Catatan memuat gambar | Hanya SHA-256 yang disimpan | `privacy/audit.py` | aktif |
| Data terkirim ke luar | Tidak ada panggilan jaringan saat inferensi | `inference/pipeline.py` | aktif |

---

## 2. Pembersihan metadata

Berkas foto membawa koordinat GPS, merek dan nomor seri perangkat, serta waktu
pengambilan. Tidak satu pun dibutuhkan untuk inspeksi kualitas, sementara
semuanya dapat mengidentifikasi orang atau lokasi pabrik.

Pendekatannya daftar-tolak menyeluruh, bukan daftar-izin: gambar diurai menjadi
piksel lalu dikodekan ulang dari nol. Menghapus medan satu per satu selalu
menyisakan risiko ada yang terlewat.

Diuji oleh `test_metadata_dibersihkan` dan `test_pembersihan_mempertahankan_ukuran`.

---

## 3. Peredaman wajah

Wajah adalah data biometrik. UU PDP No. 27 Tahun 2022 Pasal 4 ayat 2
menggolongkannya sebagai data pribadi bersifat spesifik.

Wajah diburamkan **sebelum** gambar mencapai model, bukan sesudahnya, sehingga
model tidak pernah melihatnya. Pendeteksinya YuNet, jaringan bawaan OpenCV
berukuran 230 KB.

Verifikasi dilakukan pada foto wajah sungguhan: satu wajah terdeteksi dan
diburamkan sampai tidak dapat dikenali. **Gambar itu sengaja tidak disimpan ke
repo**, karena memuat foto orang sungguhan untuk mendemonstrasikan fitur
privasi bertentangan dengan tujuan fitur itu sendiri. Uji otomatis memakai
gambar sintetik.

Keterbatasan yang dinyatakan:

1. Wajah menyamping atau sangat kecil dapat terlewat.
2. Bila berkas modelnya tidak tersedia, lapisan ini melaporkan dirinya tidak
   aktif alih-alih diam-diam dilewati. Klaim privasi yang tidak berjalan lebih
   berbahaya daripada klaim yang dinyatakan gagal.

Diuji oleh `test_gambar_tanpa_wajah_tidak_berubah` dan
`test_peredaman_melaporkan_ketersediaan`.

---

## 4. Pemrosesan sekali pakai

Gambar hidup di memori selama satu permintaan, lalu buffernya ditimpa nol.
Penimpaan tetap terjadi walau pemrosesan gagal di tengah jalan.

Menimpa dengan nol tidak menjamin apa pun terhadap penyerang yang menguasai
mesin, karena Python dapat menyalin data di balik layar. Yang dijamin: tidak
ada gambar yang sengaja disimpan, dan buffer yang masih dipegang program tidak
lagi berisi gambar.

Diuji oleh `test_buffer_dinolkan_setelah_dipakai` dan
`test_buffer_dinolkan_walau_terjadi_galat`.

---

## 5. Penyaring OCR

OCR membaca apa pun yang terlihat, termasuk nama pada seragam dan papan nama.
Yang dibutuhkan sistem hanya kode batch, sehingga keluarannya disaring dengan
daftar-izin.

**Bug yang ditemukan saat menguji:** pola pertama pada config tidak cocok
dengan format kode batch yang dicontohkan kontrak API proyek ini sendiri, yaitu
`B240815-021`, karena hanya mengizinkan satu blok angka tanpa pemisah di
tengahnya. Pola diperbaiki dan diuji ulang.

| Masukan | Hasil |
|---|---|
| `B240815-021` | lolos |
| `ABC1234567` | lolos |
| `15/08/26` | lolos |
| `LOT-4F2A91` | lolos |
| `Budi Santoso` | dibuang |
| `PT Contoh Sejahtera` | dibuang |
| `Jl. Merdeka 17` | dibuang |

OCR sendiri belum terpasang pada pipeline, sehingga penyaring ini siap tetapi
belum menerima masukan sungguhan.

Diuji oleh `test_ocr_hanya_meloloskan_kode_batch`,
`test_ocr_membuang_nama_orang`, dan `test_ocr_membuang_alamat_dan_nama_perusahaan`.

---

## 6. Catatan hanya-hash

Sistem tetap dapat menjawab "apakah gambar ini pernah diperiksa" tanpa
menyimpan gambarnya, cukup lewat SHA-256. Sidik jari itu membuktikan dua gambar
sama tetapi tidak dapat dikembalikan menjadi gambar.

Satu baris catatan memuat: sidik jari, keputusan, latensi, jumlah wajah yang
diburamkan, status pembersihan metadata, dan waktu. Tidak ada piksel.

Diuji oleh `test_audit_tidak_memuat_gambar` dan
`test_hash_berbeda_untuk_gambar_berbeda`.

---

## 7. Pemetaan ke UU PDP No. 27 Tahun 2022

| Prinsip | Pasal | Bagaimana dipenuhi |
|---|---|---|
| Pembatasan tujuan | 16 ayat 2 | Hanya piksel produk yang diproses; metadata dibuang sebelum apa pun |
| Minimalisasi data | 16 ayat 2 | Gambar tidak disimpan; catatan hanya berisi sidik jari |
| Data biometrik | 4 ayat 2 | Wajah diburamkan sebelum mencapai model |
| Akuntabilitas | 20 | Jejak audit berbasis hash yang dapat diverifikasi |
| Keamanan pemrosesan | 35 | Pemrosesan sepenuhnya lokal, tanpa panggilan jaringan |

---

## 8. Biaya dan keterbatasan

Lapisan privasi menambah sekitar 23 milidetik per gambar, dari 140 menjadi 163
milidetik. Sebagian besar dipakai peredaman wajah.

Yang belum dikerjakan:

1. Pengujian membership inference terhadap bobot model. Disebut di PRIVACY.md
   sebagai rencana dan tetap menjadi rencana.
2. Verifikasi otomatis bahwa tidak ada panggilan jaringan saat inferensi.
   Saat ini bersandar pada pembacaan kode, bukan pengujian.
3. OCR belum terpasang sehingga penyaringnya belum menerima masukan sungguhan.
