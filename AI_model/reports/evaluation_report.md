# Laporan Evaluasi Modul AI VisionQC

Ringkasan seluruh angka hasil pengukuran, disusun sebagai bahan langsung untuk
bab Metodologi dan bab Kesimpulan proposal. Setiap angka berasal dari run yang
benar-benar dijalankan; berkas sumbernya disebutkan pada tiap bagian.

Tanggal penyusunan: 18 Agustus 2026.

---

## 1. Ringkasan untuk pembaca yang terburu-buru

| Pertanyaan | Jawaban |
|---|---|
| Apakah model benar-benar di-fine-tune? | Ya. Recall deteksi naik dari 0,0000 menjadi 0,6933 dengan p = 1,52 x 10^-12 |
| Apakah peningkatannya signifikan? | Ya, uji McNemar pada deteksi dan segmentasi, keduanya p jauh di bawah 0,001 |
| Apakah sistem tahu kapan harus ragu? | Ya, lewat conformal prediction dengan cakupan terukur 0,9661 |
| Apakah ambangnya dapat dipertahankan? | Ya, ambang anomali berasal dari pemodelan ekor dan lolos uji kecocokan |
| Apakah cukup cepat untuk demo? | Ya, 153 ms untuk kedua model inti di CPU |
| Apa kelemahan terbesarnya? | Dataset hanya memuat tujuh gambar normal per split |

---

## 2. Dataset

Sumber: `reports/metrics/dataset_stats.json`.

414 gambar terbagi 291 latih, 64 validasi, dan 59 uji, memuat 643 instance
cacat dari empat kategori produk pangan dan minuman. Uji keselarasan distribusi
kelas antar split menghasilkan p = 0,5194 sehingga pembagiannya sah.

Rasio ketimpangan 20,31 dan entropi ternormalisasi 0,7556 belum memenuhi
target. Keduanya disebabkan kelangkaan kelas kontaminasi yang hanya memiliki 22
instance dan deformasi yang hanya 16.

Mask VisA ternyata menyimpan kode jenis cacat pada nilai pikselnya, tetapi
pemetaan kode ke jenis tidak ikut didistribusikan. Pemetaan itu dipulihkan
secara empiris dan diuji ulang pada seluruh gambar multi-label dengan hasil
120 dari 120 cocok. Tanpa langkah tersebut VisA hanya dapat dipakai sebagai
data biner normal atau anomali.

---

## 3. Bukti fine-tuning

Sumber: `reports/metrics/detection_comparison.json` dan
`segmentation_comparison.json`.

| Model | Metrik | Sebelum | Sesudah | Uji McNemar |
|---|---|---|---|---|
| Deteksi | Recall | 0,0000 | **0,6933** [0,5793 ; 0,7989] | 52 membaik, 0 memburuk, p = 1,52 x 10^-12 |
| Deteksi | MCC tingkat gambar | 0,090 | **0,719** | Cohen's h = 1,968 |
| Segmentasi | Recall mask | 0,0000 | **0,6000** [0,4681 ; 0,7101] | 45 membaik, 0 memburuk, p = 5,41 x 10^-11 |
| Segmentasi | IoU mask | 0,0000 | **0,7847** | Cohen's h = 1,772 |
| Segmentasi | Galat luas cacat | 15,37 poin persen | **0,37 poin persen** | - |

Baseline juga dinilai tanpa mempedulikan kelas untuk memastikan angka nol bukan
galat pengukuran. Hasilnya 0,0133, artinya model pra-latih hanya menemukan satu
dari 75 lokasi cacat dan memang tidak melihat cacatnya.

---

## 4. Deteksi anomali

Sumber: `reports/metrics/anomaly_results.json`.

Model dilatih hanya dari gambar normal sehingga cacat jenis baru tetap
terjaring. AUROC tingkat gambar berkisar 0,8476 sampai 0,9967 pada lima
kategori.

Ambang tidak dipilih tangan melainkan diturunkan dengan memodelkan ekor sebaran
skor memakai sebaran Pareto Tergeneralisasi. Uji Kolmogorov-Smirnov tidak
menolak kecocokan pada kategori mana pun, dengan p antara 0,593 dan 0,978.

Recall pada titik operasi satu persen berkisar 0,14 sampai 1,00. Rendahnya
recall pada sebagian kategori bukan kegagalan model melainkan akibat titik
operasi yang terlalu ketat untuk konteks pangan.

---

## 5. Lapisan keputusan

Sumber: `reports/metrics/calibration_results.json`.

| Bagian | Hasil |
|---|---|
| Kalibrasi | ECE 0,3222 menjadi **0,0391** lewat Platt scaling |
| Conformal | cakupan **0,9661** terhadap jaminan 0,95 |
| Ditahan ke manusia | 8,5 persen gambar |
| Ambang biaya | tidak menggeneralisasi; ambang operasi tetap 0,50 |

Temperature scaling dicoba lebih dulu dan justru memperburuk ECE menjadi
0,4038. Penyebabnya model kurang percaya diri, bukan terlalu percaya diri,
karena 88 persen data uji memang cacat. Temperature scaling tanpa intersep
tidak dapat menggeser skor ke proporsi kelas yang sebenarnya.

---

## 6. Latensi

Sumber: `reports/metrics/latency_benchmark.json`.

| Model | Median | Persentil ke-95 |
|---|---|---|
| Deteksi ONNX di CPU | 62,5 ms | 119,6 ms |
| Segmentasi ONNX di CPU | 90,9 ms | 140,3 ms |
| Kedua model | **153,4 ms** | - |

Inspeksi utuh terukur sekitar 140 ms per gambar setelah model dimuat, masih
jauh di bawah sasaran satu detik.

---

## 7. Keterbatasan yang harus disebutkan di proposal

Menyembunyikan bagian ini justru berisiko, karena penilai yang memahami
statistik akan menemukannya sendiri.

1. **Test set hanya memuat tujuh gambar normal.** Ini akar dari hampir seluruh
   keterbatasan lain: cakupan conformal kelas normal hanya 0,714, letak minimum
   biaya tidak stabil, dan gambar bersih justru ditahan alih-alih diloloskan.
2. **Recall kelas kontaminasi dan deformasi tidak informatif** karena
   dukungannya hanya tiga dan dua instance.
3. **Model anomali yang dipakai adalah PaDiM**, cadangan yang tercantum di
   config, bukan EfficientAD yang direncanakan sebagai pilihan utama.
4. **Prevalensi cacat produksi tiga persen adalah asumsi**, belum diverifikasi
   terhadap data lini produksi sungguhan.
5. **Segmentasi belum tentu konvergen**; pelatihan berhenti karena batas epoch
   tercapai, bukan karena metriknya berhenti membaik.

---

## 8. Rencana perbaikan untuk tahap Final

1. Menaikkan jumlah gambar normal secara drastis, karena satu perubahan itu
   memperbaiki cakupan conformal, kestabilan ambang biaya, dan perilaku PASS
   sekaligus.
2. Menjalankan generator cacat sintetik untuk mengangkat kelas kontaminasi dan
   deformasi.
3. Mencoba EfficientAD sebagai pengganti PaDiM.
4. Melatih segmentasi lebih lama sampai benar-benar berhenti membaik.
5. Memverifikasi prevalensi cacat terhadap data lini produksi nyata.
