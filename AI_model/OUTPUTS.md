# OUTPUTS.md - Peta Keluaran Tiap Langkah

Satu halaman untuk menjawab: **langkah mana menghasilkan apa, dan di mana
berkasnya**. Isi halaman ini diperiksa terhadap keadaan disk yang sebenarnya,
bukan disalin dari rencana.

Dokumen lain punya peran berbeda:

| Dokumen | Menjawab |
|---|---|
| [AI_MODEL_PLAN.md](./AI_MODEL_PLAN.md) | apa rencananya dan bagaimana urutannya |
| [EXPERIMENTS.md](./EXPERIMENTS.md) | berapa angkanya dan seberapa dapat dipercaya |
| [OUTPUTS.md](./OUTPUTS.md) | berkas apa yang dihasilkan dan di mana letaknya |
| [README.md](./README.md) | bagaimana cara menjalankannya |

---

## 1. Ringkasan

| Step | Nama | Status | Keluaran utama |
|---|---|---|---|
| 1 | Fondasi & Perencanaan | selesai | 6 dokumen, 3 config, `requirements.txt` |
| 2 | Akuisisi & Preprocessing Dataset | selesai | 10 modul data, 2 skrip, 3 dataset olahan, `dataset_stats.json`, 2 lembar kontak |
| 3 | Generator Cacat Sintetik | ditunda | belum ada |
| 4 | Baseline + Fine-Tune Detection | selesai | 7 modul training & evaluasi, 2 skrip, bobot terlatih, `detection_comparison.json`, 4 gambar |
| 5 | Fine-Tune Segmentation | selesai | 1 modul evaluasi mask, 1 skrip, bobot terlatih, `segmentation_comparison.json`, 3 gambar |
| 6 | Anomaly Detection + Ambang EVT | selesai | 2 modul, 1 skrip, 5 model, `anomaly_results.json`, 5 gambar |
| 7 | Lapisan Statistik & Kalibrasi | selesai | 3 modul, 1 skrip, `calibration_results.json`, 3 gambar |
| 8 | Lapisan Privasi | selesai | 5 modul privasi, 11 uji, `privacy_audit.md` |
| 9 | Decision Engine + Ekspor ONNX | selesai | mesin keputusan, ekspor ONNX, tolok ukur latensi |
| 10 | Integrasi & Laporan Evaluasi | selesai | `run_inspection()`, skema Pydantic, anotasi, pipeline, 12 uji, laporan evaluasi |

---

## 2. Tempat Penyimpanan

| Lokasi | Isi | Ikut git? |
|---|---|---|
| `configs/` | parameter yang dapat diubah manusia | ya |
| `src/visionqc_ai/` | kode modul | ya |
| `scripts/` | entrypoint yang dijalankan manual | ya |
| `reports/metrics/` | angka hasil pengukuran, format JSON | **ya, sengaja** |
| `reports/figures/` | bukti visual, format PNG | **ya, sengaja** |
| `data/raw/` | dataset mentah hasil unduhan | tidak, terlalu besar |
| `data/processed/` | dataset siap latih | tidak, dapat dibuat ulang |
| `models/` | bobot model | tidak, didistribusi lewat GitHub Release |

Isi `reports/` sengaja ikut di-commit karena itulah bukti evaluasi yang
diminta panitia. Yang dikecualikan hanya berkas berukuran besar yang dapat
dihasilkan ulang dari kode dan dataset publik.

Bobot model dikecualikan dari git tetapi **tidak** dibiarkan hilang. Berkasnya
diterbitkan sebagai GitHub Release `models-v1.0.0`, sementara daftar resmi
beserta sidik jari SHA-256-nya tetap di dalam git pada `models/models.json`.
Dengan begitu keaslian setiap unduhan dapat diperiksa tanpa perlu menyimpan
berkas besar di dalam riwayat git.

| Berkas | Ukuran | Peran |
|---|---|---|
| `models/models.json` | kecil, di dalam git | daftar resmi dan sidik jari |
| `scripts/download_models.py` | kecil, di dalam git | pengunduh dan pemeriksa |
| `models/onnx/*.onnx` | 21,2 MB, di luar git | diunduh dari Release |

---

## 3. Step 1 - Fondasi & Perencanaan

Menetapkan arah kerja. Belum ada kode yang dapat dijalankan.

| Berkas | Isi |
|---|---|
| `AI_MODEL_PLAN.md` | peta 10 langkah beserta prasyaratnya |
| `STATISTICS.md` | seluruh rumus statistik dan alasan pemakaiannya |
| `PRIVACY.md` | analisis ancaman privasi dan mitigasinya |
| `DATASET_REQUIREMENTS.md` | dataset yang dibutuhkan beserta lisensinya |
| `EXPERIMENTS.md` | kerangka tabel bukti fine-tuning |
| `configs/training.yaml` | hyperparameter, disetel untuk RTX 3050 4 GB |
| `configs/inference.yaml` | ambang keputusan, statis saat runtime |
| `requirements.txt` | daftar dependensi |

---

## 4. Step 2 - Akuisisi & Preprocessing Dataset

### Kode

| Berkas | Tanggung jawab |
|---|---|
| `src/visionqc_ai/data/taxonomy.py` | pemetaan 38 label mentah ke 6 kelas |
| `src/visionqc_ai/data/sources.py` | registri kategori dan tata letak folder mentah |
| `src/visionqc_ai/data/visa_codes.py` | rekonstruksi kode jenis cacat pada mask VisA |
| `src/visionqc_ai/data/mask_utils.py` | mask piksel menjadi kotak pembatas dan poligon |
| `src/visionqc_ai/data/records.py` | representasi antara |
| `src/visionqc_ai/data/convert_mvtec.py` | konversi MVTec AD |
| `src/visionqc_ai/data/convert_visa.py` | konversi VisA |
| `src/visionqc_ai/data/convert_goodsad.py` | konversi PKU-GoodsAD (JPEG, mask tanpa akhiran `_mask`) |
| `src/visionqc_ai/data/split.py` | split terstratifikasi, seed terkunci |
| `src/visionqc_ai/data/validate.py` | validasi statistik dataset |
| `src/visionqc_ai/data/writer.py` | penulisan tiga tata letak keluaran |
| `scripts/prepare_dataset.py` | menjalankan seluruh rantai |
| `scripts/download_models.py` | mengunduh dan memverifikasi bobot model |
| `scripts/preview_dataset.py` | lembar kontak pemeriksaan anotasi |
| `configs/dataset.yaml` | kategori, preprocessing, split, ambang validasi |

### Dataset olahan (tidak masuk git)

| Folder | Isi | Ukuran |
|---|---|---|
| `data/processed/detect/` | 2.039 gambar + label kotak pembatas + `data.yaml` | ~230 MB |
| `data/processed/seg/` | label poligon; gambarnya hard link ke `detect` | ~5 MB |
| `data/processed/anomaly/` | gambar normal dan cacat per kategori, plus `combined/` | ~1,2 GB |

Gambar pada `seg/` sengaja dibuat sebagai hard link ke `detect/` supaya tidak
ada salinan kedua yang memakan disk.

### Bukti yang tersimpan

| Berkas | Isi |
|---|---|
| `reports/metrics/dataset_stats.json` | seluruh angka: jumlah per kategori, rekonstruksi kode VisA, hasil validasi, instance yang dibuang |
| `reports/figures/dataset_samples_train.png` | anotasi digambar ulang di atas gambar olahan, split train |
| `reports/figures/dataset_samples_test.png` | idem, split test |

### Angka kunci

Diperbarui 21 Agustus 2026 setelah PKU-GoodsAD masuk.

2.039 gambar (1.428 train, 310 val, 301 test) dengan 2.239 instance cacat pada
**6 kelas**. Gambar cacat naik dari 363 menjadi 1.776, yaitu 4,9 kali lipat.

Uji keselarasan distribusi antar split lolos (p = 0,9915), entropi 0,8740
**lulus**, ukuran test set 321 instance **lulus**. Yang belum lulus tinggal
rasio ketimpangan, 35,55, dan itu seluruhnya disebabkan `kotor` yang tetap 22
instance sementara lima kelas lain tumbuh berkali lipat.
Rincian di [EXPERIMENTS.md bagian 2](./EXPERIMENTS.md).

---

## 5. Step 4 - Baseline + Fine-Tune Detection

### Kode

| Berkas | Tanggung jawab |
|---|---|
| `src/visionqc_ai/training/train_detect.py` | pembungkus fine-tuning Ultralytics |
| `src/visionqc_ai/evaluation/matching.py` | pencocokan prediksi dengan ground truth |
| `src/visionqc_ai/evaluation/metrics.py` | metrik tingkat instance dan tingkat gambar |
| `src/visionqc_ai/evaluation/bootstrap.py` | selang kepercayaan BCa dan Wilson |
| `src/visionqc_ai/evaluation/significance.py` | uji McNemar dan Cohen's h |
| `src/visionqc_ai/evaluation/detection_eval.py` | menjalankan model pada satu split |
| `scripts/train_detection.py` | menjalankan fine-tuning |
| `scripts/compare_detection.py` | membandingkan baseline dengan hasil fine-tuning |

### Bobot model (tidak masuk git)

| Berkas | Isi |
|---|---|
| `models/finetuned/detect/weights/best.pt` | bobot terbaik, epoch 137 |
| `models/finetuned/detect/weights/last.pt` | bobot epoch terakhir, 161 |
| `models/finetuned/detect/results.csv` | metrik per epoch |
| `yolo11n.pt` | bobot pra-latih COCO, diunduh otomatis |

### Bukti yang tersimpan

| Berkas | Isi |
|---|---|
| `reports/metrics/training_detect.json` | ringkasan run: epoch diminta, epoch dijalankan, lokasi bobot |
| `reports/metrics/detection_comparison.json` | seluruh metrik sebelum dan sesudah, selang kepercayaan, McNemar, Cohen's h, recall per kelas, latensi |
| `reports/figures/detection_comparison.png` | metrik dengan galat 95% dan recall per kelas |
| `reports/figures/training_curves.png` | kurva galat pelatihan dan metrik validasi |
| `reports/figures/detection_confusion_matrix.png` | confusion matrix ternormalisasi |
| `reports/figures/detection_pr_curve.png` | kurva precision-recall |

### Angka kunci

Recall kelas cacat naik dari 0,0000 menjadi 0,6933 [0,5793 ; 0,7989].
MCC tingkat gambar naik dari 0,090 menjadi 0,719.
Uji McNemar: 52 instance membaik, 0 memburuk, p = 1,52 x 10^-12.
Rincian beserta keterbatasannya di [EXPERIMENTS.md bagian 3](./EXPERIMENTS.md).

---

## 6. Step 5 - Fine-Tune Segmentation

### Kode

| Berkas | Tanggung jawab |
|---|---|
| `src/visionqc_ai/evaluation/segmentation_eval.py` | pencocokan IoU mask dan perhitungan luas cacat |
| `scripts/compare_segmentation.py` | membandingkan baseline dengan hasil fine-tuning |

Training memakai ulang `scripts/train_detection.py` dengan penanda bagian
`segmentation`, sehingga tidak ada kode training yang digandakan:

```bash
python scripts/train_detection.py --data data/processed/seg/data.yaml \
    --name seg --section segmentation
```

### Bobot model (tidak masuk git)

| Berkas | Isi |
|---|---|
| `models/finetuned/seg/weights/best.pt` | bobot terbaik, epoch 203 |
| `models/finetuned/seg/weights/last.pt` | bobot epoch terakhir, 250 |
| `models/finetuned/seg/results.csv` | metrik per epoch |
| `yolo11n-seg.pt` | bobot pra-latih COCO, diunduh otomatis |

### Bukti yang tersimpan

| Berkas | Isi |
|---|---|
| `reports/metrics/training_seg.json` | ringkasan run |
| `reports/metrics/segmentation_comparison.json` | metrik mask, IoU, galat luas cacat, McNemar, recall per kelas |
| `reports/figures/segmentation_comparison.png` | metrik, sebaran IoU mask, ketepatan estimasi luas |
| `reports/figures/segmentation_pr_curve.png` | kurva precision-recall mask |
| `reports/figures/segmentation_confusion_matrix.png` | confusion matrix ternormalisasi |

### Angka kunci

Recall mask naik dari 0,0000 menjadi 0,6000 [0,4681 ; 0,7101], IoU mask rerata
0,7847, dan MCC tingkat gambar 0,838. Galat estimasi luas cacat turun dari
15,37 menjadi **0,37 poin persen**. Uji McNemar: 45 instance membaik, 0
memburuk, p = 5,41 x 10^-11.


---

## 7. Step 6 - Anomaly Detection dan Ambang EVT

### Kode

| Berkas | Tanggung jawab |
|---|---|
| `src/visionqc_ai/statistics/evt.py` | pencocokan GPD pada ekor dan perhitungan ambang |
| `src/visionqc_ai/training/train_anomaly.py` | pembungkus anomalib, pembatas gambar latih |
| `scripts/build_combined_anomaly.py` | menggabungkan seluruh kategori menjadi `combined`, yaitu dataset yang melatih model anomali yang benar-benar dipakai saat penyajian |
| `scripts/train_anomaly.py` | melatih tiap kategori dan menurunkan ambangnya |

### Bobot model (tidak masuk git)

`models/finetuned/anomaly/` berisi checkpoint PaDiM untuk kelima kategori.

### Bukti yang tersimpan

| Berkas | Isi |
|---|---|
| `reports/metrics/anomaly_results.json` | AUROC, ambang EVT, parameter GPD, laju alarm palsu, recall per kategori |
| `reports/figures/anomaly_bottle.png` | sebaran skor dan diagram kuantil-kuantil GPD |
| `reports/figures/anomaly_chewinggum.png` | idem |
| `reports/figures/anomaly_cashew.png` | idem |
| `reports/figures/anomaly_pipe_fryum.png` | idem |
| `reports/figures/anomaly_fryum.png` | idem |

### Angka kunci

AUROC 0,848 sampai 0,997. Ambang EVT lolos uji kecocokan Kolmogorov-Smirnov di
seluruh kategori dengan p antara 0,593 dan 0,978. Recall pada titik operasi 1
persen berkisar 0,14 sampai 1,00; rendahnya recall pada sebagian kategori
menjadi dasar peninjauan titik operasi di Step 7.


---

## 8. Step 7 - Kalibrasi, Conformal, dan Ambang Biaya

### Kode

| Berkas | Tanggung jawab |
|---|---|
| `src/visionqc_ai/statistics/calibration.py` | ECE, MCE, Brier, temperature dan Platt scaling |
| `src/visionqc_ai/statistics/conformal.py` | conformal split dan Mondrian, pemeriksaan cakupan |
| `src/visionqc_ai/statistics/cost_sensitive.py` | ambang Bayes, kurva biaya, koreksi prevalensi |
| `scripts/calibrate_decision.py` | menjalankan ketiganya dan menulis hasilnya |

### Bukti yang tersimpan

| Berkas | Isi |
|---|---|
| `reports/metrics/calibration_results.json` | perbandingan tiga metode kalibrasi, kuantil conformal, cakupan, kurva biaya |
| `reports/figures/reliability_diagram.png` | keandalan sebelum dan sesudah kalibrasi |
| `reports/figures/conformal_coverage.png` | cakupan empiris dan komposisi keputusan |
| `reports/figures/cost_curve.png` | biaya per unit terhadap ambang keputusan |

### Angka kunci

Set kalibrasi diperluas dari 7 menjadi 307 gambar normal per sisi, memakai
gambar yang tidak pernah dilihat model. Setelah itu Platt scaling menurunkan
ECE menjadi 0,0145, cakupan conformal kelas normal naik dari 0,7143 menjadi
0,9381, gambar yang ditahan turun dari 83 persen menjadi 2,8 persen, dan
ambang biaya menggeneralisasi dengan penghematan 21,4 persen.

Perubahan itu sekaligus memperbaiki cacat yang paling menghambat demo: gambar
bersih kini menghasilkan PASS, bukan REVIEW.


---

## 9. Step 9 - Mesin Keputusan dan Ekspor ONNX

### Kode

| Berkas | Tanggung jawab |
|---|---|
| `src/visionqc_ai/inference/decision.py` | mesin keputusan; mode biner PASS/REJECT dan mode tiga kelas |
| `src/visionqc_ai/export/onnx_export.py` | ekspor ONNX dan pengukuran latensi |
| `scripts/export_onnx.py` | menjalankan ekspor dan tolok ukur |

### Artefak penyajian (tidak masuk git)

| Berkas | Ukuran |
|---|---|
| `models/onnx/yolo11n-defect.onnx` | 10,12 MB |
| `models/onnx/yolo11n-seg-defect.onnx` | 11,09 MB |

### Bukti yang tersimpan

| Berkas | Isi |
|---|---|
| `reports/metrics/latency_benchmark.json` | latensi median dan persentil ke-95 di CPU, ukuran berkas, spesifikasi mesin |

### Angka kunci

Latensi ONNX di CPU: detect 62,5 ms dan seg 90,9 ms median, total 153,4 ms.
Sasaran keseluruhan satu detik per gambar masih menyisakan ruang.

Mesin keputusan menggabungkan peluang terkalibrasi Platt, himpunan prediksi
conformal, ambang luas cacat, bobot keparahan kelas, dan ambang anomali hasil
teori nilai ekstrem. Seluruh nilainya statis dari `configs/inference.yaml`.

### Penilaian anomali

`src/visionqc_ai/inference/anomaly.py` menilai anomali di atas berkas ONNX dan
tersambung ke pipeline, sehingga jalur anomali benar-benar ikut menentukan
keputusan. Modelnya dilatih pada gabungan empat kategori; model satu kategori
tidak dapat dipakai karena bagi model yang hanya mengenal botol, kacang mete
pun terbaca sebagai anomali.

`scripts/export_anomaly.py` mengekspor dan **memverifikasi** hasilnya terhadap
PyTorch, karena berkas yang terbentuk belum tentu menghasilkan skor yang sama.


---

## 10. Step 10 - Integrasi dan Laporan Evaluasi

### Kode

| Berkas | Tanggung jawab |
|---|---|
| `src/visionqc_ai/__init__.py` | mengekspor `run_inspection` dan `InspectionResult` |
| `src/visionqc_ai/schemas.py` | kontrak Pydantic dengan Backend |
| `src/visionqc_ai/inference/annotate.py` | penggambaran kotak, mask, dan pita keputusan |
| `src/visionqc_ai/inference/pipeline.py` | orkestrasi dari byte gambar sampai hasil |
| `scripts/evaluate_pipeline.py` | evaluasi PASS/REJECT ujung-ke-ujung lewat ONNX yang benar-benar dipakai Backend, bukan bobot PyTorch |
| `pyproject.toml` | agar Backend dapat memasang modul ini |
| `tests/` | 46 uji: mesin keputusan, kontrak data, privasi, integrasi, dan warna anotasi |

### Cara Backend memakainya

```python
from visionqc_ai import run_inspection, InspectionResult

result: InspectionResult = run_inspection(image_bytes)
```

Model dimuat sekali pada pemanggilan pertama dan dipakai ulang sesudahnya.

### Bukti yang tersimpan

| Berkas | Isi |
|---|---|
| `reports/evaluation_report.md` | ringkasan seluruh angka, bahan langsung untuk proposal |
| `reports/figures/inspection_example.png` | contoh keluaran beranotasi yang dilihat pengguna |

### Angka kunci

Inspeksi utuh sekitar 140 ms per gambar setelah model dimuat, jauh di bawah
sasaran satu detik. Seluruh 12 uji lolos.


---

## 11. Step 8 - Lapisan Privasi

### Kode

| Berkas | Tanggung jawab |
|---|---|
| `src/visionqc_ai/privacy/exif.py` | membuang seluruh metadata lewat pengodean ulang |
| `src/visionqc_ai/privacy/face_blur.py` | mendeteksi dan memburamkan wajah dengan YuNet |
| `src/visionqc_ai/privacy/ephemeral.py` | menimpa buffer gambar dengan nol setelah dipakai |
| `src/visionqc_ai/privacy/ocr_filter.py` | daftar-izin pola kode batch |
| `src/visionqc_ai/privacy/audit.py` | catatan hanya berisi SHA-256 |

Lapisan ini berjalan di dalam `inference/pipeline.py` **sebelum** gambar
mencapai model.

### Bukti yang tersimpan

| Berkas | Isi |
|---|---|
| `reports/privacy_audit.md` | pemetaan tiap klaim privasi ke berkas dan uji yang membuktikannya |

### Angka kunci

Menambah sekitar 23 milidetik per gambar, dari 140 menjadi 163 milidetik.
Sebelas uji privasi lolos, seluruhnya memakai gambar sintetik.


---

## 12. Audit Kesiapan Sambung ke Backend (20 Agu 2026)

Pemeriksaan menyeluruh seluruh berkas modul menjelang penyerahan ke Backend.
Tujuannya bukan menambah fitur, melainkan menguji apakah yang sudah ada
benar-benar berjalan ketika dipakai sebuah layanan.

### Yang ditemukan dan diperbaiki

| Temuan | Akibat bila dibiarkan | Perbaikan |
|---|---|---|
| Pendeteksi wajah dipakai bersama tanpa kunci | Backend gugur begitu dua pengguna mengunggah bersamaan | Kunci pada `privacy/face_blur.py` dan pada `inspect()` |
| `max_file_size_mb`, `allowed_formats`, `max_dimension` tertulis di config tetapi tidak pernah diperiksa | Berkas 18 MB, BMP, dan gambar 5000 piksel diterima | Modul `inference/validation.py` |
| Galat masukan tidak dapat dibedakan dari galat sistem | Backend tidak dapat memilih antara 400 dan 500 | `InvalidImageError` |
| `warmup_on_startup: true` tidak pernah dijalankan | Permintaan pertama memakan 4,8 detik | `InspectionPipeline.warmup()` |
| Akar proyek dihitung dari kedalaman folder sumber | Salah menunjuk begitu paket dipasang ke tempat lain | `default_project_root()` dan `VISIONQC_ROOT` |
| `Defect.area_pct` selalu kosong | Medan kontrak tidak pernah terisi | Irisan mask segmentasi dengan wilayah kotak |
| `ephemeral_buffers: true` tidak pernah dipanggil pipeline | Klaim privasi tidak sesuai kode | Buffer ditimpa setelah keluaran terbentuk |

Contoh pemakaian pada README semula menuliskan `run_inspection(image_bytes,
config)`, tanda tangan yang tidak pernah ada. Sudah diperbaiki dan dijalankan
apa adanya sebagai bagian dari audit.

### Bukti

| Pemeriksaan | Hasil |
|---|---|
| Uji otomatis | 35 lolos, 11 di antaranya baru pada `tests/test_integration.py` |
| Uji regresi kunci pendeteksi wajah | gagal ketika kunci dilepas, lolos ketika dipasang |
| Delapan permintaan bersamaan pada empat thread | hasil identik dengan pemanggilan berurutan, 8 dari 8 |
| Ekspor split uji 59 gambar | tidak bergeser: normal 6/0/1, cacat 2/33/17 |
| SHA-256 keempat model terhadap manifest | cocok seluruhnya |
| Unduhan release tanpa kredensial | HTTP 200 untuk keempat berkas |
| Impor dari direktori lain sebagai paket terpasang | berhasil |
| Latensi setelah warmup | sekitar 230 milidetik per gambar |

---

## 13. Keputusan Biner Tanpa REVIEW (20 Agu 2026)

### Kode

| Berkas | Tanggung jawab |
|---|---|
| `scripts/select_binary_threshold.py` | memilih ambang biner pada set kalibrasi |
| `src/visionqc_ai/inference/decision.py` | jalur `_decide_binary` |

### Bukti yang tersimpan

| Berkas | Isi |
|---|---|
| `reports/metrics/binary_threshold.json` | ambang terpilih, kurva biaya, hasil kalibrasi dan uji |

### Angka kunci

Ambang 0,22, dipilih pada set kalibrasi dengan meminimumkan biaya. Pada uji
diperluas 359 gambar: recall 0,9615, specificity 0,9642, akurasi 0,9638,
**REVIEW nol**. Ambang lama 0,60 hanya mencapai recall 0,5769.

---

## 14. Menghasilkan Ulang Semuanya

Dari repo bersih, dengan `data/raw/` sudah terisi dataset publik:

```bash
python scripts/download_models.py     # bobot model, sebelum apa pun

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python scripts/prepare_dataset.py       # Step 2, sekitar 4 menit

# Penyeimbangan set latih. URUTANNYA MENGIKAT: keduanya menulis ke dalam
# split latih yang baru saja dibangun, jadi menjalankan ulang
# prepare_dataset.py sesudah ini akan menghapus hasilnya.
python scripts/generate_synthetic_kotor.py   # cacat kotor sintetik
python scripts/mine_hard_negatives.py        # gambar bagus yang salah dituduh

python scripts/preview_dataset.py --split train
python scripts/preview_dataset.py --split test

python scripts/train_detection.py       # Step 4, sekitar 22 menit di RTX 3050
python scripts/compare_detection.py     # evaluasi dan uji signifikansi

python scripts/train_detection.py --data data/processed/seg/data.yaml \
    --name seg --section segmentation   # Step 5, sekitar 67 menit
python scripts/compare_segmentation.py

python scripts/train_anomaly.py         # Step 6, sekitar 2 menit per kategori
python scripts/calibrate_decision.py    # Step 7, sekitar 1 menit
python scripts/export_onnx.py           # Step 9, sekitar 2 menit
```

Seed dikunci di `configs/dataset.yaml`, sehingga pembagian split akan sama
persis pada setiap pengulangan. Hasil training dapat berbeda tipis karena
sifat nondeterministik operasi CUDA tertentu.

---

## 15. Yang Belum Ada

Belum ada satu pun keluaran untuk Step 3 (generator cacat sintetik). Seluruh sel
metrik yang berkaitan di `EXPERIMENTS.md` masih bertuliskan `belum diukur` dan
tidak boleh diisi sebelum ada run yang benar-benar dijalankan.
