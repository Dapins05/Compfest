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
| 8 | Lapisan Privasi | belum | belum ada |
| 9 | Decision Engine + Ekspor ONNX | sebagian | mesin keputusan, ekspor ONNX, tolok ukur latensi; anotasi dan pipeline digeser ke Step 10 |
| 10 | Integrasi & Laporan Evaluasi | belum | belum ada |

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
| `models/` | bobot model | tidak, terlalu besar |

Isi `reports/` sengaja ikut di-commit karena itulah bukti evaluasi yang
diminta panitia. Yang dikecualikan hanya berkas berukuran besar yang dapat
dihasilkan ulang dari kode dan dataset publik.

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
| `src/visionqc_ai/data/taxonomy.py` | pemetaan 30+ label mentah ke 5 kelas |
| `src/visionqc_ai/data/sources.py` | registri kategori dan tata letak folder mentah |
| `src/visionqc_ai/data/visa_codes.py` | rekonstruksi kode jenis cacat pada mask VisA |
| `src/visionqc_ai/data/mask_utils.py` | mask piksel menjadi kotak pembatas dan poligon |
| `src/visionqc_ai/data/records.py` | representasi antara |
| `src/visionqc_ai/data/convert_mvtec.py` | konversi MVTec AD |
| `src/visionqc_ai/data/convert_visa.py` | konversi VisA |
| `src/visionqc_ai/data/split.py` | split terstratifikasi, seed terkunci |
| `src/visionqc_ai/data/validate.py` | validasi statistik dataset |
| `src/visionqc_ai/data/writer.py` | penulisan tiga tata letak keluaran |
| `scripts/prepare_dataset.py` | menjalankan seluruh rantai |
| `scripts/preview_dataset.py` | lembar kontak pemeriksaan anotasi |
| `configs/dataset.yaml` | kategori, preprocessing, split, ambang validasi |

### Dataset olahan (tidak masuk git)

| Folder | Isi | Ukuran |
|---|---|---|
| `data/processed/detect/` | 414 gambar + label kotak pembatas + `data.yaml` | 50 MB |
| `data/processed/seg/` | label poligon; gambarnya hard link ke `detect` | 1 MB |
| `data/processed/anomaly/` | gambar normal dan cacat per kategori | 252 MB |

Gambar pada `seg/` sengaja dibuat sebagai hard link ke `detect/` supaya tidak
ada salinan kedua yang memakan disk.

### Bukti yang tersimpan

| Berkas | Isi |
|---|---|
| `reports/metrics/dataset_stats.json` | seluruh angka: jumlah per kategori, rekonstruksi kode VisA, hasil validasi, instance yang dibuang |
| `reports/figures/dataset_samples_train.png` | anotasi digambar ulang di atas gambar olahan, split train |
| `reports/figures/dataset_samples_test.png` | idem, split test |

### Angka kunci

414 gambar (291 train, 64 val, 59 test) dengan 643 instance cacat.
Uji keselarasan distribusi antar split lolos dengan p = 0,5194.
Rasio ketimpangan 20,31 dan entropi 0,7556 **belum** memenuhi target.
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

Platt scaling menurunkan ECE dari 0,3222 menjadi 0,0391. Conformal prediction
mencapai cakupan 0,9661 dengan 8,5 persen gambar diserahkan ke manusia. Ambang
hasil optimasi biaya tidak menggeneralisasi ke split uji sehingga ambang
operasi tetap 0,50.


---

## 9. Step 9 - Mesin Keputusan dan Ekspor ONNX

### Kode

| Berkas | Tanggung jawab |
|---|---|
| `src/visionqc_ai/inference/decision.py` | mesin keputusan PASS, REJECT, REVIEW |
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

### Belum dibuat

`annotate.py` dan `pipeline.py` digeser ke Step 10 bersama `run_inspection()`,
karena ketiganya hanya bermakna bila dirakit sekaligus.


---

## 10. Menghasilkan Ulang Semuanya

Dari repo bersih, dengan `data/raw/` sudah terisi dataset publik:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python scripts/prepare_dataset.py       # Step 2, sekitar 4 menit
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

## 11. Yang Belum Ada

Belum ada satu pun keluaran untuk Step 3, 8, dan 10. Seluruh sel
metrik yang berkaitan di `EXPERIMENTS.md` masih bertuliskan `belum diukur` dan
tidak boleh diisi sebelum ada run yang benar-benar dijalankan.
