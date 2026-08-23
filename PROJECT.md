# PROJECT.md — VisionQC
**AI Smart Manufacturing Quality Control**
COMPFEST 18 · AIC (AI Competition) · Tahap **Penyisihan**

Dokumen teknis utama. Rules pengerjaan → [CLAUDE.md](./CLAUDE.md) · Progres → [LOG.md](./LOG.md)
**Terakhir diperbarui:** 2026-08-18 (struktur repo diselaraskan dengan kenyataan; timeline dijadwal ulang)

---

## KOTAK MERAH — BACA INI DULU

| | |
|---|---|
| **Deadline** | **25 Agustus 2026, 23.55 WIB** |
| **Hari ini** | 18 Agustus 2026 |
| **Sisa waktu** | **7 HARI** |
| **Periode kerja sah** | 17 Juni – 25 Agustus 2026 (dilarang melanjutkan proyek lama) |
| **Standby Discord** | 9 & 10 September 2026, 20.00 — panitia bisa minta live demo, jawab maks. 2 jam |
**Empat berkas wajib. Kurang satu = diskualifikasi:**
1.  Link repo GitHub (**public**, ada `README.md` + `docker compose`)
2.  Video **Proof of Work** — maks. 7 menit, YouTube **unlisted**
3.  Video **Karya Inovasi** — maks. 5 menit, YouTube **public**
4.  Proposal PDF — maks. 20 halaman

>  Versi PROJECT.md sebelumnya (arsitektur queue + dashboard analitik + auth, timeline 8 minggu)
>**melanggar batasan ruang lingkup panitia dan tidak muat di sisa waktu**. Dokumen ini adalah
> hasil perombakan total. Alasannya dirinci di §2.

---

## Daftar Isi

| # | Bagian | Untuk siapa |
|---|---|---|
| 1 | [Ringkasan Proyek](#1-ringkasan-proyek) | semua |
| 2 | [**Batasan Wajib dari Panitia**](#2-batasan-wajib-dari-panitia) | **semua — wajib baca** |
| 3 | [Deliverables & Checklist Submisi](#3-deliverables--checklist-submisi) | semua |
| 4 | [Masalah & Solusi](#4-masalah--solusi) | Anggota 3 (proposal) |
| 5 | [Daftar Fitur](#5-daftar-fitur-versi-penyisihan) | semua |
| 6 | [Tech Stack](#6-tech-stack) | dev |
| 7 | [Arsitektur Sistem](#7-arsitektur-sistem-sinkron) | dev |
| 8 | [Cara Kerja Detail](#8-cara-kerja-detail) | dev |
| 9 | [Kontrak API](#9-kontrak-api) | FE + BE |
| 10 | [Struktur Repo](#10-struktur-repo) | dev |
| 11 | [**Jobdesk Per Anggota**](#11-jobdesk-per-anggota) | semua |
| 12 | [**Workflow Git**](#12-workflow-git) | dev |
| 13 | [Workflow ML & Fine-Tuning](#13-workflow-ml--fine-tuning-wajib) | Anggota 1 |
| 14 | [**Timeline 10 Hari**](#14-timeline-10-hari) | semua |
| 15 | [Rencana Tahap Final](#15-rencana-tahap-final-fitur-yang-diparkir) | semua |
| 16 | [Panduan Proposal](#16-panduan-proposal-20-halaman) | Anggota 3 |
| 17 | [Panduan Video](#17-panduan-video) | Anggota 3 |
| 18 | [Risiko](#18-risiko--mitigasi) | semua |
| 19 | [Setup](#19-setup) | dev |
| 20 | [Keputusan Terbuka](#20-keputusan-terbuka) | semua |

---

## 1. Ringkasan Proyek
**Tema lomba:** AI for Backbone Economy → area **Smart Manufacturing**  (salah satu dari tiga area yang diizinkan: Smart Manufacturing, Smart Logistics, Smart Commerce)
**VisionQC** adalah sistem *quality control* otomatis berbasis computer vision untuk lini produksi.

### Alur inti (versi Penyisihan)

```
┌──────────────────────────────────────────────────────────────────┐
│  Pengguna unggah 1 gambar produk                                 │
│                        ↓                                          │
│  AI memeriksa (sinkron, dalam 1 request):                        │
│     ① YOLO11-detect  → cacat jenis apa? di kotak mana?           │
│     ② YOLO11-seg     → bentuk mask cacat → luas berapa persen?   │
│     ③ PaDiM          → skor anomali (menangkap cacat jenis baru) │
│     ④ PaddleOCR      → kode batch produk (DIMATIKAN, lihat F-07) │
│     ⑤ Decision Engine → PASS / REJECT (mode biner)                │
│                        ↓                                          │
│  Hasil ditampilkan: gambar beranotasi + verdict + alasan + skor  │
└──────────────────────────────────────────────────────────────────┘
```

### Nilai jual (untuk proposal & video)

| # | Keunggulan | Kenapa penting |
|---|---|---|
| 1 | **Bukan sekadar bagus/cacat** | Menunjukkan *jenis*, *lokasi*, dan *luas* cacat — tim produksi tahu harus memperbaiki apa |
| 2 | **Menangkap cacat yang belum pernah dilihat** | PaDiM dilatih **hanya dari barang bagus**, jadi cacat jenis baru tetap terjaring |
| 3 | **Ambang dari biaya, bukan tebakan** | Ambang keputusan diturunkan dengan meminimumkan biaya yang diharapkan pada set kalibrasi, bukan dipilih tangan. Sistem memutuskan sendiri tanpa menahan produk |
| 4 | **Terlacak** | Kode batch dari OCR menghubungkan hasil inspeksi ke produk spesifik |
| 5 | **Reprodusibel** | `docker compose up` — panitia bisa menjalankan sendiri dalam hitungan menit |

---

## 2. BATASAN WAJIB DARI PANITIA

> Sumber: `Persyaratan/image copy 2.png` — *"Ketentuan Batasan Ruang Lingkup MVP"*.
> Panitia menyatakan ruang lingkup proyek penyisihan **"WAJIB HANYA SAMPAI"** batasan berikut.
> Melewatinya = melanggar ketentuan lomba.

### 2.1 Tiga batasan resmi

| Lapisan |  WAJIB HANYA SAMPAI |  DILARANG / TIDAK PERLU |
|---|---|---|
| **Frontend / UI** | Alur interaksi inti: menerima **input tunggal** dari pengguna, menampilkan output AI | Dashboard analitik tingkat lanjut · sistem otentikasi kompleks · halaman riwayat pengguna |
| **Backend & Integrasi** | **Pemrosesan interaksi sinkron**. Fokus agar API/sistem lokal jalan sesuai `README.md` pakai `docker compose` | Background jobs · pipeline pencatatan data otomatis (*automated logging*) · infrastruktur database terdistribusi |
| **Model AI & Algoritma** | **Core inference** dengan parameter **statis** saat demonstrasi | Auto-tuning · skrip pengujian massal (*bulk testing*) · loop umpan balik otomatis |

### 2.2 Apa yang DIBUANG dari rencana sebelumnya, dan kenapa

Versi PROJECT.md terdahulu melanggar batasan di atas. Berikut audit lengkapnya:

| Komponen lama | Vonis | Dasar |
|---|---|---|
| RabbitMQ + Celery worker |  **DIBUANG** | Background jobs — dilarang. Backend wajib sinkron |
| Ingestor sebagai service terpisah |  **DIBUANG** | Bagian dari arsitektur async yang dilarang |
| Redis pub/sub + WebSocket realtime |  **DIBUANG** | Bukan interaksi sinkron request-response |
| PostgreSQL + TimescaleDB + Alembic + SQLAlchemy |  **DIBUANG** | Pencatatan data otomatis + infrastruktur DB — dilarang |
| MinIO object storage |  **DIBUANG** | Bagian dari infrastruktur penyimpanan yang tidak diperlukan |
| Dashboard analitik, grafik tren, Pareto chart |  **DIBUANG** | "tidak perlu dashboard analitik tingkat lanjut" |
| Halaman riwayat inspeksi + galeri cacat |  **DIBUANG** | "tidak perlu halaman riwayat pengguna" |
| Auth JWT + RBAC |  **DIBUANG** | "tidak perlu sistem otentikasi yang kompleks" |
| Antrean review + active learning |  **DIBUANG** | Loop umpan balik otomatis — dilarang |
| Model registry + aktivasi model + shadow mode |  **DIBUANG** | Auto-tuning / parameter tidak statis — dilarang |
| Panel ubah threshold saat runtime |  **DIBUANG** | Parameter wajib **statis** saat demo |
| Retensi otomatis, housekeeping, monitoring Grafana |  **DIBUANG** | Automated pipeline — dilarang |
| E2E test massal (Playwright) |  **DIKURANGI** | "skrip pengujian massal" tidak diminta. Sisakan unit test secukupnya |
| Qwen + Qdrant + embedding + XGBoost/CatBoost |  **DIBUANG** | Di luar core inference; juga tidak muat dalam 10 hari |

> **Ini bukan kerugian.** Semua yang dibuang **diparkir ke tahap Final** (§15). Panitia menyatakan
> proyek penyisihan **wajib dilanjutkan** di Final, jadi rancangan lengkapnya tetap berguna —
> hanya belum boleh diimplementasikan sekarang.

### 2.3 Batasan lain yang wajib dipatuhi

| Aturan | Konsekuensi |
|---|---|
| **Model WAJIB di-fine-tune** | Pakai pre-trained apa adanya = melanggar. Harus ada bukti fine-tuning di repo |
| Repo GitHub wajib **public** | Panitia harus bisa akses & menjalankan |
| Commit wajib **Conventional Commits** & deskriptif | Commit asal-asalan "dianggap tidak memenuhi standar pengembangan" |
| Commit & push **setiap ada perubahan** | Riwayat git = bukti pengerjaan dalam periode lomba |
| Dataset hanya dari **sumber publik** atau **sintetik** | Preprocessing wajib dilakukan & dijelaskan selama periode lomba |
| **Dilarang menampilkan institusi pendidikan** | Di repo, kode, proposal, video, slide — semuanya |
| Karya **orisinal**, dikerjakan 17 Juni – 25 Agustus 2026 | Dilarang melanjutkan proyek lama |
| Panitia berhak minta **live demo** saat penjurian | Sistem harus benar-benar jalan, bukan video editan |

---

## 3. Deliverables & Checklist Submisi

### 3.1 Empat berkas wajib

#### ① Repository GitHub
- [ ] Visibility **PUBLIC**
- [ ] `README.md` berisi **setup guide yang jelas** — panitia menjalankan sendiri di komputernya
- [ ] `docker-compose.yml` berfungsi: `docker compose up --build` langsung jalan
- [ ] Riwayat commit rapi, Conventional Commits, deskriptif
- [ ] Tidak ada nama institusi pendidikan di mana pun
- [ ] Commit terakhir **sebelum** 25 Agustus 2026 23.55 WIB

#### ② Video Proof of Work
- [ ] Durasi **maksimal 7 menit**
- [ ] YouTube, visibility **UNLISTED**
- [ ] Judul persis: `COMPFEST 18 AIC: PROOF OF WORK - [Nama Tim] - [Nama Proyek]`
- [ ] Isi: menunjukkan **proses pengerjaan** dan bahwa sistem benar-benar bekerja

#### ③ Video Karya Inovasi
- [ ] Durasi **maksimal 5 menit**
- [ ] YouTube, visibility **PUBLIC**
- [ ] Judul persis: `COMPFEST 18 AIC: [Nama Tim] - [Nama Proyek]`
- [ ] Isi: **use case aplikasi** dan **teknologi AI yang digunakan**

#### ④ Proposal PDF
- [ ] **Maksimal 20 halaman** (tidak termasuk cover, daftar pustaka, lampiran)
- [ ] Bebas plagiarisme
- [ ] Struktur bab sesuai ketentuan → panduan lengkap di §16

### 3.2 Checklist final sebelum submit (H-1)

```
[ ] Repo public & bisa di-clone orang luar
[ ] docker compose up --build DIUJI di komputer bersih / teman
[ ] README.md diikuti langkah demi langkah oleh orang yang belum pernah lihat proyek ini
[ ] Kedua video sudah diunggah, visibility & judul BENAR PERSIS
[ ] Link video sudah dites di mode incognito (unlisted ≠ private!)
[ ] Proposal PDF ≤ 20 halaman, dicek plagiarisme
[ ] TIDAK ADA nama kampus/sekolah di repo, kode, proposal, video, slide
[ ] Semua link dikumpulkan di situs COMPFEST
[ ] Submit lebih awal — boleh submit ulang, yang dinilai submisi terakhir
```

---

## 4. Masalah & Solusi

*Bahan utama untuk bab Latar Belakang di proposal.*

### 4.1 Masalah nyata

| Masalah | Dampak |
|---|---|
| Inspeksi manual oleh operator | Akurasi turun drastis setelah 2–3 jam karena kelelahan mata |
| Standar tiap operator berbeda | Barang sama bisa lolos di shift pagi, ditolak di shift malam |
| Kecepatan manusia terbatas | Menjadi *bottleneck* di lini produksi cepat |
| Cacat lolos ke pelanggan | Biaya recall & retur jauh lebih mahal daripada biaya inspeksi |
| Cacat jenis baru tidak terantisipasi | Bahan baku ganti / mesin aus → muncul cacat yang belum pernah dilabeli |

>  Anggota 3: setiap angka statistik yang dikutip di proposal **wajib punya sumber yang bisa
> ditelusuri** (R7.2). Jangan pakai angka dari ingatan.

### 4.2 Kenapa pendekatan tiga model

Pendekatan naif = satu model klasifikasi bagus/cacat. Masalahnya: butuh ribuan contoh tiap jenis
cacat (padahal cacat itu langka), tidak mendeteksi cacat jenis baru, dan tidak memberi tahu *di mana*.

| Model | Menjawab | Butuh label cacat? |
|---|---|---|
| **YOLO11-detect** | "Cacat jenis apa, di kotak mana?" | Ya |
| **YOLO11-seg** | "Bentuk persis area cacat? Berapa persen luasnya?" | Ya |
| **PaDiM** | "Apakah menyimpang dari normal?" | **Tidak** — cukup gambar bagus |
| **PaddleOCR** | "Kode batch berapa?" | Tidak (pre-trained) |

Ini juga menjadi jawaban untuk bab *"metode-metode lain yang mendukung alasan pengambilan keputusan"*
di proposal.

---

## 5. Daftar Fitur (versi Penyisihan)

Prioritas: **P0** = wajib (tanpa ini tidak ada demo) · **P1** = penting untuk nilai · **P2** = bonus.

> Semua fitur di bawah sudah disaring agar **tidak melanggar §2**. Fitur yang melanggar sudah
> dipindah ke §15 (rencana Final).

### 5.1 Fitur Inti

| ID | Fitur | Prio | Deskripsi |
|---|---|---|---|
| **F-01** | Unggah gambar tunggal | P0 | Pengguna pilih/drag 1 gambar produk. Ini "input tunggal" yang dimaksud panitia |
| **F-02** | Deteksi cacat (bbox) | P0 | YOLO11-detect **fine-tuned** → jenis cacat + lokasi + confidence |
| **F-03** | Decision engine | P0 | Gabungkan output model → `PASS` / `REJECT` + alasan yang terbaca manusia |
| **F-04** | Tampilan hasil | P0 | Gambar beranotasi (bbox+mask) + verdict + alasan + skor tiap model |
| **F-05** | Segmentasi cacat | P1 | YOLO11-seg → mask presisi → `luas_cacat / luas_objek` dalam persen |
| **F-06** | Anomaly detection | P1 | PaDiM → skor anomali. Menangkap cacat jenis baru |
| **F-07** | OCR kode batch | P1 | **Tersambung tetapi DIMATIKAN.** paddleocr 2.7.3 tidak dapat diimpor bersama NumPy 2 dan protobuf yang terpasang; naik ke 3.7 menarik opencv-contrib yang bertabrakan dengan OpenCV 5. `batch_code` selalu `null`. Alasan lengkap di `AI_model/configs/inference.yaml` |
| **F-08** | Contoh gambar siap pakai | P1 | Tombol "coba contoh" — penting agar panitia bisa menguji tanpa cari gambar sendiri |
| **F-09** | Penjelasan hasil | P1 | Teks yang menerangkan **kenapa** verdict-nya begitu — bukan sekadar angka |
| **F-10** | Heatmap anomali | P2 | Visualisasi area mencurigakan. **Belum dibangkitkan**; `anomaly.heatmap_base64` masih `null` |
| **F-11** | Grading severity | P2 | Ringan / sedang / berat berdasarkan luas + jenis cacat |
| **F-12** | Health check | P1 | `/healthz` — memastikan sistem siap saat panitia menjalankan |

### 5.2 Fitur Pendukung Penilaian

| ID | Fitur | Prio | Deskripsi |
|---|---|---|---|
| **F-13** | `docker compose up` sekali jalan | **P0** | Dinilai langsung oleh panitia (R9.3) |
| **F-14** | README setup guide | **P0** | Langkah demi langkah, bisa diikuti orang asing |
| **F-15** | Halaman "Tentang Model" | P2 | Info versi model, metrik fine-tuning, dataset — memperlihatkan kedalaman kerja |
| **F-16** | Bukti fine-tuning di repo | **P0** | Skrip training, config, log, metrik sebelum vs sesudah (R7.3) |
**Total: 16 fitur.** Ini terlihat jauh lebih sedikit dari versi sebelumnya (31 fitur) — dan memang
seharusnya begitu. Panitia menilai **fokus dan reprodusibilitas**, bukan banyaknya fitur.

---

## 6. Tech Stack

### 6.1 Frontend

| Teknologi | Peran | Alasan |
|---|---|---|
| **Next.js 15** | Aplikasi web | Satu halaman inti, mudah di-dockerize |
| **TypeScript strict** | Bahasa | Error ketahuan saat compile |
| **Tailwind CSS** | Styling | Cepat — waktu tinggal 10 hari |
| **shadcn/ui** | Komponen | Aksesibel & rapi tanpa membangun design system |
| **Zod** | Validasi respons API | Memastikan tipe tidak "bohong" saat runtime |
**Yang sengaja TIDAK dipakai:** TanStack Query/Table (tidak ada daftar data), Socket.IO (tidak ada
realtime), Recharts (tidak ada dashboard analitik — dilarang R9.1), Zustand (state cukup `useState`).

### 6.2 Backend

| Teknologi | Peran | Alasan |
|---|---|---|
| **FastAPI (Python 3.11)** | REST API sinkron | Satu bahasa dengan AI, OpenAPI otomatis, Pydantic |
| **Pydantic v2** | Skema request/response | Kontrak tipe yang tegas (R3.6) |
| **Uvicorn** | ASGI server | Standar FastAPI |
| **Pillow / OpenCV** | Pengolahan gambar | Decode, resize, gambar anotasi |
**Tanpa database, tanpa queue, tanpa object storage.** Gambar diproses di memori, hasil dikembalikan
langsung dalam respons. Ini bukan penyederhanaan malas — ini **kepatuhan pada R9.1** sekaligus
membuat `docker compose up` jauh lebih andal untuk dinilai panitia.

### 6.3 AI Stack

| Komponen | Model | Fine-tune? | Peran |
|---|---|---|---|
| **Detection** | Ultralytics **YOLO11n** |  **WAJIB** | Cari cacat: jenis + bbox |
| **Segmentation** | **YOLO11n-seg** |  **WAJIB** | Mask cacat → luas % |
| **Anomaly** | **PaDiM** |  dilatih dari data kita | Skor penyimpangan dari normal |
| **OCR** | **PaddleOCR** |  pre-trained (pelengkap) | Baca kode batch |
| **Runtime** | **ONNX Runtime** | — | Inferensi cepat di CPU |

> **R7.3 — fine-tuning itu WAJIB, bukan opsional.** Panitia menyatakan: *"Model wajib di fine
> tune sesuai dengan inovasi fitur per tim."* Bukti yang harus ada di repo: skrip training, file
> config/hyperparameter, log training, dan tabel metrik **sebelum vs sesudah** fine-tuning.
> PaddleOCR boleh tetap pre-trained karena ia komponen pelengkap, bukan model inovasi inti —
> tapi YOLO dan model anomali **harus** dilatih dengan data kita.

### 6.4 Dataset

| Sumber | Status | Catatan |
|---|---|---|
| **MVTec AD** |  Boleh — sumber publik | Standar emas anomaly detection industri. Punya gambar normal + cacat + mask ground truth |
| **DAGM 2007** |  Boleh — sumber publik | Cacat permukaan tekstur |
| **Data sintetik** |  Boleh secara eksplisit | Bisa dipakai menambah variasi |

Panitia mengizinkan dataset dari sumber publik dan data sintetik. **Yang wajib dilakukan selama
periode lomba adalah preprocessing-nya**, dan itu harus dijelaskan di proposal (§16, bab Metodologi
→ "alur dalam memperoleh dataset").

### 6.5 Tooling
**Docker Compose** (wajib) · **uv** (Python) · **pnpm** (Node) · **ruff** · **pytest**

---

## 7. Arsitektur Sistem (Sinkron)

```
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Next.js — satu halaman inti                              │  │
│  │  [ Unggah gambar ]  →  [ Periksa ]  →  [ Hasil ]          │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ POST /api/v1/inspect  (multipart, 1 gambar)
                           │ ── menunggu (sinkron) ──
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI  (Backend/)                          │
│                                                                 │
│   1. validasi gambar (format, ukuran maks)                      │
│   2. preprocessing (resize, normalisasi)                        │
│   3. ┌──────────── PIPELINE INFERENSI ────────────┐             │
│      │  YOLO11-detect  → bbox + kelas + conf      │             │
│      │  YOLO11-seg     → mask → luas %            │             │
│      │  PaDiM          → anomaly_score            │             │
│      │  PaddleOCR      → batch_code               │             │
│      └────────────────────────────────────────────┘             │
│   4. DECISION ENGINE  → PASS / REJECT (mode biner)              │
│   5. render gambar beranotasi (bbox + mask)                     │
│   6. balas JSON + gambar (base64)                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 200 OK — satu respons lengkap
                           ▼
                    Hasil tampil di browser
```

### Prinsip arsitektur

| # | Prinsip | Alasan |
|---|---|---|
| 1 | **Semuanya sinkron** — satu request, satu respons | Batasan panitia R9.1. Tidak ada queue, worker, atau polling |
| 2 | **Tanpa database** | "Tidak perlu pipeline pencatatan data otomatis" — dan menghilangkan satu sumber kegagalan saat panitia menjalankan |
| 3 | **Model dimuat sekali saat startup** | Warm-up di `lifespan` FastAPI; request pertama tidak lambat |
| 4 | **Parameter statis** | Threshold dibaca dari file config saat startup, tidak bisa diubah saat runtime (R7.4) |
| 5 | **Dua container saja** | `web` + `api`. Makin sedikit service, makin kecil kemungkinan gagal di komputer panitia |
| 6 | **Stateless** | Tidak menyimpan apa pun antar request |

### Perbandingan dengan rancangan lama

| | Lama (melanggar) | Baru (patuh) |
|---|---|---|
| Jumlah container | 8 (web, api, worker, ingestor, postgres, redis, rabbitmq, minio) | **2** (web, api) |
| Alur | async, event-driven | **sinkron** |
| Waktu setup panitia | menit, rawan gagal | **< 5 menit, satu perintah** |
| Risiko gagal saat dinilai | tinggi | **rendah** |

---

## 8. Cara Kerja Detail

### 8.1 Satu permintaan, langkah demi langkah

| # | Tahap | Yang terjadi | Estimasi |
|---|---|---|---|
| 1 | **Unggah** | Pengguna pilih gambar; frontend validasi tipe & ukuran | — |
| 2 | **Kirim** | `POST /api/v1/inspect` multipart | ~30 ms |
| 3 | **Validasi** | Cek format (JPG/PNG), ukuran maks 10 MB, dimensi minimum | ~5 ms |
| 4 | **Preprocess** | Resize ke 640×640, normalisasi, konversi tensor | ~15 ms |
| 5 | **Detect** | YOLO11-detect fine-tuned → daftar bbox + kelas + confidence | ~40 ms |
| 6 | **Segment** | YOLO11-seg → mask → `luas_cacat = area(mask)/area(objek) × 100%` | ~45 ms |
| 7 | **Anomaly** | PaDiM → `anomaly_score` (skala jarak Mahalanobis, **bukan** 0–1) | ~35 ms |
| 8 | **OCR** | PaddleOCR pada region kode - **tidak aktif**, 0 ms | ~60 ms bila dinyalakan |
| 9 | **Decide** | Decision engine (§8.2) | ~1 ms |
| 10 | **Render** | Gambar hasil beranotasi bbox + mask, encode base64 | ~25 ms |
| 11 | **Respons** | JSON lengkap dikembalikan | ~10 ms |
**Target total: ≤ 1 detik di CPU dengan ONNX.**
>  Ini **target**, bukan hasil pengukuran. Wajib diukur nyata dan dicatat di `LOG.md` sebelum
> ditulis di proposal (R7.2).

### 8.2 Decision Engine

Ini "otak" sistem dan bagian yang paling layak dijelaskan panjang di proposal.

> **Diperbarui 23 Agustus 2026.** Bagian ini semula menjelaskan keputusan TIGA
> kelas. Sejak 20 Agustus 2026 sistem berjalan pada **mode biner**: setiap
> gambar dinyatakan PASS atau REJECT, dan `REVIEW` tidak pernah dikembalikan.
> Dokumen ini tertinggal dari perubahan itu; teks di bawah adalah keadaan yang
> sebenarnya berjalan. Implementasinya di
> `AI_model/src/visionqc_ai/inference/decision.py`.

```python
# Semua ambang dibaca dari configs/inference.yaml saat startup dan bersifat
# STATIS selama sistem berjalan (R7.4, batasan panitia).

def decide(defects, defect_area_pct, anomaly_score, config) -> Verdict:

    # 1) Kelas kritis ditolak berapa pun luasnya. Saat ini hanya `kotor`,
    #    karena kontaminasi berdampak langsung pada keamanan konsumsi.
    if any(d.class_name in config.critical_classes for d in defects):
        return Verdict(REJECT, alasan="terdeteksi kotor, ditolak berapa pun luasnya")

    # 2) Luas cacat melewati batas toleransi
    if defect_area_pct > config.area_pct_threshold:          # 2.0 %
        return Verdict(REJECT, alasan="luas cacat melampaui batas")

    # 3) Cacat terdeteksi cukup meyakinkan
    if max(d.confidence for d in defects) >= config.binary_threshold:   # 0.35
        return Verdict(REJECT, alasan="cacat terdeteksi melampaui ambang keputusan")

    # 4) Menyimpang dari normal walau tidak ada cacat yang dikenali
    if anomaly_score > config.anomaly_threshold:             # 83.2714
        return Verdict(REJECT, alasan="skor anomali melampaui ambang")

    return Verdict(PASS, alasan="tidak ditemukan cacat maupun penyimpangan dari normal")
```

**Kenapa biner, bukan tiga kelas:**

| Alasan | Penjelasan |
|---|---|
| **Sistem harus memutuskan** | Lini produksi yang menyerahkan sebagian produk ke manusia tidak menyelesaikan persoalan yang ingin diotomatiskan |
| **Biayanya bukan pendapat** | Model biaya menetapkan cacat lolos Rp50.000 melawan salah-tolak Rp2.000, jadi cacat yang terdeteksi lemah lebih murah ditolak daripada ditahan |
| **Ambangnya diturunkan, bukan ditebak** | `binary_threshold` 0.35 dipilih dengan meminimumkan biaya yang diharapkan pada SET KALIBRASI; set uji tidak dilihat saat memilih |

Himpunan prediksi conformal tetap dihitung dan tetap dilaporkan lewat medan
`decision` supaya keputusan dapat ditelusuri, tetapi tidak lagi dipakai untuk
menahan keputusan.

> **Kepatuhan.** Tidak ada antrean review, penyimpanan, maupun loop retraining
> otomatis — semuanya dilarang di tahap penyisihan (R9.1) dan diparkir ke Final
> (§15). Mode tiga kelas masih ada di dalam kode di balik
> `decision.mode: three_class` dan direncanakan dipakai kembali di tahap Final.

### 8.3 Metrik yang benar untuk QC

Utamakan **recall pada kelas cacat**, bukan akurasi keseluruhan.
Kalau 97% barang bagus, model yang selalu menjawab "PASS" sudah dapat akurasi 97% — tapi tidak
berguna sama sekali. Tetapkan target recall dulu (mis. ≥ 0.90), baru maksimalkan precision di batas itu.

Ini poin bagus untuk proposal: menunjukkan tim paham **metrik mana yang benar-benar penting** di
konteks manufaktur.

---

## 9. Kontrak API

> Kontrak antara Frontend dan Backend. **Dikunci di Hari 1** supaya keduanya bisa jalan paralel.

```http
POST /api/v1/inspect          # multipart: file=<gambar>  → hasil inspeksi lengkap
GET  /api/v1/samples          # daftar gambar contoh (F-08)
GET  /api/v1/model-info       # versi & metrik model (F-15)
GET  /healthz                 # kesiapan sistem (F-12)
```
**Respons `POST /api/v1/inspect`:**

> **Diperbarui 23 Agustus 2026 dengan keluaran sungguhan.** Contoh sebelumnya
> ditulis sebagai rancangan sebelum sistemnya ada, dan beberapa nilainya sudah
> tidak berlaku: `threshold` anomali tertulis 0.75 padahal skalanya bukan 0..1
> dan nilainya 83.2714, sedangkan medan `label`, `defect_area_pct`, `decision`,
> dan `anomaly.exceeded` belum tercantum sama sekali. Contoh di bawah disalin
> apa adanya dari `POST /api/v1/inspect` atas `Backend/samples/bottle_broken_01.png`.

```json
{
  "verdict": "REJECT",
  "reason": "luas cacat 11.11 persen melampaui batas 2.00 persen",
  "confidence": 0.9358512163162231,
  "batch_code": null,
  "defects": [
    {
      "type": "pecah",
      "label": "Pecah / Retak",
      "bbox": { "x": 305, "y": 274, "w": 497, "h": 575 },
      "confidence": 0.9358512163162231,
      "area_pct": 11.1115
    }
  ],
  "defect_area_pct": 11.1115,
  "anomaly": {
    "score": 54.39267349243164,
    "threshold": 83.2714,
    "exceeded": false,
    "heatmap_base64": null
  },
  "decision": {
    "calibrated_probability": 0.9358512163162231,
    "prediction_set": ["cacat"],
    "severity": 0.8422660946846009,
    "conformal_alpha": 0.1
  },
  "annotated_image_base64": "data:image/jpeg;base64,...",
  "model_version": "visionqc-models-v1.1.0-6class",
  "latency_ms": 272
}
```

Catatan atas medan yang mudah disalahpahami:

| Medan | Keadaan sekarang |
|---|---|
| `verdict` | hanya `PASS` atau `REJECT`. `REVIEW` tidak pernah dikembalikan pada mode biner |
| `batch_code` | selalu `null`. Jalur OCR tersambung tetapi dimatikan; alasannya di `configs/inference.yaml` |
| `label` | nama kelas berbahasa Indonesia untuk ditampilkan, mis. `Pecah / Retak` |
| `anomaly.score` | skala jarak Mahalanobis PaDiM, **bukan** 0..1 |
| `anomaly.heatmap_base64` | selalu `null`; peta panas belum dibangkitkan |
| `decision` | dilaporkan untuk penelusuran, tidak dipakai menahan keputusan |

### 9.1 Sumber kebenaran kontrak

> Diputuskan 23 Agustus 2026, menggantikan rencana `packages/contracts/` sebagai folder tersendiri.

Sumber kebenaran tunggal adalah **`AI_model/src/visionqc_ai/schemas.py`**. Model Pydantic di sana
sudah menjadi tipe kembalian `run_inspection()`, jadi ia satu-satunya definisi yang pasti ikut
berubah ketika keluaran AI berubah.

```
visionqc_ai/schemas.py            <- SUMBER KEBENARAN (Pydantic)
        |
        +-- Backend  : impor langsung, pakai sebagai response_model FastAPI
        |
        +-- FastAPI  : /openapi.json dihasilkan otomatis
                |
                +-- Frontend : api.d.ts di-generate dari openapi.json
```

Alasan folder kontrak terpisah **tidak** dibuat: menyalin skema ke tempat ketiga menambah satu
lagi berkas yang bisa tidak sinkron tanpa ada yang menyadarinya, dan salinan yang menyimpang
justru merusak jaminan yang ingin diberikan kontrak itu. Backend mengimpornya langsung, sehingga
ketidakcocokan muncul sebagai galat impor atau galat tipe, bukan sebagai bug diam-diam saat
penyajian.

Frontend tetap **dilarang** menulis tipe API manual (R3.7):

```bash
pnpm dlx openapi-typescript http://localhost:8000/openapi.json -o Frontend/src/types/api.d.ts
```

**Mengubah `schemas.py` tetap berarti mengubah kontrak antar-anggota** — wajib dikabari lebih
dulu, sebagaimana R6.5 mengaturnya untuk `packages/contracts/`.

---

## 10. Struktur Repo

> Diperbarui 18 Agustus 2026. Versi sebelumnya menuliskan tata letak gaya
> monorepo (`apps/api`, `apps/web`, `ml`) yang tidak pernah terwujud. Yang
> berlaku adalah struktur nyata di bawah.

> **Diperbarui 23 Agustus 2026** mengikuti keadaan setelah Backend tersambung
> ke modul AI. Beberapa baris pada versi sebelumnya menyebut berkas yang tidak
> pernah ada: `Backend/app/main.py` (yang ada `Backend/main.py`),
> `Backend/models/` (bobot ada di `AI_model/models/`), dan `.env.example`
> (belum ada karena layanan tidak memerlukan satu pun secret).

```
compfest/
|-- README.md                      # setup guide + docker compose  [WAJIB R9.3]
|-- PROJECT.md                     # dokumen teknis
|-- docker-compose.yml             # service api; web menyusul
|-- .dockerignore                  # mempersempit konteks build
|-- .gitignore
|
|-- Frontend/                      # Next.js                     [Anggota 1]
| |-- app/page.tsx               #   halaman inti: unggah, hasil
| |-- components/                #   UploadZone, ResultCard, DefectOverlay
| |-- lib/api.ts                 #   client, tipe di-generate dari /openapi.json
|
|-- Backend/                       # FastAPI sinkron             [Anggota 2]
| |-- main.py                    #   entrypoint, lifespan, penangan galat
| |-- Dockerfile                 #   konteks build = AKAR REPO, bukan Backend/
| |-- requirements.txt           #   dependensi web saja; AI dari AI_model
| |-- app/
| |   |-- routers/inspect.py     #   endpoint inspeksi
| |   |-- routers/meta.py        #   samples + model-info
| |   |-- schemas.py             #   ekspor ulang kontrak dari modul AI (9.1)
| |   |-- config.py              #   batas unggahan, dibaca dari config AI
| |-- samples/                   #   gambar contoh (F-08)
| |-- tests/
|
|-- AI_model/                      # modul AI                    [Anggota 1]
| |-- configs/                   #   dataset, training, inference
| |-- src/visionqc_ai/
| |   |-- data/                  #   konversi, split, validasi dataset
| |   |-- training/              #   skrip fine-tuning
| |   |-- evaluation/            #   metrik, bootstrap, uji signifikansi
| |   |-- statistics/            #   conformal, kalibrasi, EVT
| |   |-- privacy/               #   EXIF, face blur, ephemeral
| |   |-- inference/             #   pipeline, decision engine, anotasi
| |-- scripts/                   #   entrypoint manual
| |-- data/  models/             #   tidak masuk git
| |-- reports/                   #   metrik & gambar evaluasi, ikut di-commit
| |-- EXPERIMENTS.md             #   tabel metrik sebelum vs sesudah
|
|-- docs/                          #                             [Anggota 3]
    |-- proposal/  video/  assets/
```

> `AI_model/` **dipakai saat runtime**, bukan sekadar arsip bukti. Backend
> mengimpor `visionqc_ai` dari sana, dan image Docker memasangnya lewat
> `pip install /app/AI_model`. Karena itu konteks build Docker adalah akar
> repo; konteks `./Backend` akan menghasilkan container yang gagal pada impor
> pertama.
>
> Selain dipakai, folder ini sekaligus menjadi bukti bahwa model benar-benar
> di-fine-tune dan bahwa preprocessing dikerjakan selama periode lomba.
>
> Isi `AI_model/reports/` sengaja ikut di-commit karena merupakan bukti
> evaluasi. Bobot model dan dataset tetap dikecualikan karena ukurannya.

---

## 11. JOBDESK PER ANGGOTA

Sisa waktu 10 hari. Pembagian ini dirancang agar **tidak ada yang menunggu**.

> **Anggota 1 memegang beban terberat** (Frontend + AI, dan fine-tuning adalah jalur kritis).
> Mitigasi ada di §11.4 — tolong dibaca dan dijalankan, bukan sekadar dibaca.

---

### ANGGOTA 1 — Frontend + AI Engineer
**Folder milik:** `AI_model/` · `AI_model/src/visionqc_ai/inference/` · `Frontend/`

#### A. Bagian AI — **JALUR KRITIS, kerjakan lebih dulu**

| # | Tugas | Deliverable | Hari |
|---|---|---|---|
| A1 | Unduh & siapkan dataset MVTec AD | Dataset terpasang, split train/val/test terkunci | H-10 |
| A2 | **Skrip preprocessing** | `AI_model/src/visionqc_ai/data/` — resize, augmentasi, konversi format YOLO. **Wajib dijelaskan di proposal** | H-10 |
| A3 | Baseline (sebelum fine-tune) | Ukur YOLO11n pre-trained apa adanya → catat metrik. Ini pembanding wajib | H-9 |
| A4 | **Fine-tune YOLO11-detect**  | Model terlatih + log + metrik sesudah. **INI WAJIB (R7.3)** | H-9 |
| A5 | **Fine-tune YOLO11-seg**  | Model mask + perhitungan luas % | H-8 |
| A6 | **Latih model anomali**  | PaDiM, dilatih dari gambar normal saja + tentukan ambang dengan teori nilai ekstrem | H-8 |
| A7 | Integrasi PaddleOCR | Modul `inference/ocr.py` beserta penyaring daftar-izin; mesinnya dimatikan karena bentrok dependensi | H-7 |
| A8 | **Decision engine** | `decision.py` — gabungkan semua jadi PASS/REJECT | H-7 |
| A9 | Ekspor ONNX + ukur latensi | `.onnx` + tabel perbandingan latensi | H-6 |
| A10 | Modul anotasi gambar | `annotate.py` — gambar bbox + mask di atas gambar asli | H-6 |
| A11 | **Isi `AI_model/EXPERIMENTS.md`** | Tabel metrik **sebelum vs sesudah** fine-tuning (R7.1, R7.3) | terus-menerus |

#### B. Bagian Frontend

| # | Tugas | Deliverable | Hari |
|---|---|---|---|
| B1 | Setup Next.js + Tailwind + shadcn | Shell aplikasi + layout | H-10 |
| B2 | Komponen unggah gambar | Drag & drop + preview + validasi (F-01) | H-9 |
| B3 | API client | `lib/api.ts` pakai tipe hasil generate dari `/openapi.json` (lihat 9.1) | H-9 |
| B4 | Tampilan hasil | Verdict besar + alasan + skor tiap model (F-04, F-09) | H-6 |
| B5 | Overlay cacat | Gambar beranotasi + toggle bbox/mask/heatmap (F-05, F-10) | H-5 |
| B6 | Tombol gambar contoh | "Coba contoh" (F-08) — **penting agar panitia mudah menguji** | H-5 |
| B7 | Halaman "Tentang Model" | Versi + metrik + dataset (F-15) | H-4 |
| B8 | Polish + loading state + error state | Tampilan rapi untuk video demo | H-4 |
**Urutan yang saya sarankan:** hari 1–4 fokus **AI sampai A6**, karena fine-tuning butuh waktu
training (dan waktu tunggu itu bisa dipakai mengerjakan B1–B3 secara paralel). Setelah pipeline AI
jadi, geser penuh ke frontend.

---

### ANGGOTA 2 — Backend Engineer
**Folder milik:** `Backend/` · `docker-compose.yml` · `README.md`

| # | Tugas | Deliverable | Hari |
|---|---|---|---|
| C1 | **Inisialisasi repo GitHub PUBLIC** | Repo + `.gitignore` + struktur folder + branch protection | **H-10 (pagi)** |
| C2 | **Kunci kontrak API**  | `schemas.py` + `openapi.json` + `api.d.ts` — semua orang menunggu ini | **H-10** |
| C3 | Skeleton FastAPI | App jalan, `/healthz` hijau, CORS (F-12) | H-10 |
| C4 | **Mock endpoint `/inspect`**  | Balas hasil palsu sesuai kontrak → Anggota 1 bisa bangun UI tanpa menunggu AI | **H-10** |
| C5 | Validasi & preprocessing gambar | Cek format/ukuran, resize, penanganan error yang jelas | H-9 |
| C6 | `config.py` threshold statis | Nilai statis dari file config (R7.4) | H-9 |
| C7 | **`docker-compose.yml`**  | 2 service, `docker compose up --build` langsung jalan (F-13) | H-8 |
| C8 | Lifespan & warm-up model | Model dimuat sekali saat startup, 1 inferensi dummy | H-7 |
| C9 | Integrasi pipeline AI ke endpoint | Pasang modul Anggota 1 menggantikan mock | H-6 |
| C10 | Endpoint samples & model-info | F-08, F-15 | H-5 |
| C11 | Penanganan error menyeluruh | Gambar rusak, model gagal muat, timeout — pesan jelas, bukan 500 telanjang | H-5 |
| C12 | Unit test secukupnya | Test decision engine + validasi. **Bukan bulk testing** (dilarang R9.1) | H-4 |
| C13 | **`README.md` setup guide**  | Langkah demi langkah (F-14, R9.3) — **ini dinilai panitia** | H-4 |
| C14 | **Uji reprodusibilitas**  | Clone bersih di komputer lain → `docker compose up` → harus jalan (R5.6) | H-3 |
| C15 | Optimasi latensi | Pastikan respons masuk akal untuk demo | H-3 |
**C1, C2, C4 adalah prioritas mutlak di hari pertama.** Selama ketiganya belum ada, dua anggota lain
tidak bisa bergerak efisien.

---

### ANGGOTA 3 — Proposal, Video & Dokumentasi
**Folder milik:** `docs/`

Ini **bukan peran pasif**. Tiga dari empat berkas wajib ada di tangan Anggota 3.

#### A. Proposal (panduan lengkap di §16)

| # | Tugas | Hari |
|---|---|---|
| D1 | Riset latar belakang + kumpulkan sumber yang bisa dikutip | H-10 |
| D2 | Studi literatur: YOLO, PaDiM, anomaly detection industri | H-9 |
| D3 | Draf bab Latar Belakang + Tujuan & Manfaat | H-8 |
| D4 | Draf bab **Metodologi** (3 sub-bab wajib — §16.2) | H-7 sampai H-5 |
| D5 | Bab "metode lain yang mendukung keputusan" | H-4 |
| D6 | Kesimpulan + daftar pustaka | H-3 |
| D7 | **Rapikan ke PDF ≤ 20 halaman + cek plagiarisme** | H-2 |

#### B. Video (panduan lengkap di §17)

| # | Tugas | Hari |
|---|---|---|
| D8 | Naskah + storyboard kedua video | H-5 |
| D9 | Rekam **Video Proof of Work** (maks 7 menit) | H-3 |
| D10 | Rekam **Video Karya Inovasi** (maks 5 menit) | H-2 |
| D11 | Edit + unggah + **cek judul & visibility** | H-2 |

#### C. Dukungan Teknis — **ini yang membuat perannya krusial**

| # | Tugas | Kenapa penting | Hari |
|---|---|---|---|
| D12 | **Bantu siapkan dataset & verifikasi label** | Pekerjaan paling makan waktu di CV. Ini yang paling meringankan Anggota 1 | H-10 s/d H-8 |
| D13 | **Kumpulkan gambar contoh** untuk F-08 | Panitia perlu bisa menguji tanpa cari gambar sendiri | H-6 |
| D14 | Diagram arsitektur & alur | Dipakai di proposal **dan** video | H-6 |
| D15 | **QA — coba sistem sebagai orang awam** | Menemukan hal membingungkan yang tidak terlihat oleh yang membuatnya | H-4 |
| D16 | **Uji `README.md`** — ikuti sendiri dari nol | Simulasi persis apa yang dilakukan panitia | H-3 |
| D17 | **Kurator `LOG.md`** | Pastikan progres tercatat (R1.1); tanya anggota lain tiap hari | tiap hari |
| D18 | **Audit "tidak ada nama institusi"** | R9.4 — cek repo, kode, proposal, video, slide | H-1 |
| D19 | **Pegang checklist submisi §3.2** | Diskualifikasi kalau ada yang kurang | H-1 |

---

### 11.4 Beban Kerja & Mitigasi

| Anggota | Porsi | Titik terberat |
|---|---|---|
| 1 — Frontend + AI | ~45% | H-10 s/d H-6 (fine-tuning + bangun UI bersamaan) |
| 2 — Backend | ~25% | H-10 (semua orang menunggu kontrak & mock) |
| 3 — Proposal + video | ~30% | H-3 s/d H-1 (proposal, dua video, checklist) |
**Cara meringankan Anggota 1 — jalankan, jangan cuma dibaca:**

1. **Anggota 2 wajib menyelesaikan mock endpoint (C4) di hari pertama.** Ini membuat Anggota 1 bisa membangun seluruh UI tanpa menunggu model selesai dilatih.
2. **Anggota 3 mengambil alih penyiapan dataset (D12).** Tidak butuh kemampuan coding, tapi menghemat berhari-hari.
3. **Fine-tune secukupnya, jangan mengejar sempurna.** Panitia mewajibkan *ada* fine-tuning yang bisa dibuktikan — bukan akurasi tertinggi. Fine-tune 30–50 epoch pada subset dataset sudah memenuhi syarat dan bisa dijelaskan dengan jujur.
4. **Fitur P2 dipotong tanpa ragu** kalau H-4 belum kelar.

### 11.5 Ritual Tim (10 hari, jadi harus ketat)

| Kapan | Apa | Durasi |
|---|---|---|
| Tiap pagi | Standup di grup: kemarin apa, hari ini apa, mentok di mana | 10 menit |
| Tiap malam | Anggota 3 update `LOG.md` + papan status | 10 menit |
| H-6 & H-3 | Demo internal: tunjukkan sistem **berjalan**, bukan bercerita | 20 menit |
| H-1 | Jalankan checklist §3.2 bersama-sama | 45 menit |

---

## 12. Workflow Git

### 12.1 Aturan dari panitia (bukan preferensi kita)

| Aturan panitia | Konsekuensi praktis |
|---|---|
| Repo wajib **PUBLIC** | Buat public sejak awal, jangan private lalu diubah di akhir |
| **Commit & push setiap ada perubahan** | Riwayat git = bukti pengerjaan dalam periode lomba. Jangan menumpuk semua jadi 1 commit besar di akhir — itu mencurigakan |
| **Conventional Commits wajib** | Commit tanpa pesan deskriptif "dianggap tidak memenuhi standar pengembangan" |
| Commit terakhir sebelum **25 Agu 2026 23.55** | Selesaikan H-1, jangan mepet |

### 12.2 Format commit (sesuai contoh panitia)

Panitia menyebut tiga tipe secara eksplisit:

```
feat: <deskripsi>       — penambahan fitur atau fungsionalitas baru
fix: <deskripsi>        — perbaikan bug atau kesalahan pada sistem
refactor: <deskripsi>   — perubahan struktur kode yang tidak mengubah fungsionalitas
```

Contoh yang baik:
```bash
feat: tambah endpoint inspeksi gambar tunggal
feat: fine-tune YOLO11n pada dataset MVTec AD kategori bottle
feat: ganti keputusan tiga kelas dengan keputusan biner tanpa REVIEW
fix: perbaiki perhitungan luas cacat saat mask kosong
refactor: pisahkan pipeline inferensi jadi modul terpisah
```

Contoh yang **buruk** (berisiko dianggap tidak memenuhi standar):
```bash
update
fix bug           (bug yang mana?)
asdasd
final version
```

>  Scope opsional (`feat(api): ...`) tetap sah menurut spesifikasi Conventional Commits, tapi
> panitia hanya mencontohkan bentuk tanpa scope. **Pakai bentuk tanpa scope** agar persis sama
> dengan contoh mereka — tidak ada gunanya mengambil risiko tafsir.

### 12.3 Struktur branch

```
main                        ← selalu bisa jalan, ini yang dinilai panitia
 ├── feat/ai-finetune-yolo    ← Anggota 1
 ├── feat/web-upload-page     ← Anggota 1
 ├── feat/api-inspect-endpoint← Anggota 2
 ├── feat/api-docker-compose  ← Anggota 2
 └── feat/docs-proposal-bab1  ← Anggota 3
```

### 12.4 Siklus harian (semua anggota)

```bash
# ── PAGI: ambil pekerjaan tim ──────────────────────────
git checkout main
git pull --rebase origin main

# ── MULAI TUGAS ────────────────────────────────────────
git checkout -b feat/ai-finetune-yolo

# ── KERJA, lalu commit (SERING, jangan ditumpuk) ───────
git add AI_model/src/visionqc_ai/training/train_yolo_detect.py
git commit -m "feat: tambah skrip fine-tuning YOLO11n untuk deteksi cacat"

# ── SEBELUM PUSH: sinkronkan ───────────────────────────
git fetch origin
git rebase origin/main
# konflik? → selesaikan → git add <file> → git rebase --continue

# ── PUSH ───────────────────────────────────────────────
git push -u origin feat/ai-finetune-yolo

# ── PR di GitHub → minta review 1 anggota → merge ──────
git checkout main && git pull --rebase origin main
git branch -d feat/ai-finetune-yolo
```

> ⏱ **Catatan realistis untuk 10 hari:** kalau proses PR terasa memperlambat, boleh menyederhanakan
> jadi *push langsung ke branch masing-masing lalu merge cepat*. Yang **tidak boleh dikorbankan**
> adalah: pesan commit deskriptif, commit sering, dan `main` selalu bisa jalan.

### 12.5  Titik temu AI ↔ Backend

Berbeda dengan rancangan lama (queue + database), pada arsitektur sinkron ini AI dan Backend hanya
bertemu di **satu titik**: **antarmuka fungsi pipeline inferensi**.

```
   ANGGOTA 2 (Backend)                        ANGGOTA 1 (AI)
   ───────────────────                        ──────────────
   Backend/app/routers/inspect.py            AI_model/src/visionqc_ai/inference/
              │                                        │
              │   ┌──────────────────────────────┐    │
              └──▶│  TITIK TEMU:                 │◀───┘
                  │  fungsi run_inspection()     │
                  │  + skema InspectionResult    │
                  │  (Backend/app/schemas.py)   │
                  └──────────────────────────────┘
```
**Kontrak yang berlaku:**

> **Diperbarui 23 Agustus 2026.** Sumber kebenaran BUKAN lagi
> `Backend/app/schemas.py`. Berkas itu kini hanya mengekspor ulang skema dari
> `AI_model/src/visionqc_ai/schemas.py` — lihat bagian 9.1. Salinan yang
> sebelumnya berdiri sendiri di Backend sudah sempat menyimpang: `Defect`
> kehilangan `label`, `InspectionResult` kehilangan `defect_area_pct` dan
> `decision`, dan `verdict` masih memuat `REVIEW`.

```python
# AI_model/src/visionqc_ai/schemas.py  — SUMBER KEBENARAN, jangan diubah sepihak

class Defect(BaseModel):
    type: str
    label: str                      # nama kelas untuk ditampilkan ke pengguna
    bbox: BBox
    confidence: float
    area_pct: float | None = None

class AnomalyResult(BaseModel):
    score: float
    threshold: float
    exceeded: bool
    heatmap_base64: str | None = None

class DecisionDetail(BaseModel):
    calibrated_probability: float
    prediction_set: list[str]
    severity: float
    conformal_alpha: float

class InspectionResult(BaseModel):
    verdict: VerdictLabel           # "PASS" atau "REJECT"; REVIEW tidak dipakai
    reason: str
    confidence: float | None
    batch_code: str | None
    defects: list[Defect]
    anomaly: AnomalyResult | None
    annotated_image_base64: str
    model_version: str
    latency_ms: int


# AI_model/src/visionqc_ai/inference/__init__.py  — DIISI ANGGOTA 1
def run_inspection(image: bytes, config: Config) -> InspectionResult:
    """Satu-satunya pintu masuk dari backend ke AI."""
    ...
```
**Cara kerja paralel tanpa saling menunggu:**

| Waktu | Anggota 2 (Backend) | Anggota 1 (AI/FE) |
|---|---|---|
| H-10 | Kunci `schemas.py`; buat `run_inspection()` **versi mock** yang membalas hasil acak sesuai skema | — |
| H-10 s/d H-6 | Bangun endpoint, validasi, docker, README — semua di atas mock | Bangun UI di atas mock; latih model secara terpisah |
| H-6 | — | Ganti isi `run_inspection()` dengan pipeline asli |
| H-6 | **Integrasi langsung jalan**, karena bentuk datanya sudah sama sejak hari pertama | |
**Aturan mengubah kontrak (R6.5):**
```
1. JANGAN ubah schemas.py sendirian tanpa memberi tahu.
2. Menambah field opsional (`| None = None`) = aman.
   Ganti nama / hapus / ubah tipe = BREAKING → bicarakan dulu.
3. Setelah berubah, Anggota 2 regenerate tipe frontend:
   pnpm dlx openapi-typescript http://localhost:8000/openapi.json        -o Frontend/src/types/api.d.ts
4. Kabari di grup chat. Anggota lain: git pull --rebase origin main
```

### 12.6 Menghindari konflik

- Tiap orang bekerja di foldernya sendiri (§10) → konflik hampir nol
- `git pull --rebase origin main` tiap pagi **dan** sebelum push
- Branch berumur < 1 hari (10 hari, jadi harus cepat)
- File yang disentuh bersama (`docker-compose.yml`, `schemas.py`) → kabari dulu di chat

Kalau konflik terjadi: buka file, gabungkan manual, **jangan menghapus kerjaan orang lain**. Kalau
ragu, tanya pemiliknya. Kalau kacau: `git rebase --abort`, lalu minta bantuan.

### 12.7 File model & dataset

Bobot model dan dataset **tidak boleh masuk git** (R4.2) — repo jadi berat dan lambat di-clone
panitia.
**`.gitignore` wajib:**
```gitignore
# Model & dataset
*.pt
*.pth
*.onnx
*.engine
AI_model/data/
AI_model/models/
Backend/models/

# Environment
.env
.venv/
__pycache__/

# Node
node_modules/
.next/
```
**Lalu bagaimana panitia mendapat modelnya?** Ini pertanyaan penting — jangan sampai `docker compose up`
gagal karena model tidak ada. Tiga opsi:

| Opsi | Cara | Rekomendasi |
|---|---|---|
| **GitHub Releases** | Unggah `.onnx` sebagai release asset; `Dockerfile`/skrip unduh otomatis saat build |  **Paling rapi** — tetap satu tempat dengan repo, otomatis |
| Hugging Face Hub | Unggah model, unduh saat build |  Bagus, sekalian menambah kredibilitas |
| Google Drive | Link di README, unduh manual |  Terakhir — panitia harus kerja manual, rawan gagal |

>  Apapun pilihannya, **`README.md` wajib menjelaskan langkahnya dengan sangat jelas**, dan
> Anggota 3 wajib mengujinya dari nol (D16). Model yang tidak bisa diunduh = sistem tidak jalan =
> nilai hangus.

---

## 13. Workflow ML & Fine-Tuning (WAJIB)

Khusus Anggota 1. **Fine-tuning bukan opsional** — panitia mewajibkannya (R7.3).

### 13.1 Alur wajib

```
1. PEROLEH DATASET
   MVTec AD (publik) → pilih 1–2 kategori yang relevan (mis. bottle, metal_nut)
   Catat: sumber, lisensi, jumlah gambar, distribusi kelas
   → ini jadi isi bab "Alur dalam memperoleh dataset" di proposal

2. PREPROCESSING (wajib dikerjakan & dijelaskan selama periode lomba)
   - Resize ke 640×640
   - Konversi anotasi ke format YOLO
   - Augmentasi: flip, rotate, brightness, noise (Albumentations)
   - Split train/val/test — TERKUNCI, test set tidak pernah dilihat saat tuning (R7.5)
   → simpan skripnya di AI_model/src/visionqc_ai/data/, ini bukti kerja

3. BASELINE — UKUR SEBELUM FINE-TUNE   WAJIB
   Jalankan YOLO11n pre-trained apa adanya pada test set. Catat metriknya.
   Tanpa angka "sebelum", kamu tidak bisa membuktikan fine-tuning memberi dampak.

4. FINE-TUNE   INI YANG DIWAJIBKAN PANITIA
   yolo detect train model=yolo11n.pt data=defect.yaml epochs=50 imgsz=640
   Simpan: config hyperparameter, log training, kurva loss

5. UKUR SESUDAH
   Evaluasi pada test set yang sama. Bandingkan dengan baseline.

6. CATAT DI AI_model/EXPERIMENTS.md
   Tabel sebelum vs sesudah — ini bukti utama kepatuhan R7.3

7. EVALUASI JUJUR
   - Confusion matrix (bukan cuma akurasi)
   - Precision-Recall curve
   - LIHAT LANGSUNG 20 kasus salah prediksi — hampir selalu memberi wawasan

8. EKSPOR ONNX
   Ukur latensi di hardware yang dipakai demo.
   Pastikan metrik tidak turun setelah konversi.

9. KUNCI PARAMETER
   Threshold masuk config sebagai nilai STATIS (R7.4, batasan panitia).
   Tidak boleh berubah sendiri saat demo.
```

### 13.2 Tabel wajib di `AI_model/EXPERIMENTS.md`

Ini **bukti kepatuhan R7.3** dan bahan langsung untuk proposal:

| Model | Tahap | Dataset | Epoch | mAP50 | Recall (cacat) | Precision | Latensi | Catatan |
|---|---|---|---|---|---|---|---|---|
| YOLO11n-detect | **sebelum** (pre-trained) | MVTec bottle test | — | ? | ? | ? | ? | baseline |
| YOLO11n-detect | **sesudah** fine-tune | MVTec bottle test | 50 | ? | ? | ? | ? | |
| YOLO11n-seg | sebelum | — | — | ? | ? | ? | ? | |
| YOLO11n-seg | sesudah | — | 50 | ? | ? | ? | ? | |
| PaDiM | dilatih dari normal | gabungan 10 kategori | — | AUROC 0.6019 | — | — | 23 Agu | lemah, lihat EXPERIMENTS.md 5.3 |

> **Isi dengan angka hasil run yang sungguhan.** Jangan pernah mengarang (R7.2). Panitia berhak
> meminta live demo dan klarifikasi saat penjurian — angka karangan akan ketahuan.

### 13.3 Realistis dalam 10 hari

Kamu **tidak perlu** mengejar model terbaik. Yang dinilai adalah:
- Apakah fine-tuning **benar-benar dilakukan** dan **bisa dibuktikan**
- Apakah **prosesnya dijelaskan dengan jelas** di proposal
- Apakah **sistemnya berjalan** saat panitia menjalankannya

Fine-tune 30–50 epoch pada 1–2 kategori MVTec AD sudah memenuhi syarat. Lebih baik model sederhana
yang benar-benar jalan dan terdokumentasi, daripada model canggih yang belum selesai saat deadline.

---

## 14. TIMELINE - SISA 7 HARI
**Hari ini 18 Agustus. Deadline 25 Agustus 23.55 WIB.**

> Timeline H-10 yang lama sudah terlewat. Yang tercapai sampai 18 Agustus:
> dataset terunduh dan Step 2 modul AI selesai. Sisi Backend dan Frontend
> masih kosong.

| Hari | Tgl | Anggota 1 (FE+AI) | Anggota 2 (BE) | Anggota 3 (Proposal) |
|---|---|---|---|---|
| **H-7** | 18 Agu | Step 2 dataset selesai; mulai Step 3-4 | **Repo public, kontrak API, mock endpoint** (C1-C4) | Riset latar belakang + studi literatur (D1-D2) |
| **H-6** | 19 Agu | **Fine-tune YOLO-detect** (Step 4); setup Next.js (B1) | Validasi gambar, config statis, **docker-compose** (C5-C7) | Draf Latar Belakang + Tujuan (D3) |
| **H-5** | 20 Agu | **Fine-tune seg + anomali** (Step 5-6); komponen unggah (B2-B3) | Lifespan & warm-up (C8) | Metodologi bagian 1-2 (D4); diagram (D14) |
| **H-4** | 21 Agu | Lapisan statistik + privasi (Step 7-8); tampilan hasil (B4) | **Integrasi pipeline AI** (C9) | Metodologi bagian 3 (D4); naskah video (D8) |
| **H-3** | 22 Agu | **Decision engine + ONNX + integrasi** (Step 9-10); overlay (B5-B6) | Endpoint samples & model-info (C10-C11) | Bab metode pendukung (D5); QA sistem (D15) |
| **H-2** | 23 Agu | Polish UI + perbaikan hasil QA (B7-B8) | Unit test, **README**, **uji reprodusibilitas** (C12-C14) | Kesimpulan (D6); **rekam Video PoW** (D9); uji README (D16) |
| **H-1** | 24 Agu | **Freeze fitur**, hanya bug kritis | Freeze; commit terakhir | **Video Inovasi + PDF final** (D7, D10-D11); audit institusi (D18) |
| **H-0** | 25 Agu | **Submit pagi hari**; checklist bagian 3.2 (D19) | | |

### Konsekuensi kehilangan tiga hari

Sisa waktu tidak lagi memungkinkan mengerjakan semua yang direncanakan.
Urutan pemotongan:

1. **F-15 halaman Tentang Model** - potong sekarang
2. **F-11 grading severity** dan **F-10 heatmap** - potong bila H-4 belum mulai
3. **Step 3 cacat sintetik** - kecilkan ke kelas `kotor` dan `deformasi` saja
4. **F-07 OCR** - potong bila H-3 belum jalan
5. **Step 5 segmentasi** - bila terpaksa, luas cacat diestimasi dari bbox dan
   keterbatasannya dinyatakan apa adanya di proposal

Yang tidak boleh dipotong: F-01, F-02, F-03, F-04, F-13, F-14, dan terutama
F-16 (bukti fine-tuning), karena itu kewajiban panitia dan bukan fitur.

### Aturan timeline yang tidak boleh dilanggar

1. **H-10 milik Anggota 2.** Kontrak API + mock endpoint harus selesai hari pertama. Tanpa itu, dua orang lain terhambat sepanjang lomba.
2. **Fine-tuning selesai H-8.** Ini jalur kritis. Kalau molor, semua ikut molor.
3. **Integrasi H-6, bukan H-2.** Jangan tunggu semua sempurna baru disatukan.
4. **H-1 freeze fitur.** Hari terakhir untuk memastikan semuanya jalan, bukan menambah fitur.
5. **Submit H-0 pagi.** Boleh submit ulang — yang dinilai submisi terakhir. Submit awal = jaring pengaman kalau internet bermasalah.

### Titik potong kalau tertinggal

Potong berurutan dari atas:
```
1. F-15 Halaman "Tentang Model"     ← buang duluan
2. F-11 Grading severity
3. F-10 Heatmap anomali
4. F-07 OCR kode batch
5. F-06 anomali PaDiM  ← kalau sangat terpaksa; kurangi nilai inovasi tapi sistem tetap jalan
```
**Tidak boleh dipotong:** F-01, F-02, F-03, F-04, F-13, F-14, F-16.
Terutama **F-16 (bukti fine-tuning)** — itu kewajiban panitia, bukan fitur.

---

## 15. Rencana Tahap Final (Fitur yang Diparkir)

Panitia menyatakan: *"Proyek yang dikerjakan saat penyisihan wajib dilanjutkan sebagai proyek yang
dikerjakan saat tahap Final."*

Jadi rancangan lengkap yang dibuang di §2.2 **tidak terbuang** — ia menjadi peta jalan tahap Final,
dan bisa disebut sebagai *rencana pengembangan* di bab Kesimpulan proposal. Ini justru memperkuat
proposal: menunjukkan tim punya visi jangka panjang, bukan sekadar demo sekali pakai.

| Prioritas | Fitur Final | Dari batasan mana |
|---|---|---|
| 1 | Ingest video/RTSP realtime + antrian pesan | "background jobs" |
| 2 | Database + riwayat inspeksi + penelusuran batch | "halaman riwayat" & "automated logging" |
| 3 | Dashboard analitik: tren, Pareto cacat, per-lini | "dashboard analitik lanjut" |
| 4 | Antrean review manusia + active learning | "loop umpan balik otomatis" |
| 5 | Model registry, shadow mode, auto-retraining | "auto-tuning" |
| 6 | Integrasi PLC/sorter nyata | — |
| 7 | Auth & RBAC multi-peran | "otentikasi kompleks" |
| 8 | Monitoring Prometheus + Grafana | — |

> **Rancangan arsitektur async yang dibuang tetap disimpan** di riwayat git dan `LOG.md`.
> Saat Final, tinggal diambil lagi.

---

## 16. Panduan Proposal (20 Halaman)

Untuk Anggota 3. Struktur bab **ditentukan panitia** — ikuti persis.

### 16.1 Struktur wajib

| Bab | Isi | Est. halaman |
|---|---|---|
| — | **Cover** (tidak dihitung) — nama tim, judul inovasi.  **TANPA nama institusi** (R9.4) | — |
| 1 | **Nama Kelompok dan Judul/Nama Inovasi** | 0,5 |
| 2 | **Latar Belakang** — masalah QC manual, data pendukung bersumber, kenapa AI (§4) | 3 |
| 3 | **Tujuan dan Manfaat Pengembangan** — apa yang dicapai, siapa yang diuntungkan, dampak ekonomi | 2 |
| 4 | **Metodologi** — tiga sub-bab wajib (§16.2) | **9** |
| 5 | **Metode-metode lain yang mendukung alasan pengambilan keputusan** (§16.3) | 3 |
| 6 | **Kesimpulan** — ringkasan + rencana tahap Final (§15) | 1,5 |
| — | **Daftar Pustaka** (tidak dihitung) | — |
| — | **Lampiran** (tidak dihitung) — screenshot, tabel metrik lengkap, diagram besar | — |
**Total inti: ~19 halaman.** Aman di bawah batas 20.

### 16.2 Bab Metodologi — tiga sub-bab yang diminta panitia

Ini bab terpenting dan paling banyak porsinya.

#### ① Alur dalam memperoleh dataset (~2,5 hal)
- Sumber: MVTec AD / DAGM — sebutkan lisensi & jumlah gambar
- Kenapa dataset ini dipilih (relevansi dengan kasus manufaktur)
- Distribusi kelas & ketidakseimbangan data
- **Preprocessing**: resize, konversi anotasi ke format YOLO, augmentasi, strategi split
- Sertakan diagram alur data

#### ② Alur pengembangan model — **tiap feature** (~4,5 hal)
Panitia meminta penjelasan **per fitur**. Jadi bagi per model:

| Sub-bagian | Isi |
|---|---|
| **Deteksi cacat (YOLO11-detect)** | Arsitektur, alasan memilih YOLO11n, hyperparameter, **metrik sebelum vs sesudah fine-tuning** |
| **Segmentasi (YOLO11-seg)** | Kenapa butuh mask (untuk hitung luas %), proses training, hasil |
| **Anomaly detection (PaDiM)** | Kenapa perlu model tanpa label cacat, cara pelatihan dari data normal, penentuan ambang dengan teori nilai ekstrem, dan AUROC yang terukur apa adanya |
| **OCR (PaddleOCR)** | Peran pelengkap, kenapa cukup pre-trained |
| **Decision Engine** | Logika biner, kenapa REVIEW ditinggalkan, penetapan ambang dari model biaya pada set kalibrasi |

#### ③ Alur integrasi model ke environment kode (~2 hal)
- Ekspor PyTorch → ONNX, alasan (latensi)
- Arsitektur sinkron FastAPI (§7) + diagram
- Pemuatan model saat startup (lifespan) & warm-up
- Kontainerisasi dengan Docker Compose
- Alur request end-to-end (§8.1)

### 16.3 Bab "Metode lain yang mendukung keputusan"

Di sinilah menjelaskan **kenapa memilih ini, bukan itu**:

| Keputusan | Alasan yang bisa dituliskan |
|---|---|
| Tiga model, bukan satu klasifikator | Cacat langka & tidak seimbang; cacat jenis baru tidak akan tertangkap model berlabel |
| PaDiM berdampingan dengan YOLO | Saling melengkapi: YOLO butuh label, PaDiM tidak |
| Keputusan biner, bukan 3 kelas | Sistem QC harus memutuskan sendiri; ambangnya diturunkan dari model biaya, bukan dipilih tangan |
| Recall diutamakan, bukan akurasi | Data tidak seimbang; cacat lolos jauh lebih mahal |
| ONNX Runtime | Latensi inferensi untuk penggunaan lini produksi |
| Arsitektur sinkron | Sesuai batasan ruang lingkup MVP + reprodusibilitas lokal |
| YOLO11n (bukan model besar) | Target penggunaan di perangkat pabrik berdaya komputasi terbatas |

### 16.4 Aturan penulisan

-  Setiap angka wajib punya sumber atau berasal dari eksperimen sendiri (R7.2)
-  Bebas plagiarisme — parafrase + sitasi yang benar
-  **Jangan sebut nama kampus/sekolah/institusi di mana pun** (R9.4)
-  Sertakan diagram (dari D14) — memecah dinding teks dan memudahkan juri

---

## 17. Panduan Video

### 17.1 Video Proof of Work — maks 7 menit, **UNLISTED**
**Judul persis:** `COMPFEST 18 AIC: PROOF OF WORK - [Nama Tim] - [Nama Proyek]`

Tujuannya membuktikan **kalian benar-benar mengerjakannya**. Saran struktur:

| Menit | Isi |
|---|---|
| 0:00–0:30 | Perkenalan tim & proyek (tanpa nama institusi) |
| 0:30–2:00 | Tur repo GitHub: struktur folder, riwayat commit, skrip training |
| 2:00–3:30 | **Bukti fine-tuning**: skrip, log training, tabel metrik sebelum vs sesudah |
| 3:30–5:00 | Menjalankan sistem dari nol: `docker compose up --build` sampai jalan |
| 5:00–6:30 | Demo langsung: unggah gambar → hasil muncul (tunjukkan kasus PASS dan REJECT) |
| 6:30–7:00 | Penutup: rencana tahap Final |

### 17.2 Video Karya Inovasi — maks 5 menit, **PUBLIC**
**Judul persis:** `COMPFEST 18 AIC: [Nama Tim] - [Nama Proyek]`

Panitia meminta video menunjukkan **use case aplikasi dan teknologi AI yang digunakan**. Ini video
"jualan" — lebih rapi dan naratif.

| Menit | Isi |
|---|---|
| 0:00–0:45 | **Masalah**: inspeksi manual di pabrik — lelah, tidak konsisten, lambat |
| 0:45–1:30 | **Solusi**: VisionQC — apa yang dilakukan sistem |
| 1:30–3:00 | **Use case**: demo alur nyata, tunjukkan cacat terdeteksi + lokasi + verdict |
| 3:00–4:15 | **Teknologi AI**: tiga model dan perannya masing-masing, kenapa kombinasinya kuat |
| 4:15–5:00 | **Dampak & penutup**: manfaat bagi industri, rencana pengembangan |

### 17.3 Checklist video (mudah terlewat, fatal akibatnya)

```
[ ] Durasi TIDAK melebihi batas (7 menit / 5 menit)
[ ] Visibility BENAR: PoW = unlisted, Inovasi = public
[ ] Judul PERSIS sesuai format panitia — cek huruf besar/kecil dan tanda hubung
[ ] Link diuji di mode incognito (unlisted ≠ private — private tidak bisa dibuka panitia!)
[ ] TIDAK ADA nama/logo institusi pendidikan (R9.4)
[ ] Audio jelas, layar terbaca
[ ] Demo yang ditampilkan sungguhan, bukan mockup
```

---

## 18. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
|  **Waktu hanya 10 hari** | Tidak selesai | Ruang lingkup sudah dipangkas sesuai batasan panitia; titik potong §14 sudah disiapkan; freeze fitur H-1 |
|  **Melanggar batasan ruang lingkup** | Nilai turun / dianggap tidak patuh | §2 jadi acuan; sebelum menambah apa pun, cek R9.1 |
|  **Lupa fine-tuning** | **Melanggar aturan wajib** | A3–A6 masuk jalur kritis; F-16 tidak boleh dipotong; bukti disimpan di `AI_model/` |
|  **Ada berkas kurang saat submit** | **Diskualifikasi** | Checklist §3.2 dijalankan H-1 oleh Anggota 3 |
|  `docker compose up` gagal di komputer panitia | Nilai reprodusibilitas hangus | C14 — uji di komputer bersih; D16 — Anggota 3 ikuti README dari nol |
|  Model tidak bisa diunduh panitia | Sistem tidak jalan | GitHub Releases + langkah jelas di README (§12.7) |
|  Nama institusi ikut ter-commit | Melanggar R9.4 | Audit D18 di H-1: repo, kode, proposal, video, slide |
|  Video salah visibility/judul | Panitia tidak bisa akses | Checklist §17.3; uji link di incognito |
|  Inferensi lambat saat demo | Demo tersendat | ONNX + YOLO11n; uji di laptop yang benar-benar dipakai demo |
|  Riwayat commit terlihat menumpuk di akhir | Dicurigai tidak dikerjakan selama periode lomba | Commit sering & deskriptif tiap hari (R6.6) |
|  Angka metrik dikarang | **Fatal** — panitia bisa minta live demo & klarifikasi | R7.2; semua angka dari run sungguhan |
|  Anggota 1 kelebihan beban | Jalur kritis macet | Mock endpoint H-10 (C4); Anggota 3 bantu dataset (D12); fine-tune secukupnya |

---

## 19. Setup
**Prasyarat:** Docker Desktop · Node 20+ · pnpm · Python 3.11 · uv · Git

### Cara panitia menjalankan (harus sesederhana ini)

```bash
git clone https://github.com/<tim>/visionqc.git
cd visionqc
cp .env.example .env
docker compose up --build
# → buka http://localhost:3000
```

>  Kalau langkah di atas gagal di komputer bersih, itu **masalah paling gawat** di proyek ini.
> Uji berkala (R5.6), bukan cuma di akhir.

### Cara developer menjalankan

```bash
# backend
cd Backend
uv sync
uv run uvicorn app.main:app --reload          # → :8000/docs

# frontend
cd Frontend
pnpm install
pnpm dev                                       # → :3000

# fine-tuning (Anggota 1)
cd ml
uv run python preprocessing/prepare_mvtec.py
uv run python training/train_yolo_detect.py
uv run python export/onnx_export.py
```

### Sebelum push

```bash
uv run ruff check . && uv run pytest
pnpm lint && pnpm typecheck
docker compose up --build        # ← R5.6
```

---

## 20. Keputusan Terbuka

| # | Pertanyaan | Kenapa mendesak |
|---|---|---|
| 1 | **Nama tim & nama proyek final?** | Dibutuhkan untuk judul video (formatnya sudah ditentukan panitia) dan cover proposal |
| 2 | **Sudah terdaftar di COMPFEST?** | 30 tim pertama dapat VPS/GPU credits gratis — sangat membantu fine-tuning |
| 3 | **Sudah ada repo GitHub?** | C1 harus jalan hari ini juga |
| 4 | **Ada GPU untuk fine-tuning?** | Menentukan berapa epoch yang realistis dalam 10 hari |
| 5 | **Kategori produk yang diinspeksi?** (botol, logam, kain, PCB) | Menentukan subset MVTec AD yang dipakai |
| 6 | **Apakah proyek ini sudah pernah dikerjakan sebelum 17 Juni 2026?** | Kalau ya, **melanggar aturan** — harus dimulai bersih |
| 7 | **Siapa yang jadi PIC standby Discord 9–10 Sept, 20.00?** | Wajib ada yang siaga, balas maks 2 jam |

Begitu diputuskan, catat di [LOG.md](./LOG.md).
