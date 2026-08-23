# VisionQC

Sistem inspeksi kualitas kemasan pangan dan minuman berbasis computer vision.
Satu gambar produk diunggah, sistem memeriksanya, lalu mengembalikan keputusan
**PASS** atau **REJECT** beserta alasannya.

```
gambar produk
     |
     +-- lapisan privasi   metadata dibuang, wajah diburamkan
     +-- YOLO11n detect    cacat jenis apa, di mana, seberapa yakin
     +-- YOLO11n seg       berapa persen luas permukaan yang cacat
     +-- PaDiM             skor penyimpangan dari produk normal
     +-- decision engine   PASS atau REJECT, dengan alasan yang terbaca
     |
hasil + gambar beranotasi
```

Enam kelas cacat dikenali: **pecah**, **gores**, **noda**, **kotor**,
**deformasi**, dan **terbuka**.

---

## Menjalankan

### Yang perlu disiapkan

- Docker dan Docker Compose
- Koneksi internet saat build pertama, untuk mengunduh bobot model (sekitar
  190 MB) dan dependensi

GPU **tidak** diperlukan. Layanan berjalan di CPU.

### Satu perintah

```bash
docker compose up --build
```

Build pertama memakan beberapa menit karena memasang PyTorch versi CPU dan
mengunduh bobot model. Build berikutnya memakai cache.

Setelah container menyala:

```bash
curl http://localhost:8000/healthz
```

```json
{
  "status": "ok",
  "components": {
    "detection": true,
    "segmentation": true,
    "anomaly": true,
    "face_blur": true,
    "ocr": false
  }
}
```

Dokumentasi API interaktif tersedia di **http://localhost:8000/docs**.

### Mencoba inspeksi

Tiga gambar contoh ikut disertakan:

```bash
curl -X POST http://localhost:8000/api/v1/inspect \
  -F "file=@Backend/samples/bottle_broken_01.png;type=image/png"
```

```json
{
  "verdict": "REJECT",
  "reason": "luas cacat 11.11 persen melampaui batas 2.00 persen",
  "confidence": 0.9359,
  "defects": [
    {
      "type": "pecah",
      "label": "Pecah / Retak",
      "bbox": { "x": 305, "y": 274, "w": 497, "h": 575 },
      "confidence": 0.9359,
      "area_pct": 11.1115
    }
  ],
  "defect_area_pct": 11.1115,
  "annotated_image_base64": "...",
  "model_version": "visionqc-models-v1.1.0-6class",
  "latency_ms": 272
}
```

---

## Endpoint

| Metode | Jalur | Kegunaan |
|---|---|---|
| `GET` | `/healthz` | kesiapan layanan dan lapisan mana yang aktif |
| `POST` | `/api/v1/inspect` | periksa satu gambar, kembalikan keputusannya |
| `GET` | `/api/v1/model-info` | versi model yang sedang dilayani |
| `GET` | `/api/v1/samples` | daftar gambar contoh |
| `GET` | `/samples/{nama}` | berkas gambar contoh |

Batas unggahan: **10 MB**, format **JPG**, **PNG**, atau **WEBP**, sisi
terpendek minimal **224 piksel**.

---

## Menjalankan tanpa Docker

Untuk pengembangan. Python 3.11.

```bash
# modul AI beserta dependensinya
pip install -e ./AI_model

# bobot model, diverifikasi dengan SHA-256
python AI_model/scripts/download_models.py

# layanan
pip install -r Backend/requirements.txt
cd Backend
uvicorn main:app --reload
```

Pengujian:

```bash
cd Backend && python -m pytest      # layanan API
cd AI_model && python -m pytest     # modul AI
```

---

## Struktur

```
compfest/
|-- docker-compose.yml       susunan layanan
|-- Backend/                 layanan FastAPI
|   |-- main.py              aplikasi, penangan galat, endpoint kesehatan
|   |-- app/routers/         endpoint inspeksi dan keterangan
|   |-- app/schemas.py       ekspor ulang kontrak dari modul AI
|   |-- app/config.py        batas unggahan, dibaca dari config modul AI
|   +-- samples/             gambar contoh
|-- AI_model/                modul AI
|   |-- src/visionqc_ai/     pipeline, mesin keputusan, lapisan privasi
|   |-- configs/             parameter inferensi, semuanya statis
|   |-- scripts/             penyiapan data, pelatihan, ekspor, evaluasi
|   +-- models/models.json   daftar bobot beserta sidik jari SHA-256
+-- PROJECT.md               dokumen teknis
```

---

## Bobot model

Bobot tidak disimpan di dalam git karena ukurannya. Yang disimpan adalah
`AI_model/models/models.json`, daftar resmi beserta sidik jari SHA-256 setiap
berkas. `download_models.py` mengunduh dari GitHub Releases lalu mencocokkan
sidik jarinya, sehingga berkas yang rusak atau tertukar tertangkap sebelum
dipakai. Langkah ini sudah termasuk di dalam build Docker.

Memeriksa bobot yang sudah ada:

```bash
python AI_model/scripts/download_models.py --check
```

---

## Catatan

**Parameter bersifat statis.** Seluruh ambang keputusan dibaca sekali dari
`AI_model/configs/inference.yaml` saat startup dan tidak berubah selama sistem
berjalan. Tidak ada penyetelan otomatis maupun endpoint untuk mengubahnya.

**Pemrosesan sinkron.** Satu permintaan membawa satu gambar dan menerima satu
hasil lengkap. Tidak ada antrean, pekerjaan latar, maupun basis data.

**Privasi.** Metadata gambar dibuang dan wajah yang kebetulan terpotret
diburamkan sebelum gambar mencapai model. Gambar tidak pernah ditulis ke disk,
dan catatan hanya menyimpan ringkasan SHA-256, bukan gambarnya.

**Kode batch belum aktif.** Medan `batch_code` selalu bernilai `null`. Jalur
OCR sudah tersambung tetapi dimatikan; alasannya tercantum di
`AI_model/configs/inference.yaml`.

Rincian pengukuran model, dataset, dan keputusan rancangan ada di
`AI_model/EXPERIMENTS.md` dan `PROJECT.md`.
