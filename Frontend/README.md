# VisionQC - antarmuka

Halaman inti VisionQC: satu citra produk masuk, satu putusan keluar.

Panduan menjalankan seluruh sistem ada di [README repositori](../README.md).
Berkas ini hanya memuat yang khusus antarmuka.

## Pengembangan

```bash
pnpm install
pnpm dev            # http://localhost:3000
```

Server pengembangan meneruskan `/api/*` dan `/samples/*` ke layanan pada
`VISIONQC_API_ORIGIN` (bawaan `http://localhost:8000`), sehingga peramban selalu
berbicara dengan satu origin dan tidak bergantung pada CORS.

## Kualitas

```bash
pnpm lint
pnpm typecheck
pnpm test
```

## Tipe API

Tipe respons **tidak pernah ditulis tangan** (R3.7). Sumbernya `/openapi.json`
milik layanan, yang diturunkan dari skema Pydantic modul AI:

```bash
pnpm gen:api        # layanan harus sedang berjalan
```

`src/lib/contract.ts` mengunci skema Zod ke tipe hasil pembangkitan itu, jadi
kontrak yang berubah tanpa tipe ikut dibangkitkan ulang akan muncul sebagai
galat kompilasi, bukan sebagai medan yang diam-diam kosong saat ditampilkan.

## Kamera

Kamera dipakai sebagai sumber citra, bukan sebagai jalur inferensi
berkelanjutan: bingkai yang diambil menjadi berkas JPEG biasa dan melewati jalur
yang sama dengan unggahan. Peramban hanya mengizinkan kamera pada konteks aman,
yaitu `localhost` atau HTTPS; membuka antarmuka lewat alamat IP jaringan tanpa
HTTPS akan mematikan tab Kamera, dan antarmuka menjelaskan hal itu di tempat.
