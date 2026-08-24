"""Bangun seluruh diagram alir proposal VisionQC."""

from __future__ import annotations

import pathlib
import sys

# Modul saudara diimpor dari folder ini sendiri. Sebelumnya berkas ini
# menyisipkan folder sementara milik sesi pembangunan lama ke sys.path, dan
# karena sisipan itu diletakkan paling depan, salinan di folder sementara
# itulah yang benar-benar dipakai membangun dokumen - bukan berkas di
# repositori ini. Akibatnya menyunting berkas di sini tidak mengubah apa pun,
# dan folder sementara itu dapat hilang kapan saja.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from flowchart import arrow, box, canvas, elbow, lane, save  # noqa: E402

# Keluaran ditaruh relatif terhadap letak berkas ini, bukan pada jalur mutlak,
# supaya skrip ini tetap berjalan di mesin mana pun yang mengkloning repositori.
# Foldernya dibuat bila belum ada; pada kloning bersih ia memang belum ada,
# karena gambar jadinya tidak ikut disimpan di git.
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = str(OUT_DIR)


# ---------------------------------------------------------------- 1. pengguna
def alur_pengguna():
    fig, ax = canvas(7.6, 9.4, (0, 10), (0, 12.6))

    y = 11.9
    box(ax, 5, y, 3.0, 0.62, "Mulai", "terminator", bold=True)
    box(ax, 5, y - 1.25, 5.4, 0.82,
        "Operator membuka antarmuka web|VisionQC di peramban")
    box(ax, 5, y - 2.55, 5.4, 1.10,
        "Memilih satu citra produk: unggah berkas,|tangkap kamera, atau gambar contoh|(JPG / PNG / WEBP, maks. 10 MB)", "user")
    box(ax, 5, y - 3.9, 4.6, 0.86, "Validasi sisi klien:|tipe berkas dan ukuran", "decision")
    box(ax, 5, y - 5.3, 5.4, 0.82,
        "Menekan tombol Periksa;|antarmuka menampilkan status memproses")
    box(ax, 5, y - 6.55, 5.4, 0.72, "Sistem menjalankan pipeline inspeksi", "ai")
    box(ax, 5, y - 7.85, 4.6, 0.86, "Keputusan sistem", "decision")

    box(ax, 2.35, y - 9.3, 3.6, 0.92, "REJECT: pita merah,|kotak dan mask cacat,|alasan penolakan", "user")
    box(ax, 7.65, y - 9.3, 3.6, 0.92, "PASS: pita hijau,|gambar tanpa penandaan,|keterangan lolos", "ai")

    box(ax, 5, y - 10.7, 5.6, 0.82,
        "Operator membaca alasan dan angka|pendukung, lalu menindaklanjuti produk")
    box(ax, 5, y - 11.6, 3.0, 0.62, "Selesai", "terminator", bold=True)

    arrow(ax, (5, y - 0.31), (5, y - 0.84))
    arrow(ax, (5, y - 1.66), (5, y - 2.00))
    arrow(ax, (5, y - 3.10), (5, y - 3.47))
    arrow(ax, (5, y - 4.33), (5, y - 4.89), "sah")
    elbow(ax, (2.7, y - 3.9), (2.3, y - 2.55), first="h", label="tidak sah")
    arrow(ax, (5, y - 5.71), (5, y - 6.19))
    arrow(ax, (5, y - 6.91), (5, y - 7.42))
    elbow(ax, (5, y - 8.28), (2.35, y - 8.84), first="v")
    elbow(ax, (5, y - 8.28), (7.65, y - 8.84), first="v")
    ax.text(3.5, y - 8.66, "REJECT", ha="center", va="center", fontsize=8.8,
            color="#33415c", fontfamily="Times New Roman",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none"), zorder=6)
    ax.text(6.5, y - 8.66, "PASS", ha="center", va="center", fontsize=8.8,
            color="#33415c", fontfamily="Times New Roman",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none"), zorder=6)
    elbow(ax, (2.35, y - 9.76), (4.4, y - 10.29), first="v")
    elbow(ax, (7.65, y - 9.76), (5.6, y - 10.29), first="v")
    arrow(ax, (5, y - 11.11), (5, y - 11.29))

    save(fig, OUT + r"\alur_pengguna.png")


# ------------------------------------------------------------ 2. arsitektur
def arsitektur_sistem():
    fig, ax = canvas(9.6, 7.4, (0, 13), (0, 10))

    lane(ax, 0.3, 6.55, 12.4, 3.1, "Peramban pengguna")
    lane(ax, 0.3, 3.15, 12.4, 3.1, "Kontainer web  ·  Next.js 15 (port 3000)")
    lane(ax, 0.3, 0.25, 12.4, 2.6, "Kontainer api  ·  FastAPI + ONNX Runtime (port 8000)")

    box(ax, 3.2, 8.0, 4.4, 0.95, "Halaman inti:|unggah gambar dan tampilkan hasil", "user")
    box(ax, 9.4, 8.0, 4.4, 0.95, "Kartu hasil:|pita keputusan, alasan,|gambar beranotasi", "user")

    box(ax, 3.2, 4.6, 4.4, 0.95, "Komponen unggah|validasi tipe dan ukuran", "process")
    box(ax, 9.4, 4.6, 4.4, 0.95, "Klien API tertik|(tipe dibangkitkan dari OpenAPI)", "process")

    box(ax, 2.6, 1.5, 3.6, 0.9, "Router FastAPI|POST /api/v1/inspect", "accent")
    box(ax, 6.5, 1.5, 3.2, 0.9, "Modul visionqc_ai|run_inspection()", "ai")
    box(ax, 10.6, 1.5, 3.4, 0.9, "Empat bobot ONNX|dimuat saat startup", "data")

    arrow(ax, (3.2, 7.52), (3.2, 5.08))
    arrow(ax, (5.4, 4.6), (7.2, 4.6))
    arrow(ax, (9.4, 5.08), (9.4, 7.52))
    elbow(ax, (3.2, 4.12), (2.6, 1.95), first="v")
    arrow(ax, (4.4, 1.5), (4.9, 1.5))
    arrow(ax, (8.1, 1.5), (8.9, 1.5))
    elbow(ax, (6.5, 1.95), (9.4, 4.12), first="v")

    ax.text(3.35, 6.3, "multipart/form-data  ·  satu gambar", fontsize=8.6,
            color="#33415c", family="Times New Roman", ha="left")
    ax.text(9.55, 6.3, "JSON InspectionResult", fontsize=8.6,
            color="#33415c", family="Times New Roman", ha="left")
    ax.text(6.5, 0.62,
            "Tanpa basis data, antrean, maupun pekerja latar: satu permintaan HTTP dilayani sampai tuntas di dalam permintaan itu.",
            fontsize=8.8, color="#5b6572", family="Times New Roman", ha="center", style="italic")

    save(fig, OUT + r"\arsitektur_sistem.png")


# --------------------------------------------------------- 3. pipeline AI
def pipeline_inferensi():
    fig, ax = canvas(9.8, 8.2, (0, 13), (0, 11))

    box(ax, 6.5, 10.4, 4.0, 0.7, "Bita gambar dari Backend", "data", bold=True)
    box(ax, 6.5, 9.25, 6.6, 0.86,
        "Lapisan privasi: dekode di memori, metadata EXIF dibuang,|wajah diburamkan (YuNet), berkas tidak pernah ditulis ke diska", "user")
    box(ax, 6.5, 8.1, 5.4, 0.66, "Prapemrosesan: ubah ukuran 640 px, normalisasi")

    lane(ax, 0.35, 5.15, 12.3, 2.45, "Empat jalur inferensi")
    box(ax, 2.0, 6.35, 2.9, 1.15, "YOLO11n-detect|jenis cacat, kotak,|keyakinan", "ai")
    box(ax, 5.35, 6.35, 2.9, 1.15, "YOLO11n-seg|mask piksel,|luas cacat (%)", "ai")
    box(ax, 8.7, 6.35, 2.9, 1.15, "PaDiM|skor anomali|(jarak Mahalanobis)", "ai")
    box(ax, 11.5, 6.35, 2.0, 1.15, "PaddleOCR|kode batch|(nonaktif)", "process")

    box(ax, 6.5, 3.95, 6.4, 0.8,
        "Mesin keputusan — seluruh ambang statis dari inference.yaml", "accent", bold=True)
    box(ax, 6.5, 2.75, 5.6, 0.66, "Penggambaran anotasi: kotak, mask, dan pita keputusan")
    box(ax, 6.5, 1.55, 6.8, 0.9,
        "InspectionResult: verdict, reason, confidence, defects[],|defect_area_pct, anomaly, decision, annotated_image_base64", "data")
    box(ax, 6.5, 0.5, 3.4, 0.62, "Kembali ke Backend", "terminator", bold=True)

    arrow(ax, (6.5, 10.05), (6.5, 9.68))
    arrow(ax, (6.5, 8.82), (6.5, 8.43))
    for cx in (2.0, 5.35, 8.7, 11.5):
        elbow(ax, (6.5, 7.77), (cx, 6.93), first="v")
        elbow(ax, (cx, 5.78), (6.5, 4.35), first="v")
    arrow(ax, (6.5, 3.55), (6.5, 3.08))
    arrow(ax, (6.5, 2.42), (6.5, 2.0))
    arrow(ax, (6.5, 1.1), (6.5, 0.81))

    save(fig, OUT + r"\pipeline_inferensi.png")


# ------------------------------------------------------- 4. decision engine
def decision_engine():
    fig, ax = canvas(8.2, 9.6, (0, 11), (0, 13))

    box(ax, 5.5, 12.4, 6.6, 0.68,
        "Masukan: daftar cacat, keyakinan tertinggi, luas cacat (%), skor anomali", "data")

    box(ax, 5.5, 11.15, 5.0, 0.95, "Ada cacat berkelas kritis|(kotor / kontaminasi)?", "decision")
    box(ax, 5.5, 9.35, 5.0, 0.95, "Luas cacat > 2,00 %|dari luas gambar?", "decision")
    box(ax, 5.5, 7.55, 5.0, 0.95, "Keyakinan tertinggi|>= 0,195?", "decision")
    box(ax, 5.5, 5.75, 5.0, 0.95, "Skor anomali|> 83,2714?", "decision")

    box(ax, 5.5, 4.15, 3.4, 0.72, "PASS", "ai", fs=12, bold=True)
    box(ax, 9.6, 8.45, 2.4, 0.72, "REJECT", "user", fs=12, bold=True)

    box(ax, 5.5, 2.9, 7.4, 0.78,
        "Alasan dalam bahasa manusia + himpunan prediksi conformal + skor keparahan", "process")
    box(ax, 5.5, 1.75, 3.6, 0.62, "Verdict dikembalikan", "terminator", bold=True)

    arrow(ax, (5.5, 12.06), (5.5, 11.63))
    for yc in (11.15, 9.35, 7.55, 5.75):
        elbow(ax, (8.0, yc), (9.6, 8.81), first="h", label="ya" if yc == 11.15 else None)
        if yc != 11.15:
            ax.text(8.6, yc + 0.16, "ya", fontsize=8.5, color="#33415c",
                    family="Times New Roman", ha="center")
    arrow(ax, (5.5, 10.68), (5.5, 9.83), "tidak")
    arrow(ax, (5.5, 8.88), (5.5, 8.03), "tidak")
    arrow(ax, (5.5, 7.08), (5.5, 6.23), "tidak")
    arrow(ax, (5.5, 5.28), (5.5, 4.51), "tidak")

    elbow(ax, (5.5, 3.79), (4.6, 3.29), first="v")
    elbow(ax, (9.6, 8.09), (6.9, 3.29), first="v")
    arrow(ax, (5.5, 2.51), (5.5, 2.06))

    ax.text(5.5, 0.75,
            "Mode biner: sistem tidak pernah menahan keputusan. Aturan dievaluasi berurutan;\naturan pertama yang terpenuhi menentukan hasil, sehingga keputusan selalu dapat ditelusuri.",
            fontsize=8.8, color="#5b6572", family="Times New Roman", ha="center",
            style="italic", linespacing=1.4)

    save(fig, OUT + r"\decision_engine.png")


# ------------------------------------------------------------- 5. dataset
def alur_dataset():
    fig, ax = canvas(9.6, 7.6, (0, 13), (0, 10.4))

    box(ax, 2.2, 9.7, 3.6, 0.66, "MVTec AD|CC BY-NC-SA 4.0", "data", fs=9)
    box(ax, 6.5, 9.7, 3.6, 0.66, "VisA|CC BY 4.0", "data", fs=9)
    box(ax, 10.8, 9.7, 3.6, 0.66, "PKU-GoodsAD|GPL-3.0", "data", fs=9)

    box(ax, 6.5, 8.35, 8.6, 0.72,
        "Penyaringan kategori: ukur luas cacat pada resolusi 640 px dari mask asli")
    box(ax, 6.5, 7.1, 5.4, 0.9, "Objek >= 12 px?", "decision")
    box(ax, 11.3, 7.1, 3.0, 0.78, "Jalur anomali saja|(fryum, 29,8 %)", "process", fs=9)

    box(ax, 6.5, 5.65, 8.8, 0.78,
        "Rekonstruksi kode mask VisA menjadi jenis cacat — divalidasi 120/120 gambar multilabel")
    box(ax, 6.5, 4.4, 8.8, 0.78,
        "Penyatuan taksonomi: 38 label mentah menjadi 6 kelas (taxonomy.py)")
    box(ax, 6.5, 3.15, 8.8, 0.86,
        "Prapemrosesan: ubah ukuran 640 px, mask menjadi poligon YOLO,|komponen < 25 px dibuang, 15 % gambar normal disisipkan sebagai latar")
    box(ax, 6.5, 1.85, 8.8, 0.78,
        "Pembagian terstratifikasi, seed 42 dikunci — 1.428 latih / 310 validasi / 301 uji")
    box(ax, 6.5, 0.6, 8.8, 0.78,
        "Validasi statistik: uji khi-kuadrat keselarasan, entropi, dan kecukupan ukuran sampel", "accent")

    for cx in (2.2, 6.5, 10.8):
        elbow(ax, (cx, 9.37), (6.5, 8.71), first="v")
    arrow(ax, (6.5, 7.99), (6.5, 7.55))
    arrow(ax, (6.5, 6.65), (6.5, 6.04), "ya")
    arrow(ax, (9.2, 7.1), (9.8, 7.1), "tidak")
    arrow(ax, (6.5, 5.26), (6.5, 4.79))
    arrow(ax, (6.5, 4.01), (6.5, 3.58))
    arrow(ax, (6.5, 2.72), (6.5, 2.24))
    arrow(ax, (6.5, 1.46), (6.5, 0.99))

    save(fig, OUT + r"\alur_dataset.png")


# ------------------------------------------------ 6. pengembangan model
def alur_model():
    fig, ax = canvas(9.6, 6.6, (0, 13), (0, 9))

    box(ax, 2.1, 8.3, 3.4, 0.68, "Bobot pra-latih COCO", "data", fs=9.5)
    box(ax, 2.1, 6.9, 3.4, 0.78, "Ukur baseline|pada set uji", "process", fs=9.5)

    box(ax, 6.5, 8.3, 4.2, 0.68, "configs/training.yaml|(hiperparameter dikunci)", "data", fs=9.5)
    box(ax, 6.5, 6.9, 4.2, 0.78, "Fine-tuning|150 / 130 epoch, seed 42", "ai", fs=9.5)

    box(ax, 6.5, 5.35, 5.0, 0.78, "Pilih epoch terbaik|dari metrik split validasi", "process")
    box(ax, 6.5, 3.9, 8.8, 0.78,
        "Evaluasi pada split uji yang belum pernah dilihat (IoU 0,5)", "accent")
    box(ax, 6.5, 2.55, 8.8, 0.86,
        "Uji signifikansi McNemar berpasangan + selang kepercayaan BCa (2.000 resampling)|+ besar efek Cohen's h")
    box(ax, 6.5, 1.2, 5.4, 0.9, "Signifikan dan|memenuhi sasaran?", "decision")
    box(ax, 11.6, 1.2, 2.6, 0.78, "Catat kegagalan,|ulangi satu perubahan", "user", fs=9)
    box(ax, 1.9, 1.2, 2.8, 0.78, "Ekspor ONNX|dan verifikasi", "terminator", fs=9.5, bold=True)

    arrow(ax, (2.1, 7.96), (2.1, 7.29))
    arrow(ax, (6.5, 7.96), (6.5, 7.29))
    elbow(ax, (2.1, 6.51), (5.2, 5.35), first="v")
    arrow(ax, (6.5, 6.51), (6.5, 5.74))
    arrow(ax, (6.5, 4.96), (6.5, 4.29))
    arrow(ax, (6.5, 3.51), (6.5, 2.98))
    arrow(ax, (6.5, 2.12), (6.5, 1.65))
    arrow(ax, (9.2, 1.2), (10.3, 1.2), "tidak")
    arrow(ax, (3.8, 1.2), (3.3, 1.2), "ya")
    elbow(ax, (11.6, 1.59), (8.6, 6.9), first="v")

    save(fig, OUT + r"\alur_pengembangan_model.png")


# ---------------------------------------------------------- 7. integrasi
def alur_integrasi():
    fig, ax = canvas(9.6, 6.4, (0, 13), (0, 8.6))

    lane(ax, 0.3, 4.6, 12.4, 3.7, "Waktu pengembangan")
    lane(ax, 0.3, 0.25, 12.4, 4.0, "Waktu pemasangan  ·  docker compose up --build")

    box(ax, 2.3, 7.35, 3.4, 0.72, "best.pt (PyTorch)", "data", fs=9.5)
    box(ax, 6.5, 7.35, 3.8, 0.72, "Ekspor ONNX|opset 12 / 16", "process", fs=9.5)
    box(ax, 10.7, 7.35, 3.6, 0.84, "Verifikasi numerik|selisih maks. 3,4 x 10⁻⁵", "accent", fs=9.5)

    box(ax, 6.5, 5.55, 8.8, 0.8,
        "Unggah ke GitHub Releases (models-v1.1.0) + catat sidik jari SHA-256 di models.json", "data")

    box(ax, 2.3, 3.35, 3.6, 0.8, "Docker build|pip install ./AI_model", "process", fs=9.5)
    box(ax, 6.6, 3.35, 3.6, 0.86, "download_models.py|unduh + cocokkan SHA-256", "accent", fs=9.5)
    box(ax, 10.8, 3.35, 3.4, 0.86, "Sidik jari cocok?", "decision", fs=9.5)

    box(ax, 10.8, 1.85, 3.4, 0.72, "Build gagal|dengan pesan jelas", "user", fs=9.5)
    box(ax, 6.6, 1.85, 4.6, 0.86, "Startup: muat 4 bobot ONNX sekali|(lifespan FastAPI) + pemanasan", "ai", fs=9.5)
    box(ax, 2.3, 1.85, 3.6, 0.86, "/healthz melaporkan|ok atau degraded", "process", fs=9.5)
    box(ax, 6.6, 0.75, 4.0, 0.6, "Layanan siap menerima permintaan", "terminator", fs=9.5, bold=True)

    arrow(ax, (4.0, 7.35), (4.6, 7.35))
    arrow(ax, (8.4, 7.35), (8.9, 7.35))
    elbow(ax, (10.7, 6.93), (6.5, 5.95), first="v")
    elbow(ax, (6.5, 5.15), (2.3, 3.75), first="v")
    arrow(ax, (4.1, 3.35), (4.8, 3.35))
    arrow(ax, (8.4, 3.35), (9.1, 3.35))
    arrow(ax, (10.8, 2.92), (10.8, 2.21), "tidak")
    elbow(ax, (10.8, 2.92), (8.9, 1.85), first="v", label="ya")
    arrow(ax, (4.3, 1.85), (4.1, 1.85))
    elbow(ax, (6.6, 1.42), (6.6, 1.05), first="v")

    save(fig, OUT + r"\alur_integrasi.png")


if __name__ == "__main__":
    alur_pengguna()
    pipeline_inferensi()
    alur_model()
    raise SystemExit(0)
    arsitektur_sistem()
    pipeline_inferensi()
    decision_engine()
    alur_dataset()
    alur_model()
    alur_integrasi()
