"""Unduh bobot model dan verifikasi keasliannya.

Bobot model tidak disimpan di git karena ukurannya, sehingga siapa pun yang
mengkloning repo ini harus mengunduhnya sekali. Skrip ini yang melakukannya.

    python scripts/download_models.py
    python scripts/download_models.py --check

Hanya memakai pustaka bawaan Python, tanpa dependensi apa pun. Ini disengaja:
skrip ini dijalankan sebelum `pip install`, sehingga mensyaratkan pustaka luar
akan membuatnya gagal justru pada langkah pertama.

Setiap berkas diverifikasi dengan SHA-256 terhadap daftar di models/models.json.
Unduhan yang rusak di tengah jalan lebih berbahaya daripada unduhan yang gagal,
karena model yang cacat tetap berjalan dan memberi hasil yang keliru tanpa
gejala apa pun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "models" / "models.json"
CHUNK = 1024 * 256


@dataclass(frozen=True)
class ModelEntry:
    """Satu baris pada daftar model."""

    name: str
    target: Path
    url: str
    size_bytes: int
    sha256: str
    required: bool
    description: str


def load_manifest(path: Path = MANIFEST) -> list[ModelEntry]:
    """Baca daftar model beserta sidik jarinya."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    base = data["base_url"].rstrip("/")
    return [
        ModelEntry(
            name=item["name"],
            target=PROJECT_ROOT / item["target"],
            url=f"{base}/{item['name']}",
            size_bytes=int(item["size_bytes"]),
            sha256=str(item["sha256"]),
            required=bool(item.get("required", True)),
            description=str(item.get("description", "")),
        )
        for item in data["models"]
    ]


def file_sha256(path: Path) -> str:
    """Sidik jari SHA-256 sebuah berkas, dibaca bertahap agar hemat memori."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify(entry: ModelEntry) -> tuple[bool, str]:
    """Periksa apakah berkas sudah ada dan sidik jarinya cocok."""
    if not entry.target.exists():
        return False, "belum ada"
    actual = entry.target.stat().st_size
    if actual != entry.size_bytes:
        return False, f"ukuran {actual} byte, seharusnya {entry.size_bytes}"
    if file_sha256(entry.target) != entry.sha256:
        return False, "sidik jari SHA-256 tidak cocok"
    return True, "cocok"


def download(entry: ModelEntry) -> None:
    """Unduh satu berkas lalu verifikasi sebelum dipindahkan ke tujuannya.

    Unduhan ditulis ke berkas sementara dan baru dipindahkan setelah sidik
    jarinya terbukti cocok, sehingga berkas tujuan tidak pernah berisi unduhan
    setengah jadi.
    """
    entry.target.parent.mkdir(parents=True, exist_ok=True)
    temporary = entry.target.with_suffix(entry.target.suffix + ".part")

    downloaded = 0
    with urllib.request.urlopen(entry.url) as response, temporary.open("wb") as out:
        while chunk := response.read(CHUNK):
            out.write(chunk)
            downloaded += len(chunk)
            if entry.size_bytes:
                percent = 100 * downloaded / entry.size_bytes
                print(f"\r    {percent:5.1f}%  {downloaded // 1024} KB", end="")
    print()

    actual = file_sha256(temporary)
    if actual != entry.sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"sidik jari tidak cocok untuk {entry.name}\n"
            f"  seharusnya : {entry.sha256}\n"
            f"  didapat    : {actual}\n"
            "Berkas dibuang. Unduhan kemungkinan terpotong atau sumbernya berubah."
        )
    temporary.replace(entry.target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unduh bobot model VisionQC")
    parser.add_argument(
        "--check",
        action="store_true",
        help="hanya periksa berkas yang ada, tanpa mengunduh",
    )
    parser.add_argument("--force", action="store_true", help="unduh ulang walau cocok")
    args = parser.parse_args()

    entries = load_manifest()
    print(f"Daftar model: {MANIFEST.relative_to(PROJECT_ROOT)}")
    print(f"{len(entries)} berkas\n")

    missing_required = False
    for entry in entries:
        ok, message = verify(entry)
        label = "wajib" if entry.required else "opsional"
        print(f"  {entry.name}  [{label}]")

        if ok and not args.force:
            print(f"    sudah ada dan {message}")
            continue
        if args.check:
            print(f"    TIDAK SIAP: {message}")
            missing_required |= entry.required
            continue

        print(f"    {message}, mengunduh dari release")
        try:
            download(entry)
            print("    selesai dan terverifikasi")
        except urllib.error.HTTPError as error:
            print(f"    GAGAL: server menjawab {error.code}")
            if error.code == 404:
                print("    Release atau berkasnya belum tersedia.")
            missing_required |= entry.required
        except (urllib.error.URLError, ValueError) as error:
            print(f"    GAGAL: {error}")
            missing_required |= entry.required

    print()
    if missing_required:
        print("Model wajib belum tersedia. Sistem tidak dapat melakukan inspeksi.")
        return 1
    print("Seluruh model wajib tersedia dan terverifikasi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
