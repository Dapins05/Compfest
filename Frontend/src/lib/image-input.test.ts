import { describe, expect, it } from "vitest";

import {
  MAX_FILE_SIZE_MB,
  MIN_IMAGE_SIDE_PX,
  formatBytes,
  validateImageDimensions,
  validateImageFile,
} from "@/lib/image-input";

function berkas(bytes: number, type: string, name = "produk.jpg"): File {
  return new File([new Uint8Array(bytes)], name, { type });
}

describe("validateImageFile", () => {
  it("menerima ketiga format yang didukung layanan", () => {
    for (const type of ["image/jpeg", "image/png", "image/webp"]) {
      expect(validateImageFile(berkas(2048, type))).toBeNull();
    }
  });

  it("menolak berkas yang bukan gambar", () => {
    const hasil = validateImageFile(berkas(2048, "text/plain", "catatan.txt"));
    expect(hasil?.code).toBe("tipe");
  });

  it("menolak berkas tanpa tipe MIME tanpa menampilkan kata kosong", () => {
    const hasil = validateImageFile(berkas(2048, ""));
    expect(hasil?.code).toBe("tipe");
    expect(hasil?.message).toContain("tanpa tipe");
  });

  it("menolak berkas kosong sebelum memeriksa tipenya", () => {
    // Berkas nol bita bertipe gambar tetap tidak dapat diperiksa, dan alasannya
    // harus menyebut isinya yang kosong, bukan tipenya.
    expect(validateImageFile(berkas(0, "image/png"))?.code).toBe("kosong");
  });

  it("menolak berkas yang melampaui batas ukuran layanan", () => {
    const melampaui = (MAX_FILE_SIZE_MB + 1) * 1024 * 1024;
    expect(validateImageFile(berkas(melampaui, "image/jpeg"))?.code).toBe("ukuran");
  });

  it("menerima berkas tepat pada batas ukuran", () => {
    // Batasnya "melampaui", bukan "mencapai". Layanan memakai pembandingan yang
    // sama, jadi berkas tepat 10 MB harus diterima keduanya.
    expect(validateImageFile(berkas(MAX_FILE_SIZE_MB * 1024 * 1024, "image/png"))).toBeNull();
  });
});

describe("validateImageDimensions", () => {
  it("menolak gambar yang sisi terpendeknya di bawah minimum", () => {
    const hasil = validateImageDimensions(1024, MIN_IMAGE_SIDE_PX - 1);
    expect(hasil?.code).toBe("dimensi");
    expect(hasil?.message).toContain(String(MIN_IMAGE_SIDE_PX - 1));
  });

  it("menerima gambar tepat pada ukuran minimum", () => {
    expect(validateImageDimensions(MIN_IMAGE_SIDE_PX, MIN_IMAGE_SIDE_PX)).toBeNull();
  });
});

describe("formatBytes", () => {
  it("memilih satuan menurut besarnya", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2 KB");
    expect(formatBytes(3 * 1024 * 1024)).toBe("3.0 MB");
  });
});
