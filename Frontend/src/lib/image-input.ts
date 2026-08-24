/**
 * Pemeriksaan berkas gambar di sisi klien.
 *
 * Batas di bawah menyalin batas yang ditegakkan layanan, dan sumber
 * kebenarannya tetap `AI_model/configs/inference.yaml`. Salinan ini sengaja
 * TIDAK dijadikan penentu: layanan tetap memeriksa ulang setiap kiriman, dan
 * jawabannyalah yang berlaku. Gunanya di sini hanya satu, yaitu memberi tahu
 * pengguna secepatnya bahwa berkasnya tidak akan diterima - tanpa membuatnya
 * menunggu satu perjalanan jaringan beserta unggahan sepuluh megabita lebih
 * dulu.
 */

export const ACCEPTED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"] as const;
export const ACCEPTED_EXTENSIONS = ".jpg,.jpeg,.png,.webp";
export const MAX_FILE_SIZE_MB = 10;

/**
 * Sisi terpendek minimum. Modul AI menolak gambar di bawah nilai ini karena
 * lebih kecil daripada masukan jaringan setelah prapemrosesan.
 */
export const MIN_IMAGE_SIDE_PX = 224;

export type RejectionCode = "tipe" | "ukuran" | "kosong" | "dimensi" | "tidak-terbaca";

export interface ImageRejection {
  code: RejectionCode;
  message: string;
}

/** Ubah jumlah bita menjadi keterangan yang terbaca manusia. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Periksa tipe dan ukuran berkas. Mengembalikan null bila berkasnya sah. */
export function validateImageFile(file: File): ImageRejection | null {
  if (file.size === 0) {
    return { code: "kosong", message: "Berkas kosong, tidak ada isi yang dapat diperiksa." };
  }

  const mime = file.type.toLowerCase();
  if (!ACCEPTED_MIME_TYPES.includes(mime as (typeof ACCEPTED_MIME_TYPES)[number])) {
    const disebut = mime === "" ? "tanpa tipe" : mime;
    return {
      code: "tipe",
      message: `Berkas ${disebut} tidak didukung. Gunakan JPG, PNG, atau WEBP.`,
    };
  }

  const megabytes = file.size / (1024 * 1024);
  if (megabytes > MAX_FILE_SIZE_MB) {
    return {
      code: "ukuran",
      message: `Ukuran ${formatBytes(file.size)} melampaui batas ${MAX_FILE_SIZE_MB} MB.`,
    };
  }

  return null;
}

/** Periksa dimensi gambar yang sudah terbaca. Null berarti sah. */
export function validateImageDimensions(width: number, height: number): ImageRejection | null {
  const terpendek = Math.min(width, height);
  if (terpendek < MIN_IMAGE_SIDE_PX) {
    return {
      code: "dimensi",
      message:
        `Sisi terpendek gambar ${terpendek} piksel, di bawah minimum ` +
        `${MIN_IMAGE_SIDE_PX} piksel. Gunakan foto beresolusi lebih tinggi.`,
    };
  }
  return null;
}
