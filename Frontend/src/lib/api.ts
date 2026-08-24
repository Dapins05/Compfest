/**
 * Klien layanan VisionQC.
 *
 * Seluruh permintaan ditujukan ke origin yang sama dengan halaman ini, lalu
 * diteruskan ke kontainer `api` oleh penulisan ulang jalur di `next.config.ts`.
 * Alamat layanan karena itu tidak pernah ikut masuk ke berkas JavaScript yang
 * dikirim ke peramban - kalau ia ditanam lewat `NEXT_PUBLIC_...`, nilainya
 * terkunci pada saat `next build`, sehingga image Docker yang sudah jadi tidak
 * dapat diarahkan ke host lain tanpa dibangun ulang.
 *
 * Setiap respons dilewatkan Zod sebelum dikembalikan. Respons yang bentuknya
 * tidak sesuai kontrak ditolak di sini, bukan dibiarkan menjadi `undefined`
 * pada saat digambar ke layar.
 */

import {
  healthStatusSchema,
  inspectionResultSchema,
  modelInfoSchema,
  sampleListSchema,
  type HealthStatus,
  type InspectionResult,
  type ModelInfo,
  type SampleImage,
} from "@/lib/contract";

export type ApiErrorKind = "masukan" | "belum-siap" | "server" | "jaringan" | "kontrak";

/** Galat yang sudah digolongkan, supaya antarmuka tahu harus menyalahkan apa. */
export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;

  constructor(kind: ApiErrorKind, message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

/**
 * Ambil pesan galat dari badan respons.
 *
 * FastAPI memakai medan `detail` untuk dua bentuk yang berbeda: untaian biasa
 * pada `HTTPException`, dan daftar galat validasi pada 422. Keduanya ditangani
 * supaya pengguna tidak pernah melihat "[object Object]".
 */
async function bacaDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail)) {
        const pesan = detail
          .map((item) =>
            typeof item === "object" && item !== null && "msg" in item
              ? String((item as { msg: unknown }).msg)
              : null,
          )
          .filter((item): item is string => item !== null);
        if (pesan.length > 0) return pesan.join("; ");
      }
    }
  } catch {
    // Badan yang bukan JSON bukan hal yang perlu dilaporkan sendiri; kode
    // status di bawah sudah cukup menjelaskan.
  }
  return `Layanan menjawab dengan kode ${response.status}.`;
}

function golongkan(status: number): ApiErrorKind {
  if (status === 503) return "belum-siap";
  if (status >= 400 && status < 500) return "masukan";
  return "server";
}

async function minta(path: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(path, init);
  } catch {
    throw new ApiError(
      "jaringan",
      "Layanan tidak dapat dihubungi. Pastikan kontainer api sedang berjalan.",
    );
  }
}

async function bacaJson<T>(
  path: string,
  parse: (data: unknown) => T,
  init?: RequestInit,
): Promise<T> {
  const response = await minta(path, init);
  if (!response.ok) {
    throw new ApiError(golongkan(response.status), await bacaDetail(response), response.status);
  }

  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new ApiError("kontrak", "Layanan mengembalikan respons yang bukan JSON.");
  }

  try {
    return parse(data);
  } catch {
    throw new ApiError(
      "kontrak",
      "Bentuk respons layanan tidak sesuai kontrak. Jalankan ulang pembangkitan tipe " +
        "(pnpm gen:api) setelah skema modul AI berubah.",
    );
  }
}

/** Periksa satu gambar. Ini satu-satunya jalur yang menjalankan model. */
export function inspectImage(file: Blob, filename: string): Promise<InspectionResult> {
  const form = new FormData();
  form.append("file", file, filename);
  return bacaJson("/api/v1/inspect", (data) => inspectionResultSchema.parse(data), {
    method: "POST",
    body: form,
  });
}

/** Kesiapan layanan beserta lapisan model yang aktif. */
export function fetchHealth(): Promise<HealthStatus> {
  return bacaJson("/api/healthz", (data) => healthStatusSchema.parse(data), {
    cache: "no-store",
  });
}

/** Keterangan model yang sedang dilayani. */
export function fetchModelInfo(): Promise<ModelInfo> {
  return bacaJson("/api/v1/model-info", (data) => modelInfoSchema.parse(data), {
    cache: "no-store",
  });
}

/** Daftar gambar contoh yang disediakan layanan. */
export function fetchSamples(): Promise<SampleImage[]> {
  return bacaJson("/api/v1/samples", (data) => sampleListSchema.parse(data), {
    cache: "no-store",
  });
}

/** Unduh satu gambar contoh sebagai berkas, supaya dapat dikirim ke /inspect. */
export async function fetchSampleFile(sample: SampleImage): Promise<File> {
  const response = await minta(sample.url);
  if (!response.ok) {
    throw new ApiError(
      golongkan(response.status),
      `Gambar contoh ${sample.name} tidak dapat diambil.`,
      response.status,
    );
  }
  const blob = await response.blob();
  return new File([blob], sample.name, { type: blob.type || "image/png" });
}
