import "server-only";

/**
 * Penerus permintaan ke layanan API.
 *
 * Berkas ini menggantikan `rewrites()` pada `next.config.ts`, dan alasannya
 * ditemukan lewat kegagalan yang sungguhan: Next menuliskan hasil `rewrites()`
 * ke `.next/routes-manifest.json` pada saat `next build`, bukan membacanya
 * ketika server menyala. Nilai `VISIONQC_API_ORIGIN` karena itu ikut terpanggang
 * ke dalam image. Di dalam docker compose, image yang dibangun tanpa variabel itu
 * membawa `http://localhost:8000`, yang dari sudut pandang container `web`
 * menunjuk ke dirinya sendiri - halaman terbuka normal, tetapi setiap permintaan
 * ke layanan gagal dengan ECONNREFUSED.
 *
 * Route handler dijalankan per permintaan, sehingga `process.env` dibaca ketika
 * permintaannya datang. Satu image yang sama dapat diarahkan ke host lain hanya
 * dengan mengganti variabel lingkungan.
 */

/** Alamat layanan API. Dibaca setiap permintaan, bukan sekali saat modul dimuat. */
function apiOrigin(): string {
  return process.env.VISIONQC_API_ORIGIN ?? "http://localhost:8000";
}

// Header yang tidak boleh diteruskan.
//
// `host` menunjuk ke server ini, bukan ke layanan tujuan, dan panjang badan
// dihitung ulang oleh fetch. Sisanya bersifat per-sambungan: ia mengatur
// sambungan antara klien dan server ini, bukan sambungan berikutnya.
//
// `expect` yang paling menentukan. curl menambahkan `Expect: 100-continue`
// sendiri untuk badan yang besar, dan undici menolak header itu dengan
// `NotSupportedError`. Meneruskannya membuat setiap unggahan besar gagal di
// proksi - tampak seperti batas ukuran, padahal bukan.
const HEADER_DILEWATI = new Set([
  "host",
  "connection",
  "content-length",
  "transfer-encoding",
  "expect",
  "keep-alive",
  "proxy-connection",
  "te",
  "trailer",
  "upgrade",
]);

/**
 * Teruskan satu permintaan ke layanan lalu kembalikan responsnya apa adanya.
 *
 * @param request permintaan yang masuk ke antarmuka
 * @param upstreamPath jalur pada layanan, diawali garis miring
 */
export async function forwardToApi(
  request: Request,
  upstreamPath: string,
): Promise<Response> {
  const masuk = new URL(request.url);
  const tujuan = new URL(apiOrigin() + upstreamPath);
  tujuan.search = masuk.search;

  const headers = new Headers();
  request.headers.forEach((nilai, nama) => {
    if (!HEADER_DILEWATI.has(nama.toLowerCase())) headers.set(nama, nilai);
  });

  // Badan dibaca utuh alih-alih dialirkan. Unggahan dibatasi 10 MB oleh layanan,
  // sehingga menahannya sebentar di memori tidak menjadi persoalan, sedangkan
  // pengaliran menuntut `duplex: "half"` yang belum ada pada tipe RequestInit.
  const berbadan = request.method !== "GET" && request.method !== "HEAD";
  const body = berbadan ? await request.arrayBuffer() : undefined;

  let hulu: Response;
  try {
    hulu = await fetch(tujuan, {
      method: request.method,
      headers,
      body,
      redirect: "manual",
      cache: "no-store",
    });
  } catch (error) {
    // Layanan mati atau tidak dapat dihubungi. 502 menyatakan itu apa adanya:
    // yang gagal adalah tujuan di belakang, bukan antarmuka ini. Penyebabnya
    // dicatat ke log server alih-alih ditelan diam-diam; galat yang tidak
    // tercatat di sini pernah membuat kegagalan unggahan besar tampak seperti
    // layanan yang mati.
    console.error("gagal meneruskan ke %s: %o", tujuan.toString(), error);
    return Response.json(
      {
        detail:
          "Layanan API tidak dapat dihubungi dari antarmuka. Pastikan container api berjalan.",
      },
      { status: 502 },
    );
  }

  const keluar = new Headers(hulu.headers);
  keluar.delete("content-encoding");
  keluar.delete("content-length");
  keluar.delete("transfer-encoding");

  return new Response(hulu.body, { status: hulu.status, headers: keluar });
}
