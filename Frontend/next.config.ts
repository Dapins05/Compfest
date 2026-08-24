import type { NextConfig } from "next";

/**
 * Penerusan permintaan ke layanan TIDAK diatur di berkas ini.
 *
 * `rewrites()` sempat dipakai di sini, lalu terbukti keliru: Next menuliskan
 * hasilnya ke `.next/routes-manifest.json` pada saat `next build`, bukan
 * membacanya ketika server menyala. Alamat layanan karena itu ikut terpanggang
 * ke dalam image, dan di dalam docker compose image yang dibangun tanpa variabel
 * itu membawa `http://localhost:8000` - yang dari sudut pandang container `web`
 * menunjuk ke dirinya sendiri. Halaman tetap terbuka normal, tetapi setiap
 * permintaan ke layanan gagal dengan ECONNREFUSED.
 *
 * Penerusannya kini ditangani route handler di `src/app/api/[...path]/route.ts`
 * dan `src/app/samples/[...path]/route.ts`, yang membaca VISIONQC_API_ORIGIN
 * pada setiap permintaan.
 */
const nextConfig: NextConfig = {
  // Keluaran mandiri memuat hanya berkas yang benar-benar dipakai saat jalan,
  // sehingga image runtime tidak perlu membawa node_modules seutuhnya. Mode ini
  // dinyalakan hanya di dalam Docker, lewat VISIONQC_STANDALONE pada Dockerfile.
  //
  // Alasannya khas Windows: menyusun keluaran mandiri menuntut pembuatan
  // symlink, dan Windows menolaknya (EPERM) kecuali Developer Mode dinyalakan
  // atau perintahnya dijalankan sebagai administrator. Menyalakannya tanpa
  // syarat membuat `pnpm build` gagal di mesin anggota tim, padahal keluaran
  // mandiri hanya dibutuhkan image kontainer.
  output: process.env.VISIONQC_STANDALONE === "1" ? "standalone" : undefined,
};

export default nextConfig;
