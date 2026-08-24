import { forwardToApi } from "@/lib/proxy";

// Dijalankan pada setiap permintaan; tidak ada yang boleh disimpan di cache,
// karena inilah jalur yang membawa hasil inspeksi.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

interface Konteks {
  params: Promise<{ path: string[] }>;
}

/**
 * Petakan jalur antarmuka ke jalur layanan.
 *
 * `/api/healthz` menunjuk ke `/healthz`, yang berada di akar layanan. Jalur itu
 * sengaja ditaruh di bawah `/api` pada sisi antarmuka supaya tidak bertabrakan
 * dengan kesehatan container `web` sendiri.
 */
function jalurLayanan(segmen: string[]): string {
  if (segmen.length === 1 && segmen[0] === "healthz") return "/healthz";
  return "/api/" + segmen.join("/");
}

async function teruskan(request: Request, konteks: Konteks): Promise<Response> {
  const { path } = await konteks.params;
  return forwardToApi(request, jalurLayanan(path));
}

export const GET = teruskan;
export const POST = teruskan;
