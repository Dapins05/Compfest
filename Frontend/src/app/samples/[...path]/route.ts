import { forwardToApi } from "@/lib/proxy";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

interface Konteks {
  params: Promise<{ path: string[] }>;
}

/** Gambar contoh dilayani container api; antarmuka hanya meneruskannya. */
export async function GET(request: Request, konteks: Konteks): Promise<Response> {
  const { path } = await konteks.params;
  return forwardToApi(request, "/samples/" + path.join("/"));
}
