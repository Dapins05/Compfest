import type { VerdictLabel } from "@/lib/contract";

export type VerdictTone = "pass" | "reject" | "hold";

export interface VerdictPresentation {
  heading: string;
  tone: VerdictTone;
  caption: string;
}

/**
 * Terjemahkan putusan menjadi bentuk yang ditampilkan.
 *
 * `REVIEW` tetap ditangani meskipun mesin keputusan berjalan pada mode biner
 * dan tidak pernah mengembalikannya. Nilai itu masih sah menurut kontrak, dan
 * cabang yang hilang akan tampak sebagai pita kosong tanpa keterangan apa pun
 * seandainya mode tiga kelas dinyalakan kembali.
 */
export function presentVerdict(verdict: VerdictLabel): VerdictPresentation {
  switch (verdict) {
    case "PASS":
      return {
        heading: "LOLOS",
        tone: "pass",
        caption: "Produk memenuhi syarat mutu",
      };
    case "REJECT":
      return {
        heading: "DITOLAK",
        tone: "reject",
        caption: "Produk tidak memenuhi syarat mutu",
      };
    case "REVIEW":
      return {
        heading: "DITAHAN",
        tone: "hold",
        caption: "Putusan diserahkan ke pemeriksa manusia",
      };
  }
}
