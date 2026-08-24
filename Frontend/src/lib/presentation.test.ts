import { describe, expect, it } from "vitest";

import { formatConfidence, formatLatency, formatPercent, formatScore } from "@/lib/format";
import { presentVerdict } from "@/lib/verdict";

describe("presentVerdict", () => {
  it("memberi nada dan judul yang berbeda untuk setiap putusan", () => {
    expect(presentVerdict("PASS").tone).toBe("pass");
    expect(presentVerdict("REJECT").tone).toBe("reject");
    expect(presentVerdict("REVIEW").tone).toBe("hold");
  });

  it("tidak pernah mengembalikan judul kosong", () => {
    // REVIEW tidak dikembalikan mesin keputusan pada mode biner, tetapi masih
    // sah menurut kontrak. Cabang yang hilang akan tampak sebagai pita tanpa
    // keterangan apa pun, bukan sebagai galat.
    for (const verdict of ["PASS", "REJECT", "REVIEW"] as const) {
      const tampilan = presentVerdict(verdict);
      expect(tampilan.heading).not.toBe("");
      expect(tampilan.caption).not.toBe("");
    }
  });
});

describe("pemformatan angka", () => {
  it("memakai koma sebagai pemisah desimal", () => {
    expect(formatPercent(11.1115)).toBe("11,11 %");
    expect(formatScore(54.39267)).toBe("54,39");
  });

  it("mengubah keyakinan menjadi persen dan menangani nilai kosong", () => {
    expect(formatConfidence(0.9358512)).toBe("93,6 %");
    expect(formatConfidence(null)).toBe("-");
    expect(formatConfidence(undefined)).toBe("-");
  });

  it("berpindah ke satuan detik ketika latensi melewati seribu milidetik", () => {
    expect(formatLatency(272)).toBe("272 ms");
    expect(formatLatency(1450)).toBe("1,45 s");
  });
});
