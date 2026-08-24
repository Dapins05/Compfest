/**
 * Pemformatan angka untuk ditampilkan.
 *
 * Seluruhnya memakai konvensi Indonesia - koma sebagai pemisah desimal -
 * sehingga angka pada antarmuka terbaca sama dengan angka pada proposal.
 */

const desimal = (angka: number, digit: number): string =>
  angka.toLocaleString("id-ID", {
    minimumFractionDigits: digit,
    maximumFractionDigits: digit,
  });

/** Persentase luas cacat, dua angka di belakang koma. */
export function formatPercent(value: number): string {
  return `${desimal(value, 2)} %`;
}

/** Keyakinan model sebagai persen bulat, mis. 0.9359 menjadi "93,6 %". */
export function formatConfidence(value: number | null | undefined): string {
  if (value === null || value === undefined) return "-";
  return `${desimal(value * 100, 1)} %`;
}

/** Skor anomali. Skalanya jarak Mahalanobis, bukan 0..1. */
export function formatScore(value: number): string {
  return desimal(value, 2);
}

/** Latensi satu permintaan. */
export function formatLatency(milliseconds: number): string {
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  return `${desimal(milliseconds / 1000, 2)} s`;
}
