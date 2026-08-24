import type { Defect } from "@/lib/contract";
import { formatConfidence, formatPercent } from "@/lib/format";

interface DefectListProps {
  defects: Defect[];
}

/**
 * Daftar cacat yang terdeteksi.
 *
 * Setiap baris memuat keyakinan model dan luasnya, bukan hanya namanya, supaya
 * operator dapat menilai sendiri seberapa kuat dasar putusan yang diambil
 * sistem alih-alih harus mempercayainya begitu saja.
 */
export function DefectList({ defects }: DefectListProps) {
  if (defects.length === 0) {
    return (
      <p className="rounded-lg border border-line bg-surface-sunken px-4 py-3 text-sm text-ink-soft">
        Tidak ada cacat yang terdeteksi pada gambar ini.
      </p>
    );
  }

  return (
    <ul className="divide-y divide-line overflow-hidden rounded-lg border border-line">
      {defects.map((defect, index) => (
        <li
          key={`${defect.type}-${defect.bbox.x}-${defect.bbox.y}-${index}`}
          className="bg-surface px-4 py-3"
        >
          <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <span className="text-sm font-medium text-ink">{defect.label}</span>
            <span className="tabular text-xs text-ink-faint">
              keyakinan {formatConfidence(defect.confidence)}
              {defect.area_pct !== null && defect.area_pct !== undefined
                ? ` · luas ${formatPercent(defect.area_pct)}`
                : ""}
            </span>
          </div>
          <div className="mt-2 flex items-center gap-3">
            <div
              className="h-1.5 flex-1 overflow-hidden rounded-full bg-line"
              role="presentation"
            >
              <div
                className="h-full rounded-full bg-brand"
                style={{ width: `${Math.min(100, Math.max(0, defect.confidence * 100))}%` }}
              />
            </div>
            <span className="tabular shrink-0 text-[11px] text-ink-faint">
              kotak {defect.bbox.x}, {defect.bbox.y} &middot; {defect.bbox.w}&times;
              {defect.bbox.h} px
            </span>
          </div>
        </li>
      ))}
    </ul>
  );
}
