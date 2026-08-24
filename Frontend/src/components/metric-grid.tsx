import type { InspectionResult } from "@/lib/contract";
import { formatConfidence, formatLatency, formatPercent, formatScore } from "@/lib/format";
import { cn } from "@/lib/utils";

interface MetricGridProps {
  result: InspectionResult;
}

function Tile({
  label,
  value,
  note,
  tone = "netral",
}: {
  label: string;
  value: string;
  note?: string;
  tone?: "netral" | "reject";
}) {
  return (
    <div className="rounded-lg border border-line bg-surface px-4 py-3">
      <div className="text-[11px] font-medium uppercase tracking-wide text-ink-faint">
        {label}
      </div>
      <div
        className={cn(
          "tabular mt-1 text-lg font-semibold",
          tone === "reject" ? "text-reject" : "text-ink",
        )}
      >
        {value}
      </div>
      {note ? <div className="mt-0.5 text-[11px] text-ink-faint">{note}</div> : null}
    </div>
  );
}

/**
 * Angka pendukung putusan.
 *
 * Ambang luas cacat sengaja tidak ditulis ulang di sini. Nilainya sudah dibawa
 * medan alasan yang datang dari layanan, dan menyalinnya ke Frontend berarti
 * menambah satu tempat lagi yang harus ikut berubah setiap kali berkas
 * konfigurasi modul AI disunting.
 */
export function MetricGrid({ result }: MetricGridProps) {
  const anomali = result.anomaly;

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <Tile
        label="Luas cacat"
        value={formatPercent(result.defect_area_pct)}
        note="terhadap luas gambar"
      />
      <Tile
        label="Keyakinan"
        value={formatConfidence(result.confidence)}
        note="cacat paling meyakinkan"
      />
      {anomali ? (
        <Tile
          label="Skor anomali"
          value={formatScore(anomali.score)}
          note={`ambang ${formatScore(anomali.threshold)}${anomali.exceeded ? " · terlampaui" : ""}`}
          tone={anomali.exceeded ? "reject" : "netral"}
        />
      ) : (
        <Tile label="Skor anomali" value="-" note="lapisan anomali tidak aktif" />
      )}
      <Tile
        label="Waktu proses"
        value={formatLatency(result.latency_ms)}
        note="inferensi di sisi layanan"
      />
    </div>
  );
}
