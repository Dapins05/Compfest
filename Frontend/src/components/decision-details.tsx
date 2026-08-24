import type { InspectionResult } from "@/lib/contract";
import { formatConfidence } from "@/lib/format";

interface DecisionDetailsProps {
  result: InspectionResult;
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5">
      <dt className="text-xs text-ink-faint">{label}</dt>
      <dd className="tabular text-xs text-ink">{value}</dd>
    </div>
  );
}

/**
 * Rincian penelusuran putusan.
 *
 * Isinya dilaporkan supaya putusan dapat ditelusuri, bukan supaya dipakai
 * mengambil keputusan sendiri: pada mode biner himpunan prediksi conformal
 * tidak lagi menahan putusan apa pun. Bagian ini dilipat secara bawaan karena
 * operator lini tidak membutuhkannya setiap kali, sedangkan penilai dan
 * pemeriksa membutuhkannya.
 */
export function DecisionDetails({ result }: DecisionDetailsProps) {
  const decision = result.decision;

  return (
    <details className="group rounded-lg border border-line bg-surface">
      <summary className="cursor-pointer select-none px-4 py-3 text-xs font-medium text-ink-soft marker:content-none">
        Rincian keputusan dan versi model
        <span className="ml-2 text-ink-faint group-open:hidden">tampilkan</span>
        <span className="ml-2 hidden text-ink-faint group-open:inline">sembunyikan</span>
      </summary>
      <dl className="divide-y divide-line border-t border-line px-4 py-2">
        {decision ? (
          <>
            <Row
              label="Peluang terkalibrasi"
              value={formatConfidence(decision.calibrated_probability)}
            />
            <Row
              label="Himpunan prediksi conformal"
              value={decision.prediction_set.join(", ") || "-"}
            />
            <Row label="Severity" value={decision.severity.toFixed(4)} />
            <Row label="Alpha conformal" value={decision.conformal_alpha.toFixed(2)} />
          </>
        ) : (
          <Row label="Rincian keputusan" value="tidak dilaporkan" />
        )}
        <Row label="Kode batch" value={result.batch_code ?? "tidak aktif"} />
        <Row label="Versi model" value={result.model_version || "-"} />
      </dl>
    </details>
  );
}
