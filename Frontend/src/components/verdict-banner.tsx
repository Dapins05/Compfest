import { CircleCheck, CircleSlash, TriangleAlert } from "lucide-react";

import type { VerdictLabel } from "@/lib/contract";
import { cn } from "@/lib/utils";
import { presentVerdict, type VerdictTone } from "@/lib/verdict";

interface VerdictBannerProps {
  verdict: VerdictLabel;
  reason: string;
}

const nada: Record<VerdictTone, string> = {
  pass: "border-pass-line bg-pass-soft text-pass",
  reject: "border-reject-line bg-reject-soft text-reject",
  hold: "border-hold-line bg-hold-soft text-hold",
};

const ikon = {
  pass: CircleCheck,
  reject: CircleSlash,
  hold: TriangleAlert,
} as const;

/**
 * Pita putusan.
 *
 * Warna tidak pernah menjadi satu-satunya penanda: setiap putusan disertai kata
 * dan ikon yang berbeda bentuk, sehingga tetap terbaca pada cetakan hitam putih
 * maupun oleh mata yang sulit membedakan merah dan hijau.
 */
export function VerdictBanner({ verdict, reason }: VerdictBannerProps) {
  const tampilan = presentVerdict(verdict);
  const Ikon = ikon[tampilan.tone];

  return (
    <div
      role="status"
      aria-live="polite"
      className={cn("flex items-start gap-4 rounded-lg border px-5 py-4", nada[tampilan.tone])}
    >
      <Ikon className="mt-0.5 size-7 shrink-0" aria-hidden />
      <div className="min-w-0">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
          <span className="text-xl font-semibold tracking-tight">{tampilan.heading}</span>
          <span className="text-xs opacity-80">{tampilan.caption}</span>
        </div>
        <p className="mt-1 text-sm leading-relaxed text-ink-soft">{reason}</p>
      </div>
    </div>
  );
}
