"use client";

import { CircleAlert, Loader2, ScanSearch } from "lucide-react";
import { useEffect, useState } from "react";

import { DecisionDetails } from "@/components/decision-details";
import { DefectList } from "@/components/defect-list";
import { MetricGrid } from "@/components/metric-grid";
import { VerdictBanner } from "@/components/verdict-banner";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import type { InspectionResult } from "@/lib/contract";
import { cn } from "@/lib/utils";

export type PanelPhase = "kosong" | "memeriksa" | "selesai" | "galat";

interface ResultPanelProps {
  phase: PanelPhase;
  result: InspectionResult | null;
  error: string | null;
  originalUrl: string | null;
}

/**
 * Medan `annotated_image_base64` sudah berupa data URI utuh pada layanan yang
 * berjalan sekarang. Awalannya tetap ditambahkan bila hilang, karena nama
 * medannya menjanjikan base64 telanjang dan bentuk itu masih sah menurut
 * kontrak.
 */
function sumberGambar(base64: string): string {
  return base64.startsWith("data:") ? base64 : `data:image/jpeg;base64,${base64}`;
}

export function ResultPanel({ phase, result, error, originalUrl }: ResultPanelProps) {
  const [tampilAsli, setTampilAsli] = useState(false);

  // Setiap hasil baru dimulai dari gambar beranotasi, bukan dari pilihan yang
  // tersisa dari pemeriksaan sebelumnya.
  useEffect(() => setTampilAsli(false), [result]);

  return (
    <Card className="flex min-h-[30rem] flex-col">
      <CardHeader
        title="Hasil pemeriksaan"
        description="Putusan sistem beserta angka yang mendasarinya"
        icon={<ScanSearch className="size-4" aria-hidden />}
        action={
          phase === "selesai" && result && originalUrl ? (
            <div
              role="group"
              aria-label="Tampilan gambar"
              className="flex shrink-0 rounded-md border border-line p-0.5"
            >
              {(
                [
                  ["Beranotasi", false],
                  ["Asli", true],
                ] as const
              ).map(([label, nilai]) => (
                <button
                  key={label}
                  type="button"
                  aria-pressed={tampilAsli === nilai}
                  onClick={() => setTampilAsli(nilai)}
                  className={cn(
                    "rounded px-2.5 py-1 text-xs font-medium transition-colors",
                    tampilAsli === nilai
                      ? "bg-brand text-white"
                      : "text-ink-faint hover:text-ink",
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
          ) : null
        }
      />

      <CardBody className="flex flex-1 flex-col gap-4">
        {phase === "kosong" ? (
          <Placeholder
            icon={<ScanSearch className="size-6" aria-hidden />}
            title="Belum ada pemeriksaan"
            body="Pilih satu citra produk di panel kiri, lalu tekan Periksa. Hasilnya muncul di sini."
          />
        ) : null}

        {phase === "memeriksa" ? (
          <Placeholder
            icon={<Loader2 className="size-6 animate-spin" aria-hidden />}
            title="Sedang memeriksa"
            body="Citra sedang melewati deteksi, segmentasi, dan pemeriksaan anomali di sisi layanan."
          />
        ) : null}

        {phase === "galat" && error ? (
          <div
            role="alert"
            className="flex items-start gap-3 rounded-lg border border-reject-line bg-reject-soft px-4 py-3"
          >
            <CircleAlert className="mt-0.5 size-5 shrink-0 text-reject" aria-hidden />
            <div>
              <p className="text-sm font-medium text-reject">Pemeriksaan tidak dapat diselesaikan</p>
              <p className="mt-0.5 text-sm leading-relaxed text-ink-soft">{error}</p>
            </div>
          </div>
        ) : null}

        {phase === "selesai" && result ? (
          <>
            <VerdictBanner verdict={result.verdict} reason={result.reason} />

            <div className="flex justify-center overflow-hidden rounded-lg border border-line bg-surface-sunken">
              {/* Gambar hasil datang sebagai data URI di dalam respons dan tidak
                  punya URL yang dapat diambil ulang, sehingga tidak ada yang
                  dapat dioptimalkan next/image di sini. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={
                  tampilAsli && originalUrl
                    ? originalUrl
                    : sumberGambar(result.annotated_image_base64)
                }
                alt={
                  tampilAsli
                    ? "Citra produk sebelum diperiksa"
                    : "Citra produk dengan penandaan cacat dari sistem"
                }
                className="max-h-[26rem] w-auto max-w-full object-contain"
              />
            </div>

            <MetricGrid result={result} />
            <DefectList defects={result.defects ?? []} />
            <DecisionDetails result={result} />
          </>
        ) : null}
      </CardBody>
    </Card>
  );
}

function Placeholder({
  icon,
  title,
  body,
}: {
  icon: React.ReactNode;
  title: string;
  body: string;
}) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 px-8 py-16 text-center">
      <span className="flex size-12 items-center justify-center rounded-full bg-surface-sunken text-ink-faint">
        {icon}
      </span>
      <p className="text-sm font-medium text-ink">{title}</p>
      <p className="max-w-sm text-sm leading-relaxed text-ink-faint">{body}</p>
    </div>
  );
}
