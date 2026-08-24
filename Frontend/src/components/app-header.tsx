"use client";

import { ScanEye } from "lucide-react";
import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { fetchHealth, fetchModelInfo } from "@/lib/api";
import type { HealthStatus, ModelInfo } from "@/lib/contract";
import { cn } from "@/lib/utils";

type Kesiapan = "memeriksa" | "siap" | "terbatas" | "mati";

/**
 * Kepala halaman beserta lampu kesiapan layanan.
 *
 * Kesiapan ditanyakan sekali saat halaman dibuka, bukan diulang berkala.
 * Penjajakan berkala akan menjadi pemantauan latar, dan itu di luar batas
 * ruang lingkup tahap penyisihan; kalau layanan mati di tengah jalan, kegagalan
 * itu tetap muncul pada saat pemeriksaan dijalankan.
 */
export function AppHeader() {
  const [kesiapan, setKesiapan] = useState<Kesiapan>("memeriksa");
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);

  useEffect(() => {
    let batal = false;

    fetchHealth()
      .then((status) => {
        if (batal) return;
        setHealth(status);
        setKesiapan(status.status === "ok" ? "siap" : "terbatas");
      })
      .catch(() => {
        if (!batal) setKesiapan("mati");
      });

    fetchModelInfo()
      .then((info) => {
        if (!batal) setModel(info);
      })
      .catch(() => undefined);

    return () => {
      batal = true;
    };
  }, []);

  const lampu: Record<Kesiapan, { warna: string; teks: string }> = {
    memeriksa: { warna: "bg-ink-faint", teks: "menghubungi layanan" },
    siap: { warna: "bg-pass", teks: "layanan siap" },
    terbatas: { warna: "bg-hold", teks: "layanan berjalan terbatas" },
    mati: { warna: "bg-reject", teks: "layanan tidak terhubung" },
  };

  const aktif = health
    ? Object.entries(health.components)
        .filter(([, hidup]) => hidup)
        .map(([nama]) => nama)
    : [];

  return (
    <header className="border-b border-line bg-surface">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-4 sm:px-6">
        <div className="flex items-center gap-3">
          <span className="flex size-9 items-center justify-center rounded-lg bg-brand text-white">
            <ScanEye className="size-5" aria-hidden />
          </span>
          <div>
            <h1 className="text-base font-semibold leading-tight tracking-tight text-ink">
              VisionQC
            </h1>
            <p className="text-xs leading-tight text-ink-faint">
              Inspeksi mutu kemasan pangan dan minuman
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {model ? (
            // Nama versi panjang dan tidak punya titik pemenggalan alami.
            // Tanpa pemotongan, satu lencana ini melebarkan seluruh halaman di
            // layar ponsel dan membuat isinya meluber ke kanan.
            <Badge
              tone="neutral"
              title={`${model.version} - ${model.dataset}`}
              className="max-w-[15rem] overflow-hidden"
            >
              <span className="tabular truncate">{model.version}</span>
            </Badge>
          ) : null}
          {aktif.length > 0 ? (
            <Badge tone="brand" title={`Lapisan aktif: ${aktif.join(", ")}`}>
              {aktif.length} lapisan aktif
            </Badge>
          ) : null}
          <span className="inline-flex items-center gap-2 rounded-md border border-line bg-surface-sunken px-2 py-0.5 text-xs text-ink-soft">
            <span
              className={cn("size-2 rounded-full", lampu[kesiapan].warna)}
              aria-hidden
            />
            {lampu[kesiapan].teks}
          </span>
        </div>
      </div>
    </header>
  );
}
