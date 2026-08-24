"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { AppHeader } from "@/components/app-header";
import { ResultPanel, type PanelPhase } from "@/components/result-panel";
import { SourcePanel, type ChosenImage } from "@/components/source-panel";
import { ApiError, inspectImage } from "@/lib/api";
import type { InspectionResult } from "@/lib/contract";
import { validateImageDimensions, validateImageFile } from "@/lib/image-input";
import { readImageSize } from "@/lib/read-image";

/**
 * Halaman inti VisionQC.
 *
 * Satu citra masuk, satu putusan keluar. Tidak ada riwayat pemeriksaan, tidak
 * ada papan analitik, dan tidak ada keadaan yang bertahan setelah halaman
 * ditutup - ketiganya di luar batas ruang lingkup tahap penyisihan, dan
 * ketiadaannya juga berarti tidak ada citra pengguna yang tersimpan di mana
 * pun.
 */
export default function Page() {
  const [chosen, setChosen] = useState<ChosenImage | null>(null);
  const [rejection, setRejection] = useState<string | null>(null);
  const [phase, setPhase] = useState<PanelPhase>("kosong");
  const [result, setResult] = useState<InspectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Object URL dilepas sendiri; kalau tidak, setiap gambar yang pernah dipilih
  // tetap tertahan di memori peramban sampai tabnya ditutup.
  const previousUrl = useRef<string | null>(null);
  const lepaskan = useCallback(() => {
    if (previousUrl.current) URL.revokeObjectURL(previousUrl.current);
    previousUrl.current = null;
  }, []);
  useEffect(() => lepaskan, [lepaskan]);

  const clear = useCallback(() => {
    lepaskan();
    setChosen(null);
    setRejection(null);
    setResult(null);
    setError(null);
    setPhase("kosong");
  }, [lepaskan]);

  const select = useCallback(
    async (file: File) => {
      setResult(null);
      setError(null);
      setPhase("kosong");

      const ditolak = validateImageFile(file);
      if (ditolak) {
        lepaskan();
        setChosen(null);
        setRejection(ditolak.message);
        return;
      }

      const url = URL.createObjectURL(file);
      try {
        const { width, height } = await readImageSize(url);
        const dimensiDitolak = validateImageDimensions(width, height);
        if (dimensiDitolak) {
          URL.revokeObjectURL(url);
          lepaskan();
          setChosen(null);
          setRejection(dimensiDitolak.message);
          return;
        }
      } catch {
        URL.revokeObjectURL(url);
        lepaskan();
        setChosen(null);
        setRejection("Berkas ini tidak dapat dibaca sebagai gambar.");
        return;
      }

      lepaskan();
      previousUrl.current = url;
      setRejection(null);
      setChosen({ file, previewUrl: url });
    },
    [lepaskan],
  );

  const inspect = useCallback(async () => {
    if (!chosen) return;
    setPhase("memeriksa");
    setError(null);
    setResult(null);

    try {
      setResult(await inspectImage(chosen.file, chosen.file.name));
      setPhase("selesai");
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught.message
          : "Terjadi kesalahan yang tidak dikenali saat memeriksa citra.",
      );
      setPhase("galat");
    }
  }, [chosen]);

  return (
    <div className="flex min-h-dvh flex-col">
      <AppHeader />

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-5 sm:px-6 sm:py-6">
        <div className="grid items-start gap-5 sm:gap-6 lg:grid-cols-[minmax(0,25rem)_minmax(0,1fr)]">
          <SourcePanel
            chosen={chosen}
            rejection={rejection}
            busy={phase === "memeriksa"}
            onSelect={(file) => void select(file)}
            onClear={clear}
            onInspect={() => void inspect()}
          />
          <ResultPanel
            phase={phase}
            result={result}
            error={error}
            originalUrl={chosen?.previewUrl ?? null}
          />
        </div>
      </main>

      <footer className="border-t border-line px-4 py-4 sm:px-6">
        <p className="mx-auto max-w-7xl text-xs leading-relaxed text-ink-faint">
          Seluruh ambang keputusan bersifat statis dan dibaca sekali saat layanan dinyalakan.
          Citra diproses di dalam permintaan, tidak pernah ditulis ke disk, dan wajah yang
          kebetulan terpotret diburamkan sebelum citra mencapai model.
        </p>
      </footer>
    </div>
  );
}
