"use client";

import { AlertCircle, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";

import { fetchSamples, fetchSampleFile } from "@/lib/api";
import type { SampleImage } from "@/lib/contract";
import { cn } from "@/lib/utils";

interface SamplePickerProps {
  onSelect: (file: File) => void;
  disabled?: boolean;
}

/**
 * Daftar gambar contoh yang dilayani `/api/v1/samples`.
 *
 * Gunanya supaya sistem dapat dicoba tanpa menyiapkan foto lebih dulu - antara
 * lain oleh penilai yang menjalankan repositori ini di komputernya sendiri.
 * Berkas contohnya diunduh kembali dari layanan lalu dikirim lewat jalur yang
 * sama persis dengan unggahan biasa, sehingga tidak ada jalan pintas yang
 * hanya bekerja untuk contoh.
 */
export function SamplePicker({ onSelect, disabled = false }: SamplePickerProps) {
  const [samples, setSamples] = useState<SampleImage[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [memuat, setMemuat] = useState<string | null>(null);

  useEffect(() => {
    let batal = false;
    fetchSamples()
      .then((daftar) => {
        if (!batal) setSamples(daftar);
      })
      .catch(() => {
        if (!batal) setError("Daftar gambar contoh tidak dapat diambil dari layanan.");
      });
    return () => {
      batal = true;
    };
  }, []);

  const pilih = async (sample: SampleImage) => {
    setMemuat(sample.name);
    try {
      onSelect(await fetchSampleFile(sample));
    } catch {
      setError(`Gambar contoh ${sample.name} tidak dapat diambil.`);
    } finally {
      setMemuat(null);
    }
  };

  if (error) {
    return (
      <div className="flex aspect-4/3 flex-col items-center justify-center gap-2 rounded-lg border border-line bg-surface-sunken px-6 text-center">
        <AlertCircle className="size-5 text-ink-faint" aria-hidden />
        <p className="text-sm text-ink-soft">{error}</p>
      </div>
    );
  }

  if (samples === null) {
    return (
      <div className="flex aspect-4/3 items-center justify-center rounded-lg border border-line bg-surface-sunken">
        <Loader2 className="size-5 animate-spin text-ink-faint" aria-hidden />
        <span className="sr-only">Memuat daftar gambar contoh</span>
      </div>
    );
  }

  if (samples.length === 0) {
    return (
      <div className="flex aspect-4/3 items-center justify-center rounded-lg border border-line bg-surface-sunken px-6 text-center">
        <p className="text-sm text-ink-soft">Layanan tidak menyediakan gambar contoh.</p>
      </div>
    );
  }

  return (
    <div className="grid aspect-4/3 grid-cols-3 content-start gap-2 overflow-y-auto rounded-lg border border-line bg-surface-sunken p-2">
      {samples.map((sample) => (
        <button
          key={sample.name}
          type="button"
          disabled={disabled || memuat !== null}
          onClick={() => void pilih(sample)}
          title={sample.name}
          className={cn(
            "group relative aspect-square overflow-hidden rounded-md border border-line",
            "bg-surface transition-colors hover:border-brand disabled:opacity-50",
          )}
        >
          {/* Gambar contoh dilayani oleh kontainer api dan hanya perlu
              ditampilkan apa adanya, sehingga pengoptimalan next/image tidak
              memberi apa-apa selain satu lapisan tambahan. */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={sample.url}
            alt={sample.name}
            className="h-full w-full object-cover"
            loading="lazy"
          />
          {memuat === sample.name ? (
            <span className="absolute inset-0 flex items-center justify-center bg-ink/45">
              <Loader2 className="size-4 animate-spin text-white" aria-hidden />
            </span>
          ) : null}
        </button>
      ))}
    </div>
  );
}
