"use client";

import { Camera, CircleAlert, Images, Loader2, ScanLine, Upload, X } from "lucide-react";
import { useState } from "react";

import { CameraCapture } from "@/components/camera-capture";
import { SamplePicker } from "@/components/sample-picker";
import { UploadDropzone } from "@/components/upload-dropzone";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { formatBytes } from "@/lib/image-input";
import { cn } from "@/lib/utils";

export type SourceTab = "unggah" | "kamera" | "contoh";

export interface ChosenImage {
  file: File;
  previewUrl: string;
}

interface SourcePanelProps {
  chosen: ChosenImage | null;
  rejection: string | null;
  busy: boolean;
  onSelect: (file: File) => void;
  onClear: () => void;
  onInspect: () => void;
}

const tabs: { id: SourceTab; label: string; icon: typeof Upload }[] = [
  { id: "unggah", label: "Unggah", icon: Upload },
  { id: "kamera", label: "Kamera", icon: Camera },
  { id: "contoh", label: "Contoh", icon: Images },
];

/**
 * Panel sumber citra.
 *
 * Ketiga sumber bermuara pada satu berkas dan satu tombol Periksa. Kamera tidak
 * mendapat jalur pemeriksaan tersendiri: bingkai yang diambil menjadi berkas
 * JPEG biasa dan melewati pemeriksaan sisi klien yang sama dengan unggahan,
 * sehingga hanya ada satu alur yang perlu dipercaya.
 */
export function SourcePanel({
  chosen,
  rejection,
  busy,
  onSelect,
  onClear,
  onInspect,
}: SourcePanelProps) {
  const [tab, setTab] = useState<SourceTab>("unggah");

  return (
    <Card className="flex flex-col">
      <CardHeader
        title="Citra produk"
        description="Satu citra untuk satu pemeriksaan"
        icon={<ScanLine className="size-4" aria-hidden />}
      />

      <CardBody className="flex flex-1 flex-col gap-4">
        <div role="tablist" aria-label="Sumber citra" className="flex gap-1 rounded-lg bg-surface-sunken p-1">
          {tabs.map(({ id, label, icon: Ikon }) => (
            <button
              key={id}
              role="tab"
              type="button"
              id={`tab-${id}`}
              aria-selected={tab === id}
              aria-controls={`panel-${id}`}
              onClick={() => setTab(id)}
              className={cn(
                "flex flex-1 items-center justify-center gap-2 rounded-md px-3 py-2",
                "text-sm font-medium transition-colors",
                tab === id
                  ? "bg-surface text-ink shadow-[0_1px_2px_rgba(15,21,31,0.08)]"
                  : "text-ink-faint hover:text-ink",
              )}
            >
              <Ikon className="size-4" aria-hidden />
              {label}
            </button>
          ))}
        </div>

        {/* Ketiga panel tetap terpasang dan hanya disembunyikan lewat CSS. Kalau
            panel yang tidak aktif dilepas, aliran kamera ikut mati setiap kali
            operator berpindah tab, dan menyalakannya kembali menuntut izin
            peramban lagi. */}
        {tabs.map(({ id }) => (
          <div
            key={id}
            role="tabpanel"
            id={`panel-${id}`}
            aria-labelledby={`tab-${id}`}
            hidden={tab !== id}
          >
            {id === "unggah" ? <UploadDropzone onSelect={onSelect} disabled={busy} /> : null}
            {id === "kamera" ? <CameraCapture onCapture={onSelect} disabled={busy} /> : null}
            {id === "contoh" ? <SamplePicker onSelect={onSelect} disabled={busy} /> : null}
          </div>
        ))}

        {rejection ? (
          <div
            role="alert"
            className="flex items-start gap-2.5 rounded-lg border border-reject-line bg-reject-soft px-3.5 py-2.5"
          >
            <CircleAlert className="mt-0.5 size-4 shrink-0 text-reject" aria-hidden />
            <p className="text-xs leading-relaxed text-ink-soft">{rejection}</p>
          </div>
        ) : null}

        {chosen ? (
          <div className="flex items-center gap-3 rounded-lg border border-line bg-surface-sunken p-2.5">
            {/* Pratinjau berasal dari object URL berkas lokal; tidak ada yang
                dapat dioptimalkan next/image di sini. */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={chosen.previewUrl}
              alt="Pratinjau citra yang dipilih"
              className="size-12 shrink-0 rounded-md border border-line object-cover"
            />
            <div className="min-w-0 flex-1">
              <p className="truncate text-xs font-medium text-ink">{chosen.file.name}</p>
              <p className="tabular text-[11px] text-ink-faint">
                {formatBytes(chosen.file.size)}
              </p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClear}
              disabled={busy}
              aria-label="Bersihkan pilihan"
            >
              <X aria-hidden />
            </Button>
          </div>
        ) : null}

        <div className="mt-auto pt-1">
          <Button size="lg" className="w-full" onClick={onInspect} disabled={!chosen || busy}>
            {busy ? <Loader2 className="animate-spin" aria-hidden /> : <ScanLine aria-hidden />}
            {busy ? "Memeriksa" : "Periksa"}
          </Button>
        </div>
      </CardBody>
    </Card>
  );
}
