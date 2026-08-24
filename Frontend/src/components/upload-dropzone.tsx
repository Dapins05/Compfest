"use client";

import { ImageUp } from "lucide-react";
import { useId, useRef, useState, type DragEvent } from "react";

import {
  ACCEPTED_EXTENSIONS,
  MAX_FILE_SIZE_MB,
  MIN_IMAGE_SIDE_PX,
} from "@/lib/image-input";
import { cn } from "@/lib/utils";

interface UploadDropzoneProps {
  onSelect: (file: File) => void;
  disabled?: boolean;
}

/**
 * Pemilih berkas dengan dukungan seret dan lepas.
 *
 * Berkas yang terpilih diteruskan apa adanya ke pemanggil; pemeriksaan tipe,
 * ukuran, dan dimensi dilakukan di satu tempat pada halaman supaya pesan
 * penolakannya sama untuk ketiga sumber citra.
 */
export function UploadDropzone({ onSelect, disabled = false }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const inputId = useId();
  const [dragging, setDragging] = useState(false);

  const terima = (files: FileList | null) => {
    const file = files?.[0];
    if (file) onSelect(file);
  };

  const onDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    setDragging(false);
    if (!disabled) terima(event.dataTransfer.files);
  };

  return (
    <label
      htmlFor={inputId}
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={cn(
        "flex aspect-4/3 cursor-pointer flex-col items-center justify-center gap-3",
        "rounded-lg border-2 border-dashed px-6 text-center transition-colors",
        dragging
          ? "border-brand bg-brand-soft"
          : "border-line-strong bg-surface-sunken hover:border-brand hover:bg-brand-soft/40",
        disabled && "pointer-events-none opacity-50",
      )}
    >
      <span
        className={cn(
          "flex size-11 items-center justify-center rounded-full",
          dragging ? "bg-brand text-white" : "bg-surface text-ink-faint",
        )}
      >
        <ImageUp className="size-5" aria-hidden />
      </span>
      <span className="text-sm font-medium text-ink">
        Seret foto ke sini, atau klik untuk memilih
      </span>
      <span className="text-xs leading-relaxed text-ink-faint">
        JPG, PNG, atau WEBP &middot; maksimum {MAX_FILE_SIZE_MB} MB &middot; sisi terpendek
        minimal {MIN_IMAGE_SIDE_PX} piksel
      </span>
      <input
        ref={inputRef}
        id={inputId}
        type="file"
        accept={ACCEPTED_EXTENSIONS}
        disabled={disabled}
        className="sr-only"
        onChange={(event) => {
          terima(event.target.files);
          // Dikosongkan supaya memilih berkas yang sama dua kali berturut-turut
          // tetap memicu perubahan.
          event.target.value = "";
        }}
      />
    </label>
  );
}
