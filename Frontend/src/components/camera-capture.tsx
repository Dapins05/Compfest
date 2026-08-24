"use client";

/**
 * Pengambilan citra lewat kamera.
 *
 * Kamera dipakai sebagai sumber citra, bukan sebagai jalur inferensi
 * berkelanjutan: aliran video hanya ditampilkan, dan model baru berjalan ketika
 * operator menekan tombol ambil lalu menekan Periksa. Batas ruang lingkup
 * penyisihan menuntut antarmuka menerima MASUKAN TUNGGAL dan melarang gelung
 * otomatis pada sisi model, sehingga memeriksa setiap bingkai video termasuk
 * yang tidak boleh dibangun pada tahap ini.
 *
 * Ada dua jalur, dan yang menentukan bukan jenis perangkatnya melainkan apakah
 * halaman dibuka pada konteks aman:
 *
 *   viewfinder - `getUserMedia` tersedia, jendela bidik tampil di dalam halaman.
 *                Berlaku pada localhost dan pada HTTPS.
 *   native     - `getUserMedia` tidak tersedia, misalnya ketika antarmuka
 *                dibuka dari ponsel lewat http://alamat-ip:3000. Pengambilan
 *                dialihkan ke aplikasi kamera bawaan perangkat, yang tidak
 *                menuntut konteks aman dan tetap mengembalikan satu berkas
 *                gambar. Inilah yang membuat pengujian dari ponsel di jaringan
 *                pabrik tetap dapat dilakukan tanpa menyiapkan sertifikat.
 */

import { Camera, CameraOff, RefreshCw, ShieldAlert, Smartphone } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type CameraState = "mati" | "meminta" | "hidup" | "galat";
type Capability = "memeriksa" | "viewfinder" | "native";

interface CameraCaptureProps {
  onCapture: (file: File) => void;
  disabled?: boolean;
}

/** Ubah galat getUserMedia menjadi kalimat yang menyebut jalan keluarnya. */
function jelaskanGalat(error: unknown): string {
  const nama = error instanceof DOMException ? error.name : "";
  switch (nama) {
    case "NotAllowedError":
    case "SecurityError":
      return (
        "Izin kamera ditolak peramban. Buka setelan situs pada peramban, " +
        "izinkan kamera, lalu coba lagi."
      );
    case "NotFoundError":
    case "OverconstrainedError":
      return "Tidak ada kamera yang terdeteksi pada perangkat ini.";
    case "NotReadableError":
      return "Kamera sedang dipakai aplikasi lain. Tutup aplikasi itu lalu coba lagi.";
    default:
      return "Kamera tidak dapat dinyalakan pada perangkat ini.";
  }
}

export function CameraCapture({ onCapture, disabled = false }: CameraCaptureProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const nativeInputRef = useRef<HTMLInputElement>(null);

  const [capability, setCapability] = useState<Capability>("memeriksa");
  const [state, setState] = useState<CameraState>("mati");
  const [message, setMessage] = useState<string>("");
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string>("");

  // Diperiksa setelah komponen terpasang, bukan saat render, supaya penyajian
  // di server dan di peramban menghasilkan markah yang sama.
  useEffect(() => {
    // Diperiksa lewat `typeof`, bukan lewat nilai kebenarannya: pada tipe DOM
    // `getUserMedia` selalu terdefinisi, sehingga pemeriksaan biasa dianggap
    // TypeScript sebagai syarat yang tidak pernah salah. Yang sesungguhnya
    // hilang pada konteks tidak aman adalah `mediaDevices` itu sendiri.
    setCapability(
      typeof navigator !== "undefined" &&
        typeof navigator.mediaDevices?.getUserMedia === "function"
        ? "viewfinder"
        : "native",
    );
  }, []);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setState("mati");
  }, []);

  const start = useCallback(async (pilihan?: string) => {
    setState("meminta");
    setMessage("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: pilihan
          ? { deviceId: { exact: pilihan } }
          : { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 1280 } },
        audio: false,
      });

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => undefined);
      }
      setState("hidup");

      // Label perangkat baru terisi setelah izin diberikan, jadi daftarnya
      // sengaja diambil sesudah aliran menyala, bukan sebelumnya.
      const semua = await navigator.mediaDevices.enumerateDevices();
      setDevices(semua.filter((item) => item.kind === "videoinput"));
      const aktif = stream.getVideoTracks()[0]?.getSettings().deviceId ?? "";
      setDeviceId(pilihan ?? aktif);
    } catch (error) {
      setState("galat");
      setMessage(jelaskanGalat(error));
    }
  }, []);

  useEffect(() => stop, [stop]);

  const capture = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const stempel = new Date().toISOString().replace(/[:.]/g, "-");
        onCapture(new File([blob], "kamera-" + stempel + ".jpg", { type: "image/jpeg" }));
      },
      "image/jpeg",
      0.92,
    );
  }, [onCapture]);

  if (capability === "native") {
    return (
      <div className="space-y-3">
        <div className="flex aspect-4/3 flex-col items-center justify-center gap-3 rounded-lg border border-line bg-surface-sunken px-6 text-center">
          <span className="flex size-11 items-center justify-center rounded-full bg-surface text-ink-faint">
            <Smartphone className="size-5" aria-hidden />
          </span>
          <p className="text-sm font-medium text-ink">Pakai kamera perangkat</p>
          <p className="max-w-xs text-xs leading-relaxed text-ink-faint">
            Peramban hanya menampilkan jendela bidik di dalam halaman pada koneksi aman.
            Halaman ini dibuka lewat koneksi biasa, jadi pengambilan dialihkan ke aplikasi
            kamera bawaan perangkat. Hasilnya diperiksa lewat jalur yang sama persis.
          </p>
          <Button
            size="sm"
            variant="outline"
            disabled={disabled}
            onClick={() => nativeInputRef.current?.click()}
          >
            <Camera aria-hidden />
            Buka kamera
          </Button>
        </div>
        <input
          ref={nativeInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          className="sr-only"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onCapture(file);
            event.target.value = "";
          }}
        />
      </div>
    );
  }

  const sudut = [
    "left-0 top-0 border-l-2 border-t-2 rounded-tl-md",
    "right-0 top-0 border-r-2 border-t-2 rounded-tr-md",
    "left-0 bottom-0 border-b-2 border-l-2 rounded-bl-md",
    "right-0 bottom-0 border-b-2 border-r-2 rounded-br-md",
  ];

  return (
    <div className="space-y-3">
      <div className="relative aspect-4/3 overflow-hidden rounded-lg border border-line bg-ink">
        <video
          ref={videoRef}
          playsInline
          muted
          className="h-full w-full object-cover"
          aria-label="Pratinjau kamera"
        />

        {state === "hidup" ? (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <div className="relative aspect-square h-[78%]">
              {sudut.map((posisi) => (
                <span key={posisi} className={cn("absolute size-8 border-white/70", posisi)} />
              ))}
            </div>
          </div>
        ) : null}

        {state !== "hidup" ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6 text-center">
            {state === "galat" ? (
              <ShieldAlert className="size-7 text-reject-line" aria-hidden />
            ) : (
              <CameraOff className="size-7 text-white/45" aria-hidden />
            )}
            <p className="max-w-xs text-sm leading-relaxed text-white/75">
              {state === "galat"
                ? message
                : state === "meminta"
                  ? "Menunggu izin kamera dari peramban."
                  : capability === "memeriksa"
                    ? "Memeriksa dukungan kamera."
                    : "Kamera belum dinyalakan."}
            </p>
            {state !== "meminta" && capability === "viewfinder" ? (
              <Button size="sm" variant="outline" onClick={() => void start()}>
                <Camera aria-hidden />
                {state === "galat" ? "Coba lagi" : "Nyalakan kamera"}
              </Button>
            ) : null}
          </div>
        ) : null}
      </div>

      {state === "hidup" ? (
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={capture} disabled={disabled} className="flex-1">
            <Camera aria-hidden />
            Ambil gambar
          </Button>
          <Button variant="outline" onClick={stop} aria-label="Matikan kamera">
            <CameraOff aria-hidden />
          </Button>
          {devices.length > 1 ? (
            <label className="flex w-full items-center gap-2 text-xs text-ink-faint">
              <RefreshCw className="size-3.5" aria-hidden />
              <span className="shrink-0">Perangkat</span>
              <select
                value={deviceId}
                onChange={(event) => void start(event.target.value)}
                className="min-w-0 flex-1 rounded-md border border-line bg-surface px-2 py-1.5 text-xs text-ink"
              >
                {devices.map((device, index) => (
                  <option key={device.deviceId} value={device.deviceId}>
                    {device.label || "Kamera " + String(index + 1)}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
