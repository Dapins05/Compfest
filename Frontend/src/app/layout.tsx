import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "VisionQC - Inspeksi Mutu Kemasan",
  description:
    "Inspeksi kualitas kemasan pangan dan minuman berbasis computer vision. " +
    "Satu citra produk diperiksa, hasilnya PASS atau REJECT beserta alasannya.",
};

export const viewport: Viewport = {
  themeColor: "#f4f6f9",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="id">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body>
    </html>
  );
}
