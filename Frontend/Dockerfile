# Image antarmuka VisionQC.
#
# Konteks build adalah folder Frontend. Berbeda dengan layanan api, antarmuka
# tidak mengimpor apa pun dari AI_model: tipe API dibangkitkan lebih dulu dari
# /openapi.json dan sudah tersimpan di src/types/api.d.ts, sehingga build ini
# tidak menuntut layanan menyala.

FROM node:22-alpine AS deps
WORKDIR /app
RUN npm install --global pnpm@11.10.0

# pnpm-workspace.yaml ikut disalin karena di sanalah paket yang boleh
# menjalankan skrip build didaftarkan. Tanpanya `pnpm install` keluar dengan
# kode 1 dan build berhenti sebelum satu berkas pun dikompilasi.
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

FROM node:22-alpine AS builder
WORKDIR /app
RUN npm install --global pnpm@11.10.0
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Keluaran mandiri hanya dinyalakan di sini; lihat alasannya di next.config.ts.
ENV VISIONQC_STANDALONE=1
ENV NEXT_TELEMETRY_DISABLED=1
RUN pnpm build

FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

# Alamat layanan dibaca ketika server dinyalakan, bukan ketika image dibangun,
# sehingga image yang sama dapat diarahkan ke host lain tanpa dibangun ulang.
ENV VISIONQC_API_ORIGIN=http://api:8000

RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 nextjs

COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Tidak ada `COPY public`: aplikasi ini tidak punya aset statis di luar berkas
# yang sudah dibundel, dan folder `public/` yang kosong tidak dilacak git -
# sehingga penyalinannya justru menggagalkan build dari kloning bersih.

USER nextjs
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD node -e "fetch('http://127.0.0.1:3000/').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server.js"]
