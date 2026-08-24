import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  test: {
    // Yang diuji di sini adalah fungsi murni: aturan penolakan berkas,
    // pemetaan putusan, dan pemformatan angka. Ketiganya tidak menyentuh DOM,
    // jadi lingkungan node sudah memadai dan tidak perlu menarik jsdom.
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
