import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  base: "./",
  server: { fs: { allow: [".."] } },
  worker: { format: "es" },
  // fonts are inlined (src/fonts.ts) so every build is self-contained; other assets keep Vite's default limit
  build: { assetsInlineLimit: (file: string) => (file.endsWith(".woff2") ? true : undefined) },
  test: { testTimeout: 60000 },
});
