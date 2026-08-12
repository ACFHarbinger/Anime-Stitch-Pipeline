import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const base = process.env.SITE_BASE || "/";

export default defineConfig({
  base,
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1024,
  },
});
