import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// Deployed alongside the MkDocs portal under gh-pages/app/ (see
// .github/workflows/docs.yml), i.e. https://<owner>.github.io/Image-Toolkit/app/
// — override via SITE_BASE for other deploy targets (PR previews, forks).
const base = process.env.SITE_BASE || "/";

export default defineConfig({
  base,
  plugins: [vue()],
  server: {
    fs: {
      // allow importing markdown from ../ (docs/) and ../../ (repo root docs
      // reference, e.g. docs/moon/) outside the website/ project root
      allow: [".."],
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
