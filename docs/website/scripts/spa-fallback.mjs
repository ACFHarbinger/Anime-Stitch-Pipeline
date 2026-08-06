#!/usr/bin/env node
// GitHub Pages has no server-side rewrite rule, so a hard refresh (or a
// deep link) on any client-side route 404s. The standard workaround: a
// 404.html identical to index.html — GH Pages serves it for unknown paths,
// and Vue Router (createWebHistory) then takes over client-side.
import { copyFileSync, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const dir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "dist");
const src = path.join(dir, "index.html");
if (existsSync(src)) {
  copyFileSync(src, path.join(dir, "404.html"));
  console.log("[spa-fallback] wrote dist/404.html");
}
