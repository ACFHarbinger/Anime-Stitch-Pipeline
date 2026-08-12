import { Camera, GitBranch, ScanLine } from "lucide-react";

export const site = {
  shortName: "ASP",
  name: "Anime Stitch Pipeline",
  eyebrow: "PANORAMA STITCHING / VISUAL STRUCTURE",
  badge: "STITCH LAB / v2.0",
  heroTitle: ["Trace the", "unseen."],
  heroDescription: "A high-performance panorama engine for anime panning shots, built around constrained motion, robust estimation, seam-aware compositing, and honest evaluation.",
  accent: "cyan",
  repository: "https://github.com/ACFHarbinger/Anime-Stitch-Pipeline",
  modules: [
    { number: "01", title: "Capture & Index", text: "Keep source frames, references, and provenance attached from ingestion onward.", detail: "Build a searchable sequence workspace before alignment begins.", action: "Read the architecture", href: "https://github.com/ACFHarbinger/Anime-Stitch-Pipeline/blob/main/docs/ARCHITECTURE.md", icon: Camera },
    { number: "02", title: "Align Structure", text: "Estimate constrained 2D motion and reject unstable feature matches.", detail: "GNC-TLS residual weighting and bundle adjustment make the signal inspectable.", action: "Inspect the pipeline", href: "#pipeline", icon: ScanLine },
    { number: "03", title: "Compose Evidence", text: "Route seams around protected cel regions and preserve review artifacts.", detail: "Render panoramas with seam diagnostics instead of hiding quality behind one score.", action: "Read the benchmarks", href: "https://github.com/ACFHarbinger/Anime-Stitch-Pipeline/blob/main/docs/BENCHMARKS.md", icon: GitBranch },
  ],
  stages: ["INGEST", "MATCH", "GNC-TLS", "BUNDLE", "SEAM", "RENDER"],
};
