# Glossary

## Template terms

| Term | Meaning |
| --- | --- |
| ADR | Architecture Decision Record — a short document capturing a significant, hard-to-reverse technical decision and its rationale. |
| C4 Model | A layered way to diagram software architecture: Context, Container, Component, Code. |
| Module | One of this repo's top-level directories (`base/`, `backend/`, `gui/`, `frontend/`). |
| Recipe | A named command defined in the root `justfile`. |

## ASP domain terms

| Term | Meaning |
| --- | --- |
| ARAP | As-Rigid-As-Possible — a mesh-deformation regularizer (Sýkora 2009) used to warp foreground poses toward agreement without distorting anatomy; see `alignment`/`compositing` Stage 8.5. |
| Bundle adjustment / GNC-TLS | Joint optimization of frame offsets from pairwise matches; GNC-TLS (Yang 2020) is the robust (outlier-tolerant) variant used here. |
| Coherence (structural) | Whether an output image "parses as one picture of one character" — no duplicated/misordered anatomy. The property no automated metric in this project currently measures directly; see `docs/reports/ASP_Critical_Evaluation_2026-07-08.md` §6.1. |
| dy_cv gate | A quality gate that falls back to SCANS when the vertical-scroll step coefficient of variation indicates irregular (non-uniform) scrolling — ASP's assumptions hold poorly on such sequences. |
| GT-SSIM / aligned-SSIM | Structural similarity vs. a hand-made ground-truth panorama, computed after ECC alignment; "aligned" restricts the comparison to the actually-overlapping content region. |
| Guarded fallback | A test where a quality gate rejected the ASP composite and SCANS (below) was substituted instead — a safety win, not a quality win. |
| Hold detection | Identifying near-duplicate consecutive frames ("holds," common in on-twos/on-threes anime) via perceptual hashing + robust statistics, used both for frame selection and (in Overmix's design) animation-phase grouping. |
| HybridStitch (`HybridStitchPanel`) | The GUI's manual/interactive stitching tool (control points, color correction, seam painting, mesh warp) — architecturally independent of the automated `AnimeStitchPipeline`. |
| Phase-consistent compositing | Compositing strategy that never blends frames from two different animation phases into one seam — see roadmap Phase 2.2–2.3. |
| Single-pose escalation | Falling back to a single frame's pose for a region instead of averaging/warping two conflicting poses — the project's "master principle" (never average two conflicting poses). |
| SCANS | The OpenCV `cv2.Stitcher`-based simple stitch used throughout as the baseline ASP must match or exceed. |

> Add further terms here as they enter common use in the roadmap/reports — this list is not exhaustive.
