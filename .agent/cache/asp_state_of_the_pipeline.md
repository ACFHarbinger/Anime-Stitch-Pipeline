# ASP — State of the Pipeline (Post-Trim, 2026-07-08)

*This document describes the Anime Stitch Pipeline as it exists after the S200 "great
trim" (commits `43aa60c`…, 2026-07-08). It replaces the pre-trim state document now
archived at `archive/agent/cache/asp_state_of_the_pipeline.md`. Read that archive and
`research/ASP_Critical_Evaluation_2026-07-08.md` for the full history of what was
removed and why.*

---

## 1. What the ASP Is Now

A single, small, measurable pipeline: the S160-benchmarked default path with
everything unverified removed. Numbers:

| Surface | Pre-trim | Post-trim |
|---|---|---|
| `backend/src/animation/` | 30,640 lines / 49 files | ~12,700 lines / 27 files |
| `compositing.py` | 6,939 lines / 145 functions | 2,184 lines / ~60 functions |
| `pipeline.py` | 6,536 lines | 2,227 lines |
| `canvas.py` | 2,209 lines | 419 lines |
| `ASP_*` env flags | 387 | 43 |
| Config schema (`config.py`) | ~490 keys | 43 keys |
| Quality gates | ~150 (most unverified) | 8 (see §3) |
| `bench_anime_stitch.py` | 5,603 lines / ~40 metrics | ~3,400 lines / 12 metrics |
| C++ `base/src/animation/` | 5,753 lines / 11 files | 4,189 lines / 9 files |
| Animation test suite | 2,230 collected | 684 collected (632 pass, 0 fail) |

Removed subsystems (code deleted; concepts documented in the archive):
MFSR (DCT/PSO/DRL/diffusion SR), RLHF (reward model, trainer, feedback tab),
ToonCrafter seam synthesis, SRStitcher diffusion fill, Real-ESRGAN post-SR,
ProPainter background completion, AnimeInterp/SGM + CamFlow flow engines,
GroundingDINO NL masking, MLLM scoring, HITL presets, hybrid 4K export,
wave correction, trajectory smoothing, the ~40 default-ON §5.x pipeline CV gates,
the ~40 §5.x bench comparative gates, the 24 pre-DP seam escalation gates, the
zone normalization chain (chroma/lum/sat/contrast/hue), Poisson/MultiBand/canvas-DP
blend alternates, and ~90 default-OFF experiment flags.

## 2. The 13-Stage Core Path

| Stage | What runs (all default-ON, no hidden flags) | File |
|---|---|---|
| 0 | Smart frame selection: greedy 50px displacement + hold detection (MAD + dHash) + blur/contrast rejection | `ingestion/frame_selection.py` |
| 1–2 | Load, sort by numeric suffix, width-normalise (Lanczos-4) | `alignment/canvas.py` |
| 3 | BaSiC flat-field (when `use_basic`) | `rendering/photometric.py` |
| 4 | BiRefNet fg/bg masks (SAM-2 via `ASP_USE_SAM2=1`) | `ingestion/masking.py` |
| 4.5 | Background scalar photometric normalisation (adaptive clamp §1.4B) | `core/pipeline.py` |
| 5–6 | Matching cascade: EfficientLoFTR → kornia LoFTR → ALIKED+LightGlue → template match → phase correlation → RoMa; static-edge rejection (§1.2A/C), spread + bg-ratio filters | `alignment/matching.py` |
| 6.5 | Spatial dedup (<25px), §1.15 connectivity gate, §2.9C high-conf retry set | `core/pipeline.py` |
| 7 | GNC-TLS bundle adjustment (Geman-McClure outer loop, Cauchy LM inner, spanning-tree inlier pre-filter) | `alignment/bundle_adjust.py` |
| 7b | Affine validation (ratio / adaptive min_gap / rotation / scale) + retry chain → PANORAMA → SCANS fallback | `core/validation.py` |
| 7c | **dy_cv gate (§4.7)**: step-CV ≥ 1.5 (adaptive floor 0.8 for N≥8) → SCANS. The catastrophic-failure guard (test77-class) | `core/pipeline.py` |
| 8 | SEA-RAFT flow refine (when available) / ECC sub-pixel refine | `flow/flow_refine.py`, `alignment/ecc.py` |
| 9 | Canvas construction + midplane shift; horizontal-scroll → SCANS | `alignment/canvas.py` |
| 10 | A5 foreground-excluded temporal median (C++ `render_median`, GPU optional) | `rendering/rendering.py` |
| 10.5 | Multi-frame coverage gate (≥30% rows with ≥2 frames) | `core/pipeline.py` |
| 8.5/11 | Foreground composite (see §3) | `rendering/compositing.py` |
| 12.5–13 | Scroll-axis content trim, TELEA border fill, morphological crop, save | `core/pipeline.py`, `alignment/canvas.py` |

### Stage 11 composite, in order
1. Warp frames + masks to canvas (`_warp_inputs`).
2. Bg-scalar normalisation per frame (`_normalize_warped_frames`, coherence skip-mask).
3. Boundary optimisation ±SEARCH_RANGE with bg-weighted similarity (C++ fast path) +
   feather table + §1.6B gain-driven feather floor.
4. **Stage 8.5 fg registration** per seam: RAFT/DIS flow → ARAP Push+Regularise (C++
   `arap_push_regularise`; LSD collinearity embedded) → symmetric midpoint warp →
   A6 single-pose escalation at fixed `post_warp_diff > 22` lum (or residual > 90px).
   User overrides (`seam_overrides`: force single pose, waypoints) honoured.
5. Post-registration feather adaptation (±50% by post_diff).
6. §4.10 sequential global gain equalization across warped frames.
7. Pairwise DP seams (C++ `seam_cut`, tiered cost map §1.6A/§3.15A, DSFN
   per-pixel soft weight, Laplacian blend with §4.1/§4.4 per-zone blocks
   gain/lum compensation, single-pose soft edge + S16 colour match).
   §4.2 GraphCut global seam (`base.seam.graphcut_seam_find`, ≤0.4 MPix seam
   estimation + §3.33 GC feather) exists behind `ASP_GRAPHCUT_SEAM=1` but is
   **default OFF**: its first measurement (2026-07-09) scored seam_visibility
   20–80 vs the DP path's 2–16.
8. Black-pixel fill, §1.106 seam lum-step audit (log + `seam_meta_out` only).

## 3. The Gate Set (complete list — keep it this small)

| Gate | Where | Fires |
|---|---|---|
| §1.15 edge-graph connectivity | pre-BA | disconnected graph → SCANS |
| Affine validation (ratio/min_gap/rot/scale) + retry chain | Stage 7b | invalid solve → retries → PANORAMA → SCANS |
| §4.7 dy_cv (1.5, adaptive) | post-BA | irregular scroll → SCANS |
| Horizontal scroll axis | Stage 9.5 | horizontal → SCANS |
| Stage 10.5 multi-frame coverage (0.30) | post-render | sparse coverage → SCANS |
| A6 single-pose escalation (22 lum / 90 px) | per seam | pose gap → one coherent pose |
| Bench CompositeGate (SC>max(38,2×scans), SB>max(35,2×scans)) | benchmark | banded render → SCANS output |
| Bench GhostGate (siqe > max(40,2×sim)) + SeamVisGate (sv > max(35,3×sim)) | benchmark | ghosted/hard-cut composite → SCANS output |

Rule going forward (from the critical evaluation): **a new gate must displace an old
one, and nothing ships default-ON without a full-corpus benchmark.**

## 4. Benchmark & Metrics

Run: `just asp-benchmark` (full 97) · `just asp-benchmark-verify` (test04/08/09/27/57)
· data dir `dump/` · results `backend/benchmark/output/anime_stitch_*.json` · report
`dump/output/benchmark_report.md`.

Metric set (12, all validated; the ~28 removed metrics are enumerated in the archive):
`sharpness`, `coverage`, `seam_gradient`, `color_entropy`, `edge_energy_score`
(double-Sobel — **sharpness proxy, not ghosting**), `ghosting_siqe` (FFT autocorr —
the true ghosting metric), `seam_coherence`, `seam_visibility`, per-seam ghost
scores/max, `cqas` (weighted no-GT aggregate), plus SSIM/PSNR vs simple stitch and
raw + ECC-aligned SSIM vs the 55 ground truths.

Verdict logic unchanged in structure (GT-SSIM based when GT exists, cv-metrics
`_auto_verdict` otherwise) but now scores ghosting via `ghosting_siqe`.
**Caveat carried over from the evaluation: no automated verdict here measures
structural coherence; side-by-side visual review remains mandatory before believing
any `asp_better`.**

## 5. Benchmark Results — Trimmed Pipeline

*(Full analysis: `asp_benchmark_2026-07.md` in this directory.)*

| Run | asp_better | comparable | simple_better | GT-SSIM ASP/simple | aligned |
|---|---|---|---|---|---|
| 2026-06-01 (S0–5, 96t) | 8/55 GT | 24 | 23 | 0.669 / 0.695 | — |
| 2026-06-21 (S142, 97t) | 9 | 41 | 46 | 0.659 / 0.699 | — |
| 2026-06-23 (S160, 97t) | 10 | 41 | 45 | 0.653 / 0.693 | 0.680 / 0.720 |
| **2026-07-09 (post-trim)** | **27** | **41** | **29** | **0.665 / 0.696** | **0.693 / 0.718** |

Post-trim run: 87 s/test (−32%), 51 true composites, 46 guarded fallbacks
(SeamVisGate 24, render gate 21, alignment 1). The verdict jump is largely
honest fallback behavior (SCANS-on-preprocessed-frames beats raw cv2.Stitcher
on many irregular tests); the composite-quality gap vs simple stitch narrowed
(aligned −0.040 → −0.025) but its root cause — pose gaps at frame selection —
is architectural and unchanged. Also measured this run: §4.2 GraphCut seams
score seam_visibility 20–80 vs the DP path's 2–16 → defaulted OFF.

## 6. What To Do Next (unchanged from the critical evaluation §9)

1. **Human visual coherence rating** as the primary metric; calibrate automated
   metrics against it; hard coherence gate in verdicts.
2. **One change → one benchmark → keep or revert.** The 5-test subset per change,
   full 97 per feature.
3. The highest-value unexplored direction is **animation-phase grouping at ingestion**
   (Overmix `AnimationSeparator` idea) followed by per-pixel phase-consistent
   reconstruction — see evaluation §9.2 for the coherence-first architecture sketch.
4. Candidate A/B experiments already implemented and flag-gated OFF, in priority
   order: `ASP_HOLD_AVERAGE=1` (Overmix sub-pixel hold averaging), `ASP_USE_SAM2=1`,
   `ASP_POSE_WINDOW_PX=80` (DINOv2 pose-consistent selection — needs the
   GT-coupling measurement fix first).
