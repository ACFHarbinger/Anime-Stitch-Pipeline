# Anime Stitch Pipeline — Context Prompt

**Intent:** Initialize a high-context session for working on `AnimeStitchPipeline`. Load all architectural knowledge, known bugs, and test infrastructure before making any changes.

---

## The Prompt

You are an expert computer vision engineer working on the `AnimeStitchPipeline` in Image-Toolkit. This is a 13-stage research pipeline that stitches sequential anime/manga frame captures into vertical panoramas.

### Architecture in one sentence

Frames → BaSiC photometric correction → BiRefNet foreground masking → LoFTR/TemplateMatch pairwise matching → bundle adjustment → ECC sub-pixel refinement → canvas geometry → temporal median render → (optional MFSR) → hard-partition foreground composite → crop.

### Key source files

| File | Stage | What it does |
|------|-------|-------------|
| `backend/src/animation/pipeline.py` | Orchestrator | Full 13-stage flow; also `_filter_edges`, coverage gate |
| `backend/src/animation/compositing.py` | Stage 11 | Hard-partition composite — primary source of seam improvements |
| `backend/src/animation/rendering.py` | Stage 9 | Temporal median render with per-pixel gain |
| `backend/src/animation/bundle_adjust.py` | Stage 7 | Global bundle adjustment (LM, GNC cauchy loss) |
| `backend/src/animation/matching.py` | Stages 5–6 | Pairwise LoFTR + TemplateMatch |
| `backend/src/animation/canvas.py` | Stage 8 | Canvas geometry, `_compute_canvas`, `_crop_to_valid`, TELEA fill |
| `backend/src/animation/masking.py` | Stage 4 | BiRefNet foreground masks |
| `backend/src/animation/ecc.py` | Stage 8 | ECC sub-pixel refinement |
| `backend/src/animation/fg_register.py` | Stage 8.5 | Flow-guided foreground pose registration (ARAP-Push) |
| `backend/src/animation/frame_selection.py` | Pre-pipeline | Smart frame selection — hold detection, DINOv2, near-dup filter |
| `backend/src/animation/config.py` | Config | §1.8A TOML config loader (`load_asp_config`) |
| `backend/src/animation/validation.py` | Post-7 | Affine health check; min_gap threshold = 25px |
| `backend/src/constants/animation.py` | Constants | Centralised pipeline constants (FEATHER_MAX, FEATHER_MIN, etc.) |
| `backend/src/core/image_merger.py` | Reference | Simple stitch (`_merge_images_scan_stitch`) — the quality target |

### Full architecture reference

Read `docs/ARCHITECTURE.md` — it has complete Mermaid diagrams for both the simple stitch and the 13-stage pipeline with all parameters and branch conditions.

### Research context

Read the consolidated research reference before proposing algorithmic changes:
- `research/Image_Stitching_Research.md` — the complete, single-source stitching reference (merges all 14 prior reports). Covers geometric foundations, Perfect-vs-Scan-Stitch audit, feature matching, optical flow, spatially-varying warps, the foreground-assembly paradigm (§8), photometric correction, segmentation, seam-finding, blending, background reconstruction, the 14-stage pipeline spec, evaluation, and failure/fallback taxonomy.
- For generation work: `research/Image_Generation_Research.md`.

### Overmix reference implementation

`Overmix/src/` contains a production C++ image stitching tool. Key files to consult:
- `aligners/RecursiveAligner.cpp` — hierarchical alignment (robust to bad pairwise matches)
- `aligners/AverageAligner.cpp` — average-frame-based alignment
- `comparators/MultiScaleComparator.cpp` — multi-scale matching comparator
- `renders/AverageRender.cpp` — weighted average rendering
- `renders/StatisticsRender.cpp` — median/statistics-based rendering

---

## Current State (2026-06-15, Session 123)

The pipeline has received 123 sessions of improvements (S1–S123). 777 unit tests in `backend/test/animation/` (2 skipped: pyav); 18 GUI tests. SCANS fallbacks reduced from 51/96 → 4/96 genuine fallbacks. Key improvements shipped across sessions:

**Frame selection & pre-processing (S6–S110):**
- Hold detection (`_detect_hold_blocks`, `_detect_hold_blocks_dhash`), DINOv2 frame selection, temporal variance pre-filter, near-dup luma filter, dHash animation hold detection, scene-change edge rejection, response-based hold refinement
- S97 — `_reject_blurry_frames(thumbs, paths, blur_threshold, thumb_size=64)` in `frame_selection.py`: drops interior frames with Laplacian variance (uint8 scale) below threshold; first/last always kept; wired as step 1a-b between §1.2D temporal-variance filter and hold detection; `BLUR_REJECT_THRESH=50.0` in constants; `ASP_BLUR_REJECT_THRESH=50.0` to enable
- S110 — `_reject_low_contrast_frames(thumbs, paths, contrast_threshold)` in `frame_selection.py`: drops interior frames with `np.std(thumb*255)` below threshold; catches flash/whiteout panels with near-zero pixel variance that §1.2E misses (high Laplacian at borders, zero interior texture → spurious zero-displacement edges); first/last always kept; wired as step 1b-a after §1.2E; `CONTRAST_THRESH=15.0` in constants; `ASP_CONTRAST_THRESH=15.0` to enable

**Compositing — seam pre-escalation & post-composite correction (S98–S99, S122):**
- S98 — `_seam_zone_texture_energy(fa, fb, boundary, half_band=30) → float` in `compositing.py`: mean Laplacian variance in ±30px band around seam across both warped frames; flat-colour zones (low value) → pre-escalate to single-pose before ARAP (aperture problem guard); `_SEAM_LOW_TEXTURE_THRESH` flag (default 0.0=off; `ASP_SEAM_LOW_TEXTURE_THRESH=5.0`); `SEAM_LOW_TEXTURE_THRESH=5.0` in constants; wired between §1.20 and ARAP call in FG-registration loop
- S99 — `_fg_gradient_cost(canvas_zone, weight=1.0) → np.ndarray` in `compositing.py`: normalized Laplacian magnitude (values in [0, weight], shape (H, W)) on canvas zone; for fg-interior pixels (cost ≥ 1.0) in `_build_seam_cost_map`, adds gradient cost so character outline pixels are more expensive than flat fill — DP steers seam away from hairline-break-risk pixels; background pixels unaffected; `_LINE_GRAD_WEIGHT` flag (default 0.0=off; `ASP_LINE_GRAD_WEIGHT=1.0`); `LINE_GRAD_WEIGHT=1.0` in constants
- S122 — `_seam_chroma_equalize(canvas, boundaries, band_px=20, min_shift=3.0) → np.ndarray` in `compositing.py`: §1.56 post-composite chroma seam correction — complement to §1.21 (`_seam_lum_equalize`); converts strip reference bands to CIE LAB, measures mean a/b shift above vs below each boundary, applies linear additive ramp over `band_px` rows to close chroma gap; L* left unchanged (§1.21 handles luminance); targets colour-temperature and hue banding that equal-BGR luminance shifts miss; `_SEAM_CHROMA_EQ` flag (default OFF; `ASP_SEAM_CHROMA_EQ=1`); wired immediately after §1.21 in `_composite_foreground`; `SEAM_CHROMA_EQ_BAND_PX=20`, `SEAM_CHROMA_EQ_MIN_SHIFT=3.0` in constants; `"ASP_SEAM_CHROMA_EQ"` in `_CONFIG_SCHEMA`

**Pipeline quality gates (S13–S121):**
- S101 — `_compute_bg_coverage_fraction(bg_masks) → float` in `pipeline.py`: mean fraction of bg pixels (> 127) across all valid (non-None) masks; gate wired between Stage 4 and Stage 4.5 — when `_MIN_BG_FRACTION > 0.0` and fraction < threshold, returns SCANS fallback immediately (fg-dominant scene where bg normalisation, bg-masked LoFTR, and phase correlation would all operate on noise); `MIN_BG_FRACTION=0.05` in constants; `ASP_MIN_BG_FRACTION=0.05` recommended; `_compute_bg_coverage_fraction` exported in `__all__`
- S103 — `_compute_render_coverage(valid_mask) → float` in `pipeline.py`: fraction of canvas pixels with valid_mask > 0 (reached by ≥1 warped frame); gate between Stage 10.2 and Stage 10.5 — catches geometry failure where all frames pile into a dense clump leaving most canvas black (passes Stage 10.5 row check within the clump); `RENDER_MIN_COVERAGE=0.30` in constants; `ASP_RENDER_MIN_COVERAGE=0.30` recommended; `_compute_render_coverage` exported in `__all__`
- S107 — `_compute_adj_edge_coverage(edges, n_frames) → float` in `pipeline.py`: counts distinct adjacent pairs `(|i−j|=1)` with ≥1 edge using canonical `(min(i,j), max(i,j))` set keys; returns `len(covered)/(n_frames-1)`, vacuously 1.0 for n_frames≤1; gate wired between §1.16 MST weight and Stage 7 BA; catches skip-edge-dominated graphs that pass §1.15/§1.16 but have no local displacement anchors for BA interpolation; `ADJ_COVERAGE_MIN=0.60` in constants; `ASP_ADJ_COVERAGE_MIN=0.60` recommended; `_compute_adj_edge_coverage` exported in `__all__`
- S108 — `_compute_max_adjacent_gap(affines, frames) → float` in `pipeline.py`: for each pair (i,i+1) computes gap = leading edge of frame i+1 minus trailing edge of frame i along dominant scroll axis (ty_span≥tx_span → vertical, else horizontal); returns max gap, 0.0 for N<2; gate wired after §1.17 and before Stage 9.5; fires when gap > `_MAX_ADJACENT_GAP_PX` threshold; catches BA "stretch" failure where total span looks correct but two adjacent frames are placed with an uncovered strip between them — complementary to §1.17 (collapse) and §1.39 (post-render, expensive); `MAX_ADJACENT_GAP_PX=100.0` in constants; `ASP_MAX_ADJACENT_GAP_PX=100.0` recommended; `_compute_max_adjacent_gap` exported in `__all__`
- S109 — `_compute_canvas_width_ratio(canvas_w, frames) → float` in `pipeline.py`: `canvas_w / median_frame_w`; 1.0 for empty frames; gate wired between §1.44 and Stage 9.5; fires when ratio > `_MAX_CANVAS_WIDTH_RATIO`; catches BA tx-drift in a nominally vertical-scroll sequence — §3.14 does not fire (ty_span dominates), §1.17 does not fire (vertical span is correct), yet the canvas grows to 2–4× frame width; `MAX_CANVAS_WIDTH_RATIO=1.5` in constants; `ASP_MAX_CANVAS_WIDTH_RATIO=1.5` recommended; `_compute_canvas_width_ratio` exported in `__all__`
- S111 — `_compute_sign_inconsistency_rate(edges) → float` in `pipeline.py`: filters to adjacent edges (`|i-j|=1`), determines dominant axis (larger median |displacement|), returns minority-sign fraction `min(n_pos, n_neg)/len(nonzero)`; 0.0 for <2 adjacent edges (safe fallback); gate wired after §1.43 and before Stage 7 BA; catches mixed-sign displacement graphs where some adjacent edges report opposite scroll direction to the majority (wrong-peak PC/TM matches) — complementary to §1.12 Kendall-τ (post-BA monotonicity) since §1.47 operates on raw edge displacements pre-BA; `SIGN_INCONSISTENCY_MAX=0.20` in constants; `ASP_SIGN_INCONSISTENCY_MAX=0.20` recommended; `_compute_sign_inconsistency_rate` exported in `__all__`
- S112 — `_compute_adj_disp_cv(edges) → float` in `pipeline.py`: filters to adjacent edges (`|i-j|=1`), selects dominant axis (larger median |disp|), returns `std(|mags|)/mean(|mags|)`; 0.0 for <2 adjacent edges or zero mean; gate wired after §1.47 and before Stage 7 BA; catches wrong-harmonic PC peaks (e.g. 2× expected step) and non-adjacent TM jumps — an outlier that *agrees* on scroll direction (passes §1.47) but reports 10× typical magnitude still triggers this gate; `ADJ_DISP_CV_MAX=0.50` in constants; `ASP_ADJ_DISP_CV_MAX=0.50` recommended; `_compute_adj_disp_cv` exported in `__all__`
- S113 — `_compute_adj_min_weight(edges) → float` in `pipeline.py`: returns `min(e["weight"] for e in adj_edges)` where adj_edges = `{e : |i-j|=1}`; 1.0 for no adjacent edges (safe no-op); gate wired between §1.43 (coverage) and §1.47 (sign consistency); fills the gap between §1.16 (MST mean weight) and §1.43 (existence check): a pair may have an edge and the MST mean may look acceptable, but if that specific edge's weight is near zero the compositing seam at that boundary is ill-placed regardless of BA solution quality; `ADJ_MIN_WEIGHT=0.20` in constants; `ASP_ADJ_MIN_WEIGHT=0.20` recommended; `_compute_adj_min_weight` exported in `__all__`
- S114 — `_compute_ba_max_residual(edges, affines) → float` in `pipeline.py`: for each edge (i→j) computes `‖observed_disp − (affines[j].t − affines[i].t)‖₂`; returns maximum across all edges; 0.0 for empty edges/affines (safe no-op); gate wired between Stage 7 (BA output) and Stage 7b (affine validation); directly targets Category B failures where a single high-weight wrong LoFTR match corrupts BA — GNC/Cauchy down-weights it but cannot fully suppress it, producing a large residual (50–500 px) even in the solved frame placement; `BA_RESIDUAL_MAX=200.0` in constants; `ASP_BA_RESIDUAL_MAX=200.0` recommended; `_compute_ba_max_residual` exported in `__all__`
- S115 — `_compute_min_adjacent_overlap(affines, frames) → float` in `pipeline.py`: for each pair (i, i+1) computes `trailing_edge(i) − leading_edge(i+1)` along dominant scroll axis; returns minimum across all N-1 pairs; 0.0 for <2 frames (safe no-op); gate wired alongside §1.44 (after Stage 9 canvas, before Stage 9.5 confidence); complementary to §1.44 (catches negative overlap = gap); §1.51 catches *positive but thin* overlap (1–19 px) where the blend zone exists but is too narrow for reliable DP seam cutting or FEATHER_MIN=80 satisfaction; `MIN_ADJACENT_OVERLAP_PX=20.0` in constants; `ASP_MIN_ADJACENT_OVERLAP_PX=20.0` recommended; `_compute_min_adjacent_overlap` exported in `__all__`
- S116 — `_compute_ba_weighted_mean_residual(edges, affines) → float` in `pipeline.py`: computes `Σ(w_i × r_i) / Σ(w_i)` where `r_i = ‖observed_disp − (affines[j].t − affines[i].t)‖₂` and `w_i = e["weight"]`; 0.0 for empty input or all-zero weights (safe no-op); gate wired immediately after §1.50 (between Stage 7 and Stage 7b); complementary to §1.50 (max residual = single outlier); §1.52 catches *systematic* BA drift where all edges are moderately wrong (40–60px each), passing §1.50's threshold but indicating the entire global frame placement is unreliable (e.g., repeated background texture biasing the phase-correlation response surface); `BA_WMEAN_RESIDUAL_MAX=30.0` in constants; `ASP_BA_WMEAN_RESIDUAL_MAX=30.0` recommended; `_compute_ba_weighted_mean_residual` exported in `__all__`
- S117 — `_compute_canvas_memory_mb(canvas_h, canvas_w) → float` in `pipeline.py`: returns `canvas_h × canvas_w × 3 × 4 / 1024²` (float32 RGB lower-bound footprint in MB); 0.0 for zero/negative dimensions (safe no-op); gate wired immediately after Stage 9 midplane-shift canvas recompute and before §3.14 scroll-axis check; `CANVAS_MAX_DIM=32768` prevents individually extreme dimensions but not extreme products (e.g. 32768×1920 ≈ 720 MB, still within CANVAS_MAX_DIM yet risky on 4 GB systems when combined with warped-frame buffers at 3–6× canvas size); fires early before any allocation → SCANS fallback; `CANVAS_MAX_MEMORY_MB=2048.0` in constants; `ASP_CANVAS_MAX_MEMORY_MB=2048.0` recommended; `_compute_canvas_memory_mb` exported in `__all__`
- S121 — `_smooth_affine_trajectory(affines, sigma, iqr_threshold) → (smoothed, was_applied)` in `pipeline.py`: Gaussian 1D smooth on BA-solved tx/ty sequences; IQR of abs adjacent steps computed per axis; smoother skips when `max(IQR_tx, IQR_ty) ≤ iqr_threshold` (clean linear scrolls untouched); `scipy.ndimage.gaussian_filter1d(mode='nearest')` applied to both sequences when activated; rotation/scale components unchanged; wired post-§1.55 rotation gate and before Stage 7b; targets the 4 confirmed genuine fallbacks (tests 54/59/73/89) caused by phase-correlation jitter in non-linear/multi-axis scroll; `TRAJ_SMOOTH_SIGMA=1.5`, `TRAJ_SMOOTH_IQR_THRESH=10.0` in constants; `ASP_TRAJ_SMOOTH_SIGMA` + `ASP_TRAJ_SMOOTH_IQR_THRESH` in `_CONFIG_SCHEMA`; `_smooth_affine_trajectory` exported in `__all__`
- S120 — `_compute_max_affine_rotation_deg(affines) → float` in `pipeline.py`: extracts rotation via `arctan2(M[1,0], M[0,0])` for each 2×3 affine; returns max absolute value in degrees; 0.0 for empty list; gate wired post-Stage-7 BA between §1.52 weighted-mean residual and Stage 7b affine validation; catches corrupted LoFTR matches onto rotationally-similar texture patches (repeated borders, mirrored panels) that produce seemingly valid BA residuals but unreliable translations; pure-scroll captures have near-zero rotation on all affines; `MAX_AFFINE_ROTATION_DEG=5.0` in constants; `ASP_MAX_AFFINE_ROTATION_DEG=5.0` recommended; `_compute_max_affine_rotation_deg` exported in `__all__`
- S118 — `_compute_render_luma_std(canvas, valid_mask) → float` in `pipeline.py`: computes std of `mean(BGR)` per pixel for all pixels where `valid_mask > 0`; 0.0 for None inputs or zero valid pixels (safe no-op); gate wired after §1.39 render coverage and before Stage 10.5 multi-frame coverage; distinct from §1.39 (coverage *quantity*): §1.54 checks luminance *variety* — std near zero means all covered pixels share the same luminance; catches BaSiC over-correction (all frames normalized to same mean luma), silent warp collapse (all frames to same canvas region), or hold-block leakage (single repeated frame); `RENDER_LUMA_STD_MIN=5.0` in constants; `ASP_RENDER_LUMA_STD_MIN=5.0` recommended; `_compute_render_luma_std` exported in `__all__`

**Background fill (S106):**
- S106 — `_linear_interp_zero_bg(canvas, zero_mask)` in `bg_complete.py`: per-channel linear blend between nearest known pixel above and below each zero-coverage gap; boundary gaps fall back to NN copy; eliminates discrete color step artefact that NN-copy produces at gap midpoint when above/below tones differ; `_INTERP_BG_FILL` flag (default OFF; `ASP_INTERP_BG_FILL=1`)

**Rendering photometric correction (S104–S105):**
- S104 — `_adaptive_render_gain_clamp(ref_lum) → (lo, hi)` in `rendering.py`: same continuous formula as §1.4B (`clamp_width = 0.26 − 0.12 × ref_lum/255`); replaces fixed [0.88, 1.12] in `_compute_sequential_color_gains`; ±26% at black, ±14% at white; `_ADAPTIVE_RENDER_GAIN` flag (default OFF; `ASP_ADAPTIVE_RENDER_GAIN=1`)
- S105 — `_check_gain_chain_drift(gains, max_ratio) → bool` in `rendering.py`: `cum = prod(gains, axis=0)`; returns True when `|log(cum_c)| > log(max_ratio)` for any channel; called after the main loop in `_compute_sequential_color_gains` — on True, resets all gains/biases to identity (no correction better than 10× drifted chain); `_GAIN_DRIFT_MAX` flag (default 0.0=off; `ASP_GAIN_DRIFT_MAX=2.0`); `GAIN_DRIFT_MAX=2.0` in constants

**Bundle adjustment & matching (S6–S102):**
- GNC-TLS robust loss (Geman-McClure continuation), adaptive GNC f_scale, spanning-tree inlier filter, adaptive min-step threshold, static edge rejection, scale normalization, high-confidence edge re-solve (Retry 0), adaptive rotation/scale thresholds, MST weight gate, edge graph connectivity check, per-channel BGR scene-change gate, Kendall-τ monotonicity check
- S93 — `_triangular_consistency_filter(edges, max_residual_px)` in `pipeline.py`: penalises weakest edge (weight × 0.5) in triangles where leg-sum vs hypotenuse L2 residual > threshold; addresses wrong adjacent edges that geometric filter never questioned; `ASP_TRI_CONSISTENCY=80.0` to enable
- S100 — `_compute_translation_spread(pts_i, pts_j) → (mad_dx, mad_dy)` in `matching.py`: MAD of per-match displacements around the median translation; high MAD indicates LoFTR matches are scattered (bimodal from fg/bg confusion) → edge rejected, falls through to template match / phase correlation; `MATCH_SPREAD_CEIL=30.0` in constants; `ASP_MATCH_SPREAD_CEIL=30.0` to enable
- S102 — `_compute_bg_match_ratio(n_bg_pts, n_total_pts) → float` in `matching.py`: fraction of LoFTR matches that land on background pixels; `n_loftr_total` captured before bg-filter block; after bg-filtering, when `_LOFTR_BG_RATIO_MIN > 0.0` and ratio < threshold, LoFTR edge rejected (`pts1` zeroed) and falls through to ALIKED/template-match/phase-corr; catches fg-dominated pairs where n_bg ≥ 20 but only 5–10% of total matches — sparse clustered bg matches produce a noisy median displacement even after passing the §1.20 count check; `LOFTR_BG_RATIO_MIN=0.15` in constants; `ASP_LOFTR_BG_RATIO_MIN=0.15` recommended

**Diagnostic overlay & HITL seam inspector (S94–S97):**
- S94 — `_annotate_seams(canvas, boundaries, seam_post_diffs, seam_single_pose)` in `compositing.py`: coloured horizontal lines at each seam boundary (green/amber/red by alignment quality) with text labels; `ASP_SEAM_OVERLAY=1` to enable
- S95 — `seam_meta_out`/`seam_overrides` params in `_composite_foreground()`: `seam_meta_out` is a mutable dict populated with `{"boundaries", "seam_post_diffs", "seam_single_pose"}` on return; `seam_overrides` maps seam k → `{"force_single_pose": bool, "force_blend": bool, "waypoints": List[Tuple[int,int]]}` (canvas-space y coords)
- S96 — `_extract_seam_crops(canvas, boundaries, band_px=50) → Dict[int, np.ndarray]` in `compositing.py`: crops ±50px rows around each seam boundary, clamped to canvas bounds; `seam_crops` key added to `seam_meta_out`; `_SeamCard` in `SeamDiagnosticDialog` displays crop thumbnail when provided
- S123 — `_seam_cut()` gains `waypoints: Optional[List[Tuple[int,int]]]` (zone-local coords). Three-part hard guarantee: pre-DP inf-injection (forces forward pass through waypoint pixel), forced traceback (overrides 3-neighbour argmin at waypoint columns), post-smooth re-stamp (re-applies after `_smooth_seam_path`/`_clamp_seam_path`). Canvas-space waypoints from `seam_overrides[k]["waypoints"]` auto-converted to zone-local via `y -= y0_zone`; `_seam_job` tuple extended to 7 elements.

**Canvas & alignment (S13–S61):**
- Multi-frame canvas coverage gate (Stage 10.5), PANORAMA stitcher fallback, TELEA border fill, scroll-axis detection wired, canvas span utilization gate, adaptive min-gap threshold, adaptive boundary search

**Compositing & seam quality (S10–S46):**
- Seam DP vectorization (§1.5A), parallel seam pre-computation, seam path cache (§1.5D), tiered seam cost, per-pixel DSFN ramp, bg-mask-aware DSFN, Poisson seam blend, seam color match, single-pose soft-edge, per-DSFN ramp, seam color similarity gate (BGR), seam path smoothing, seam path boundary clamp, seam hard corridor barrier, adaptive SP soft-edge width, adaptive SP escalation threshold, fg-density feather cap, tight-step preemptive escalation, post-composite luma equalization, seam-step post-composite gate, zone minimum height guard, seam path instability escalation, seam FG penetration escalation, SemanticStitch column barrier, GNC-TLS compositing, multi-scale spatially-varying gain, gain-adaptive feather minimum

**Photometric normalization (S18–S49):**
- Continuous adaptive gain clamp (§1.4B), per-pair coherence gate, bg-gain unclamped override, histogram CDF matching, per-frame exposure outlier rejection, bg-only normalization coverage floor, background zero-coverage fill (`bg_complete.py`)

**Masking & segmentation (S79–S83):**
- SAM-2 wired via `_USE_SAM2` flag (S79 + S80), Grounded SAM-2 (Issue 10A1), BiRefNet two-channel selector, foreground-masked DINOv2, Otsu bg mask for phase correlation (`_otsu_bg_mask_pair`, S80)
- S83 — `_compute_fg_masks_sam2_stateful()` + `_cleanup_sam2_state()` in `masking.py` — live SAM-2 predictor+state kept alive across HITL boundary (uses `mkdtemp`, no `reset_state`/`del`); `AnimeStitchPipeline` stores `_sam2_predictor/_sam2_inference_state/_sam2_tmp_dir/_sam2_frame_h/_sam2_frame_w`; `_cleanup_sam2_state()` method frees GPU+disk after checkpoint 1.5 dialog closes; HITL checkpoint 1.5 data dict now includes live state keys so `_refine_cb` in `stitch_tab.py` calls `_refine_masks_with_clicks` with the real predictor

**Config & infra (S27–S44):**
- TOML config loader (`asp_config.toml`, auto-loaded), config schema validation (14 `ASP_*` keys), config → env injection with `setdefault`, `_reload_scans_frames` (on-demand SCANS reload)

**HITL (S79–S84):**
- `QWaitCondition`/`QMutex` staged execution; 4 HITL signals + pause points; `SelectionReviewDialog`, `EdgeReviewDialog`, `CanvasInspectorDialog`, `CoverageHeatmapDialog`; all 4 in `gui/src/dialogs/`
- S81 — HITL checkpoint 1.5 (mask review): `sig_review_masks` + `set_mask_override()` in `stitch_worker.py`; `MaskReviewDialog` in `gui/src/dialogs/mask_review_dialog.py` with `_ClickOverlay` (left=pos/right=neg SAM-2 prompts) + `_RefinementWorker(QThread)`
- S81 — `backend/src/animation/grounding.py` (new): lazy GroundingDINO wrapper (`_detect_objects`, `_detect_best_box`, `_detect_exclusion_mask`); graceful ImportError fallback; `GROUNDING_DINO_CKPT`/`CFG` env vars
- S81 — `masking.py`: `_compute_fg_masks_grounded_sam2()` (text prompt → DINO bbox → SAM-2 propagation) + `_refine_masks_with_clicks()` (pos/neg click re-propagation)
- S81 — `backend/src/animation/data_serialization.py` (new): `COCOAnnotationBuilder` (fg segmentation, seam-exclusion, frame-selection annotations; RLE via pycocotools, polygon fallback; atomic write) + `LabelStudioExporter` (model predictions + human annotations for RLHF preference learning) + `create_session_serializers()` factory
- S81 — `_build_seam_cost_map()` / `_composite_foreground()` gain `exclusion_masks` param — NL seam routing: cost=1e6 hard barrier where mask>127, forcing DP seam away from named objects
- S82 — `exclusion_masks` threaded end-to-end: `AnimeStitchPipeline.exclusion_masks` instance attr → Stage 11 → `_composite_foreground`; `StitchWorker.set_exclusion_masks()` + HITL checkpoint 1.5 wiring; `MaskReviewDialog` seam-exclusion section (GroundingDINO detect button + `sig_exclusion_masks_accepted` signal); COCO+LS auto-save at checkpoint 1.5 (→ `~/.image-toolkit/hitl_annotations/`)
- S82 — `backend/src/animation/video_ingestion.py` (new, Issue 9): `VideoIngestionStream` + `ingest_video()` — PyAV proxy-first decode at ¼ res, telecine-drop dedup, uniform/keyframe/smart selection, full-res seek-based per-frame decode; `ASP_VIDEO_PROXY_SCALE/MAX_FRAMES/TELECINE_MAD/KEYFRAMES_ONLY` env vars; graceful `pip install av` fallback
- S83 — Live SAM-2 state preservation across HITL checkpoint boundary: `_compute_fg_masks_sam2_stateful()` stateful variant returns `(masks, predictor, state, tmp_dir, H, W)`; `AnimeStitchPipeline._compute_fg_masks()` stores tuple on `self`; checkpoint 1.5 data dict passes live state; `_refine_cb` in `stitch_tab.py` now calls `_refine_masks_with_clicks(predictor, state, ...)` for real; `_cleanup_sam2_state()` frees GPU/disk after dialog closes; 10 new tests in `test_masking.py`
- S84 — Video ingestion HITL + "From Video" GUI mode: `sig_review_video = Signal(object)` (checkpoint 0); `_hitl_video_pause()` blocks on `_hitl_mutex`; `StitchWorker.run()` ingests video via `ingest_video()` into `mkdtemp` before pipeline, emits `sig_review_video` in HITL mode, applies `frame_override`; `SelectionReviewDialog` configurable `title` param; `stitch_tab.py` "From Video Source" checkbox + `_video_input_widget` + `_on_hitl_review_video()`; 5 new GUI tests in `TestStitchWorkerVideoPath`
- S85 — HITL Checkpoint 3.5 seam boundary editor: `_compute_initial_boundaries(affines, frames) → np.ndarray` extracted to `compositing.py` (`__all__`); `_composite_foreground()` + `AnimeStitchPipeline._composite_foreground()` gain `preset_boundaries: Optional[np.ndarray] = None`; `StitchWorker` `sig_review_boundaries` + `set_boundary_override()` + checkpoint 3.5 block between Stage 10/11; `boundary_editor_dialog.py` (new) — `_DraggableLine(QGraphicsLineItem)` + `BoundaryEditorDialog` with draggable N-1 seam lines + "Reset to Auto" + `adjusted_boundaries()`; `stitch_tab._on_hitl_review_boundaries()`; 5 new tests `TestComputeInitialBoundaries`
- S86 — HITL Checkpoint 4.5 post-composite seam painter: `paint_mask: Optional[np.ndarray]` param in `_composite_foreground()` appended to `_eff_exclusion` list (canvas-space uint8, zone-sliced identically to `exclusion_masks`); `AnimeStitchPipeline._composite_foreground()` wrapper updated; `StitchWorker` `sig_review_composite` + `set_paint_mask()` + re-composite while-loop at checkpoint 4.5 (breaks on accept, re-runs with new mask on `SeamPainterDialog.RECOMPOSITE=2`); `seam_painter_dialog.py` (new) — `_PaintCanvas(QLabel)` with alpha-overlay left-drag paint / right-drag erase, `paint_mask_preview()` → uint8 alpha channel, `full_resolution_mask()` upscales via `INTER_NEAREST`; `stitch_tab._on_hitl_review_composite()`; 5 new tests `TestPaintMask`; total backend/test/animation/ suite: 577 tests (2 skipped)
- S87 — HITL Checkpoint 5 final output RLHF feedback: `StitchWorker` `sig_review_output = Signal(object)` + `set_output_feedback(overall_rating, annotations)` + `"output"` in signal map + checkpoint 5 block after Stage 13 (lazy-imports `FeedbackStore`/`StitchAnnotation`, calls `add_from_image()`, logs result); `final_output_review_dialog.py` (new) — `_AddFlawDialog` (flaw_type QComboBox from `RLHF_FLAW_TYPES` + severity QDoubleSpinBox) + `FinalOutputReviewDialog` (canvas preview + overall-quality slider 0–10 in 0.5 steps + flaw annotation QListWidget + Save/Skip); `stitch_tab._on_hitl_review_output()` wired; 7 new tests in `test_rlhf_feedback.py` (FeedbackStore add/iter/count/roundtrip/from_image/empty/malformed) → **584 tests passing**
- S92 — HITL Session Viewer: `gui/src/dialogs/hitl_session_viewer_dialog.py` (new) — `_list_sessions()` (mtime-sorted), `_load_session_meta()` (JSON-only, no numpy decode), `_format_session_info()` (checkpoint label map `_CHECKPOINT_LABELS`); `HITLSessionViewerDialog` — QSplitter list/detail; Load-for-Replay (sets `_selected_path`, accepts); Delete (QMessageBox confirm + unlink + refresh); Export (shutil.copy2 + DontUseNativeDialog); Refresh; `selected_path()` accessor; `session_dir` param for testability; `stitch_tab.py`: "Browse Sessions…" button + `_on_browse_sessions()` handler. 8 tests `TestListSessions`/`TestFormatSessionInfo`/`TestHITLSessionViewerDialog` → **18 GUI tests** (backend 598 unchanged)
- S91 — Canvas Inspector rotation/scale editor: `_rot_angles` + `_scale_factors` per-frame lists; `QDoubleSpinBox` for rotation (±180°, step 0.5°) and scale (0.1–3.0, step 0.01); `_update_transform_controls()` populates spinboxes on frame select; `_on_rot/scale_changed()` update list + call `setRotation/setScale()` on drag item; `setTransformOriginPoint(fw/2, fh/2)` for center-pivot; `_reset_frame()` zeroes rot/scale; `adjusted_affines()` applies `R(θ,s) @ orig_2x2` before tx/ty nudge; 5 tests `TestCanvasInspectorRotScale` → **10 GUI tests in test_canvas_inspector_dialog.py** (backend 598 unchanged)
- S90 — Canvas Inspector drag-to-reposition: `_DraggableFrameItem(QGraphicsRectItem)` in `canvas_inspector_dialog.py` — `ItemIsMovable | ItemSendsGeometryChanges | ItemIsSelectable`; `itemChange(ItemPositionChange)` writes nudge in-place from proposed scene pos; thumbnail pixmaps as child items move with rect; `_populate_scene()` creates drag items; scene selection syncs list widget; configurable `_step_spin` (QSpinBox, 1–200, default 10) replaces hardcoded ±10px; `_nudge()` + `_reset_frame()` call `setPos()` on drag item; `gui/test/test_canvas_inspector_dialog.py` (new): 5 tests `TestCanvasInspectorDrag` → **+5 GUI tests** (backend 598 unchanged)
- S89 — HITL Checkpoint 2 manual edge entry: `_build_manual_edge(i, j, dx, dy, weight=0.9) → dict` in `pipeline.py` (pure-translation M, single-point pts, `method="manual"`, weight clipped, exported in `__all__`); `_ManualEdgeDialog(QDialog)` in `edge_review_dialog.py` — i/j spinboxes bounded by n_frames, dx/dy QDoubleSpinBox, weight 0.9 default; `EdgeReviewDialog` gains `_manual_edges` list + `_n_frames` + "Add Edge…" toolbar button + `_on_add_edge()`; `_populate()` renders manual edges in purple dotted lines + purple table rows (always-on, uncheckable); `accepted_edges()` returns filtered originals + all manual entries; `StitchWorker` Checkpoint 2 updated to convert `method="manual"` overrides to full pipeline edges via `_build_manual_edge()`; bugfix: `_on_hitl_review_edges()` fixed to pass `data=data` to constructor; 5 tests `TestBuildManualEdge` → **598 tests passing**
- S88 — HITL session persistence & replay: `backend/src/animation/hitl_session.py` (new) — `_encode_array`/`_decode_array` (numpy ↔ base64-JSON, 8 MB skip threshold for large arrays); `_to_json`/`_from_json` recursive converters; `save_session(overrides, path)` writes `{version, timestamp, checkpoints}` JSON; `load_session(path)` restores override dicts with numpy decoded; `autosave_path()` timestamped path under `~/.config/image-toolkit/hitl_sessions/`; `StitchWorker` gains `session_path: Optional[str]` → loads replay dict at init; `_hitl_session_overrides` dict accumulates non-cancel overrides; autosaves after Stage 13 success; `current_session_path` property; `save_session(path)` public method; `_make_hitl_pause_cb()` and `_hitl_video_pause()` now replay stored overrides without blocking when `hitl_mode=False`; `stitch_tab.py` "Load Session…" button + `_session_path_label` + `_on_load_session()` + session path shown in success dialog; 9 tests in `test_hitl_session.py` (ndarray codec ×4, save/load ×5) → **593 tests passing**

**Benchmark infra:**
- `_compute_rlhf_score`, `_ghosting_score_v2` (autocorrelation double-edge), `seam_bhattacharyya_distances`, per-seam SIQE ghost scores, `_compute_aligned_ssim` (MOTION_EUCLIDEAN)

**Current test corpus: 97 tests (asp_test01–asp_test97)**
- asp_test97 added 2026-06-13 ("Akane wa Tsumare Somerareru - 02", 90 frames, horizontal-ish scroll, 16 frames selected)
- asp_test07 replaced 2026-06-13 (new "Akane wa Tsumare Somerareru - 01" dataset, 182 frames, 28 frames selected)
- Ground truth available for 55/97 tests

**Benchmark results (2026-06-13, test07 + test97 only):**
- test07: SC=23.6 (ASP) vs 43.2 (simple); verdict=comparable; sharpness 111.75 vs 38.81; ghosting_siqe 22.98 vs 91.65; 28 frames, 1593×3841 output
- test97: SC=10.6 (ASP) vs 14.3 (simple); verdict=simple_better (coverage 86% vs 98%); ghosting_siqe 33.2 vs 61.79; 16 frames, 2505×1859 output

**The CV sharpness metric (Laplacian variance) is inverted** — hard seam edges inflate sharpness. Use `seam_coherence` as primary quality proxy (≤18 good, 18–28 moderate, >28 severe). `ghosting_siqe` (§3.8A autocorrelation) is more reliable than `ghosting_score` for detecting double-edges.

**Visually confirmed good outputs:** asp_test28, asp_test58.

---

## Phase 2 Architecture (Issue 9 & 10 — Next Generation)

Issue 10 (Multi-modal HITL) is **implemented in S81–S96**. Issue 9 (Video ingestion) is implemented in S82 + S84.

| Module | Location | Status |
|--------|----------|--------|
| `VideoIngestionStream` | `backend/src/animation/video_ingestion.py` | ✅ **S82** — proxy-first decode, telecine dedup, smart/uniform/keyframe selection |
| `grounding.py` functions | `backend/src/animation/grounding.py` | ✅ **S81** — GroundingDINO wrapper + exclusion mask |
| `_compute_fg_masks_grounded_sam2` | `backend/src/animation/masking.py` | ✅ **S81** — text → DINO bbox → SAM-2 propagation |
| `_refine_masks_with_clicks` | `backend/src/animation/masking.py` | ✅ **S81** — pos/neg click SAM-2 re-propagation |
| `_compute_fg_masks_sam2_stateful` | `backend/src/animation/masking.py` | ✅ **S83** — live predictor+state preserved across HITL |
| `_cleanup_sam2_state` | `backend/src/animation/masking.py` | ✅ **S83** — GPU/disk cleanup after HITL dialog closes |
| `COCOAnnotationBuilder` | `backend/src/animation/data_serialization.py` | ✅ **S81** — COCO JSON + RLE/polygon encoding |
| `LabelStudioExporter` | `backend/src/animation/data_serialization.py` | ✅ **S81** — Label Studio tasks with RLHF delta |
| `MaskReviewDialog` | `gui/src/dialogs/mask_review_dialog.py` | ✅ **S81** — click overlay + refinement worker |
| `exclusion_masks` in compositing | `compositing.py` | ✅ **S81** — NL seam routing hard barrier |
| HITL checkpoint 0 (video review) | `gui/src/helpers/models/stitch_worker.py` | ✅ **S84** — `sig_review_video` + `_hitl_video_pause()` + "From Video" GUI |
| HITL checkpoint 3.5 (boundary editor) | `gui/src/dialogs/boundary_editor_dialog.py` | ✅ **S85** — draggable seam lines; `_compute_initial_boundaries` + `preset_boundaries` param |
| HITL checkpoint 4.5 (seam painter) | `gui/src/dialogs/seam_painter_dialog.py` | ✅ **S86** — paint/erase seam exclusion; `paint_mask` + `_eff_exclusion`; re-composite while-loop |
| HITL checkpoint 5 (RLHF feedback) | `gui/src/dialogs/final_output_review_dialog.py` | ✅ **S87** — quality slider + flaw annotations; `FeedbackStore.add_from_image()` after save |
| HITL session persistence & replay | `backend/src/animation/hitl_session.py` | ✅ **S88** — `save_session`/`load_session`; autosave after run; replay without blocking |
| HITL Checkpoint 2 manual edge entry | `gui/src/dialogs/edge_review_dialog.py`, `backend/src/animation/pipeline.py` | ✅ **S89** — `_ManualEdgeDialog` + `_build_manual_edge()`; purple dotted rendering; `method="manual"` conversion at StitchWorker |
| Canvas Inspector drag-to-reposition | `gui/src/dialogs/canvas_inspector_dialog.py` | ✅ **S90** — `_DraggableFrameItem` with `ItemIsMovable + ItemSendsGeometryChanges`; thumbnail children; configurable step spinbox |
| Canvas Inspector rotation/scale editor | `gui/src/dialogs/canvas_inspector_dialog.py` | ✅ **S91** — per-frame `_rot_angles`/`_scale_factors`; `QDoubleSpinBox` controls; `R(θ,s) @ orig_2x2` in `adjusted_affines()`; center-pivot via `setTransformOriginPoint` |
| HITL Session Viewer | `gui/src/dialogs/hitl_session_viewer_dialog.py` | ✅ **S92** — browse/inspect/delete/export sessions; `selected_path()` → load for replay; "Browse Sessions…" button in stitch_tab |
| Triangular consistency filter | `backend/src/animation/pipeline.py` | ✅ **S93** — `_triangular_consistency_filter()`: penalise weakest edge in inconsistent triangles; `ASP_TRI_CONSISTENCY=80.0` |
| Seam overlay diagnostic annotation | `backend/src/animation/compositing.py` | ✅ **S94** — `_annotate_seams()`: coloured lines (green/amber/red) + text labels at seam boundaries; `ASP_SEAM_OVERLAY=1` |
| HITL checkpoint 4.6 (seam inspector) | `gui/src/dialogs/seam_diagnostic_dialog.py` | ✅ **S95–S96** — `SeamDiagnosticDialog`: per-seam cards (diff/SP/override checkboxes + ±50px crop thumbnail); `force_single_pose`/`force_blend` via `seam_meta_out`+`seam_overrides`; S96 adds `seam_crops` key to `seam_meta_out` |

**S81 Known limitation resolved (S83):** Live SAM-2 state is preserved across the HITL checkpoint boundary via `_compute_fg_masks_sam2_stateful()`. The predictor is passed through the checkpoint 1.5 data dict and `_refine_cb` in `stitch_tab.py` now calls `_refine_masks_with_clicks` with the live predictor when SAM-2 is active (`ASP_USE_SAM2=1`).

See `research/ASP_High_Value_Issues_Report.md` Issues 9 & 10 and `research/Upgrading Anime Stitch Pipeline.md` for the full spec.

---

## Known Issues Summary (96-test corpus, 2026-06-07 S27 baseline)

### SCANS fallbacks (4 genuine, down from 51/96)

| Test | Cause |
|------|-------|
| test54 | Genuine: min_gap below floor after validation retries |
| test59 | Genuine: bundle ratio failure |
| test73 | Genuine: extreme diagonal (dx_cv=25.3) |
| test89 | Genuine: ratio=4.0 bundle failure |

### Remaining compositing issues

Most ASP-succeeded tests now pass validation. Remaining seam quality issues:

| Category | Approx. count | Seam∇ proxy |
|----------|--------------|-------------|
| Catastrophic — severe color banding | ~5–10 | >12 |
| Poor — visible seams | ~10–15 | 7–12 |
| Moderate — seams visible but usable | ~20–25 | 4–7 |
| Good — genuine improvement | ~15–20 | <5 |

### Open roadmap items

- **§1.9A Fallback path purity** ✅ Fixed (S28) — `_spatial_dedup_frames()` syncs `scans_frames` after spatial dedup
- **§1.10A RLHF post-run quality gate** — call `reward_model.predict(output)` after each run; log score; flag < 0.6 for review
- **§2.x Diagnostics** — per-test pipeline trace, benchmark comparison dashboard

---

## Test Corpus (97 datasets, asp_test01–asp_test97)

Datasets are in `data/asp_testXX/` (zero-padded). Frames are consecutive video frames (~42ms intervals) smart-selected by phase-correlation to ~18 frames/dataset (50px step target).

**Numbering history:**
- Original 94-test benchmark corpus
- `asp_test25` added (*Akane wa Tsumare Somerareru - 02*, ~223 frames); old test25–94 shifted +1
- `asp_test96` added (*Ajisai no Chiru Koro ni - 01*, ~139 frames)
- `asp_test07` replaced 2026-06-13 with new "Akane wa Tsumare Somerareru - 01" dataset (182 frames)
- `asp_test97` added 2026-06-13 (*Akane wa Tsumare Somerareru - 02*, 90 frames, 16 selected)

**Ground truth images:** 55 of 97 tests have a reference panorama in `data/ground_truth/asp_testXX.{png,jpg,jpeg}`. These are used by the benchmark for SSIM/PSNR comparison vs. GT — the most reliable quality signal available.

Tests WITH ground truth: 1, 2, 4, 5, 6, 8, 9, 11, 12, 14, 15, 16, 17, 20, 25, 26, 27, 31, 32, 33, 34, 37, 42, 43, 44, 45, 46, 49, 50, 52, 54, 57, 58, 59, 65, 70, 72, 74, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 88, 89, 90, 91, 92, 95, 96

### Representative dataset table (selected tests)

| Dataset | Frames | Seam∇ | Fallback? | Visual quality | Notes |
|---------|-------:|------:|-----------|---------------|-------|
| `asp_test01` | 16 | 9.05 | N | Poor | Hard color step mid-image |
| `asp_test03` | 5 | 6.86 | N | Moderate | Very few frames, limited compositing |
| `asp_test04` | 23 | 8.64 | N | **Catastrophic** | 4+ color strips, duplicate limbs |
| `asp_test07` | 11 | 7.59 | N | Moderate | Close-up, appears coherent |
| `asp_test08` | 14 | 10.12 | N | **Catastrophic** | Character 3× ghosted |
| `asp_test10` | 8 | 4.23 | N | Moderate | Few frames, low seam gradient |
| `asp_test11` | 11 | 10.25 | N | **Catastrophic** | Severe color banding |
| `asp_test13` | 14 | 4.89 | Y | SCANS | ratio=10.6 outlier bundle |
| `asp_test17` | 19 | 5.68 | N | Moderate | Subtle banding, mostly coherent |
| `asp_test18` | 19 | 1.56 | N | Likely ok | Composite bypassed (horizontal scroll detect) |
| `asp_test07` | 28 sel | SC=23.6 | N | Comparable | New dataset 2026-06-13 (182 raw frames); *Akane wa Tsumare Somerareru - 01* |
| `asp_test25` | NEW | — | — | New test | *Akane wa Tsumare Somerareru - 02* sequence; GT available |
| `asp_test26` | 11 | 10.35 | N | **Catastrophic** | Was old test25; extreme color break; GT available |
| `asp_test28` | 21 | 9.89 | N | ✅ **Good** | Was old test27; proper vertical panorama; GT available |
| `asp_test35` | 6 | 13.85 | N | Moderate | Was old test34; seam visible but extends scene; GT available |
| `asp_test37` | 26 | 3.92 | N | **Catastrophic** | Was old test36; swimsuit changes color per strip; GT available |
| `asp_test38` | 14 | 0.87 | Y | SCANS (clean) | Was old test37; very low seam — SCANS is good; GT available |
| `asp_test43` | 23 | 3.31 | Y | SCANS (clean) | Was old test42; low seam — SCANS is good; GT available |
| `asp_test48` | 9 | 1.50 | Y | SCANS (clean) | Was old test47; very low seam — SCANS is good |
| `asp_test58` | 27 | 5.82 | N | ✅ **Good** | Was old test57; clean extended coverage; GT available |
| `asp_test61` | 19 | 7.44 | N | Poor | Was old test60; banding but covers more than simple |
| `asp_test70` | 23 | 1.77 | Y | SCANS (clean) | Was old test69; very low seam — SCANS is good |
| `asp_test79` | 30 | 10.58 | N | Likely poor | Was old test78; slowest (289s), high seam; GT available |
| `asp_test86` | 29 | 8.76 | N | **Catastrophic** | Was old test85; multiple harsh color bands; GT available |
| `asp_test88` | 7 | 15.24 | N | Likely catastrophic | Was old test87; highest seam gradient |
| `asp_test89` | 22 | 4.52 | Y | SCANS | Was old test88; ratio=4.0 bundle failure; GT available |
| `asp_test91` | 17 | 1.78 | Y | SCANS (clean) | Was old test90; very low seam — SCANS is clean; GT available |
| `asp_test96` | NEW | — | — | New test | *Ajisai no Chiru Koro ni - 01* sequence; GT available |
| `asp_test97` | 16 sel | SC=10.6 | N | simple_better (coverage) | New 2026-06-13; *Akane wa Tsumare Somerareru - 02* (90 raw frames); horizontal scroll; 2505×1859 output |

**Seam∇ (seam_gradient)** is the best available diagnostic metric: `< 5` = likely clean or SCANS fallback; `5–8` = moderate seam; `> 8` = likely poor or catastrophic.

### Diagnostic snippet — check seam gradient across all datasets

```bash
source .venv/bin/activate && python3 -c "
import json, os
with open('backend/benchmark/results/anime_stitch_20260601_152735.json') as f:
    d = json.load(f)
for ds in sorted(d['datasets'], key=lambda x: x['metrics_asp'].get('seam_gradient', 0), reverse=True):
    ah = ds['affine_health']
    seam = ds['metrics_asp'].get('seam_gradient', 0)
    fb = 'FB' if ds['used_fallback'] else '  '
    print(f\"{ds['name']} {fb} seam={seam:.2f} ratio={ah['ratio']:.1f} gap={ah['min_gap_px']:.0f}px\")
"
```

---

## Benchmark: Selective Test Run

The benchmark (`backend/benchmark/bench_anime_stitch.py`) now supports selective test execution for fast iteration:

```bash
source .venv/bin/activate

# Run specific tests (fastest feedback)
python3 -m backend.benchmark.bench_anime_stitch --tests asp_test04 asp_test28 asp_test58

# Run a numeric range
python3 -m backend.benchmark.bench_anime_stitch --range 1-10

# Run comma-separated test numbers
python3 -m backend.benchmark.bench_anime_stitch --range 4,8,27,57

# First N tests
python3 -m backend.benchmark.bench_anime_stitch --first 5

# Skip already-processed datasets
python3 -m backend.benchmark.bench_anime_stitch --skip-done

# Re-run the known good and known bad tests for regression checking
python3 -m backend.benchmark.bench_anime_stitch --tests asp_test04 asp_test08 asp_test28 asp_test37 asp_test58 asp_test86
```

### Recommended test subsets

| Purpose | Command |
|---------|---------|
| Quick sanity (2 good + 2 bad + 1 SCANS) | `--tests asp_test28 asp_test58 asp_test04 asp_test86 asp_test38` |
| Full catastrophic failures (new numbers) | `--range 4,8,11,26,37,86,88` |
| Borderline tests (near-50px threshold, new numbers) | `--range 38,59,72,74,80,81,85,90` |
| Tests with ground truth | `--range 1,2,4,5,6,8,9,11,12,14,15,16,17,20,25,26,27` |
| New tests only | `--tests asp_test25 asp_test96` |

---

## Quality target

The **simple stitch** (`_merge_images_scan_stitch`) is consistently the better output in most cases for this corpus. The pipeline's goal should be to match or exceed it. Right now it does NOT do so in the majority of tests.

Visual quality standard (in priority order):
1. No severe horizontal color bands (adjacent strips must match within ±15 luminance units)
2. No body-part duplication at strip seams
3. No ghosting (3+ frames per canvas row required for temporal median)
4. Natural brightness transitions at frame boundaries

### Fast iteration loop

Do NOT re-run GPU-heavy stages (BiRefNet, LoFTR) when iterating on compositing. Use the pre-computed stage outputs from any already-processed dataset:

```bash
# Check which datasets have pre-computed stages
ls data/asp_test27/output/panorama_stages/
# stage02_normalised_frame*.png, stage04_bgmask_frame*.png,
# stage08_canvas_info.json, stage09_temporal_render.png, stage11_fg_composite.png

# Run only compositing from saved stages (adapt run_pipeline_v2.py)
source .venv/bin/activate
python3 archive/run_pipeline_v2.py
```

### Unit test suite

```bash
source .venv/bin/activate
pytest backend/test/animation/ -q          # 567 tests (~30s, no GPU) — 2 skipped (pyav)
pytest backend/test/animation/ -k "canvas" # run a specific module
```

| File | Covers |
|------|--------|
| `test_bundle_adjust.py` | Stage 7 LM solver — frame clustering, anchor frame, outlier edges |
| `test_filter_edges.py`  | `_filter_edges` — wrong-sign, gross outliers, geometric consistency |
| `test_canvas.py`        | `_compute_canvas`, `_crop_to_valid` — overcrop, horizontal scroll, TELEA fill |
| `test_affine_validation.py` | `_validate_affines` spec — ratio, min_gap, rotation, scale |
| `test_compositing.py`   | `_diff_to_feather`, `_global_gain_normalize`, `_composite_foreground`, seam functions |
| `test_rendering.py`     | `_render_median`, `_render_first`, ghosting detection, baselines |
| `test_frame_selection.py` | `smart_select_frames`, hold detection, `_near_dup_luma_filter` |
| `test_fg_register.py`   | Stage 8.5 fg pose registration — ARAP Push, flow-guided alignment |
| `test_bench_metrics.py` | `_compute_aligned_ssim`, benchmark metric helpers |
| `test_config.py`        | §1.8A `load_asp_config` — TOML loading, env injection, setdefault precedence |
| `test_pipeline.py`      | §1.9A `_spatial_dedup_frames` — dedup logic, scans_frames sync, edge reindex |

---

## Constraints

- NEVER skip MFSR by default in the production pipeline — only skip it in test scripts. The GUI exposes an "enable MFSR" toggle.
- Do NOT add `QPixmap`, Qt, or GUI imports inside any `backend/src/animation/` file.
- Keep `_composite_foreground` signature unchanged — called by pipeline, GUI worker, and test scripts.
- Gains applied in `_render_median` and `_composite_foreground` are independent — do not confuse them.
- `has_content = src.max(axis=2) > 0` must stay at `> 0` — dark pixels with max=1–10 are real content.
- Stage 11 uses `INTER_LINEAR`, not `INTER_LANCZOS4` — Lanczos4 produces halos at silhouette edges.

My first task is: read the anime pipeline issues and analysis reports in `.agent/cache/*.md`, then read the pipeline source code in `backend/src/animation/`, then understand the current visual failures and architectural root causes, and propose or implement the fixes described in the priority list above.
