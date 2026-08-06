---
trigger: anime_stitch
description: Rules for modifying the AnimeStitchPipeline — compositing, rendering, alignment, and photometric correction code in backend/src/animation/.
---

You are working on the anime image stitching pipeline in `backend/src/animation/`. Apply the following rules at all times.

---

## Iteration & Testing

- **Run the unit test suite first.** Before and after any change to `backend/src/animation/`, run `pytest backend/test/animation/ -q`. The suite has 772 backend tests (2 skipped: pyav) + 18 GUI tests (10 in `test_canvas_inspector_dialog.py`, 8 in `test_hitl_session_viewer_dialog.py`) covering all issue categories with no GPU dependency (~30s). A regression here is a hard blocker.
- **Update tests when fixing documented bugs.** Tests that document broken behavior (e.g. near-zero edge clustering) have a comment saying "update this assertion after the fix". Find the test and flip the assertion to verify the corrected behavior.
- **Always use the fast iteration loop.** Do not re-run BiRefNet or LoFTR to test compositing changes. Load pre-computed stages from `data/asp_test1/output/panorama_stages/` via `archive/run_pipeline_v2.py`. Only re-run full GPU stages when changing Stages 1–8.
- **View the output image after every run.** Use the Read tool on the `.png` output to visually inspect the result before claiming success. Do not rely solely on printed gain/delta values.
- **Compare against the simple stitch.** The reference target is `data/asp_test1/output/simple_stitch.png`. No horizontal bands, no brightness discontinuities, no block artifacts.
- **Test all datasets.** A fix that helps `asp_test1/` (8 frames) must not break `asp_test6/` (the positive baseline), `asp_test3/` (11 frames, Stage 9 failure), `asp_test2/` (10 frames, alignment failure), or the datasets asp_test4–asp_test22.

---

## Photometric Corrections

- **LS normalization clamp is adaptive (§1.4B, S24).** `_adaptive_gain_clamp(ref_lum, frame_lum)` in `compositing.py` uses a continuous formula: `clamp_width = 0.26 − 0.12 × (ref_lum/255)`. This gives ±26% for pure-black scenes and ±14% for pure-white — no binary threshold. Do NOT replace this with a fixed clamp or the old `(0.95, 1.05)` range. The wide clamp is needed for dark-scene animated frames where gains outside ±7% are common. The natural scene brightness gradient between panels is real; the clamp's purpose is to bound gross measurement error, not to artificially flatten it.
- **No per-zone gain correction.** The `gain_seam` measurement in `_composite_foreground` compares different scene elements at the same canvas row — it measures scene content, not photometric calibration. Keep it as `gain_seam = np.ones(3)`.
- **No post-composite seam ramp inside feather zones.** `_apply_canvas_seam_correction` was designed for the hard-partition-only case. When wide feather blends are active, the ramp correction applies inside the blend zone and creates new visible bands. It is intentionally disabled.
- **Do not re-enable** any of these without first verifying on all three test datasets.
- **Stage 11 bg normalization is bg-only, scalar, BT.601.** The gain in `_composite_foreground` is computed from background-classified pixels only (`bg_sel = warped_bg[i] & (warped_list[i].max(axis=2) > 10)`) and applied to background pixels only. The gain is a single scalar (BT.601 luminance: B×0.114 + G×0.587 + R×0.299), not per-channel. The clamp is now adaptive (see §1.4B above). Per-channel gain introduced hue shifts on warm/red-dominant backgrounds. Do NOT revert to per-channel or per-pixel gain.
- **Per-pair coherence gate (§1.4A, S18).** `_coherence_skip_mask` in `compositing.py` generates a per-frame skip mask — only frames in adjacent pairs whose luminance diff exceeds `coherence_limit=20.0` skip normalization. This replaced the old global `_skip_normalization` flag that excluded ALL frames when any single pair was bad. Do not revert to the global flag.
- **Laplacian blend ramp is proportional, not fixed.** `ramp_px = max(20.0, zone_h * 0.12)` ensures the brightness gradient is spread across enough pixels to fall below visual threshold. Do NOT use a fixed ramp width.
- **test12 S11 > S09 is a known structural limitation.** test12 has inter-frame luminance differences of ~30 units between adjacent frames that exceed the adaptive gain cap. Stage 9 temporal median smooths these; Stage 11 per-zone composite preserves them. This is NOT a bug in Stage 11 — it is a fundamental property of the input.

---

## Feather Zones

- **Allow feather zone overlap.** Adjacent feather zones are allowed to overlap. The `num/denom` accumulation in the composite chunk loop handles this correctly by averaging contributions. Do NOT add a boundary-spacing cap that prevents overlap.
- **Cap feathers by natural overlap only.** Use `min(nat_overlap // 2, FEATHER_MAX)`. The `nat_overlap` is the physical overlap between adjacent frames (typically ~1895px for these datasets), not the boundary spacing.
- **Adaptive feather refinement (§1.3A, S12).** After FG registration, `seam_post_diffs[k] < 8.0` → widen feather 1.5× (cap `FEATHER_MAX=300`); `> 16.0` → narrow 0.75× (floor `FEATHER_MIN=80`). This runs after the initial overlap-cap pass and is governed by the `seam_post_diffs` dict; make sure it is initialized to `{}` before the registration loop.
- **Gain-adaptive feather minimum (§1.6B, S22).** `_gain_to_min_feather(gain_diff)` widens `feathers[k]` when adjacent frames have extreme gain mismatch (`gain_diff > 0.267`). Formula: `min(120, max(40, int(gain_diff × 300)))`. This prevents a visibly narrow feather zone from locking in a hard brightness step.
- **The DP seam path governs gain taper, not blend alpha.** `d_seam` (per-column seam path distance) is used for `t_blend` (gain correction taper). The blend alpha `t_lin` must use `d_flat = local_ys - float(y_cut)` (flat horizontal). Never use the seam path for the blend alpha — it causes irregular brightness boundaries.

---

## Alignment

- **Bundle adjustment output must be validated.** After `_bundle_adjust_affine`, check that the sorted `ty` values form a monotonically increasing sequence with roughly equal spacing (within a 3× factor of the median gap). If not, flag alignment failure and fall back to the simple stitch. Do not let broken affines propagate to Stage 9.
- **Never assume frame input order equals scroll order.** The pairwise matcher must detect the scroll direction from the data, not from filename order.
- **RANSAC-style outlier rejection is needed in bundle adjust.** After initial LM solve, compute per-edge residuals. Edges with residual > 3× median should be removed and the system re-solved. This prevents a few bad LoFTR matches from collapsing the entire alignment.
- **Reject near-zero dy matches in `_filter_edges`.** Any pairwise match with `|dy| < 50px` is a near-zero match — either the same frame matched against itself, a wrong-direction match, or a repeated-content false positive. These produce frame clustering (multiple frames at essentially the same canvas y-position) and must be discarded before bundle adjustment.
- **Stage 9 ghosting is always caused by bad affines.** If the temporal render is ghosted, do not debug `_render_median` first — check the affines in `stage08_canvas_info.json`. The rendering logic is correct; overlapping frames mean the canvas geometry is wrong.
- **Diagonal scroll (tx drift) is not yet supported.** Datasets where the camera pans both vertically and horizontally will always fail until `_compute_canvas` in `canvas.py` is updated to use the full affine tx offset. Symptom: simple_stitch is wider than the pipeline panorama and has staircase black borders. Test7 is the canonical example.
- **Check the full affine matrix, not just ty/tx.** test18 has ratio=1.1× with min_gap=327px but catastrophic Stage 9 ghosting because the off-diagonal elements of the affine matrices indicate large rotation. `_validate_affines` must check `|a[0][1]|` and `|a[1][0]|` (off-diagonal); if either exceeds 0.1, the frame has problematic rotation and must be flagged.
- **Detect scroll axis before compositing.** If `ty_range < 0.1 * tx_range`, the source frames represent a horizontal scroll (like test20: tx=0–1857px, ty≈0). Applying vertical strip compositing to a horizontal scroll produces horizontal seam bands. Log a warning and either fall back to `_merge_images_scan_stitch` or switch to horizontal strip mode.
- **Detect co-located duplicate frames.** Before bundle adjustment, reject frame pairs where both `|dy| < min_step` AND `|dx| < min_step`. In `_validate_affines`, flag any dataset where `min_gap == 0` — it has co-located frames that will ghosting the top/bottom strips.
- **⚠️ The pipeline is failing catastrophically on most 94-test cases.** Visual inspection confirms severe horizontal color banding, body-part duplication, and mismatched colors between strips in ~60–80% of ASP-succeeded tests. The CV sharpness metric is completely wrong for evaluating this — hard seam edges inflate the sharpness score. Do NOT use the benchmark "asp_better" verdict as evidence of quality. Only seam_gradient < 5 with a visually confirmed clean image should count as a success.
- **Confirmed good baselines (visually verified, new numbering):** asp_test28 (proper vertical panorama; was test27), asp_test58 (clean extended coverage; was test57). All other claimed "asp_better" tests should be individually verified before use as baselines.
- **Test number shift:** Old tests 25–94 are now numbered 26–95. New asp_test25 and asp_test96 were inserted/appended. Total corpus: 96 tests. Ground truth available for 55 of 96 tests in `data/ground_truth/`.
- **Root cause of failures:** These are animated video scenes, not static scrolling art. Phase correlation measures whole-frame displacement including character animation, not just camera movement. The temporal median requires ≥3 frames per canvas row; with 50px inter-frame steps and 1080px frames, most rows have only 1 frame — no temporal averaging occurs.
- **The dominant fallback cause is min_gap < 50px.** Reducing to 25px adds more "succeeded" tests but does not fix the compositing quality failures in those tests.

---

## Stage 11 Compositing — INTER_LINEAR and has_content

- **Stage 11 uses INTER_LINEAR, not INTER_LANCZOS4.** This is intentional: Lanczos4's negative side-lobes create dark halos at sharp silhouette edges. INTER_LINEAR has no significant ringing. Do NOT change the interpolation flag in the `warpAffine` calls inside `_composite_foreground`.
- **`has_content = src.max(axis=2) > 0` must stay at `> 0`.** Changing to `> 10` was tested and caused a +5.25 regression on test12 by silently dropping legitimate dark content pixels. Spatial analysis confirmed 95% of pixels with 0 < max ≤ 10 in dark frames are interior content, not boundary artifacts. The Lanczos ringing problem exists in Stage 9 (rendering.py, INTER_LANCZOS4) — not Stage 11.
- **Do not use pixel value thresholds to detect frame boundaries.** The correct approach for geometry-based content detection would be to warp a binary mask with INTER_NEAREST. Pixel value thresholds conflate "no content" with "very dark content."

## Rendering (Stage 9)

- **MFSR is disabled in test scripts — not in production.** The `run_pipeline_v2.py` test script skips MFSR because DCT-based MFSR introduces 8×8 block artifacts in flat cel-shaded regions. The production pipeline exposes a GUI toggle. Do not remove the MFSR code; keep it behind the toggle.
- **The rendering gain clamp `(0.88, 1.12)` in `_render_median` is independent** from the LS clamp in `_composite_foreground`. These are different correction passes on different data — do not conflate them.
- **`_FADE_ROWS = 40` in rendering.py** (increased from 20). This smooths the temporal median at each frame's canvas entry/exit. Do not reduce it below 40 without checking for regressions on test19.
- **The `_FADE_ROWS` gate `count_no_i >= 1` must stay.** The fade-out is intentionally skipped for pixels where no other frame is present (exclusive zones). Removing this gate darkens natural scene top/bottom edges in exclusive columns — confirmed to cause +10–110 unit regressions across 8 datasets.

## Seam Metric

The old `max(abs(diff(row_mean_lum)))` metric has two artifact modes: (1) it counts the canvas top/bottom edge as a seam; (2) exclusive-zone columns appearing mid-panorama inflate the metric even when interior content is smooth. Use the corrected metric (see `cache/anime_stitch_pipeline_issues.md §7B.2`): skip rows where fewer than 5% of columns have pixels > 10, trim 5 rows from top/bottom, then take max |diff| of row means.

**test19 "16-unit seam" is largely a metric artifact.** Of the 16 reported units: ~12 come from the coverage change at row 1402 (frame 3's exclusive zone cols 3840–4229 appearing), and ~4 come from a genuine 5-column animation phase seam at cols 2541–2546. Only the latter is a visual artifact, and it requires phase-aware temporal rendering to fix — not a compositing change.

---

## Video Ingestion (Phase 2 — Issue 9)

- **New module: `backend/src/animation/video_ingestion.py`** — `VideoIngestionStream` wraps PyAV (`av.open()`). It exposes `get_frame(idx, full_res=True)`, `get_proxy_frames(stride=5)`, and `decimate_duplicates(mad_threshold=0.01)`. Do NOT use Decord (memory leaks, color deviation) or cv2.VideoCapture (unreliable seeking past GOP boundaries). Use PyAV exclusively.
- **Proxy-first decode.** The proxy stream (I-frame-only at ¼ resolution via `av.seek + decode`) must always run before full-resolution decode. `smart_select_frames()` receives proxy frames; only selected frame indices are decoded at full resolution. This keeps memory under ~100 MB for 300-frame inputs.
- **Hybrid `pipeline.run()` signature.** `AnimeStitchPipeline.run()` accepts `video_path: str | None` alongside `image_paths: List[str] | None`. If `video_path` is provided and `image_paths` is None, the pipeline uses `VideoIngestionStream` for ingestion. If both are provided, `image_paths` are the high-res keyframes and `video_path` is used for tracking only (Phase 9C).
- **No blocking decode on the main thread.** `VideoIngestionStream` frame reads must happen inside `QRunnable`/`QThread` workers if triggered from the GUI. The `VideoIngestionStream` class itself is thread-safe (PyAV containers opened per-thread).
- **Telecine-aware duplicate detection.** `decimate_duplicates()` runs on proxy frames before `smart_select_frames()`. It complements but does not replace `_detect_hold_blocks()` — hold detection operates on already-unique frames.

## Multi-Modal HITL (Phase 2 — Issue 10)

- **No blocking calls in HITL checkpoint dialogs.** GroundingDINO inference and SAM-2 re-propagation during click refinement must run in a `QThread` with progress indication. The dialog's event loop must remain responsive for click capture. Never call model inference from a slot handler directly.
- **Grounding DINO wrapper lives in `backend/src/animation/grounding.py`**, not in `masking.py`. `masking.py` calls into `grounding.py` via `_compute_fg_masks_grounded_sam2(frames, text_prompt, ...)`. No model weight loading in `masking.py` directly.
- **Click-refinement prompts are additive.** Positive clicks (`pos_clicks`) and negative clicks (`neg_clicks`) are accumulated across the session dialog's lifetime; they are all passed to SAM-2 in one `predict()` call per refinement. Do NOT restart SAM-2's state from scratch on each click.
- **Seam exclusion masks are injected at `_build_seam_cost_map()`.** The `exclusion_masks: List[np.ndarray] | None = None` parameter receives GroundingDINO-detected exclusion regions. Each mask pixel gets `cost = 1e6` (hard barrier). The fallback (all-background columns) must still be available when no exclusion masks are provided.
- **`COCOAnnotationBuilder` is side-effect-only.** `data_serialization.py`'s `COCOAnnotationBuilder.save()` must never raise. Write to a temp file then `os.replace()` atomically. Serialization failures must be logged as warnings, never as pipeline-blocking errors.
- **RLHF reward model is read-only during inference.** The `StitchRewardModel.predict()` call in `bench_anime_stitch.py` must never modify model weights. Fine-tuning runs are offline-only (separate `scripts/finetune_*.py` scripts). Never call `.train()` during a benchmark or pipeline run.

## Code Quality

- **No Qt imports in `backend/src/animation/`.** These modules must be importable from headless scripts.
- **Do not change `_composite_foreground`'s public signature.** It is called from the pipeline, the GUI worker, and test scripts. All changes must be backward-compatible.
- **Print diagnostic output for every meaningful decision.** The `[Stitch]` prefix print statements (boundary positions, feather sizes, LS gains, DP path ranges) are essential for diagnosing issues in logs. Do not remove or silence them.
- **Pipeline constants live in `backend/src/constants/animation.py`**, not at the top of individual module files. Key constants: `FEATHER_MAX=300`, `FEATHER_MIN=80`, `FEATHER_TABLE`, `CANVAS_MAX_DIM`, `MIN_EXPECTED_STEP=25`, `SPATIAL_DEDUP_PX=25`, `NEAR_DUP_LUMA_THRESH=3.0`. Import them with `from backend.src.constants.animation import ...`. Do not duplicate magic numbers inline.
- **Runtime overrides via env vars or TOML (§1.8A, S27).** Any constant controlled by an env var (e.g. `ASP_NEAR_DUP_LUMA`, `ASP_HOLD_THRESHOLD`, `ASP_SP_SOFT_PX`) can also be set via `asp_config.toml` using `load_asp_config()` from `backend.src.animation.config`. File values use `setdefault` so explicit env vars always win.
- **`scans_frames` is synced after every frame-drop pass (§1.9A, S28).** When any dedup step (pre-stage-5 luma dedup or post-stage-6 spatial dedup) drops frames from `frames`, `scans_frames` must be updated with the same `keep_idx`. This ensures all SCANS fallback paths receive the same frame subset the main pipeline has committed to. The spatial dedup now uses `_spatial_dedup_frames()` (module-level, in `pipeline.py`) which always syncs `scans_frames`.
