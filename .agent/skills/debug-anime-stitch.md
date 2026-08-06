---
description: Diagnostic checklist for AnimeStitchPipeline failures — seam bands, Stage 9 ghosting, alignment failure, and photometric artifacts.
---

You are diagnosing a failure in `AnimeStitchPipeline` (`backend/src/animation/`). Use this checklist to identify the root cause before making any changes.

---

## Step 0 — Identify the failure category

Run the pipeline and examine the output image and printed log. Map the symptom to one of the four categories below.

| Symptom | Category |
|---------|----------|
| Visible horizontal bright/dark bands or stripes across the output | **A — Stage 11 seam** |
| Ghosted / doubled character outlines, semi-transparent overlay | **B — Stage 9 ghosting** |
| Frames in wrong vertical order, upside-down sections, large content gaps | **C — Alignment failure** |
| 8×8 grid pattern across flat-color regions | **D — MFSR block artifact** |
| Image crops to wrong region or has large black borders | **E — Canvas geometry** |
| Pipeline panorama is substantially wider in simple_stitch, staircase black borders | **F — Diagonal scroll (tx drift)** |
| Good ty/tx values but catastrophic Stage 9 ghosting | **G — Affine rotation/scale mismatch** |
| Wide landscape panorama (wider than tall) with horizontal seam bands | **H — Pure horizontal scroll** |

**Critical triage rule:** If the output shows ghosting (B), ALWAYS check the affines first (Category C/G). Do not debug `_render_median` until you have confirmed:
1. max_gap/median_gap ratio < 3×
2. min_gap > 25px (no co-located frames — threshold was lowered from 50px in S11)
3. Off-diagonal affine elements < 0.1 (no rotation)

Exception confirmed: test18 has ratio=1.1× with min_gap=327px but catastrophic ghosting from affine rotation (Category G).

---

## Category A — Stage 11 Seam / Brightness Bands

### A1 — Isolate Stage 11

Check whether Stage 9 output is already clean:

```bash
# View stage09 temporal render
python3 -c "
from PIL import Image
img = Image.open('data/asp_test1/output/panorama_stages/stage09_temporal_render.png')
img.show()
"
```

If stage09 is clean and the band appears only in stage11 (final output), the issue is in `_composite_foreground` (`backend/src/animation/compositing.py`).

### A2 — Check gain corrections

Look at the printed log for the compositing run:

```
[Stitch]   LS gains: F7=X F6=X ...
[Stitch]     Boundary B1 (frames 6/5): ... gain=[X,X,X]
[Stitch]     Seam check B1 ...  ΔB=X ΔG=X ΔR=X
```

**Red flags:**
- LS gains outside adaptive clamp range — `clamp_width = 0.26 − 0.12 × (ref_lum/255)`; check `_adaptive_gain_clamp` in `compositing.py`
- `gain=[0.85, 0.83, 0.83]` style — per-zone gain correction is active and overcorrecting; set `gain_seam = np.ones(3)`
- `Seam ramp B1 ...` lines — post-composite ramp is active inside feather zones; disable `_apply_canvas_seam_correction`

### A3 — Check feather overlap

```python
# In the log, look for:
# [Stitch]   Feathers (gap-capped): B0=Xpx B1=Xpx ...
# Then check boundary spacing:
# B0 at y=988, B1 at y=1157 → gap=169px
# If feathers are 84px (half of 169), zones are non-overlapping → creates two sequential transitions
```

If `B0=84px B1=84px` and the gap is 169px, the boundary-spacing cap is active. Remove it — feathers should be capped by `nat_overlap//2`, not boundary spacing.

### A4 — Verify blend alpha is flat horizontal

In `compositing.py`, find the composite chunk loop. Confirm `t_lin` is computed as:

```python
d_flat = local_ys - float(y_cut)  # flat horizontal
t_lin = np.clip((d_flat + zone_half_f) / (2.0 * zone_half_f), 0.0, 1.0)
```

If it uses `d_seam` instead of `d_flat`, the per-column seam path is bending the brightness ramp horizontally — replace with `d_flat`.

### A5 — Verify zone_half_f is full zone width

Confirm:
```python
zone_half_f = max(1, (y1_f - y0_f - 1) // 2)
```
Not a fixed constant like `SEAM_THIN_HF = 8`. A fixed 8px blend inside a 500px zone causes a visible 16px seam line.

---

## Category B — Stage 9 Ghosting / Temporal Render Failure

### B1 — Check source frames

```bash
# View BaSiC-corrected frames (stage03) or normalised frames (stage02)
ls data/asp_test3/output/panorama_stages/stage03_basic_corrected_frame*.png
# Open a few and check they look correct
```

If any frame is all-black, nearly-black, or inverted, BaSiC correction failed for that frame. The rendering will collapse with a black strip in the output.

### B2 — Check BiRefNet masks

```bash
ls data/asp_test3/output/panorama_stages/stage04_bgmask_frame*.png
```

Open several masks. Background pixels should be white (255), foreground (character) pixels black (0). An inverted mask (all-white or all-black) causes LS normalization and boundary search to use wrong pixels.

### B3 — Check affines before rendering

```bash
python3 -c "
import json
with open('/path/to/stage08_canvas_info.json') as f:
    d = json.load(f)
aff = d['affines_final']
tys = sorted([(i, aff[i][1][2]) for i in range(len(aff))], key=lambda x: x[1])
for rank, (fi, ty) in enumerate(tys):
    print(f'rank{rank}: frame{fi} ty={ty:.1f}')
"
```

If the `ty` values are not monotonically increasing with roughly equal spacing, alignment failed (see Category C) and must be fixed before Stage 9.

### B4 — Rendering gain clamp

In `backend/src/animation/rendering.py`, `_render_median` applies a per-pixel gain correction clamped to `(0.88, 1.12)`. If the scene has large natural brightness variation, this clamp may be insufficient or excessive. Check whether ghosting disappears when the clamp is widened to `(0.75, 1.35)` for diagnosis.

---

## Category C — Alignment Failure

### C1 — Inspect the affines

```bash
python3 -c "
import json, numpy as np
with open('/path/to/stage08_canvas_info.json') as f:
    d = json.load(f)
aff = d['affines_final']
tys = [aff[i][1][2] for i in range(len(aff))]
tys_sorted = sorted(tys)
gaps = np.diff(tys_sorted)
print('ty values (sorted):', [f'{t:.0f}' for t in tys_sorted])
print('gaps:', [f'{g:.0f}' for g in gaps])
print('median gap:', f'{np.median(gaps):.0f}')
print('max/median ratio:', f'{gaps.max()/np.median(gaps):.1f}')
"
```

**Red flags:**
- Any gap > 5× the median gap — frame is misplaced
- Any gap < 10px — two frames are essentially at the same position (degenerate)
- `ty` sequence is not monotonic after sorting by frame index — wrong frame ordering

### C2 — Check edge list

In `bundle_adjust.py`, add temporary debug output to print all edges before and after the LM solve:

```python
print("Edges before bundle adjust:")
for e in edges:
    print(f"  {e['src']} -> {e['dst']}: dx={e['dx']:.1f} dy={e['dy']:.1f}")
```

Look for edges with anomalous `dy` values:
- Expected: `dy` for a vertical scroll is ~200–400px per frame step, all negative (going down)
- Wrong-direction: positive `dy` for a downward scroll, or magnitude far from the mode

### C2b — Check for near-zero match clustering

A more subtle failure than non-monotonic ordering: many frames placed within 5–30px of each other. Symptom: `ty` values are sorted correctly but many gaps are < 50px.

```bash
python3 -c "
import json, numpy as np
with open('/path/to/stage08_canvas_info.json') as f:
    d = json.load(f)
aff = d['affines_final']
tys = sorted(aff[i][1][2] for i in range(len(aff)))
gaps = np.diff(tys)
n_clustered = (gaps < 50).sum()
print(f'Gaps < 50px: {n_clustered} of {len(gaps)}')
print('Small gaps:', gaps[gaps < 50])
"
```
If more than 1–2 gaps are < 50px, LoFTR produced near-zero dy matches (same-scene or repeated-content frame pairs). Fix: add `|dy| < 50px` rejection to `_filter_edges` in `pipeline.py`.

### C3 — Verify `_filter_edges` is catching wrong-direction matches

In `pipeline.py`, `_filter_edges` uses velocity consistency. Check its output count:

```
# From pipeline log:
# X edges after filtering
```

If the count before filtering is close to the count after, `_filter_edges` is not rejecting bad edges. Check the velocity threshold and whether the majority of edges are correct.

### C4 — Quick fix: robust bundle adjust

Add residual-based outlier rejection after the initial LM solve in `bundle_adjust.py`:

```python
# After initial solve, compute per-edge residuals
residuals = []
for e in edges:
    predicted_dy = affines[e['dst']][1,2] - affines[e['src']][1,2]
    residuals.append(abs(predicted_dy - e['dy']))
median_res = np.median(residuals)
# Remove edges with residual > 3x median
edges_clean = [e for e, r in zip(edges, residuals) if r <= 3 * median_res]
# Re-solve with clean edges
```

---

## Category D — MFSR Block Artifacts

**Diagnosis:** Visible 8×8 regular grid pattern across the entire image, especially on flat-color regions (skin, backgrounds).

**Fix:** Disable MFSR in the pipeline call. In `run_pipeline_v2.py` and test scripts, MFSR is already skipped. In the GUI, the "MFSR" toggle should default to OFF.

**If MFSR must be used:** The DCT-based MFSR in `backend/src/animation/mfsr/` uses 8×8 block DCT operations. For anime cel-shading, a spatial-domain SR model (e.g., RealESRGAN from `backend/src/models/`) produces far better results.

---

## Category E — Canvas Geometry / Crop Issues

### E1 — Check `_compute_canvas`

```bash
python3 -c "
import json
with open('/path/to/stage08_canvas_info.json') as f:
    d = json.load(f)
print('Canvas:', d['canvas_w'], 'x', d['canvas_h'])
aff = d['affines_final']
# Check all frames fit within canvas
for i, a in enumerate(aff):
    print(f'frame{i}: ty={a[1][2]:.0f}')
"
```

If `canvas_h` is much larger than expected (e.g., 2× the sum of frame heights), there are likely sign errors in the translation — some frames have negative `ty`, pushing the canvas origin far up.

### E2 — Check `_crop_to_valid`

`_crop_to_valid` finds the bounding box of non-zero pixels. If all frames share a common horizontal offset but the canvas was computed with the wrong origin, the valid region may not span the full expected height.

---

## Category F — Diagonal Scroll (tx drift)

**Diagnosis:** Pipeline panorama is narrower than the simple stitch (e.g., 4603 vs 5530px for test7). Simple stitch shows staircase-step black borders at all four edges — indicating the frames have both vertical (ty) and horizontal (tx) scroll components.

**Root cause:** `_compute_canvas` in `canvas.py` uses only the ty component of each affine, placing all frames on the same x-column. Horizontal camera drift is discarded, making correct alignment geometrically impossible.

**Fix required:** Update `_compute_canvas` to use the full `(tx, ty)` affine translation when placing frames, then update the compositing stage to handle 2D strip regions rather than pure horizontal bands.

**Verification:**
```bash
python3 -c "
import json, numpy as np
with open('/path/to/stage08_canvas_info.json') as f:
    d = json.load(f)
aff = d['affines_final']
txs = [aff[i][0][2] for i in range(len(aff))]
print('tx values:', [f'{t:.0f}' for t in txs])
print('tx range:', f'{min(txs):.0f} to {max(txs):.0f}')
"
```
If tx range > 200px, the dataset has significant horizontal drift and requires diagonal-scroll support. Test7 canonical example: tx range expected ~500px.

---

## Category G — Affine Rotation/Scale Mismatch

**Diagnosis:** Good ty/tx translation values (ratio < 2×, min_gap > 50px) but catastrophic Stage 9 ghosting. The pipeline health check passes but the actual frame warps are wrong because the affine matrices have large off-diagonal elements (rotation) or diagonal deviations from 1.0 (scale).

**Canonical example:** test18 — ratio=1.1×, min_gap=327px, stage09 catastrophically ghosted.

**Diagnostic:**
```python
import json
with open('/path/to/stage08_canvas_info.json') as f:
    d = json.load(f)
for i, a in enumerate(d['affines_final']):
    rot = (abs(a[0][1]) + abs(a[1][0])) / 2
    scale_dev = abs(a[0][0] - 1.0) + abs(a[1][1] - 1.0)
    flag = ' *** ROTATION WARNING' if rot > 0.05 else ''
    flag += ' *** SCALE WARNING' if scale_dev > 0.05 else ''
    print(f'frame{i}: rot={rot:.3f} scale_dev={scale_dev:.3f}{flag}')
```

**Fix:** Extend `_validate_affines` to reject frames where `|a[0][1]| > 0.1` or `|a[1][0]| > 0.1`. These frames have rotation that `_compute_canvas` ignores, causing wide overlap regions even though translation centres are correct.

---

## Category H — Pure Horizontal Scroll

**Diagnosis:** Pipeline panorama is wider than tall (landscape format) with multiple horizontal seam bands despite apparently clean alignment. Affines show ty≈0 for all frames and tx varying from 0 to 1000+px.

**Canonical example:** test20 — ty: [0,0,4,2,2,3,0], tx: [1857,1482,1130,942,688,315,0].

**Diagnostic:**
```python
import json, numpy as np
with open('/path/to/stage08_canvas_info.json') as f:
    d = json.load(f)
aff = d['affines_final']
tys = [aff[i][1][2] for i in range(len(aff))]
txs = [aff[i][0][2] for i in range(len(aff))]
ty_range = max(tys) - min(tys)
tx_range = max(txs) - min(txs)
print(f'ty_range={ty_range:.0f}  tx_range={tx_range:.0f}  ratio={tx_range/(ty_range+1):.1f}')
if tx_range > 10 * ty_range:
    print('*** HORIZONTAL SCROLL DETECTED — vertical strip model will fail')
```

**Fix (temporary):** Fall back to `_merge_images_scan_stitch` when horizontal scroll is detected.
**Fix (proper):** Implement horizontal strip mode — sort frames by tx, assign canvas columns to frames, run DP seam cut along vertical lines.

---

## Unit Test Suite

Before and after any fix, run the automated test suite. It covers all issue categories with synthetic data — no GPU required:

```bash
source .venv/bin/activate
pytest backend/test/animation/ -q                          # 498 tests (~30s)
pytest backend/test/animation/test_filter_edges.py -v      # Category C — alignment
pytest backend/test/animation/test_bundle_adjust.py -v     # Category C — clustering
pytest backend/test/animation/test_canvas.py -v            # Category E — canvas/crop
pytest backend/test/animation/test_affine_validation.py -v # Category C/G — validate
pytest backend/test/animation/test_compositing.py -v       # Category A — seam/gain
pytest backend/test/animation/test_rendering.py -v         # Category B — ghosting
pytest backend/test/animation/test_frame_selection.py -v   # hold/near-dup filtering
pytest backend/test/animation/test_config.py -v            # §1.8A TOML config
```

The suite documents both correct behavior (assertions that must pass) and known bugs (assertions that document the broken state, marked in the test docstring). When you fix a documented bug, find the corresponding test and update the assertion to verify the correct behavior.

Shared test helpers live in `backend/test/conftest.py`: `make_frame`, `make_edge`, `make_translation_affine`, `make_rotation_affine`, `compute_ty_gaps`.

---

## Diagnostic Quick Reference

```bash
# Check all stage outputs exist and are non-zero
find /path/to/panorama_stages/ -name "*.png" -empty

# Print LS gains from a run
grep "LS gains" /tmp/pipeline_run.log

# Check boundary positions and feather widths
grep "Boundary\|Feathers" /tmp/pipeline_run.log

# Check DP seam path ranges
grep "DP path" /tmp/pipeline_run.log

# Run compositing only (fast, no GPU)
source .venv/bin/activate && python3 archive/run_pipeline_v2.py 2>&1 | tee /tmp/pipeline_run.log
```
