# ASP §1.3/§3.2 GraphCut Seam Post-Mortem — Second Measurement (2026-07-27)

**Verdict: still rejected. `ASP_GRAPHCUT_SEAM` stays default OFF.** This is a
post-mortem, not a roadmap item — see `moon/roadmaps/asp.md` §1.3 for the
one-line pointer back here.

## What was attempted

The first measurement (2026-07-09 trim baseline) found seam_visibility 20-80
vs the DP path's 2-16, attributed to three wiring gaps: hard ownership cut,
±8px feather, no per-seam photometric correction. This session fixed all
three:

1. **Distance-transform feathering** (`_feather_gc_boundaries` rewritten):
   replaced the per-column "last owned row" linear ramp — a poor
   approximation for a GraphCut boundary that can meander in any direction —
   with `cv2.distanceTransform`-based signed-distance blending that follows
   the true 2D ownership boundary shape.
2. **Local per-boundary blocks-gain correction**: within each blend band's
   bounding box, `_blocks_gain_compensate` (the same helper the DP path
   already relies on) corrects frame i+1 to match frame i's local photometry
   before blending, addressing "no per-seam photometric correction."
3. **Widened feather width**: `ASP_GC_FEATHER_PX` default 8→96px, bringing it
   onto the same scale as the DP path's typical 100-300px feathers (an
   order-of-magnitude mismatch was itself a plausible major contributor to
   the original sv 20-80 finding).

**Not attempted**: the anime edge-cost rule (`cost ∝ 1 − edge_strength`, so
seams follow line-art). `cv::detail::GraphCutSeamFinder` uses OpenCV's fixed
`COST_COLOR_GRAD` internally with no exposed hook for a custom per-pixel cost
weighting at the Python binding level — implementing this would mean
reimplementing the min-cut algorithm in `base/src/animation/seam.cpp` from
scratch, well beyond a "fix the wiring" scope. Flagged as a separate,
larger, uncertain-value research item if GraphCut is ever revisited.

## Measurement (5-test verify, `asp_test04/08/09/27/57`)

Comparing the DP path's real (gate-passing or gate-rejected) composite
quality against GraphCut's real attempted composite quality (both taken from
`fallback_reason` gate numbers where a fallback occurred, since `metrics_asp`
on a fallback test reflects the SCANS image, not the rejected ASP attempt):

| test | DP real sv/gate | GraphCut real sv/gate | verdict |
|------|------------------|------------------------|---------|
| test04 | sb=35.8 (barely failed composite_gate_sb) | **passed the gate** but sharpness=3150 (vs DP-fallback's 36.6) — see below | new defect, not a win |
| test08 | sv=28.15 (passed) | sv=90.0 (failed) | **worse** |
| test09 | sv=39.8 (already failed) | sv=55.1 (failed) | **worse** |
| test27 | sv=16.14 (passed) | sc=23.7/sb=61.7 (failed, different gate) | **worse** |
| test57 | sv=11.03 (passed) | sv=75.6 (failed) | **worse** |

4 of 5 tests got measurably worse on the real (pre-fallback) seam quality.
The one test that passed the composite gate (test04) produced a **visibly
corrupted image** — dense, near-periodic horizontal scan-line artifacts
across almost the entire canvas, not localized to a seam region. The
Laplacian-variance sharpness metric scored this as "sharp" (3150 vs a normal
~40-150 range) because it's fooled by fine periodic noise — a clear case of
an automated metric disagreeing with a two-second look at the image.

## Root cause of the new defect (hypothesis, not fully confirmed)

`_execute_graphcut_composite` runs `GraphCutSeamFinder` on a heavily
downscaled proxy (`_GC_SEAM_EST_MPIX = 0.4`, giving ~0.29-0.33× scale on
these canvases — see the `est scale=` log line) and upscales the resulting
ownership masks back to full resolution with `cv2.INTER_NEAREST`. On flat,
low-texture cel-shaded anime backgrounds, `COST_COLOR_GRAD` gets very little
gradient signal to distinguish a good cut path from a bad one, and the
min-cut solver plausibly degenerates into a fragmented partition — many thin
alternating ownership bands rather than one clean boundary. Nearest-neighbor
upscaling preserves that fragmentation at full resolution, and this
session's *wider* 96px feather then blends aggressively across many of those
thin alternating bands in immediate succession, producing exactly the kind
of repeating scan-line/moiré pattern visible in `asp_test04`'s output.

This is architectural, not a wiring bug: no amount of feathering or local
gain correction fixes a fragmented input partition. Two possible real fixes,
neither attempted here (out of "fix the wiring" scope):
- Run seam estimation at a less-aggressive downscale (cost: the roadmap's
  own comment notes full-resolution min-cut is "O(hours) on tall canvases" —
  needs a real runtime/quality tradeoff study, not a quick change).
- Post-process the raw ownership masks (morphological open/close, or
  keep-largest-connected-component per frame) to remove sub-pixel-scale
  fragments before feathering — a plausible next experiment, but a new one,
  not validated here.

## Disposition

- `ASP_GRAPHCUT_SEAM` stays default OFF (it always was).
- The distance-transform feathering + local gain correction code is kept in
  `compositing.py` — it's strictly gated behind the flag, harmless to the
  default path (confirmed via a flag-OFF 5-test verify showing zero change),
  and is a more correct primitive than the old per-column ramp even though it
  didn't solve the aggregate problem. A future attempt at the fragmentation
  fix would build on this rather than needing to redo it.
- DSeam (the report-flagged fast alternative) was not evaluated — moot until
  the fragmentation issue is understood, since DSeam has the same
  low-res-estimation architecture question.
- **Recommendation: do not revisit GraphCut again without first addressing
  the ownership-fragmentation hypothesis above** — the wiring gaps
  identified in the first post-mortem are fixed, but a new, deeper, and more
  severe problem surfaced by fixing them.
