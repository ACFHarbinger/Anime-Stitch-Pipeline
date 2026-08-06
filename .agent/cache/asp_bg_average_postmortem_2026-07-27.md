# ASP §2.5 Overmix-style Background Averaging Post-Mortem — Second Measurement (2026-07-27)

**Verdict: still rejected. `ASP_BG_AVERAGE` stays default OFF.** This is a
post-mortem, not a roadmap item — see `moon/roadmaps/asp.md` §2.5 for the
one-line pointer back here.

## What was attempted

The first measurement (S220) found visible horizontal strip-banding and
attributed it to an abrupt switch between pure median (count==2) and pure
mean (count≥3) at the geographic boundary where confirmed-background sample
count changes. This session fixed that specific issue:

- `_render_median`'s Case-2 block (`rendering.py`) now computes a blend
  weight that ramps linearly from 0 at count==2 to 1 at
  count==`ASP_BG_AVERAGE_FULL_AT` (new env var, default 5), instead of
  switching abruptly at count==3. No more single-pixel-of-count value
  discontinuity.
- `config.py` registers `ASP_BG_AVERAGE_FULL_AT` in the schema and the
  rendering `_DUMP_SECTIONS` list.
- 670 animation tests still green (`pytest backend/test/animation/
  --skip-gpu`); the smoothing is a pure refinement of the existing gated
  code path.

## Measurement

Re-ran the 5-test verify (`asp_test04/08/09/27/57`) with `ASP_BG_AVERAGE=1`
and the smoothing fix in place. `asp_test04` — the test that most clearly
showed banding in S220 — still shows severe visible corruption in its
background region, confirmed by directly viewing the composite (not
inferred from metrics). Isolating the cause:

1. Ran `asp_test04` alone with `ASP_BG_AVERAGE` unset (default OFF): clean,
   coherent composite, zero banding.
2. Ran it again with `ASP_BG_AVERAGE=1` (smoothing fix active): same region
   shows a patchwork of solid-color ~30px blocks with jarring color
   transitions between them — visually distinct from the diffuse
   strip-banding described in S220.
3. Cropped and compared both outputs at the same canvas coordinates: the
   affected region corresponds to a small rectangular area in the source
   frames carrying fine block-structured overlay content (present in this
   particular source clip). In the OFF composite, the median picks one
   frame's block pattern cleanly, so it renders as a single self-consistent
   block grid. In the ON composite, the mean blends multiple frames' block
   patterns together — and because each frame's block grid shifts slightly
   relative to the others under the frame-to-frame warp, averaging produces
   a jarring multi-colored patchwork rather than one coherent grid.

This is a **different root cause** from the S220 hypothesis. It is not
about the abruptness of the mean/median transition — even a partial
(~30-40%) blend weight is enough to visibly corrupt this content, because
averaging misaligned block-structured overlays doesn't average out to
something reasonable the way averaging photometric sensor noise does; the
blocks are a rigid grid uncorrelated with the underlying scene content, and
warp misalignment between frames means "the same" block in two frames maps
to different canvas pixels.

## Why the smoothing fix didn't help

The smoothing fix only changes *how gradually* the blend weight changes
across the count==2 → count≥`FULL_AT` transition. It does nothing to
address *whether blending itself is safe* at a given canvas location. Any
non-zero blend weight over block-structured, warp-misaligned content will
show some degree of this artifact — smoothing reduces the amplitude of nothing
here, because the defect isn't a discontinuity artifact, it's a content
artifact.

## Disposition

- `ASP_BG_AVERAGE` stays default OFF (it always was).
- The smoothing fix (`ASP_BG_AVERAGE_FULL_AT`, ramped blend weight) is kept
  — it's strictly gated behind the flag, harmless to the default path, and
  is a more principled primitive than the old abrupt switch even though it
  doesn't solve the aggregate problem. A future attempt would build on this
  rather than needing to redo it.
- **Recommendation: do not revisit this feature again without a way to
  detect or exclude fine block-structured overlay regions from mean
  blending** (e.g. per-pixel local-variance/edge-density gating, or simply
  restricting averaging to canvas regions where all contributing frames'
  content agrees closely after warp, independent of raw sample count) —
  the count-based abruptness identified in S220 is fixed, but the deeper
  problem (mean blending assumes clean photometric noise, not
  warp-misaligned structured content) was never actually about the count
  boundary at all.
