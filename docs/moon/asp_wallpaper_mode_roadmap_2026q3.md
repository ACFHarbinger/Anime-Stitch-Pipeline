# ASP Wallpaper Mode Roadmap (2026Q3)

**Status:** Locked, ready for Slice 1 implementation.
**Origin:** Brainstormed 2026-08-18 by the full team (Claude, Gemini,
deepseek, opencode) via `.agent/bus/2026-08-18.md`, triggered by a
4-way independent analysis of `asp_test97`. Confirmed with Harbinger.

## Relationship to `ROADMAP.md`

Additive, not a replacement. `ROADMAP.md`'s benchmarking mission (match/
exceed the OpenCV Stitcher — "SCANS") stays primary; `asp_evaluations_*.json`
stays the human-rating label pipeline, now also used to validate wallpaper
mode via ad-hoc team visual review (no new rubric for Slice 1).

## Diagnosis (asp_test97, 4-way confirmed)

- Motion tracking is not the failure — clean pan/walk tracking across every
  independent read (deepseek: 90-95% inlier rate, all 89 frame-pairs).
- Root bug: the canvas finalize/crop step crops to the *intersection* of
  frame footprints (maximizes fill%), which clips a frame-dominating subject
  instead of framing them.
- Camera is static in asp_test97; the character walks (opencode, phase-
  correlation on background vs. full-frame) — corrects an earlier
  "diagonal pan" misreading from the visual-only passes.
- Subject is rigid in-frame: 2.4-3.1% residual motion after alignment
  (deepseek). This is a compositing problem, not a motion-blending one.
- Hugin's apparent edge on this test case is mostly "doesn't discard pan/
  walk extent," not superior stitching — the finalize-step fix below should
  narrow the real Hugin-only-wins set considerably (re-measure post-Slice-1,
  see Deferred).

## Architecture: Hero-Cel + Background Plate Compositing

(opencode's proposal, cross-reviewed and endorsed by Gemini, confirmed with
Harbinger — supersedes the earlier bbox-union-crop-only framing once
Harbinger confirmed a *coherent figure* is the most important output
property.)

1. **Hero frame selection.** Score every candidate frame:
   `S_f = w1·Area_norm(mask) + w2·LaplacianVariance(I⊙mask) + w3·Symmetry(mask) − w4·BorderIntersection(mask)`
   (Gemini). Heavy `BorderIntersection` penalty avoids picking a frame where
   the figure is truncated by the frame edge. Auto-select the top-scoring
   frame as hero; GUI thumbnail scrubber lets the artist override the pose
   before baking.
2. **Hero-cel extraction.** FG-mask the hero frame (existing machinery) →
   figure + alpha matte.
3. **Background plate.** Temporal median aggregation across the non-hero
   frames, excluding the hero-cel footprint, fills the hero-shaped hole;
   classical inpainting (Telea / Poisson seamless clone) smooths edges.
4. **Composite.** Rigid-place the hero-cel on the plate at its **original
   registered position** — Harbinger's call: preserve the character's
   natural scene placement, do not reposition to thirds/center. Subject
   residual motion is only 2.4-3.1%, so this is a rigid placement, not a
   warp/tracking problem.
5. **Aspect framer.** User-selected aspect (16:9 / 9:16 / 21:9). Hard
   constraint: fully contain the hero figure. Extend background (inpaint)
   to fill the rest of the target canvas around the figure's natural
   position — not a repositioning objective.

**Multi-character:** multi-cel compositing — connected-component contours
with temporal IoU continuity; subjects of comparable area (>35% each) all
get composited rather than the smaller one being dropped.

**Inpainting policy (hybrid, tiered):**
- Tier 1 (default, <500ms budget): temporal median + classical Telea/
  Poisson.
- Tier 2 (on-demand): local generative diffusion outpaint (SD3/ComfyUI)
  when the target-aspect void exceeds 10% of canvas; hard cap 25% of
  canvas.

## Module breakdown

`submodules/ASP/backend/src/rendering/wallpaper/`:
- `_hero_selector.py` — `S_f` ranking + hero-cel + alpha matte extraction.
- `_plate_builder.py` — temporal median plate + exclusion mask + classical
  inpainting; also owns the joint gain-compensation solve (see Slice 1).
- `_cel_compositor.py` — rigid anchor registration + Poisson blending
  (`cv2.seamlessClone`).
- `_aspect_framer.py` — aspect solver (16:9/9:16/21:9), background
  extension around the figure's natural position.
- `manager.py` / `wallpaper_pipeline.py` — orchestrator, `ASP_MODE=wallpaper`,
  registers its stages into the existing `PipelineSession` protocol
  (M1a/b/c, `backend/src/core/pipeline/session.py`) rather than a parallel
  runner.

**Entry points:** `devtool wallpaper <clip> [--aspect 16:9|9:16|21:9]
[--quality fast|balanced|max] [--estimate]`; a GUI wallpaper export surface
adjacent to the Benchmarks plugin, with the hero-frame thumbnail scrubber.

## Slice 1 (ready to implement)

`_hero_selector.py` + `_plate_builder.py` + `_cel_compositor.py` +
`_aspect_framer.py` → hero-figure composite at a user-selected aspect,
Tier-1 hybrid inpainting, GUI override scrubber.

Also in scope for this slice: **joint least-squares gain compensation**
across plate-contributing frames (the standard Brown & Lowe panorama
approach, deepseek's proposal) — fixes the measured bug where `asp_test97`
trusted frame 0's gain estimate despite it having almost no clean
background to sample (`gains.png`). Lives in `_plate_builder.py`; small,
directly motivated by a measured defect, not speculative.

## Deferred (Slice 2+, re-measure after Slice 1 lands)

- **Fallback routing gate → Hugin**, extending the existing `dy_cv`→SCANS
  pattern. Re-run the metrics-correlation routing study *after* Slice 1 —
  much of Hugin's edge on `asp_test97` was the crop bug this slice fixes,
  so the real Hugin-only-wins set (wide-baseline/near-projection-
  singularity pans, severe lighting gradients) is likely narrower than it
  looks today.
- ASP contributes pre-processing (gain normalization) and post-processing
  (seam healing, rectification) around whichever engine a case routes to —
  never a bare hand-off.
- **Routing classifier** (metrics → engine), supervised, using the existing
  `asp_evaluations_*.json` label pipeline; a contextual bandit once there's
  enough labeled routing history.
- **Plate-frame coverage optimization** — which non-hero frames feed the
  plate is a set-cover/facility-location problem distinct from hero
  *selection* (now a scoring function, not set-cover, per Harbinger's
  answer to pivot frame selection). Worth revisiting once Slice 1's plate
  builder has real usage data.
- **Bayesian hyperparameter tuning** against the benchmark corpus (gate
  thresholds, `ASP_HOLD_BG_SUB`, ...) — orthogonal to the pivot, lower
  priority.

## Discipline note

`ROADMAP.md` §5.1 documents why a prior PSO/DRL/RLHF stack was deleted in
the S200 "great trim": no coherence metric to optimize against, no A/B
discipline, shipped default-ON without measurement. Nothing in this
roadmap repeats that category — crop-window solving and joint gain
compensation are small, exact/closed-form or grid-searchable problems, not
learned policies needing reward models or training data. Anything that
*does* cross into learned/trained territory later (the routing classifier/
bandit, Bayesian hyperparameter tuning) must follow `ROADMAP.md`'s Ground
Rules: one change → one benchmark → keep or revert, human coherence rating
as the success criterion, never a proxy metric alone.
