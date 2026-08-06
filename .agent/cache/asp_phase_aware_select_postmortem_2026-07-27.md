# ASP §2.4 Phase-Aware Frame Selection Post-Mortem (2026-07-27)

**Verdict: rejected. `ASP_PHASE_AWARE_SELECT` stays default OFF.** This is a
post-mortem, not a roadmap item — see `moon/roadmaps/asp.md` §2.4 for the
one-line pointer back here.

## What was attempted

`smart_select_frames`'s Pass 2 (pose-consistent local refinement, itself
gated behind `ASP_POSE_WINDOW_PX>0` and off by default) already penalises a
candidate that stays in the *same hold block* as the previous anchor
(`_SAME_HOLD_PENALTY`) — identical pose, zero registration benefit. §2.4
asked for the coarser, opposite-direction bias: penalise a candidate that
would *cross into a different animation phase* than the previous anchor,
since cross-phase pairs are harder to align/composite than within-phase
ones (§2.3 already refuses to midpoint-warp across a phase boundary at
composite time).

Implementation (`backend/src/animation/ingestion/frame_selection.py`):

- Factored the §2.2 change-point clustering math out of
  `detect_animation_phases` into `_phase_ids_from_hashes(hashes, z_thresh)`
  so it could be reused without a redundant thumbnail reload.
- In Pass 2, when `ASP_PHASE_AWARE_SELECT=1`, computed `_cand_phase_ids` on
  the *candidate* pool (this pass's own `thumbs`, post pre-filters, post
  optional hold-averaging — not the final selected set, which doesn't exist
  yet at selection time) and added `ASP_PHASE_CROSS_PENALTY` (default 0.05,
  matching `_SAME_HOLD_PENALTY`'s magnitude) to a candidate's tie-break
  score when its candidate-phase differs from the anchor's.
- Registered both env vars in `config.py`'s schema and TOML dump section.
- 670 animation tests still green (`pytest backend/test/animation/
  --skip-gpu`).

## Measurement pitfall (caught before drawing any conclusion)

The first 5-test run compared `ASP_PHASE_AWARE_SELECT=1` against the
default configuration and found **zero measurable difference** — every
metric byte-identical across all 5 tests. This looked like "no effect" but
was actually a broken comparison: Pass 2 itself is gated behind
`ASP_POSE_WINDOW_PX`, which **defaults to 0** (disabled). The phase-cross
penalty lives inside Pass 2's scoring loop, so with Pass 2 off, the new
code never executes at all — the comparison was OFF-vs-OFF, not measuring
anything. Re-ran with `ASP_POSE_WINDOW_PX=80` (Pass 2's own documented
value) set in *both* arms, isolating phase-awareness as the only variable.
`[PhaseSelect]` log lines confirmed the code path fired (3-15 candidate
phases detected per test) and a `diff` of `[PoseSelect] Slot ...`
substitution lines confirmed real, different selection decisions between
the two runs.

## Measurement (5-test verify, Pass 2 on in both arms)

| test | post_warp_diff OFF→ON | seam_coherence | seam_visibility | fallback OFF→ON | verdict OFF→ON |
|------|------------------------|-----------------|-------------------|-------------------|------------------|
| test04 | 11.85→12.74 (worse) | 29.82→27.04 (better) | 16.16→18.69 (worse) | False→False | simple_better→simple_better |
| test08 | 12.36→13.42 (worse) | 12.31→12.32 (flat) | 32.01→29.75 (better) | False→False | simple_better→comparable |
| test09 | 2.925→2.937 (flat) | flat | flat | True→True | comparable→comparable |
| test27 | 4.809→4.809 (flat) | flat | flat | False→False | simple_better→simple_better |
| test57 | 12.58→11.06 (better) | 20.94→30.58 (worse) | 2.79→17.13 (much worse) | **True→False** | comparable→simple_better |

Mean `post_warp_diff` across the 5: **8.90 (OFF) → 8.99 (ON)** — flat to
slightly worse, failing the roadmap's own stated success criterion ("mean
seam post_warp_diff drops").

## Visual inspection (the part that actually decided this)

- **test57** is the one test where behavior changed qualitatively: it
  previously fell back to a SCANS (OpenCV) composite (a real ASP attempt
  didn't clear the composite gate). With phase-aware selection on, frame
  selection changed enough that a real ASP composite *did* clear the gate
  — but viewing it shows clear vertical strip/seam discontinuities and
  visible misregistration, especially down the right side of the canvas.
  This is a worse result than the clean SCANS fallback it replaced, even
  though ghosting_siqe improved a lot (83.0→17.7) — another instance of a
  single metric disagreeing with a direct look at the image.
- **test08** was already a clear ASP loss in both arms (pre-existing
  vertical seam artifacts visible in both OFF and ON composites, unrelated
  to this change) — the verdict flip (simple_better→comparable) doesn't
  correspond to a meaningful visual difference; both outputs are bad.
- test04/09/27 were not visually re-inspected beyond their metric deltas,
  since none produced a qualitative behavior change (fallback status
  unchanged) and the deltas were small.

## Why this didn't work

The phase-cross penalty is a purely local, per-slot tie-break — it doesn't
know whether nudging one slot's selection will change whether the
*composite gate* later decides a real attempt is safe enough to keep, only
whether the immediate neighbor-frame pose similarity looks better. On
test57, a locally-better-looking substitution apparently pushed the overall
frame set into a configuration the render/composite gate scored as
"good enough to attempt" when it previously wasn't — but the actual
composite that produced was worse than the safe fallback it displaced. A
purely local per-slot signal isn't sufficient to predict that outcome.

## Disposition

- `ASP_PHASE_AWARE_SELECT` stays default OFF. Because it's nested inside
  Pass 2 (itself default OFF via `ASP_POSE_WINDOW_PX=0`), this change has
  zero effect on the default pipeline regardless.
- The code (the `_phase_ids_from_hashes` refactor, the candidate-phase
  computation, the tie-break penalty) is kept — the refactor is a
  harmless, more-reusable primitive on its own, and the tie-break wiring is
  strictly flag-gated.
- **Recommendation: do not re-attempt as a local per-slot penalty.** If
  phase-awareness is revisited, it needs a way to evaluate the effect on
  the *eventual* composite/render-gate outcome (e.g. only apply the bias
  when it doesn't change whether a phase-boundary-adjacent test would have
  fallen back), not just the immediate neighbor-similarity score — the
  test57 case is exactly the failure mode of a locally-greedy signal
  changing a global fallback/attempt decision for the worse.
