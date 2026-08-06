# ASP §3.1 Joint Canvas-Space Blocks-Gain Solve — First Measurement (2026-07-27)

**Verdict: mixed. `ASP_JOINT_GAIN_SOLVE` stays default OFF** — genuine wins on
2/5 tests, roughly neutral on 2/5, one new regression on 1/5. Not a clean
enough result to flip the default, but not a flat rejection either — see
"Disposition" for the specific, fixable gap found.

## What was implemented

Replaced `_equalize_warped_gains`'s sequential pairwise chain (frame 0 as
reference, each subsequent frame corrected to match its already-corrected
predecessor — the drift-over-long-chains problem the roadmap named) with
`_joint_gain_solve`: the Brown-Lowe (2007) formulation, one linear
least-squares system over **all** overlapping frame pairs' bg-only mean
luminance simultaneously, with a gain-prior term regularizing each frame's
gain toward 1.0 (the same formulation `cv2::detail::GainCompensator` uses
for classical panorama stitching). Scalar per-frame gain (not per-channel,
not spatial blocks), clamped to [0.5, 2.0], applied only to each frame's own
background pixels — matching the roadmap's "bg-pixels-only, luminance-scalar,
clamped" spec exactly. Gated behind `ASP_JOINT_GAIN_SOLVE` (default OFF).

## Measurement (5-test verify, `asp_test04/08/09/27/57`)

| test | seam_coherence | seam_visibility | sharpness | ghosting_siqe | fallback OFF→ON |
|------|-----------------|-------------------|-----------|-----------------|--------------------|
| test04 | 28.33→28.37 (flat) | 2.03→32.66 (**worse**) | 36.6→132.2 (better) | 67.0→15.8 (better) | **True→False** |
| test08 | 13.01→16.47 (worse) | 28.15→10.44 (**better**) | 279→244 (slightly worse) | 32.4→41.6 (worse) | False→False |
| test09 | 19.54→22.72 (worse) | 1.96→6.34 (still low) | 80.1→95.7 (better) | 78.0→47.5 (better) | **True→False** |
| test27 | 30.25→28.22 (better) | 16.14→6.62 (**better**) | 157.1→126.8 (worse) | 50.4→48.6 (flat) | False→False |
| test57 | 28.26→24.84 (better) | 11.03→11.66 (flat) | 46.6→82.2 (better) | 20.6→25.5 (worse) | False→False |

## Visual inspection (the part that decided this)

- **test08 — real win.** The OFF composite has clearly visible banding
  cutting through the arm/shirt; the ON composite noticeably reduces it,
  matching the seam_visibility improvement (28.15→10.44).
- **test09 — real win, and a bonus.** OFF was a SCANS fallback; ON produces
  a clean real composite that actually shows *more* content (a background
  character cropped out of the SCANS version becomes visible). No new
  defects. Matches the ghosting_siqe improvement (78.0→47.5).
- **test27 — modest, real improvement.** Both versions still show the same
  pre-existing multi-copy ghosting artifact (a pose/motion-blend issue,
  unrelated to photometric gain) around the pom-poms/hands, but it's
  slightly cleaner in ON, matching the seam_visibility gain.
- **test57 — a wash.** Both versions show essentially the same strip-banding
  pattern at similar severity; the sharpness/seam_coherence deltas don't
  correspond to a visible quality difference either direction.
- **test04 — a real new regression.** OFF fell back to SCANS (composite
  gate: `asp_sb=35.8` vs. `limit=35.0` — failed by a hair). ON's aggregate
  banding score evidently improved enough to cross that gate and produce a
  real composite instead — but that composite shows clearly visible
  horizontal banding across the character's torso, a defect the
  seam_visibility metric correctly flags (2.03→32.66, a large jump) even
  though sharpness and ghosting both look better in isolation. This is
  exactly the "never trust a metric without looking" case the roadmap's own
  ground rules warn about, on the composite-quality metric this time, not
  the coarser gate metric.

## Root cause of the test04 regression

The composite gate that decides fallback-vs-real uses `sb` (an aggregate
strip-banding statistic over the whole composite). The joint gain solve
improved that aggregate score enough to cross the gate's threshold — but
crossing an *aggregate* threshold doesn't guarantee zero *local* defects.
`seam_visibility` (the worst single adjacent-row luminance jump) measures
something finer-grained, and it shows the joint solve's globally-optimal
gain assignment still leaves a real local mismatch at specific frame
boundaries in this test — plausibly because those particular adjacent
frames have limited bg overlap for the solver to constrain them against
each other confidently, so the prior term (which only pulls toward gain=1.0,
not toward agreement with a specific neighbor) doesn't fully compensate.

## Disposition

- `ASP_JOINT_GAIN_SOLVE` stays default OFF. Two genuine wins and no clear
  regressions except test04 is a promising start, but the project's own
  "never worse than fallback" objective means a single case where enabling
  this trades a safe fallback for a visibly banded composite is disqualifying
  for a default flip, even with 2/5 real wins elsewhere.
- The implementation itself (`_joint_gain_solve`/`_apply_joint_gain_solve`
  in `compositing.py`) is kept — it's correctly gated, matches the roadmap's
  exact spec, and the wins on test08/test09 show the underlying approach is
  sound, not just theoretically appealing.
- **Recommended follow-up** (not attempted here, to keep this one change /
  one measurement): add a finer-grained local check (e.g., worst-adjacent-
  row-jump, mirroring `seam_visibility`) to the composite gate itself when
  `ASP_JOINT_GAIN_SOLVE` is active, so a test like test04 — where the
  aggregate gate passes but a local defect remains — still falls back
  safely instead of shipping a banded composite. That's a gate-design
  change, not a gain-solve change, and deserves its own separate
  measurement rather than being bundled into this one.
- A larger validation run (more than 5 tests) would help distinguish
  whether test04's regression is a one-off (limited bg overlap on that
  specific frame pair) or systematic — worth doing before ever considering
  a default flip, per this project's standard scaling discipline.
