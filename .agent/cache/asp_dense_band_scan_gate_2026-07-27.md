# ASP §3.1 Follow-up — Dense Band-Scan Composite Gate (2026-07-27)

**Verdict: REJECTED, zero measured value. Reverted, not shipped.**

## Motivation

The `ASP_JOINT_GAIN_SOLVE` postmortem
(`.agent/cache/asp_joint_gain_solve_postmortem_2026-07-27.md`) recommended a
follow-up: "add a finer-grained local check (e.g., worst-adjacent-row-jump,
mirroring `seam_visibility`) to the composite gate itself... so a test like
test04 — where the aggregate gate passes but a local defect remains — still
falls back safely." The same test04 gate-threshold fragility was independently
observed a second time in the ToonOut masking fix
(`.agent/cache/asp_toonout_fix_2026-07-27.md`), reinforcing it as a real,
recurring gate-design gap rather than a one-off.

## What was implemented (and reverted)

`_dense_band_scan_score()` in `backend/benchmark/bench_anime_stitch.py`: a
dense sliding-window version of `_strip_banding_score`. The existing
`_strip_banding_score` only samples a band around each frame's own affine
canvas-entry row — as many samples as there are input frames — so a banding
defect strictly inside a frame's own extent (not at a frame-to-frame
boundary) is structurally invisible to it. The new metric scanned every 10
rows across the full canvas instead, closing that specific sampling gap.
Wired into the existing composite gate (not a new gate name — folded into
`composite_gate_*`, respecting the ≤10-gate budget) with an adaptive
floor/SCANS-ratio limit matching the `sc`/`sb` pattern, floor=45 (chosen as a
conservative starting point above `sb`'s 35, since dense sampling structurally
produces a max ≥ the sparse sampling's max on the same image).

## Measurement (5-test verify, `asp_test04/08/09/27/57`, two conditions)

| test | asp sd (OFF) | asp sd (`ASP_JOINT_GAIN_SOLVE=1`) | sd floor |
|------|-------------|-----------------------------------|----------|
| test04 | 5.6 | 6.8 | 45 |
| test08 | 20.6 | 20.9 | 45 |
| test09 | 6.7 | 6.7 | 45 |
| test27 | 10.8 | 10.8 | 45 |
| test57 | 5.5 | 14.6 | 45 |

Across all 10 data points, `sd` never approached the floor — the largest
observed value (20.9) is well under half of 45. **The new gate never fired,
in either condition, on any test.** It had zero measured effect on any
fallback/real verdict.

## The bigger finding: the original test04 regression no longer reproduces

The joint-gain-solve postmortem's test04 regression was measured *before*
the ToonOut masking fix (S230) landed. Re-running the same
`ASP_JOINT_GAIN_SOLVE=1` condition now, **against the current codebase where
ToonOut is already the shipped default**, shows test04's aggregate `sb` at
**40.4**, over the existing 35.0 floor — the *pre-existing* `composite_gate_sb`
check now correctly triggers the fallback on its own, without any new metric
needed. ToonOut's masking accuracy improvement changed the underlying pixel
composition enough that the specific "aggregate `sb` barely passes while a
local defect remains" scenario documented twice against the *older* codebase
state does not currently exist in the *current* one. This is a genuinely
different, more useful finding than the gate addition itself: **the
regression this follow-up was built to catch appears to already be closed
by an unrelated, earlier fix**, not by anything built in this session.

## Disposition

- Reverted `_dense_band_scan_score` and its gate wiring entirely — per the
  roadmap's anti-goal ("no new quality gates... without... a full-corpus
  run... zero measured value"), an addition that never fires across the only
  available measurement sample is exactly the pattern to avoid, regardless
  of how sound its motivating theory is.
- The underlying architectural gap in `_strip_banding_score` (frame-boundary-
  anchored sampling, not a full scan) is real and still exists in the
  codebase, but is currently unexercised by any known failing case. Parked,
  not built — do not re-attempt without first finding a live test where
  `sb`/`seam_visibility` both pass despite a visible local defect *in the
  current, ToonOut-inclusive codebase state* (the old test04 case no longer
  qualifies).
- `ASP_JOINT_GAIN_SOLVE` itself: given test04's specific regression no longer
  reproduces, its disposition should be re-measured fresh rather than
  inherited from the pre-ToonOut postmortem — flagged as a candidate for a
  future session's one-change/one-benchmark re-verification, not decided
  here (out of scope for this follow-up, which was gate-design only).

## Addendum (2026-07-27, same day) — re-verified against the current baseline

Re-ran the same 5-test set with `ASP_JOINT_GAIN_SOLVE=1` against the current
(ToonOut-inclusive) codebase to check the flagged re-verification. Result is
cleaner than the original postmortem: **test04 now safely reverts to SCANS**
(previously the regression case) via the pre-existing `sb` check;
**test08/test57 are SCANS under both ON and OFF** (unaffected either way,
since ToonOut's own fix already moved them to fallback regardless of gain
solve); **test09 flips SCANS→real** (a coverage win, matching the original
assessment) and **test27 stays real→real** (modest improvement, also
matching). Zero regressions on this 5-test sample — a meaningfully better
picture than the original mixed verdict.

**However**, spot-checking test09's actual output image directly shows
visible horizontal banding/streaking artifacts still present *despite*
passing every automated gate (composite, ghost, seam-vis) — the same
"gates pass, real defect remains" pattern this whole investigation started
from, just not severe enough to trip any current threshold. This is exactly
why the roadmap's ground rule #2 requires human coherence rating, not
automated metrics alone, before any default flip — confirmed again here,
not refuted by the cleaner gate-level result above.

**Disposition unchanged: `ASP_JOINT_GAIN_SOLVE` stays default OFF.** The
5-test re-verify is encouraging (no regressions, one coverage win) but per
ground rule #1 a full-97 run is required before any default flip regardless,
and per ground rule #2 the Phase 0.1 human rating pass — still open — is the
actual gating requirement, not something to substitute with ad hoc image
inspection of individual tests. `moon/roadmaps/asp.md` §3.1 updated to
reflect the cleaner re-verify result while keeping the flag OFF.
