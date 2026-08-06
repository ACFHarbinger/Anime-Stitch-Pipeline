# ASP Phase 4 — Fallback Class Re-Examination (2026-07-27)

18-test targeted batch (`asp_test04/08/09/27/57/49/32/35/41/48/58/59/51/33/55/60/37/44`,
`anime_stitch_20260727_223002.json`), chosen to span every current fallback
class rather than an arbitrary sample: the 5-test verify set (known
baseline) plus every distinct `fallback_reason` prefix from the most recent
larger sample available (`anime_stitch_20260726_144009.json`, 37 tests,
pre-ToonOut) — `alignment_failed` (test49), `composite_gate_sb` (test32/35/
41/48, sampled 4 of 7), `composite_gate_sc` (test58/59, both), `ghost_gate_
siqe` (test51, only one), `seam_vis_gate` (test33/55/60, sampled 3 of 4),
plus two `real_composite` controls (test37/44). Clean run, no resource-danger
triggers, thread-cap fix continues to hold at this larger (18-test) scale —
the biggest single step-up in scale since the fix landed.

## Finding 1 — classifications shifted substantially since 2026-07-26

Comparing against the older 37-test sample for the 13 overlapping tests
(the other 5 are the already-tracked verify set): **8 of 13 changed fallback
class** between the two runs, driven by the accumulated fixes landed
2026-07-27 (ToonOut masking, aligned-SSIM benchmark fix, and this session's
own gate/gain-solve investigations) rather than any deliberate Phase-4 work:

| test | old class (07-26) | new class (07-27) |
|------|--------------------|--------------------|
| test49 | `alignment_failed` | `composite_gate_sb` |
| test32 | `composite_gate_sb` | `seam_vis_gate` |
| test35 | `composite_gate_sb` | **real composite** |
| test41 | `composite_gate_sb` | `seam_vis_gate` |
| test58 | `composite_gate_sc` | `composite_gate_sb` |
| test51 | `ghost_gate_siqe` | `seam_vis_gate` |
| test33 | `seam_vis_gate` | **real composite** |
| test55 | `seam_vis_gate` | **real composite** |
| test60 | `seam_vis_gate` | `composite_gate_sb` |
| test37 | **real composite** | `seam_vis_gate` |

Net: 3 tests improved to real composites (test33/35/55) that previously
failed outright; 1 test (test37) regressed from real to a safe fallback —
consistent with the already-documented ToonOut pattern (a masking-accuracy
improvement can reveal a defect a cruder mask previously hid, trading a
flawed real composite for a safe fallback, which is a net positive under
this project's "never worse than fallback" objective, not a loss). **Any
Phase-4 fallback-class accounting done before 2026-07-27's fixes landed is
now stale** — this 18-test batch is a more current reference point, though
still far short of the full-97 census the phase's exit gate actually wants.

## Finding 2 — `alignment_failed` (test49) is resolved as a class, not just this test

The roadmap's Phase 4 text singles out test49 for individual diagnosis as
the corpus's one `alignment_failed` case. It no longer is: test49 now
reaches the composite stage and fails at `composite_gate_sb` instead
(`sc=39.3/limit 53.2, sb=47.1/limit 35.0`, `mean_post_warp_diff=20.3`, 3
detected phases). Whatever combination of this session's fixes (most likely
ToonOut's masking-accuracy improvement, since that changes the BiRefNet
output feeding bundle adjustment and the alignment health check) resolved
the outright alignment failure. **The item is superseded, not closed as
originally scoped** — test49 is no longer a special alignment case; it now
belongs to the (large) `composite_gate_sb` class discussed below.

## Finding 3 — `seam_vis_gate` is the dominant class on this sample (7/18, 39%), and it is NOT homogeneous

The roadmap's own hypothesis ("their failed composites are mostly
pose-blend artifacts") is only partially right. Cross-referencing each
`seam_vis_gate` test's `seam_visibility` score against the already-existing
`mean_post_warp_diff` metric (§0.4c — mean seam pose-registration residual)
splits the class cleanly into two different root causes:

| test | seam_visibility | mean_post_warp_diff | likely root cause |
|------|-----------------:|----------------------:|--------------------|
| test41 | 63.8 | 44.7 | pose-blend (high both) |
| test32 | 39.0 | 18.3 | mixed/moderate |
| test51 | 60.6 | 16.4 | **high sv, low post_warp_diff — photometric, not pose** |
| test08 | 143.3 | 15.3 | **very high sv, low post_warp_diff — photometric, not pose** |
| test57 | 40.5 | 12.0 | photometric-leaning |
| test37 | 65.6 | 6.7 | **high sv, very low post_warp_diff — clearly photometric** |
| test09 | 39.8 | 3.0 | **clearly photometric** |

Several of the most severe seam_visibility scores (test08's 143.3, test37's
65.6, test51's 60.6) occur on tests with *low* pose-residual — meaning the
composite's actual frame registration is fine, and the visible defect is
banding/exposure mismatch, not torn anatomy or misaligned content. This
contradicts a "mostly pose-blend" characterization for the *worst* cases in
this sample specifically, even though it holds for test41.

## Finding 4 — testing whether `ASP_JOINT_GAIN_SOLVE` rescues the photometric subset: partial success, and a real caution

Given `ASP_JOINT_GAIN_SOLVE` (§3.1, default OFF) already showed a genuine
win on the lowest-post-warp-diff case in the original 5-test set (test09),
ran the same flag against all 7 `seam_vis_gate` tests from this batch
(`anime_stitch_20260727_224956.json`):

| test | post_warp_diff | OFF | ON | outcome |
|------|----------------:|-----|-----|---------|
| test09 | 3.0 | SCANS | **REAL** | flip — coverage win |
| test32 | 18.3 | SCANS | **REAL** | flip — win |
| test37 | 6.7 | SCANS | SCANS (sv 65.6→42.2) | improved a lot, not enough to cross the gate |
| test08 | 15.3 | SCANS | SCANS (sb 35.9, just over the 35 floor) | improved, narrowly still fails |
| test57 | 12.0 | SCANS | SCANS (sb 36.8, just over) | improved, narrowly still fails |
| test41 | 44.7 | SCANS | SCANS (sb **80.4**, far worse) | **regressed** |
| test51 | 16.4 | SCANS | SCANS (sb **53.1**, much worse) | **regressed** |

2/7 genuinely flip to real composites (both low/moderate post_warp_diff, as
hypothesized). 3/7 (test37/08/57) show real improvement in their gate
scores without quite crossing the threshold — encouraging but inconclusive.
**But test51 breaks the clean story**: it has a comparably low
post_warp_diff to test32 (16.4 vs 18.3) yet responds in the *opposite*
direction under gain solve (composite_gate_sb sb jumps from passing to
53.1, a severe regression) rather than improving like test32 did. This
matches the original §3.1 postmortem's own root-cause theory (limited
background overlap for the solver to constrain some frame pairs against
each other, rather than a clean function of pose-residual alone) —
`mean_post_warp_diff` correlates with gain-solve's effect direction but is
**not a reliable enough discriminator to build an automatic per-test
dispatch rule from**. Tried and explicitly not pursued further, rather than
shipping a heuristic the data doesn't actually support.

## Finding 5 — the flagged test51 follow-up: checked a second discriminator (frame/pair count), also insufficient

The dedicated follow-up flagged above: does frame count (and hence the
number of overlapping pairs `_joint_gain_solve` has to build its
least-squares system from) explain why test51 regresses while test32
improves, despite similar `mean_post_warp_diff`? `_joint_gain_solve` builds
its system over **all** pairs with sufficient shared background overlap
(not just adjacent frames), so fewer selected frames means fewer possible
constraining pairs — a plausible structural reason a sparse sequence's gain
solution would be more fragile:

| test | final frames | max possible pairs | gain-solve outcome |
|------|---------------:|---------------------:|----------------------|
| test51 | 8  | 28  | **regressed** |
| test41 | 10 | 45  | **regressed** |
| test08 | 9  | 36  | improved (didn't cross gate) |
| test32 | 17 | 136 | **flip — win** |
| test09 | 21 | 210 | **flip — win** |
| test37 | 26 | 325 | improved (didn't cross gate) |
| test57 | 26 | 325 | improved (didn't cross gate) |

The two regressions (test51, test41) do have the lowest pair counts, which
is suggestive — but **test08 breaks the pattern**: it has the second-fewest
pairs (36) yet *improved* rather than regressing, the opposite of what a
clean "fewer pairs = more fragile = more likely to regress" rule would
predict. **Conclusion: frame/pair count, like `mean_post_warp_diff`, is
correlated but not reliable enough on its own to predict
`ASP_JOINT_GAIN_SOLVE`'s effect direction.** Two independent, plausible
cheap heuristics have now each been checked and each falls short on a real
counter-example — this isn't a matter of finding the right one-line
discriminator; a full per-test measurement (full-97 run, or a dedicated
per-test A/B pass) is genuinely required before `ASP_JOINT_GAIN_SOLVE` can
be conditionally dispatched with any confidence. Closing this follow-up
here rather than trying a third heuristic on the same 7-test sample, which
risks overfitting an explanation to too little data.

## Disposition

- **No new code shipped from this analysis** — this is diagnostic, per the
  roadmap's Phase-4 instruction to "re-examine what remains," not a measured
  code change. `ASP_JOINT_GAIN_SOLVE` stays default OFF (unchanged from the
  S242 re-verification); no automatic post-warp-diff-based gating was built,
  since test51's counter-example shows the signal isn't clean enough to
  trust for that purpose.
- **Phase-4 roadmap text needs updating**: test49 is no longer the corpus's
  `alignment_failed` case (resolved as a side effect of other fixes, now a
  `composite_gate_sb` case like many others); the `seam_vis_gate` class
  should be understood as a mix of pose-blend and photometric failures, not
  uniformly pose-blend, with `mean_post_warp_diff` as a useful (if noisy)
  diagnostic split.
- **Genuinely useful next step, not attempted here**: since `test51` is the
  one clear counter-example to the post-warp-diff pattern, understanding
  *why* it responds oppositely (limited bg overlap? a specific frame pair?)
  would be worth a dedicated, narrow follow-up before concluding anything
  stronger about `ASP_JOINT_GAIN_SOLVE`'s scope of applicability.
