# ASP Benchmark Analysis — Post-Trim Full Run (2026-07-09)

*Run: `anime_stitch_20260709_030853.json`, 97 datasets, 8,816 s (87 s/test).
Code: trimmed pipeline (S200 trim + fixes), pairwise-DP seam path, BiRefNet
restored, SeamVisGate floor 35. Report: `dump/output/benchmark_report.md`.
Baselines referenced: S160 full run 2026-06-23 (`anime_stitch_20260623_234305.json`).*

---

## 1. Headline Numbers vs S160

| Metric | S160 (2026-06-23) | Post-trim (2026-07-09) | Δ |
|---|---|---|---|
| Verdicts (97t) | 10 asp / 41 comp / 45 simple / 1 insuf | **27 asp / 41 comp / 29 simple / 0 insuf** | asp_better ×2.7 |
| GT verdicts (55t) | 9 asp / 22 comp / 26 simple (S142 basis) | 7 asp / 24 comp / 24 simple | ~flat |
| GT-SSIM (raw, 55t) | 0.653 / 0.693 (−0.041) | 0.665 / 0.696 (−0.032) | gap −22% |
| Aligned SSIM (55t) | 0.680 / 0.720 (−0.040) | **0.693 / 0.718 (−0.025)** | gap −37% |
| seam_visibility (avg) | 25.77 / 4.21 (6.1×) | **12.13 / 3.32 (3.7×)** | halved |
| ghosting_siqe (avg) | 36.2 / 72.3 | 55.3 / 75.1 | still better; see §4 |
| sharpness (avg) | 96.7 / 64.3 | 88.6 / 63.4 | ~same advantage |
| coverage (avg ASP) | 0.979 | 0.987 | — |
| Runtime | 128 s/test | **87 s/test** | −32% |
| True ASP composites | 88/97 "passed" (gates mostly off) | 51/97 | see §3 |

**Read the verdict jump honestly.** The 27 `asp_better` includes fallback outputs:
46/97 tests now emit SCANS-on-preprocessed-frames instead of an ASP composite,
and that output often beats the raw `cv2.Stitcher` simple stitch (better frame
selection + photometric preprocessing + no crop failures). The genuinely-composited
wins with GT confirmation are the same family as always (test17, test31, test44,
test96). The **aligned-SSIM gap improvement (−0.040 → −0.025)** is the more
meaningful signal, and it comes from three sources: BiRefNet restored (masks were
silently broken since 2026-07-05), the SeamVisGate replacing catastrophic
composites with sane SCANS output, and the removal of unverified default-ON gates.

## 2. What Changed Since S160 (and what each change did)

1. **Trim itself (S200)** — behavior-preserving on the DP path; runtime −32%
   (fewer per-seam audits, no §5.x metric cascade, C++ paths retained).
2. **BiRefNet fixed** — the July-5 vendoring had broken mask loading entirely
   (ruff stripped `eval()`-resolved imports; weights filename mismatch). Every
   stage that depends on fg/bg separation was silently degraded before the fix.
3. **GraphCut §4.2 measured for the first time — and defaulted OFF.** With the
   cv2-style ≤0.4 MPix seam-estimation downscale it runs in seconds, but its
   composites scored seam_visibility 20–80 vs the DP path's 2–16 on the same
   tests (hard ownership cut, ±8 px feather, no per-seam gain compensation).
   The gate-era's "highest-impact fix" failed its first real measurement as wired.
4. **SeamVisGate recalibrated (floor 20 → 35)** — at 20 it silently replaced the
   majority of ASP output with SCANS (simple-stitch sv is 1.5–4, so the 3× ratio
   term never binds). At 35 it catches exactly the catastrophic family.

## 3. Fallback Anatomy (46/97)

| Reason | N | Notes |
|---|---|---|
| seam_vis_gate (≥35 or 3×) | 24 | includes the old catastrophic family (test82 sv=158.8, test74, test34…) — their SCANS outputs are now coherent and often GT-competitive |
| composite_gate_sb (render banding) | 19 | the historical ~40% render-gate cluster: fg-dominant, high-animation scenes |
| composite_gate_sc | 2 | |
| alignment_failed | 1 | test49 (min_gap), the perennial |

This matches the corpus structure identified in June: roughly half the corpus is
scenes where multi-frame pose-blend assembly cannot work with the current
architecture, and the correct behavior is a guarded fallback. The gates now do
that job with 3 mechanisms instead of ~50.

## 4. Metric Caveats

- **ghosting_siqe ASP average rose (36→55)** because 46 outputs are now
  SCANS-family (periodic hard cuts score high on siqe by construction). On true
  ASP composites the siqe advantage persists (e.g. test57: 21.8 vs 81.9).
- `seam_visibility` fell partly *because* the gate removes the worst composites
  from the pool — it is a post-gate output metric, not a composite-quality delta.
- **No automated verdict here measures structural coherence.** Visual audit of
  this run: test96 is a real, dramatic win (full-body reconstruction vs simple's
  bottom-third crop); test17 is clean (the pre-trim blocky gain patches are
  gone); test44 wins coverage but has a shear-banded head; test27/test57 true
  composites still show the Class-A wide-feather pose-ghosting family. The
  fundamental ceiling (frame-selection pose gaps) is unchanged, exactly as the
  critical evaluation predicted — the trim removed noise, not the wall.

## 5. Standing Conclusions

1. The trimmed pipeline reproduces the S160-class results with a third less
   runtime, 43 flags instead of 387, and 8 gates instead of ~150 — every future
   change is now measurable in one 2.5 h run.
2. The verdict-level gain (10→27 asp_better) is mostly *honest fallback
   behavior*, not composite-quality progress. The composite-quality gap
   (aligned −0.025) and its root cause (pose gaps at selection time) remain.
3. GraphCut-as-wired is not the fix; if revisited it needs GC-boundary
   photometric correction + wider feathering, benchmarked before defaulting.
4. Next steps stay as in `research/ASP_Critical_Evaluation_2026-07-08.md` §9:
   human coherence ratings, then animation-phase grouping at ingestion
   (`ASP_HOLD_AVERAGE=1` A/B is the cheapest first experiment).
