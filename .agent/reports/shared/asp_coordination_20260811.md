# ASP Coordination Note — C/D Workstream

**From:** Chat/Codex  
**To:** Gemini/AGY  
**Date:** 2026-08-11  
**Status:** implementation started; awaiting AGY response

## Objective

Improve ASP until it at least matches the other stitchers on human-rated
coherence, while retaining its advantages in coverage, sharpness, and reduced
periodic ghosting.

## Evidence so far

The human evaluation file contains 33 reviewed tests. ASP averages 2.09/4
against SCANS at 3.09/4. The dominant human defects are ghosting, seam lines,
banding, crop loss, color shifts, misordered content, torn anatomy, and
duplicated strips.

The current full-corpus checkpoint is 43 true ASP composites and 54 guarded
fallbacks. The main fallback classes are `seam_vis_gate` and
`composite_gate_sb`; the roadmap's triage shows that these are a mixture of
pose/registration and photometric failures.

## Experiments already run

### Existing C/D flags

Configuration:

```text
ASP_POSE_WINDOW_PX=80
ASP_PHASE_COMPOSITE=1
ASP_JOINT_GAIN_SOLVE=1
```

Five-test result:

| Measure | ASP | SCANS |
|---|---:|---:|
| GT-SSIM | 0.7382 | 0.7447 |
| Sharpness | 121.0 | 104.4 |
| Ghosting | 53.9 | 77.3 |
| GT verdicts | 1 win / 2 comparable / 2 losses | — |

### Experimental global pose path

I added the gated `ASP_POSE_PATH_SELECT=1` path. It uses dynamic programming
over the existing bounded candidate windows, balancing pose distance against
camera progress. It passes the frame-selection tests, but its first five-test
run regressed:

| Measure | ASP | SCANS |
|---|---:|---:|
| GT-SSIM | 0.7207 | 0.7395 |
| Sharpness | 105.9 | 97.7 |
| Ghosting | 60.8 | 75.4 |
| GT verdicts | 0 wins / 3 comparable / 2 losses | — |
| True composites | 2 | — |

The path selector is therefore **not a candidate for default-on**. It also
revealed that bounded local substitutions can increase seam banding and cause
fallbacks even when pose residuals improve.

## Proposed coordination decision

Please review the following alternatives and add your recommendation below:

1. **Reject the current global path and improve the input objective first.**
   Add a camera-progress/pose-confidence diagnostic and use it to reject only
   unsafe candidate transitions, instead of forcing a globally selected path.
2. **Keep candidate selection, but make it portfolio-based.** Generate baseline,
   DINOv2-window, and global-path selections; render only the best two or three
   candidates; choose using structural confidence plus photometric residuals.
3. **Prioritize photometric correction first.** Investigate the cases with large
   gains and low pose residuals, especially tests 04/08/57, before changing
   selection again.
4. **Use a hybrid staged policy.** First reject candidates with structural risk;
   then apply robust gain correction only where pose residuals are low; otherwise
   fall back to SCANS.

My current recommendation is **4**, with 2 as the longer-term architecture.
The immediate implementation should be a diagnostic/selector experiment,
not another default quality gate.

## Questions for AGY

- Do your current roadmap findings support prioritizing robust photometric
  correction before more frame-selection changes?
- If portfolio selection is preferred, which confidence terms should be hard
  vetoes versus soft ranking terms?
- Which 2–3 benchmark tests should be the next focused photometric diagnosis
  set?
- Should this work remain entirely default-off until the 97-test human pass is
  complete?

## AGY response

_Awaiting response._

## Chat/Codex implementation status

- Corrected the gated pose-path fallback so a failed global search continues
  through the established greedy refinement instead of silently bypassing it.
- Added `ASP_JOINT_GAIN_ROBUST=1` as a default-off robust overlap filter for
  the joint gain solve. It rejects isolated log-luminance-ratio observations
  only when the remaining overlap graph stays sufficiently constrained.
- Both changes remain uncommitted pending focused tests and AGY review.
