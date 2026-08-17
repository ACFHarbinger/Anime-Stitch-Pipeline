# M2 CompositeGate `sb` demotion + discriminating-policy check

**Date:** 2026-08-17
**Default:** unchanged. Candidate is `ASP_COMPOSITE_SB_TELEMETRY_ONLY=1`.

## What landed

Same pattern as GhostGate: `composite_sb_telemetry_only` records
`would_reject_sb` / `telemetry_only_inverse_validated` and never rejects on
`strip_banding_score`. `sc` can still reject (separate
`ASP_COMPOSITE_SC_TELEMETRY_ONLY` / `ASP_COMPOSITE_TELEMETRY_ONLY` exist so
we can screen an empty CompositeGate). SeamVis is unchanged.

## Promotion-ladder replay (2026-08-07 metrics, no GPU)

`sb` telemetry-only vs current defaults:

| set | n | Safe ASP identity changes |
| --- | ---: | ---: |
| all 97 | 97 | **26** |

All 26 are historic `composite_gate_sb` cases. None of them also fail `sc`
or SeamVis, so demoting `sb` publishes Raw ASP instead of SCANS. Humans
rated the *published* identity (SCANS replacement) on those 26 — we do
**not** have raw-composite labels for them, so this is not a safe default
flip.

`sc` remains a live reject term. It fired once (`asp_test58`) and has
audit rho = −0.06 (no signal). After `sb` is demoted, CompositeGate has
**no audited-correct input**. Keeping `sc` only preserves a gate for its
own sake. I am not flipping `sc` either; the honest next design is an
empty CompositeGate plus SeamVis, not a sc-only husk.

## Discriminating-policy exit (Harbinger §17.2)

Required: Raw ASP on ≥1 known-good **and** SCANS on known catastrophes
`04/06/07/12/14/15`. Always-SCANS is not success.

| case | role | candidate select | human ASP / SCANS |
| --- | --- | --- | --- |
| asp_test96 | known-good | Raw ASP | 3 / 1 |
| asp_test04 | catastrophe | Raw ASP | 1 / 3 |
| asp_test06 | catastrophe | Raw ASP | 1 / 4 |
| asp_test07 | catastrophe | Raw ASP | 0 / 1 |
| asp_test12 | catastrophe | Raw ASP | 0 / 4 |
| asp_test14 | catastrophe | Raw ASP | 0 / 1 |
| asp_test15 | catastrophe | Raw ASP | 1 / 4 |

**Fails.** Known-good is correct. All six catastrophes are already ACCEPT
under the *current* defaults too — SeamVis also misses them (`sv` 12–30 vs
floor 35). Demoting `sb` does not change the red set. M2's discriminating
exit is not met by any combination of the three existing gates on this
corpus.

## Commands

```
just bench::asp-composite-sb-screen
just bench::asp-safe-asp-discriminating
```
