# SeamVis cannot be retuned to pass the M2 discriminating exit

**Date:** 2026-08-17
**Follows:** discriminating-policy FAIL
  (`m2_composite_sb_and_discriminating_20260817.md`)

The previous pass showed SeamVis (floor 35, ratio 3) misses all six
catastrophes. This pass asks whether *any* `(floor, ratio)` pair would
catch those six and still keep known-good `asp_test96`.

## Result

**Infeasible. 0 pairs** on a 0–40 × {1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10}
grid.

| case | role | asp_sv | sim_sv |
| --- | --- | ---: | ---: |
| asp_test15 | catastrophe | **12.55** | 3.07 |
| asp_test14 | catastrophe | 12.59 | 2.01 |
| asp_test04 | catastrophe | 17.48 | 2.28 |
| asp_test07 | catastrophe | 18.73 | 5.71 |
| asp_test12 | catastrophe | 23.06 | 3.18 |
| asp_test06 | catastrophe | 29.58 | 2.71 |
| asp_test96 | known-good | **32.20** | 4.89 |

Known-good `asp_test96` has a *higher* `seam_visibility` than every
catastrophe. Catching test15 (`sv=12.55`) requires `floor < 12.55` and
`ratio < 12.55/3.07 ≈ 4.1`. Keeping test96 requires `floor ≥ 32.2` or
`ratio ≥ 32.2/4.89 ≈ 6.58`. Those regions do not overlap.

Gemini's M2.5a finding that `seam_visibility` is corpus-aligned (ρ=+0.43,
stronger on photometric defects) is compatible with this: the red-set
catastrophes are structural (torn / misordered / low coherence), not
high-sv seam cuts. The metric that works in aggregate is the wrong
cut-point for this exit set.

## Implication

M2's discriminating exit cannot be met by retuning Composite/Ghost/SeamVis
thresholds. It needs a **new** structural signal (or a human/HITL veto),
not another floor change. No default flipped.

```
just bench::asp-seamvis-threshold-sweep
```
