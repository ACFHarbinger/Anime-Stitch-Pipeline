# M3 coherence_v2 red-set screen (compositor A/B)

**Date:** 2026-08-17
**Default:** still off. Not promoted.

## What was measured

Same warped proxy inputs, two compositors:

- **default:** live Laplacian seam loop
- **v2:** `ASP_COHERENCE_V2=1` ownership apply

Proxy, not a rematch: 6 evenly subsampled dump frames, 0.25 scale, vertical
affines from the 2026-08-07 median `dy_steps`. Human labels still describe
the published default path, not these v2 canvases.

```
just bench::asp-coherence-v2-redset
```

## Crop-loss gate (M3 exit: improve red set without increasing crop loss)

**FAIL.** 6/7 cases lose coverage/area, including known-good test96.

| case | role | cov default→v2 | area default→v2 | crop loss |
| --- | --- | --- | --- | --- |
| asp_test04 | catastrophe | 0.986→0.839 | 161k→142k | yes |
| asp_test06 | catastrophe | 1.000→0.745 | 175k→130k | yes |
| asp_test07 | catastrophe | 0.950→0.945 | same box | no |
| asp_test12 | catastrophe | 0.977→0.847 | 171k→156k | yes |
| asp_test14 | catastrophe | 0.995→0.884 | 167k→160k | yes |
| asp_test15 | catastrophe | 0.984→0.928 | 194k→186k | yes |
| asp_test96 | known-good | 1.000→0.783 | 166k→130k | yes |

First-claim owner-take-all leaves holes where the winning pose has no
pixels — that is crop loss by construction on this apply slice. Seam
visibility sometimes drops (test04 89→18) but that is not a promotion
signal while coverage falls.

JSON for the dashboard A/B:
`docs/website/public/data/coherence_v2_redset.json`
