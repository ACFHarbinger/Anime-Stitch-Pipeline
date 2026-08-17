# M3 coherence_v2 red-set screen (compositor A/B)

**Date:** 2026-08-17 (updated after exclusive-keep fix)
**Default:** still off.

## Fix that unblocked the crop gate

Owner-take-all on the *union* connected component painted exclusive
coverage of the losing pose with the winner's empty pixels (holes).
Assignment is now:

- A-only → A, B-only → B
- only A∩B is contested (one owner, or whole-overlap handoff if no corridor)

## Crop-loss gate (re-run)

Proxy A/B: 6 subsampled frames, 0.25 scale, median-dy affines.
Human labels still describe the published default path.

**PASS.** 0/7 crop-loss increases. Known-good test96 coverage 1.000→1.000.

| case | cov default→v2 | area | crop loss |
| --- | --- | --- | --- |
| asp_test04 | 0.986→0.994 | same | no |
| asp_test06 | 1.000→1.000 | same | no |
| asp_test07 | 0.950→0.946 | same | no |
| asp_test12 | 0.977→0.971 | same | no |
| asp_test15 | 0.984→0.966 | same | no |
| asp_test96 | 1.000→1.000 | same | no |

Structural proxies are mixed (fracture down on test96 64.8→39.3, up on
test14 59.5→79.7). That is **not** a human screen and not an M3 exit.

Not promoted. Sidecar: `docs/website/public/data/coherence_v2_redset.json`.

## Full-resolution renders (2026-08-17)

`--scale 1.0 --max-frames 8` for Harbinger visual review. Crop gate still
**0/7**. PNGs:

`docs/website/public/data/coherence_v2/{case}_default.png` and `{case}_v2.png`

JSON includes `renders.default_url` / `coherence_v2_url` (`/data/coherence_v2/...`).
Still compositor-only (even subsample + median-dy). Still not a human rating.
