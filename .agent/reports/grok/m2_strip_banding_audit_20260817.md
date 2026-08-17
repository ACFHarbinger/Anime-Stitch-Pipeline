# M2 follow-up — `strip_banding_score` instrumentation and correlation

**Date:** 2026-08-17
**Track:** ASP #31 / roadmap §5 M2
**Follows:** `.agent/reports/claude/m2_gate_signal_correlation_audit_20260817.md`

Claude's audit left CompositeGate's `strip_banding_score` input **unaudited**
because `_compute_all_metrics` never emitted it (2026-07 metric-zoo trim).
This pass wires the score back into the persisted metric dict, extends the
audit tool to recompute it from saved panoramas, and reports rho. **No gate
default changed.**

## What landed

1. `_compute_all_metrics` now writes `strip_banding_score`. Without affines
   the function still returns `0.0`, matching CompositeGate's intentional
   `scans_sb = 0.0` quirk. Future runs persist the number; historical JSONs
   do not have it.
2. `audit_gate_correlation.py` accepts `--recompute-missing` (optional
   `--images-root`, `--max-image-date`, `--no-prefer-raw`) and prints a
   `used_fallback` split for this metric.
3. Tests: `test_strip_banding_instrumentation.py`,
   `test_audit_gate_correlation.py` (6 passed). Existing
   `test_bench_metrics.py` CORE_KEYS / stale-key list updated to match.

## Headline number

```
.venv/bin/python backend/benchmark/audit_gate_correlation.py \
    --run backend/benchmark/output/anime_stitch_20260807_045552.json \
    --labels data/benchmarks/asp_evaluations_20260810.json \
    --recompute-missing
```

| metric | rho | p | n | verdict |
| --- | ---: | ---: | ---: | --- |
| `ghosting_siqe` | -0.600 | <0.0001 | 97 | inverse (unchanged) |
| `edge_energy_score` | -0.531 | <0.0001 | 97 | inverse (unchanged) |
| `sharpness` | -0.471 | <0.0001 | 97 | inverse (unchanged) |
| **`strip_banding_score`** | **-0.417** | **<0.0001** | **97** | **inverse / misleading** |
| `seam_coherence` | -0.062 | 0.544 | 97 | no signal (unchanged) |
| `seam_visibility` | +0.425 | <0.0001 | 97 | works (unchanged) |
| `seam_gradient` | +0.473 | <0.0001 | 97 | works (unchanged) |

`strip_banding_score` is the fourth-worst inverse metric in the set — same
family as GhostGate's signal, not like SeamVis.

## Pairing caveat (do not hide this)

The 2026-08-07 run JSON has no `strip_banding_score`. Values were recomputed
from `dump/output/` panoramas + `alignment.affines` (`ty` only). Image
mtimes: **12 from 2026-08-07, 1 from 2026-08-15, 84 from 2026-08-16**.
Human labels remain the 2026-08-10 review of the 2026-08-07 published pair.
This is **not** a pure 2026-08-07 pairing.

Date-locked check (`--max-image-date 2026-08-07`): n=12, rho=**-0.525**,
p=0.080 (same direction, under-powered).

## Fallback split (same recompute)

CompositeGate already used high `sb` to replace Raw ASP with SCANS, and
humans rated the *published* identity. That could fake an inverse if
fallbacks (higher human "ASP" scores) also show higher recomputed `sb`.

| subset | rho | p | n | mean sb | mean human Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | -0.417 | <0.0001 | 97 | 28.3 | -0.629 |
| true composite (`used_fallback=false`) | **-0.365** | **0.016** | 43 | 21.3 | -1.233 |
| fallback | -0.119 | 0.390 | 54 | 33.8 | -0.148 |

The inverse **survives on the 43 true composites**. Quartiles go the wrong
way: lowest-`sb` quarter has mean human Δ **-1.21**; highest-`sb` quarter
has mean Δ **0.00**. Low strip-banding is not a human-quality win.

## What this does to CompositeGate

| Gate | Signals | Status after this pass |
| --- | --- | --- |
| GhostGate | `ghosting_score_v2` | inverse (rho=-0.60) — still the worst |
| CompositeGate | `seam_coherence` + `strip_banding_score` | **no audited-correct input** (sc: no signal, sb: inverse) |
| SeamVisGate | `seam_visibility_score` | still the only working gate (rho=+0.43) |

## Not done this turn (promotion ladder)

No default flipped. Chat/Codex still owns the GhostGate / `cqas` replacement
design review of Claude's method. The next one-change experiment I will
propose, after that review:

- Demote CompositeGate's `sb` term to telemetry-only (keep the log line,
  stop it from rejecting), five-case screen, then stratified, then 97.
- Do **not** raise or lower `ASP_GATE_SB` as a first experiment — the
  signal itself is inverse, so retuning the floor cannot make it
  discriminating in the M2 sense.

A consolidated post-M1 97-case metrics JSON will persist `strip_banding_score`
directly and retire the image-recompute path for this metric.
