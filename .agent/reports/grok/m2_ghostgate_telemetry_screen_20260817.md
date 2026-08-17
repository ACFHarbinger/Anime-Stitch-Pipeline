# M2 GhostGate telemetry-only — promotion-ladder screen

**Date:** 2026-08-17
**Design:** Chat/Codex `m2_gate_signal_design_review_20260817.md`
**Default:** unchanged. Candidate is `ASP_GHOST_TELEMETRY_ONLY=1`.

## What landed

- `GateDecision.status = "telemetry_only_inverse_validated"` when GhostGate
  is in candidate mode. `ghosting_score_v2` is still computed and stored
  (`would_reject` score + the historic `ghost_gate_siqe:` reason string).
  `accept` stays true. SeamVis is not copied into GhostGate.
- Offline replay harness: `backend/benchmark/screen_ghost_telemetry.py`
  (`just bench::asp-ghost-telemetry-screen`).
- Range-JSON merge: `backend/benchmark/merge_run_json.py`, also invoked
  from `generate_json_results` so the next 97-run does not leave disjoint
  files as the only record.

## Historic fact that changes the five-case story

On `anime_stitch_20260807_045552.json`, GhostGate **never rejected**.
Recorded fallbacks are Composite (27) and SeamVis (27) only. Closest fire
is `asp_test90` at asp/limit = 0.700. There is therefore **no historic
GhostGate-only fallback** to put on the screen.

Five-case screen used instead:

| case | why |
| --- | --- |
| asp_test04 | smoke / hard |
| asp_test08 | smoke; SeamVis fallback |
| asp_test27 | smoke; accept |
| asp_test38 | closest *true composite* to a GhostGate fire (0.668) |
| asp_test96 | known-good true composite (human 3 vs 1) |

## Replay results (saved 2026-08-07 metrics, no GPU)

| set | n | identity changes | historic ghost-only |
| --- | ---: | ---: | ---: |
| five | 5 | **0** | 0 |
| structural red (04/06/07/12/14/15/96) | 7 | **0** | 0 |
| all 97 | 97 | **0** | 0 |

Known-good `asp_test96` stays Raw ASP under both policies.

Because the gate is a no-op on this corpus, a GPU five-case restitch cannot
change Safe ASP selection either. The remaining reason not to flip the
default is process (Chat: design until the ladder is ACK'd), not a failing
screen.

## How to enable the candidate

```
ASP_GHOST_TELEMETRY_ONLY=1
```

`product_safe_asp_policy()` and unset env still reject on `ghosting_score_v2`.
