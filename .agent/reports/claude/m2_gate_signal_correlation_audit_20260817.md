# M2 gate-signal audit — Spearman correlation vs human labels (2026-08-17)

**Scope:** roadmap §5 M2, issue #31, deliverable "Audit existing gates against
the completed human labels; demote/remove signals inversely correlated with
human quality." This report grounds that deliverable in a reproducible
computation. It does **not** change any gate default — that is a separate,
reviewed change per the roadmap's promotion ladder (one change → five-case
screen → stratified set → all 97, non-regression required).

## What was missing

The 2026-08-15 bus entry (Chat/Codex) cited "sharpness, edge-energy, and the
current ghosting score are inversely correlated with human ASP-v-SCANS deltas
(Spearman -0.47, -0.53, -0.60)" as the evidentiary basis for M2's gate-demotion
requirement. No script or saved analysis backing that claim exists anywhere in
`.agent/reports/` (checked `chat/`, `shared/`, and the parent Image-Toolkit
repo's `.agent/reports/`) — it was an unlinked number. This report reproduces
it independently, extends it to every no-reference metric the benchmark
computes, and maps the result onto the actual gate code in
`backend/src/core/pipeline/safety_policy.py`.

## Method

New tool: `backend/benchmark/audit_gate_correlation.py`.

For each of the 97 reviewed cases in `data/benchmarks/asp_evaluations_20260810.json`,
correlate `(ASP metric − SCANS metric)` against `(human ASP score − human SCANS
score)`, using the pre-M0 2026-08-07 baseline run
(`backend/benchmark/output/anime_stitch_20260807_045552.json` — the only saved
run with per-case metrics for all 97 names; the post-M1 ungated run only exists
in disjoint partial ranges, see "Data gap" below). Each metric's delta is
re-oriented so that positive always means "the metric says ASP got better",
per that metric's own higher/lower-is-better design intent (documented in
`safety_metrics.py`/`bench_anime_stitch.py` docstrings — not asserted here).
A well-behaved no-reference metric should then score `rho > 0`.

```
.venv/bin/python backend/benchmark/audit_gate_correlation.py \
    --run backend/benchmark/output/anime_stitch_20260807_045552.json \
    --labels data/benchmarks/asp_evaluations_20260810.json
```

## Results (n=97, all reviewed cases used, none skipped)

| metric | rho | p | verdict |
| --- | ---: | ---: | --- |
| `ghosting_siqe` | **-0.600** | <0.0001 | inverse / misleading |
| `edge_energy_score` | **-0.531** | <0.0001 | inverse / misleading |
| `sharpness` | **-0.471** | <0.0001 | inverse / misleading |
| `color_entropy` | -0.210 | 0.039 | inverse / misleading (weaker) |
| `cqas` | -0.091 | 0.374 | no signal |
| `seam_coherence` | -0.062 | 0.544 | no signal |
| `coverage` | +0.091 | 0.373 | no signal |
| `seam_visibility` | **+0.425** | <0.0001 | works as intended |
| `seam_gradient` | +0.473 | <0.0001 | works as intended |

The three headline numbers from the bus (sharpness -0.47, edge-energy -0.53,
ghosting -0.60) reproduce exactly. Two findings beyond the original claim:

1. **`seam_visibility` and `seam_gradient` are not inversely correlated — they
   work as intended** (rho +0.43 / +0.47). These should *not* be lumped into
   a blanket "no-reference metrics are untrustworthy" conclusion.
2. **`cqas` (the Composite Quality Aggregate Score — the single scalar used
   across dashboards/reports for the 43+ GT-less cases) has no measurable
   correlation with human judgment** (rho=-0.09, not significant). This was
   not previously flagged. It is a weighted blend of `ghosting_siqe` (weight
   0.35, inverse), `seam_visibility` (0.30, correct), `seam_coherence` (0.20,
   no signal), `sharpness` (0.15, inverse), and `canvas_gain_uniformity`
   (0.15 — see gap below). The two largest-weighted inverse/no-signal
   components (0.35 + 0.15 + 0.20 = 0.70 of total weight) are cancelling out
   the one strong correct signal (`seam_visibility`, 0.30). This explains the
   previously-reported 59.8% automated-verdict-vs-human-ordering agreement
   without needing a separate hypothesis.

## Mapping onto the actual Safe ASP gates

`safety_policy.py`'s three gates use exactly these metrics:

| Gate | Signal(s) used | Per this audit |
| --- | --- | --- |
| `GhostGate` | `ghosting_score_v2` (== `ghosting_siqe`) | **inverse (rho=-0.60)** — the metric that decides this gate is the single worst-scoring signal audited. |
| `SeamVisGate` | `seam_visibility_score` | works as intended (rho=+0.43) — no change indicated. |
| `CompositeGate` | `seam_coherence` (audited, no signal) + `strip_banding_score` (**unaudited**) | `seam_coherence` component carries no measurable signal on its own; `strip_banding_score` is imported into `bench_anime_stitch.py` (`_strip_banding_score`, line 75) but never called anywhere in `_compute_all_metrics` — it has **zero** correlation coverage against the human-labeled corpus. It cannot currently be judged "inverse" or "correct" — it has simply never been measured. |

## Recommendation (not applied — needs the promotion-ladder review)

- `GhostGate` is the strongest single candidate for demotion/rework: it gates
  on the metric with the worst measured inverse correlation in the whole
  audit. Options for a follow-up change: drop it, replace its signal with
  something audited-correct (e.g. a `seam_visibility`-style discontinuity
  measure instead of the FFT-autocorrelation ghost score), or keep it only as
  a non-authoritative telemetry field rather than an accept/reject gate.
- `CompositeGate`'s `strip_banding_score` component needs to be added to
  `_compute_all_metrics` (or a standalone audit script run directly against
  saved outputs) before anyone can claim it is or isn't trustworthy — right
  now it is an unaudited gate input, which is itself worth fixing regardless
  of what the number turns out to be.
- `SeamVisGate` has real support from this data; leave it as the reference
  "gate that works" while the other two are reworked.
- `cqas` needs the same fix path as `GhostGate` (drop/reweight the inverse
  `ghosting_siqe` and no-signal `seam_coherence` components) before it can be
  trusted as the single-scalar ranking used in reports/dashboards for
  GT-less cases. This overlaps M2.5a (#32, "per-defect-category correlation
  analysis... data-driven subset selection") — flagging there so it isn't
  independently rediscovered; this report only establishes that `cqas`
  currently fails the audit, not a replacement formula.

## Data gap (flagging, not fixing here)

The post-M1 ungated 97-case run (2026-08-16, the one M2 should really be
audited against) was executed in disjoint ranges across multiple manual
invocations after `_checkpoint.json`-based resume was needed (see the
Image-Toolkit bus, Claude/grok 2026-08-16 entries). No single JSON with all 97
post-M1 cases' `metrics_asp`/`metrics_simple` was ever saved — the checkpoint
file was cleared after completion and each range wrote its own partial output
file. This audit therefore necessarily used the pre-M0 2026-08-07 baseline
(the same evidence base the M0 relabel and the original bus claim used, so the
comparison is apples-to-apples with what's already been decided on), not the
newer ungated run. Re-running this audit against a consolidated post-M1 97-case
file would be a good cheap follow-up once one exists — worth having the next
full-corpus run merge its per-range outputs into one file instead of leaving
them scattered, purely as an evidence-hygiene improvement.

## Verification

`.venv/bin/python backend/benchmark/audit_gate_correlation.py --run ... --labels ...`
run live, output matches this report exactly (n=97, 0 skipped). No pipeline
code changed; no pytest run needed (new file has no existing test surface).
