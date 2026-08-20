# M2.5b — Learned Human-Coherence Proxy: Feasibility Spike (issue #33)

**Verdict: not yet feasible.** Do not build a learned proxy metric on the
current corpus. Revisit once the SFW corpus (#38-41) adds labeled cases.

## Data

Source: `backend/benchmark/output/anime_stitch_latest_consolidated.json`,
the frozen post-M1 ungated 97-run (#30, closed 2026-08-20), cross-referenced
against the reviewed human labels already embedded per case
(`human_coherence.asp`, 0-4 coherence score).

Filtered to cases where `used_fallback == False` (genuine Raw ASP composites
— a SCANS-fallback case relabeled `asp` must not train the proxy as if it
were real ASP output) and `human_coherence.reviewed == True`:

**27 true composites with reviewed labels** — smaller than the 43 cited in
the roadmap's §3 evidence baseline, because that number was measured against
the pre-M1 (2026-08-07) gated corpus; the M1 canonical-adapter pipeline
changed which cases produce a genuine (non-fallback) Raw ASP result. This is
the real, current denominator for this spike, not the older figure.

Source-sequence grouping (parsed from the frame filename's episode-title
prefix, to prevent near-duplicate frames of the same episode leaking across
a train/validation split): **27 distinct groups for 27 cases** — every true
composite in this corpus happens to come from a different source video, so
leave-one-group-out degenerates to plain leave-one-out here. Not a bug in
the grouping logic; a property of this particular case selection.

## Method

`backend/benchmark/learned_proxy_feasibility.py` (new, this issue):

1. Per-feature Pearson correlation of the 9 already-computed `metrics_asp`
   signals (sharpness, coverage, seam_gradient, color_entropy,
   edge_energy_score, ghosting_siqe, seam_coherence, seam_visibility,
   strip_banding_score) against the human ASP coherence score.
2. A ridge regression (closed-form, dependency-free, lam=1.0, features
   z-scored on the training fold only) over all 9 features, evaluated with
   leave-one-group-out CV, compared against a constant-mean baseline
   (predict the training fold's mean coherence score).

No deep model was attempted — `AnimeStitchNet`/`StitchTrainer` was
deliberately not reused (per the issue: it's a 4-DoF alignment regressor,
not a quality model), and a from-scratch deep model on n=27 would be
guaranteed to overfit far worse than the classical baseline already does.

## Results

| Feature | Pearson r vs human ASP coherence |
|---|---:|
| color_entropy | +0.328 |
| ghosting_siqe | +0.176 |
| seam_coherence | +0.160 |
| seam_gradient | +0.143 |
| sharpness | -0.148 |
| edge_energy_score | +0.078 |
| coverage | +0.052 |
| seam_visibility | -0.063 |
| strip_banding_score | -0.031 |

No feature individually clears |r| = 0.35. `color_entropy` is the strongest
signal found and it's weak — consistent with the M2 gate audit's finding
that the existing metric family (sharpness, edge_energy, ghosting) tracks
human judgment poorly or inversely (§3, rho -0.47/-0.53/-0.60 in the larger
mixed corpus).

Leave-one-group-out (=leave-one-out here):

- Baseline RMSE (predict training-fold mean): **1.058**
- Ridge-on-9-features RMSE: **1.536**
- **Ridge is 45% worse than the baseline.** With 9 features and 27 samples
  (≈3 samples/feature), even lam=1.0 ridge regularization is not enough to
  prevent the model from fitting fold-specific noise instead of signal.

## Conclusion

At n=27 true composites, no calibrated model built on the existing metric
family beats predicting the mean. This is not a implementation gap to fix
with a fancier model — it is a data-size and signal-quality ceiling:

1. **Too few samples.** 27 cases is below the threshold where any
   multi-feature model reliably beats a constant baseline, regardless of
   method.
2. **Weak available signal.** Even with infinite data, the existing metrics
   are individually weak predictors (max |r|=0.328) — a proxy built only on
   these features has a low ceiling until new, better-correlated signals
   exist (this is exactly what M2.5a's anime-adapted CV metrics work
   should eventually supply — see `.agent/reports/gemini/m2_5a_*`).

## Recommendation

- **Do not gate anything on a learned proxy now.** Correctly non-gating per
  the issue; this spike reinforces that decision rather than challenging it.
- **Revisit after the SFW corpus (#38-41)** adds labeled cases — even
  doubling or tripling n would meaningfully change the bias/variance
  picture for a 9-feature linear model, though a real answer likely still
  needs 100+ true composites.
- **Revisit after M2.5a's anime-adapted metrics land as usable features** —
  the current feature set's low individual correlations are the binding
  constraint, not the modeling method.
- Keep `backend/benchmark/learned_proxy_feasibility.py` as the reusable
  revalidation script the issue calls for — rerun it at each corpus
  milestone (roadmap M2.5 deliverable: "revalidated against held-out human
  ratings every time the corpus grows").
