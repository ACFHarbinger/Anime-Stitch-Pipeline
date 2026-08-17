"""M2 gate-signal audit: Spearman correlation of no-reference metrics vs human judgment.

Roadmap §5 M2 requires auditing the existing Safe ASP gate signals against the
completed human labels, and demoting/removing signals that are inversely
correlated with human quality. Grounds that requirement in a reproducible
computation instead of a one-off, unlinked number.

For each reviewed test case, correlates (ASP metric − SCANS metric) against
(human ASP score − human SCANS score), after re-orienting each metric's delta
so that positive always means "the metric says ASP got better" (per that
metric's own higher/lower-is-better design intent). A well-behaved no-reference
metric should then have rho > 0; rho < 0 means the metric moves the wrong way
relative to human preference and should not gate/demote toward SCANS on its
own value.

Usage::

    .venv/bin/python backend/benchmark/audit_gate_correlation.py \\
        --run backend/benchmark/output/anime_stitch_20260807_045552.json \\
        --labels data/benchmarks/asp_evaluations_20260810.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import spearmanr

METRIC_KEYS = [
    "sharpness", "edge_energy_score", "ghosting_siqe", "seam_coherence",
    "seam_visibility", "cqas", "coverage", "color_entropy", "seam_gradient",
]

# Higher raw value = better output, per each metric's own docstring in
# safety_metrics.py / bench_anime_stitch.py. Lower-is-better metrics get their
# delta negated before correlating.
HIGHER_IS_BETTER = {
    "sharpness": True,
    "edge_energy_score": True,
    "ghosting_siqe": False,   # higher = more periodic ghosting (worse)
    "seam_coherence": False,  # higher = more row-luminance variance/banding (worse)
    "seam_visibility": False,  # higher = harder seam cut (worse)
    "cqas": True,              # already a normalized goodness aggregate
    "coverage": True,
    "color_entropy": True,    # not a clear quality direction; reference only
    "seam_gradient": False,   # higher = more discontinuity at seam rows (worse)
}


def audit(run_path: Path, labels_path: Path) -> list[tuple[str, float, float, int]]:
    runs = json.loads(run_path.read_text())["datasets"]
    human = json.loads(labels_path.read_text())

    human_delta: list[float] = []
    metric_delta: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    skipped = 0
    for d in runs:
        h = human.get(d["name"])
        ma, ms = d.get("metrics_asp") or {}, d.get("metrics_simple") or {}
        if not h or not h.get("reviewed") or not ma or not ms:
            skipped += 1
            continue
        human_delta.append(h["asp"] - h["simple"])
        for k in METRIC_KEYS:
            va, vs = ma.get(k), ms.get(k)
            metric_delta[k].append(va - vs if va is not None and vs is not None else float("nan"))

    print(f"n_used={len(human_delta)} skipped={skipped}")
    results = []
    for k in METRIC_KEYS:
        pairs = [(a, b) for a, b in zip(metric_delta[k], human_delta) if not math.isnan(a)]
        if len(pairs) < 5:
            continue
        sign = 1.0 if HIGHER_IS_BETTER[k] else -1.0
        rho, p = spearmanr([sign * a for a, _ in pairs], [b for _, b in pairs])
        results.append((k, rho, p, len(pairs)))

    results.sort(key=lambda r: r[1])
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="benchmark run JSON with metrics_asp/metrics_simple")
    ap.add_argument("--labels", type=Path, required=True, help="asp_evaluations_*.json human-label file")
    args = ap.parse_args()

    results = audit(args.run, args.labels)
    print()
    print("Sign convention: delta re-oriented so positive = 'metric says ASP got better'.")
    print("rho > 0 = metric agrees with human preference direction (as intended).")
    print("rho < 0 = metric is INVERSELY correlated with human preference (misleading).")
    print()
    for k, rho, p, n in results:
        flag = (
            "  <-- INVERSE / MISLEADING vs human judgment" if rho < -0.2
            else "  (works as intended)" if rho > 0.2
            else "  (no clear signal)"
        )
        print(f"{k:20s} rho={rho:+.3f} p={p:.4f} n={n}{flag}")


if __name__ == "__main__":
    main()
