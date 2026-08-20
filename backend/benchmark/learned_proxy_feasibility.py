"""
M2.5b feasibility spike (issue #33): can a learned proxy predict human
coherence from existing per-case metrics?

Non-gating diagnostic only. This script does NOT ship a model — it reports
whether one is worth building at the current corpus size, and by how much
a simple calibrated baseline beats a naive constant-mean predictor under a
leakage-safe, grouped-by-source-sequence cross-validation.

Usage::

    PYTHONPATH=<image-toolkit-root> .venv/bin/python \\
        backend/benchmark/learned_proxy_feasibility.py

Reads ``backend/benchmark/output/anime_stitch_latest_consolidated.json``
(the frozen post-M1 ungated 97-run, issue #30). Writes nothing except a
report to stdout; the caller is responsible for saving findings.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

_ASP_ROOT = Path(__file__).resolve().parents[2]
_CONSOLIDATED = _ASP_ROOT / "backend/benchmark/output/anime_stitch_latest_consolidated.json"

# Features already computed per-case by the benchmark harness. Deliberately
# restricted to Raw-ASP-side signals available without a new GPU run.
_FEATURES = [
    "sharpness",
    "coverage",
    "seam_gradient",
    "color_entropy",
    "edge_energy_score",
    "ghosting_siqe",
    "seam_coherence",
    "seam_visibility",
    "strip_banding_score",
]

_SOURCE_RE = re.compile(r"^(.*?)\s*-\s*\d+\s*\[")


def _source_group(anime_path: str, name: str) -> str:
    """Best-effort source-sequence key so CV splits don't leak near-duplicate
    frames of the same episode across train/validation. Falls back to the
    case name itself (worst case: no grouping benefit, never worse than
    ungrouped)."""
    stem = Path(anime_path).name if anime_path else ""
    m = _SOURCE_RE.match(stem)
    return m.group(1).strip() if m else name


def load_cases() -> list[dict]:
    doc = json.loads(_CONSOLIDATED.read_text())
    cases = []
    for ds in doc["datasets"]:
        if ds.get("used_fallback"):
            continue  # not a genuine Raw ASP composite — would mislabel a SCANS/fallback image as ASP
        hc = ds.get("human_coherence") or {}
        if not hc.get("reviewed") or hc.get("asp") is None:
            continue
        feats = ds.get("metrics_asp") or {}
        if any(feats.get(f) is None for f in _FEATURES):
            continue
        cases.append(
            {
                "name": ds["name"],
                "group": _source_group(ds.get("anime_path", ""), ds["name"]),
                "y": float(hc["asp"]),
                "x": [float(feats[f]) for f in _FEATURES],
            }
        )
    return cases


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(set(xs)) < 2:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _ridge_fit(X: list[list[float]], y: list[float], lam: float = 1.0) -> list[float]:
    """Closed-form ridge regression via Gauss-Jordan — no numpy dependency
    beyond what's already vendored elsewhere; kept dependency-free here
    since this is a throwaway diagnostic, not shipped model code."""
    n = len(X)
    p = len(X[0]) + 1  # +1 bias column
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    rows = [[1.0] + row for row in X]
    for i in range(n):
        for a in range(p):
            b[a] += rows[i][a] * y[i]
            for c in range(p):
                A[a][c] += rows[i][a] * rows[i][c]
    for i in range(1, p):  # ridge penalty, skip bias
        A[i][i] += lam
    # Gauss-Jordan elimination
    M = [A[i] + [b[i]] for i in range(p)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-9:
            continue
        M[col], M[pivot] = M[pivot], M[col]
        piv = M[col][col]
        M[col] = [v / piv for v in M[col]]
        for r in range(p):
            if r != col:
                f = M[r][col]
                M[r] = [v - f * M[col][idx] for idx, v in enumerate(M[r])]
    return [M[i][p] for i in range(p)]


def _predict(coefs: list[float], x: list[float]) -> float:
    return coefs[0] + sum(c * v for c, v in zip(coefs[1:], x))


def leave_one_group_out(cases: list[dict]) -> dict:
    groups = sorted({c["group"] for c in cases})
    baseline_errs, model_errs = [], []
    for g in groups:
        train = [c for c in cases if c["group"] != g]
        test = [c for c in cases if c["group"] == g]
        if not train or not test:
            continue
        y_train = [c["y"] for c in train]
        mean_y = statistics.fmean(y_train)
        # Normalize features (train stats only) so ridge isn't dominated by scale.
        p = len(_FEATURES)
        mu = [statistics.fmean(c["x"][i] for c in train) for i in range(p)]
        sd = [statistics.pstdev(c["x"][i] for c in train) or 1.0 for i in range(p)]
        X_train = [[(c["x"][i] - mu[i]) / sd[i] for i in range(p)] for c in train]
        coefs = _ridge_fit(X_train, y_train, lam=1.0)
        for c in test:
            x_norm = [(c["x"][i] - mu[i]) / sd[i] for i in range(p)]
            pred = _predict(coefs, x_norm)
            model_errs.append((pred - c["y"]) ** 2)
            baseline_errs.append((mean_y - c["y"]) ** 2)
    return {
        "n_groups": len(groups),
        "n_cases": len(cases),
        "baseline_rmse": statistics.fmean(baseline_errs) ** 0.5 if baseline_errs else float("nan"),
        "model_rmse": statistics.fmean(model_errs) ** 0.5 if model_errs else float("nan"),
    }


def main() -> None:
    cases = load_cases()
    print(f"Loaded {len(cases)} true Raw ASP composites with reviewed human labels.")
    print(f"Distinct source-sequence groups: {len(set(c['group'] for c in cases))}")
    print()
    print("Per-feature Pearson r against human ASP coherence score:")
    ys = [c["y"] for c in cases]
    for i, feat in enumerate(_FEATURES):
        xs = [c["x"][i] for c in cases]
        print(f"  {feat:24s} r = {_pearson(xs, ys):+.3f}")
    print()
    result = leave_one_group_out(cases)
    print("Leave-one-source-group-out ridge regression vs constant-mean baseline:")
    print(f"  groups={result['n_groups']}  cases={result['n_cases']}")
    print(f"  baseline RMSE (predict train mean) = {result['baseline_rmse']:.3f}")
    print(f"  ridge-on-existing-metrics RMSE     = {result['model_rmse']:.3f}")
    if result["baseline_rmse"] and result["model_rmse"] < result["baseline_rmse"]:
        improvement = 100 * (1 - result["model_rmse"] / result["baseline_rmse"])
        print(f"  -> {improvement:.1f}% RMSE reduction over baseline")
    else:
        print("  -> no improvement over predicting the training-fold mean")


if __name__ == "__main__":
    main()
