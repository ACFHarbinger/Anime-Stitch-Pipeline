"""Similarity-based benchmark subset selection (Milestone §M2.5a).

Derives representative mini-benchmark subsets for fast-iteration deltas
without requiring a full 97-case benchmark run.

Constructs multi-dimensional feature vectors combining:
1. Automated image metrics (seam visibility, gradient, sharpness, ghosting, etc.)
2. Structural alignment features (frame count, displacement step, aspect ratio)
3. Human evaluation annotations (ASP/SCANS scores, defect indicator vectors)

Provides:
- Balanced representative subset selection (K-Medoids / stratified clustering)
- Scoped domain subsets (e.g. structural alignment focus for M3/M4, seam/photometric focus for M5)
- Representativeness verification (Spearman rank fidelity vs full 97-case corpus)

Usage:
    .venv/bin/python backend/benchmark/subset_selection.py \\
        --run backend/benchmark/output/anime_stitch_20260807_045552.json \\
        --labels data/benchmarks/asp_evaluations_20260810.json \\
        --k 12 --mode balanced \\
        --json-out docs/website/public/data/benchmark_subsets.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

ASP_ROOT = Path(__file__).resolve().parents[2]

DEFECT_KEYS = [
    "ghosting",
    "crop_loss",
    "color_shift",
    "torn_anatomy",
    "seam_line",
    "banding",
    "misordered_content",
    "blur",
    "duplicated_strip",
    "geometry_warp",
    "other",
]

CORE_METRIC_KEYS = [
    "seam_visibility",
    "seam_gradient",
    "sharpness",
    "ghosting_siqe",
    "coverage",
]


@dataclass
class TestCaseFeatures:
    name: str
    h_asp: float
    h_simple: float
    h_delta: float
    defects: set[str]
    used_fallback: bool
    metrics_asp: dict[str, float]
    metrics_simple: dict[str, float]
    feature_vector: np.ndarray


def extract_features(
    datasets: list[dict[str, Any]],
    human_labels: dict[str, Any],
) -> list[TestCaseFeatures]:
    """Extract normalized feature representations for each reviewed test case."""
    raw_cases: list[dict[str, Any]] = []

    for d in datasets:
        name = d["name"]
        h = human_labels.get(name)
        if not h or not h.get("reviewed"):
            continue
        ma = dict(d.get("metrics_asp") or {})
        ms = dict(d.get("metrics_simple") or {})
        if not ma or not ms:
            continue

        h_asp = float(h.get("asp", 0.0))
        h_simple = float(h.get("simple", 0.0))
        h_delta = h_asp - h_simple
        defects = set(h.get("defects", []))
        fb = bool(d.get("used_fallback"))

        raw_cases.append({
            "name": name,
            "h_asp": h_asp,
            "h_simple": h_simple,
            "h_delta": h_delta,
            "defects": defects,
            "used_fallback": fb,
            "metrics_asp": ma,
            "metrics_simple": ms,
        })

    if not raw_cases:
        return []

    # Build numeric feature matrix for normalization
    matrix_rows: list[list[float]] = []
    for c in raw_cases:
        row: list[float] = [
            c["h_asp"],
            c["h_simple"],
            c["h_delta"],
            1.0 if c["used_fallback"] else 0.0,
        ]
        # Defect binary indicators
        for def_k in DEFECT_KEYS:
            row.append(1.0 if def_k in c["defects"] else 0.0)

        # Core metric values (ASP)
        for m_k in CORE_METRIC_KEYS:
            row.append(float(c["metrics_asp"].get(m_k, 0.0)))

        matrix_rows.append(row)

    mat = np.array(matrix_rows, dtype=np.float32)
    # Z-score normalization per column with epsilon std floor
    mean = np.mean(mat, axis=0)
    std = np.std(mat, axis=0)
    std[std < 1e-6] = 1.0
    norm_mat = (mat - mean) / std

    cases: list[TestCaseFeatures] = []
    for idx, c in enumerate(raw_cases):
        cases.append(
            TestCaseFeatures(
                name=c["name"],
                h_asp=c["h_asp"],
                h_simple=c["h_simple"],
                h_delta=c["h_delta"],
                defects=c["defects"],
                used_fallback=c["used_fallback"],
                metrics_asp=c["metrics_asp"],
                metrics_simple=c["metrics_simple"],
                feature_vector=norm_mat[idx],
            )
        )

    return cases


def select_k_medoids_subset(
    cases: list[TestCaseFeatures],
    k: int,
    *,
    weights: np.ndarray | None = None,
) -> list[TestCaseFeatures]:
    """Select k representative cases using greedy K-Medoids / MaxMin facility location."""
    if k >= len(cases):
        return cases

    feat_matrix = np.stack([c.feature_vector for c in cases])
    if weights is not None:
        feat_matrix = feat_matrix * weights

    dist_matrix = cdist(feat_matrix, feat_matrix, metric="euclidean")

    # Start with the medoid that minimizes sum of distances to all other points
    medoid_indices = [int(np.argmin(np.sum(dist_matrix, axis=1)))]

    # Greedily add points that maximize minimum distance to existing medoids
    while len(medoid_indices) < k:
        min_dists = np.min(dist_matrix[:, medoid_indices], axis=1)
        next_idx = int(np.argmax(min_dists))
        medoid_indices.append(next_idx)

    return [cases[i] for i in sorted(medoid_indices)]


def select_domain_scoped_subset(
    cases: list[TestCaseFeatures],
    k: int,
    domain: str,
) -> list[TestCaseFeatures]:
    """Select a subset focused on a specific pipeline failure domain.

    - 'structural': alignment / torn anatomy / pose ordering / duplicated strips (M3/M4 focus)
    - 'photometric': banding / seam lines / color shifts (M5 focus)
    - 'temporal': ghosting / blur (Stage 9 focus)
    - 'balanced': stratified across all failure classes
    """
    if domain == "structural":
        target_defects = {"torn_anatomy", "misordered_content", "duplicated_strip", "geometry_warp"}
    elif domain == "photometric":
        target_defects = {"banding", "seam_line", "color_shift"}
    elif domain == "temporal":
        target_defects = {"ghosting", "blur"}
    else:
        return select_k_medoids_subset(cases, k)

    # Filter cases exhibiting at least one target defect
    domain_cases = [c for c in cases if c.defects.intersection(target_defects)]
    if len(domain_cases) <= k:
        # Pad with nearest neighbors if domain pool is smaller than k
        remaining = [c for c in cases if c not in domain_cases]
        pad = select_k_medoids_subset(remaining, k - len(domain_cases))
        return domain_cases + pad

    return select_k_medoids_subset(domain_cases, k)


def evaluate_subset_fidelity(
    full_cases: list[TestCaseFeatures],
    subset_cases: list[TestCaseFeatures],
) -> dict[str, Any]:
    """Compute representativeness statistics comparing subset to the full corpus."""
    subset_names = {c.name for c in subset_cases}
    
    # 1. Defect Coverage Ratio
    full_defects = Counter(d for c in full_cases for d in c.defects)
    sub_defects = Counter(d for c in subset_cases for d in c.defects)
    coverage_ratio = len(sub_defects) / max(1, len(full_defects))

    # 2. Score Distribution Fidelity (Mean & Variance alignment)
    full_mean_asp = float(np.mean([c.h_asp for c in full_cases]))
    sub_mean_asp = float(np.mean([c.h_asp for c in subset_cases]))
    full_mean_simple = float(np.mean([c.h_simple for c in full_cases]))
    sub_mean_simple = float(np.mean([c.h_simple for c in subset_cases]))

    # 3. Preference Distribution Match
    full_prefs = Counter("asp" if c.h_delta > 0 else "simple" if c.h_delta < 0 else "tie" for c in full_cases)
    sub_prefs = Counter("asp" if c.h_delta > 0 else "simple" if c.h_delta < 0 else "tie" for c in subset_cases)
    
    # 4. Metric Rank Fidelity: Correlate oriented metric deltas across full vs subset
    metric_rhos_full: dict[str, float] = {}
    metric_rhos_sub: dict[str, float] = {}
    for m in CORE_METRIC_KEYS:
        pairs_f = [(c.metrics_asp.get(m, 0.0) - c.metrics_simple.get(m, 0.0), c.h_delta) for c in full_cases]
        pairs_s = [(c.metrics_asp.get(m, 0.0) - c.metrics_simple.get(m, 0.0), c.h_delta) for c in subset_cases]
        rf, _ = spearmanr([a for a, _ in pairs_f], [b for _, b in pairs_f])
        rs, _ = spearmanr([a for a, _ in pairs_s], [b for _, b in pairs_s])
        metric_rhos_full[m] = round(float(rf), 3) if not math.isnan(rf) else 0.0
        metric_rhos_sub[m] = round(float(rs), 3) if not math.isnan(rs) else 0.0

    return {
        "subset_size": len(subset_cases),
        "full_corpus_size": len(full_cases),
        "defect_coverage_ratio": round(coverage_ratio, 3),
        "covered_defects_count": len(sub_defects),
        "total_defects_count": len(full_defects),
        "mean_asp_score": {
            "subset": round(sub_mean_asp, 2),
            "full": round(full_mean_asp, 2),
            "abs_error": round(abs(sub_mean_asp - full_mean_asp), 2),
        },
        "mean_simple_score": {
            "subset": round(sub_mean_simple, 2),
            "full": round(full_mean_simple, 2),
            "abs_error": round(abs(sub_mean_simple - full_mean_simple), 2),
        },
        "preference_distribution": {
            "subset": {k: round(v / len(subset_cases), 3) for k, v in sub_prefs.items()},
            "full": {k: round(v / len(full_cases), 3) for k, v in full_prefs.items()},
        },
        "metric_rank_correlation_fidelity": {
            m: {"subset_rho": metric_rhos_sub[m], "full_rho": metric_rhos_full[m]}
            for m in CORE_METRIC_KEYS
        },
    }


def generate_all_standard_subsets(
    cases: list[TestCaseFeatures],
) -> dict[str, Any]:
    """Generate the standard suite of representative mini-benchmarks."""
    # 1. Balanced Smoke Set (K=10)
    sub_balanced_10 = select_k_medoids_subset(cases, 10)
    fid_balanced_10 = evaluate_subset_fidelity(cases, sub_balanced_10)

    # 2. Balanced Medium Set (K=20)
    sub_balanced_20 = select_k_medoids_subset(cases, 20)
    fid_balanced_20 = evaluate_subset_fidelity(cases, sub_balanced_20)

    # 3. Structural Alignment Focus (M3/M4, K=12)
    sub_structural_12 = select_domain_scoped_subset(cases, 12, "structural")
    fid_structural_12 = evaluate_subset_fidelity(cases, sub_structural_12)

    # 4. Photometric & Seam Focus (M5, K=12)
    sub_photometric_12 = select_domain_scoped_subset(cases, 12, "photometric")
    fid_photometric_12 = evaluate_subset_fidelity(cases, sub_photometric_12)

    return {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "total_corpus_cases": len(cases),
        "subsets": {
            "balanced_smoke_10": {
                "label": "Balanced Smoke Set (10 Cases)",
                "description": "Fast ~30s smoke validation covering all core defect archetypes with minimal variance to full corpus means.",
                "target_milestone": "General CI / Smoke",
                "cases": [c.name for c in sub_balanced_10],
                "fidelity": fid_balanced_10,
            },
            "balanced_medium_20": {
                "label": "Balanced Stratified Set (20 Cases)",
                "description": "High-fidelity 20-case mini-benchmark for comprehensive pull request checks.",
                "target_milestone": "Pre-Merge Gate",
                "cases": [c.name for c in sub_balanced_20],
                "fidelity": fid_balanced_20,
            },
            "structural_red_set_12": {
                "label": "Structural Alignment Set (12 Cases)",
                "description": "M3/M4 targeted benchmark focusing on torn anatomy, affine misordering, and duplicated strips.",
                "target_milestone": "M3 / M4 (Alignment)",
                "cases": [c.name for c in sub_structural_12],
                "fidelity": fid_structural_12,
            },
            "photometric_seam_set_12": {
                "label": "Photometric & Seams Set (12 Cases)",
                "description": "M5 targeted benchmark focusing on horizontal banding, color shift, and seam cut visibility.",
                "target_milestone": "M5 (Photometrics)",
                "cases": [c.name for c in sub_photometric_12],
                "fidelity": fid_photometric_12,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Path to automated benchmark run JSON",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to human evaluations JSON",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=12,
        help="Target subset size (default: 12)",
    )
    parser.add_argument(
        "--mode",
        choices=["balanced", "structural", "photometric", "temporal", "all_standard"],
        default="all_standard",
        help="Selection mode / target domain (default: all_standard)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to output the subset definitions as JSON",
    )
    args = parser.parse_args()

    run_data = json.loads(args.run.read_text())
    human_labels = json.loads(args.labels.read_text())
    cases = extract_features(run_data.get("datasets", []), human_labels)

    if args.mode == "all_standard":
        out_data = generate_all_standard_subsets(cases)
        print("=" * 80)
        print("M2.5a DATA-DRIVEN BENCHMARK SUBSETS GENERATED")
        print("=" * 80)
        for key, s in out_data["subsets"].items():
            fid = s["fidelity"]
            print(f"• {s['label']} ({len(s['cases'])} cases)")
            print(f"  Target: {s['target_milestone']}")
            print(f"  Cases: {', '.join(s['cases'])}")
            print(f"  Mean ASP Abs Error: {fid['mean_asp_score']['abs_error']:.2f} | Defect Coverage: {fid['defect_coverage_ratio']*100:.1f}%")
            print()
    else:
        if args.mode == "balanced":
            subset = select_k_medoids_subset(cases, args.k)
        else:
            subset = select_domain_scoped_subset(cases, args.k, args.mode)

        fidelity = evaluate_subset_fidelity(cases, subset)
        out_data = {
            "mode": args.mode,
            "k": args.k,
            "cases": [c.name for c in subset],
            "fidelity": fidelity,
        }
        print(f"Selected {len(subset)} cases for mode '{args.mode}':")
        print(", ".join(c.name for c in subset))
        print(f"Defect coverage: {fidelity['defect_coverage_ratio']*100:.1f}%")
        print(f"Mean ASP: subset={fidelity['mean_asp_score']['subset']} vs full={fidelity['mean_asp_score']['full']}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out_data, indent=2) + "\n")
        print(f"Exported JSON to {args.json_out}")


if __name__ == "__main__":
    main()
