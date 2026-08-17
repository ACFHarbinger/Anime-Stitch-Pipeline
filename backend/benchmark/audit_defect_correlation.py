"""M2.5a (#32) Defect-Category & Stage-Attributed Signal Correlation Audit.

Roadmap §5 M2.5a extends the corpus-wide metric audit to analyze which failure
classes (torn anatomy, banding, color shift, crop loss, ...) each no-reference
metric actually tracks versus inverts on, and correlates metric behavior against
attributed pipeline stages.

Outputs:
1. Subset correlation: Spearman rho on cases where defect D is present.
2. Defect incidence correlation: correlation between oriented metric delta and
   defect absence indicator (1 = defect absent / clean, 0 = defect present).
3. Stage attribution mapping failure classes to responsible pipeline stages.
4. JSON export for website dashboard visualization.

Usage:
    .venv/bin/python backend/benchmark/audit_defect_correlation.py \\
        --run backend/benchmark/output/anime_stitch_20260807_045552.json \\
        --labels data/benchmarks/asp_evaluations_20260810.json \\
        --json-out docs/website/public/data/defect_correlation_matrix.json
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

from scipy.stats import spearmanr

ASP_ROOT = Path(__file__).resolve().parents[2]

METRIC_KEYS = [
    "sharpness",
    "edge_energy_score",
    "ghosting_siqe",
    "seam_coherence",
    "seam_visibility",
    "cqas_v1_legacy",
    "coverage",
    "color_entropy",
    "seam_gradient",
    "strip_banding_score",
]

HIGHER_IS_BETTER: dict[str, bool] = {
    "sharpness": True,
    "edge_energy_score": True,
    "ghosting_siqe": False,  # lower = less ghosting
    "seam_coherence": False,  # lower = less variance/banding
    "seam_visibility": False,  # lower = softer seam transition
    "cqas_v1_legacy": True,  # historic diagnostic aggregate
    "coverage": True,
    "color_entropy": True,  # reference metric
    "seam_gradient": False,  # lower = smoother seam gradient
    "strip_banding_score": False,  # lower = less inter-strip jump
}

METRIC_LABELS: dict[str, str] = {
    "sharpness": "Sobel Sharpness",
    "edge_energy_score": "Edge Energy",
    "ghosting_siqe": "SIQE Ghosting",
    "seam_coherence": "Seam Coherence",
    "seam_visibility": "Seam Visibility",
    "cqas_v1_legacy": "CQAS v1 (Legacy)",
    "coverage": "Coverage Ratio",
    "color_entropy": "Color Entropy",
    "seam_gradient": "Seam Gradient",
    "strip_banding_score": "Strip Banding",
}

# Pipeline Stage Attribution for Failure Classes
DEFECT_STAGE_MAP: dict[str, dict[str, str]] = {
    "torn_anatomy": {
        "stage_id": "stage_05_08",
        "stage_name": "Stage 5–8: Feature Matching & Bundle Adjustment",
        "category": "structural",
    },
    "misordered_content": {
        "stage_id": "stage_05_08",
        "stage_name": "Stage 5–8: Affine Pose Ordering",
        "category": "structural",
    },
    "duplicated_strip": {
        "stage_id": "stage_05_08",
        "stage_name": "Stage 5–8: Pairwise Displacement Filter",
        "category": "structural",
    },
    "geometry_warp": {
        "stage_id": "stage_05_08",
        "stage_name": "Stage 5–8: Affine/Homography Solve",
        "category": "structural",
    },
    "ghosting": {
        "stage_id": "stage_09",
        "stage_name": "Stage 9: Temporal Median & Foreground Masking",
        "category": "temporal",
    },
    "crop_loss": {
        "stage_id": "stage_08_09",
        "stage_name": "Stage 8–9: Canvas Geometry & Crop Bounds",
        "category": "canvas",
    },
    "blur": {
        "stage_id": "stage_09",
        "stage_name": "Stage 9: Multi-Frame Temporal Blending",
        "category": "temporal",
    },
    "banding": {
        "stage_id": "stage_11",
        "stage_name": "Stage 11: Least-Squares Photometric Normalization",
        "category": "photometric",
    },
    "color_shift": {
        "stage_id": "stage_11",
        "stage_name": "Stage 11: Color Balance & Gain Normalization",
        "category": "photometric",
    },
    "seam_line": {
        "stage_id": "stage_11",
        "stage_name": "Stage 11: Dynamic Programming Seam Cutting",
        "category": "photometric",
    },
    "other": {
        "stage_id": "unassigned",
        "stage_name": "Uncategorized / Miscellaneous",
        "category": "other",
    },
}


def classify_diagnosis(rho: float | None, n: int) -> str:
    if rho is None or n < 5 or math.isnan(rho):
        return "insufficient_data"
    if rho > 0.2:
        return "tracks_quality"
    if rho < -0.2:
        return "inverse_misleading"
    return "no_signal"


def compute_defect_correlation_matrix(
    run_data: dict[str, Any],
    human_labels: dict[str, Any],
    *,
    min_defect_samples: int = 5,
) -> dict[str, Any]:
    """Compute the full per-defect and per-stage correlation matrix."""
    datasets = run_data.get("datasets", [])
    
    cases: list[dict[str, Any]] = []
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
        defects = list(h.get("defects", []))
        fb = bool(d.get("used_fallback"))
        
        m_deltas: dict[str, float | None] = {}
        for k in METRIC_KEYS:
            legacy_key = "cqas" if k == "cqas_v1_legacy" else k
            va = ma.get(k, ma.get(legacy_key))
            vs = ms.get(k, ms.get(legacy_key))
            if va is not None and vs is not None:
                sign = 1.0 if HIGHER_IS_BETTER[k] else -1.0
                m_deltas[k] = float(sign * (va - vs))
            else:
                m_deltas[k] = None
        
        cases.append({
            "name": name,
            "h_asp": h_asp,
            "h_simple": h_simple,
            "h_delta": h_delta,
            "defects": set(defects),
            "fallback": fb,
            "m_deltas": m_deltas,
        })

    total_cases = len(cases)
    
    # Identify all observed defect categories
    defect_counts = Counter(d for c in cases for d in c["defects"])
    all_defects = sorted(defect_counts.keys(), key=lambda d: (-defect_counts[d], d))
    
    # 1. Overall Corpus Correlation
    overall_metrics: dict[str, dict[str, Any]] = {}
    for k in METRIC_KEYS:
        pairs = [(c["m_deltas"][k], c["h_delta"]) for c in cases if c["m_deltas"][k] is not None]
        if len(pairs) >= 5:
            rho, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
            r_val = round(float(rho), 3) if not math.isnan(rho) else None
            p_val = round(float(p), 4) if not math.isnan(p) else None
            overall_metrics[k] = {
                "metric_key": k,
                "label": METRIC_LABELS.get(k, k),
                "higher_is_better": HIGHER_IS_BETTER[k],
                "rho": r_val,
                "p_value": p_val,
                "n": len(pairs),
                "diagnosis": classify_diagnosis(r_val, len(pairs)),
            }
        else:
            overall_metrics[k] = {
                "metric_key": k,
                "label": METRIC_LABELS.get(k, k),
                "higher_is_better": HIGHER_IS_BETTER[k],
                "rho": None,
                "p_value": None,
                "n": len(pairs),
                "diagnosis": "insufficient_data",
            }

    # 2. Per-Defect Subset Correlation (rho on cases exhibiting defect D)
    defect_subset_matrix: dict[str, dict[str, Any]] = {}
    
    # 3. Defect Absence / Detection Correlation (point-biserial with clean indicator)
    defect_detection_matrix: dict[str, dict[str, Any]] = {}
    
    defect_summaries: dict[str, dict[str, Any]] = {}

    for defect in all_defects:
        count = defect_counts[defect]
        sub = [c for c in cases if defect in c["defects"]]
        stage_info = DEFECT_STAGE_MAP.get(defect, {
            "stage_id": "unknown",
            "stage_name": "Unknown Stage",
            "category": "other",
        })
        
        mean_asp = sum(c["h_asp"] for c in sub) / len(sub) if sub else 0.0
        mean_simple = sum(c["h_simple"] for c in sub) / len(sub) if sub else 0.0
        
        defect_summaries[defect] = {
            "defect": defect,
            "label": defect.replace("_", " ").title(),
            "count": count,
            "prevalence_pct": round((count / total_cases) * 100, 1) if total_cases > 0 else 0.0,
            "mean_asp_score": round(mean_asp, 2),
            "mean_simple_score": round(mean_simple, 2),
            "stage_id": stage_info["stage_id"],
            "stage_name": stage_info["stage_name"],
            "category": stage_info["category"],
        }

        # Subset correlation
        subset_row: dict[str, Any] = {}
        for k in METRIC_KEYS:
            pairs = [(c["m_deltas"][k], c["h_delta"]) for c in sub if c["m_deltas"][k] is not None]
            if len(pairs) >= min_defect_samples:
                rho, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
                r_val = round(float(rho), 3) if not math.isnan(rho) else None
                p_val = round(float(p), 4) if not math.isnan(p) else None
                subset_row[k] = {
                    "rho": r_val,
                    "p_value": p_val,
                    "n": len(pairs),
                    "diagnosis": classify_diagnosis(r_val, len(pairs)),
                }
            else:
                subset_row[k] = {
                    "rho": None,
                    "p_value": None,
                    "n": len(pairs),
                    "diagnosis": "insufficient_data",
                }
        defect_subset_matrix[defect] = subset_row

        # Detection correlation (indicator: 1 if defect absent / clean, 0 if defect present)
        detection_row: dict[str, Any] = {}
        for k in METRIC_KEYS:
            pairs = [
                (c["m_deltas"][k], 1 if defect not in c["defects"] else 0)
                for c in cases
                if c["m_deltas"][k] is not None
            ]
            if len(pairs) >= 5:
                rho, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
                r_val = round(float(rho), 3) if not math.isnan(rho) else None
                p_val = round(float(p), 4) if not math.isnan(p) else None
                detection_row[k] = {
                    "rho": r_val,
                    "p_value": p_val,
                    "n": len(pairs),
                    "diagnosis": classify_diagnosis(r_val, len(pairs)),
                }
            else:
                detection_row[k] = {
                    "rho": None,
                    "p_value": None,
                    "n": len(pairs),
                    "diagnosis": "insufficient_data",
                }
        defect_detection_matrix[defect] = detection_row

    # 4. Stage Group Summary Aggregations
    stage_groups: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "stage_id": "",
        "stage_name": "",
        "category": "",
        "defects": [],
        "total_defect_instances": 0,
        "metrics_avg_rho": {},
    })

    for defect, summary in defect_summaries.items():
        cat = summary["category"]
        grp = stage_groups[cat]
        grp["stage_id"] = summary["stage_id"]
        grp["stage_name"] = summary["stage_name"]
        grp["category"] = cat
        grp["defects"].append(defect)
        grp["total_defect_instances"] += summary["count"]

    # Compute average rho per stage category
    for cat, grp in stage_groups.items():
        avg_rho: dict[str, float | None] = {}
        for k in METRIC_KEYS:
            rhos = [
                defect_detection_matrix[d][k]["rho"]
                for d in grp["defects"]
                if defect_detection_matrix[d][k]["rho"] is not None
            ]
            avg_rho[k] = round(sum(rhos) / len(rhos), 3) if rhos else None
        grp["metrics_avg_rho"] = avg_rho

    return {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "total_reviewed_cases": total_cases,
        "metric_catalog": {
            k: {
                "key": k,
                "label": METRIC_LABELS.get(k, k),
                "higher_is_better": HIGHER_IS_BETTER[k],
            }
            for k in METRIC_KEYS
        },
        "overall_corpus_correlation": overall_metrics,
        "defect_summaries": defect_summaries,
        "defect_subset_correlation": defect_subset_matrix,
        "defect_detection_correlation": defect_detection_matrix,
        "stage_groups": dict(stage_groups),
    }


def format_cli_report(matrix_data: dict[str, Any]) -> str:
    """Format an ASCII / markdown statistical report for CLI output."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("M2.5a (#32) PER-DEFECT & STAGE-ATTRIBUTED CORRELATION AUDIT")
    lines.append("=" * 80)
    lines.append(f"Total reviewed cases analyzed: {matrix_data['total_reviewed_cases']}")
    lines.append("")
    
    # 1. Overall Corpus Ranking
    lines.append("1. OVERALL CORPUS METRIC RANKING (Oriented delta vs human delta):")
    lines.append("-" * 75)
    lines.append(f"{'Metric':<24} {'Oriented rho':>14} {'p-value':>10} {'N':>6}  {'Diagnosis'}")
    lines.append("-" * 75)
    sorted_overall = sorted(
        matrix_data["overall_corpus_correlation"].values(),
        key=lambda x: (x["rho"] if x["rho"] is not None else -999),
        reverse=True,
    )
    for m in sorted_overall:
        rho_str = f"{m['rho']:>+14.3f}" if m["rho"] is not None else f"{'N/A':>14}"
        p_str = f"{m['p_value']:>10.4f}" if m["p_value"] is not None else f"{'N/A':>10}"
        lines.append(f"{m['label']:<24} {rho_str} {p_str} {m['n']:>6}  {m['diagnosis'].upper()}")
    lines.append("-" * 75)
    lines.append("")

    # 2. Defect Detection Correlation Matrix
    lines.append("2. DEFECT ABSENCE DETECTION MATRIX (Positive rho = metric rewards clean outputs):")
    lines.append("   (Measures whether higher metric delta discriminates against specific defect classes)")
    lines.append("-" * 105)
    header = f"{'Defect (Stage)':<26}" + "".join(f"{k[:9]:>8}" for k in METRIC_KEYS)
    lines.append(header)
    lines.append("-" * 105)
    
    for defect, summary in matrix_data["defect_summaries"].items():
        row_label = f"{summary['label']} ({summary['count']}) [{summary['category'][:4].upper()}]"
        row = f"{row_label:<26}"
        for k in METRIC_KEYS:
            cell = matrix_data["defect_detection_correlation"][defect][k]
            if cell["rho"] is not None:
                row += f"{cell['rho']:>+8.2f}"
            else:
                row += f"{'--':>8}"
        lines.append(row)
    lines.append("-" * 105)
    lines.append("")

    # 3. Stage-Attribution Insights
    lines.append("3. PIPELINE STAGE ATTRIBUTION SUMMARY (Average Discrimination rho):")
    lines.append("-" * 80)
    for cat, grp in matrix_data["stage_groups"].items():
        lines.append(f"• Category [{cat.upper()}] — {grp['stage_name']}")
        lines.append(f"  Defects: {', '.join(grp['defects'])} (Total tags: {grp['total_defect_instances']})")
        top_tracking = sorted(
            [(k, v) for k, v in grp["metrics_avg_rho"].items() if v is not None],
            key=lambda x: x[1],
            reverse=True,
        )
        if top_tracking:
            best_k, best_v = top_tracking[0]
            worst_k, worst_v = top_tracking[-1]
            lines.append(f"  Best tracking metric:  {METRIC_LABELS[best_k]} (avg rho = {best_v:+.3f})")
            lines.append(f"  Worst inverting metric: {METRIC_LABELS[worst_k]} (avg rho = {worst_v:+.3f})")
        lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Path to automated benchmark run JSON (e.g. anime_stitch_*.json)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to human evaluations JSON (asp_evaluations_*.json)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to output the full correlation matrix as JSON",
    )
    args = parser.parse_args()

    run_data = json.loads(args.run.read_text())
    human_labels = json.loads(args.labels.read_text())

    matrix = compute_defect_correlation_matrix(run_data, human_labels)
    report = format_cli_report(matrix)
    print(report)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(matrix, indent=2) + "\n")
        print(f"Exported JSON matrix to {args.json_out}")


if __name__ == "__main__":
    main()
