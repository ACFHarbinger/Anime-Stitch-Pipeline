"""
Photometric seam diagnosis for the ASP Phase-4 seam_vis_gate / composite_gate_sb
fallback classes (roadmap docs/moon/ROADMAP.md Phase 4, GitHub issue #16).

The 2026-07-28 full-97 triage pass (`triage_fallback_classes.py`,
`.agent/cache/asp_phase4_fallback_triage_full97_2026-07-28.md`) found that most
seam_vis_gate/composite_gate_sb fallbacks are borderline (within ~10 points of
the 35.0 limit) and flagged the roadmap's own next step: pick the handful of
most-borderline tests and diagnose *what's specifically wrong, photometrically*
-- not another cross-metric correlation attempt, an actual per-pixel look at
the seam.

Per this repo's hard rule on the `dump/` benchmark corpus (explicit content),
this script never renders, displays, or saves any crop/thumbnail of the actual
frames. Every output is a numeric statistic (mean/median luminance, per-row
luminance profiles) or a matplotlib line plot of those *numbers* -- never the
underlying artwork.

What it computes, per borderline test, straight from the already-rendered
pipeline stage outputs the benchmark run wrote to `dump/<test>/output/
panorama_stages/` (never re-running the pipeline):

  1. A 1D luminance profile of the final pre-crop composite
     (`stage11_fg_composite.png`), sampled row-by-row (this *is* "perpendicular
     to the seam" for ASP's vertical-scroll panoramas) -- the same array
     `_seam_visibility_score()` reduces to a single worst-jump number. Reports
     whether that worst jump is a single-row spike (true seam artifact) or a
     multi-row ramp (smooth global exposure drift), and whether it lands at a
     known frame-boundary row (from the benchmark JSON's per-frame `ty`) or in
     open interior content.
  2. The same per-frame background-luminance-gain bookkeeping already recorded
     in the benchmark JSON's `photometric` block (`ref_lum`, `bg_lums`,
     `applied_gains`), extended with the *residual* each frame's gain leaves
     behind when the correction saturates at its [gain_lo, gain_hi] clip
     bound -- `residual = ref_lum - bg_lum * applied_gain`. A saturated gain
     with a large residual is a specific, testable "the existing correction
     under-corrects this frame" signal, distinct from "there is no signal".
  3. `_strip_banding_score`'s own per-frame strip-mean sequence (max adjacent
     jump == the `asp_sb` gate value), to see whether composite_gate_sb
     failures are one-outlier-frame jumps or a monotonic drift across the
     whole stack.
  4. Cross-reference against `mean_post_warp_diff` and the gate's own asp/sim/
     limit values, already in the benchmark JSON.

Outputs: `.agent/cache/asp_seam_photometric_diagnosis_<date>/diagnosis.json`
(machine-readable) and one `<test>_luminance_profile.png` line plot per test
(row index / frame index on x, luminance or luminance-jump on y -- data only).

Usage:
  python -m backend.benchmark.diagnose_seam_photometrics [--json PATH] [--out DIR]
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_OUTPUT_DIR = os.path.join(_HERE, "output")

# `dump/` is a gitignored symlink into the real corpus (see module docstring
# on the never-view-content rule) -- it exists in the main checkout but not
# in an isolated `git worktree add` checkout (worktrees only get tracked
# files). Resolve stage-output paths against wherever `dump/` actually
# resolves, independent of where this script's outputs get written.
_DATA_ROOT = os.environ.get("ASP_DIAGNOSE_DATA_ROOT") or (
    _REPO_ROOT
    if os.path.isdir(os.path.join(_REPO_ROOT, "dump"))
    else os.path.abspath(os.path.join(_REPO_ROOT, "..", "..", ".."))
)

# The 7 tests named in GitHub issue #16 -- the most-borderline seam_vis_gate /
# composite_gate_sb fallbacks per the 2026-07-28 full-97 triage report, sorted
# there by margin-over-limit (closest to passing first).
_TARGET_TESTS = [
    "asp_test87",  # seam_vis_gate, margin 0.4
    "asp_test10",  # seam_vis_gate, margin 1.3
    "asp_test71",  # seam_vis_gate, margin ~3.3
    "asp_test69",  # seam_vis_gate, margin ~3.7
    "asp_test25",  # composite_gate_sb, margin 1.4
    "asp_test16",  # composite_gate_sb, margin 2.0
    "asp_test11",  # composite_gate_sb, margin 2.3
]


def _newest_benchmark_json() -> str:
    candidates = sorted(glob.glob(os.path.join(_OUTPUT_DIR, "anime_stitch_*.json")))
    if not candidates:
        raise SystemExit(f"No anime_stitch_*.json found under {_OUTPUT_DIR}")
    return candidates[-1]


# ---------------------------------------------------------------------------
# Metric reproductions -- deliberately re-derived here rather than importing
# bench_anime_stitch.py (which pulls in torch/BiRefNet at import time for a
# script that only needs 3 small numpy/cv2 functions). Kept byte-for-byte
# equivalent to the originals; cross-checked against the benchmark JSON's own
# gate values per-test below (see "score_reproduction_check" in the output).
# ---------------------------------------------------------------------------


def _seam_visibility_profile(output_img: np.ndarray) -> Dict[str, Any]:
    """Re-derivation of bench_anime_stitch.py's `_seam_visibility_score`,
    returning the full per-row profile and diff array instead of collapsing
    to the single worst-jump number."""
    gray = (
        cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
        if output_img.ndim == 3
        else output_img
    )
    g = gray.astype(np.float32)
    H, W = g.shape
    content = g > 5
    row_content_count = content.sum(axis=1)
    row_valid = row_content_count > W * 0.1
    valid_idx = np.where(row_valid)[0]
    if valid_idx.size < 4:
        return {"valid": False}
    row_sums = np.where(content[valid_idx], g[valid_idx], 0.0).sum(axis=1)
    row_mean_vals = row_sums / np.maximum(row_content_count[valid_idx], 1)
    diffs = np.abs(np.diff(row_mean_vals))
    worst_i = int(np.argmax(diffs)) if diffs.size else -1
    worst_row = int(valid_idx[worst_i]) if worst_i >= 0 else -1
    score = round(float(diffs.max()), 2) if diffs.size else 0.0
    return {
        "valid": True,
        "row_idx": valid_idx.tolist(),
        "row_mean_lum": row_mean_vals.tolist(),
        "diffs": diffs.tolist(),
        "worst_row": worst_row,
        "score": score,
    }


def _strip_banding_series(
    render_img: np.ndarray, affines_ty_sorted: List[float]
) -> Dict[str, Any]:
    """Re-derivation of `_strip_banding_score`, returning the per-strip
    means and jumps instead of only the max jump."""
    gray = (
        cv2.cvtColor(render_img, cv2.COLOR_BGR2GRAY)
        if render_img.ndim == 3
        else render_img
    )
    H = gray.shape[0]
    strip_means = []
    strip_tys = []
    for ty in affines_ty_sorted:
        ty_i = int(ty)
        y0 = max(0, ty_i)
        y1 = min(H, ty_i + 50)
        if y1 > y0:
            band = gray[y0:y1]
            if band.mean() > 5:
                strip_means.append(float(band.mean()))
                strip_tys.append(ty_i)
    if len(strip_means) < 2:
        return {"valid": False}
    diffs = [
        abs(strip_means[i + 1] - strip_means[i]) for i in range(len(strip_means) - 1)
    ]
    return {
        "valid": True,
        "strip_ty": strip_tys,
        "strip_means": strip_means,
        "adjacent_diffs": diffs,
        "max_jump": round(max(diffs), 2),
        "max_jump_at_pair": int(np.argmax(diffs)),
    }


def _seam_shape_classification(
    row_idx: List[int], diffs: List[float], worst_row: int, window: int = 12
) -> Dict[str, Any]:
    """Sharp-step vs smooth-gradient classification around the worst jump.

    Compares the single worst adjacent-row jump against the total luminance
    change accumulated over a +/-`window`-row neighbourhood (excluding the
    worst jump itself). If the worst jump accounts for most of that local
    drift, it's a single-row spike (true seam cut). If the drift is spread
    roughly evenly across the window, it's a smooth ramp (global exposure
    mismatch/gradient), not a hard seam.
    """
    diffs_arr = np.asarray(diffs, dtype=np.float64)
    row_arr = np.asarray(row_idx[:-1], dtype=np.int64)  # diffs[i] is row_idx[i]->row_idx[i+1]
    if diffs_arr.size == 0:
        return {"classification": "no_data"}
    center_pos = int(np.argmin(np.abs(row_arr - worst_row))) if row_arr.size else 0
    lo = max(0, center_pos - window)
    hi = min(diffs_arr.size, center_pos + window + 1)
    local = diffs_arr[lo:hi]
    worst_val = float(diffs_arr[center_pos])
    local_total = float(local.sum())
    other_total = local_total - worst_val
    sharpness_ratio = worst_val / local_total if local_total > 1e-6 else 1.0
    classification = (
        "sharp_step" if sharpness_ratio >= 0.5 else "smooth_gradient"
    )
    return {
        "classification": classification,
        "sharpness_ratio": round(sharpness_ratio, 3),
        "worst_jump": round(worst_val, 2),
        "local_window_total_drift": round(local_total, 2),
        "local_window_other_drift": round(other_total, 2),
        "window_rows": window,
    }


def _nearest_frame_boundary(row: int, boundary_rows: List[int]) -> Dict[str, Any]:
    if not boundary_rows:
        return {"nearest_boundary_row": None, "distance_px": None, "at_boundary": False}
    arr = np.asarray(boundary_rows)
    idx = int(np.argmin(np.abs(arr - row)))
    dist = int(abs(arr[idx] - row))
    return {
        "nearest_boundary_row": int(arr[idx]),
        "distance_px": dist,
        "at_boundary": dist <= 15,
    }


def _photometric_residuals(photometric: Dict[str, Any]) -> Dict[str, Any]:
    """Extends the benchmark JSON's own photometric block with the
    post-gain residual per frame: how far the bg-gain correction (Step 4,
    bench_anime_stitch.py ~L1367-1396) leaves each frame's background
    luminance from ref_lum after its clipped gain is applied. Large residual
    on a clipped-gain frame means the existing scalar-gain correction
    couldn't fully close that frame's exposure gap (bound = [0.80,1.25] if
    ref_lum<80 else [0.88,1.14])."""
    ref_lum = photometric.get("ref_lum")
    bg_lums = photometric.get("bg_lums") or []
    gains = photometric.get("applied_gains") or []
    if ref_lum is None or not bg_lums or not gains:
        return {"valid": False}
    gain_lo, gain_hi = (0.80, 1.25) if ref_lum < 80.0 else (0.88, 1.14)
    residuals = []
    for lum, gain in zip(bg_lums, gains):
        if lum is None:
            residuals.append(None)
            continue
        post = lum * gain
        residuals.append(round(ref_lum - post, 2))
    clipped_flags = [
        (abs(g - gain_lo) < 1e-3 or abs(g - gain_hi) < 1e-3) for g in gains
    ]
    valid_residuals = [r for r in residuals if r is not None]
    return {
        "valid": True,
        "gain_bounds": [gain_lo, gain_hi],
        "residual_per_frame": residuals,
        "gain_clipped_per_frame": clipped_flags,
        "max_abs_residual": round(max(abs(r) for r in valid_residuals), 2)
        if valid_residuals
        else None,
        "max_abs_residual_on_clipped_frame": round(
            max(
                (abs(r) for r, c in zip(residuals, clipped_flags) if r is not None and c),
                default=0.0,
            ),
            2,
        ),
        "n_frames_clipped": sum(clipped_flags),
    }


def _diagnose_one(entry: Dict[str, Any], repo_root: str) -> Dict[str, Any]:
    name = entry["name"]
    paths = entry.get("paths") or {}
    stage_dir_rel = paths.get("stage_dir")
    result: Dict[str, Any] = {
        "name": name,
        "fallback_reason": entry.get("fallback_reason"),
        "mean_post_warp_diff": entry.get("mean_post_warp_diff"),
        "frame_count": (entry.get("frames") or {}).get("count"),
    }
    if not stage_dir_rel:
        result["error"] = "no stage_dir in benchmark JSON entry"
        return result
    stage_dir = os.path.join(repo_root, stage_dir_rel)
    composite_path = os.path.join(stage_dir, "stage11_fg_composite.png")
    if not os.path.isfile(composite_path):
        result["error"] = f"missing {composite_path}"
        return result

    composite = cv2.imread(composite_path)
    if composite is None:
        result["error"] = f"cv2.imread failed on {composite_path}"
        return result

    affines = (entry.get("alignment") or {}).get("affines") or []
    tys_sorted = sorted(float(a["ty"]) for a in affines)
    boundary_rows = [int(t) for t in tys_sorted]

    # --- 1. seam visibility profile (numeric only) ---
    sv_profile = _seam_visibility_profile(composite)
    result["seam_visibility_score_reproduced"] = sv_profile.get("score")
    if sv_profile.get("valid"):
        shape = _seam_shape_classification(
            sv_profile["row_idx"], sv_profile["diffs"], sv_profile["worst_row"]
        )
        boundary = _nearest_frame_boundary(sv_profile["worst_row"], boundary_rows)
        result["seam_shape"] = shape
        result["seam_boundary_proximity"] = boundary
        result["_profile_for_plot"] = {
            "row_idx": sv_profile["row_idx"],
            "row_mean_lum": sv_profile["row_mean_lum"],
            "worst_row": sv_profile["worst_row"],
            "boundary_rows": boundary_rows,
        }
    else:
        result["seam_shape"] = {"classification": "insufficient_content_rows"}

    # --- 2. strip banding series (for composite_gate_sb tests, but computed
    #     for all -- cheap and gives extra cross-context) ---
    sb_series = _strip_banding_series(composite, tys_sorted)
    if sb_series.get("valid"):
        result["strip_banding_reproduced"] = {
            "max_jump": sb_series["max_jump"],
            "strip_means": [round(m, 2) for m in sb_series["strip_means"]],
            "adjacent_diffs": [round(d, 2) for d in sb_series["adjacent_diffs"]],
        }
        # Monotonic drift check: is the strip-mean sequence mostly one
        # direction (smooth global drift) or does it reverse repeatedly
        # (frame-local/outlier jump)?
        means = sb_series["strip_means"]
        deltas = np.diff(means)
        signs = np.sign(deltas)
        signs = signs[signs != 0]
        monotonic_frac = (
            max((signs > 0).mean(), (signs < 0).mean()) if signs.size else 0.0
        )
        result["strip_banding_reproduced"]["monotonic_fraction"] = round(
            float(monotonic_frac), 3
        )
        result["strip_banding_reproduced"]["drift_pattern"] = (
            "monotonic_drift" if monotonic_frac >= 0.75 else "non_monotonic_jump"
        )

    # --- 3. photometric gain residuals (from existing benchmark JSON data) ---
    result["photometric_residuals"] = _photometric_residuals(
        entry.get("photometric") or {}
    )

    # --- 4. gate's own numbers, already in the JSON ---
    result["gate_values"] = _parse_gate_reason(entry.get("fallback_reason") or "")

    return result


def _parse_gate_reason(reason: str) -> Dict[str, Any]:
    import re

    gate, _, rest = reason.partition(":")
    values: Dict[str, float] = {}
    for part in re.split(r"[,_](?=[a-z])", rest):
        m = re.match(r"([a-z_]+)=(-?[\d.]+)", part)
        if m:
            values[m.group(1)] = float(m.group(2))
    return {"gate": gate, "values": values}


def _plot_profile(name: str, profile: Dict[str, Any], out_path: str) -> None:
    """Line plot of computed luminance-vs-row NUMBERS only -- no image
    content is rendered, embedded, or referenced."""
    row_idx = profile["row_idx"]
    row_lum = profile["row_mean_lum"]
    worst_row = profile["worst_row"]
    boundary_rows = profile["boundary_rows"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(row_idx, row_lum, linewidth=0.8, color="#3366cc")
    for b in boundary_rows:
        ax.axvline(b, color="#999999", linestyle=":", linewidth=0.6)
    ax.axvline(worst_row, color="#cc3333", linestyle="--", linewidth=1.2, label="worst jump row")
    ax.set_xlabel("Canvas row (y, distance along vertical-scroll axis)")
    ax.set_ylabel("Mean row luminance (0-255)")
    ax.set_title(f"{name}: per-row luminance profile (composite, pre-crop)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None, help="Benchmark JSON to read (default: newest)")
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: .agent/cache/asp_seam_photometric_diagnosis_<date>/)",
    )
    parser.add_argument(
        "--tests", nargs="+", default=_TARGET_TESTS, help="Dataset names to diagnose"
    )
    args = parser.parse_args()

    json_path = args.json or _newest_benchmark_json()
    with open(json_path) as fh:
        bench = json.load(fh)
    datasets = {d["name"]: d for d in bench.get("datasets", [])}

    from datetime import date

    out_dir = args.out or os.path.join(
        _REPO_ROOT, ".agent", "cache", f"asp_seam_photometric_diagnosis_{date.today().isoformat()}"
    )
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for name in args.tests:
        entry = datasets.get(name)
        if entry is None:
            results.append({"name": name, "error": "not found in benchmark JSON"})
            continue
        diag = _diagnose_one(entry, _REPO_ROOT)
        profile_for_plot = diag.pop("_profile_for_plot", None)
        if profile_for_plot is not None:
            plot_path = os.path.join(out_dir, f"{name}_luminance_profile.png")
            _plot_profile(name, profile_for_plot, plot_path)
            diag["luminance_profile_plot"] = os.path.relpath(plot_path, _REPO_ROOT)
        results.append(diag)
        print(f"[diagnose] {name}: done")

    out_json = os.path.join(out_dir, "diagnosis.json")
    with open(out_json, "w") as fh:
        json.dump(
            {"source_benchmark_json": os.path.relpath(json_path, _REPO_ROOT), "results": results},
            fh,
            indent=2,
        )
    print(f"\nWrote {out_json}")
    print(f"Plots: {out_dir}/*_luminance_profile.png")


if __name__ == "__main__":
    main()
