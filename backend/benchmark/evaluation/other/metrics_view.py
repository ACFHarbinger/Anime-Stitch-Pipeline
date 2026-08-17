"""Flattens one dataset's ``bench_anime_stitch.py`` result block into
display-ready rows and chart series.

The old dashboard loaded this entire block and then showed none of it (issue
#123 defect 7). Everything here is a pure read of already-computed benchmark
output — no metric is recomputed, so a number in the UI is always the same
number the report and the verdict logic used.

Metric directions are taken from the definitions in ``bench_anime_stitch.py``
rather than guessed, since several read backwards from their names:
``seam_coherence`` is the std of per-row mean luminance (a *banding* proxy, so
lower is better despite "coherence"), ``edge_energy_score`` is a double-Sobel
sharpness proxy and explicitly *not* a ghosting measure, and ``ghosting_siqe``
is the true ghosting signal on a 0-100 scale where lower is clean.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from ..constants.schema import (
    IMAGE_ASP,
    IMAGE_HUGIN,
    IMAGE_OVERMIX,
    IMAGE_SIMPLE,
    SCORABLE_KEYS,
)

HIGHER_BETTER = "higher"
LOWER_BETTER = "lower"
NEUTRAL = "neutral"

# Which `metrics_*` block in the results JSON belongs to which comparator key.
METRICS_BLOCK = {
    IMAGE_ASP: "metrics_asp",
    IMAGE_SIMPLE: "metrics_simple",
    IMAGE_OVERMIX: "metrics_overmix",
    IMAGE_HUGIN: "metrics_hugin",
}

# (json key, display label, direction, unit/format hint)
CV_METRICS: tuple[tuple[str, str, str, str], ...] = (
    ("cqas_v1_legacy", "CQAS v1 (legacy diagnostic; not a verdict)", NEUTRAL, "{:.4f}"),
    ("sharpness", "Sharpness (Laplacian)", HIGHER_BETTER, "{:.2f}"),
    ("ghosting_siqe", "Ghosting (SIQE 0-100)", LOWER_BETTER, "{:.2f}"),
    ("seam_visibility", "Seam visibility", LOWER_BETTER, "{:.2f}"),
    ("seam_coherence", "Seam coherence (banding)", LOWER_BETTER, "{:.2f}"),
    ("seam_gradient", "Mean seam gradient", LOWER_BETTER, "{:.3f}"),
    ("edge_energy_score", "Edge energy (sharpness proxy)", HIGHER_BETTER, "{:.4f}"),
    ("coverage", "Coverage", HIGHER_BETTER, "{:.4f}"),
    ("color_entropy", "Colour entropy", NEUTRAL, "{:.4f}"),
    ("ghost_seam_max", "Worst per-seam ghost", LOWER_BETTER, "{:.2f}"),
)

GT_METRICS: tuple[tuple[str, str, str, str], ...] = (
    ("aligned_ssim_vs_gt", "Aligned SSIM vs GT", HIGHER_BETTER, "{:.4f}"),
    ("ssim_vs_gt", "Raw SSIM vs GT", HIGHER_BETTER, "{:.4f}"),
    ("psnr_vs_gt", "PSNR vs GT (dB)", HIGHER_BETTER, "{:.2f}"),
    ("seam_coherence", "Seam coherence (banding)", LOWER_BETTER, "{:.2f}"),
)

# ghosting_siqe taxonomy from `_compute_cqas` / the GhostGate comments:
# 0 = clean, 30+ = ghost likely, 60+ = ghost confirmed. Used to colour the
# per-seam quality strip (#69's 11.1) against the pipeline's own thresholds
# rather than a generic 0.8/0.6 split.
GHOST_GOOD_MAX = 30.0
GHOST_WARN_MAX = 60.0

# `alignment.dy_steps` outlier rule from #69's 11.2: a step beyond 2x the
# median is the frame that hurts bundle adjustment.
STEP_OUTLIER_FACTOR = 2.0

# `photometric.applied_gains` deviating from 1.0 by more than this is the
# banding-risk flag from #69's 11.3.
GAIN_DEVIATION_FLAG = 0.15

# Absolute 0-1 normalizers for individual diagnostic axes. These historic
# thresholds were once used by CQAS v1 but no aggregate score is shown or used
# for ranking: CQAS v1 failed the human-label audit. Min-max-across-comparators
# remains actively misleading because it pins the winner at 1.0 and the loser
# at 0.0 regardless of how close they actually are.
RADAR_SCALES: tuple[tuple[str, str, float, str], ...] = (
    ("sharpness", "Sharpness", 100.0, HIGHER_BETTER),
    ("coverage", "Coverage", 1.0, HIGHER_BETTER),
    ("ghosting_siqe", "Ghosting", 60.0, LOWER_BETTER),
    ("seam_visibility", "Seam visibility", 25.0, LOWER_BETTER),
    ("seam_coherence", "Banding", 50.0, LOWER_BETTER),
)


def radar_value(metric_key: str, value: float | None) -> float | None:
    """Normalize one individual diagnostic axis to 0-1 (1 = best)."""
    for key, _label, reference, direction in RADAR_SCALES:
        if key != metric_key:
            continue
        if value is None:
            return None
        ratio = value / reference
        score = ratio if direction == HIGHER_BETTER else 1.0 - ratio
        return float(min(1.0, max(0.0, score)))
    return None


def radar_rows(entry: dict, keys: Sequence[str] | None = None) -> list[MetricRow]:
    """Radar axes as ``MetricRow``s carrying *normalized* values, so the chart
    builder stays a pure plot of numbers computed here."""
    keys = list(keys) if keys is not None else present_comparators(entry)
    rows = []
    for metric_key, label, _reference, direction in RADAR_SCALES:
        values = {
            key: radar_value(
                metric_key,
                _as_float((entry.get(METRICS_BLOCK[key]) or {}).get(metric_key)),
            )
            for key in keys
        }
        if any(v is not None for v in values.values()):
            rows.append(MetricRow(metric_key, label, direction, "{:.3f}", values))
    return rows


@dataclasses.dataclass
class MetricRow:
    """One metric across every comparator that reported it."""

    key: str
    label: str
    direction: str
    fmt: str
    values: dict[str, float | None]

    def best_key(self) -> str | None:
        """Which comparator wins this row, or ``None`` when the metric has no
        direction or fewer than two comparators reported it."""
        if self.direction == NEUTRAL:
            return None
        present = {k: v for k, v in self.values.items() if v is not None}
        if len(present) < 2:
            return None
        pick = max if self.direction == HIGHER_BETTER else min
        return pick(present, key=lambda k: present[k])

    def formatted(self, key: str) -> str:
        value = self.values.get(key)
        return "—" if value is None else self.fmt.format(value)


def _as_float(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def present_comparators(entry: dict) -> list[str]:
    """Comparator keys with a non-empty metrics block in this result."""
    return [key for key in SCORABLE_KEYS if entry.get(METRICS_BLOCK[key])]


def cv_metric_rows(entry: dict, keys: Sequence[str] | None = None) -> list[MetricRow]:
    keys = list(keys) if keys is not None else present_comparators(entry)
    rows = []
    for metric_key, label, direction, fmt in CV_METRICS:
        values = {
            key: _as_float(
                (entry.get(METRICS_BLOCK[key]) or {}).get(
                    metric_key,
                    # Old result JSONs used the unversioned name.  Preserve
                    # their diagnostic visibility without restoring a rankable
                    # aggregate field.
                    (entry.get(METRICS_BLOCK[key]) or {}).get("cqas")
                    if metric_key == "cqas_v1_legacy"
                    else None,
                )
            )
            for key in keys
        }
        if any(v is not None for v in values.values()):
            rows.append(MetricRow(metric_key, label, direction, fmt, values))
    return rows


def gt_metric_rows(entry: dict) -> list[MetricRow]:
    """Ground-truth comparison rows (#69's 11.5). Empty when this test has no
    ground truth, which is 42 of the 97."""
    gt = entry.get("ground_truth") or {}
    if not gt.get("available"):
        return []
    blocks = {IMAGE_ASP: gt.get("metrics_asp") or {}, IMAGE_SIMPLE: gt.get("metrics_simple") or {}}
    rows = []
    for metric_key, label, direction, fmt in GT_METRICS:
        values = {key: _as_float(block.get(metric_key)) for key, block in blocks.items()}
        if any(v is not None for v in values.values()):
            rows.append(MetricRow(metric_key, label, direction, fmt, values))
    return rows


def headline_facts(entry: dict) -> list[tuple[str, str]]:
    """The short "what happened to this test" summary, as label/value pairs."""
    if not entry:
        return [("Metrics", "no benchmark result found for this test")]
    comparison = entry.get("comparison") or {}
    gt = entry.get("ground_truth") or {}
    frames = entry.get("frames") or {}
    time = entry.get("time") or {}
    facts: list[tuple[str, str]] = [
        ("Verdict", str(comparison.get("verdict", "—"))),
        ("Verdict source", str(comparison.get("verdict_source", "—"))),
        ("Composite", "SCANS fallback" if entry.get("used_fallback") else "true ASP composite"),
    ]
    if entry.get("fallback_reason"):
        facts.append(("Fallback reason", str(entry["fallback_reason"])))
    facts.append(("Ground truth", "available" if gt.get("available") else "none"))
    if frames.get("count") is not None:
        facts.append(("Frames used", f"{frames['count']} @ {frames.get('source_w')}x{frames.get('source_h')}"))
    phases = entry.get("phases") or {}
    if phases.get("count") is not None:
        facts.append(("Animation phases", str(phases["count"])))
    if entry.get("mean_post_warp_diff") is not None:
        facts.append(("Mean post-warp diff", f"{_as_float(entry['mean_post_warp_diff']):.2f}"))
    health = entry.get("affine_health") or {}
    if health:
        facts.append((
            "Affine health",
            f"{'ok' if health.get('valid') else 'INVALID'} — {health.get('reason', '—')}",
        ))
    if time.get("total_sec") is not None:
        facts.append(("Total time", f"{_as_float(time['total_sec']):.1f} s"))
    return facts


def pipeline_config_rows(entry: dict) -> list[tuple[str, str]]:
    config = entry.get("pipeline_config") or {}
    return [(key, str(value)) for key, value in sorted(config.items())]


# ---------------------------------------------------------------------------
# Chart series (#69's 11.1-11.4)
# ---------------------------------------------------------------------------


def seam_ghost_series(entry: dict, keys: Sequence[str] | None = None) -> dict[str, list[float]]:
    """Per-seam SIQE ghost scores per comparator (11.1). Often empty — the
    pipeline only emits these for true multi-strip composites."""
    keys = list(keys) if keys is not None else present_comparators(entry)
    series = {}
    for key in keys:
        scores = (entry.get(METRICS_BLOCK[key]) or {}).get("ghost_seam_scores") or []
        floats = [f for f in (_as_float(s) for s in scores) if f is not None]
        if floats:
            series[key] = floats
    return series


def ghost_band(score: float) -> str:
    """Threshold band for a per-seam ghost score: ``good``/``warn``/``bad``."""
    if score < GHOST_GOOD_MAX:
        return "good"
    if score < GHOST_WARN_MAX:
        return "warn"
    return "bad"


@dataclasses.dataclass
class AlignmentSeries:
    frames: list[int]
    tx: list[float]
    ty: list[float]
    dx_steps: list[float]
    dy_steps: list[float]
    dx_cv: float | None
    dy_cv: float | None
    outlier_steps: list[int]  # indices into dy_steps beyond 2x the median


def alignment_series(entry: dict) -> AlignmentSeries | None:
    """Per-frame translation and inter-frame steps (11.2)."""
    alignment = entry.get("alignment") or {}
    affines = alignment.get("affines") or []
    if not affines:
        return None
    frames = [int(a.get("frame", i)) for i, a in enumerate(affines)]
    tx = [_as_float(a.get("tx")) or 0.0 for a in affines]
    ty = [_as_float(a.get("ty")) or 0.0 for a in affines]
    dy_steps = [_as_float(s) or 0.0 for s in alignment.get("dy_steps") or []]
    dx_steps = [_as_float(s) or 0.0 for s in alignment.get("dx_steps") or []]
    outliers: list[int] = []
    if dy_steps:
        magnitudes = sorted(abs(s) for s in dy_steps)
        median = magnitudes[len(magnitudes) // 2]
        if median > 0:
            limit = STEP_OUTLIER_FACTOR * median
            outliers = [i for i, s in enumerate(dy_steps) if abs(s) > limit]
    return AlignmentSeries(
        frames=frames, tx=tx, ty=ty,
        dx_steps=dx_steps, dy_steps=dy_steps,
        dx_cv=_as_float(alignment.get("dx_cv")),
        dy_cv=_as_float(alignment.get("dy_cv")),
        outlier_steps=outliers,
    )


@dataclasses.dataclass
class PhotometricSeries:
    bg_lums: list[float]
    applied_gains: list[float]
    ref_lum: float | None
    frames_corrected: int | None
    gain_range: list[float] | None
    flagged_frames: list[int]  # gains deviating from 1.0 by >15%


def photometric_series(entry: dict) -> PhotometricSeries | None:
    """Per-frame background luminance and applied gain (11.3)."""
    photometric = entry.get("photometric") or {}
    gains = [_as_float(g) for g in photometric.get("applied_gains") or []]
    lums = [_as_float(v) for v in photometric.get("bg_lums") or []]
    if not gains and not lums:
        return None
    clean_gains = [g if g is not None else 1.0 for g in gains]
    return PhotometricSeries(
        bg_lums=[v if v is not None else 0.0 for v in lums],
        applied_gains=clean_gains,
        ref_lum=_as_float(photometric.get("ref_lum")),
        frames_corrected=photometric.get("frames_corrected"),
        gain_range=photometric.get("gain_range"),
        flagged_frames=[i for i, g in enumerate(clean_gains) if abs(g - 1.0) > GAIN_DEVIATION_FLAG],
    )


@dataclasses.dataclass
class MatchingSummary:
    raw_edges: int | None
    filtered_edges: int | None
    methods: dict[str, int]
    weights: list[float]
    n_pts: list[int]
    frame_gaps: list[int]  # j - i per edge; 1 = adjacent, >1 = a skip link


def matching_summary(entry: dict) -> MatchingSummary | None:
    """Matcher method mix and per-edge weight/support (11.4)."""
    matching = entry.get("matching") or {}
    edges = matching.get("edges") or []
    if not edges and not matching.get("methods"):
        return None
    weights, n_pts, gaps = [], [], []
    for edge in edges:
        weight = _as_float(edge.get("weight"))
        pts = edge.get("n_pts")
        if weight is None or pts is None:
            continue
        weights.append(weight)
        n_pts.append(int(pts))
        gaps.append(int(edge.get("j", 0)) - int(edge.get("i", 0)))
    return MatchingSummary(
        raw_edges=matching.get("raw_edges"),
        filtered_edges=matching.get("filtered_edges"),
        methods={str(k): int(v) for k, v in (matching.get("methods") or {}).items()},
        weights=weights, n_pts=n_pts, frame_gaps=gaps,
    )


def timing_breakdown(entry: dict) -> list[tuple[str, float]]:
    """Per-stage wall time, largest first, excluding the total and the
    non-time counters that share the ``time`` block."""
    time = entry.get("time") or {}
    stages = []
    for key, value in time.items():
        if key == "total_sec" or not key.endswith("_sec"):
            continue
        seconds = _as_float(value)
        if seconds:
            stages.append((key[: -len("_sec")].replace("_", " "), seconds))
    return sorted(stages, key=lambda kv: kv[1], reverse=True)


def frame_selection_stages(entry: dict) -> list[tuple[str, int]]:
    """Frame counts through the selection funnel (11.7's per-test view)."""
    selection = entry.get("frame_selection") or {}
    order = (
        ("original_count", "original"),
        ("smart_select_count", "after smart select"),
        ("spatial_dedup_count", "after spatial dedup"),
        ("final_count", "final"),
    )
    return [(label, int(selection[key])) for key, label in order if selection.get(key) is not None]
