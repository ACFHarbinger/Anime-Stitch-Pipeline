"""Per-test pipeline diagnostics charts — issue #69 (Analytics Phase 11)
sub-items 11.1-11.5, absorbed into the evaluation tool because they answer
"why did *this* test score badly" while a human is looking at it. The
corpus-wide half of Phase 11 (11.6 memory waterfall, 11.9 cross-run
regression, 11.10 experiment tracker) stays in #69.

Every builder takes the already-flattened series from ``other/metrics_view.py``
and returns a matplotlib ``Figure``. Nothing here recomputes a metric, so a
chart can never disagree with the report or the verdict logic. Any series the
benchmark didn't emit for a given test renders an explanation instead of an
empty axes — that is the normal case, not an error: 42 of 97 tests have no
ground truth, and per-seam ghost scores only exist for true multi-strip
composites.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from matplotlib.figure import Figure

from ..constants.schema import COMPARATOR_TITLES
from ..constants.user_interface import COL_ACCENT, COL_BAD, COL_GOOD, COL_TEXT_DIM, COL_WARN
from ..other import metrics_view as mv
from .figure_theme import empty_figure, themed_figure, themed_legend

_BAND_COLORS = {"good": COL_GOOD, "warn": COL_WARN, "bad": COL_BAD}
_SERIES_COLORS = (COL_ACCENT, COL_WARN, COL_GOOD, COL_BAD, "#c792ea")


def _title(key: str) -> str:
    return COMPARATOR_TITLES.get(key, key)


# ---------------------------------------------------------------------------
# 11.1 — Per-seam quality strip
# ---------------------------------------------------------------------------


def seam_quality_figure(entry: Dict) -> Figure:
    """Per-seam SIQE ghost score, one bar per inter-strip boundary, coloured
    against the pipeline's own ghost thresholds (clean < 30, ghost likely
    30-60, ghost confirmed >= 60 — from ``_compute_cqas``).

    ``ghosting_siqe`` is a whole-image average; this is the localised version
    that says *which* seam dragged it down.
    """
    series = mv.seam_ghost_series(entry)
    if not series:
        return empty_figure(
            "No per-seam ghost scores in this result.\n"
            "The pipeline only emits them for true multi-strip composites —\n"
            "a SCANS fallback has no ASP seams to score."
        )
    fig, axes = themed_figure(figsize=(8.5, 3.4 * len(series)), n_axes=len(series), nrows=len(series))
    if len(series) == 1:
        axes = [axes] if not isinstance(axes, list) else axes
    for ax, (key, scores) in zip(axes, series.items(), strict=False):
        positions = np.arange(1, len(scores) + 1)
        colors = [_BAND_COLORS[mv.ghost_band(s)] for s in scores]
        ax.bar(positions, scores, color=colors, edgecolor="#111", linewidth=0.5)
        ax.axhline(mv.GHOST_GOOD_MAX, color=COL_WARN, linestyle="--", linewidth=0.9, alpha=0.8)
        ax.axhline(mv.GHOST_WARN_MAX, color=COL_BAD, linestyle="--", linewidth=0.9, alpha=0.8)
        worst = int(np.argmax(scores))
        ax.annotate(
            f"worst seam #{worst + 1}: {scores[worst]:.1f}",
            xy=(worst + 1, scores[worst]), xytext=(0, 8), textcoords="offset points",
            ha="center", color=COL_TEXT_DIM, fontsize=8,
        )
        ax.set_title(f"{_title(key)} — per-seam ghost score (lower = cleaner)")
        ax.set_xlabel("Seam boundary (top to bottom)")
        ax.set_ylabel("SIQE 0-100")
        ax.set_xticks(positions)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 11.2 — Alignment drift
# ---------------------------------------------------------------------------


def alignment_drift_figure(entry: Dict) -> Figure:
    """Per-frame tx/ty and the inter-frame steps, with the >2x-median step
    outliers flagged — those are the frames that hurt bundle adjustment.

    A high ``dy_cv`` (non-uniform scroll speed) is the documented common
    cause of fallbacks, so both coefficients of variation are in the title
    rather than buried in a table.
    """
    series = mv.alignment_series(entry)
    if series is None:
        return empty_figure("No alignment.affines block in this result.")

    fig, axes = themed_figure(figsize=(8.5, 6.4), n_axes=2, nrows=2)
    ax_pos, ax_step = axes

    ax_pos.plot(series.frames, series.ty, color=COL_ACCENT, linewidth=1.6, marker="o",
                markersize=3, label="ty (scroll axis)")
    ax_pos.plot(series.frames, series.tx, color=COL_WARN, linewidth=1.4, marker="o",
                markersize=3, label="tx")
    ax_pos.set_title("Per-Frame Translation (bundle-adjusted affines)")
    ax_pos.set_xlabel("Frame")
    ax_pos.set_ylabel("Offset (px)")
    ax_pos.grid(True, color="#333", linewidth=0.5, alpha=0.6)
    themed_legend(ax_pos, fontsize=8)

    if series.dy_steps:
        step_x = np.arange(len(series.dy_steps))
        colors = [COL_BAD if i in series.outlier_steps else COL_ACCENT for i in step_x]
        ax_step.bar(step_x, series.dy_steps, color=colors, edgecolor="#111", linewidth=0.4,
                    label="dy step")
        if series.dx_steps:
            ax_step.plot(np.arange(len(series.dx_steps)), series.dx_steps, color=COL_WARN,
                         linewidth=1.2, marker="o", markersize=3, label="dx step")
        median = float(np.median([abs(s) for s in series.dy_steps]))
        ax_step.axhline(mv.STEP_OUTLIER_FACTOR * median, color=COL_BAD, linestyle="--",
                        linewidth=0.9, alpha=0.8, label=f"{mv.STEP_OUTLIER_FACTOR:g}x median")
        flagged = (
            f" — {len(series.outlier_steps)} outlier step(s): {series.outlier_steps}"
            if series.outlier_steps else " — no outlier steps"
        )
        ax_step.set_title(
            f"Inter-Frame Steps  (dy_cv={series.dy_cv:.4g}, dx_cv={series.dx_cv:.4g}){flagged}"
            if series.dy_cv is not None and series.dx_cv is not None
            else f"Inter-Frame Steps{flagged}"
        )
        ax_step.set_xlabel("Step index (frame i → i+1)")
        ax_step.set_ylabel("Delta (px)")
        ax_step.grid(True, color="#333", linewidth=0.5, alpha=0.6)
        themed_legend(ax_step, fontsize=8)
    else:
        ax_step.text(0.5, 0.5, "No inter-frame step data", ha="center", va="center",
                     color=COL_TEXT_DIM)
        ax_step.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 11.3 — Photometric correction profile
# ---------------------------------------------------------------------------


def photometric_figure(entry: Dict) -> Figure:
    """Per-frame background luminance (bars) against the applied gain (line,
    right axis), with gains deviating from 1.0 by more than 15% flagged —
    those are the frames most likely to introduce visible colour banding
    after compositing.
    """
    series = mv.photometric_series(entry)
    if series is None:
        return empty_figure("No photometric block in this result.")

    fig, ax = themed_figure(figsize=(8.5, 4.2))
    frames = np.arange(max(len(series.bg_lums), len(series.applied_gains)))
    if series.bg_lums:
        ax.bar(np.arange(len(series.bg_lums)), series.bg_lums, color="#2c5f7a",
               edgecolor="#111", linewidth=0.4, label="bg luminance")
    if series.ref_lum is not None:
        ax.axhline(series.ref_lum, color=COL_ACCENT, linestyle="--", linewidth=1.0,
                   alpha=0.9, label=f"reference luminance ({series.ref_lum:.1f})")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Background luminance")
    ax.grid(True, color="#333", linewidth=0.5, alpha=0.6)

    ax_gain = ax.twinx()
    ax_gain.set_facecolor("none")
    ax_gain.tick_params(colors="white")
    ax_gain.yaxis.label.set_color("white")
    if series.applied_gains:
        gain_x = np.arange(len(series.applied_gains))
        ax_gain.plot(gain_x, series.applied_gains, color=COL_WARN, linewidth=1.6,
                     marker="o", markersize=3, label="applied gain")
        if series.flagged_frames:
            ax_gain.scatter(
                series.flagged_frames, [series.applied_gains[i] for i in series.flagged_frames],
                color=COL_BAD, s=48, zorder=5, marker="X",
                label=f">{mv.GAIN_DEVIATION_FLAG:.0%} deviation",
            )
        ax_gain.axhline(1.0, color=COL_TEXT_DIM, linewidth=0.8, alpha=0.7)
    ax_gain.set_ylabel("Applied gain")

    corrected = series.frames_corrected
    span = (
        f"gain range {series.gain_range[0]:.3g}-{series.gain_range[-1]:.3g}"
        if series.gain_range else "gain range n/a"
    )
    ax.set_title(
        f"Photometric Profile — {corrected if corrected is not None else '?'}/{len(frames)} "
        f"frames corrected, {span}"
    )
    handles, labels = ax.get_legend_handles_labels()
    gain_handles, gain_labels = ax_gain.get_legend_handles_labels()
    legend = ax.legend(handles + gain_handles, labels + gain_labels, facecolor="#1a1a2e",
                       edgecolor="#444", labelcolor="white", fontsize=8, loc="best")
    if legend is not None:
        legend.get_frame().set_alpha(0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 11.4 — Edge quality & matching breakdown
# ---------------------------------------------------------------------------


def matching_figure(entry: Dict) -> Figure:
    """Matcher method mix (donut) beside per-edge weight vs support (scatter).

    High weight + high ``n_pts`` edges are reliable; low weight + few points
    are the noise candidates the spanning-tree filter is meant to drop. Point
    colour encodes the frame gap, since a skip-link (j - i > 1) failing is a
    different problem from an adjacent pair failing.
    """
    summary = mv.matching_summary(entry)
    if summary is None:
        return empty_figure("No matching block in this result.")

    fig, axes = themed_figure(figsize=(9.5, 4.2), n_axes=2)
    ax_mix, ax_scatter = axes

    if summary.methods:
        labels = list(summary.methods)
        counts = [summary.methods[k] for k in labels]
        colors = [_SERIES_COLORS[i % len(_SERIES_COLORS)] for i in range(len(labels))]
        ax_mix.pie(counts, labels=labels, colors=colors, autopct="%1.0f%%",
                   textprops={"color": "white", "fontsize": 8},
                   wedgeprops={"width": 0.45, "edgecolor": "#111"})
        ax_mix.set_title("Matcher Method Mix")
    else:
        ax_mix.text(0.5, 0.5, "No method breakdown", ha="center", va="center", color=COL_TEXT_DIM)
        ax_mix.axis("off")

    if summary.weights:
        gaps = np.array(summary.frame_gaps)
        scatter = ax_scatter.scatter(
            summary.n_pts, summary.weights, c=gaps, cmap="viridis", s=28,
            edgecolors="#111", linewidths=0.4,
        )
        bar = fig.colorbar(scatter, ax=ax_scatter, fraction=0.046)
        bar.set_label("frame gap (j − i)", color="white")
        bar.ax.tick_params(colors="white")
        ax_scatter.set_xlabel("Support (n_pts)")
        ax_scatter.set_ylabel("Edge weight")
        ax_scatter.grid(True, color="#333", linewidth=0.5, alpha=0.6)
        kept = summary.filtered_edges
        raw = summary.raw_edges
        ratio = f" ({100.0 * kept / raw:.0f}% kept)" if kept and raw else ""
        ax_scatter.set_title(f"Edge Quality — {kept}/{raw} edges after filter{ratio}")
    else:
        ax_scatter.text(0.5, 0.5, "No per-edge data", ha="center", va="center", color=COL_TEXT_DIM)
        ax_scatter.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 11.5 — Ground truth comparison
# ---------------------------------------------------------------------------


def gt_comparison_figure(entry: Dict, previous_entry: Optional[Dict] = None) -> Figure:
    """Grouped bars of every GT metric for ASP vs Simple.

    ``previous_entry`` (the same dataset from an older results JSON) turns on
    regression detection: a metric that moved against its good direction by
    more than 3% gets called out, matching the 3% margin
    ``bench_anime_stitch.py``'s own ``_gt_verdict`` uses to avoid
    noise-driven verdict flips.
    """
    rows = mv.gt_metric_rows(entry)
    if not rows:
        return empty_figure(
            "This test has no ground truth.\n"
            "42 of the 97 corpus tests are GT-less by design — use CQAS and\n"
            "the no-reference metrics for those."
        )
    previous_rows = {r.key: r for r in mv.gt_metric_rows(previous_entry)} if previous_entry else {}

    fig, ax = themed_figure(figsize=(8.5, 4.4))
    labels = [r.label for r in rows]
    positions = np.arange(len(rows))
    width = 0.36
    for offset, key, color in ((-width / 2, "asp", COL_ACCENT), (width / 2, "simple", COL_WARN)):
        # PSNR is on a dB scale an order of magnitude above the SSIM rows, so
        # bars are drawn against each row's own max — the comparison that
        # matters is ASP vs Simple within a row, not across rows.
        values, raw = [], []
        for row in rows:
            value = row.values.get(key)
            raw.append(value)
            row_max = max((v for v in row.values.values() if v is not None), default=1.0) or 1.0
            values.append((value / row_max) if value is not None else 0.0)
        bars = ax.bar(positions + offset, values, width, color=color, edgecolor="#111",
                      linewidth=0.5, label=_title(key))
        for bar, value in zip(bars, raw, strict=False):
            if value is None:
                continue
            ax.annotate(f"{value:.4g}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        color="white", fontsize=7)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("Value, normalized per metric")
    ax.set_ylim(0, 1.22)
    ax.grid(True, axis="y", color="#333", linewidth=0.5, alpha=0.6)
    themed_legend(ax, fontsize=8)

    regressions = []
    for row in rows:
        prev = previous_rows.get(row.key)
        if prev is None:
            continue
        for key in ("asp", "simple"):
            now, before = row.values.get(key), prev.values.get(key)
            if now is None or before is None or not before:
                continue
            delta = (now - before) / abs(before)
            worse = delta < -0.03 if row.direction == mv.HIGHER_BETTER else delta > 0.03
            if worse:
                regressions.append(f"{_title(key)} {row.label} {delta:+.1%}")
    verdict = (entry.get("ground_truth") or {}).get("verdict", "—")
    title = f"Ground Truth Comparison — GT verdict: {verdict}"
    if regressions:
        title += f"\nREGRESSION vs previous run: {'; '.join(regressions)}"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Supporting per-test charts
# ---------------------------------------------------------------------------


def timing_figure(entry: Dict) -> Figure:
    """Where this test's wall time went, largest stage first."""
    stages = mv.timing_breakdown(entry)
    if not stages:
        return empty_figure("No stage timings in this result.")
    fig, ax = themed_figure(figsize=(8.0, 0.45 * len(stages) + 1.6))
    labels = [name for name, _ in stages][::-1]
    values = [seconds for _, seconds in stages][::-1]
    ax.barh(np.arange(len(values)), values, color=COL_ACCENT, edgecolor="#111", linewidth=0.5)
    for i, value in enumerate(values):
        ax.annotate(f"{value:.1f}s", xy=(value, i), xytext=(4, 0), textcoords="offset points",
                    va="center", color="white", fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    total = (entry.get("time") or {}).get("total_sec")
    ax.set_title(f"Stage Timings — {total:.1f} s total" if total else "Stage Timings")
    ax.set_xlabel("Seconds")
    ax.grid(True, axis="x", color="#333", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return fig


def frame_selection_figure(entry: Dict) -> Figure:
    """This test's slice of 11.7's frame-selection funnel: how many frames
    survived each reduction stage."""
    stages = mv.frame_selection_stages(entry)
    if not stages:
        return empty_figure("No frame_selection block in this result.")
    fig, ax = themed_figure(figsize=(8.0, 3.8))
    labels = [name for name, _ in stages]
    counts = [count for _, count in stages]
    original = counts[0] or 1
    dropped = [original - c for c in counts]
    positions = np.arange(len(labels))
    ax.bar(positions, counts, color=COL_GOOD, edgecolor="#111", linewidth=0.5, label="kept")
    ax.bar(positions, dropped, bottom=counts, color="#3a3a52", edgecolor="#111",
           linewidth=0.5, label="dropped")
    for i, count in enumerate(counts):
        ax.annotate(f"{count}", xy=(i, count / 2), ha="center", va="center",
                    color="#0e0e18", fontsize=9, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("Frames")
    mode = (entry.get("frame_selection") or {}).get("selection_mode", "—")
    drop_pct = 100.0 * (original - counts[-1]) / original
    ax.set_title(f"Frame Selection — mode '{mode}', {drop_pct:.0f}% dropped overall")
    themed_legend(ax, fontsize=8)
    fig.tight_layout()
    return fig


def cv_metric_radar_figure(entry: Dict) -> Figure:
    """No-reference metrics on one radar, one polygon per comparator — the
    fast "who wins on what, and by how much" read before drilling into the
    table.

    Axes are normalized against ``_compute_cqas``'s own absolute references
    (ghosting/60, seam visibility/25, banding/50, sharpness/100, coverage and
    CQAS already 0-1), so a radius means what the pipeline's aggregate quality
    score means by it and 1.0 is "as good as the metric's scale goes".
    Deliberately *not* min-max normalized across the comparators present: with
    two comparators that pins the winner at 1.0 and the loser at 0.0 on every
    axis, which draws a dramatic star out of a 0.1% difference.
    """
    rows = mv.radar_rows(entry)
    keys = mv.present_comparators(entry)
    if not rows or len(keys) < 2:
        return empty_figure("Not enough comparator metrics for a radar (need 2+ with metrics).")

    angles = np.linspace(0, 2 * np.pi, len(rows), endpoint=False).tolist()
    angles.append(angles[0])
    fig, ax = themed_figure(figsize=(6.6, 5.6), projection="polar")
    for i, key in enumerate(keys):
        values = [row.values.get(key) or 0.0 for row in rows]
        values.append(values[0])
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        ax.plot(angles, values, color=color, linewidth=1.8, label=_title(key))
        ax.fill(angles, values, color=color, alpha=0.14)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([r.label for r in rows], fontsize=8, color="white")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=7, color=COL_TEXT_DIM)
    ax.set_title("No-Reference Metrics — CQAS scale, outward = better", fontsize=10)
    ax.grid(color="#444", linewidth=0.5)
    themed_legend(ax, fontsize=8, loc="upper right", bbox_to_anchor=(1.22, 1.12))
    fig.tight_layout()
    return fig


DIAGNOSTICS = (
    ("Metric radar", cv_metric_radar_figure),
    ("Per-seam ghost (11.1)", seam_quality_figure),
    ("Alignment drift (11.2)", alignment_drift_figure),
    ("Photometric (11.3)", photometric_figure),
    ("Matching (11.4)", matching_figure),
    ("Ground truth (11.5)", gt_comparison_figure),
    ("Frame selection", frame_selection_figure),
    ("Stage timings", timing_figure),
)
