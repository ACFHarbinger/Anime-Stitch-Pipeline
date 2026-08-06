"""Shared dark-theme matplotlib styling for every figure the tool builds.

Kept visually consistent with ``bench_anime_stitch.py``'s own plot styling
(``#12121f`` figure background, white text/ticks, inferno/plasma/magma
colormaps) so live visualizations read as the same tool family as its static
report plots.

This lives in ``logic/`` rather than ``ui/`` on purpose: every figure builder
is a pure ``ndarray``/dict -> ``Figure`` function with no Qt dependency, so the
same builders serve the Qt inspector, the FiftyOne surface, and a headless
script. ``ui/mpl_canvas.py`` imports *this*, never the reverse — the old
package had ``logic`` importing ``ui`` for exactly this helper.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

from matplotlib.figure import Figure

from ..constants.user_interface import AX_BG, COL_TEXT_DIM, FIG_BG


def style_axis(ax) -> None:
    ax.set_facecolor(AX_BG)
    ax.title.set_color("white")
    ax.tick_params(colors="white")
    for axis_name in ("xaxis", "yaxis", "zaxis"):
        axis = getattr(ax, axis_name, None)
        if axis is not None:
            axis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


def themed_figure(
    figsize: Tuple[float, float] = (6.0, 4.5),
    n_axes: int = 1,
    projection: Optional[str] = None,
    nrows: int = 1,
) -> Union[Tuple[Figure, object], Tuple[Figure, List]]:
    """A dark-themed figure with ``n_axes`` styled axes.

    ``nrows`` lays the axes out in a grid instead of a single row, for the
    diagnostics panels that stack two related series vertically.
    """
    fig = Figure(figsize=figsize, dpi=100)
    fig.patch.set_facecolor(FIG_BG)
    if n_axes == 1:
        ax = fig.add_subplot(111, projection=projection)
        style_axis(ax)
        return fig, ax
    ncols = -(-n_axes // nrows)
    axes = [fig.add_subplot(nrows, ncols, i + 1, projection=projection) for i in range(n_axes)]
    for ax in axes:
        style_axis(ax)
    return fig, axes


def themed_legend(ax, **kwargs):
    """A legend that stays readable on the dark axes background."""
    legend = ax.legend(facecolor=AX_BG, edgecolor="#444", labelcolor="white", **kwargs)
    if legend is not None:
        legend.get_frame().set_alpha(0.9)
    return legend


def empty_figure(message: str, figsize: Tuple[float, float] = (6.0, 3.0)) -> Figure:
    """Placeholder for "this test has no data for that chart" — every
    diagnostic here is optional in the benchmark JSON (42 of 97 tests have no
    ground truth, per-seam ghost scores are only emitted for true multi-strip
    composites), so a missing series must render an explanation rather than an
    empty axes or an exception."""
    fig, ax = themed_figure(figsize=figsize)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=COL_TEXT_DIM, wrap=True)
    ax.axis("off")
    return fig
