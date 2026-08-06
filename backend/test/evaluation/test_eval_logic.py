"""
Tests for the evaluation tool's pure computation layer (issue #123).

The headline case is defect 3 — "FFT Spectrum is incorrectly shown as the same
for all images". The root cause was rendering, not maths: the raw ``log1p(|F|)``
array went to ``imshow`` on matplotlib's default linear autoscale, where the DC
spike owns the whole colour range, so every natural image came out as the same
near-uniform field with one bright dot. So the tests here assert the two
properties that fix it: that the percentile window is much narrower than the raw
range (contrast actually gets stretched), and that the radial power profile
separates images the 2D spectrum renders alike.

Everything in ``logic/`` is a pure ``ndarray``/``dict`` -> ``ndarray``/``Figure``
transform with no Qt import, which is what lets this run headless.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _repo_root)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from asp_backend.evaluation.constants.logic import (  # noqa: E402
    FFT_PERCENTILE_HI,
    FFT_PERCENTILE_LO,
    VIZ_MAX_EDGE,
)
from asp_backend.evaluation.logic import comparison_maps as cm  # noqa: E402
from asp_backend.evaluation.logic import visualizations_basic as vb  # noqa: E402
from asp_backend.evaluation.logic import visualizations_matching as vm  # noqa: E402
from asp_backend.evaluation.logic.figure_theme import empty_figure, themed_figure  # noqa: E402


def _sharp(h=240, w=200, seed=0) -> np.ndarray:
    """High-frequency texture — stands in for a sharp composite."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _blurred(h=240, w=200, seed=0) -> np.ndarray:
    """The same content with its high frequencies removed."""
    import cv2

    return cv2.GaussianBlur(_sharp(h, w, seed), (21, 21), 0)


def _banded(h=240, w=200, period=12) -> np.ndarray:
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    img[::period] = 190
    return img


# ---------------------------------------------------------------------------
# defect 3 — the FFT view
# ---------------------------------------------------------------------------


def test_percentile_window_is_narrower_than_the_raw_range():
    """The actual fix: clipping to p1-p99.5 puts the spectral structure across
    the colormap instead of leaving it in a sliver above the DC spike."""
    magnitude = vb.fft_log_magnitude(_sharp())
    lo, hi = np.percentile(magnitude, [FFT_PERCENTILE_LO, FFT_PERCENTILE_HI])
    raw_span = float(magnitude.max() - magnitude.min())
    assert hi > lo
    assert (hi - lo) < raw_span
    # The DC spike must be excluded by the upper clip, or nothing was gained.
    assert magnitude.max() > hi


def test_radial_profile_separates_sharp_from_blurred():
    """A blur is invisible in the 2D spectrum image but obvious in the profile's
    high-frequency tail — which is why the profile is plotted beside it."""
    _, sharp_profile = vb.radial_power_profile(vb.fft_log_magnitude(_sharp()))
    _, blur_profile = vb.radial_power_profile(vb.fft_log_magnitude(_blurred()))
    n = min(len(sharp_profile), len(blur_profile))
    tail = slice(int(0.6 * n), n)
    assert sharp_profile[tail].mean() > blur_profile[tail].mean() + 0.5


def test_radial_profile_is_monotonic_in_frequency_axis():
    freqs, profile = vb.radial_power_profile(vb.fft_log_magnitude(_sharp()))
    assert len(freqs) == len(profile)
    assert np.all(np.diff(freqs) > 0)
    assert freqs[0] > 0.0 and freqs[-1] <= 1.0


def test_radial_profile_normalizes_each_axis_so_aspect_is_not_anisotropy():
    """A non-square panorama must not read as anisotropic purely because of its
    aspect ratio — the corpus has 1703x1704 next to 2197x2972."""
    tall = _sharp(400, 150, seed=1)
    wide = _sharp(150, 400, seed=1)
    _, tall_profile = vb.radial_power_profile(vb.fft_log_magnitude(tall))
    _, wide_profile = vb.radial_power_profile(vb.fft_log_magnitude(wide))
    n = min(len(tall_profile), len(wide_profile))
    assert np.abs(tall_profile[:n] - wide_profile[:n]).max() < 1.5


def test_constant_image_does_not_divide_by_zero():
    flat = np.full((80, 90, 3), 77, dtype=np.uint8)
    figure = vb.fft_magnitude_figure(flat)
    assert figure is not None


def test_fft_profile_comparison_plots_one_curve_per_image():
    figure = vb.fft_profile_comparison_figure({"asp": _sharp(), "simple": _blurred()})
    axis = figure.axes[0]
    assert len(axis.lines) == 2
    assert {t.get_text() for t in axis.get_legend().get_texts()} == {"asp", "simple"}


def test_fft_profile_comparison_skips_missing_images():
    figure = vb.fft_profile_comparison_figure({"asp": _sharp(), "hugin": None})
    assert len(figure.axes[0].lines) == 1


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------


def test_large_rasters_are_downsampled_before_plotting():
    big = np.zeros((2050, 1917, 3), dtype=np.uint8)
    assert max(vb._downsample(big).shape[:2]) <= VIZ_MAX_EDGE


def test_small_rasters_are_left_alone():
    small = np.zeros((64, 48, 3), dtype=np.uint8)
    assert vb._downsample(small).shape == small.shape


# ---------------------------------------------------------------------------
# Per-image figures all build
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", [
    vb.color_channel_figure,
    vb.cumulative_histogram_figure,
    vb.scatter_2d_figure,
    vb.scatter_3d_figure,
    vb.spatial_scatter_figure,
    vb.intensity_heatmap_figure,
    vb.gradient_heatmap_figure,
    vb.fft_magnitude_figure,
    vb.row_luminance_profile_figure,
])
def test_single_image_figures_build(builder):
    assert builder(_sharp(120, 100)) is not None


def test_row_luminance_profile_reports_the_seam_coherence_std():
    """The plot's title carries the same std the seam_coherence metric is, so a
    banding artifact reported as a scalar can be seen as its cause."""
    figure = vb.row_luminance_profile_figure(_banded())
    rows = _banded()[..., 0].astype(np.float32).mean(axis=1)
    assert f"{float(rows.std()):.2f}" in figure.axes[0].get_title()


@pytest.mark.parametrize("method", ["orb", "sift"])
def test_feature_match_figures_build(method):
    assert vm.feature_match_figure(_sharp(), _sharp(seed=1), method) is not None


def test_feature_match_handles_featureless_input():
    flat = np.full((100, 100, 3), 200, dtype=np.uint8)
    figure = vm.feature_match_figure(flat, flat, "orb")
    assert "no features" in figure.axes[0].get_title()


def test_optical_flow_resizes_a_mismatched_pair():
    assert vm.optical_flow_hsv_figure(_sharp(120, 100), _sharp(90, 140, seed=2)) is not None


# ---------------------------------------------------------------------------
# Comparison maps
# ---------------------------------------------------------------------------


def test_shape_note_only_warns_when_canvases_differ():
    a = _sharp(100, 100)
    assert cm.shape_note(a, _sharp(100, 100, seed=3)) is None
    note = cm.shape_note(a, _sharp(140, 120))
    assert note is not None and "resampled" in note


def test_identical_images_score_perfect_ssim():
    a = _sharp()
    result = cm.ssim_heatmap(a, a.copy())
    assert result.score == pytest.approx(1.0, abs=1e-6)
    assert result.heatmap.shape == a.shape


def test_ssim_returns_the_score_not_just_a_heatmap():
    """defect 7: the old compare tab computed this and threw it away."""
    result = cm.ssim_heatmap(_sharp(), _blurred())
    assert 0.0 <= result.score < 1.0
    assert isinstance(result.exact, bool)


def test_abs_diff_amplification_increases_the_signal():
    a, b = _sharp(), _blurred()
    plain = 255 - cm.abs_diff_inverted(a, b, amplify=1.0).astype(np.int32)
    loud = 255 - cm.abs_diff_inverted(a, b, amplify=4.0).astype(np.int32)
    assert loud.mean() > plain.mean()


@pytest.mark.parametrize("alpha,expected", [(1.0, 10), (0.0, 200), (0.5, 105)])
def test_alpha_blend_endpoints(alpha, expected):
    a = np.full((40, 40, 3), 10, dtype=np.uint8)
    b = np.full((40, 40, 3), 200, dtype=np.uint8)
    assert cm.alpha_blend(a, b, alpha).mean() == pytest.approx(expected, abs=1.0)


@pytest.mark.parametrize("alpha", [-2.0, 1.7])
def test_alpha_blend_clamps_out_of_range_alpha(alpha):
    a = np.full((20, 20, 3), 10, dtype=np.uint8)
    b = np.full((20, 20, 3), 200, dtype=np.uint8)
    blended = cm.alpha_blend(a, b, alpha)
    assert 10 <= blended.mean() <= 200


def test_swipe_composite_takes_a_from_one_side_and_b_from_the_other():
    a = np.full((40, 100, 3), 10, dtype=np.uint8)
    b = np.full((40, 100, 3), 200, dtype=np.uint8)
    out, split = cm.swipe_composite(a, b, 0.5, vertical=True)
    assert split == 50
    assert out[:, :split].max() == 10 and out[:, split:].min() == 200


def test_swipe_composite_horizontal_divider():
    a = np.full((100, 40, 3), 10, dtype=np.uint8)
    b = np.full((100, 40, 3), 200, dtype=np.uint8)
    out, split = cm.swipe_composite(a, b, 0.25, vertical=False)
    assert split == 25
    assert out[:split].max() == 10 and out[split:].min() == 200


def test_checkerboard_alternates_and_honours_tile_size():
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = np.full((64, 64, 3), 255, dtype=np.uint8)
    out = cm.checkerboard_mosaic(a, b, tile=32)
    assert out[0, 0].max() == 0 and out[0, 32].min() == 255
    assert out[32, 0].min() == 255 and out[32, 32].max() == 0


def test_false_color_puts_a_in_red_and_b_in_cyan():
    a = np.full((20, 20, 3), 255, dtype=np.uint8)
    b = np.zeros((20, 20, 3), dtype=np.uint8)
    out = cm.false_color_overlay(a, b)
    assert out[..., 2].min() == 255  # R carries A
    assert out[..., 0].max() == 0 and out[..., 1].max() == 0  # G+B (cyan) carry B


def test_contour_bounding_finds_a_planted_difference():
    a = np.zeros((200, 200, 3), dtype=np.uint8)
    b = a.copy()
    b[60:140, 60:140] = 255
    _annotated, boxes = cm.contour_bounding(a, b)
    assert len(boxes) >= 1
    x, y, w, h = max(boxes, key=lambda box: box[2] * box[3])
    assert w > 40 and h > 40


def test_contour_threshold_is_respected():
    a = np.zeros((120, 120, 3), dtype=np.uint8)
    b = a.copy()
    b[40:80, 40:80] = 20  # a faint difference
    _, loose = cm.contour_bounding(a, b, thresh=5)
    _, strict = cm.contour_bounding(a, b, thresh=100)
    assert len(loose) >= len(strict)
    assert strict == []


def test_edge_overlay_marks_b_edges_in_magenta():
    a = np.full((60, 60, 3), 80, dtype=np.uint8)
    b = a.copy()
    b[:, 30:] = 240  # a hard vertical edge
    out = cm.edge_overlay(a, b)
    magenta = (out[..., 0] == 255) & (out[..., 1] == 0) & (out[..., 2] == 255)
    assert magenta.any()


def test_difference_stats_are_zero_for_identical_images():
    a = _sharp()
    stats = cm.difference_stats(a, a.copy())
    assert stats["mean_abs_diff"] == 0.0
    assert stats["changed_pct_gt2"] == 0.0


def test_pixel_value_grid_dumps_rgb_triples():
    region = np.zeros((4, 4, 3), dtype=np.uint8)
    region[..., 2] = 200  # R in BGR order
    text = cm.pixel_value_grid(region)
    assert "(200,  0,  0)" in text


def test_region_stats_report_per_channel_values():
    region = np.zeros((6, 6, 3), dtype=np.uint8)
    region[..., 2] = 120
    stats = cm.region_stats(region)
    assert stats["channels"]["R"]["mean"] == pytest.approx(120.0)
    assert stats["channels"]["B"]["max"] == 0
    assert (stats["height"], stats["width"]) == (6, 6)


# ---------------------------------------------------------------------------
# Figure theming
# ---------------------------------------------------------------------------


def test_themed_figure_supports_multi_axes_grids():
    _figure, axes = themed_figure(n_axes=4, nrows=2)
    assert len(axes) == 4


def test_themed_figure_styles_a_3d_axes_zaxis():
    """The old helper missed zaxis and papered over it with a per-call color=."""
    _figure, axis = themed_figure(projection="3d")
    assert axis.zaxis.label.get_color() == "white"


def test_empty_figure_carries_the_explanation():
    figure = empty_figure("no ground truth for this test")
    assert any("no ground truth" in t.get_text() for t in figure.axes[0].texts)
