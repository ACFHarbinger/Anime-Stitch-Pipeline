"""Tests for the pure-math wallpaper aspect geometry helpers.

Pure integer/float geometry only — no numpy, no cv2, no fixtures.  Mirrors
the style of ``test_wallpaper_plate_builder.py``; boxes are canvas pixel
``(x0, y0, x1, y1)`` with ``x1``/``y1`` exclusive.
"""

from __future__ import annotations

import pytest
from asp_backend.rendering.wallpaper import (
    fit_window_containing_bbox,
    parse_aspect_ratio,
    union_bbox,
)

SIXTEEN_NINE = 16.0 / 9.0


def test_parse_aspect_ratio_sixteen_nine() -> None:
    assert parse_aspect_ratio("16:9") == pytest.approx(SIXTEEN_NINE)


def test_parse_aspect_ratio_nine_sixteen() -> None:
    assert parse_aspect_ratio("9:16") == pytest.approx(9.0 / 16.0)


def test_parse_aspect_ratio_twenty_one_nine() -> None:
    assert parse_aspect_ratio("21:9") == pytest.approx(21.0 / 9.0)


def test_parse_aspect_ratio_invalid_specs_rejected() -> None:
    for bad in ("4:3", "16:10", "16/9", "16:9 ", "16:9:1", ""):
        with pytest.raises(ValueError, match="unsupported"):
            parse_aspect_ratio(bad)


def test_union_bbox_single_box() -> None:
    assert union_bbox([(2, 3, 8, 9)]) == (2, 3, 8, 9)


def test_union_bbox_overlapping_boxes() -> None:
    boxes = [(0, 0, 4, 4), (2, 2, 6, 6)]
    assert union_bbox(boxes) == (0, 0, 6, 6)


def test_union_bbox_disjoint_boxes() -> None:
    boxes = [(0, 0, 2, 2), (10, 10, 12, 12)]
    assert union_bbox(boxes) == (0, 0, 12, 12)


def test_union_bbox_containment_and_negative_coordinates() -> None:
    boxes = [(-5, -5, 5, 5), (-2, 3, 4, 8), (0, 0, 10, 10)]
    assert union_bbox(boxes) == (-5, -5, 10, 10)


def test_union_bbox_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        union_bbox([])


def test_fit_window_exact_aspect_returns_bbox() -> None:
    # 160x90 is exactly 16:9; the minimal window is the bbox itself.
    bbox = (0, 0, 160, 90)
    result = fit_window_containing_bbox(bbox, SIXTEEN_NINE, (200, 300))
    assert result == bbox


def test_fit_window_pads_width_for_tall_bbox() -> None:
    # 90x160 is taller than 16:9 -> the window must grow in width only.
    result = fit_window_containing_bbox((0, 0, 90, 160), SIXTEEN_NINE, (200, 300))
    assert result == (-98, 0, 188, 160)
    assert result[0] <= 0 and result[2] >= 90
    assert (result[2] - result[0]) > 90
    assert (result[3] - result[1]) == 160


def test_fit_window_pads_height_for_wide_bbox() -> None:
    # 200x90 is wider than 16:9 -> the window must grow in height only.
    result = fit_window_containing_bbox((0, 0, 200, 90), SIXTEEN_NINE, (200, 300))
    assert result == (0, -12, 200, 102)
    assert result[1] <= 0 and result[3] >= 90
    assert (result[3] - result[1]) > 90
    assert (result[2] - result[0]) == 200


def test_fit_window_does_not_clamp_to_canvas() -> None:
    # Canvas is only 100 tall but the window needs 114 -> must NOT shrink.
    canvas = (100, 200)
    result = fit_window_containing_bbox((50, 50, 250, 150), SIXTEEN_NINE, canvas)
    assert result == (50, 43, 250, 157)
    assert result[0] >= 50 and result[2] <= 250
    assert (result[3] - result[1]) > canvas[0]


def test_fit_window_negative_aspect_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        fit_window_containing_bbox((0, 0, 10, 10), 0.0, (100, 100))