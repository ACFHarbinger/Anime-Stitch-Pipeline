"""Comparison tab: ImageJ/DiffImg/ImageMagick-style pixel comparisons between
any two comparators, with live controls.

Every knob is a real slider driving the same parameterised functions in
``logic/comparison_maps.py`` that a headless export would call — blend alpha,
swipe position and axis, checkerboard tile size, contour blur/threshold, and
difference amplification. The old tab had a fixed 50% blend, a fixed 64px tile,
fixed contour parameters, and a swipe widget whose own scaled ``QLabel`` couldn't
be zoomed.

Results land in a zoomable ``ImagePanel`` (via ``ToolTabBase``), and each map is
annotated with the scalar difference statistics and a canvas-size caveat when
the two comparators had different dimensions — which they usually do.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QSlider

from ..constants.schema import IMAGE_ASP, IMAGE_SIMPLE
from ..logic import comparison_maps as cm
from .tool_tab_base import ToolTabBase


def _slider(minimum: int, maximum: int, value: int) -> QSlider:
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(value)
    return slider


class ComparisonTab(ToolTabBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._a = self._comparator_combo()
        self._b = self._comparator_combo()
        self._a.currentIndexChanged.connect(self.refresh)
        self._b.currentIndexChanged.connect(self.refresh)
        self._add_control_row("A", self._a)
        self._add_control_row("B", self._b)

        self._alpha = _slider(0, 100, 50)
        self._split = _slider(0, 100, 50)
        self._tile = _slider(8, 256, 64)
        self._amplify = _slider(10, 100, 10)  # tenths: 1.0x - 10.0x
        self._blur = _slider(1, 61, 15)
        self._thresh = _slider(1, 120, 25)
        self._vertical = QCheckBox("Vertical swipe divider")
        self._vertical.setChecked(True)
        for widget in (self._alpha, self._split, self._tile, self._amplify, self._blur, self._thresh):
            widget.valueChanged.connect(self.refresh)
        self._vertical.toggled.connect(self.refresh)
        self._add_control_row("Blend α", self._alpha)
        self._add_control_row("Swipe", self._split)
        self._add_control_row("Tile px", self._tile)
        self._add_control_row("Diff ×", self._amplify)
        self._add_control_row("Blur k", self._blur)
        self._add_control_row("Threshold", self._thresh)
        self._add_control(self._vertical)

        self._add_tool("Absolute difference", self._diff)
        self._add_tool("SSIM heatmap", self._ssim)
        self._add_tool("False colour (red/cyan)", self._false_color)
        self._add_tool("Alpha blend", self._blend)
        self._add_tool("Swipe composite", self._swipe)
        self._add_tool("Checkerboard mosaic", self._checkerboard)
        self._add_tool("Edge overlay", self._edges)
        self._add_tool("Contour-bounded diff", self._contours)
        self._add_tool("Difference statistics", self._stats)

    # -- context -------------------------------------------------------------

    def _on_context_changed(self) -> None:
        keys = self.available()
        self._populate_combo(self._a, keys)
        self._populate_combo(self._b, keys)
        if IMAGE_ASP in keys and self._a.currentData() not in keys:
            self._a.setCurrentIndex(keys.index(IMAGE_ASP))
        if self._b.currentData() == self._a.currentData() and len(keys) > 1:
            fallback = IMAGE_SIMPLE if IMAGE_SIMPLE in keys and self._a.currentData() != IMAGE_SIMPLE else None
            target = fallback or next(k for k in keys if k != self._a.currentData())
            self._b.setCurrentIndex(keys.index(target))

    def _cache_key(self, name: str) -> str:
        return "|".join(str(v) for v in (
            name, self._a.currentData(), self._b.currentData(),
            self._alpha.value(), self._split.value(), self._tile.value(),
            self._amplify.value(), self._blur.value(), self._thresh.value(),
            self._vertical.isChecked(),
        ))

    def _pair(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        a = self.image(self._a.currentData())
        b = self.image(self._b.currentData())
        if a is None or b is None:
            return None
        if self._a.currentData() == self._b.currentData():
            self.set_status("Pick two different comparators.")
            return None
        return a, b

    def _annotate(self, a: np.ndarray, b: np.ndarray, extra: str = "") -> None:
        parts = []
        if extra:
            parts.append(extra)
        stats = cm.difference_stats(a, b)
        parts.append(
            f"mean|Δ|={stats['mean_abs_diff']:.2f}  p99={stats['p99_abs_diff']:.0f}  "
            f"max={stats['max_abs_diff']:.0f}  "
            f"changed>2: {stats['changed_pct_gt2']:.1f}%  >10: {stats['changed_pct_gt10']:.1f}%"
        )
        note = cm.shape_note(a, b)
        if note:
            parts.append(note)
        self.set_status("   ".join(parts))

    # -- tools ---------------------------------------------------------------

    def _diff(self):
        pair = self._pair()
        if pair is None:
            return None
        a, b = pair
        out = cm.abs_diff_inverted(a, b, amplify=self._amplify.value() / 10.0)
        self._annotate(a, b, f"Absolute difference, inverted, ×{self._amplify.value() / 10.0:.1f}")
        return out

    def _ssim(self):
        pair = self._pair()
        if pair is None:
            return None
        a, b = pair
        result = cm.ssim_heatmap(a, b)
        kind = "SSIM" if result.exact else "abs-diff fallback (skimage missing)"
        self._annotate(a, b, f"{kind} score={result.score:.4f}")
        return result.heatmap

    def _false_color(self):
        pair = self._pair()
        if pair is None:
            return None
        a, b = pair
        self._annotate(a, b, "Red = A, cyan = B; fringes are misalignment")
        return cm.false_color_overlay(a, b)

    def _blend(self):
        pair = self._pair()
        if pair is None:
            return None
        a, b = pair
        alpha = self._alpha.value() / 100.0
        self._annotate(a, b, f"Blend α={alpha:.2f} (A weight)")
        return cm.alpha_blend(a, b, alpha)

    def _swipe(self):
        pair = self._pair()
        if pair is None:
            return None
        a, b = pair
        out, split = cm.swipe_composite(
            a, b, self._split.value() / 100.0, vertical=self._vertical.isChecked()
        )
        axis = "x" if self._vertical.isChecked() else "y"
        self._annotate(a, b, f"Swipe at {axis}={split}px — A before, B after")
        return out

    def _checkerboard(self):
        pair = self._pair()
        if pair is None:
            return None
        a, b = pair
        self._annotate(a, b, f"Checkerboard, {self._tile.value()}px tiles")
        return cm.checkerboard_mosaic(a, b, self._tile.value())

    def _edges(self):
        pair = self._pair()
        if pair is None:
            return None
        a, b = pair
        self._annotate(a, b, "B's line art over A — the sharpest misalignment read on flat cels")
        return cm.edge_overlay(a, b)

    def _contours(self):
        pair = self._pair()
        if pair is None:
            return None
        a, b = pair
        annotated, boxes = cm.contour_bounding(
            a, b, blur_ksize=self._blur.value(), thresh=self._thresh.value()
        )
        self._annotate(a, b, f"{len(boxes)} changed region(s) above threshold")
        return annotated

    def _stats(self):
        pair = self._pair()
        if pair is None:
            return None
        a, b = pair
        stats = cm.difference_stats(a, b)
        ssim = cm.ssim_heatmap(a, b)
        lines = [
            f"A: {self._a.currentData()}  {a.shape[1]}x{a.shape[0]}",
            f"B: {self._b.currentData()}  {b.shape[1]}x{b.shape[0]}",
            "",
            f"SSIM (grayscale, whole canvas)  {ssim.score:.6f}" + ("" if ssim.exact else "  [fallback]"),
            "",
        ]
        lines += [f"{key:22s} {value:.4f}" for key, value in stats.items()]
        note = cm.shape_note(a, b)
        if note:
            lines += ["", note]
        self.set_status("Difference statistics")
        return "\n".join(lines)
