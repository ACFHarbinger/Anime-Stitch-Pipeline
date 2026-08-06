"""Visualisations tab: per-image pixel-data views, plus the cross-image
feature-match and optical-flow views.

Fixes issue #123 defect 9: the old ``_run_pair`` ignored the source selector
entirely and always ran ASP-vs-Simple, so ASP-vs-ground-truth — the comparison
the objective is actually defined against — was unreachable. Pair tools now read
two independent selectors.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..constants.schema import IMAGE_ASP, IMAGE_GROUND_TRUTH, IMAGE_SIMPLE
from ..logic import visualizations_basic as vb
from ..logic import visualizations_matching as vm
from .tool_tab_base import ToolTabBase


class VisualizationTab(ToolTabBase):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._source = self._comparator_combo()
        self._source.currentIndexChanged.connect(self.refresh)
        self._add_control_row("Image", self._source)

        self._pair_b = self._comparator_combo()
        self._pair_b.currentIndexChanged.connect(self.refresh)
        self._add_control_row("Compare to", self._pair_b)

        self._add_tool("Color channels", lambda: self._single(vb.color_channel_figure))
        self._add_tool("Cumulative histogram", lambda: self._single(vb.cumulative_histogram_figure))
        self._add_tool("Row luminance (banding)", lambda: self._single(vb.row_luminance_profile_figure))
        self._add_tool("Intensity heatmap", lambda: self._single(vb.intensity_heatmap_figure))
        self._add_tool("Gradient heatmap", lambda: self._single(vb.gradient_heatmap_figure))
        self._add_tool("FFT spectrum + profile", lambda: self._single(vb.fft_magnitude_figure))
        self._add_tool("FFT profiles — all", self._fft_all)
        self._add_tool("2D scatter (R vs G)", lambda: self._single(vb.scatter_2d_figure))
        self._add_tool("3D scatter (RGB)", lambda: self._single(vb.scatter_3d_figure))
        self._add_tool("Spatial scatter (edges)", lambda: self._single(vb.spatial_scatter_figure))
        self._add_tool("ORB matches", lambda: self._pair(lambda a, b: vm.feature_match_figure(a, b, "orb")))
        self._add_tool("SIFT matches", lambda: self._pair(lambda a, b: vm.feature_match_figure(a, b, "sift")))
        self._add_tool("Optical flow (HSV)", lambda: self._pair(vm.optical_flow_hsv_figure))

    def _on_context_changed(self) -> None:
        keys = self.available()
        self._populate_combo(self._source, keys)
        self._populate_combo(self._pair_b, keys)
        # Default the pair target to something other than the source, so the
        # first click on a pair tool isn't a self-comparison.
        if self._source.currentData() == self._pair_b.currentData() and len(keys) > 1:
            preferred = [k for k in (IMAGE_GROUND_TRUTH, IMAGE_SIMPLE, IMAGE_ASP) if k in keys]
            for key in preferred:
                if key != self._source.currentData():
                    self._pair_b.setCurrentIndex(keys.index(key))
                    break

    def _cache_key(self, name: str) -> str:
        return f"{name}|{self._source.currentData()}|{self._pair_b.currentData()}"

    def _selected(self) -> Optional[np.ndarray]:
        return self.image(self._source.currentData())

    def _single(self, fn):
        img = self._selected()
        return None if img is None else fn(img)

    def _pair(self, fn):
        a = self.image(self._source.currentData())
        b = self.image(self._pair_b.currentData())
        if a is None or b is None:
            return None
        if self._source.currentData() == self._pair_b.currentData():
            self.set_status("Pick two different images to compare.")
            return None
        return fn(a, b)

    def _fft_all(self):
        images = {k: v for k, v in self._images.items() if v is not None}
        return vb.fft_profile_comparison_figure(images) if images else None
