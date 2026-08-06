"""Diagnostics tab: the per-test pipeline charts from ``logic/diagnostics.py``
(issue #69's 11.1-11.5 plus timings, frame selection and the metric radar).

Also carries the cross-run selector that turns on 11.5's regression detection:
pick an older ``anime_stitch_*.json`` and the ground-truth chart annotates any
metric that moved against its good direction by more than the 3% margin
``_gt_verdict`` itself uses.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from PySide6.QtWidgets import QComboBox

from ..logic import diagnostics as dg
from ..other import discovery
from .tool_tab_base import ToolTabBase


class DiagnosticsTab(ToolTabBase):
    def __init__(self, repo_root: str, parent=None):
        super().__init__(parent)
        self._repo_root = repo_root
        self._name: Optional[str] = None
        self._runs: List[str] = discovery.results_files(repo_root)

        self._baseline = QComboBox()
        self._baseline.addItem("none (no regression check)", None)
        # Newest first after the current run — a user comparing against
        # "the previous run" wants it at the top, not 40 files down.
        for path in reversed(self._runs[:-1]):
            self._baseline.addItem(os.path.basename(path), path)
        self._baseline.currentIndexChanged.connect(self.refresh)
        self._add_control_row("Baseline run", self._baseline)

        for label, builder in dg.DIAGNOSTICS:
            self._add_tool(label, self._make_handler(builder))

    def set_dataset(self, name: Optional[str]) -> None:
        self._name = name

    def _make_handler(self, builder):
        def handler():
            if builder is dg.gt_comparison_figure:
                return builder(self._metrics, self._baseline_entry())
            return builder(self._metrics)

        return handler

    def _baseline_entry(self) -> Optional[Dict]:
        path = self._baseline.currentData()
        if not path or not self._name:
            return None
        return discovery.load_metrics(self._repo_root, path).get(self._name)

    def _cache_key(self, name: str) -> str:
        return f"{name}|{self._baseline.currentData()}"
