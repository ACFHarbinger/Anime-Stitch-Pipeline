"""Coherence V2 A/B Evaluation tab for the ASP native evaluator.

Displays the Single-Pose vs. Seam-Loop Compositor A/B evaluation suite:
- Evaluates Stage 2 character pose coherence and anatomical fidelity.
- Loads pre-computed benchmark PNG renders and telemetry metadata.
- Interactive deep-zoom visual comparison: Side-by-Side, Swipe, Alpha Blend, Diff Map, SSIM.
- Telemetry readout: Line Art Fracture, Seam Visibility, Cel Flatness, Background Corridor, Single-Pose Handoff.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..logic import comparison_maps as cm
from .image_panel import ImagePanel
from .theme import subtle


def _slider(minimum: int, maximum: int, value: int) -> QSlider:
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setValue(value)
    return slider


class CoherenceV2Tab(QWidget):
    """Coherence V2 single-pose vs baseline comparison tab."""

    def __init__(self, repo_root: Path | str | None = None, parent=None):
        super().__init__(parent)
        self.repo_root = Path(repo_root) if repo_root else self._find_repo_root()
        self._ab_data: Dict[str, Any] = {}
        self._redset_data: Dict[str, Any] = {}
        self._cases: List[Dict[str, Any]] = []
        self._current_case: Optional[Dict[str, Any]] = None
        self._img_default: Optional[np.ndarray] = None
        self._img_v2: Optional[np.ndarray] = None

        self._load_datasets()
        self._setup_ui()
        self._populate_cases()

    @staticmethod
    def _find_repo_root() -> Path:
        cur = Path(__file__).resolve()
        for p in [cur] + list(cur.parents):
            if (p / ".agent").exists() or (p / "docs" / "website").exists():
                return p
        return Path.cwd()

    def _load_datasets(self) -> None:
        data_candidates = [
            self.repo_root / "docs" / "website" / "public" / "data",
            self.repo_root / "data" / "benchmark",
            Path.cwd() / "docs" / "website" / "public" / "data",
        ]
        data_dir = None
        for cand in data_candidates:
            if (cand / "coherence_v2_ab_eval.json").exists():
                data_dir = cand
                break

        if data_dir is not None:
            ab_file = data_dir / "coherence_v2_ab_eval.json"
            redset_file = data_dir / "coherence_v2_redset.json"
            if ab_file.exists():
                try:
                    with open(ab_file, encoding="utf-8") as f:
                        self._ab_data = json.load(f)
                except Exception:
                    self._ab_data = {}
            if redset_file.exists():
                try:
                    with open(redset_file, encoding="utf-8") as f:
                        self._redset_data = json.load(f)
                except Exception:
                    self._redset_data = {}

        self._cases = self._ab_data.get("cases", [])

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left Panel: Controls, Case Selection & Telemetry ---
        left_widget = QWidget()
        left_widget.setMinimumWidth(320)
        left_widget.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        # Header Banner
        header_box = QFrame()
        header_box.setStyleSheet(
            "QFrame { background-color: #1e222b; border: 1px solid #333a46; border-radius: 6px; padding: 6px; }"
        )
        h_layout = QVBoxLayout(header_box)
        h_layout.setContentsMargins(4, 4, 4, 4)
        h_layout.setSpacing(2)
        title_lbl = QLabel("✨ Coherence V2 Evaluator")
        title_lbl.setFont(QFont("sans-serif", 10, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color: #2ecc71;")
        desc_lbl = QLabel("Single-Pose vs. Seam-Loop Compositor A/B Comparison")
        desc_lbl.setStyleSheet("color: #a0a6b2; font-size: 11px;")
        desc_lbl.setWordWrap(True)
        h_layout.addWidget(title_lbl)
        h_layout.addWidget(desc_lbl)
        left_layout.addWidget(header_box)

        # Filter row
        filter_row = QHBoxLayout()
        filter_lbl = QLabel("Filter:")
        filter_lbl.setStyleSheet("color: #c5c9d3; font-size: 11px; font-weight: bold;")
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["All Cases", "Red-Set Cases Only", "Anatomy Improved Only"])
        self.combo_filter.currentIndexChanged.connect(self._populate_cases)
        filter_row.addWidget(filter_lbl)
        filter_row.addWidget(self.combo_filter, stretch=1)
        left_layout.addLayout(filter_row)

        # Case List
        left_layout.addWidget(QLabel("Select Evaluation Case:"))
        self.case_list = QListWidget()
        self.case_list.setStyleSheet(
            """
            QListWidget {
                background-color: #1a1d24;
                border: 1px solid #333a46;
                border-radius: 4px;
                color: #e2e8f0;
            }
            QListWidget::item {
                padding: 4px 6px;
                border-bottom: 1px solid #232730;
            }
            QListWidget::item:selected {
                background-color: #2b3b52;
                color: #60a5fa;
            }
            """
        )
        self.case_list.currentRowChanged.connect(self._on_case_selected)
        left_layout.addWidget(self.case_list, stretch=2)

        # Comparison Controls Group
        ctrl_box = QFrame()
        ctrl_box.setStyleSheet(
            "QFrame { background-color: #1e222b; border: 1px solid #333a46; border-radius: 6px; padding: 6px; }"
        )
        ctrl_layout = QVBoxLayout(ctrl_box)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)
        ctrl_layout.setSpacing(4)

        # View Mode Combo
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("View Mode:")
        mode_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #c5c9d3;")
        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "Side-by-Side (A | B)",
            "Swipe Divider (Slider)",
            "Alpha Blend (Crossfade)",
            "Diff Map (Amplified)",
            "SSIM Heatmap",
            "False-Color Fringes",
            "View Baseline Only (A)",
            "View Coherence V2 Only (B)",
        ])
        self.combo_mode.currentIndexChanged.connect(self._refresh_view)
        mode_row.addWidget(mode_lbl)
        mode_row.addWidget(self.combo_mode, stretch=1)
        ctrl_layout.addLayout(mode_row)

        # Sliders
        self.slider_swipe = _slider(0, 100, 50)
        self.slider_swipe.valueChanged.connect(self._refresh_view)
        self.slider_alpha = _slider(0, 100, 50)
        self.slider_alpha.valueChanged.connect(self._refresh_view)
        self.slider_diff = _slider(10, 100, 15)
        self.slider_diff.valueChanged.connect(self._refresh_view)
        self.chk_vertical = QCheckBox("Vertical swipe")
        self.chk_vertical.setChecked(True)
        self.chk_vertical.toggled.connect(self._refresh_view)

        ctrl_layout.addLayout(self._make_slider_row("Swipe %:", self.slider_swipe))
        ctrl_layout.addLayout(self._make_slider_row("Blend α:", self.slider_alpha))
        ctrl_layout.addLayout(self._make_slider_row("Diff ×:", self.slider_diff))
        ctrl_layout.addWidget(self.chk_vertical)
        left_layout.addWidget(ctrl_box)

        # Telemetry & Metrics Card
        self.lbl_metrics_card = QLabel()
        self.lbl_metrics_card.setWordWrap(True)
        self.lbl_metrics_card.setStyleSheet(
            """
            QLabel {
                background-color: #14171d;
                border: 1px solid #29303d;
                border-radius: 6px;
                padding: 8px;
                color: #cbd5e1;
                font-family: monospace;
                font-size: 11px;
            }
            """
        )
        left_layout.addWidget(self.lbl_metrics_card, stretch=2)

        splitter.addWidget(left_widget)

        # --- Right Panel: Zoomable Image Display ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        self.image_panel = ImagePanel("coherence_v2", "Coherence V2 A/B")
        right_layout.addWidget(self.image_panel, stretch=1)

        self.lbl_status = subtle("Select a case to inspect.")
        self.lbl_status.setStyleSheet("padding: 4px 8px; color: #8b949e; font-size: 11px;")
        right_layout.addWidget(self.lbl_status)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        outer.addWidget(splitter)

    @staticmethod
    def _make_slider_row(label: str, slider: QSlider) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setMinimumWidth(55)
        lbl.setStyleSheet("font-size: 11px; color: #94a3b8;")
        row.addWidget(lbl)
        row.addWidget(slider, stretch=1)
        return row

    def _populate_cases(self) -> None:
        self.case_list.blockSignals(True)
        self.case_list.clear()

        filter_mode = self.combo_filter.currentIndex()
        # 0: All, 1: Redset Only, 2: Improved Anatomy Only

        redset_names = {c.get("name") for c in self._redset_data.get("cases", [])}

        for c in self._cases:
            name = c.get("name", "unknown")
            verdict = c.get("engineering_verdict", "")
            is_improved = verdict == "improves_anatomy"
            is_redset = name in redset_names

            if filter_mode == 1 and not is_redset:
                continue
            if filter_mode == 2 and not is_improved:
                continue

            item_text = name
            if is_improved:
                item_text += "  [✓ Anatomy]"
            if is_redset:
                item_text += "  (Red-Set)"

            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, c)
            if is_improved:
                item.setForeground(QColor("#4ade80"))
            elif is_redset:
                item.setForeground(QColor("#f87171"))
            self.case_list.addItem(item)

        self.case_list.blockSignals(False)
        if self.case_list.count() > 0:
            self.case_list.setCurrentRow(0)
            self._on_case_selected(0)
        else:
            self._current_case = None
            self._update_metrics_display()
            self._refresh_view()

    def _on_case_selected(self, row: int) -> None:
        item = self.case_list.item(row)
        if not item:
            return
        case_data = item.data(Qt.ItemDataRole.UserRole)
        self._current_case = case_data
        self._load_case_images(case_data.get("name", ""))
        self._update_metrics_display()
        self._refresh_view()

    def _load_case_images(self, case_name: str) -> None:
        self._img_default = None
        self._img_v2 = None
        if not case_name:
            return

        candidates = [
            self.repo_root / "docs" / "website" / "public" / "data" / "coherence_v2",
            self.repo_root / "data" / "benchmark" / "coherence_v2",
            Path.cwd() / "docs" / "website" / "public" / "data" / "coherence_v2",
        ]
        dir_found = None
        for c in candidates:
            if c.exists():
                dir_found = c
                break

        if dir_found:
            default_path = dir_found / f"{case_name}_default.png"
            v2_path = dir_found / f"{case_name}_v2.png"

            if default_path.exists():
                self._img_default = cv2.imread(str(default_path))
            if v2_path.exists():
                self._img_v2 = cv2.imread(str(v2_path))

    def _update_metrics_display(self) -> None:
        if not self._current_case:
            self.lbl_metrics_card.setText("No case selected.")
            return

        c = self._current_case
        name = c.get("name", "—")
        category = c.get("target_category", "structural")
        verdict = c.get("engineering_verdict", "parity")
        notes = c.get("notes", "—")

        m_base = c.get("metrics_baseline", {})
        m_v2 = c.get("metrics_coherence_v2", {})
        delta = c.get("delta_metrics", {})

        laf_base = m_base.get("line_art_fracture", 0.0)
        laf_v2 = m_v2.get("line_art_fracture", 0.0)
        laf_d = delta.get("line_art_fracture_delta", 0.0)

        seam_base = m_base.get("seam_visibility", 0.0)
        seam_v2 = m_v2.get("seam_visibility", 0.0)
        seam_d = delta.get("seam_visibility_delta", 0.0)

        flat_base = m_base.get("cel_flatness", 0.0)
        flat_v2 = m_v2.get("cel_flatness", 0.0)
        flat_d = delta.get("cel_flatness_delta", 0.0)

        has_corridor = c.get("has_background_corridor", False)
        handoff = c.get("handoff_occurred", False)

        laf_color = "#4ade80" if laf_d < 0 else "#94a3b8"
        corridor_str = (
            "<span style='color:#4ade80;'>Feasible Path</span>"
            if has_corridor
            else "<span style='color:#fbbf24;'>Obstructed (All-FG)</span>"
        )
        handoff_str = (
            "<span style='color:#38bdf8;'>Active Handoff</span>"
            if handoff
            else "<span style='color:#94a3b8;'>Direct Corridor</span>"
        )
        verdict_color = "#4ade80" if verdict == "improves_anatomy" else "#94a3b8"

        html = f"""
        <div style='line-height: 1.4;'>
          <b>Case:</b> <span style='color:#60a5fa;'>{name}</span> &bull; <b>Target:</b> {category}<br>
          <b>Verdict:</b> <span style='color:{verdict_color}; font-weight:bold;'>{verdict}</span><br>
          <hr style='border: 0.5px solid #333a46;'>
          <b>Line Art Fracture:</b> {laf_base:.1f} &rarr; <span style='color:#4ade80;'>{laf_v2:.1f}</span>
          (<span style='color:{laf_color};'>{laf_d:+.1f}</span>)<br>
          <b>Seam Visibility:</b> {seam_base:.1f} &rarr; {seam_v2:.1f} ({seam_d:+.1f})<br>
          <b>Cel Flatness:</b> {flat_base:.2f} &rarr; {flat_v2:.2f} ({flat_d:+.2f})<br>
          <hr style='border: 0.5px solid #333a46;'>
          <b>BG Corridor:</b> {corridor_str}<br>
          <b>Single-Pose Handoff:</b> {handoff_str}<br>
          <b>Notes:</b> <span style='color:#94a3b8;'>{notes}</span>
        </div>
        """
        self.lbl_metrics_card.setText(html)

    def _refresh_view(self) -> None:
        if self._img_default is None and self._img_v2 is None:
            self.image_panel.set_image(None)
            self.lbl_status.setText("No images found for current case.")
            return

        a = self._img_default
        b = self._img_v2
        if a is None and b is not None:
            a = b.copy()
        if b is None and a is not None:
            b = a.copy()

        mode = self.combo_mode.currentIndex()
        # 0: Side-by-side, 1: Swipe, 2: Blend, 3: Diff, 4: SSIM, 5: False Color, 6: A only, 7: B only

        out_img: Optional[np.ndarray] = None
        status_text = ""

        if mode == 0:  # Side-by-Side
            ha, wa = a.shape[:2]
            hb, wb = b.shape[:2]
            max_h = max(ha, hb)
            pad_a = cv2.copyMakeBorder(a, 0, max_h - ha, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)) if ha < max_h else a
            pad_b = cv2.copyMakeBorder(b, 0, max_h - hb, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)) if hb < max_h else b
            divider = np.zeros((max_h, 3, 3), dtype=np.uint8)
            divider[:, :, 1] = 255  # Green divider line
            out_img = np.hstack([pad_a, divider, pad_b])
            status_text = f"Side-by-Side: [Left: Baseline Default | Right: Coherence V2] ({out_img.shape[1]}x{out_img.shape[0]})"
        elif mode == 1:  # Swipe
            split = self.slider_swipe.value() / 100.0
            vertical = self.chk_vertical.isChecked()
            composite, split_px = cm.swipe_composite(a, b, split=split, vertical=vertical)
            # Draw split line
            line_color = (0, 255, 0)
            if vertical and 0 <= split_px < composite.shape[1]:
                cv2.line(composite, (split_px, 0), (split_px, composite.shape[0]), line_color, 2)
            elif not vertical and 0 <= split_px < composite.shape[0]:
                cv2.line(composite, (0, split_px), (composite.shape[1], split_px), line_color, 2)
            out_img = composite
            status_text = f"Swipe Wipe at {int(split*100)}% (Divider: {'Vertical' if vertical else 'Horizontal'})"
        elif mode == 2:  # Alpha Blend
            alpha = self.slider_alpha.value() / 100.0
            out_img = cm.alpha_blend(a, b, alpha=alpha)
            status_text = f"Alpha Blend: {int(alpha*100)}% Baseline / {int((1.0-alpha)*100)}% Coherence V2"
        elif mode == 3:  # Diff Map
            amp = self.slider_diff.value() / 10.0
            out_img = cm.abs_diff_inverted(a, b, amplify=amp)
            status_text = f"Inverted Diff Map (Amplification: {amp:.1f}x)"
        elif mode == 4:  # SSIM Heatmap
            res = cm.ssim_heatmap(a, b)
            out_img = res.heatmap
            status_text = f"SSIM Heatmap (Mean SSIM: {res.score:.4f})"
        elif mode == 5:  # False-color
            out_img = cm.false_color_overlay(a, b)
            status_text = "False-Color Overlay: Red = Baseline, Cyan = Coherence V2"
        elif mode == 6:  # A Only
            out_img = a
            status_text = f"Baseline Default Stitch ({a.shape[1]}x{a.shape[0]})"
        elif mode == 7:  # B Only
            out_img = b
            status_text = f"Coherence V2 Stitch ({b.shape[1]}x{b.shape[0]})"

        self.image_panel.set_image(out_img)
        self.lbl_status.setText(status_text)

    def set_context(self, images: dict, metrics: dict, name: str) -> None:
        """Called by main evaluator window when dataset changes."""
        for i in range(self.case_list.count()):
            item = self.case_list.item(i)
            c = item.data(Qt.ItemDataRole.UserRole)
            if c and c.get("name") == name:
                self.case_list.setCurrentRow(i)
                break


__all__ = ["CoherenceV2Tab"]
