"""
HITL Checkpoint 5 — Final Output RLHF Feedback Dialog (S87).

Shows the finished stitch and collects an overall quality rating plus optional
flaw annotations. Feedback is persisted to FeedbackStore by the caller.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from gui.src.constants.elements import _PREVIEW_MAX_PX

try:
    from backend.src.constants.animation import RLHF_FLAW_TYPES
except Exception:
    RLHF_FLAW_TYPES = ["seam", "ghosting", "misalignment", "color_mismatch", "blur"]


class _AddFlawDialog(QDialog):
    """Minimal dialog to capture a single flaw annotation (type + severity)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Flaw")
        layout = QFormLayout(self)

        self._type_combo = QComboBox()
        self._type_combo.addItems(RLHF_FLAW_TYPES)
        layout.addRow("Flaw type:", self._type_combo)

        self._severity = QDoubleSpinBox()
        self._severity.setRange(0.0, 1.0)
        self._severity.setSingleStep(0.1)
        self._severity.setValue(0.5)
        self._severity.setDecimals(2)
        layout.addRow("Severity (0–1):", self._severity)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def flaw_dict(self) -> dict:
        return {
            "x": 0.0,
            "y": 0.0,
            "w": 1.0,
            "h": 1.0,
            "flaw_type": self._type_combo.currentText(),
            "severity": round(self._severity.value(), 2),
            "description": "",
        }


class FinalOutputReviewDialog(QDialog):
    """
    Post-run HITL dialog: shows the finished panorama, collects a quality
    rating and optional flaw annotations, and saves them to FeedbackStore.

    Dialog result codes:
      - Accepted  → user submitted feedback (get_feedback() returns the dict)
      - Rejected  → user skipped (no feedback saved)
    """

    def __init__(self, data: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Step 5 — Rate Output Quality")
        self.resize(740, 580)
        self._feedback: Optional[dict] = None

        canvas_preview: Optional[np.ndarray] = data.get("canvas_preview")
        self._output_path: str = data.get("output_path", "")
        self._pipeline_config: dict = data.get("pipeline_config", {})

        root = QVBoxLayout(self)

        # -- Preview --
        self._preview_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        if canvas_preview is not None:
            self._set_preview(canvas_preview)
        else:
            self._preview_label.setText("(no preview available)")
        root.addWidget(self._preview_label, stretch=2)

        # -- Rating --
        rating_box = QGroupBox("Overall Quality (0 = worst, 10 = perfect)")
        rating_layout = QHBoxLayout(rating_box)

        self._rating_slider = QSlider(Qt.Orientation.Horizontal)
        self._rating_slider.setRange(0, 20)  # 0–20 maps to 0.0–10.0 in 0.5 steps
        self._rating_slider.setValue(14)  # default 7.0
        self._rating_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._rating_slider.setTickInterval(2)
        rating_layout.addWidget(self._rating_slider, stretch=1)

        self._rating_label = QLabel("7.0 / 10")
        self._rating_label.setMinimumWidth(60)
        rating_layout.addWidget(self._rating_label)

        self._rating_slider.valueChanged.connect(self._on_rating_changed)
        root.addWidget(rating_box)

        # -- Flaw annotations --
        flaw_box = QGroupBox("Flaw Annotations (optional)")
        flaw_layout = QVBoxLayout(flaw_box)

        self._flaw_list = QListWidget()
        flaw_layout.addWidget(self._flaw_list)

        btn_row = QHBoxLayout()
        self._add_flaw_btn = QPushButton("Add Flaw…")
        self._add_flaw_btn.clicked.connect(self._on_add_flaw)
        self._remove_flaw_btn = QPushButton("Remove Selected")
        self._remove_flaw_btn.clicked.connect(self._on_remove_flaw)
        btn_row.addWidget(self._add_flaw_btn)
        btn_row.addWidget(self._remove_flaw_btn)
        btn_row.addStretch()
        flaw_layout.addLayout(btn_row)
        root.addWidget(flaw_box)

        # -- Buttons --
        self._save_btn = QPushButton("Save Feedback and Continue")
        self._save_btn.setDefault(True)
        self._skip_btn = QPushButton("Skip (no feedback)")
        self._save_btn.clicked.connect(self._on_save)
        self._skip_btn.clicked.connect(self.reject)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self._skip_btn)
        btn_layout.addWidget(self._save_btn)
        root.addLayout(btn_layout)

    # ---------------------------------------------------------------------- #

    def _set_preview(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        scale = min(1.0, _PREVIEW_MAX_PX / max(h, w, 1))
        if scale < 1.0:
            bgr = cv2.resize(
                bgr,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        self._preview_label.setPixmap(QPixmap.fromImage(qimg))

    def _on_rating_changed(self, value: int) -> None:
        rating = value / 2.0
        self._rating_label.setText(f"{rating:.1f} / 10")

    def _on_add_flaw(self) -> None:
        dlg = _AddFlawDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            flaw = dlg.flaw_dict()
            item = QListWidgetItem(
                f"{flaw['flaw_type']}  severity={flaw['severity']:.2f}"
            )
            item.setData(Qt.ItemDataRole.UserRole, flaw)
            self._flaw_list.addItem(item)

    def _on_remove_flaw(self) -> None:
        for item in self._flaw_list.selectedItems():
            self._flaw_list.takeItem(self._flaw_list.row(item))

    def _on_save(self) -> None:
        overall_rating = self._rating_slider.value() / 2.0
        annotations: List[dict] = []
        for i in range(self._flaw_list.count()):
            item = self._flaw_list.item(i)
            annotations.append(item.data(Qt.ItemDataRole.UserRole))
        self._feedback = {
            "overall_rating": overall_rating,
            "annotations": annotations,
            "output_path": self._output_path,
            "pipeline_config": self._pipeline_config,
        }
        self.accept()

    # ---------------------------------------------------------------------- #

    def get_feedback(self) -> Optional[dict]:
        """Returns collected feedback dict, or None if the user skipped."""
        return self._feedback
