"""
MaskReviewDialog — Issue 10A2 (click-based segmentation review).

Displays a frame with its SAM-2 foreground mask overlaid. The user can:
  - Left-click  → add a positive prompt (expand the mask to include this point)
  - Right-click → add a negative prompt (exclude this point from the mask)
  - Type a character description and click "Re-segment" to re-run Grounded SAM-2
  - Click "Accept" to confirm the current masks and resume the pipeline

The dialog communicates back to StitchWorker via:
  ``sig_mask_accepted`` — list[Optional[ndarray]] refined fg masks

The former GroundingDINO "Seam Exclusion (NL routing)" feature (Issue 10A3) was
removed with the hitl.grounding module in the 2026-07 ASP trim; seam routing
overrides are now done visually via the SeamPainterDialog / SeamDiagnosticDialog
(waypoints, force-single-pose, painted barriers) instead of text prompts.

Architecture: all model inference is delegated to a ``_RefinementWorker`` QThread.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
from gui.src.constants.elements import (
    _CLICK_RADIUS,
    _MAX_DISPLAY_H,
    _NEGATIVE_COLOR,
    _OVERLAY_ALPHA,
    _POSITIVE_COLOR,
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
)

# ── constants ────────────────────────────────────────────────────────────────


# ── helpers ──────────────────────────────────────────────────────────────────


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _render_overlay(
    frame: np.ndarray,
    mask: np.ndarray | None,
    pos_clicks: list[tuple[int, int]],
    neg_clicks: list[tuple[int, int]],
) -> np.ndarray:
    """Render frame with mask overlay and click dots into a uint8 BGR image."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Mask overlay: foreground (mask==0) gets a teal tint
    if mask is not None:
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        fg = m < 128
        overlay = out.copy()
        overlay[fg] = (
            (
                overlay[fg].astype(np.float32) * (1 - _OVERLAY_ALPHA)
                + np.array([120, 200, 60], dtype=np.float32) * _OVERLAY_ALPHA
            )
            .clip(0, 255)
            .astype(np.uint8)
        )
        out = overlay

    # Draw click dots
    for cx, cy in pos_clicks:
        cv2.circle(out, (cx, cy), _CLICK_RADIUS, _POSITIVE_COLOR, -1)
        cv2.circle(out, (cx, cy), _CLICK_RADIUS + 1, (0, 100, 0), 1)
    for cx, cy in neg_clicks:
        cv2.circle(out, (cx, cy), _CLICK_RADIUS, _NEGATIVE_COLOR, -1)
        cv2.circle(out, (cx, cy), _CLICK_RADIUS + 1, (0, 0, 100), 1)

    return out


# ── refinement worker ─────────────────────────────────────────────────────────


class _RefinementWorker(QThread):
    """
    Runs mask refinement (Grounded SAM-2 or click re-propagation) off the main thread.
    """

    sig_done = Signal(list)  # list[Optional[np.ndarray]] — refined masks
    sig_error = Signal(str)

    def __init__(self, refine_fn: Callable, parent=None):
        super().__init__(parent)
        self._refine_fn = refine_fn

    def run(self):
        try:
            masks = self._refine_fn()
            self.sig_done.emit(masks)
        except Exception as e:
            self.sig_error.emit(str(e))


# ── click-capture overlay widget ──────────────────────────────────────────────


class _ClickOverlay(QLabel):
    """
    QLabel subclass that captures mouse clicks and maps them to frame coordinates.
    Left-click → positive prompt; right-click → negative prompt.
    """

    sig_pos_click = Signal(int, int)  # (frame_x, frame_y)
    sig_neg_click = Signal(int, int)

    def __init__(self, frame_w: int, frame_h: int, parent=None):
        super().__init__(parent)
        self._frame_w = frame_w
        self._frame_h = frame_h
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event: QMouseEvent):
        # Map display coordinates to frame coordinates
        if self.pixmap() is None or self.pixmap().isNull():
            return
        pm = self.pixmap()
        disp_w, disp_h = pm.width(), pm.height()
        fx = int(event.position().x() / disp_w * self._frame_w)
        fy = int(event.position().y() / disp_h * self._frame_h)
        fx = max(0, min(fx, self._frame_w - 1))
        fy = max(0, min(fy, self._frame_h - 1))
        if event.button() == Qt.MouseButton.LeftButton:
            self.sig_pos_click.emit(fx, fy)
        elif event.button() == Qt.MouseButton.RightButton:
            self.sig_neg_click.emit(fx, fy)


# ── main dialog ───────────────────────────────────────────────────────────────


class MaskReviewDialog(QDialog):
    """
    HITL mask review dialog — Issue 10A2 (click-based refinement) + 10A1 (text re-segment).

    Signals
    -------
    sig_mask_accepted : emitted with the final list[Optional[ndarray]] when user clicks Accept.

    Parameters
    ----------
    data : dict from the "masks" HITL checkpoint:
        "frames"       : list[np.ndarray]          — BGR frames
        "bg_masks"     : list[Optional[np.ndarray]] — 255=bg, 0=fg masks
        "image_paths"  : list[str]
    refine_callback : optional callable(text_prompt, pos_clicks, neg_clicks, frame_idx)
        that returns list[Optional[np.ndarray]].  When None the "Re-segment" button is
        disabled but click-prompt refinement still records clicks for annotation export.
    """

    sig_mask_accepted = Signal(object)  # list[Optional[np.ndarray]]

    def __init__(self, data: dict, refine_callback=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mask Review — Foreground Segmentation (Stage 4.5)")
        self.resize(900, 720)

        self._frames: list[np.ndarray] = list(data.get("frames", []))
        self._masks: list[np.ndarray | None] = list(data.get("bg_masks", []))
        self._paths: list[str] = list(data.get("image_paths", []))
        self._refine_callback = refine_callback

        # Ensure mask list is same length as frames
        while len(self._masks) < len(self._frames):
            self._masks.append(None)

        self._current_idx = 0
        self._pos_clicks: list[tuple[int, int]] = []
        self._neg_clicks: list[tuple[int, int]] = []

        self._refine_thread: QThread | None = None
        self._refine_worker: _RefinementWorker | None = None

        self._frame_h, self._frame_w = (
            (self._frames[0].shape[0], self._frames[0].shape[1])
            if self._frames
            else (1080, 1920)
        )

        self._build_ui()
        self._refresh_display()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Header instructions
        hdr = QLabel(
            "Left-click = include region (positive).  Right-click = exclude region (negative).  "
            "Click <b>Apply Clicks</b> to re-run SAM-2 with your prompts, or type a description "
            "and click <b>Re-segment</b> to re-run Grounded SAM-2."
        )
        hdr.setWordWrap(True)
        hdr.setStyleSheet("color: #bbb; font-size: 11px;")
        root.addWidget(hdr)

        # Frame selector row
        frame_sel_row = QHBoxLayout()
        self._btn_prev = QPushButton("◀ Prev")
        self._btn_prev.clicked.connect(self._prev_frame)
        self._btn_next = QPushButton("Next ▶")
        self._btn_next.clicked.connect(self._next_frame)
        self._frame_label = QLabel()
        self._frame_label.setStyleSheet("font-size: 11px; color: #aaa;")
        frame_sel_row.addWidget(self._btn_prev)
        frame_sel_row.addWidget(self._btn_next)
        frame_sel_row.addWidget(self._frame_label, stretch=1)
        root.addLayout(frame_sel_row)

        # Click-capture overlay inside a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._overlay_label = _ClickOverlay(self._frame_w, self._frame_h)
        self._overlay_label.sig_pos_click.connect(self._on_pos_click)
        self._overlay_label.sig_neg_click.connect(self._on_neg_click)
        scroll.setWidget(self._overlay_label)
        root.addWidget(scroll, stretch=1)

        # Click count display
        self._click_label = QLabel("Positive clicks: 0 | Negative clicks: 0")
        self._click_label.setStyleSheet("font-size: 10px; color: #888;")
        root.addWidget(self._click_label)

        # Text prompt row
        text_row = QHBoxLayout()
        self._text_input = QLineEdit()
        self._text_input.setPlaceholderText(
            "Describe the character (e.g. 'girl with blue sailor uniform and red hair')…"
        )
        self._btn_resegment = QPushButton("Re-segment (Grounded SAM-2)")
        self._btn_resegment.clicked.connect(self._on_resegment)
        self._btn_resegment.setEnabled(self._refine_callback is not None)
        text_row.addWidget(self._text_input, stretch=1)
        text_row.addWidget(self._btn_resegment)
        root.addLayout(text_row)

        # Apply clicks row
        clicks_row = QHBoxLayout()
        self._btn_apply_clicks = QPushButton("Apply Clicks (SAM-2 re-propagate)")
        self._btn_apply_clicks.clicked.connect(self._on_apply_clicks)
        self._btn_apply_clicks.setEnabled(self._refine_callback is not None)
        self._btn_clear_clicks = QPushButton("Clear Clicks")
        self._btn_clear_clicks.clicked.connect(self._clear_clicks)
        clicks_row.addWidget(self._btn_apply_clicks)
        clicks_row.addWidget(self._btn_clear_clicks)
        root.addLayout(clicks_row)

        # Progress bar (hidden when idle)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(4)
        root.addWidget(self._progress)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(line)

        # Accept / Skip / Cancel row
        bottom_row = QHBoxLayout()
        self._btn_accept = QPushButton("Accept Masks and Resume")
        self._btn_accept.setDefault(True)
        self._btn_accept.clicked.connect(self._on_accept)
        btn_skip = QPushButton("Skip (use original masks)")
        btn_skip.clicked.connect(self._on_skip)
        btn_cancel = QPushButton("Cancel Pipeline")
        btn_cancel.clicked.connect(self.reject)
        bottom_row.addStretch()
        bottom_row.addWidget(self._btn_accept)
        bottom_row.addWidget(btn_skip)
        bottom_row.addWidget(btn_cancel)
        root.addLayout(bottom_row)

    # ── display ───────────────────────────────────────────────────────────────

    def _refresh_display(self):
        idx = self._current_idx
        n = len(self._frames)
        if not self._frames or idx >= n:
            return

        frame = self._frames[idx]
        mask = self._masks[idx] if idx < len(self._masks) else None

        # Scale frame for display
        h, w = frame.shape[:2]
        scale = min(1.0, _MAX_DISPLAY_H / h)
        disp_w, disp_h = max(1, int(w * scale)), max(1, int(h * scale))

        rendered = _render_overlay(frame, mask, self._pos_clicks, self._neg_clicks)
        rendered_small = cv2.resize(
            rendered, (disp_w, disp_h), interpolation=cv2.INTER_AREA
        )

        self._overlay_label.setPixmap(QPixmap.fromImage(_bgr_to_qimage(rendered_small)))
        self._overlay_label.setFixedSize(disp_w, disp_h)

        path = self._paths[idx] if idx < len(self._paths) else f"frame {idx}"
        import os

        self._frame_label.setText(
            f"Frame {idx + 1} / {n}  —  {os.path.basename(path)}"
            f"  {'(no mask)' if mask is None else '(mask active)'}"
        )
        self._btn_prev.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < n - 1)
        self._click_label.setText(
            f"Positive clicks: {len(self._pos_clicks)} | Negative clicks: {len(self._neg_clicks)}"
        )

    # ── frame navigation ──────────────────────────────────────────────────────

    def _prev_frame(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self._clear_clicks(silent=True)
            self._refresh_display()

    def _next_frame(self):
        if self._current_idx < len(self._frames) - 1:
            self._current_idx += 1
            self._clear_clicks(silent=True)
            self._refresh_display()

    # ── click handlers ────────────────────────────────────────────────────────

    def _on_pos_click(self, fx: int, fy: int):
        self._pos_clicks.append((fx, fy))
        self._refresh_display()

    def _on_neg_click(self, fx: int, fy: int):
        self._neg_clicks.append((fx, fy))
        self._refresh_display()

    def _clear_clicks(self, silent: bool = False):
        self._pos_clicks.clear()
        self._neg_clicks.clear()
        if not silent:
            self._refresh_display()

    # ── refinement ────────────────────────────────────────────────────────────

    def _set_busy(self, busy: bool):
        self._progress.setVisible(busy)
        self._btn_accept.setEnabled(not busy)
        self._btn_resegment.setEnabled(not busy and self._refine_callback is not None)
        self._btn_apply_clicks.setEnabled(
            not busy and self._refine_callback is not None
        )

    def _on_resegment(self):
        """Re-run Grounded SAM-2 with the user's text prompt."""
        text = self._text_input.text().strip()
        if not text:
            self._text_input.setPlaceholderText("Please enter a description first…")
            return
        if self._refine_callback is None:
            return
        self._set_busy(True)
        fn = lambda: self._refine_callback(text, [], [], self._current_idx)  # noqa: E731
        self._launch_refinement(fn)

    def _on_apply_clicks(self):
        """Apply accumulated positive/negative clicks via SAM-2 re-propagation."""
        if not self._pos_clicks and not self._neg_clicks:
            return
        if self._refine_callback is None:
            return
        self._set_busy(True)
        pos = list(self._pos_clicks)
        neg = list(self._neg_clicks)
        idx = self._current_idx
        fn = lambda: self._refine_callback("", pos, neg, idx)  # noqa: E731
        self._launch_refinement(fn)

    def _launch_refinement(self, fn):
        self._refine_worker = _RefinementWorker(fn)
        self._refine_thread = self._refine_worker
        self._refine_worker.sig_done.connect(self._on_refinement_done)
        self._refine_worker.sig_error.connect(self._on_refinement_error)
        self._refine_worker.start()

    def _on_refinement_done(self, new_masks: list):
        self._refine_thread.quit() # pyrefly: ignore [missing-attribute]
        self._refine_thread.wait() # pyrefly: ignore [missing-attribute]
        self._refine_thread = None
        if new_masks:
            # Replace our mask list with the refined masks
            for i, m in enumerate(new_masks):
                if i < len(self._masks) and m is not None:
                    self._masks[i] = m
        self._clear_clicks(silent=True)
        self._set_busy(False)
        self._refresh_display()

    def _on_refinement_error(self, msg: str):
        if self._refine_thread:
            self._refine_thread.quit()
            self._refine_thread.wait()
            self._refine_thread = None
        self._set_busy(False)
        import warnings

        warnings.warn(f"[ASP] Mask refinement error: {msg}", RuntimeWarning, stacklevel=2)
        self._refresh_display()

    # ── accept / skip ─────────────────────────────────────────────────────────

    def _on_accept(self):
        self.sig_mask_accepted.emit(list(self._masks))
        self.accept()

    def _on_skip(self):
        # Emit original masks (from data dict) unchanged
        self.sig_mask_accepted.emit(list(self._masks))
        self.accept()

    def accepted_masks(self) -> list[np.ndarray | None]:
        """Access masks after dialog.exec() returns Accepted."""
        return list(self._masks)
