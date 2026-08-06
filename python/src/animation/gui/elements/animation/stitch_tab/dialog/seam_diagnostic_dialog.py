"""§2.4A/C/§2.11B — Seam Registration Inspector dialog (S95–S96, S124).

Surfaces per-seam diagnostic data from HITL checkpoint 4.6:
- post_warp_diff coloured green/amber/red (same thresholds as §2.4B overlay)
- single-pose escalation badge
- ±50px seam zone crop thumbnail (§2.4C, S96)
- "Force single-pose" / "Force blend" checkboxes per seam (mutually exclusive)
- "Accept" / "Cancel" buttons; "Accept" returns the user's per-seam overrides

S124 adds §2.11B — interactive waypoint placement on the canvas preview.
When "Add Waypoints" mode is active every left-click on the canvas preview
adds a waypoint assigned to the nearest seam boundary; waypoints are stored in
the override dict under ``seam_overrides[k]["waypoints"]`` which the backend
(§2.11A, S123) threads through ``_seam_cut()`` to force the DP seam through
user-designated pixels.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from backend.src.constants import SEAM_OVERLAY_AMBER_THRESH, SEAM_OVERLAY_RED_THRESH
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ── §2.11B: waypoint canvas ───────────────────────────────────────────────────

class _WaypointCanvas(QLabel):
    """Canvas preview with interactive waypoint placement (§2.11B, S124).

    In waypoint mode (:meth:`set_active` True) a left-click on the preview
    adds a canvas-space ``(x, y)`` waypoint assigned to the seam boundary whose
    y-position is nearest the click's canvas-y coordinate.  Waypoints are
    rendered as coloured dots with seam-index labels; a distinct colour is used
    per seam index (cycling over 10 palette entries for >10 seams).

    :meth:`all_waypoints` returns the current dict for :meth:`get_overrides`.
    :meth:`clear_seam_waypoints` removes all waypoints for a single seam.
    """

    waypoint_changed: Signal = Signal()  # emitted whenever the waypoints dict changes

    # Per-seam colour palette (cycles for >10 seams)
    _PALETTE: List[str] = [
        "#ff4444", "#44aa44", "#4488ff", "#cc8800",
        "#aa44aa", "#008888", "#ff8844", "#8844ff",
        "#44ff88", "#ff4488",
    ]

    def __init__(
        self,
        base_pixmap: QPixmap,
        canvas_w: int,
        canvas_h: int,
        boundaries: List[float],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._base_pix: QPixmap = base_pixmap
        self._canvas_w: int = max(1, canvas_w)
        self._canvas_h: int = max(1, canvas_h)
        self._boundaries: List[float] = list(boundaries)
        pw, ph = max(1, base_pixmap.width()), max(1, base_pixmap.height())
        self._scale_x: float = pw / self._canvas_w
        self._scale_y: float = ph / self._canvas_h
        self._active: bool = False
        self._waypoints: Dict[int, List[Tuple[int, int]]] = {}
        self._redraw()

    # ── public API ───────────────────────────────────────────────────────────

    def set_active(self, active: bool) -> None:
        """Enable or disable waypoint-placement mode."""
        self._active = active
        cursor = Qt.CursorShape.CrossCursor if active else Qt.CursorShape.ArrowCursor
        self.setCursor(cursor)

    def clear_seam_waypoints(self, k: int) -> None:
        """Remove all waypoints for seam *k* and redraw."""
        if self._waypoints.pop(k, None) is not None:
            self._redraw()
            self.waypoint_changed.emit()

    def all_waypoints(self) -> Dict[int, List[Tuple[int, int]]]:
        """Return a shallow copy of the current waypoints dict (seam_k → list)."""
        return {k: list(v) for k, v in self._waypoints.items() if v}

    def waypoint_count(self, k: int) -> int:
        """Number of waypoints currently set for seam *k*."""
        return len(self._waypoints.get(k, []))

    # ── Qt overrides ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._active and event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            cx = int(pos.x() / self._scale_x)
            cy = int(pos.y() / self._scale_y)
            cx = max(0, min(self._canvas_w - 1, cx))
            cy = max(0, min(self._canvas_h - 1, cy))
            k = self._nearest_seam(cy)
            if k >= 0:
                self._waypoints.setdefault(k, []).append((cx, cy))
                self._redraw()
                self.waypoint_changed.emit()
        super().mousePressEvent(event)

    # ── internals ────────────────────────────────────────────────────────────

    def _nearest_seam(self, canvas_y: int) -> int:
        """Return the seam index whose boundary y is closest to *canvas_y*."""
        if not self._boundaries:
            return -1
        return min(range(len(self._boundaries)), key=lambda i: abs(canvas_y - self._boundaries[i]))

    def _redraw(self) -> None:
        """Composite base pixmap with waypoint dots and update label."""
        pix = self._base_pix.copy()
        if self._waypoints:
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            font = QFont()
            font.setPointSize(7)
            painter.setFont(font)
            for k, wps in self._waypoints.items():
                colour = QColor(self._PALETTE[k % len(self._PALETTE)])
                painter.setPen(QPen(colour, 2))
                painter.setBrush(colour)
                for cx, cy in wps:
                    dx = int(cx * self._scale_x)
                    dy = int(cy * self._scale_y)
                    painter.drawEllipse(dx - 4, dy - 4, 8, 8)
                    # Tiny label so the user can see which seam each dot belongs to
                    painter.setPen(QPen(QColor("white"), 1))
                    painter.drawText(dx + 6, dy + 4, f"S{k}")
                    painter.setPen(QPen(colour, 2))
            painter.end()
        self.setPixmap(pix)


# ── §2.10C: user-drawn flow field canvas ─────────────────────────────────────

class _FlowArrowCanvas(QLabel):
    """Canvas overlay that lets the user draw displacement arrows (§2.10C).

    In draw mode the user clicks once (origin) and releases at the tip of the
    desired displacement arrow.  Each arrow records ``(x, y, dx, dy)`` in
    canvas-pixel space.  A "Clear" button removes all arrows.

    :meth:`all_flow_arrows` returns the list for :meth:`get_overrides`.
    """

    flow_changed: Signal = Signal()

    _ARROW_COLOR = QColor(255, 180, 0)

    def __init__(
        self,
        base_pixmap: QPixmap,
        canvas_w: int,
        canvas_h: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._base_pix: QPixmap = base_pixmap
        self._canvas_w: int = max(1, canvas_w)
        self._canvas_h: int = max(1, canvas_h)
        pw = max(1, base_pixmap.width())
        ph = max(1, base_pixmap.height())
        self._scale_x: float = pw / self._canvas_w
        self._scale_y: float = ph / self._canvas_h
        self._active: bool = False
        self._origin: Optional[Tuple[float, float]] = None  # current drag origin (canvas px)
        self._arrows: List[Tuple[float, float, float, float]] = []  # (x, y, dx, dy)
        self._redraw()

    def set_active(self, active: bool) -> None:
        self._active = active
        self.setCursor(Qt.CursorShape.CrossCursor if active else Qt.CursorShape.ArrowCursor)

    def all_flow_arrows(self) -> List[Tuple[float, float, float, float]]:
        return list(self._arrows)

    def clear(self) -> None:
        self._arrows.clear()
        self._origin = None
        self._redraw()
        self.flow_changed.emit()

    def mousePressEvent(self, event) -> None:
        if self._active and event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            cx = pos.x() / self._scale_x
            cy = pos.y() / self._scale_y
            self._origin = (cx, cy)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._active and event.button() == Qt.MouseButton.LeftButton and self._origin is not None:
            pos = event.position()
            ex = pos.x() / self._scale_x
            ey = pos.y() / self._scale_y
            ox, oy = self._origin
            self._origin = None
            dx = ex - ox
            dy = ey - oy
            if abs(dx) > 2 or abs(dy) > 2:  # ignore micro-clicks
                self._arrows.append((ox, oy, dx, dy))
                self._redraw()
                self.flow_changed.emit()
        super().mouseReleaseEvent(event)

    def _redraw(self) -> None:
        pix = self._base_pix.copy()
        if self._arrows:
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(self._ARROW_COLOR, 2)
            painter.setPen(pen)
            for ox, oy, dx, dy in self._arrows:
                sx0 = int(ox * self._scale_x)
                sy0 = int(oy * self._scale_y)
                sx1 = int((ox + dx) * self._scale_x)
                sy1 = int((oy + dy) * self._scale_y)
                painter.drawLine(sx0, sy0, sx1, sy1)
                # Arrowhead dot
                painter.setBrush(self._ARROW_COLOR)
                painter.drawEllipse(sx1 - 3, sy1 - 3, 6, 6)
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.end()
        self.setPixmap(pix)


# ── per-seam diagnostic card ──────────────────────────────────────────────────

class _SeamCard(QFrame):
    """Compact per-seam diagnostic card.

    Displays seam index, boundary y-position, post_warp_diff (coloured),
    single-pose status, optional ±50px crop thumbnail (§2.4C), two
    mutually-exclusive override checkboxes, and (§2.11B) a waypoint count
    badge with a "Clear WPs" button.
    """

    def __init__(
        self,
        k: int,
        boundary_y: float,
        post_diff: float,
        is_single_pose: bool,
        crop: Optional[np.ndarray] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._k = k
        self._post_diff = post_diff
        self._is_single_pose = is_single_pose

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        has_crop = crop is not None and crop.size > 0
        self.setMinimumHeight(56 + (70 if has_crop else 0))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        row = QHBoxLayout()
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(12)
        outer.addLayout(row)

        # ── Seam identity ──────────────────────────────────────────────────
        id_lbl = QLabel(f"S{k}")
        id_lbl.setFixedWidth(28)
        id_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        id_lbl.setStyleSheet("font-weight:bold; font-size:12px;")
        row.addWidget(id_lbl)

        # Boundary position
        pos_lbl = QLabel(f"y={int(boundary_y)}px")
        pos_lbl.setFixedWidth(64)
        row.addWidget(pos_lbl)

        # post_warp_diff with colour coding
        if is_single_pose or post_diff >= SEAM_OVERLAY_RED_THRESH:
            diff_colour = "#cc2200"
        elif post_diff >= SEAM_OVERLAY_AMBER_THRESH:
            diff_colour = "#cc7700"
        else:
            diff_colour = "#007700"

        diff_text = f"diff={post_diff:.1f}" if post_diff < 90.0 else "diff=forced"
        diff_lbl = QLabel(diff_text)
        diff_lbl.setStyleSheet(
            f"color:{diff_colour}; font-weight:bold; min-width:80px;"
        )
        row.addWidget(diff_lbl)

        # Single-pose badge
        sp_lbl = QLabel("SP" if is_single_pose else "—")
        sp_lbl.setFixedWidth(28)
        sp_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if is_single_pose:
            sp_lbl.setStyleSheet(
                "background:#cc2200; color:white; border-radius:3px; font-size:10px;"
            )
        row.addWidget(sp_lbl)

        row.addStretch(1)

        # ── §2.11B: waypoint count badge + clear button ────────────────────
        self._wp_lbl = QLabel("0 wps")
        self._wp_lbl.setFixedWidth(44)
        self._wp_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._wp_lbl.setStyleSheet("color:gray; font-size:10px;")
        row.addWidget(self._wp_lbl)

        self._btn_clear_wps = QPushButton("Clear WPs")
        self._btn_clear_wps.setFixedWidth(68)
        self._btn_clear_wps.setToolTip(f"Remove all waypoints for seam S{k}")
        self._btn_clear_wps.setVisible(False)
        row.addWidget(self._btn_clear_wps)

        # Override checkboxes (mutually exclusive)
        self._cb_sp = QCheckBox("Force SP")
        self._cb_sp.setToolTip("Force this seam to use the dominant single-pose frame (no blend)")
        self._cb_blend = QCheckBox("Force blend")
        self._cb_blend.setToolTip("Force this seam to use DSFN blend even if post_warp_diff is high")
        self._cb_sp.toggled.connect(self._on_sp_toggled)
        self._cb_blend.toggled.connect(self._on_blend_toggled)
        row.addWidget(self._cb_sp)
        row.addWidget(self._cb_blend)

        # ── §2.4C: seam zone crop thumbnail ───────────────────────────────
        if has_crop:
            pix = self._make_crop_pixmap(crop, max_width=300, max_height=64)
            crop_lbl = QLabel()
            crop_lbl.setPixmap(pix)
            crop_lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            crop_lbl.setContentsMargins(8, 0, 8, 4)
            outer.addWidget(crop_lbl)

    # ── §2.11B: waypoint count update ────────────────────────────────────────

    def update_waypoint_count(self, n: int) -> None:
        """Refresh waypoint badge.  Shows "N wps" in blue when N > 0."""
        label = f"{n} wp{'s' if n != 1 else ''}"
        self._wp_lbl.setText(label)
        if n > 0:
            self._wp_lbl.setStyleSheet(
                "color:#3366cc; font-size:10px; font-weight:bold;"
            )
        else:
            self._wp_lbl.setStyleSheet("color:gray; font-size:10px;")
        self._btn_clear_wps.setVisible(n > 0)

    @property
    def clear_waypoints_button(self) -> QPushButton:
        """Return the 'Clear WPs' button so the dialog can connect its signal."""
        return self._btn_clear_wps

    # ── §2.4C helper ────────────────────────────────────────────────────────

    @staticmethod
    def _make_crop_pixmap(
        arr: np.ndarray, max_width: int = 300, max_height: int = 64
    ) -> QPixmap:
        """Scale a BGR crop array to fit within (max_width, max_height) and return QPixmap."""
        h, w = arr.shape[:2]
        scale = min(max_width / max(w, 1), max_height / max(h, 1), 1.0)
        if scale < 1.0:
            # pyrefly: ignore [no-matching-overload]
            arr = cv2.resize(
                arr, (max(1, int(w * scale)), max(1, int(h * scale))), cv2.INTER_AREA
            )
            h, w = arr.shape[:2]
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    # ── mutual exclusion ────────────────────────────────────────────────────

    def _on_sp_toggled(self, checked: bool) -> None:
        if checked:
            self._cb_blend.blockSignals(True)
            self._cb_blend.setChecked(False)
            self._cb_blend.blockSignals(False)

    def _on_blend_toggled(self, checked: bool) -> None:
        if checked:
            self._cb_sp.blockSignals(True)
            self._cb_sp.setChecked(False)
            self._cb_sp.blockSignals(False)

    # ── accessors ───────────────────────────────────────────────────────────

    @property
    def seam_index(self) -> int:
        return self._k

    def force_single_pose(self) -> bool:
        return self._cb_sp.isChecked()

    def force_blend(self) -> bool:
        return self._cb_blend.isChecked()

    def has_override(self) -> bool:
        return self._cb_sp.isChecked() or self._cb_blend.isChecked()


# ── main dialog ───────────────────────────────────────────────────────────────

class SeamDiagnosticDialog(QDialog):
    """§2.4A — Per-seam diagnostic panel with per-seam overrides and waypoints.

    Shows a scrollable list of :class:`_SeamCard` widgets (one per seam boundary,
    sorted worst-first by post_warp_diff) alongside an interactive canvas preview.

    §2.11B (S124): the canvas preview is a :class:`_WaypointCanvas` that
    accepts left-clicks when "Add Waypoints" mode is active.  Each click
    plants a waypoint on the nearest seam boundary in canvas space.  On Accept
    the waypoints are included in the override dict returned by
    :meth:`get_overrides` under the ``"waypoints"`` key, which the backend
    (§2.11A, S123) threads through ``_seam_cut()`` to force the DP seam path
    through the user-designated pixels.
    """

    def __init__(
        self,
        data: dict,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Parameters
        ----------
        data:
            Dict from HITL checkpoint 4.6 with keys:
            ``canvas_preview`` (np.ndarray BGR), ``boundaries`` (list[float]),
            ``seam_post_diffs`` (dict[int, float]),
            ``seam_single_pose_keys`` (list[int]),
            ``canvas_h`` (int), ``canvas_w`` (int).
        """
        super().__init__(parent)
        self.setWindowTitle("Seam Registration Inspector — §2.4A/§2.11B")
        self.setMinimumSize(700, 450)
        self.setSizeGripEnabled(True)

        boundaries: List[float] = data.get("boundaries", [])
        seam_post_diffs: dict = data.get("seam_post_diffs", {})
        sp_keys: List[int] = data.get("seam_single_pose_keys", [])
        canvas_preview: Optional[np.ndarray] = data.get("canvas_preview")
        seam_crops: dict = data.get("seam_crops", {})
        canvas_h: int = int(data.get("canvas_h", 0))
        canvas_w: int = int(data.get("canvas_w", 0))

        self._canvas: Optional[_WaypointCanvas] = None
        self._flow_canvas: Optional[_FlowArrowCanvas] = None

        # ── Root layout ────────────────────────────────────────────────────
        root = QHBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(10, 10, 10, 10)

        # ── Left panel: waypoint canvas + mode button ──────────────────────
        if canvas_preview is not None and canvas_preview.size > 0:
            left = QVBoxLayout()
            left.setSpacing(4)
            root.addLayout(left)

            _pix = self._np_to_pixmap(canvas_preview, max_width=260)
            _cw = canvas_w if canvas_w > 0 else max(1, _pix.width())
            _ch = canvas_h if canvas_h > 0 else max(1, _pix.height())
            self._canvas = _WaypointCanvas(
                _pix,
                canvas_w=_cw,
                canvas_h=_ch,
                boundaries=boundaries,
            )
            self._canvas.setAlignment(Qt.AlignmentFlag.AlignTop)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
            )
            self._canvas.waypoint_changed.connect(self._on_waypoint_changed)
            left.addWidget(self._canvas)

            # §2.11B: "Add Waypoints" toggle button
            self._btn_wps = QPushButton("📍 Add Waypoints")
            self._btn_wps.setCheckable(True)
            self._btn_wps.setToolTip(
                "Click seam positions on the canvas preview to force the DP seam path\n"
                "through those pixels (§2.11A backend already handles routing)."
            )
            self._btn_wps.toggled.connect(self._on_wp_mode_toggled)
            left.addWidget(self._btn_wps)

            # §2.10C: Flow Arrow canvas (shared pixmap, overlaid on canvas preview)
            self._flow_canvas = _FlowArrowCanvas(_pix, canvas_w=_cw, canvas_h=_ch)
            self._flow_canvas.setAlignment(Qt.AlignmentFlag.AlignTop)
            self._flow_canvas.setSizePolicy(
                QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
            )
            self._flow_canvas.flow_changed.connect(self._on_flow_changed)
            self._flow_canvas.setVisible(False)  # shown only in draw-flow mode
            left.addWidget(self._flow_canvas)

            btn_flow_row = QHBoxLayout()
            self._btn_flow = QPushButton("↗ Draw Flow")
            self._btn_flow.setCheckable(True)
            self._btn_flow.setToolTip(
                "§2.10C  Click-drag arrows on the seam crop to define displacement vectors.\n"
                "These override RAFT/DIS flow for the re-composite pass.\n"
                "Stored as 'flow_arrows' in the seam override dict."
            )
            self._btn_flow.toggled.connect(self._on_flow_mode_toggled)
            self._btn_flow_clear = QPushButton("Clear Flow")
            self._btn_flow_clear.setToolTip("Remove all drawn flow arrows for this seam.")
            self._btn_flow_clear.clicked.connect(self._on_flow_clear)
            self._btn_flow_clear.setEnabled(False)
            btn_flow_row.addWidget(self._btn_flow)
            btn_flow_row.addWidget(self._btn_flow_clear)
            left.addLayout(btn_flow_row)

            left.addStretch(1)

        # ── Right panel: seam cards + buttons ─────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(6)
        root.addLayout(right)

        # Header
        header_lbl = QLabel(
            f"<b>{len(boundaries)} seam(s)</b> — sorted by alignment score (worst first)"
        )
        right.addWidget(header_lbl)

        # Scroll area with seam cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        card_layout = QVBoxLayout(inner)
        card_layout.setSpacing(4)
        card_layout.setContentsMargins(4, 4, 4, 4)

        # Build cards, sort worst-first by post_diff
        seam_info: List[tuple] = []
        for k, by in enumerate(boundaries):
            diff = float(seam_post_diffs.get(k, 0.0))
            is_sp = k in sp_keys
            seam_info.append((k, by, diff, is_sp))
        seam_info.sort(key=lambda t: (t[3], t[2]), reverse=True)  # SP first, then high diff

        self._cards: List[_SeamCard] = []
        for k, by, diff, is_sp in seam_info:
            card = _SeamCard(k, by, diff, is_sp, crop=seam_crops.get(k))
            if self._canvas is not None:
                # §2.11B: wire "Clear WPs" button on each card
                card.clear_waypoints_button.clicked.connect(
                    lambda _checked=False, _k=k: self._canvas.clear_seam_waypoints(_k)  # type: ignore[union-attr]
                )
            card_layout.addWidget(card)
            self._cards.append(card)

        card_layout.addStretch(1)
        scroll.setWidget(inner)
        right.addWidget(scroll, 1)

        # Legend
        legend = QLabel(
            "<small>"
            "<span style='color:#007700'>■</span> good (diff&lt;10)  "
            "<span style='color:#cc7700'>■</span> moderate (10–22)  "
            "<span style='color:#cc2200'>■</span> poor / single-pose  "
            "SP = already in single-pose fallback  "
            "📍 = click canvas to add seam waypoints"
            "</small>"
        )
        legend.setWordWrap(True)
        right.addWidget(legend)

        # Dialog buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.button(QDialogButtonBox.StandardButton.Ok).setText("Accept and Continue")
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        right.addWidget(btn_box)

    # ── §2.11B: waypoint mode slots ──────────────────────────────────────────

    def _on_wp_mode_toggled(self, active: bool) -> None:
        """Enable/disable waypoint-placement mode on the canvas."""
        if self._canvas is not None:
            self._canvas.set_active(active)

    def _on_waypoint_changed(self) -> None:
        """Refresh waypoint count badges on all cards when canvas updates."""
        if self._canvas is None:
            return
        for card in self._cards:
            n = self._canvas.waypoint_count(card.seam_index)
            card.update_waypoint_count(n)

    def _on_flow_mode_toggled(self, active: bool) -> None:
        """§2.10C: Switch the left panel between waypoint canvas and flow-arrow canvas."""
        if self._flow_canvas is None:
            return
        if active:
            if self._canvas is not None:
                self._canvas.setVisible(False)
                if hasattr(self, "_btn_wps"):
                    self._btn_wps.setChecked(False)
            self._flow_canvas.setVisible(True)
            self._flow_canvas.set_active(True)
        else:
            self._flow_canvas.setVisible(False)
            self._flow_canvas.set_active(False)
            if self._canvas is not None:
                self._canvas.setVisible(True)

    def _on_flow_changed(self) -> None:
        """Enable/disable the clear button based on arrow count."""
        if self._flow_canvas is not None and hasattr(self, "_btn_flow_clear"):
            self._btn_flow_clear.setEnabled(bool(self._flow_canvas.all_flow_arrows()))

    def _on_flow_clear(self) -> None:
        if self._flow_canvas is not None:
            self._flow_canvas.clear()

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _np_to_pixmap(arr: np.ndarray, max_width: int = 260) -> QPixmap:
        """Convert a BGR numpy array to a QPixmap scaled to *max_width*."""
        h, w = arr.shape[:2]
        if w > max_width:
            scale = max_width / w
            arr = cv2.resize(arr, (max_width, max(1, int(h * scale))), cv2.INTER_AREA) # pyrefly: ignore [no-matching-overload]
            h, w = arr.shape[:2]
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    # ── public API ───────────────────────────────────────────────────────────

    def get_overrides(self) -> Dict[int, dict]:
        """Return per-seam override dict (only non-default seams included).

        Each entry may contain any combination of:
        - ``"force_single_pose"`` (bool) — skip ARAP, use dominant frame
        - ``"force_blend"`` (bool) — force Laplacian blend regardless of diff
        - ``"waypoints"`` (List[Tuple[int,int]]) — canvas-space (x,y) pairs
          that ``_seam_cut()`` must route through (§2.11A)
        - ``"flow_arrows"`` (List[Tuple[float,float,float,float]]) — user-drawn
          displacement arrows ``(x, y, dx, dy)`` for §2.10C flow override
        """
        all_wps: Dict[int, List[Tuple[int, int]]] = (
            self._canvas.all_waypoints() if self._canvas is not None else {}
        )
        flow_arrows_global: List[Tuple[float, float, float, float]] = (
            self._flow_canvas.all_flow_arrows() if self._flow_canvas is not None else []
        )
        # Collect all seams with any kind of override
        seam_keys: set = {card.seam_index for card in self._cards if card.has_override()}
        seam_keys.update(all_wps.keys())
        # Flow arrows apply globally (to all seams) when present — caller can
        # decide per-seam routing; store under every seam key if any card is focused,
        # or just attach to seam 0 when no card override exists.
        if flow_arrows_global:
            if seam_keys:
                for _k in set(seam_keys):
                    pass  # each seam gets the arrows below
            else:
                seam_keys.add(0)

        result: Dict[int, dict] = {}
        for k in seam_keys:
            entry: dict = {}
            card = next((c for c in self._cards if c.seam_index == k), None)
            if card is not None:
                if card.force_single_pose():
                    entry["force_single_pose"] = True
                if card.force_blend():
                    entry["force_blend"] = True
            wps = all_wps.get(k, [])
            if wps:
                entry["waypoints"] = wps
            if flow_arrows_global:
                entry["flow_arrows"] = flow_arrows_global
            if entry:
                result[k] = entry

        return result
