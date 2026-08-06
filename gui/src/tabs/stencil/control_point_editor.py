import math
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import (
    QPointF,
    QRectF,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ....constants import (
    STITCH_CP_COLORS,
    STITCH_THUMB_H,
    STITCH_THUMB_W,
)
from ....styles import apply_shadow_effect
from ....windows.settings.splitter_persistence import persist_splitter


def _load_thumb(path: str, w: int = STITCH_THUMB_W, h: int = STITCH_THUMB_H) -> QPixmap:
    bgr = cv2.imread(path)
    if bgr is None:
        pm = QPixmap(w, h)
        pm.fill(QColor("#333"))
        return pm
    ih, iw = bgr.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qi = QImage(rgb.data, nw, nh, nw * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qi)


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


def _apply_color_correction(bgr: np.ndarray, cc: dict) -> np.ndarray:
    """Apply per-frame color correction dict to a BGR uint8 image."""
    if not cc:
        return bgr
    img = bgr.astype(np.float32)
    # Brightness [-100 .. +100]
    img += float(cc.get("brightness", 0))
    # Contrast  [0.1 .. 3.0, 1.0 = no change]
    img = (img - 127.5) * float(cc.get("contrast", 1.0)) + 127.5
    # Saturation [0.0 .. 3.0, 1.0 = no change]
    sat = float(cc.get("saturation", 1.0))
    if sat != 1.0:
        gray = img.mean(axis=2, keepdims=True)
        img = gray + sat * (img - gray)
    # Gamma [0.2 .. 5.0, 1.0 = no change]
    gamma = float(cc.get("gamma", 1.0))
    if gamma != 1.0:
        img = np.sign(img) * (np.abs(img / 255.0) ** (1.0 / gamma)) * 255.0
    # White balance — temperature offset in blue/red channels [-50 .. +50]
    temp = float(cc.get("temperature", 0))
    img[:, :, 0] -= temp  # blue
    img[:, :, 2] += temp  # red
    return np.clip(img, 0, 255).astype(np.uint8)


class _CPDot(QGraphicsEllipseItem):
    """A draggable numbered control-point dot."""

    R = 8  # radius in scene pixels

    def __init__(self, canvas: "_CPCanvas", idx: int):
        super().__init__(-self.R, -self.R, 2 * self.R, 2 * self.R)
        self._canvas = canvas
        self._idx = idx
        color = STITCH_CP_COLORS[idx % len(STITCH_CP_COLORS)]
        self.setBrush(QBrush(color))
        self.setPen(QPen(QColor("white"), 1.5))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(20)

    # -- position change hook --
    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._canvas._on_dot_moved(self._idx)
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        painter.setPen(QPen(QColor("white")))
        painter.setFont(QFont("Arial", max(6, int(self.R * 0.85)), QFont.Weight.Bold))
        painter.drawText(
            QRectF(-self.R, -self.R, 2 * self.R, 2 * self.R),
            Qt.AlignmentFlag.AlignCenter,
            str(self._idx + 1),
        )

    def contextMenuEvent(self, event):
        self._canvas.remove_point(self._idx)


class _CPCanvas(QGraphicsView):
    """
    Single-image canvas for placing control points.

    Left-click  : add a point (in "add" mode)
    Left-drag   : move an existing point
    Right-click : remove nearest point
    """

    point_added = Signal(int, float, float)  # idx, x_img, y_img
    point_moved = Signal(int, float, float)
    point_removed = Signal(int)

    def __init__(self, label: str = "", parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._label_str = label
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._dots: List[_CPDot] = []
        self._img_w = 1
        self._img_h = 1
        self._mode = "add"  # "add" | "view"
        self._current_path: str = ""

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor("#1a1a1a")))
        self.setMinimumSize(300, 220)

    # ── public API ──────────────────────────────────────────────────────

    def load_image(self, path: str, corrections: dict | None = None):
        self._current_path = path
        bgr = cv2.imread(path)
        if bgr is None:
            return
        if corrections:
            bgr = _apply_color_correction(bgr, corrections)
        self._img_h, self._img_w = bgr.shape[:2]
        qi = _bgr_to_qimage(bgr)
        pm = QPixmap.fromImage(qi)
        scene = self._scene
        if self._pix_item is not None:
            scene.removeItem(self._pix_item)
        self._pix_item = QGraphicsPixmapItem(pm)
        scene.addItem(self._pix_item)
        scene.setSceneRect(0, 0, self._img_w, self._img_h)
        self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        QTimer.singleShot(50, self._fit_image)
        QTimer.singleShot(200, self._fit_image)

    def set_mode(self, mode: str):
        self._mode = mode
        if mode == "add":
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def add_point(self, x_img: float, y_img: float) -> int:
        idx = len(self._dots)
        dot = _CPDot(self, idx)
        dot.setPos(x_img, y_img)
        self._scene.addItem(dot)
        self._dots.append(dot)
        self.point_added.emit(idx, x_img, y_img)
        return idx

    def set_point_pos(self, idx: int, x_img: float, y_img: float):
        if 0 <= idx < len(self._dots):
            self._dots[idx].setPos(x_img, y_img)

    def remove_point(self, idx: int):
        if not (0 <= idx < len(self._dots)):
            return
        dot = self._dots.pop(idx)
        self.scene().removeItem(dot)
        # Renumber remaining dots
        for i, d in enumerate(self._dots):
            d._idx = i
            d.update()
        self.point_removed.emit(idx)

    def clear_points(self):
        for dot in self._dots:
            self.scene().removeItem(dot)
        self._dots.clear()

    def point_positions(self) -> List[Tuple[float, float]]:
        return [(d.pos().x(), d.pos().y()) for d in self._dots]

    # ── internal callbacks ───────────────────────────────────────────

    def _on_dot_moved(self, idx: int):
        if 0 <= idx < len(self._dots):
            p = self._dots[idx].pos()
            self.point_moved.emit(idx, p.x(), p.y())

    # ── event overrides ──────────────────────────────────────────────

    def mousePressEvent(self, event):
        if self._mode == "add" and event.button() == Qt.MouseButton.LeftButton:
            # Only add if not clicking an existing dot
            pt = self.mapToScene(event.position().toPoint())
            items = self._scene.items(pt)
            if not any(isinstance(it, _CPDot) for it in items):
                self.add_point(pt.x(), pt.y())
                return
        if event.button() == Qt.MouseButton.RightButton:
            pt = self.mapToScene(event.position().toPoint())
            self._remove_nearest(pt)
            return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def _remove_nearest(self, scene_pt: QPointF):
        if not self._dots:
            return
        best_i, best_d = 0, float("inf")
        for i, dot in enumerate(self._dots):
            dx = dot.pos().x() - scene_pt.x()
            dy = dot.pos().y() - scene_pt.y()
            d = dx * dx + dy * dy
            if d < best_d:
                best_d, best_i = d, i
        if best_d < (50**2):
            self.remove_point(best_i)

    def _fit_image(self):
        if self._pix_item is not None:
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_image()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_image()


class ControlPointEditor(QWidget):
    """
    Two _CPCanvas panels side by side.  Manages synchronized point
    lists: left[i] ↔ right[i].

    Solve modes
    -----------
    DLT (exact)   : cv2.findHomography(pts, method=0) — passes through
                    every pair, no outlier rejection.  Use when you trust
                    all your correspondences.
    DLT + RANSAC  : cv2.findHomography(pts, method=cv2.RANSAC) — robust
                    to a few bad user clicks.
    Auto + Manual : run ORB first, then append user pairs on top; re-solve.
    """

    homography_solved = Signal(np.ndarray, float)  # H (3×3), mean_reprojection_err

    def __init__(self, parent=None):
        super().__init__(parent)
        self._H: Optional[np.ndarray] = None
        self._cc_a: dict = {}
        self._cc_b: dict = {}
        self._auto_pts_a: List[Tuple[float, float]] = []
        self._auto_pts_b: List[Tuple[float, float]] = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Toolbar ──────────────────────────────────────────────────
        tb = QHBoxLayout()
        self._mode_btn = QPushButton("✎ Add Points")
        self._mode_btn.setCheckable(True)
        self._mode_btn.setChecked(True)
        self._mode_btn.toggled.connect(self._on_mode_toggle)
        apply_shadow_effect(self._mode_btn, radius=4, y_offset=2)
        tb.addWidget(self._mode_btn)

        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._clear_all)
        apply_shadow_effect(btn_clear, radius=4, y_offset=2)
        tb.addWidget(btn_clear)

        btn_auto = QPushButton("⚡ Auto-Detect (ORB)")
        btn_auto.setToolTip(
            "Run ORB+RANSAC to populate initial correspondences automatically."
        )
        btn_auto.clicked.connect(self._auto_detect)
        apply_shadow_effect(btn_auto, radius=4, y_offset=2)
        tb.addWidget(btn_auto)

        btn_suggest = QPushButton("💡 Suggest Next")
        btn_suggest.setToolTip(
            "Propose the best unused ORB match as the next point pair."
        )
        btn_suggest.clicked.connect(self._suggest_next)
        apply_shadow_effect(btn_suggest, radius=4, y_offset=2)
        tb.addWidget(btn_suggest)

        tb.addStretch()

        self._solve_mode = QComboBox()
        self._solve_mode.addItems(["DLT (exact)", "DLT + RANSAC", "Auto + Manual"])
        self._solve_mode.setToolTip(
            "DLT (exact): homography passes through every pair — "
            "trusts all user points.\n"
            "DLT + RANSAC: robust to a few bad clicks.\n"
            "Auto + Manual: appends user pairs on top of auto-detected ones."
        )
        tb.addWidget(QLabel("Solve:"))
        tb.addWidget(self._solve_mode)

        btn_solve = QPushButton("🔧 Solve Homography")
        btn_solve.setStyleSheet(
            "background:#1976D2; color:white; font-weight:bold; padding:4px 10px;"
        )
        btn_solve.clicked.connect(self._solve)
        apply_shadow_effect(btn_solve, radius=4, y_offset=2)
        tb.addWidget(btn_solve)

        root.addLayout(tb)

        # ── Dual canvas ───────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setContentsMargins(0, 0, 0, 0)
        self._lbl_a = QLabel("Frame A  (click to place points)")
        self._lbl_a.setStyleSheet("color:#aaa; font-size:10px;")
        left_l.addWidget(self._lbl_a)
        self._canvas_a = _CPCanvas("A")
        self._canvas_a.point_added.connect(self._on_a_added)
        self._canvas_a.point_moved.connect(self._on_a_moved)
        self._canvas_a.point_removed.connect(self._on_a_removed)
        left_l.addWidget(self._canvas_a)
        splitter.addWidget(left_w)

        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(0, 0, 0, 0)
        self._lbl_b = QLabel("Frame B  (click to place matching points)")
        self._lbl_b.setStyleSheet("color:#aaa; font-size:10px;")
        right_l.addWidget(self._lbl_b)
        self._canvas_b = _CPCanvas("B")
        self._canvas_b.point_added.connect(self._on_b_added)
        self._canvas_b.point_moved.connect(self._on_b_moved)
        self._canvas_b.point_removed.connect(self._on_b_removed)
        right_l.addWidget(self._canvas_b)
        splitter.addWidget(right_w)

        persist_splitter(splitter, "HybridStitchPanel/dual_canvas")
        root.addWidget(splitter, 1)

        # ── Status ───────────────────────────────────────────────────
        self._status = QLabel("Load two frames, then click to place correspondences.")
        self._status.setStyleSheet("color:#aaa; font-size:10px;")
        root.addWidget(self._status)

    # ── Public API ──────────────────────────────────────────────────

    def load_pair(
        self,
        path_a: str,
        path_b: str,
        cc_a: dict | None = None,
        cc_b: dict | None = None,
    ):
        self._path_a = path_a
        self._path_b = path_b
        self._cc_a = cc_a or {}
        self._cc_b = cc_b or {}
        self._auto_pts_a.clear()
        self._auto_pts_b.clear()
        self._canvas_a.clear_points()
        self._canvas_b.clear_points()
        self._canvas_a.load_image(path_a, self._cc_a)
        self._canvas_b.load_image(path_b, self._cc_b)
        self._lbl_a.setText(f"Frame A:  {os.path.basename(path_a)}")
        self._lbl_b.setText(f"Frame B:  {os.path.basename(path_b)}")
        self._status.setText(
            "Frames loaded. Place ≥4 corresponding points, then Solve."
        )

    def get_homography(self) -> Optional[np.ndarray]:
        return self._H

    def fit_views(self):
        """Re-fit both canvases — call after the tab becomes visible."""
        self._canvas_a._fit_image()
        self._canvas_b._fit_image()

    # ── Slots ────────────────────────────────────────────────────────

    @Slot(bool)
    def _on_mode_toggle(self, checked: bool):
        mode = "add" if checked else "view"
        self._canvas_a.set_mode(mode)
        self._canvas_b.set_mode(mode)
        self._mode_btn.setText("✎ Add Points" if checked else "✋ Pan/Zoom")

    def _clear_all(self):
        self._canvas_a.clear_points()
        self._canvas_b.clear_points()
        self._auto_pts_a.clear()
        self._auto_pts_b.clear()
        self._H = None
        self._status.setText("Points cleared.")

    def _auto_detect(self):
        if not hasattr(self, "_path_a"):
            return
        self.setCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            pts_a, pts_b = self._run_orb(self._path_a, self._path_b)
            if len(pts_a) < 4:
                self._status.setText("Auto-detect: fewer than 4 matches found.")
                return
            self._canvas_a.clear_points()
            self._canvas_b.clear_points()
            self._auto_pts_a = list(pts_a)
            self._auto_pts_b = list(pts_b)
            for x, y in pts_a:
                self._canvas_a.add_point(x, y)
            for x, y in pts_b:
                self._canvas_b.add_point(x, y)
            self._status.setText(
                f"Auto-detect: {len(pts_a)} ORB+RANSAC matches loaded.  "
                "Add/move points to refine, then Solve."
            )
        finally:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def _suggest_next(self):
        """Suggest the next best unused ORB match near the least-constrained region."""
        if not hasattr(self, "_path_a"):
            return
        if not self._auto_pts_a:
            self._auto_detect()
            return
        manual_a = set(self._canvas_a.point_positions())
        unused = [
            (i, pa, pb)
            for i, (pa, pb) in enumerate(zip(self._auto_pts_a, self._auto_pts_b, strict=False))
            if pa not in manual_a
        ]
        if not unused:
            self._status.setText("No unused auto-matches left to suggest.")
            return

        def min_dist_to_existing(xy):
            ex = self._canvas_a.point_positions()
            if not ex:
                return float("inf")
            return min(math.hypot(xy[0] - p[0], xy[1] - p[1]) for p in ex)

        _, pa, pb = max(unused, key=lambda t: min_dist_to_existing(t[1]))
        self._canvas_a.add_point(pa[0], pa[1])
        self._canvas_b.add_point(pb[0], pb[1])
        self._status.setText(f"Suggested pair added at A=({pa[0]:.0f},{pa[1]:.0f}).")

    def _solve(self):
        pts_a_raw = self._canvas_a.point_positions()
        pts_b_raw = self._canvas_b.point_positions()
        n = min(len(pts_a_raw), len(pts_b_raw))
        if n < 4:
            self._status.setText(f"Need ≥4 matching pairs; have {n}.")
            return

        mode = self._solve_mode.currentText()
        pts_a = np.array(pts_a_raw[:n], dtype=np.float32)
        pts_b = np.array(pts_b_raw[:n], dtype=np.float32)
        if mode == "Auto + Manual" and self._auto_pts_a:
            ap = np.array(self._auto_pts_a, dtype=np.float32)
            bp = np.array(self._auto_pts_b, dtype=np.float32)
            pts_a = np.vstack([pts_a, ap])
            pts_b = np.vstack([pts_b, bp])

        cv_method = cv2.RANSAC if "RANSAC" in mode else 0
        H, mask = cv2.findHomography(pts_a, pts_b, cv_method, 3.0)
        if H is None:
            self._status.setText("Homography solve failed — collinear points?")
            return

        self._H = H
        n_in = int(mask.sum()) if mask is not None else n
        pts_h = cv2.perspectiveTransform(pts_a.reshape(-1, 1, 2), H).reshape(-1, 2)
        errs = np.linalg.norm(pts_h - pts_b, axis=1)
        mean_e = float(errs.mean())

        self._status.setText(
            f"H solved via {mode} — {n} pairs,  "
            f"{n_in} inliers,  mean reprojection error {mean_e:.2f} px."
        )
        self.homography_solved.emit(H, mean_e)

    @staticmethod
    def _run_orb(path_a: str, path_b: str, max_pts: int = 50) -> Tuple[list, list]:
        bgr_a = cv2.imread(path_a)
        bgr_b = cv2.imread(path_b)
        if bgr_a is None or bgr_b is None:
            return [], []

        def _scale(img):
            h, w = img.shape[:2]
            s = min(1.0, 1280 / w)
            if s < 1.0:
                img = cv2.resize(
                    img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA
                )
            return img, s

        a_s, sa = _scale(bgr_a)
        b_s, sb = _scale(bgr_b)

        orb = cv2.ORB_create(nfeatures=1000)  # pyrefly: ignore[missing-attribute]
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        ka, da = orb.detectAndCompute(cv2.cvtColor(a_s, cv2.COLOR_BGR2GRAY), None)
        kb, db = orb.detectAndCompute(cv2.cvtColor(b_s, cv2.COLOR_BGR2GRAY), None)
        if da is None or db is None or len(ka) < 4 or len(kb) < 4:
            return [], []

        matches = bf.knnMatch(da, db, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            return [], []

        src = np.array([ka[m.queryIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)
        dst = np.array([kb[m.trainIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is None:
            return [], []

        inlier_idx = [i for i, m in enumerate(good) if mask[i, 0]]
        inlier_idx = inlier_idx[:max_pts]

        pts_a = [
            (ka[good[i].queryIdx].pt[0] / sa, ka[good[i].queryIdx].pt[1] / sa)
            for i in inlier_idx
        ]
        pts_b = [
            (kb[good[i].trainIdx].pt[0] / sb, kb[good[i].trainIdx].pt[1] / sb)
            for i in inlier_idx
        ]
        return pts_a, pts_b

    def _on_a_added(self, idx, x, y):
        while len(self._canvas_b.point_positions()) <= idx:
            self._canvas_b.add_point(x, y)
        self._status.setText(
            f"Point {idx + 1} added to A.  Now click the matching location in Frame B."
        )

    def _on_b_added(self, idx, x, y):
        n_a = len(self._canvas_a.point_positions())
        if n_a <= idx:
            self._status.setText(
                f"Point {idx + 1} added to B.  "
                "Now click the matching location in Frame A."
            )

    def _on_a_moved(self, idx, x, y):
        pass

    def _on_b_moved(self, idx, x, y):
        pass

    def _on_a_removed(self, idx):
        if idx < len(self._canvas_b.point_positions()):
            self._canvas_b.remove_point(idx)

    def _on_b_removed(self, idx):
        if idx < len(self._canvas_a.point_positions()):
            self._canvas_a.remove_point(idx)
