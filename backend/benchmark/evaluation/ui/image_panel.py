"""ImagePanel: a QGraphicsView deep-zoom/pan display for one comparator image,
with a working pixel-value overlay and interactive bbox/point annotation.

Two things the old ``panel_base.py`` got wrong, both fixed here:

**Zoom was discontinuous** (issue #123 defect 6). ``fit_to_view()`` set
``_zoom = 1.0`` while the actual transform was the fit scale, so the first
wheel notch called ``set_zoom(1.15)``, reset the transform and jumped straight
to 1.15x *native pixels* — a violent leap to near-100% on a 1700px panorama.
Zoom is now tracked as a factor *relative to the fitted view*, with the real
device scale being ``fit_scale * zoom``. That also makes cross-panel locking
meaningful: comparators have genuinely different canvas sizes (1703x1704 vs
1917x2050 vs 2972x2197 on test01), so locking absolute pixel scale would show
each one a different fraction of its content. Locking the *relative* factor
plus a normalized viewport centre shows the same part of each image at the same
apparent magnification, and syncs pan as well as zoom — the old sync passed
only the scale factor, so panned panels drifted apart.

**Pixel Value Mode did nothing** (defect 1). ``set_display_mode()`` stored a
string and repainted, and nothing ever read it — the ``PIXEL_GRID_ZOOM_THRESHOLD``
constant was imported and unused. It is now implemented, in ``pixel_overlay.py``:
a per-pixel grid once pixels are big enough to separate, and numeric RGB triples
once they are big enough to hold text, bounded so a large visible region can't
try to paint a million labels.
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QSizePolicy,
)

from ..constants.user_interface import (
    COL_BBOX,
    COL_BBOX_ACTIVE,
    COL_POINT,
    DISPLAY_PIXEL,
    DISPLAY_RAW,
    MODE_BBOX,
    MODE_NAVIGATE,
    MODE_POINT,
    MODE_PROBE,
    PIXEL_GRID_ZOOM_THRESHOLD,
    ZOOM_MAX,
    ZOOM_MIN,
    ZOOM_STEP,
)
from ..other.schema import BoundingBox
from . import pixel_overlay

# ---------------------------------------------------------------------------
# Debug instrumentation — mirrors _DBG_INSPECTOR in main_window.py.
# Set to False to silence all [DBG-PANEL] output.
# ---------------------------------------------------------------------------
_DBG_PANEL: bool = False


def _dbg_panel(key: str, *parts) -> None:  # noqa: D103
    if _DBG_PANEL:
        print(f"[DBG-PANEL][{key}]", *parts, flush=True)


def bgr_to_qimage(img: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()


class ImagePanel(QGraphicsView):
    # (zoom factor relative to fit, normalized viewport centre x, y)
    viewChanged = Signal(float, float, float)
    bboxDrawn = Signal(object)  # dict: normalized x/y/w/h (label/defect added by caller)
    pointPicked = Signal(float, float)  # normalized x, y within this panel's image
    regionPicked = Signal(float, float, float, float)  # normalized x, y, w, h — a link endpoint drawn as a box
    pixelHovered = Signal(int, int, object)  # pixel x, y, BGR tuple (or None off-image)
    pixelPinned = Signal(int, int, object)
    focusRequested = Signal()

    def __init__(self, key: str, title: str, parent=None):
        super().__init__(parent)
        self.key = key
        self.title = title
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._image_bgr: np.ndarray | None = None
        self._fit_scale = 1.0
        self._zoom = 1.0  # multiple of the fitted view, not of native pixels
        self._mode = MODE_NAVIGATE
        self._display_mode = DISPLAY_RAW
        self._rubber_origin: QPointF | None = None
        self._rubber_item: QGraphicsRectItem | None = None
        self._link_origin: QPointF | None = None
        self._link_rubber: QGraphicsRectItem | None = None
        self._overlay_items: list[QGraphicsRectItem] = []
        self._syncing = False
        self._pinned: tuple[int, int] | None = None
        self._hover_view_pos: QPoint | None = None

    # -- image loading -------------------------------------------------------

    def set_image(self, img: np.ndarray | None) -> None:
        self._image_bgr = img
        self._scene.clear()
        self._pixmap_item = None
        self._overlay_items = []
        self._pinned = None
        if img is None:
            self._scene.setSceneRect(QRectF())
            self.viewport().update()
            return
        pix = QPixmap.fromImage(bgr_to_qimage(img))
        self._pixmap_item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self.fit_to_view()

    def has_image(self) -> bool:
        return self._pixmap_item is not None

    def image_size(self) -> tuple[int, int] | None:
        if self._image_bgr is None:
            return None
        h, w = self._image_bgr.shape[:2]
        return w, h

    def current_image(self) -> np.ndarray | None:
        return self._image_bgr

    # -- zoom / pan ----------------------------------------------------------

    def _compute_fit_scale(self) -> float:
        size = self.image_size()
        if size is None:
            return 1.0
        w, h = size
        view = self.viewport().size()
        if not w or not h or view.width() <= 0 or view.height() <= 0:
            return 1.0
        return min(view.width() / w, view.height() / h)

    def fit_to_view(self, emit: bool = True) -> None:
        if not self.has_image():
            return
        prev_fit = self._fit_scale
        self._fit_scale = self._compute_fit_scale()
        self._zoom = 1.0
        self._apply_transform()
        center = self._pixmap_item.boundingRect().center()
        self.centerOn(center)
        _dbg_panel(
            self.key,
            f"fit_to_view: viewport={self.viewport().width()}x{self.viewport().height()}"
            f" | prev_fit_scale={prev_fit:.4f} -> fit_scale={self._fit_scale:.4f}"
            f" | scene_center=({center.x():.1f},{center.y():.1f})"
        )
        if emit:
            self._emit_view_changed()

    def zoom(self) -> float:
        """Current magnification as a multiple of the fitted view."""
        return self._zoom

    def native_scale(self) -> float:
        """Current magnification in native image pixels (1.0 = 100%)."""
        return self._fit_scale * self._zoom

    def _apply_transform(self) -> None:
        scale = self.native_scale()
        self.setTransform(QTransform.fromScale(scale, scale))
        self.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            scale < PIXEL_GRID_ZOOM_THRESHOLD,
        )

    def set_zoom(self, factor: float, anchor: QPointF | None = None, emit: bool = True) -> None:
        if not self.has_image():
            return
        factor = max(ZOOM_MIN, min(ZOOM_MAX, factor))
        if anchor is None:
            anchor = self.mapToScene(self.viewport().rect().center())
            viewport_pos = self.viewport().rect().center()
        else:
            viewport_pos = self.mapFromScene(anchor)
        self._zoom = factor
        self._apply_transform()
        delta = self.mapToScene(viewport_pos) - anchor
        center = self.mapToScene(self.viewport().rect().center()) - delta
        self.centerOn(center)
        if emit:
            self._emit_view_changed()

    def center_norm(self) -> tuple[float, float]:
        size = self.image_size()
        if size is None:
            return 0.5, 0.5
        w, h = size
        center = self.mapToScene(self.viewport().rect().center())
        return center.x() / w, center.y() / h

    def _emit_view_changed(self) -> None:
        if self._syncing:
            return
        cx, cy = self.center_norm()
        self.viewChanged.emit(self._zoom, cx, cy)

    def apply_external_view(self, factor: float, cx: float, cy: float) -> None:
        if not self.has_image():
            return
        self._syncing = True
        try:
            self._zoom = max(ZOOM_MIN, min(ZOOM_MAX, factor))
            self._apply_transform()
            w, h = self.image_size()
            self.centerOn(QPointF(cx * w, cy * h))
        finally:
            self._syncing = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self.has_image():
            return
        step = ZOOM_STEP if event.angleDelta().y() > 0 else (1.0 / ZOOM_STEP)
        self.set_zoom(self._zoom * step, anchor=self.mapToScene(event.position().toPoint()))
        event.accept()

    def sizeHint(self):  # noqa: D102 - Qt override
        from PySide6.QtCore import QSize
        return QSize(400, 300)

    def minimumSizeHint(self):  # noqa: D102 - Qt override
        from PySide6.QtCore import QSize
        return QSize(200, 150)

    def resizeEvent(self, event) -> None:  # noqa: D102 - Qt override
        super().resizeEvent(event)
        if not self.has_image():
            return
        previous_fit = self._fit_scale
        self._fit_scale = self._compute_fit_scale()
        _dbg_panel(
            self.key,
            f"resizeEvent: old={event.oldSize().width()}x{event.oldSize().height()}"
            f" -> new={self.width()}x{self.height()}"
            f" | viewport={self.viewport().width()}x{self.viewport().height()}"
            f" | fit_scale: {previous_fit:.4f} -> {self._fit_scale:.4f}"
            f" | zoom={self._zoom:.4f}"
            f" | sizePolicy={self.sizePolicy().horizontalPolicy().name}/{self.sizePolicy().verticalPolicy().name}"
        )
        if previous_fit != self._fit_scale:
            self._apply_transform()
            if abs(self._zoom - 1.0) < 1e-4:
                self.centerOn(self._pixmap_item.boundingRect().center())
            else:
                w, h = self.image_size()
                cx, cy = self.center_norm()
                self.centerOn(QPointF(cx * w, cy * h))

    # -- modes ---------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if mode in (MODE_NAVIGATE, MODE_PROBE)
            else QGraphicsView.DragMode.NoDrag
        )
        cursors = {
            MODE_BBOX: Qt.CursorShape.CrossCursor,
            MODE_POINT: Qt.CursorShape.PointingHandCursor,
            MODE_PROBE: Qt.CursorShape.CrossCursor,
        }
        self.viewport().setCursor(cursors.get(mode, Qt.CursorShape.ArrowCursor))

    def set_display_mode(self, mode: str) -> None:
        self._display_mode = mode
        if mode != DISPLAY_PIXEL:
            self._hover_view_pos = None
        self.viewport().update()

    def display_mode(self) -> str:
        return self._display_mode

    # -- mouse ---------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.focusRequested.emit()
        if event.button() == Qt.MouseButton.LeftButton and self.has_image():
            if self._mode == MODE_BBOX:
                self._rubber_origin = self.mapToScene(event.position().toPoint())
                self._rubber_item = QGraphicsRectItem(QRectF(self._rubber_origin, self._rubber_origin))
                self._rubber_item.setPen(QPen(QColor(COL_BBOX_ACTIVE), 0))
                self._rubber_item.setBrush(QBrush(QColor(255, 107, 107, 45)))
                self._scene.addItem(self._rubber_item)
                return
            if self._mode == MODE_POINT:
                self._link_origin = self.mapToScene(event.position().toPoint())
                self._link_rubber = QGraphicsRectItem(QRectF(self._link_origin, self._link_origin))
                self._link_rubber.setPen(QPen(QColor(COL_POINT), 0, Qt.PenStyle.DotLine))
                self._scene.addItem(self._link_rubber)
                return
            if self._mode == MODE_PROBE:
                self._pin_pixel(event.position().toPoint())
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._mode == MODE_BBOX and self._rubber_item is not None and self._rubber_origin is not None:
            rect = QRectF(self._rubber_origin, self.mapToScene(event.position().toPoint())).normalized()
            self._rubber_item.setRect(rect)
        elif self._mode == MODE_POINT and self._link_rubber is not None and self._link_origin is not None:
            rect = QRectF(self._link_origin, self.mapToScene(event.position().toPoint())).normalized()
            self._link_rubber.setRect(rect)
        else:
            pos = event.position().toPoint()
            self._emit_pixel_hover(pos)
            if self._display_mode == DISPLAY_PIXEL:
                self._hover_view_pos = pos
                self.viewport().update()  # redraw the magnifier at the new position
        super().mouseMoveEvent(event)
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag and (
            event.buttons() & Qt.MouseButton.LeftButton
        ):
            self._emit_view_changed()  # keep locked panels following a pan

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._mode == MODE_BBOX and self._rubber_item is not None:
            rect = self._rubber_item.rect()
            self._scene.removeItem(self._rubber_item)
            self._rubber_item = None
            self._rubber_origin = None
            # 3 *scene* px was the old threshold, which at a fitted view of a
            # 1700px panorama is a sub-pixel drag; require a region big enough
            # to be a real defect instead.
            if rect.width() >= 4 and rect.height() >= 4:
                self._finish_bbox(rect)
            return
        if self._mode == MODE_POINT and self._link_rubber is not None:
            rect = self._link_rubber.rect()
            origin = self._link_origin
            self._scene.removeItem(self._link_rubber)
            self._link_rubber = None
            self._link_origin = None
            # Below the drag threshold: treat it as a click at the press point,
            # not the (essentially noise-sized) rectangle a shaky click drags
            # out — same 4px floor MODE_BBOX uses to call a drag a real region.
            if rect.width() >= 4 and rect.height() >= 4:
                self._finish_region(rect)
            elif origin is not None:
                self._finish_point(origin)
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: D102 - Qt override
        self.pixelHovered.emit(-1, -1, None)
        if self._hover_view_pos is not None:
            self._hover_view_pos = None
            self.viewport().update()  # drop the magnifier once the cursor leaves
        super().leaveEvent(event)

    # -- annotation ----------------------------------------------------------

    def _finish_bbox(self, rect: QRectF) -> None:
        size = self.image_size()
        if size is None:
            return
        w, h = size
        self.bboxDrawn.emit({
            "x": max(0.0, rect.x() / w),
            "y": max(0.0, rect.y() / h),
            "w": min(1.0, rect.width() / w),
            "h": min(1.0, rect.height() / h),
        })

    def _finish_point(self, pt: QPointF) -> None:
        size = self.image_size()
        if size is None:
            return
        w, h = size
        if not (0 <= pt.x() <= w and 0 <= pt.y() <= h):
            return
        marker = QGraphicsRectItem(pt.x() - 3, pt.y() - 3, 6, 6)
        marker.setPen(QPen(QColor(COL_POINT), 0))
        self._scene.addItem(marker)
        self._overlay_items.append(marker)
        self.pointPicked.emit(pt.x() / w, pt.y() / h)

    def _finish_region(self, rect: QRectF) -> None:
        """A dragged (rather than clicked) link endpoint — a region instead
        of a point, e.g. "this whole torn area" rather than one pixel of it."""
        size = self.image_size()
        if size is None:
            return
        w, h = size
        marker = QGraphicsRectItem(rect)
        marker.setPen(QPen(QColor(COL_POINT), 0, Qt.PenStyle.DashLine))
        self._scene.addItem(marker)
        self._overlay_items.append(marker)
        self.regionPicked.emit(
            max(0.0, rect.x() / w), max(0.0, rect.y() / h),
            min(1.0, rect.width() / w), min(1.0, rect.height() / h),
        )

    def _add_bbox_item(self, rect: QRectF, color: str, label: str = "") -> None:
        item = QGraphicsRectItem(rect)
        # Cosmetic pen (width 0): a fixed scene width would render as a
        # hairline when fitted and a fat slab at 30x zoom.
        item.setPen(QPen(QColor(color), 0))
        item.setToolTip(label)
        self._scene.addItem(item)
        self._overlay_items.append(item)

    def restore_bboxes(self, bboxes: Iterable[BoundingBox]) -> None:
        size = self.image_size()
        if size is None:
            return
        w, h = size
        for b in bboxes:
            if b.image != self.key:
                continue
            self._add_bbox_item(
                QRectF(b.x * w, b.y * h, b.w * w, b.h * h),
                COL_BBOX,
                b.label or b.defect,
            )

    def clear_overlays(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items = []

    def scene_point_to_view(self, x_norm: float, y_norm: float) -> QPointF | None:
        size = self.image_size()
        if size is None:
            return None
        w, h = size
        return self.mapFromScene(QPointF(x_norm * w, y_norm * h))

    # -- pixel inspection ----------------------------------------------------

    def pixel_at(self, view_pos) -> tuple[int, int, tuple[int, int, int]] | None:
        if self._image_bgr is None:
            return None
        scene_pt = self.mapToScene(view_pos)
        x, y = int(np.floor(scene_pt.x())), int(np.floor(scene_pt.y()))
        h, w = self._image_bgr.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None
        return x, y, tuple(int(v) for v in self._image_bgr[y, x])

    def _emit_pixel_hover(self, view_pos) -> None:
        found = self.pixel_at(view_pos)
        if found is None:
            self.pixelHovered.emit(-1, -1, None)
        else:
            x, y, bgr = found
            self.pixelHovered.emit(x, y, bgr)

    def _pin_pixel(self, view_pos) -> None:
        found = self.pixel_at(view_pos)
        if found is None:
            return
        x, y, bgr = found
        self._pinned = (x, y)
        self.pixelPinned.emit(x, y, bgr)
        self.viewport().update()

    def pixel_region(
        self, x_norm: float, y_norm: float, w_norm: float, h_norm: float
    ) -> np.ndarray | None:
        """BGR crop for a normalized bbox — feeds the numeric pixel dump and
        every comparison/visualization tool's "selected region" input."""
        if self._image_bgr is None:
            return None
        h, w = self._image_bgr.shape[:2]
        x0, y0 = max(0, int(x_norm * w)), max(0, int(y_norm * h))
        x1, y1 = min(w, int((x_norm + w_norm) * w)), min(h, int((y_norm + h_norm) * h))
        if x1 <= x0 or y1 <= y0:
            return None
        return self._image_bgr[y0:y1, x0:x1].copy()

    def visible_region_norm(self) -> tuple[float, float, float, float] | None:
        """The currently visible part of the image as a normalized rect — lets
        a tool act on "what I'm looking at" without drawing a box."""
        size = self.image_size()
        if size is None:
            return None
        w, h = size
        rect = self.mapToScene(self.viewport().rect()).boundingRect()
        x0 = max(0.0, rect.x() / w)
        y0 = max(0.0, rect.y() / h)
        x1 = min(1.0, (rect.x() + rect.width()) / w)
        y1 = min(1.0, (rect.y() + rect.height()) / h)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1 - x0, y1 - y0

    # -- pixel-value overlay (Pixel Value Mode) ------------------------------

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: D102 - Qt override
        super().drawForeground(painter, rect)
        pixel_overlay.draw(self, painter, rect)
