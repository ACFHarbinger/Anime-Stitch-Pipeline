"""Pixel Value Mode's foreground painting for ``ImagePanel``.

This is the feature that was a complete no-op before the rebuild (issue #123
defect 1): ``set_display_mode()`` stored a string that nothing ever read, and
``PIXEL_GRID_ZOOM_THRESHOLD`` was imported and unused.

Free functions over the panel rather than methods on it, so ``image_panel.py``
stays within the repo's 500-LoC file budget (§5.17 / issues #121-#122) and the
painting logic — the part with the subtle coordinate-space rules below — can be
read on its own.

Two rules that are easy to get wrong here:

- **Labels are drawn in device coordinates, with the world transform reset.**
  Sizing a font in *scene* units means its device size gets multiplied by the
  view scale, and past ~30x native that asks FreeType for enormous glyphs, which
  fails outright (``render glyph failed err=62``) and takes the process down.
  That was a real crash, found by the offscreen tests.
- **Both thresholds are in native scale, not zoom factor.** Zoom is relative to
  the fitted view, so the same zoom factor is a wildly different pixel size on a
  1703x1704 panorama than on a 300px crop.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from ..constants.user_interface import (
    COL_ACCENT,
    COL_BORDER,
    COL_POINT,
    COL_SURFACE_HI,
    COL_TEXT,
    DISPLAY_PIXEL,
    PIXEL_GRID_ZOOM_THRESHOLD,
    PIXEL_MAGNIFIER_CELL,
    PIXEL_MAGNIFIER_MARGIN,
    PIXEL_MAGNIFIER_RADIUS,
    PIXEL_TEXT_MAX_CELLS,
    PIXEL_TEXT_ZOOM_THRESHOLD,
)

# Luma above which a cell is light enough to need dark text.
_DARK_TEXT_LUMA = 140


def draw(panel, painter: QPainter, rect: QRectF) -> None:
    """Paint the pinned-probe marker, the always-on hover magnifier, and — once
    already zoomed in far enough — the full in-image pixel grid too."""
    if panel._pinned is not None:
        draw_pin(panel, painter)
    if panel._display_mode != DISPLAY_PIXEL or panel._image_bgr is None:
        return

    draw_magnifier(panel, painter)

    scale = panel.native_scale()
    if scale < PIXEL_GRID_ZOOM_THRESHOLD:
        return

    bounds = _visible_cells(panel, rect)
    if bounds is None:
        return
    x0, y0, x1, y1 = bounds

    painter.save()
    painter.setPen(QPen(QColor(255, 255, 255, 45), 0))
    for x in range(x0, x1 + 1):
        painter.drawLine(QPointF(x, y0), QPointF(x, y1))
    for y in range(y0, y1 + 1):
        painter.drawLine(QPointF(x0, y), QPointF(x1, y))

    cells = (x1 - x0) * (y1 - y0)
    if scale >= PIXEL_TEXT_ZOOM_THRESHOLD and cells <= PIXEL_TEXT_MAX_CELLS:
        draw_pixel_values(panel, painter, x0, y0, x1, y1, scale)
    painter.restore()


def _hovered_pixel(panel) -> tuple[int, int] | None:
    view_pos = getattr(panel, "_hover_view_pos", None)
    if view_pos is None:
        return None
    found = panel.pixel_at(view_pos)
    if found is None:
        return None
    x, y, _bgr = found
    return x, y


def draw_magnifier(panel, painter: QPainter) -> None:
    """A fixed-size, fixed-position (bottom-right of the viewport) inset
    showing the actual RGB values around the cursor, read directly from the
    source array — so it works at any zoom, including the fit-to-view zoom a
    test opens at, where the in-image grid below is far too fine-grained to
    draw at all.
    """
    hovered = _hovered_pixel(panel)
    if hovered is None:
        return
    hx, hy = hovered
    h, w = panel._image_bgr.shape[:2]
    radius = PIXEL_MAGNIFIER_RADIUS
    x0, y0 = max(0, hx - radius), max(0, hy - radius)
    x1, y1 = min(w, hx + radius + 1), min(h, hy + radius + 1)
    cols, rows = x1 - x0, y1 - y0
    if cols <= 0 or rows <= 0:
        return

    cell = PIXEL_MAGNIFIER_CELL
    header_h = cell + 2
    box_w, box_h = cols * cell, rows * cell + header_h
    viewport_rect = panel.viewport().rect()
    origin = QPoint(
        viewport_rect.width() - box_w - PIXEL_MAGNIFIER_MARGIN,
        viewport_rect.height() - box_h - PIXEL_MAGNIFIER_MARGIN,
    )

    painter.save()
    painter.resetTransform()
    painter.setPen(QPen(QColor(COL_BORDER), 1))
    painter.setBrush(QColor(COL_SURFACE_HI))
    painter.drawRect(QRect(origin, origin + QPoint(box_w, box_h)))

    header_font = QFont("monospace")
    header_font.setPixelSize(max(9, cell - 3))
    painter.setFont(header_font)
    painter.setPen(QColor(COL_TEXT))
    painter.drawText(
        QRect(origin.x(), origin.y(), box_w, header_h),
        Qt.AlignmentFlag.AlignCenter,
        f"({hx}, {hy})",
    )

    region = panel._image_bgr[y0:y1, x0:x1]
    luma = region[..., 0] * 0.114 + region[..., 1] * 0.587 + region[..., 2] * 0.299
    value_font = QFont("monospace")
    value_font.setPixelSize(max(6, int(cell / 3.4)))
    painter.setFont(value_font)
    grid_top = origin.y() + header_h
    for j in range(rows):
        for i in range(cols):
            b, g, r = (int(v) for v in region[j, i])
            cell_x, cell_y = origin.x() + i * cell, grid_top + j * cell
            painter.fillRect(QRect(cell_x, cell_y, cell, cell), QColor(r, g, b))
            painter.setPen(QColor(0, 0, 0) if luma[j, i] > _DARK_TEXT_LUMA else QColor(255, 255, 255))
            painter.drawText(
                QRect(cell_x, cell_y, cell, cell),
                Qt.AlignmentFlag.AlignCenter,
                f"{r}\n{g}\n{b}",
            )
    # Highlight the exact hovered cell so it's clear which one is under the cursor.
    painter.setPen(QPen(QColor(COL_ACCENT), 2))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(QRect(origin.x() + (hx - x0) * cell, grid_top + (hy - y0) * cell, cell, cell))
    painter.restore()


def _visible_cells(panel, rect: QRectF):
    """The exposed rect clipped to the image, as integer pixel bounds."""
    h, w = panel._image_bgr.shape[:2]
    x0 = max(0, int(np.floor(rect.left())))
    y0 = max(0, int(np.floor(rect.top())))
    x1 = min(w, int(np.ceil(rect.right())) + 1)
    y1 = min(h, int(np.ceil(rect.bottom())) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def draw_pixel_values(panel, painter: QPainter, x0: int, y0: int, x1: int, y1: int, scale: float) -> None:
    """Draw each visible pixel's RGB triple inside its grid cell.

    Device coordinates, world transform reset — see the module docstring for why
    a scene-space font is a crash rather than a cosmetic problem.
    """
    painter.save()
    painter.resetTransform()
    font = QFont("monospace")
    # Three stacked lines have to fit in a cell `scale` device px tall.
    font.setPixelSize(max(5, min(28, int(scale / 4.5))))
    painter.setFont(font)
    region = panel._image_bgr[y0:y1, x0:x1]
    luma = region[..., 0] * 0.114 + region[..., 1] * 0.587 + region[..., 2] * 0.299
    for j in range(y1 - y0):
        for i in range(x1 - x0):
            b, g, r = (int(v) for v in region[j, i])
            top_left = panel.mapFromScene(QPointF(x0 + i, y0 + j))
            bottom_right = panel.mapFromScene(QPointF(x0 + i + 1, y0 + j + 1))
            # Flip the label against the cell's own brightness so the text stays
            # readable over both flat white and flat black cels.
            painter.setPen(QColor(0, 0, 0) if luma[j, i] > _DARK_TEXT_LUMA else QColor(255, 255, 255))
            painter.drawText(
                QRectF(top_left, bottom_right),
                Qt.AlignmentFlag.AlignCenter,
                f"{r}\n{g}\n{b}",
            )
    painter.restore()


def draw_pin(panel, painter: QPainter) -> None:
    """Crosshair on the pinned probe pixel, sized in scene units so its arms
    stay a constant on-screen length as the zoom changes."""
    x, y = panel._pinned
    painter.save()
    painter.setPen(QPen(QColor(COL_POINT), 0))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(QRectF(x, y, 1, 1))
    span = 6.0 / max(panel.native_scale(), 1e-6)
    cx, cy = x + 0.5, y + 0.5
    painter.drawLine(QPointF(cx - span, cy), QPointF(cx - 1, cy))
    painter.drawLine(QPointF(cx + 1, cy), QPointF(cx + span, cy))
    painter.drawLine(QPointF(cx, cy - span), QPointF(cx, cy - 1))
    painter.drawLine(QPointF(cx, cy + 1), QPointF(cx, cy + span))
    painter.restore()
