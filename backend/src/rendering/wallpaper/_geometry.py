"""Pure-math geometry helpers for the wallpaper-mode aspect framer (#429).

A small, dependency-free building block for the aspect solver in
``_aspect_framer.py``: it owns only the coordinate geometry (aspect parsing,
bbox unions, and the minimal aspect-fixed window that contains a bbox) with
no image I/O, no OpenCV, and no inpainting logic.  The real framer (#429)
decides how to fill window regions that extend past the canvas (e.g. via
``BackgroundPlate`` + Tier-2 outpaint per the roadmap).

Rect convention throughout: canvas pixel boxes are ``(x0, y0, x1, y1)`` with
``x1``/``y1`` exclusive, matching the rest of the wallpaper stack.
"""

from __future__ import annotations

import math

# The three aspect targets locked in the wallpaper-mode roadmap (2026Q3).
# ``spec`` -> width / height, so "16:9" is wider than tall and "9:16" taller.
_ASPECT_RATIOS: dict[str, float] = {
    "16:9": 16.0 / 9.0,
    "9:16": 9.0 / 16.0,
    "21:9": 21.0 / 9.0,
}


def parse_aspect_ratio(spec: str) -> float:
    """Parse one of the roadmap's locked wallpaper aspect specs.

    Only ``"16:9"``, ``"9:16"``, and ``"21:9"`` are valid (width/height, the
    three targets in ``asp_wallpaper_mode_roadmap_2026q3.md``).  Returns the
    width/height ratio as a ``float`` (e.g. ``"16:9"`` -> ``16 / 9``).

    Parameters
    ----------
    spec:
        The raw aspect spec string.

    Returns
    -------
    float
        Width divided by height.

    Raises
    ------
    ValueError
        If ``spec`` is not one of the three locked specs.  Deliberately not a
        generic ``"W:H"`` parser — only the roadmap targets are supported.
    """
    if spec not in _ASPECT_RATIOS:
        raise ValueError(
            f"unsupported aspect ratio {spec!r}; only {sorted(_ASPECT_RATIOS)} are valid"
        )
    return _ASPECT_RATIOS[spec]


def union_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """Smallest ``(x0, y0, x1, y1)`` box containing every input box.

    Boxes are ``(x0, y0, x1, y1)`` in canvas pixel coordinates with ``x1``/``y1``
    exclusive.  Handles overlapping, disjoint, and negative-coordinate boxes.

    Parameters
    ----------
    boxes:
        Non-empty list of boxes to union.

    Returns
    -------
    tuple[int, int, int, int]
        The union ``(x0, y0, x1, y1)``.

    Raises
    ------
    ValueError
        If ``boxes`` is empty.
    """
    if not boxes:
        raise ValueError("boxes must be non-empty")
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def fit_window_containing_bbox(
    bbox: tuple[int, int, int, int],
    aspect: float,
    canvas_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Smallest aspect-fixed window fully containing ``bbox``, on its center.

    The window has width/height == ``aspect``, is centered on ``bbox``'s own
    center, and is just large enough to contain ``bbox`` in both dimensions.
    ``canvas_size`` is ``(H, W)`` and is informational only: the returned
    window is NOT clamped to it — it may have negative coordinates or extend
    past ``H``/``W`` when the bbox demands it.  How to fill that overflow is
    the aspect framer's job (#429).

    The integer result is rounded outward (``floor`` on the low edges, ``ceil``
    on the high edges) so the returned window always contains ``bbox`` even
    when the exact aspect-fixed window lands on fractional coordinates.

    Parameters
    ----------
    bbox:
        Box to contain, ``(x0, y0, x1, y1)`` with ``x1``/``y1`` exclusive.
    aspect:
        Target width/height ratio; must be positive.
    canvas_size:
        ``(H, W)`` of the canvas.  Used only as the overflow reference; the
        result is never clipped to it.

    Returns
    -------
    tuple[int, int, int, int]
        The window ``(x0, y0, x1, y1)``, possibly outside ``canvas_size``.

    Raises
    ------
    ValueError
        If ``aspect`` is not positive.
    """
    if aspect <= 0:
        raise ValueError(f"aspect must be positive, got {aspect}")
    x0, y0, x1, y1 = bbox
    bbox_w = x1 - x0
    bbox_h = y1 - y0
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    h = max(bbox_h, bbox_w / aspect)
    w = aspect * h
    return (
        math.floor(cx - w / 2.0),
        math.floor(cy - h / 2.0),
        math.ceil(cx + w / 2.0),
        math.ceil(cy + h / 2.0),
    )


__all__ = ["parse_aspect_ratio", "union_bbox", "fit_window_containing_bbox"]