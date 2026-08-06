"""Small standalone geometry utilities used by foreground registration."""

from __future__ import annotations

import cv2
import numpy as np


def _seam_taper(
    h: int,
    w: int,
    seam_pos: int,
    taper_px: float,
    axis: int = 0,
) -> np.ndarray:
    """
    (h, w) float32 taper weight, 1.0 at ``seam_pos`` decaying linearly to 0.0
    at ``±taper_px``.  ``axis=0`` → taper along rows (vertical scroll seam);
    ``axis=1`` → taper along columns (horizontal scroll seam).
    """
    if axis == 0:
        coord = np.arange(h, dtype=np.float32)[:, None]
        dist = np.abs(coord - float(seam_pos))
        w_line = np.clip(1.0 - dist / max(taper_px, 1.0), 0.0, 1.0)  # (h,1)
        return np.broadcast_to(w_line, (h, w)).copy()
    else:
        coord = np.arange(w, dtype=np.float32)[None, :]
        dist = np.abs(coord - float(seam_pos))
        w_line = np.clip(1.0 - dist / max(taper_px, 1.0), 0.0, 1.0)  # (1,w)
        return np.broadcast_to(w_line, (h, w)).copy()


def _remap_by_displacement(img: np.ndarray, disp: np.ndarray) -> np.ndarray:
    """
    Resample ``img`` at position ``(x + disp_x, y + disp_y)`` per pixel.

    ``disp`` is an (H, W, 2) field (dx, dy).  Pixels whose source maps outside
    the image retain their original value (BORDER_TRANSPARENT fallback) to
    avoid the BORDER_REPLICATE edge-smear artefact that creates corrupted
    corner regions in the composite when the warp shifts pixels off-canvas.
    """
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    map_x = grid_x + disp[:, :, 0]
    map_y = grid_y + disp[:, :, 1]
    remapped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # Restore original pixels wherever the source coordinate mapped outside
    # the valid frame bounds — those remap to black (0) which is also content
    # in some scenes, so we use the source validity mask instead.
    out_of_bounds = (map_x < 0) | (map_x >= w) | (map_y < 0) | (map_y >= h)
    if out_of_bounds.any():
        out3 = np.stack([out_of_bounds] * 3, axis=2) if img.ndim == 3 else out_of_bounds
        remapped[out3] = img[out3]
    return remapped


__all__ = ["_seam_taper", "_remap_by_displacement"]
