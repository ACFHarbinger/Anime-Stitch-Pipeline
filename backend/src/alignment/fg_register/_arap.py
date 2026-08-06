"""ARAP (As-Rigid-As-Possible) Push + Regularise (Sýkora 2009).

Push: per-cell SAD block matching gives each grid cell an independent
appearance-optimal displacement — critical for flat cel-shaded regions
where gradient-based optical flow (RAFT, DIS) is ambiguous.
Regularise: smooths the per-cell translations globally (with an optional
LSD line-collinearity constraint) so adjacent cells don't move in
contradictory directions, preventing line-art strokes from bending.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import base as _batch_fgreg
    if (
        getattr(_batch_fgreg, "__file__", None) is None
        or not hasattr(_batch_fgreg, "fg_register")
    ):
        raise ImportError("compiled base.fg_register extension not available")
    _BATCH_FGREG = True
except ImportError:
    _batch_fgreg = None
    _BATCH_FGREG = False

# ARAP Push phase (Sýkora 2009 block-matching Push before the Regularise step).
# Enable: ASP_ARAP_PUSH=1 (default ON).
# Disable: ASP_ARAP_PUSH=0 for A/B comparison vs pure Regularise.
_ARAP_PUSH_ENABLED = os.environ.get("ASP_ARAP_PUSH", "1") != "0"


def _arap_push(
    img_a: np.ndarray,
    img_b: np.ndarray,
    fg_mask: np.ndarray,
    initial_flow: np.ndarray,
    cell_size: int = 16,
    search_range: int = 24,
    min_fg_frac: float = 0.25,
    improvement_threshold: float = 0.15,
) -> np.ndarray:
    """
    ARAP Push phase (Sýkora 2009) — per-cell block matching to find better
    rigid translations before the Regularise phase smooths them.

    The Push phase decouples neighbouring cells so each can independently jump
    to its local appearance optimum via SAD (sum of absolute differences) block
    matching.  Unlike gradient-based optical flow (RAFT, DIS), block matching
    does not require local intensity gradients — it finds the best-matching
    displacement even in large flat cel-shaded regions where the aperture problem
    renders gradient methods ambiguous.

    After Push, the per-cell translations are passed to :func:`_arap_regularise`
    for global consistency (no two adjacent cells should move in wildly different
    directions).  The Push–Regularise cycle is the full Sýkora ARAP algorithm;
    the previous ASP implementation omitted the Push phase.

    Parameters
    ----------
    img_a, img_b : (H, W[, 3]) uint8
        The two canvas-aligned frame crops (seam band).
    fg_mask : (H, W) bool/uint8
        True / > 127 = foreground character pixels.  Only fg cells are pushed;
        background cells keep the initial flow.
    initial_flow : (H, W, 2) float32
        Initial per-pixel flow from the dense flow stage (RAFT/DIS).  Used both
        as the centre of the per-cell search window and as the fallback when
        block matching finds no improvement.
    cell_size : int
        Grid cell size (px).  Smaller cells = finer-grained push (more accurate)
        but slower.  Default 16 matches the ARAP regularise grid.
    search_range : int
        Half-width of the per-cell SAD search window (px).  The block matching
        looks in a (2×search_range+1)² area centred on the initial flow estimate.
    min_fg_frac : float
        Minimum fraction of a cell's pixels that must be foreground for the Push
        to run on that cell.  Background-dominated cells keep the initial flow.
    improvement_threshold : float
        Minimum fractional SAD reduction required to accept the Push displacement
        over the initial flow's displacement.  Prevents noise-driven switches.

    Returns
    -------
    (H, W, 2) float32 — updated flow with per-cell block-matched translations
    for fg cells where a clear improvement was found; otherwise identical to
    initial_flow.
    """
    H, W = initial_flow.shape[:2]
    out = initial_flow.copy()

    # Convert to grayscale for appearance-based matching
    if img_a.ndim == 3:
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray_a = img_a.astype(np.float32)
        gray_b = img_b.astype(np.float32)

    fg = (fg_mask > 127) if fg_mask.dtype != bool else fg_mask
    fg_float = fg.astype(np.float32)

    n_cells_y = max(1, H // cell_size)
    n_cells_x = max(1, W // cell_size)
    min_fg_pixels = cell_size * cell_size * min_fg_frac

    for ci in range(n_cells_y):
        y0 = ci * cell_size
        y1 = min(H, y0 + cell_size)
        for cj in range(n_cells_x):
            x0 = cj * cell_size
            x1 = min(W, x0 + cell_size)

            fg_in_cell = int(fg_float[y0:y1, x0:x1].sum())
            if fg_in_cell < min_fg_pixels:
                continue  # not enough character content — keep initial flow

            # Per-cell initial flow estimate (robust median)
            cell_flow = initial_flow[y0:y1, x0:x1]
            init_dx = float(np.median(cell_flow[:, :, 0]))
            init_dy = float(np.median(cell_flow[:, :, 1]))

            # Search window in img_b centred at the initial flow estimate
            sy0 = max(0, y0 + round(init_dy) - search_range)
            sy1 = min(H, y1 + round(init_dy) + search_range)
            sx0 = max(0, x0 + round(init_dx) - search_range)
            sx1 = min(W, x1 + round(init_dx) + search_range)

            template = gray_a[y0:y1, x0:x1]
            search = gray_b[sy0:sy1, sx0:sx1]

            th, tw = template.shape
            if search.shape[0] < th or search.shape[1] < tw:
                continue  # search window too small (frame edge)

            # Compute baseline SAD at the initial flow location
            base_by = y0 + round(init_dy)
            base_bx = x0 + round(init_dx)
            if (0 <= base_by < H - th + 1) and (0 <= base_bx < W - tw + 1):
                base_patch = gray_b[base_by : base_by + th, base_bx : base_bx + tw]
                if base_patch.shape == template.shape:
                    base_sad = float(np.abs(template - base_patch).mean())
                else:
                    base_sad = float("inf")
            else:
                base_sad = float("inf")

            # SAD block matching in search window
            result = cv2.matchTemplate(search, template, cv2.TM_SQDIFF)
            _, _, min_loc, _ = cv2.minMaxLoc(result)
            best_sad = float(result[min_loc[1], min_loc[0]]) / (th * tw)

            # Accept only if Push found a genuinely better match
            if base_sad == float("inf") or best_sad < base_sad * (
                1.0 - improvement_threshold
            ):
                # Convert match location back to absolute displacement
                best_dy = float(sy0 + min_loc[1]) - float(y0)
                best_dx = float(sx0 + min_loc[0]) - float(x0)
                # Update the flow in this cell (only for fg pixels)
                cell_fg = fg[y0:y1, x0:x1]
                out[y0:y1, x0:x1, 0] = np.where(cell_fg, best_dx, out[y0:y1, x0:x1, 0])
                out[y0:y1, x0:x1, 1] = np.where(cell_fg, best_dy, out[y0:y1, x0:x1, 1])

    return out


def _arap_regularise(  # noqa: C901
    flow: np.ndarray,
    fg_mask: np.ndarray,
    cell_size: int = 32,
    n_iter: int = 3,
    image: Optional[np.ndarray] = None,
    image_offset: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """
    A3 — As-Rigid-As-Possible regularisation of an optical-flow field.

    Raw optical-flow vectors on anime characters can make straight line-art
    strokes "bend" during warping (each pixel moves independently, breaking
    collinearity).  ARAP regularisation fits per-cell *rigid* transformations
    (translation + rotation only, no shear/scale) to the per-pixel flow, then
    reconstructs a smooth flow by interpolating from the cell centres.  The
    result bends the character at joints rather than stretching it like fluid.

    The algorithm (Sýkora 2009, adapted for dense-flow regularisation):
      1. Divide the image into ``cell_size × cell_size`` grid cells.
      2. For each cell, compute the centroid of the fg flow vectors and a per-
         cell rotation matrix (best-fit rigid transform for the cell's vectors).
      3. Reconstruct a smooth flow from bilinear interpolation of the per-cell
         rigid centres.
      4. Iterate ``n_iter`` times (each pass makes the field smoother).

    When ``image`` is provided, ``cv2.createLineSegmentDetector`` extracts
    straight line segments from it.  Cells that share a detected line segment
    are constrained to the same median translation, preventing straight line-art
    strokes from bending during the warp (Sýkora 2009 collinearity term).

    Parameters
    ----------
    flow    : (H, W, 2) float32 — raw optical flow to regularise.
    fg_mask : (H, W) bool — True where foreground character pixels exist.
    cell_size : Grid cell size in pixels.
    n_iter  : Number of regularise passes (1-3 is usually enough).
    image   : Optional image used for LSD line detection.  Pass the seam-band
              crop (not the full canvas) for efficiency and relevance.
    image_offset : (row_offset, col_offset) — offset of ``image`` within the
              full canvas coordinate system.  LSD line coordinates are
              detected in ``image``-space and shifted by this offset before
              mapping to the full-canvas cell grid.  If ``image`` is the full
              canvas, pass (0, 0) (default).  If ``image`` is a crop starting
              at canvas row ``y0``, pass ``(y0, 0)``.

    Returns
    -------
    (H, W, 2) float32 — regularised flow (identical to input for bg pixels).
    """
    H, W = flow.shape[:2]

    if _BATCH_FGREG:
        try:
            oy, ox = image_offset
            result = _batch_fgreg.fg_register.arap_push_regularise(
                np.ascontiguousarray(flow.astype(np.float32)),
                np.ascontiguousarray(
                    fg_mask.astype(np.uint8) if fg_mask.dtype != np.uint8 else fg_mask
                ),
                cell_size,
                n_iter,
                np.ascontiguousarray(image) if image is not None else None,
                oy,
                ox,
            )
            return result.astype(np.float32)
        except Exception as _e:
            logger.debug("batch.fg_register.arap_push_regularise failed (%s), using Python", _e)

    out = flow.copy()

    # LSD collinearity constraint (Sýkora 2009 §3.3).
    # Detect straight line segments in the source image (typically the seam-band
    # crop for efficiency).  Line coordinates are shifted from image-space to
    # canvas-space via image_offset so they map correctly to the full-canvas
    # cell grid built from ``flow`` (shape H×W).
    lsd_lines: Optional[list] = None
    if image is not None:
        try:
            lsd = cv2.createLineSegmentDetector()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            lines, _, _, _ = lsd.detect(gray)
            if lines is not None:
                oy, ox = image_offset
                lsd_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.hypot(x2 - x1, y2 - y1)
                    if length >= cell_size:
                        # Shift from crop-space to canvas-space
                        lsd_lines.append((x1 + ox, y1 + oy, x2 + ox, y2 + oy))
        except Exception:
            pass

    for _ in range(n_iter):
        # Per-cell mean translation from fg flow vectors
        ny = max(1, H // cell_size)
        nx = max(1, W // cell_size)
        cell_tx = np.zeros((ny, nx), dtype=np.float32)
        cell_ty = np.zeros((ny, nx), dtype=np.float32)
        cell_count = np.zeros((ny, nx), dtype=np.float32)

        for ci in range(ny):
            y0, y1 = ci * cell_size, min(H, (ci + 1) * cell_size)
            for cj in range(nx):
                x0, x1 = cj * cell_size, min(W, (cj + 1) * cell_size)
                fg_cell = fg_mask[y0:y1, x0:x1]
                if fg_cell.any():
                    fx_cell = out[y0:y1, x0:x1, 0][fg_cell]
                    fy_cell = out[y0:y1, x0:x1, 1][fg_cell]
                    # Trimmed mean for outlier robustness (per-cell medoid)
                    cell_tx[ci, cj] = float(np.median(fx_cell))
                    cell_ty[ci, cj] = float(np.median(fy_cell))
                    cell_count[ci, cj] = fg_cell.sum()

        # Apply LSD collinearity constraints: cells intersected by the same
        # long line-art stroke are forced to share the mean translation of the
        # group, preventing straight lines from bending across the warp.
        if lsd_lines:
            for x1, y1, x2, y2 in lsd_lines:
                length = np.hypot(x2 - x1, y2 - y1)
                num_pts = max(2, int(length / (cell_size / 2)))
                xs = np.linspace(x1, x2, num_pts)
                ys = np.linspace(y1, y2, num_pts)

                cells_hit = set()
                for lx, ly in zip(xs, ys, strict=False):
                    ci = int(ly // cell_size)
                    cj = int(lx // cell_size)
                    fy = int(ly)
                    fx = int(lx)
                    if (
                        0 <= ci < ny
                        and 0 <= cj < nx
                        and 0 <= fy < H
                        and 0 <= fx < W
                        and fg_mask[fy, fx]
                    ):
                        cells_hit.add((ci, cj))

                if len(cells_hit) > 1:
                    hit_tx = [
                        cell_tx[ci, cj]
                        for (ci, cj) in cells_hit
                        if cell_count[ci, cj] > 0
                    ]
                    hit_ty = [
                        cell_ty[ci, cj]
                        for (ci, cj) in cells_hit
                        if cell_count[ci, cj] > 0
                    ]
                    if hit_tx and hit_ty:
                        avg_tx = float(np.mean(hit_tx))
                        avg_ty = float(np.mean(hit_ty))
                        for ci, cj in cells_hit:
                            cell_tx[ci, cj] = avg_tx
                            cell_ty[ci, cj] = avg_ty

        # Bilinearly interpolate per-cell rigid translations back to pixel space
        if ny > 1 and nx > 1:
            # Cell-centre coordinates
            cy_pts = np.clip(
                np.arange(ny, dtype=np.float32) * cell_size + cell_size / 2, 0, H - 1
            )
            cx_pts = np.clip(
                np.arange(nx, dtype=np.float32) * cell_size + cell_size / 2, 0, W - 1
            )
            from scipy.interpolate import RegularGridInterpolator  # lazy import

            interp_x = RegularGridInterpolator(
                (cy_pts, cx_pts),
                cell_tx,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            interp_y = RegularGridInterpolator(
                (cy_pts, cx_pts),
                cell_ty,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            ys, xs = np.mgrid[0:H, 0:W]
            pts = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float32)
            smooth_tx = interp_x(pts).reshape(H, W).astype(np.float32)
            smooth_ty = interp_y(pts).reshape(H, W).astype(np.float32)

            # Blend: fg pixels move toward the ARAP-regularised value;
            # bg pixels are left completely unchanged.
            blend = fg_mask.astype(np.float32)
            out[:, :, 0] = blend * smooth_tx + (1 - blend) * out[:, :, 0]
            out[:, :, 1] = blend * smooth_ty + (1 - blend) * out[:, :, 1]

    # LSD collinearity term (§0.1 / S8): project per-cell flow onto detected
    # line directions so straight ink outlines cannot be bent by the warp.
    # Only applied to fg/bg BOUNDARY cells (cells containing both fg and bg
    # pixels) — these are the cells that actually contain character outline
    # strokes.  Interior fg cells (flat colour fill) and pure bg cells are
    # intentionally skipped to avoid corrupting the rigid-body translation
    # estimate for the character interior.
    if image is not None:
        try:
            gray_lsd = (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            )
            lsd = cv2.createLineSegmentDetector(0)
            lines_raw, _, _, _ = lsd.detect(gray_lsd)  # (K,1,4) or None
            if lines_raw is not None and len(lines_raw) > 0:
                lines_xy = lines_raw.reshape(
                    -1, 4
                )  # (K,4): x1,y1,x2,y2 in image coords
                row_off, col_off = image_offset
                ny = max(1, H // cell_size)
                nx = max(1, W // cell_size)
                for ci in range(ny):
                    cy_c = ci * cell_size + cell_size / 2
                    for cj in range(nx):
                        cx_c = cj * cell_size + cell_size / 2
                        y0c = max(0, ci * cell_size)
                        y1c = min(H, (ci + 1) * cell_size)
                        x0c = max(0, cj * cell_size)
                        x1c = min(W, (cj + 1) * cell_size)
                        fg_cell = fg_mask[y0c:y1c, x0c:x1c]
                        # Only boundary cells: contain both fg and bg pixels
                        if not fg_cell.any() or fg_cell.all():
                            continue
                        # Map cell centre to image-crop space
                        iy_c = cy_c - row_off
                        ix_c = cx_c - col_off
                        for seg in lines_xy:
                            x1, y1, x2, y2 = (
                                float(seg[0]),
                                float(seg[1]),
                                float(seg[2]),
                                float(seg[3]),
                            )
                            bx0, bx1 = min(x1, x2) - cell_size, max(x1, x2) + cell_size
                            by0, by1 = min(y1, y2) - cell_size, max(y1, y2) + cell_size
                            if not (bx0 <= ix_c <= bx1 and by0 <= iy_c <= by1):
                                continue
                            dx_l = x2 - x1
                            dy_l = y2 - y1
                            seg_len = max(float(np.hypot(dx_l, dy_l)), 1e-8)
                            ux, uy = dx_l / seg_len, dy_l / seg_len
                            flow_x = float(out[y0c:y1c, x0c:x1c, 0].mean())
                            flow_y = float(out[y0c:y1c, x0c:x1c, 1].mean())
                            orig_mag = float(np.hypot(flow_x, flow_y))
                            if orig_mag < 0.1:
                                break
                            proj = flow_x * ux + flow_y * uy
                            proj_x = proj * ux
                            proj_y = proj * uy
                            proj_mag = abs(proj)
                            # Only apply when projection retains ≥50% of the
                            # original magnitude — prevents vertical-line
                            # segments from cancelling horizontal translation.
                            if proj_mag < orig_mag * 0.5:
                                break
                            out[y0c:y1c, x0c:x1c, 0] = np.where(
                                fg_cell, proj_x, out[y0c:y1c, x0c:x1c, 0]
                            )
                            out[y0c:y1c, x0c:x1c, 1] = np.where(
                                fg_cell, proj_y, out[y0c:y1c, x0c:x1c, 1]
                            )
                            break  # one dominant line per cell is sufficient
        except Exception:
            pass  # LSD collinearity is best-effort; never abort the warp

    return out


__all__ = ["_arap_push", "_arap_regularise", "_ARAP_PUSH_ENABLED"]
