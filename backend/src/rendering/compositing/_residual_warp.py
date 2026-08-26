"""P3: Regularized Thin Plate Spline (TPS) residual background-only local warp candidate.

Implements the P3 candidate contract from the ASP quality proposal:
1. After the existing global affine fit, estimate a regularized Thin Plate Spline (TPS)
   residual from correspondences restricted to agreed background regions.
2. Safety check: Reject residual warps with high bending energy (distortion risk) or
   excessive maximum displacement (> max_residual_px), falling back to global affine.
3. Apply residual deformation strictly to confirmed background samples; smoothly fade
   the residual to zero near foreground boundaries so character anatomy is never torn.
4. Gated behind default-off ``ASP_RESIDUAL_WARP=1``.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def residual_warp_enabled() -> bool:
    """True when ASP_RESIDUAL_WARP is enabled."""
    return os.environ.get("ASP_RESIDUAL_WARP", "0") == "1"


def compute_tps_bending_energy(
    pts_src: np.ndarray,
    displacements: np.ndarray,
    regularization: float = 1.0,
) -> float:
    """Compute the Thin Plate Spline bending energy of a 2D displacement field.

    Parameters
    ----------
    pts_src : (K, 2) float array of source anchor coordinates (x, y).
    displacements : (K, 2) float array of residual displacement vectors (dx, dy).
    regularization : float, lambda smoothing parameter.

    Returns
    -------
    bending_energy : float, normalized bending energy measure.
    """
    K = len(pts_src)
    if K < 4:
        return 0.0

    # Pairwise distances
    diff = pts_src[:, None, :] - pts_src[None, :, :]
    r = np.linalg.norm(diff, axis=2)

    # U(r) = r^2 * log(r^2) = 2 * r^2 * log(r) for r > 0, else 0
    with np.errstate(divide="ignore", invalid="ignore"):
        U = np.where(r > 1e-6, (r**2) * np.log(r**2), 0.0)

    # Augmented TPS linear system:
    # [ K + lambda*I   P ] [ W ] = [ V ]
    # [ P^T            0 ] [ A ]   [ 0 ]
    P = np.column_stack([np.ones(K), pts_src])  # (K, 3)
    L = np.zeros((K + 3, K + 3), dtype=np.float64)
    L[:K, :K] = U + regularization * np.eye(K)
    L[:K, K:] = P
    L[K:, :K] = P.T

    rhs = np.zeros((K + 3, 2), dtype=np.float64)
    rhs[:K] = displacements

    try:
        sol = np.linalg.solve(L, rhs)
        W = sol[:K]  # (K, 2)
        # Bending energy = trace(W^T * U * W)
        bending = float(np.trace(W.T @ U @ W)) / float(K)
        return max(0.0, bending)
    except np.linalg.LinAlgError:
        return float("inf")


def fit_background_residual_tps(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    global_affine: np.ndarray,
    canvas_H: int,
    canvas_W: int,
    *,
    smoothing: float = 1.0,
    max_bending: float = 0.05,
    max_residual_px: float = 15.0,
    grid_step: int = 16,
) -> tuple[np.ndarray | None, dict]:
    """Fit a regularized TPS residual flow field from background correspondences.

    Parameters
    ----------
    src_pts : (K, 2) array of coordinates in source frame (x, y).
    dst_pts : (K, 2) array of corresponding target canvas coordinates (x, y).
    global_affine : (2, 3) global affine matrix mapping src -> canvas.
    canvas_H, canvas_W : Canvas dimensions.
    smoothing : float, RBF TPS regularizer smoothing parameter.
    max_bending : float, max allowed bending energy before fallback.
    max_residual_px : float, max allowed residual displacement vector magnitude.
    grid_step : int, downsampled grid step for efficient evaluation.

    Returns
    -------
    (residual_flow, metadata) :
        residual_flow: (canvas_H, canvas_W, 2) float32 array of (dx, dy) canvas offsets,
                       or None if rejected due to high distortion.
        metadata: dict of fit diagnostics.
    """
    meta: dict = {"n_anchors": len(src_pts), "applied": False}
    if len(src_pts) < 4:
        meta["reason"] = "insufficient_anchors"
        return None, meta

    # Predicted canvas positions under global affine
    src_h = np.column_stack([src_pts, np.ones(len(src_pts))])
    pred_canvas = (global_affine @ src_h.T).T  # (K, 2)

    # Residuals: delta = target - predicted
    residuals = dst_pts - pred_canvas
    res_norms = np.linalg.norm(residuals, axis=1)
    max_observed_res = float(res_norms.max()) if len(res_norms) > 0 else 0.0
    meta["max_residual_px"] = max_observed_res

    if max_observed_res > max_residual_px:
        meta["reason"] = f"max_residual_exceeded ({max_observed_res:.1f}px > {max_residual_px:.1f}px)"
        return None, meta

    # Bending energy check
    bending = compute_tps_bending_energy(pred_canvas, residuals, regularization=smoothing)
    meta["bending_energy"] = bending

    if bending > max_bending:
        meta["reason"] = f"high_bending_energy ({bending:.4f} > {max_bending:.4f})"
        return None, meta

    # Evaluate RBF TPS on downsampled grid
    from scipy.interpolate import RBFInterpolator  # lazy import

    gh = int(np.ceil(canvas_H / float(grid_step)))
    gw = int(np.ceil(canvas_W / float(grid_step)))
    gy, gx = np.mgrid[0:gh, 0:gw]
    grid_pts = np.column_stack([gx.ravel() * grid_step, gy.ravel() * grid_step]).astype(np.float64)

    try:
        rbf_x = RBFInterpolator(pred_canvas, residuals[:, 0], kernel="thin_plate_spline", smoothing=smoothing)
        rbf_y = RBFInterpolator(pred_canvas, residuals[:, 1], kernel="thin_plate_spline", smoothing=smoothing)

        grid_dx = rbf_x(grid_pts).reshape(gh, gw).astype(np.float32)
        grid_dy = rbf_y(grid_pts).reshape(gh, gw).astype(np.float32)

        # Upsample to full canvas resolution
        dense_dx = cv2.resize(grid_dx, (canvas_W, canvas_H), interpolation=cv2.INTER_LINEAR)
        dense_dy = cv2.resize(grid_dy, (canvas_W, canvas_H), interpolation=cv2.INTER_LINEAR)

        flow = np.stack([dense_dx, dense_dy], axis=2)
        meta["applied"] = True
        return flow, meta
    except Exception as e:
        meta["reason"] = f"tps_fit_failed: {e}"
        return None, meta


def apply_residual_warp_to_frame(
    frame: np.ndarray,
    bg_mask: np.ndarray | None,
    global_affine: np.ndarray,
    residual_flow: np.ndarray | None,
    canvas_H: int,
    canvas_W: int,
    *,
    blend_margin_px: int = 12,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Warp frame onto canvas with background-only residual TPS deformation.

    Parameters
    ----------
    frame : (H, W, 3) uint8 BGR source frame.
    bg_mask : (H, W) uint8 background mask (255=bg, 0=fg).
    global_affine : (2, 3) global affine matrix mapping frame -> canvas.
    residual_flow : (canvas_H, canvas_W, 2) float32 residual flow field or None.
    canvas_H, canvas_W : Canvas dimensions.
    blend_margin_px : int, feather margin to smoothly fade residual to zero near foreground.

    Returns
    -------
    (warped_frame, warped_bg_mask) :
        warped_frame: (canvas_H, canvas_W, 3) uint8 canvas-mapped frame.
        warped_bg_mask: (canvas_H, canvas_W) bool background presence mask.
    """
    # 1. Base affine inverse map
    M_3x3 = np.eye(3, dtype=np.float64)
    M_3x3[:2, :3] = global_affine[:2, :3]
    inv_M = np.linalg.inv(M_3x3)[:2, :3].astype(np.float32)

    # Canvas coordinate grid (x_c, y_c)
    cy, cx = np.mgrid[0:canvas_H, 0:canvas_W].astype(np.float32)

    # Warp background mask to canvas space using global affine
    if bg_mask is not None:
        warped_bg_raw = cv2.warpAffine(
            bg_mask.astype(np.uint8),
            global_affine,
            (canvas_W, canvas_H),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        bg_presence = warped_bg_raw > 127
    else:
        warped_bg_raw = np.full((canvas_H, canvas_W), 255, dtype=np.uint8)
        bg_presence = np.ones((canvas_H, canvas_W), dtype=bool)

    if residual_flow is not None and bg_presence.any():
        # Modulate residual flow by background distance transform
        # so residual displacement fades to zero near character foreground
        bg_dist = cv2.distanceTransform(bg_presence.astype(np.uint8), cv2.DIST_L2, 3)
        weight = np.clip(bg_dist / float(max(1, blend_margin_px)), 0.0, 1.0)

        # Modulated canvas query coordinates: (cx - dx * w, cy - dy * w)
        mod_cx = cx - residual_flow[..., 0] * weight
        mod_cy = cy - residual_flow[..., 1] * weight
    else:
        mod_cx = cx
        mod_cy = cy

    # Map modulated canvas coordinates back to frame coordinates via inv_M
    map_x = inv_M[0, 0] * mod_cx + inv_M[0, 1] * mod_cy + inv_M[0, 2]
    map_y = inv_M[1, 0] * mod_cx + inv_M[1, 1] * mod_cy + inv_M[1, 2]

    # Remap frame to canvas
    warped_frame = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return warped_frame, bg_presence


__all__ = [
    "residual_warp_enabled",
    "compute_tps_bending_energy",
    "fit_background_residual_tps",
    "apply_residual_warp_to_frame",
]
