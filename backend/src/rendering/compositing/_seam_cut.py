"""DP/C++ seam-cut path finding between two overlapping zones."""

from __future__ import annotations

import cv2
import numpy as np
from backend.src.constants import LUMINANCE_WEIGHTS
from scipy.ndimage import minimum_filter1d as _min_filt1d

from ._native import BATCH_AVAILABLE, batch


def _compute_seam_energy(
    img1: np.ndarray,
    img2: np.ndarray,
    edge_weight: float,
    sem_cost: np.ndarray | None,
    sem_weight: float,
) -> np.ndarray:
    diff = cv2.absdiff(img1, img2).astype(np.float32).dot(LUMINANCE_WEIGHTS)
    gx_d = cv2.Sobel(diff, cv2.CV_32F, 1, 0, ksize=3)
    gy_d = cv2.Sobel(diff, cv2.CV_32F, 0, 1, ksize=3)
    energy = diff + 0.5 * (np.abs(gx_d) + np.abs(gy_d))

    for img in (img1, img2):
        gray = img.astype(np.float32).dot(LUMINANCE_WEIGHTS)
        gx_i = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy_i = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        energy += edge_weight * (np.abs(gx_i) + np.abs(gy_i))

    # P2.4 — Semantic character boundary avoidance (hard veto for character cels)
    if sem_cost is not None and sem_cost.shape == energy.shape:
        energy += sem_weight * sem_cost + 1e6 * (sem_cost > 0.1).astype(np.float32)

    return energy


def _seam_cut_batch(
    img1: np.ndarray,
    img2: np.ndarray,
    edge_weight: float,
    sem_cost: np.ndarray | None,
    sem_weight: float,
    waypoints: list[tuple[int, int]] | None,
) -> np.ndarray:
    w_list = []
    if waypoints:
        w_list = [-1] * img1.shape[1]
        for wx, wy in waypoints:
            if 0 <= wx < len(w_list):
                w_list[wx] = wy
    c_cost = (sem_cost * sem_weight) if sem_cost is not None else None
    return batch.seam.seam_cut(img1, img2, c_cost, w_list, 0.0, edge_weight)


def _prepare_waypoints(
    W_e: int,
    h_e: int,
    waypoints: list[tuple[int, int]] | None,
) -> tuple[dict[int, int], np.ndarray]:
    wp_force: dict[int, int] = {}
    wp_inf_mask = np.zeros((W_e, h_e), dtype=bool)
    if waypoints:
        for x_wp, y_wp in waypoints:
            if 0 <= x_wp < W_e and 0 <= y_wp < h_e:
                wp_force[x_wp] = y_wp
                wp_inf_mask[x_wp, :] = True
                wp_inf_mask[x_wp, y_wp] = False
    return wp_force, wp_inf_mask


def _seam_cut_python(
    img1: np.ndarray,
    img2: np.ndarray,
    edge_weight: float = 15.0,
    sem_cost: np.ndarray | None = None,
    sem_weight: float = 200.0,
    waypoints: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """
    DP seam cut that strongly avoids outlines in *either* frame.

    Energy = diff(img1,img2) + grad(diff) + edge_weight*(edges_in_img1 + edges_in_img2)
             + sem_weight * sem_cost   (P2.4 — character boundary avoidance)

    Returns path[x] = y-offset in [0, h-1] for the minimum-energy horizontal
    cut running left→right across the (h × W × 3) slices.

    §2.11A: *waypoints* is an optional list of ``(x, y)`` pairs in zone-local
    coordinates (x = column 0..W-1, y = row 0..h-1).  Each waypoint forces the
    seam to pass through that exact pixel by setting all other rows in column x
    to ``+inf`` before the DP forward pass.  The seam then fans out from the
    forced pixel in subsequent columns, so 3-connectivity is preserved.
    """
    energy = _compute_seam_energy(img1, img2, edge_weight, sem_cost, sem_weight)

    # Transpose (h, W) → (W, h) so DP runs left→right; path[x] = y-offset
    E = energy.T.copy()
    W_e, h_e = E.shape

    # §2.11A: Intelligent Scissors waypoint injection.
    wp_force, wp_inf_mask = _prepare_waypoints(W_e, h_e, waypoints)
    E[wp_inf_mask] = np.inf

    # §1.5A: Vectorized DP forward pass.
    for i in range(1, W_e):
        E[i] += _min_filt1d(E[i - 1], size=3, mode="constant", cval=np.inf)

    # Traceback: avoid per-step Python list allocation by using slice argmin.
    path = np.empty(W_e, dtype=np.int32)
    j = wp_force.get(W_e - 1, int(E[W_e - 1].argmin()))
    path[W_e - 1] = j
    for i in range(W_e - 2, -1, -1):
        if i in wp_force:
            j = wp_force[i]  # §2.11A: hard-force seam through waypoint
        else:
            j_lo = max(0, j - 1)
            j_hi = min(h_e, j + 2)  # exclusive
            j = j_lo + int(E[i, j_lo:j_hi].argmin())
        path[i] = j
    return path  # path[x] in [0, zone_h-1]  — post-processing lives in _seam_cut


def _seam_cut(
    img1: np.ndarray,
    img2: np.ndarray,
    edge_weight: float = 15.0,
    sem_cost: np.ndarray | None = None,
    sem_weight: float = 200.0,
    waypoints: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Public seam-cut entry point.  Dispatches to C++ or Python DP backend,
    then re-applies §2.11A waypoint hard-pins so both backends behave
    identically in production."""
    if BATCH_AVAILABLE:
        path = _seam_cut_batch(img1, img2, edge_weight, sem_cost, sem_weight, waypoints)
    else:
        path = _seam_cut_python(img1, img2, edge_weight, sem_cost, sem_weight, waypoints)

    h_e = img1.shape[0]  # zone height (path rows are in [0, h_e-1])

    # §2.11A: re-apply waypoints so backend post-processing cannot displace
    # the user-specified seam positions.  Hard constraints win.
    if waypoints:
        W_e = img1.shape[1]
        for x_wp, y_wp in waypoints:
            if 0 <= x_wp < W_e and 0 <= y_wp < h_e:
                path[x_wp] = y_wp
    return path


__all__ = [
    "_compute_seam_energy",
    "_seam_cut_batch",
    "_prepare_waypoints",
    "_seam_cut_python",
    "_seam_cut",
]
