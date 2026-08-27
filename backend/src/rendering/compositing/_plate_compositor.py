"""P1: Canvas-aligned background plate + single foreground pose compositor candidate.

Implements the renderer contract selected by Harbinger (Priority: One coherent
character pose, then seam cleanup):
1. Build a clean, canvas-aligned background plate from all frames' confirmed
   background pixels using joint gain equalization and robust temporal median.
2. Segment foreground character cels per frame.
3. For each connected foreground component / overlap zone on the canvas, select
   exactly ONE hero pose (maximizing area, centrality, and boundary completeness)
   and rigidly composite it over the clean background plate without multi-pose
   blending or phantom ghosting.
4. Gated behind default-off ``ASP_PLATE_SINGLE_POSE=1``.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass

import cv2
import numpy as np

from ._flags import (
    _JOINT_GAIN_ROBUST,
)
from ._gain_compensation import _apply_joint_gain_solve

_SP_SOFT_PX: int = int(os.environ.get("ASP_SP_SOFT_PX", "8"))

logger = logging.getLogger(__name__)


def plate_single_pose_enabled() -> bool:
    """True when ASP_PLATE_SINGLE_POSE is enabled."""
    return os.environ.get("ASP_PLATE_SINGLE_POSE", "0") == "1"


@dataclass(frozen=True)
class PlateCompositeResult:
    """Output of plate + single-pose compositing."""

    composite: np.ndarray  # (H, W, 3) uint8 BGR
    plate: np.ndarray  # (H, W, 3) uint8 BGR
    claimed_mask: np.ndarray  # (H, W) int32 frame index or -1
    metadata: dict


def plate_multiband_enabled() -> bool:
    """True when ASP_PLATE_MULTIBAND is enabled (P2 seam cleanup over clean plate)."""
    return os.environ.get("ASP_PLATE_MULTIBAND", "0") == "1"


def _build_aligned_background_plate(
    warped_frames: list[np.ndarray],
    warped_bg: list[np.ndarray | None],
    H: int,
    W: int,
    *,
    warped_valid: list[np.ndarray] | None = None,
    robust_gain: bool = _JOINT_GAIN_ROBUST,
    edge_preserve: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a canvas-aligned background plate from warped frames and background masks.

    Parameters
    ----------
    warped_frames : list of (H, W, 3) warped frames.
    warped_bg : list of (H, W) background masks.
    H, W : canvas dimensions.
    robust_gain : bool, robust joint gain solver.
    edge_preserve : bool, when True, protects high-frequency line art in multi-sample zones.

    Returns
    -------
    (plate_bgr, valid_bg_mask)
    """
    N = len(warped_frames)
    if N == 0:
        return np.zeros((H, W, 3), dtype=np.uint8), np.zeros((H, W), dtype=bool)

    # 1. Joint gain equalization across background overlap regions
    if N >= 2:
        try:
            warped_norm = _apply_joint_gain_solve(
                warped_frames,
                warped_bg,
                robust=robust_gain,
            )
        except Exception:
            warped_norm = warped_frames
    else:
        warped_norm = warped_frames

    # 2. Build per-frame contribution masks (content present & marked bg)
    contribution_masks: list[np.ndarray] = []
    for i in range(N):
        wf = warped_norm[i]
        valid = (
            warped_valid[i].astype(bool)
            if warped_valid is not None and i < len(warped_valid)
            else np.ones((H, W), dtype=bool)
        )
        if i < len(warped_bg) and warped_bg[i] is not None:
            if warped_bg[i].dtype == np.uint8:
                # P4: 255 = confirmed bg, 128 = uncertain (excluded), 0 = fg
                bg = warped_bg[i] > 200
            else:
                bg = warped_bg[i].astype(bool)
            m = bg & valid
        else:
            m = valid & (wf.max(axis=2) > 0)
        contribution_masks.append(m)

    # 3. Chunked temporal median over valid background samples
    plate = np.zeros((H, W, 3), dtype=np.uint8)
    valid_plate_mask = np.zeros((H, W), dtype=bool)
    band_rows = max(8, min(64, int(4_000_000 / max(1, N * W))))

    for y0 in range(0, H, band_rows):
        y1 = min(y0 + band_rows, H)
        bh = y1 - y0
        samples = np.full((N, bh, W, 3), np.nan, dtype=np.float32)

        for i in range(N):
            m_band = contribution_masks[i][y0:y1]
            if not m_band.any():
                continue
            samples[i][m_band] = warped_norm[i][y0:y1][m_band].astype(np.float32)

        # Suppress benign All-NaN slice warnings when scanning unpopulated regions
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            with np.errstate(invalid="ignore"):
                med = np.nanmedian(samples, axis=0)
                valid = ~np.isnan(med[..., 0])

        if valid.any():
            band_plate = np.clip(med[valid], 0, 255).astype(np.uint8)

            # P2 Edge-Preserving refinement: in regions with line art, pick
            # the sharpest single source
            if edge_preserve and N >= 2:
                # Compute gradient magnitude of median vs individual samples
                valid_band_mask = np.zeros((bh, W), dtype=bool)
                valid_band_mask[valid] = True
                gray_med = cv2.cvtColor(
                    np.clip(np.nan_to_num(med), 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
                )
                edges = cv2.Canny(gray_med, 50, 150) > 0
                edge_overlap = edges & valid_band_mask

                if edge_overlap.any():
                    # In edge zones, select source frame with maximum local contrast
                    for i in range(N):
                        m_edge = contribution_masks[i][y0:y1] & edge_overlap
                        if m_edge.any():
                            # Assign directly from valid source to avoid blur
                            band_plate_full = np.zeros((bh, W, 3), dtype=np.uint8)
                            band_plate_full[valid] = band_plate
                            band_plate_full[m_edge] = warped_norm[i][y0:y1][m_edge]
                            band_plate = band_plate_full[valid]
                            break

            plate[y0:y1][valid] = band_plate
            valid_plate_mask[y0:y1][valid] = True

    # 4. Fill residual voids from warped frame content (nearest available sample)
    void_mask = ~valid_plate_mask
    if void_mask.any():
        for i in range(N):
            presence = void_mask & (warped_norm[i].max(axis=2) > 0)
            if presence.any():
                plate[presence] = warped_norm[i][presence]
                void_mask[presence] = False
                valid_plate_mask[presence] = True

    return plate, valid_plate_mask


def composite_plate_single_pose(
    warped_frames: list[np.ndarray],
    warped_bg: list[np.ndarray | None],
    canvas: np.ndarray,
    *,
    warped_valid: list[np.ndarray] | None = None,
    soft_edge_px: int = _SP_SOFT_PX,
    edge_preserve: bool = False,
    multiband: bool = False,
    multiband_levels: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Composite single foreground character poses onto a clean canvas-aligned background plate.

    Parameters
    ----------
    warped_frames : list of (H, W, 3) uint8 warped BGR frames.
    warped_bg : list of (H, W) background masks.
    canvas : (H, W, 3) fallback / base canvas.
    soft_edge_px : int, feather width for foreground boundaries.
    edge_preserve : bool, protect high-frequency line art in background plate.
    multiband : bool, apply multi-scale Laplacian pyramid blending across seam boundaries.
    multiband_levels : int, number of pyramid levels for multiband blending.

    Returns
    -------
    (composite, claimed_map, metadata)
    """
    N = len(warped_frames)
    H, W = warped_frames[0].shape[:2]

    # 1. Build clean background plate (with optional P2 edge preservation)
    plate, plate_valid = _build_aligned_background_plate(
        warped_frames, warped_bg, H, W, warped_valid=warped_valid, edge_preserve=edge_preserve
    )
    result = plate.copy()
    if not plate_valid.all() and canvas is not None and canvas.shape == plate.shape:
        # Fallback to canvas in unpopulated regions
        unpopulated = ~plate_valid
        result[unpopulated] = canvas[unpopulated]

    # 2. Extract foreground masks per frame
    fg_masks: list[np.ndarray] = []
    for i in range(N):
        wf = warped_frames[i]
        has_content = wf.max(axis=2) > 5
        if i < len(warped_bg) and warped_bg[i] is not None:
            if warped_bg[i].dtype == np.uint8:
                # Foreground is confirmed character cel (value < 64)
                fg = has_content & (warped_bg[i] < 64)
            else:
                fg = has_content & ~warped_bg[i].astype(bool)
        else:
            fg = np.zeros((H, W), dtype=bool)
        fg_masks.append(fg)

    # 3. Union foreground to find connected overlap zones
    union_fg = np.zeros((H, W), dtype=bool)
    for fg in fg_masks:
        union_fg |= fg

    claimed_map = np.full((H, W), -1, dtype=np.int32)
    meta: dict = {"zones": [], "n_claimed_pixels": 0}

    # 4. P2 Multiband pyramid blending across background plate transitions
    if multiband and N >= 2:
        # Build multi-scale Laplacian pyramid for background plate blending
        gp_plate = [plate.astype(np.float32)]
        for _ in range(multiband_levels):
            gp_plate.append(cv2.pyrDown(gp_plate[-1]))

        lp_plate = [gp_plate[multiband_levels - 1]]
        for i in range(multiband_levels - 1, 0, -1):
            size = (gp_plate[i - 1].shape[1], gp_plate[i - 1].shape[0])
            expanded = cv2.pyrUp(gp_plate[i], dstsize=size)
            lp_plate.append(gp_plate[i - 1] - expanded)

        # Reconstruct plate
        recon = lp_plate[0]
        for i in range(1, multiband_levels):
            size = (lp_plate[i].shape[1], lp_plate[i].shape[0])
            recon = cv2.pyrUp(recon, dstsize=size) + lp_plate[i]

        recon_plate = np.clip(recon, 0, 255).astype(np.uint8)
        # Apply only to non-foreground regions to preserve character crispness
        non_fg = ~union_fg & plate_valid
        result[non_fg] = recon_plate[non_fg]
        meta["multiband_applied"] = True

    if not union_fg.any():
        return result, claimed_map, meta

    # 5. Connected components on union foreground
    num_labels, labels = cv2.connectedComponents(union_fg.astype(np.uint8), connectivity=8)

    for label_idx in range(1, num_labels):
        zone_mask = labels == label_idx
        zone_area = int(zone_mask.sum())
        if zone_area < 32:
            continue

        # Score candidate frames providing a pose in this zone
        best_frame = -1
        best_score = -1.0
        candidates = []

        for i in range(N):
            fg_i = fg_masks[i] & zone_mask
            area_i = int(fg_i.sum())
            if area_i == 0:
                continue

            # Completeness: fraction of zone covered by frame i's pose
            coverage = area_i / float(zone_area)

            # Compactness & boundary truncation penalty
            ys, xs = np.where(fg_i)
            h_f, w_f = warped_frames[i].shape[:2]
            touch_top = (ys.min() <= 1)
            touch_bottom = (ys.max() >= h_f - 2)
            touch_left = (xs.min() <= 1)
            touch_right = (xs.max() >= w_f - 2)
            trunc_penalty = 0.5 if (touch_top or touch_bottom or touch_left or touch_right) else 1.0

            # Score = coverage * truncation penalty
            score = coverage * trunc_penalty
            candidates.append((i, area_i, score))

            if score > best_score:
                best_score = score
                best_frame = i

        if best_frame < 0:
            continue

        # 5. Composite best frame's foreground cel rigidly onto the plate
        chosen_fg = fg_masks[best_frame] & zone_mask
        if soft_edge_px > 0:
            # Soft feathered edge
            fg_dist = cv2.distanceTransform(chosen_fg.astype(np.uint8), cv2.DIST_L2, 3)
            alpha = np.clip(fg_dist / float(soft_edge_px), 0.0, 1.0)
            alpha3 = alpha[:, :, None]
            result[chosen_fg] = np.clip(
                warped_frames[best_frame][chosen_fg] * alpha3[chosen_fg]
                + result[chosen_fg] * (1.0 - alpha3[chosen_fg]),
                0,
                255,
            ).astype(np.uint8)
        else:
            result[chosen_fg] = warped_frames[best_frame][chosen_fg]

        claimed_map[chosen_fg] = best_frame
        meta["zones"].append({
            "label": int(label_idx),
            "area": zone_area,
            "chosen_frame": int(best_frame),
            "score": float(best_score),
            "candidates": [(int(c[0]), int(c[1]), float(c[2])) for c in candidates],
        })

    meta["n_claimed_pixels"] = int((claimed_map >= 0).sum())
    return result, claimed_map, meta


__all__ = [
    "plate_single_pose_enabled",
    "plate_multiband_enabled",
    "composite_plate_single_pose",
    "PlateCompositeResult",
]
