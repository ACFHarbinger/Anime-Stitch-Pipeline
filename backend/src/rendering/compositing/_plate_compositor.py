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
_PLATE_MIN_BG_SAMPLES: int = max(1, int(os.environ.get("ASP_PLATE_MIN_BG_SAMPLES", "2")))

logger = logging.getLogger(__name__)


def plate_single_pose_enabled() -> bool:
    """True when ASP_PLATE_SINGLE_POSE is enabled."""
    return os.environ.get("ASP_PLATE_SINGLE_POSE", "0") == "1"


def plate_multiphase_enabled() -> bool:
    """True when the default-off piecewise multi-phase P1 candidate is enabled."""
    return os.environ.get("ASP_PLATE_MULTIPHASE", "0") == "1"


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


def plate_single_pose_safe_for_phases(
    phase_ids: list[int] | None,
    n_frames: int,
    source_has_multiple_phases: bool = False,
) -> bool:
    """A single plate requires aligned, single-phase provenance."""
    return (
        not source_has_multiple_phases
        and (
            not phase_ids
            or (len(phase_ids) == n_frames and len(set(phase_ids)) <= 1)
        )
    )


def _build_aligned_background_plate(
    warped_frames: list[np.ndarray],
    warped_bg: list[np.ndarray | None],
    H: int,
    W: int,
    *,
    warped_valid: list[np.ndarray] | None = None,
    robust_gain: bool = _JOINT_GAIN_ROBUST,
    edge_preserve: bool = False,
    min_background_samples: int = 1,
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

        sample_count = np.count_nonzero(~np.isnan(samples[..., 0]), axis=0)

        # Suppress benign All-NaN slice warnings when scanning unpopulated regions
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            with np.errstate(invalid="ignore"):
                med = np.nanmedian(samples, axis=0)
                valid = (~np.isnan(med[..., 0])) & (sample_count >= min_background_samples)

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

    # 4. Fill residual voids only when a single source is accepted. P1 uses
    # the temporal canvas for uncorroborated plate pixels to avoid strip seams.
    void_mask = ~valid_plate_mask
    if min_background_samples <= 1 and void_mask.any():
        for i in range(N):
            valid = (
                warped_valid[i].astype(bool)
                if warped_valid is not None and i < len(warped_valid)
                else np.ones((H, W), dtype=bool)
            )
            presence = void_mask & valid & (warped_norm[i].max(axis=2) > 0)
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
    return_plate_valid: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict] | tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
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
        warped_frames,
        warped_bg,
        H,
        W,
        warped_valid=warped_valid,
        edge_preserve=edge_preserve,
        min_background_samples=min(_PLATE_MIN_BG_SAMPLES, len(warped_frames)),
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
        if return_plate_valid:
            return result, claimed_map, meta, plate_valid
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
    if return_plate_valid:
        return result, claimed_map, meta, plate_valid
    return result, claimed_map, meta


def _laplacian_blend(left: np.ndarray, right: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Blend two equal canvas images with a soft mask in Laplacian space."""
    levels = min(4, max(1, int(np.log2(min(left.shape[:2]))) - 2))
    gp_left = [left.astype(np.float32)]
    gp_right = [right.astype(np.float32)]
    gp_alpha = [alpha.astype(np.float32)]
    for _ in range(levels):
        gp_left.append(cv2.pyrDown(gp_left[-1]))
        gp_right.append(cv2.pyrDown(gp_right[-1]))
        gp_alpha.append(cv2.pyrDown(gp_alpha[-1]))
    lp_left = [gp_left[-1]]
    lp_right = [gp_right[-1]]
    for level in range(levels, 0, -1):
        size = (gp_left[level - 1].shape[1], gp_left[level - 1].shape[0])
        lp_left.append(gp_left[level - 1] - cv2.pyrUp(gp_left[level], dstsize=size))
        lp_right.append(gp_right[level - 1] - cv2.pyrUp(gp_right[level], dstsize=size))
    merged = lp_left[0] * gp_alpha[-1][:, :, None] + lp_right[0] * (1.0 - gp_alpha[-1][:, :, None])
    for level in range(1, len(lp_left)):
        size = (lp_left[level].shape[1], lp_left[level].shape[0])
        merged = cv2.pyrUp(merged, dstsize=size)
        alpha_level = gp_alpha[-1 - level][:, :, None]
        merged += lp_left[level] * alpha_level + lp_right[level] * (1.0 - alpha_level)
    return np.clip(merged, 0, 255).astype(np.uint8)


# Target half-width (rows) of the plate-to-plate transition. Bounded per side
# only by how far each plate's own valid content reaches from the seam -- NOT
# by the mutual valid overlap. Contiguous canvas-``ty`` phase bands overlap by
# only a handful of rows (often zero), which collapsed the old feather to a
# hard cut and drove the ``seam_vis_gate`` failures on the multi-phase join.
_PHASE_JOIN_HALF_PX: int = 48


def _blend_phase_plates(
    left: np.ndarray,
    left_claimed: np.ndarray,
    left_valid: np.ndarray,
    right: np.ndarray,
    right_claimed: np.ndarray,
    right_valid: np.ndarray,
    seam_y: int,
    *,
    blend_width: int = _PHASE_JOIN_HALF_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Join adjacent physical phase bands without feathering hero cels.

    The transition ramp spans ``+/- blend_width`` rows around ``seam_y``,
    clamped only by each plate's own valid extent (so it reaches past a narrow
    or empty mutual overlap into each plate's solo-valid region). A tapered
    low-frequency offset first pulls both plates toward their shared seam-strip
    mean, so the ramp only has to hide residual mid-frequency detail; the
    offset decays to zero at the band edges, leaving far-field plate brightness
    untouched. Any pre-composited hero cel crosses the join verbatim.
    """
    H, W = left.shape[:2]
    rows = np.arange(H, dtype=np.float32)

    left_rows = np.flatnonzero(left_valid.any(axis=1))
    right_rows = np.flatnonzero(right_valid.any(axis=1))
    if len(left_rows) == 0:
        return right.copy(), right_claimed.copy()
    if len(right_rows) == 0:
        return left.copy(), left_claimed.copy()

    up = min(int(blend_width), seam_y - int(left_rows[0]))
    down = min(int(blend_width), int(right_rows[-1]) - seam_y)
    half = max(1, min(up, down))
    y0 = max(0, seam_y - half)
    y1 = min(H, seam_y + half + 1)

    # Smoothstep ramp: 1.0 (all left) at y0 -> 0.0 (all right) at y1 - 1.
    t = np.clip((rows - y0) / max(1, y1 - 1 - y0), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)

    result = left.copy()
    claimed = left_claimed.copy()

    # Background ownership outside the ramp stays a hard hand-off at seam_y.
    right_only = right_valid & ~left_valid
    right_side = np.broadcast_to((rows >= seam_y)[:, None], (H, W))
    use_right = right_only | (left_valid & right_valid & right_side)
    result[use_right] = right[use_right]
    claimed[use_right] = right_claimed[use_right]

    # Tapered low-frequency reconciliation, measured on the background
    # (non-cel) seam strip that each plate individually validates.
    lf = left.astype(np.float32)
    rf = right.astype(np.float32)
    strip = max(2, half // 2)
    ls = slice(max(0, seam_y - strip), seam_y)
    rs = slice(seam_y, min(H, seam_y + strip))
    left_strip = left_valid[ls] & (left_claimed[ls] < 0) & (left[ls].max(axis=2) > 5)
    right_strip = right_valid[rs] & (right_claimed[rs] < 0) & (right[rs].max(axis=2) > 5)
    left_vals = lf[ls][left_strip]
    right_vals = rf[rs][right_strip]
    if left_vals.size and right_vals.size:
        left_mean = left_vals.mean(axis=0)
        right_mean = right_vals.mean(axis=0)
        midpoint = 0.5 * (left_mean + right_mean)
        taper = np.clip(1.0 - np.abs(rows - seam_y) / max(1, half), 0.0, 1.0)
        taper = (taper * taper * (3.0 - 2.0 * taper))[:, None, None]
        lf = lf + taper * (midpoint - left_mean)
        rf = rf + taper * (midpoint - right_mean)

    # Inside the ramp, blend each plate's own valid pixels (Laplacian space,
    # cropped to a neighbourhood of the band so the pyramid alpha resolves the
    # ramp instead of averaging it away at the coarsest level).
    band = np.zeros((H, W), dtype=bool)
    band[y0:y1] = True
    blend_bg = band & (left_claimed < 0) & (right_claimed < 0) & (left_valid | right_valid)
    if blend_bg.any():
        left_fill = np.where(left_valid[..., None], lf, rf)
        right_fill = np.where(right_valid[..., None], rf, lf)
        cy0 = max(0, y0 - half)
        cy1 = min(H, y1 + half)
        crop_alpha = np.ascontiguousarray(
            np.broadcast_to((1.0 - t)[cy0:cy1, None], (cy1 - cy0, W))
        )
        merged_crop = _laplacian_blend(
            np.ascontiguousarray(np.clip(left_fill[cy0:cy1], 0, 255).astype(np.uint8)),
            np.ascontiguousarray(np.clip(right_fill[cy0:cy1], 0, 255).astype(np.uint8)),
            crop_alpha,
        )
        blend_local = blend_bg[cy0:cy1]
        result[cy0:cy1][blend_local] = merged_crop[blend_local]

    # Apply upper-band then lower-band cels: in an overlap, the lower physical
    # band wins. Each source already carries composite_plate_single_pose's
    # silhouette feather, so never apply the background transition ramp here.
    for source, source_claimed in (
        (left, left_claimed),
        (right, right_claimed),
    ):
        cel = source_claimed >= 0
        if not cel.any():
            continue
        result[cel] = source[cel]
        claimed[cel] = source_claimed[cel]
    return result, claimed


def composite_plate_multiphase(
    warped_frames: list[np.ndarray],
    warped_bg: list[np.ndarray | None],
    canvas: np.ndarray,
    warped_valid: list[np.ndarray],
    spans: list[tuple[int, int, int]],
    physical_phase_order: list[int],
    *,
    edge_preserve: bool,
    multiband: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build P1 plates for phase spans and join them in canvas-``ty`` order."""
    phase_results: dict[int, tuple[np.ndarray, np.ndarray, dict, np.ndarray, int, int]] = {}
    for phase, start, end in spans:
        result, claimed, meta, plate_valid = composite_plate_single_pose(
            warped_frames[start:end + 1],
            warped_bg[start:end + 1],
            canvas,
            warped_valid=warped_valid[start:end + 1],
            edge_preserve=edge_preserve,
            multiband=multiband,
            return_plate_valid=True,
        )
        global_claimed = claimed.copy()
        claimed_pixels = global_claimed >= 0
        global_claimed[claimed_pixels] += start
        phase_results[phase] = (result, global_claimed, meta, plate_valid, start, end)

    first = physical_phase_order[0]
    result, claimed, _meta, valid, _start, _end = phase_results[first]
    ownership_spans: list[dict] = []
    for phase in physical_phase_order:
        _result, phase_claimed, meta, _valid, start, end = phase_results[phase]
        zones = []
        for zone in meta["zones"]:
            zone_out = dict(zone)
            zone_out["chosen_frame"] += start
            zone_out["candidates"] = [
                (idx + start, area, score) for idx, area, score in zone["candidates"]
            ]
            zones.append(zone_out)
        ownership_spans.append({"phase": phase, "start": start, "end": end, "zones": zones,
                                "n_claimed_pixels": int((phase_claimed >= 0).sum())})

    for phase in physical_phase_order[1:]:
        right, right_claimed, _meta, right_valid, _start, _end = phase_results[phase]
        left_rows = np.flatnonzero(valid.any(axis=1))
        right_rows = np.flatnonzero(right_valid.any(axis=1))
        if len(left_rows) == 0 or len(right_rows) == 0:
            continue
        seam_y = int(round((left_rows[-1] + right_rows[0]) / 2.0))
        result, claimed = _blend_phase_plates(
            result, claimed, valid, right, right_claimed, right_valid, seam_y
        )
        valid |= right_valid

    metadata = {
        "phase_order": physical_phase_order,
        "spans": ownership_spans,
        "n_claimed_pixels": int((claimed >= 0).sum()),
    }
    return result, claimed, metadata


__all__ = [
    "plate_single_pose_enabled",
    "plate_multiphase_enabled",
    "plate_multiband_enabled",
    "plate_single_pose_safe_for_phases",
    "composite_plate_single_pose",
    "composite_plate_multiphase",
    "PlateCompositeResult",
]
