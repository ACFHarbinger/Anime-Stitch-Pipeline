"""Foreground pose registration (Stage 8.5) — the core fix for strip-seam
character misalignment.

Problem
-------
The ASP camera model is translation-only.  It aligns the *background* across
frames perfectly, but the character is *animating* between the frames being
stitched (300–800 ms apart), so its body parts land in two different poses on
either side of every strip seam → the torn / doubled edges visible in test09.

Approach (see research/ASP_Foreground_Assembly_Research.md §5)
------------------------------------------------------------
Both frames are first warped into the same canvas coordinate system by the
existing affines, so the *background* is already aligned.  Any optical flow
that remains on the *foreground* between the two canvas-aligned frames is the
pure animation residual ``A_animation`` (the camera component ``T_camera`` is
removed by the alignment-aware warping — it does not need to be subtracted
explicitly).

We then re-pose both frames' foreground toward the *midpoint* pose
(StabStitch++ bidirectional principle: halves the maximum per-frame
distortion).  The warp magnitude is tapered to zero away from the seam so the
correction is localised to the boundary where it matters (SC-AOF blend-band
principle) and never disturbs canvas regions a single frame owns outright.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from backend.src.constants import (
    FG_REG_MAX_RESIDUAL as _FG_REG_MAX_RESIDUAL_DEFAULT,
)
from backend.src.constants import (
    FG_REG_MIN_FG_PIXELS,
    FG_REG_SMOOTH_SIGMA,
    FG_REG_TAPER_PX,
)

from ._arap import _ARAP_PUSH_ENABLED, _arap_push, _arap_regularise
from ._flow import _dense_flow
from ._geometry import _remap_by_displacement, _seam_taper

# Allow benchmark sweeps to tune the warp-vs-single-pose threshold without an
# edit/rebuild cycle.  Lower → more seams take the single-pose fallback (one
# coherent character pose, no blend) instead of an imperfect re-pose+blend that
# can leave faint edge doubling.
try:
    FG_REG_MAX_RESIDUAL = float(
        os.environ.get("ASP_FG_MAX_RESIDUAL", _FG_REG_MAX_RESIDUAL_DEFAULT)
    )
except ValueError:
    FG_REG_MAX_RESIDUAL = _FG_REG_MAX_RESIDUAL_DEFAULT


def register_foreground_at_seam(  # noqa: C901
    warped_a: np.ndarray,
    warped_b: np.ndarray,
    fg_a: np.ndarray,
    fg_b: np.ndarray,
    seam_pos: int,
    axis: int = 0,
    taper_px: float = FG_REG_TAPER_PX,
    max_residual: float = FG_REG_MAX_RESIDUAL,
    smooth_sigma: float = FG_REG_SMOOTH_SIGMA,
    alpha_a: float = 0.5,
    alpha_b: float = 0.5,
    flow_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Re-pose the foreground of two canvas-aligned frames toward a shared target
    pose in a tapered band around the seam, so character body parts line up
    across the strip boundary.

    Parameters
    ----------
    warped_a, warped_b : (H, W, 3) uint8
        The two adjacent frames already warped into canvas coordinates (so the
        background is aligned).  ``a`` owns the canvas on one side of
        ``seam_pos``; ``b`` owns the other.
    fg_a, fg_b : (H, W) bool or uint8
        Foreground masks for each warped frame (True / >127 = foreground
        character pixel).  Flow / warp is applied to foreground only.
    seam_pos : int
        Canvas row (axis=0) or column (axis=1) of the ownership boundary.
    axis : int
        0 = vertical scroll (horizontal seam), 1 = horizontal scroll.
    taper_px : float
        Half-width of the seam band over which the warp magnitude tapers to 0.
    max_residual : float
        If the median foreground residual exceeds this, the animation gap is too
        large to warp safely → return inputs unchanged and signal ``warped=False``
        so the caller can fall back to single-pose-per-component (Eden-2006).
    smooth_sigma : float
        Gaussian sigma to smooth the residual flow before warping (suppresses
        per-pixel flow noise that would tear flat cel regions).
    alpha_a, alpha_b : float in [0, 1]
        Fraction of the flow applied to frame a and b respectively.
        alpha_a + alpha_b = 1.0 (enforced).
        Default 0.5/0.5 = symmetric midpoint warp.
        Global-reference strategy sets alpha proportional to temporal distance
        from the reference strip, so strips near the reference warp less and
        distant strips warp more — preventing drift accumulation.
    flow_override : (H, W, 2) float32 array, optional
        §2.10C: If provided, skip RAFT/DIS/ARAP flow estimation entirely and
        use this pre-computed dense flow field directly.  Intended for the
        HITL user-drawn flow tool where the user sketches displacement arrows
        in the SeamDiagnosticDialog and the GUI converts them to a dense field
        via ``_sparse_flow_to_dense()``.  The field is still tapered and
        Gaussian-smoothed before application.

    Returns
    -------
    (adj_a, adj_b, info) where ``adj_a``/``adj_b`` are the re-posed frames and
    ``info`` carries diagnostics (``warped`` bool, ``residual`` median px,
    ``fg_pixels`` in the seam band).
    """
    # Normalise alphas so they sum to 1.
    total_alpha = alpha_a + alpha_b
    if total_alpha > 0:
        alpha_a = alpha_a / total_alpha
        alpha_b = alpha_b / total_alpha
    else:
        alpha_a = alpha_b = 0.5
    h, w = warped_a.shape[:2]
    fa = (fg_a > 127) if fg_a.dtype != bool else fg_a
    fb = (fg_b > 127) if fg_b.dtype != bool else fg_b

    # Seam band: only the region within taper_px of the seam matters.
    taper = _seam_taper(h, w, seam_pos, taper_px, axis=axis)  # (h,w) in [0,1]
    band = taper > 0.0
    fa_band = fa & band
    fb_band = fb & band
    fg_union = fa_band | fb_band
    n_fg = int(fg_union.sum())

    # Dominant frame in the band = the one carrying the more complete character
    # instance (more foreground pixels).  Used by the single-pose fallback (A6).
    n_a = int(fa_band.sum())
    n_b = int(fb_band.sum())
    dominant = "a" if n_a >= n_b else "b"

    info = {
        "warped": False,
        "fallback": False,
        "residual": 0.0,
        "fg_pixels": n_fg,
        "dominant": dominant,
    }

    if n_fg < FG_REG_MIN_FG_PIXELS:
        # No meaningful character content crosses this seam — nothing to fix.
        return warped_a, warped_b, info

    # Dense flow a → b (camera already removed by canvas alignment ⇒ residual
    # foreground flow is the animation motion).
    # §2.10C: When a user-drawn flow override is provided, skip RAFT/DIS and
    # use it directly (still tapered and smoothed below).
    if flow_override is not None and flow_override.shape == (h, w, 2):
        flow = flow_override.astype(np.float32)
    else:
        # Compute flow only on the SEAM BAND CROP (±taper_px around seam_pos) so
        # RAFT/DIS sees the relevant region at higher relative resolution instead
        # of being diluted across the full canvas (which can be 2000+ px tall).
        # This also avoids VRAM pressure from full-canvas RAFT inference.
        if axis == 0:
            y0_crop = max(0, seam_pos - int(taper_px) - 16)
            y1_crop = min(h, seam_pos + int(taper_px) + 16)
            crop_a = warped_a[y0_crop:y1_crop, :]
            crop_b = warped_b[y0_crop:y1_crop, :]
            flow_crop = _dense_flow(crop_a, crop_b)
            flow = np.zeros((h, w, 2), dtype=np.float32)
            flow[y0_crop:y1_crop, :] = flow_crop
        else:
            x0_crop = max(0, seam_pos - int(taper_px) - 16)
            x1_crop = min(w, seam_pos + int(taper_px) + 16)
            crop_a = warped_a[:, x0_crop:x1_crop]
            crop_b = warped_b[:, x0_crop:x1_crop]
            flow_crop = _dense_flow(crop_a, crop_b)
            flow = np.zeros((h, w, 2), dtype=np.float32)
            flow[:, x0_crop:x1_crop] = flow_crop

    # A3 — ARAP Push + Regularise (full Sýkora 2009 algorithm).
    #
    # Push: per-cell SAD block matching on the seam-band crops gives each cell
    # an independent appearance-optimal displacement.  Critical for flat
    # cel-shaded regions where RAFT/DIS gradient-based flow is ambiguous.
    # Regularise: smooth the per-cell translations globally so adjacent cells
    # don't move in contradictory directions (prevents line-art bending).
    # Previously only Regularise was present; Push was omitted.
    try:
        crop_fg = fg_union[y0_crop:y1_crop, :] if axis == 0 else fg_union[:, x0_crop:x1_crop]

        if _ARAP_PUSH_ENABLED:
            # ARAP Push uses flow_crop as the initial estimate centre for block
            # matching.  If SGM ran above, flow_crop already has better initial
            # estimates for fg cells → Push refines from a better starting point.
            pushed = _arap_push(
                crop_a, crop_b, crop_fg, flow_crop, cell_size=16, search_range=24
            )
            if axis == 0:
                flow[y0_crop:y1_crop, :] = pushed
            else:
                flow[:, x0_crop:x1_crop] = pushed

        # LSD collinearity — pass the seam-band crop as the image source (faster
        # than full-canvas LSD, directly relevant to the active seam region).
        # image_offset shifts detected line coordinates from crop-space to
        # full-canvas cell-grid space so the constraint maps correctly.
        if axis == 0:
            flow = _arap_regularise(
                flow,
                fg_union,
                cell_size=16,
                n_iter=2,
                image=crop_a,
                image_offset=(y0_crop, 0),
            )
        else:
            flow = _arap_regularise(
                flow,
                fg_union,
                cell_size=16,
                n_iter=2,
                image=crop_a,
                image_offset=(0, x0_crop),
            )
    except Exception:
        if smooth_sigma > 0:
            flow[:, :, 0] = cv2.GaussianBlur(flow[:, :, 0], (0, 0), smooth_sigma)
            flow[:, :, 1] = cv2.GaussianBlur(flow[:, :, 1], (0, 0), smooth_sigma)

    # Magnitude of the residual on foreground pixels in the band.
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    fg_mag = mag[fg_union]
    med_residual = float(np.median(fg_mag)) if fg_mag.size else 0.0
    info["residual"] = round(med_residual, 2)

    if med_residual > max_residual:
        # Animation gap too large for a safe warp — signal the single-pose
        # fallback (A6): the caller should take the foreground in this seam band
        # from the dominant frame only, avoiding a two-pose double image.
        info["fallback"] = True
        return warped_a, warped_b, info

    if med_residual < 0.5:
        # Already aligned (near-static foreground at this seam) — skip.
        info["warped"] = False
        return warped_a, warped_b, info

    # Per-pixel warp weight: taper × foreground membership (warp fg only).
    w_a = (taper * fa.astype(np.float32))[:, :, None]  # (h,w,1)
    w_b = (taper * fb.astype(np.float32))[:, :, None]

    # Asymmetric re-posing toward the global reference pose.
    # flow is the vector from a→b.  In remap_by_displacement, disp is the
    # *source* offset: output[x] = input[x+disp].  So:
    #   disp_a = -alpha_a·flow  → samples frame_a content from x+alpha_a·flow,
    #                              which SHIFTS it by +alpha_a·flow (toward b).
    #   disp_b = +alpha_b·flow  → samples frame_b content from x-alpha_b·flow,
    #                              which SHIFTS it by -alpha_b·flow (toward a).
    # When alpha_a=alpha_b=0.5 this is the symmetric midpoint warp.
    # When alpha_a=1, alpha_b=0: frame a moves fully toward b (b is reference).
    disp_a = -alpha_a * flow * w_a
    disp_b = +alpha_b * flow * w_b

    # Valid-content masks: positions where the warped canvas has actual pixels.
    valid_a = warped_a.max(axis=2) > 0
    valid_b = warped_b.max(axis=2) > 0

    adj_a = _remap_by_displacement(warped_a, disp_a)
    adj_b = _remap_by_displacement(warped_b, disp_b)

    # Keep background untouched: restore original where the pixel was not warped
    # foreground (so only the character is re-posed, never the aligned bg).
    keep_a = ~(fa & band)
    keep_b = ~(fb & band)
    adj_a[keep_a] = warped_a[keep_a]
    adj_b[keep_b] = warped_b[keep_b]

    # Never introduce content where the original had none — the warp must not
    # extend canvas pixels into previously-empty boundary regions.
    adj_a[~valid_a] = 0
    adj_b[~valid_b] = 0

    # Post-warp verification: measure remaining foreground colour discrepancy
    # in a narrow strip centred on the seam.  A large post-warp diff means
    # the ARAP-regularised warp still left a significant pose mismatch that
    # will cause visible ghosting in the Laplacian blend zone.
    seam_strip_h = max(1, int(taper_px * 0.2))
    if axis == 0:
        y0_s = max(0, seam_pos - seam_strip_h)
        y1_s = min(h, seam_pos + seam_strip_h)
        strip_a = adj_a[y0_s:y1_s].astype(np.float32)
        strip_b = adj_b[y0_s:y1_s].astype(np.float32)
        fg_strip = fg_union[y0_s:y1_s]
    else:
        x0_s = max(0, seam_pos - seam_strip_h)
        x1_s = min(w, seam_pos + seam_strip_h)
        strip_a = adj_a[:, x0_s:x1_s].astype(np.float32)
        strip_b = adj_b[:, x0_s:x1_s].astype(np.float32)
        fg_strip = fg_union[:, x0_s:x1_s]

    diff_fg = float(np.abs(strip_a - strip_b).mean(axis=2)[fg_strip].mean()) if fg_strip.any() else 0.0
    info["post_warp_diff"] = round(diff_fg, 2)

    info["warped"] = True
    return adj_a, adj_b, info


__all__ = ["register_foreground_at_seam", "FG_REG_MAX_RESIDUAL"]
