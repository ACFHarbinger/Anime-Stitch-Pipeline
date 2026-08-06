"""
Photometric correction stages: BaSiC flat-field + radial vignetting.

Functions are standalone; pass the (optional) BaSiCWrapper instance via
``basic_wrapper``.  Returns the corrected frames plus the per-frame baseline
list captured during fitting (or None when unavailable).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

from backend.src.animation.core.stateless import _luma

try:
    import base as _batch_photo
    _BATCH_PHOTO = (
        getattr(_batch_photo, "__file__", None) is not None
        and hasattr(_batch_photo, "exposure")
        and hasattr(_batch_photo.exposure, "correct_vignetting")
    )
    if not _BATCH_PHOTO:
        raise ImportError("compiled base.exposure extension not available")
except ImportError:
    _batch_photo = None  # type: ignore[assignment]
    _BATCH_PHOTO = False


def _apply_basic(
    frames: List[np.ndarray],
    basic_wrapper,
) -> Tuple[List[np.ndarray], Optional[List[float]]]:
    """
    Apply BaSiC spatial flat-field correction only.

    For stitching we deliberately do NOT apply the per-frame dimming
    baseline (b_i) correction.  BaSiC's b_i makes each frame's mean
    brightness equal to the stack median, which destroys the natural
    inter-frame brightness continuity and causes colour discontinuities
    at seam boundaries.  The spatial flat-field F (vignette/shading)
    is the only correction we apply here; inter-frame colour differences
    are handled later by histogram matching inside _render_median.

    Returns
    -------
    (corrected_frames, baselines) — ``baselines`` is the per-frame dimming
    scalar list returned by BaSiC.fit (or None if the legacy API is used).
    """
    print("[Stitch]   Fitting BaSiC flat-field (spatial correction only)…")

    if hasattr(basic_wrapper, "fit"):
        # New API: fit to get flat_field, then apply WITHOUT per-frame b_i
        flat, dark, baselines = basic_wrapper.fit(frames, luma_only=True)
        baselines_list = baselines.tolist()
        dim_frames = [i for i, bi in enumerate(baselines) if bi < 0.75]
        if dim_frames:
            print(
                f"[Stitch]   Broadcast-dimming detected in frames: {dim_frames} "
                f"(b_i correction deferred to renderer)"
            )
        # Apply flat-field only (b=1.0 -> no per-frame brightness change)
        corrected = [
            basic_wrapper.apply_correction(img, baseline_override=1.0) for img in frames
        ]
        return corrected, baselines_list

    # Legacy fallback
    if hasattr(basic_wrapper, "process_batch"):
        return basic_wrapper.process_batch(frames), None
    basic_wrapper.estimate_profiles(frames)
    return [basic_wrapper.apply_correction(img) for img in frames], None


def _correct_vignetting(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Estimate and remove digital vignetting (radial darkening).
    Assumes the intensity follows I_obs = I_true * V(r), where V(r) is a radial gain.
    """
    if not frames:
        return frames
    H, W = frames[0].shape[:2]
    cx, cy = W / 2, H / 2

    # Create radial distance map
    yy, xx = np.mgrid[:H, :W]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = np.sqrt(cx**2 + cy**2)
    rr_norm = rr / r_max

    # Sample luma from a few frames to estimate k
    lumas = []
    for i in np.linspace(0, len(frames) - 1, 5, dtype=int):
        lumas.append(cv2.resize(_luma(frames[i]), (320, int(320 * H / W))))
    avg_luma = np.percentile(np.array(lumas), 75, axis=0)  # 75th percentile is robust

    # Fit 1D profile
    sh, sw = avg_luma.shape
    scx, scy = sw / 2, sh / 2
    syy, sxx = np.mgrid[:sh, :sw]
    srr = np.sqrt((sxx - scx) ** 2 + (syy - scy) ** 2) / np.sqrt(scx**2 + scy**2)

    # Fit quadratic: G(r) = 1 / (1 + k*r^2) -> inverse gain 1 + k*r^2
    n_samples = min(3000, sh * sw)
    indices = np.linspace(0, sh * sw - 1, n_samples, dtype=int)
    r_samples = srr.flatten()[indices]
    v_samples = avg_luma.flatten()[indices]

    def resid(p):
        # p = [baseline, k]
        return v_samples - (p[0] / (1 + p[1] * r_samples**2))

    res = least_squares(resid, [v_samples.max(), 0.05], bounds=([0, 0], [255, 0.5]))
    base_val, k_val = res.x

    # Soften the correction to avoid over-whitening corners
    k_val *= 0.7

    if k_val < 0.01:
        return frames  # No significant vignette detected

    # Very conservative correction
    k_val *= 0.4

    print(f"[Stitch]   Vignette correction applied (k={k_val:.4f}).")
    gain_map = (1.0 + k_val * rr_norm**2).astype(np.float32)

    # Apply correction
    corrected = []
    gm_h, gm_w = gain_map.shape[:2]

    for img in frames:
        h, w = img.shape[:2]
        curr_gain = gain_map
        if (h, w) != (gm_h, gm_w):
            curr_gain = cv2.resize(gain_map, (w, h), interpolation=cv2.INTER_LINEAR)

        if _BATCH_PHOTO:
            try:
                corrected.append(
                    np.asarray(_batch_photo.exposure.correct_vignetting(img, curr_gain))
                )
                continue
            except Exception:
                pass

        img_f = img.astype(np.float32)
        for c in range(3):
            img_f[..., c] *= curr_gain
        corrected.append(np.clip(img_f, 0, 255).astype(np.uint8))
    return corrected


__all__ = ["_apply_basic", "_correct_vignetting"]
