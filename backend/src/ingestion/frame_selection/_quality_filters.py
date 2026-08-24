"""Pre/post-selection frame quality filters (§1.2D, §1.2E, §1.46, §1.2B).

Each filter drops frames that carry no useful new information for the
stitch pipeline: near-static triplets, blurry/artifacted frames, low-contrast
(flash/whiteout) frames, and near-duplicate selected frames.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from ._native import _BATCH_FSEL, _batch

# §1.2D: Temporal variance filter — drops interior frames whose mean per-pixel
# variance across the (i-1, i, i+1) thumbnail triplet is below this threshold.
# Thumbnails are in [0, 1] float32.  Default 0.0 = disabled.
# Suggested value for enabling: 1e-3 (ASP_TEMPORAL_VAR_THRESH=0.001).
try:
    _TEMPORAL_VAR_THRESH = float(os.environ.get("ASP_TEMPORAL_VAR_THRESH", "0.0"))
except ValueError:
    _TEMPORAL_VAR_THRESH = 0.0

# §1.2E: Blur/artifact frame pre-rejection — Laplacian variance gate.
# Interior frames whose 64×64 thumbnail Laplacian variance (uint8 scale) is
# below the threshold are dropped before hold detection.  First/last always
# kept.  Default 0.0 = disabled.  Suggested value: ASP_BLUR_REJECT_THRESH=50.0.
try:
    _BLUR_REJECT_THRESH = float(os.environ.get("ASP_BLUR_REJECT_THRESH", "0.0"))
except ValueError:
    _BLUR_REJECT_THRESH = 0.0

# §1.46: Low-contrast frame pre-rejection — pixel std gate.
# Interior frames whose grayscale thumbnail std (in [0,255] scale) is below
# the threshold are dropped before hold detection.  Flash/whiteout/bloom frames
# have near-zero std — LoFTR and phase correlation have no texture to anchor on.
# Distinct from §1.2E (Laplacian blur): a sharp flash frame has high Laplacian
# but zero matchable texture.  First/last always kept.
# Default 0.0 = disabled.  Suggested value: ASP_CONTRAST_THRESH=15.0.
try:
    _CONTRAST_REJECT_THRESH = float(os.environ.get("ASP_CONTRAST_THRESH", "0.0"))
except ValueError:
    _CONTRAST_REJECT_THRESH = 0.0

# §1.2B: Near-duplicate post-filter for the selected frame list.
# Consecutive selected frames with mean grayscale diff < threshold are
# collapsed (first of each near-dup run kept; last frame always retained).
# Default 0.0 = disabled.  Enable with e.g. ASP_NEAR_DUP_LUMA=5.0.
try:
    _NEAR_DUP_LUMA = float(os.environ.get("ASP_NEAR_DUP_LUMA", "0.0"))
except ValueError:
    _NEAR_DUP_LUMA = 0.0

# Stage-1 shot-boundary pre-filter. A scene cut has no stable geometric
# relation to the preceding shot, so retain one contiguous shot rather than
# handing an impossible cross-cut edge to matching. Default 0.0 = disabled.
try:
    _SHOT_BOUNDARY_THRESH = float(os.environ.get("ASP_SHOT_BOUNDARY_THRESH", "0.0"))
except ValueError:
    _SHOT_BOUNDARY_THRESH = 0.0


def _filter_shot_boundaries(
    thumbs: list[np.ndarray],
    paths: list[str],
    threshold: float,
) -> tuple[list[np.ndarray], list[str], int]:
    """Keep the longest contiguous shot when a histogram discontinuity occurs.

    Bhattacharyya distance between 32-bin grayscale histograms is stable across
    threads and does not require a model. Ties retain the earlier shot, keeping
    results deterministic. A zero threshold preserves the current selector
    exactly.
    """
    if threshold <= 0.0 or len(thumbs) < 3:
        return list(thumbs), list(paths), 0

    starts = [0]
    previous = cv2.normalize(
        cv2.calcHist([thumbs[0]], [0], None, [32], [0.0, 1.0]), None
    )
    for i, thumb in enumerate(thumbs[1:], start=1):
        current = cv2.normalize(
            cv2.calcHist([thumb], [0], None, [32], [0.0, 1.0]), None
        )
        if cv2.compareHist(previous, current, cv2.HISTCMP_BHATTACHARYYA) >= threshold:
            starts.append(i)
        previous = current
    starts.append(len(thumbs))
    if len(starts) == 2:
        return list(thumbs), list(paths), 0

    segments = list(zip(starts, starts[1:], strict=False))
    start, end = max(segments, key=lambda segment: segment[1] - segment[0])
    return list(thumbs[start:end]), list(paths[start:end]), len(thumbs) - (end - start)


def _temporal_variance_filter(
    thumbs: list[np.ndarray],
    paths: list[str],
    sigma_threshold: float = 1e-3,
) -> tuple[list[np.ndarray], list[str], int]:
    """§1.2D — Drop near-static interior frames using temporal variance across triplets.

    For each interior frame *i* (not first or last) compute the mean per-pixel
    variance across the consecutive thumbnail triplet
    (thumbs[i-1], thumbs[i], thumbs[i+1]).  Thumbnails are in [0, 1] float32.
    If the mean variance is below ``sigma_threshold``, the frame contributes no
    new motion information and is dropped.

    **Why this is different from the other near-dup filters:**
    - §1.2A / §1.2C operate on displacement edges — they require a non-zero
      match displacement to detect statics.
    - §1.2B compares *selected* frames post-selection.
    - §1.2D acts on the raw thumbnail sequence pre-selection, catching frames
      where both the camera and the character are stationary, regardless of
      whether the matching step would later produce a near-zero edge for them.

    First and last frames are always kept to preserve canvas extent.

    Parameters
    ----------
    thumbs:
        Grayscale float32 thumbnails in [0, 1].  Length N.
    paths:
        Corresponding frame file paths.  Length N.
    sigma_threshold:
        Mean per-pixel variance (in [0, 1]² space) below which a frame is
        considered static.  Default 1e-3 (approx. std ≈ 0.032, i.e., ~8 lum
        units of inter-frame noise amplitude).

    Returns
    -------
    (filtered_thumbs, filtered_paths, n_dropped)
    """
    N = len(thumbs)
    if N < 3 or sigma_threshold <= 0.0:
        return list(thumbs), list(paths), 0

    if _BATCH_FSEL:
        try:
            u8 = [np.ascontiguousarray(
                      np.clip(t * 255, 0, 255).astype(np.uint8)
                      if t.dtype != np.uint8 else t)
                  for t in thumbs]
            # C++ sigma_threshold is in [0,1]² space (same as Python)
            ft, fp = _batch.frame_selection.temporal_variance_filter(
                u8, list(paths), sigma_threshold)
            n_dropped = N - len(ft)
            # Recover original float32 thumb objects by path matching
            path_to_thumb = {p: t for p, t in zip(paths, thumbs, strict=False)}
            kept_thumbs = [path_to_thumb.get(p, u8[i]) for i, p in enumerate(fp)]
            return kept_thumbs, list(fp), n_dropped
        except Exception:
            pass

    keep = [True] * N
    for i in range(1, N - 1):
        a, b, c = thumbs[i - 1], thumbs[i], thumbs[i + 1]
        h = min(a.shape[0], b.shape[0], c.shape[0])
        w = min(a.shape[1], b.shape[1], c.shape[1])
        stack = np.stack([a[:h, :w], b[:h, :w], c[:h, :w]], axis=0)
        if float(np.mean(np.var(stack, axis=0))) < sigma_threshold:
            keep[i] = False

    n_dropped = keep.count(False)
    return (
        [t for t, k in zip(thumbs, keep, strict=False) if k],
        [p for p, k in zip(paths, keep, strict=False) if k],
        n_dropped,
    )


def _reject_blurry_frames(
    thumbs: list[np.ndarray],
    paths: list[str],
    blur_threshold: float,
    thumb_size: int = 64,
) -> tuple[list[np.ndarray], list[str], int]:
    """§1.2E — Drop interior frames with Laplacian variance below blur_threshold.

    Resizes each grayscale float32 thumbnail to ``thumb_size``×``thumb_size``,
    converts to uint8, and measures the variance of the Laplacian.  Sharp frames
    produce high Laplacian variance; blurry or severe-artifact frames have low
    values because high-frequency edge energy is suppressed.

    First and last frames are always kept to preserve canvas extent.

    Parameters
    ----------
    thumbs:
        Grayscale float32 thumbnails in [0, 1].  Length N.
    paths:
        Corresponding frame file paths.  Length N.
    blur_threshold:
        Laplacian variance floor (uint8 scale, 0–255²).  Interior frames below
        this value are dropped.  0.0 = disabled (no frames dropped).
        Suggested: 50.0 for anime key-frames.
    thumb_size:
        Edge size for internal resize before Laplacian (default 64).

    Returns
    -------
    (filtered_thumbs, filtered_paths, n_dropped)
    """
    if blur_threshold <= 0.0 or len(thumbs) < 3:
        return list(thumbs), list(paths), 0

    keep = [True] * len(thumbs)
    for i in range(1, len(thumbs) - 1):
        small = cv2.resize(
            thumbs[i], (thumb_size, thumb_size), interpolation=cv2.INTER_AREA
        )
        gray_u8 = (np.clip(small, 0.0, 1.0) * 255).astype(np.uint8)
        lap_var = float(cv2.Laplacian(gray_u8, cv2.CV_32F).var())
        if lap_var < blur_threshold:
            keep[i] = False

    n_dropped = keep.count(False)
    return (
        [t for t, k in zip(thumbs, keep, strict=False) if k],
        [p for p, k in zip(paths, keep, strict=False) if k],
        n_dropped,
    )


def _reject_low_contrast_frames(
    thumbs: list[np.ndarray],
    paths: list[str],
    contrast_threshold: float,
) -> tuple[list[np.ndarray], list[str], int]:
    """§1.46: Drop interior frames with pixel std below *contrast_threshold*.

    Measures contrast as the standard deviation of the grayscale thumbnail in
    the [0, 255] scale.  Near-uniform frames (flash panels, whiteout effects,
    bloom overexposure, fade-to-white transitions) produce std ≈ 0–10 lum
    units.  Such frames offer no reliable texture for LoFTR keypoint matching
    or phase-correlation peaks.

    This complements §1.2E (Laplacian blur): a sharp white-flash frame scores
    high Laplacian (crisp edges wherever the flash meets non-white content) but
    contributes zero matchable internal texture to the frame interior.

    First and last frames are always kept to preserve the canvas extent.

    Parameters
    ----------
    thumbs:
        Grayscale float32 thumbnails in [0, 1].  Length N.
    paths:
        Corresponding frame file paths.  Length N.
    contrast_threshold:
        Pixel std floor in [0, 255] units.  Interior frames below this value
        are dropped.  0.0 = disabled (returns inputs unchanged).
        Suggested: 15.0 for anime (flash frame std ≈ 0–8, normal frame ≈ 30–80).

    Returns
    -------
    (filtered_thumbs, filtered_paths, n_dropped)
    """
    if contrast_threshold <= 0.0 or len(thumbs) < 3:
        return list(thumbs), list(paths), 0

    keep = [True] * len(thumbs)
    for i in range(1, len(thumbs) - 1):
        gray_255 = np.clip(thumbs[i], 0.0, 1.0) * 255.0
        contrast = float(np.std(gray_255))
        if contrast < contrast_threshold:
            keep[i] = False

    n_dropped = keep.count(False)
    return (
        [t for t, k in zip(thumbs, keep, strict=False) if k],
        [p for p, k in zip(paths, keep, strict=False) if k],
        n_dropped,
    )


def _near_dup_luma_filter(
    selected_thumbs: list[np.ndarray],
    selected_paths: list[str],
    threshold: float = 5.0,
) -> list[str]:
    """
    §1.2B: Drop consecutive near-duplicate frames from the selected list.

    Compares each consecutive pair in the ALREADY-SELECTED frame list using
    mean absolute grayscale difference on thumbnail images.  When the diff is
    below ``threshold`` (luma units, 0–255 scale) the later frame is dropped —
    it adds negligible new content to the canvas and only introduces noise in
    bundle adjustment and the temporal median.

    The first frame is always kept.  The last frame is always retained even if
    it is a near-duplicate of the preceding frame (preserves full canvas extent).

    Set ``threshold=0.0`` to disable (returns ``selected_paths`` unchanged).
    Default is 5.0 luma units — catches camera steps well below the ~10-luma
    noise floor while leaving legitimate slow-scroll frames intact.
    """
    if threshold <= 0.0 or len(selected_paths) <= 2:
        return selected_paths

    if _BATCH_FSEL:
        try:
            # C++ expects uint8; thumbs from _load_thumbs_parallel are float32 [0,1]
            u8 = [
                (np.clip(t * 255, 0, 255).astype(np.uint8) if t.dtype != np.uint8 else t)
                for t in selected_thumbs
            ]
            _, kept_paths = _batch.frame_selection.near_dup_luma_filter(
                u8, list(selected_paths), float(threshold)
            )
            return list(kept_paths)
        except Exception:
            pass

    def _to_gray_f32(t: np.ndarray) -> np.ndarray:
        """Return float32 luma from 2D grayscale or 3D BGR thumb."""
        if t.ndim == 2:
            return t.astype(np.float32)
        return cv2.cvtColor(t, cv2.COLOR_BGR2GRAY).astype(np.float32)

    keep: list[int] = [0]
    # Determine threshold scale: float32 [0,1] thumbs need threshold in [0,1];
    # uint8 thumbs compare in [0,255] space directly.
    _is_float_thumb = selected_thumbs[0].dtype != np.uint8
    _thr_scaled = threshold / 255.0 if _is_float_thumb else threshold
    for i in range(1, len(selected_paths)):
        prev = keep[-1]
        g_cur = _to_gray_f32(selected_thumbs[i])
        g_prev = _to_gray_f32(selected_thumbs[prev])
        # Resize to common dims when thumbnails differ
        if g_cur.shape != g_prev.shape:
            h = min(g_cur.shape[0], g_prev.shape[0])
            w = min(g_cur.shape[1], g_prev.shape[1])
            g_cur = cv2.resize(g_cur, (w, h))
            g_prev = cv2.resize(g_prev, (w, h))
        diff = float(np.abs(g_cur - g_prev).mean())
        if diff >= _thr_scaled:
            keep.append(i)

    # Always include last frame
    last = len(selected_paths) - 1
    if keep[-1] != last:
        keep.append(last)

    return [selected_paths[i] for i in keep]


__all__ = [
    "_TEMPORAL_VAR_THRESH",
    "_BLUR_REJECT_THRESH",
    "_CONTRAST_REJECT_THRESH",
    "_NEAR_DUP_LUMA",
    "_temporal_variance_filter",
    "_reject_blurry_frames",
    "_reject_low_contrast_frames",
    "_near_dup_luma_filter",
]
