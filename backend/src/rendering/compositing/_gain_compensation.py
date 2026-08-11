"""Per-block and joint-solve luminance/gain compensation between frame pairs."""

from __future__ import annotations

import cv2
import numpy as np
from backend.src.constants import LUMINANCE_WEIGHTS

from ._flags import _JOINT_GAIN_SIGMA_G, _JOINT_GAIN_SIGMA_N
from ._native import BATCH_AVAILABLE, batch


def _blocks_gain_compensate(
    fa_zone: np.ndarray,
    fb_zone: np.ndarray,
    block_size: int = 32,
) -> np.ndarray:
    """§4.1: Spatial blocks gain compensation (S160).

    Divides the blend zone into *block_size* × *block_size* blocks and computes
    a per-block per-channel BGR gain ratio ``mean(fa_block) / mean(fb_block)``.
    A bilinear-resized (H, W, 3) gain map is applied to *fb_zone* to correct
    strip-level banding that global scalar gain normalisation cannot handle.
    Gain is clamped to [0.5, 2.0] before application.  Blocks where the
    fb-channel mean is < 1.0 (near-black) use gain=1.0 (safe no-op).

    Returns a uint8 copy of fb_zone with the spatial gain applied.
    """
    if fa_zone.size == 0 or fb_zone.size == 0:
        return fb_zone.copy()
    if BATCH_AVAILABLE and hasattr(batch, "compositing") and hasattr(
        batch.compositing, "blocks_gain_compensate_pair"
    ):
        try:
            return np.asarray(
                batch.compositing.blocks_gain_compensate_pair(fa_zone, fb_zone, block_size)
            )
        except Exception:
            pass
    H, W = fb_zone.shape[:2]
    bs = max(1, block_size)
    n_rows = max(1, (H + bs - 1) // bs)
    n_cols = max(1, (W + bs - 1) // bs)
    gain_grid = np.ones((n_rows, n_cols, 3), dtype=np.float32)
    for ri in range(n_rows):
        r0 = ri * bs
        r1 = min(r0 + bs, H)
        for ci in range(n_cols):
            c0 = ci * bs
            c1 = min(c0 + bs, W)
            fa_b = fa_zone[r0:r1, c0:c1].astype(np.float32)
            fb_b = fb_zone[r0:r1, c0:c1].astype(np.float32)
            for ch in range(3):
                m_fb = fb_b[:, :, ch].mean()
                m_fa = fa_b[:, :, ch].mean()
                gain_grid[ri, ci, ch] = (m_fa / m_fb) if m_fb >= 1.0 else 1.0
    gain_map = cv2.resize(gain_grid, (W, H), interpolation=cv2.INTER_LINEAR)
    gain_map = np.clip(gain_map, 0.5, 2.0)
    result = np.clip(fb_zone.astype(np.float32) * gain_map, 0, 255).astype(np.uint8)
    return result


def _blocks_lum_compensate(
    fa_zone: np.ndarray,
    fb_zone: np.ndarray,
    block_size: int = 32,
) -> np.ndarray:
    """§4.4: LAB L-channel blocks gain compensation (S160).

    Like ``_blocks_gain_compensate`` but uses the LAB L-channel ratio as a
    scalar gain applied uniformly to all BGR channels.  This avoids the colour
    cast that per-channel BGR gain can produce when any channel's mean is near
    zero in a block.  Gain clamped to [0.5, 2.0].

    Returns a uint8 copy of fb_zone with the spatial L-gain applied.
    """
    if fa_zone.size == 0 or fb_zone.size == 0:
        return fb_zone.copy()
    if BATCH_AVAILABLE and hasattr(batch, "compositing") and hasattr(
        batch.compositing, "blocks_lum_compensate_pair"
    ):
        try:
            return np.asarray(
                batch.compositing.blocks_lum_compensate_pair(fa_zone, fb_zone, block_size)
            )
        except Exception:
            pass
    H, W = fb_zone.shape[:2]
    fa_lab = cv2.cvtColor(fa_zone, cv2.COLOR_BGR2LAB).astype(np.float32)
    fb_lab = cv2.cvtColor(fb_zone, cv2.COLOR_BGR2LAB).astype(np.float32)
    bs = max(1, block_size)
    n_rows = max(1, (H + bs - 1) // bs)
    n_cols = max(1, (W + bs - 1) // bs)
    gain_grid = np.ones((n_rows, n_cols), dtype=np.float32)
    for ri in range(n_rows):
        r0 = ri * bs
        r1 = min(r0 + bs, H)
        for ci in range(n_cols):
            c0 = ci * bs
            c1 = min(c0 + bs, W)
            m_fb_l = float(fb_lab[r0:r1, c0:c1, 0].mean())
            m_fa_l = float(fa_lab[r0:r1, c0:c1, 0].mean())
            gain_grid[ri, ci] = m_fa_l / max(1.0, m_fb_l)
    gain_map = cv2.resize(gain_grid, (W, H), interpolation=cv2.INTER_LINEAR)
    gain_map = np.clip(gain_map, 0.5, 2.0)
    result = np.clip(
        fb_zone.astype(np.float32) * gain_map[:, :, np.newaxis], 0, 255
    ).astype(np.uint8)
    return result


def _adaptive_gain_clamp(ref_lum: float, frame_lum: float) -> float:
    """Scalar luminance gain with §1.4B continuous adaptive clip (S24).

    Clamp width linearly interpolates between ±26 % (pure-black scene) and
    ±14 % (pure-white scene): ``clamp_width = 0.26 - 0.12 × (ref_lum / 255)``.
    This removes the discontinuity at the S18 ref_lum=80 threshold while
    keeping the same endpoints ([0.86, 1.14] at ref=255).  Scalar (not
    per-channel) to avoid hue shift.
    """
    clamp_width = 0.26 - 0.12 * (ref_lum / 255.0)
    lo = 1.0 - clamp_width
    hi = 1.0 + clamp_width
    return float(np.clip(ref_lum / max(frame_lum, 1.0), lo, hi))


def _bg_gain_unclamped(
    ref_lum: float,
    frame_lum: float,
    override_threshold: float = 0.20,
) -> float:
    """§1.4C — Background-only gain that lifts the clamp when needed.

    When ``_adaptive_gain_clamp`` would reduce the ideal correction by more
    than ``override_threshold`` (default 20%), return the raw ideal gain so
    that background pixels receive the full correction.  For small deviations
    (clamp cut ≤ 20%) the clamped value is returned unchanged.

    Background pixels tolerate aggressive correction because:
    1. They are large uniform regions — clipping is less visible.
    2. Character skin tones (which motivated the clamp) are already excluded
       from the bg-only application site.

    Parameters
    ----------
    ref_lum : float
        Reference median background luminance (scene median, [0, 255]).
    frame_lum : float
        This frame's median background luminance.
    override_threshold : float
        Fraction of the ideal correction that the clamp may cut before we
        bypass it.  0.20 means the clamp may reduce the ideal by at most 20 %.
    """
    if frame_lum <= 0.0:
        return 1.0
    ideal = ref_lum / frame_lum
    clamped = _adaptive_gain_clamp(ref_lum, frame_lum)
    if ideal == 0.0:
        return clamped
    cut = abs(ideal - clamped) / abs(ideal)
    return ideal if cut > override_threshold else clamped


def _joint_gain_solve(
    warped_frames: list[np.ndarray],
    warped_bg: list[np.ndarray | None],
    sigma_n: float = 10.0,
    sigma_g: float = 0.1,
    robust: bool = False,
) -> np.ndarray:
    """§3.1: Brown-Lowe (2007) joint multi-image gain compensation.

    One linear least-squares system over ALL overlapping frame pairs'
    bg-only mean luminance, with a gain-prior term regularizing each frame's
    gain toward 1.0 (prevents the degenerate all-shrink/all-grow solutions a
    plain pairwise-ratio system admits). Minimizes:

        sum_(i,j overlapping) n_ij/sigma_n^2 * (g_i*I_ij - g_j*I_ji)^2
      + sum_i n_i/sigma_g^2 * (1 - g_i)^2

    where I_ij is frame i's bg-only mean luminance in its overlap with frame
    j, n_ij the overlap pixel count, and n_i the total overlap-pixel count
    frame i participates in (matches cv2's own detail::GainCompensator).

    Returns a length-N scalar gain array clamped to [0.5, 2.0]. Frames with
    no valid overlap get gain=1.0 (untouched).
    """
    N = len(warped_frames)
    if N < 2:
        return np.ones(N, dtype=np.float64)

    overlaps: list[tuple[int, int, float, float, int]] = []
    for i in range(N):
        bg_i = warped_bg[i]
        if bg_i is None:
            continue
        has_i = warped_frames[i].max(axis=2) > 10
        for j in range(i + 1, N):
            bg_j = warped_bg[j]
            if bg_j is None:
                continue
            shared = bg_i & bg_j
            if not shared.any():
                continue
            has_j = warped_frames[j].max(axis=2) > 10
            sel = shared & has_i & has_j
            n_px = int(sel.sum())
            if n_px < 200:
                continue
            mean_i = float(
                warped_frames[i][sel].astype(np.float64).dot(LUMINANCE_WEIGHTS).mean()
            )
            mean_j = float(
                warped_frames[j][sel].astype(np.float64).dot(LUMINANCE_WEIGHTS).mean()
            )
            if mean_i < 1.0 or mean_j < 1.0:
                continue
            overlaps.append((i, j, mean_i, mean_j, n_px))

    if not overlaps:
        return np.ones(N, dtype=np.float64)

    if robust and len(overlaps) >= 4:
        # A pair can be contaminated by foreground leakage, a hard seam, or a
        # transient animation cel.  Its log luminance ratio is an exposure
        # observation only when it agrees with the other overlap observations.
        # Reject isolated outliers with a robust MAD threshold; retain all
        # observations when the sample has no useful spread estimate.
        log_ratios = np.asarray(
            [np.log(max(mean_i, 1e-6) / max(mean_j, 1e-6)) for _, _, mean_i, mean_j, _ in overlaps],
            dtype=np.float64,
        )
        center = float(np.median(log_ratios))
        mad = float(np.median(np.abs(log_ratios - center)))
        tolerance = max(0.10, 3.0 * 1.4826 * mad)
        filtered = [
            item
            for item, ratio in zip(overlaps, log_ratios, strict=False)
            if abs(float(ratio) - center) <= tolerance
        ]
        # Keep the original system well-constrained.  This is a diagnostic
        # safeguard, not permission to solve a nearly empty graph.
        if len(filtered) >= max(3, N - 1) and _overlap_graph_connected(filtered, N):
            overlaps = filtered

    inv_sigma_n2 = 1.0 / (sigma_n * sigma_n)
    inv_sigma_g2 = 1.0 / (sigma_g * sigma_g)

    A = np.zeros((N, N), dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)
    frame_npx = np.zeros(N, dtype=np.float64)

    for i, j, mean_i, mean_j, n_px in overlaps:
        A[i, i] += n_px * (mean_i**2) * inv_sigma_n2
        A[j, j] += n_px * (mean_j**2) * inv_sigma_n2
        A[i, j] -= n_px * mean_i * mean_j * inv_sigma_n2
        A[j, i] -= n_px * mean_i * mean_j * inv_sigma_n2
        frame_npx[i] += n_px
        frame_npx[j] += n_px

    for i in range(N):
        w = frame_npx[i] * inv_sigma_g2
        A[i, i] += w
        b[i] += w  # prior target g_i = 1.0

    try:
        gains = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        gains = np.ones(N, dtype=np.float64)

    return np.clip(gains, 0.5, 2.0)


def _overlap_graph_connected(
    overlaps: list[tuple[int, int, float, float, int]],
    num_frames: int,
) -> bool:
    """Return whether every frame remains connected by overlap observations."""
    if num_frames <= 1:
        return True
    graph = [set() for _ in range(num_frames)]
    for i, j, *_ in overlaps:
        if 0 <= i < num_frames and 0 <= j < num_frames:
            graph[i].add(j)
            graph[j].add(i)
    seen = {0}
    stack = [0]
    while stack:
        for neighbour in graph[stack.pop()]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return len(seen) == num_frames


def _apply_joint_gain_solve(
    warped_frames: list[np.ndarray],
    warped_bg: list[np.ndarray | None],
    robust: bool = False,
) -> list[np.ndarray]:
    """Solve §3.1's joint gain system and apply each frame's scalar gain to
    its own bg-only pixels (matches _normalize_single_frame's convention —
    foreground/character pixels are never touched by a background exposure
    correction)."""
    gains = _joint_gain_solve(
        warped_frames,
        warped_bg,
        sigma_n=_JOINT_GAIN_SIGMA_N,
        sigma_g=_JOINT_GAIN_SIGMA_G,
        robust=robust,
    )
    result = []
    for i, wf in enumerate(warped_frames):
        gain = float(gains[i])
        if warped_bg[i] is None or abs(gain - 1.0) < 1e-6:
            result.append(wf.copy())
            continue
        out = wf.astype(np.float32)
        bg_sel = warped_bg[i] & (wf.max(axis=2) > 10)
        out[bg_sel] = np.clip(out[bg_sel] * gain, 0, 255)
        result.append(out.astype(np.uint8))
    return result


def _equalize_warped_gains(
    warped_frames: list[np.ndarray],
    block_size: int = 32,
) -> list[np.ndarray]:
    """§4.10: Equalize inter-frame luminance before seam finding.

    Sequential pairwise gain compensation: frame 0 is the reference; each
    subsequent frame is corrected to match its (already-corrected) predecessor.
    Only pixels where BOTH adjacent frames have valid content (max channel > 0)
    are altered — black/transparent border fill regions are left untouched.
    """
    if len(warped_frames) < 2:
        return [f.copy() for f in warped_frames]
    result: list[np.ndarray] = [warped_frames[0].copy()]
    for i in range(1, len(warped_frames)):
        prev = result[i - 1]
        curr = warped_frames[i]
        has_prev = prev.max(axis=2) > 0
        has_curr = curr.max(axis=2) > 0
        both_valid = has_prev & has_curr
        if not both_valid.any():
            result.append(curr.copy())
            continue
        corrected = _blocks_gain_compensate(prev, curr, block_size=block_size)
        out = curr.copy()
        out[both_valid] = corrected[both_valid]
        result.append(out)
    return result


__all__ = [
    "_blocks_gain_compensate",
    "_blocks_lum_compensate",
    "_adaptive_gain_clamp",
    "_bg_gain_unclamped",
    "_joint_gain_solve",
    "_overlap_graph_connected",
    "_apply_joint_gain_solve",
    "_equalize_warped_gains",
]
