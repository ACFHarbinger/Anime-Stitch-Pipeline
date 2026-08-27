"""Wallpaper-mode background plate builder (Slice 1, #427).

Temporal-median background plate over the non-hero frames with the hero-cel
footprint excluded, Brown-Lowe joint gain compensation, and Tier-1 classical
inpainting (Telea) for the residual no-background hole.  Tier-2 generative
outpaint is a no-op hook here for Slice 1 — the trigger lives in
_aspect_framer (#429), which consumes :attr:`BackgroundPlate.void_ratio`.

The plate is built on the shared canvas convention used by the rest of the
rendering stack: every frame is warped to a common ``(H, W)`` canvas via its
affine, and per-frame background masks say which pixels are known-not-
foreground.  A canvas pixel accepts a sample from frame *i* only when that
frame covers it and its background mask marks it background.  The hero
frame's own footprint (``hero_cel_footprint``) is additionally excluded from
that single frame's samples — so the character's hero pose is never averaged
into the plate, while the non-hero frames (whose character stands elsewhere)
fill the hero-shaped region from the times the background was visible.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from asp_backend.rendering.compositing._flags import (
    _JOINT_GAIN_SIGMA_G,
    _JOINT_GAIN_SIGMA_N,
)
from asp_backend.rendering.compositing._gain_compensation import (
    _apply_joint_gain_solve,
    _joint_gain_solve,
)
from asp_backend.rendering.compositing._normalization import _warp_inputs

logger = logging.getLogger(__name__)

# Tier-1 inpainting is classical (Telea).  Tier-2 generative outpaint engages
# only when the aspect-framer void is large (see #429); the Slice-1 default
# below mirrors the roadmap's 25% canvas hard cap.
_MAX_INPAINT_FRACTION = 0.25
_TIER2_MIN_VOID_FRACTION = 0.10

# The Python nanmedian path is a dev-scale fallback; the production <500ms
# Tier-1 budget is delivered by the native fast path (the same one
# `_render_median` uses) that #430 wires up.  Until then, cap the temporal
# axis with even decimation so long clips (asp_test97 = 90 frames) stay
# usable — a robust median is unaffected by dropping the middle of a sorted
# stack, so even subsampling keeps the background estimate sound.
_MAX_TEMPORAL_SAMPLES = 40

Generator = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class BackgroundPlate:
    """Result of :func:`build_background_plate`.

    All arrays are canvas-space ``(H, W)`` / ``(H, W, 3)``.
    """

    plate: np.ndarray
    valid_mask: np.ndarray
    sample_count: np.ndarray
    gains: np.ndarray
    inpainted_mask: np.ndarray
    void_ratio: float


def _contribution_masks(
    warped_frames: list[np.ndarray],
    warped_bg: list[np.ndarray | None],
    hero_cel_footprint: np.ndarray,
    hero_frame_idx: int | None,
) -> list[np.ndarray]:
    """Per-frame canvas-space masks of pixels that may feed the plate.

    A pixel contributes from frame *i* iff the frame has content there
    (``max(channel) > 10``, matching ``_apply_joint_gain_solve``'s own
    convention) and the background mask marks it background (or is absent).
    Only the hero frame additionally drops its own ``hero_cel_footprint`` —
    the non-hero frames show true background there and must fill it.
    """
    masks: list[np.ndarray] = []
    for i, wf in enumerate(warped_frames):
        presence = wf.max(axis=2) > 10
        if warped_bg[i] is not None:
            bg_bool = (
                warped_bg[i] > 127
                if warped_bg[i].dtype == np.uint8
                else warped_bg[i].astype(bool)
            )
        else:
            bg_bool = presence
        m = bg_bool & presence
        if i == hero_frame_idx:
            m = m & ~hero_cel_footprint
        masks.append(m)
    return masks


def _temporal_median_plate(
    warped_frames: list[np.ndarray],
    contribution: list[np.ndarray],
    band_rows: int | None = None,
    max_samples: int = _MAX_TEMPORAL_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Chunked temporal nanmedian over the contributing samples.

    Bounds memory by processing the canvas in horizontal bands; the sample
    stack for one band is ``(N, band_h, W, 3)`` float32 — frame-major so each
    frame only ever writes its own slice (a pixel/slot-per-frame layout
    avoids cross-frame slot corruption).  When more than ``max_samples``
    frames are supplied, they are evenly decimated first (median-robust; see
    ``_MAX_TEMPORAL_SAMPLES``).  Pixels with no contributing sample stay
    black with ``sample_count == 0``.
    """
    N = len(warped_frames)
    if N > max_samples:
        idx = np.unique(np.linspace(0, N - 1, max_samples).astype(np.int64))
        warped_frames = [warped_frames[i] for i in idx]
        contribution = [contribution[i] for i in idx]
        N = len(warped_frames)
    H, W = warped_frames[0].shape[:2]
    if band_rows is None:
        band_rows = max(8, min(64, int(4_000_000 / max(1, N))))
    plate = np.zeros((H, W, 3), dtype=np.uint8)
    count = np.zeros((H, W), dtype=np.int32)
    for y0 in range(0, H, band_rows):
        y1 = min(y0 + band_rows, H)
        samples = np.full((N, y1 - y0, W, 3), np.nan, dtype=np.float32)
        for i, wf in enumerate(warped_frames):
            m = contribution[i][y0:y1]
            if not m.any():
                continue
            samples[i][m] = wf[y0:y1][m].astype(np.float32)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            with np.errstate(invalid="ignore"):
                med = np.nanmedian(samples, axis=0)
                valid = ~np.isnan(med[..., 0])
                cnt = np.sum(~np.isnan(samples[..., 0]), axis=0)
        if valid.any():
            plate[y0:y1][valid] = np.clip(med[valid], 0, 255).astype(np.uint8)
            count[y0:y1][valid] = cnt[valid].astype(np.int32)
    return plate, count


def _inpaint_holes(
    plate: np.ndarray,
    hole_mask: np.ndarray,
    radius: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Telea-inpaint the no-background hole.  Returns (filled, inpainted_mask)."""
    if not hole_mask.any():
        return plate.copy(), np.zeros_like(hole_mask)
    filled = cv2.inpaint(plate, (hole_mask.astype(np.uint8) * 255), radius, cv2.INPAINT_TELEA)
    return filled, hole_mask


def _generative_outpaint(
    plate: np.ndarray,
    void_mask: np.ndarray,
    *,
    generator: Generator | None = None,
    min_void_fraction: float = _TIER2_MIN_VOID_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Tier-2 generative outpaint hook (Slice 1: no-op unless given a generator).

    Called by the aspect framer (#429) when a target-aspect void exceeds
    ``min_void_fraction``.  ``generator`` is ``(plate, void_mask) -> (out, mask)``.
    """
    if generator is None or not void_mask.any():
        return plate, np.zeros_like(void_mask)
    ratio = float(void_mask.sum()) / float(void_mask.size)
    if ratio < min_void_fraction:
        return plate, np.zeros_like(void_mask)
    return generator(plate, void_mask)


def build_background_plate(
    frames: list[np.ndarray],
    affines: list[np.ndarray],
    bg_masks: list[np.ndarray | None],
    hero_cel_footprint: np.ndarray,
    canvas_size: tuple[int, int],
    *,
    hero_frame_idx: int | None = None,
    robust: bool = False,
    inpaint_radius: int = 5,
    max_inpaint_fraction: float = _MAX_INPAINT_FRACTION,
    band_rows: int | None = None,
    max_samples: int = _MAX_TEMPORAL_SAMPLES,
) -> BackgroundPlate:
    """Build the background plate for a wallpaper-mode composite.

    Parameters
    ----------
    frames:
        BGR uint8 frames to sample the plate from.  Pass the *non-hero*
        frames (the ones where the character stands away from the hero cel);
        the hero frame itself is optional and only useful via
        ``hero_frame_idx``.
    affines:
        Per-frame ``2x3`` frame→canvas affine transforms.
    bg_masks:
        Per-frame frame-space background masks (uint8, True = background);
        ``None`` falls back to frame presence.
    hero_cel_footprint:
        ``(H, W)`` bool — the hero-cel region on the canvas.  Excluded from
        the hero frame's samples so its pose never enters the plate.
    canvas_size:
        ``(H, W)`` of the canvas.
    hero_frame_idx:
        Index of the hero frame inside ``frames``, if it is included; only
        that frame's footprint is excluded.  ``None`` (default) means
        ``frames`` are all non-hero and the footprint is purely informational.
    robust:
        Passed to the joint gain solve (rejects isolated outlier overlap
        observations).
    inpaint_radius:
        Telea radius for the residual hole.
    max_inpaint_fraction:
        Hard cap on the classical-inpaint area; a hole above this is still
        inpainted classically for Slice 1 but leaves ``void_ratio`` high so
        #429 can route to Tier-2 generative outpaint.
    band_rows:
        Horizontal band height for the chunked median (None = auto).
    max_samples:
        Temporal decimation cap for the median stack (see
        ``_MAX_TEMPORAL_SAMPLES``).
    """
    if not frames or len(frames) != len(affines) or len(frames) != len(bg_masks):
        raise ValueError("frames, affines, bg_masks must be non-empty and same length")
    if hero_frame_idx is not None and not 0 <= hero_frame_idx < len(frames):
        raise ValueError(f"hero_frame_idx {hero_frame_idx} out of range for {len(frames)} frames")
    H, W = canvas_size
    if hero_cel_footprint.shape != (H, W):
        raise ValueError(
            f"hero_cel_footprint shape {hero_cel_footprint.shape} != canvas {canvas_size}"
        )

    warped_frames, warped_bg = _warp_inputs(frames, affines, bg_masks, H, W, len(frames))
    gains = _joint_gain_solve(
        warped_frames,
        warped_bg,
        sigma_n=_JOINT_GAIN_SIGMA_N,
        sigma_g=_JOINT_GAIN_SIGMA_G,
        robust=robust,
    )
    corrected = _apply_joint_gain_solve(warped_frames, warped_bg, robust=robust)

    contribution = _contribution_masks(corrected, warped_bg, hero_cel_footprint, hero_frame_idx)
    plate, sample_count = _temporal_median_plate(
        corrected, contribution, band_rows=band_rows, max_samples=max_samples
    )

    scene = np.zeros((H, W), dtype=bool)
    for wf in warped_frames:
        scene |= wf.max(axis=2) > 10
    hole = (sample_count == 0) & scene
    filled, inpainted_mask = _inpaint_holes(plate, hole, inpaint_radius)
    void_ratio = float(hole.sum()) / float(H * W)

    if void_ratio > max_inpaint_fraction:
        logger.warning(
            "plate void %.1f%% exceeds Tier-1 cap %.1f%% — aspect framer may route to "
            "Tier-2 generative outpaint",
            void_ratio * 100,
            max_inpaint_fraction * 100,
        )

    return BackgroundPlate(
        plate=filled,
        valid_mask=sample_count > 0,
        sample_count=sample_count,
        gains=gains,
        inpainted_mask=inpainted_mask,
        void_ratio=void_ratio,
    )


__all__ = [
    "BackgroundPlate",
    "build_background_plate",
    "_generative_outpaint",
    "_contribution_masks",
    "_temporal_median_plate",
]