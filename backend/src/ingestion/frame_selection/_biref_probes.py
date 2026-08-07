"""BiRefNet probe masks for camera-displacement/pose-similarity in ``smart_select_frames``.

Extracted from ``smart_select_frames`` step 2 as a standalone, individually
testable function -- pure code motion, no logic change. Samples a handful of
frames (first/quarter/mid/three-quarter/last), runs BiRefNet on each, and
accumulates an intersection background mask (``_TWO_CHANNEL_SELECT``) and a
union foreground mask (used by Pass 2's ``_fg_center_diff`` fallback when
DINOv2 is unavailable).
"""

from __future__ import annotations

import contextlib
import os

import cv2
import numpy as np

try:
    import torch as _torch
except ImportError:
    _torch = None  # type: ignore[assignment]

# Two-channel selection using BiRefNet background masks for cleaner camera
# displacement estimates.  Disabled by default — BiRefNet overhead is significant
# and the approach changed frame timing in ways that hurt GT-SSIM.
_TWO_CHANNEL_SELECT = os.environ.get("ASP_TWO_CHANNEL_SELECT", "0") != "0"


def _compute_biref_probe_masks(
    N: int,
    thumbs: list[np.ndarray],
    frames_paths: list[str],
    pw: float,
    dinov2_features: np.ndarray | None,
    verbose: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run BiRefNet on 5 probe frames and return (bg_thumb_mask, fg_thumb_mask).

    ``bg_thumb_mask`` (intersection of probe bg masks) is only populated when
    ``_TWO_CHANNEL_SELECT`` is enabled. ``fg_thumb_mask`` (union of probe fg
    masks) is only populated when Pass 2 is active (``pw > 0``) and needs it
    as the ``_fg_center_diff`` fallback (DINOv2 unavailable). Returns
    ``(None, None)`` when neither is needed, BiRefNet is unavailable, or no
    probe frame loaded successfully.
    """
    bg_thumb_mask: np.ndarray | None = None
    fg_thumb_mask: np.ndarray | None = None

    needs_biref_probes = _TWO_CHANNEL_SELECT or (pw > 0 and dinov2_features is None)
    if not needs_biref_probes:
        return bg_thumb_mask, fg_thumb_mask

    try:
        from backend.src.models.wrappers.birefnet_wrapper import (
            BiRefNetWrapper as _BiRefNet,
        )  # §3.14 lazy

        _biref = _BiRefNet()
        _probe_idxs = sorted({0, N // 4, N // 2, 3 * N // 4, N - 1})
        _th_shape = thumbs[0].shape[:2]
        _bg_accum = np.ones(_th_shape, dtype=np.float32)
        _fg_accum = np.zeros(_th_shape, dtype=np.float32)
        _n_ok = 0
        for _pi in _probe_idxs:
            _full = cv2.imread(frames_paths[_pi])
            if _full is None:
                continue
            _mk = _biref.get_mask(_full)
            _th = thumbs[_pi].shape
            _bg_prob = cv2.resize(
                (_mk > 127).astype(np.float32),
                (_th[1], _th[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            _fg_prob = 1.0 - _bg_prob
            _bg_accum = np.minimum(_bg_accum, _bg_prob)
            _fg_accum = np.maximum(_fg_accum, _fg_prob)
            _n_ok += 1
        with contextlib.suppress(Exception):
            _biref.offload()
        del _biref
        import gc as _gc
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        if _n_ok > 0:
            if _TWO_CHANNEL_SELECT:
                bg_cov = float(_bg_accum.mean())
                if verbose:
                    print(
                        f"  [SmartSelect] BiRefNet bg mask: {bg_cov:.0%}, {_n_ok} probes"
                    )
                bg_thumb_mask = _bg_accum if bg_cov >= 0.10 else None
            if pw > 0:
                fg_cov = float(_fg_accum.mean())
                if verbose:
                    print(
                        f"  [SmartSelect] BiRefNet fg mask: {fg_cov:.0%} fg coverage, "
                        f"{_n_ok} probes"
                    )
                fg_thumb_mask = _fg_accum if fg_cov >= 0.05 else None
    except Exception as _e:
        if verbose:
            print(
                f"  [SmartSelect] BiRefNet unavailable ({_e}); "
                "using central-crop pose diff"
            )

    return bg_thumb_mask, fg_thumb_mask


__all__ = ["_compute_biref_probe_masks", "_TWO_CHANNEL_SELECT"]
