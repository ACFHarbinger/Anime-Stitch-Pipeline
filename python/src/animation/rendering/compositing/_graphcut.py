"""GraphCut global multi-image seam composite (§4.2), gated by _GRAPHCUT_SEAM."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from ._flags import _GC_FEATHER_PX, _GRAPHCUT_SEAM
from ._gain_compensation import _blocks_gain_compensate
from ._native import BATCH_AVAILABLE, batch
from ._seam_cache import _extract_seam_crops


def _feather_gc_boundaries(
    result: np.ndarray,
    ownership_masks: List[np.ndarray],
    warped_frames: List[np.ndarray],
    feather_px: int = 96,
) -> np.ndarray:
    """§3.33 / §1.3 GraphCut post-mortem fix (2026-07-27).

    Two changes from the original per-column linear-ramp version, targeting
    the two identified wiring gaps (the theory — graph-cut with hard t-links
    — was never in question; see roadmap §1.3):

    1. **Distance-transform feathering** instead of a per-column vertical
       ramp. A GraphCut boundary can meander in any direction (unlike the
       DP path's horizontal-strip zones); a per-column "last owned row"
       ramp only approximates a horizontal seam and produces a visibly
       wrong blend wherever the cut runs diagonally or doubles back.
       ``cv2.distanceTransform`` on each ownership mask gives the true
       Euclidean distance to the boundary in every direction, so the
       blend follows the actual 2D seam shape.
    2. **Local per-boundary blocks-gain correction** within the blend
       band, before blending — the pre-seam global gain equalization
       (§4.10) corrects frame-to-frame luminance overall, but leaves
       residual local mismatch at the specific boundary a graph cut
       chose (which needn't align with where the global correction was
       most accurate). Reuses the same `_blocks_gain_compensate` the DP
       path already relies on.

    Only pixels where both frames have content (non-black) are blended;
    all-black pixels are skipped so gap-fill work is not undone.
    """
    N = len(ownership_masks)
    if N < 2 or feather_px <= 0:
        return result
    out = result.copy()
    for i in range(N - 1):
        own_i = (ownership_masks[i] > 127).astype(np.uint8)
        own_next = (ownership_masks[i + 1] > 127).astype(np.uint8)
        if not own_i.any() or not own_next.any():
            continue

        content_i = warped_frames[i].max(axis=2) > 0
        content_next = warped_frames[i + 1].max(axis=2) > 0

        # Signed distance to the boundary: positive = deep inside frame i's
        # territory, negative = deep inside frame i+1's. Near-zero at the cut.
        dist_i = cv2.distanceTransform(own_i * 255, cv2.DIST_L2, 5)
        dist_next = cv2.distanceTransform(own_next * 255, cv2.DIST_L2, 5)
        signed_dist = dist_i - dist_next

        in_band = (np.abs(signed_dist) <= feather_px) & content_i & content_next
        if not in_band.any():
            continue

        # Local gain correction over the blend band's bounding box, before
        # blending — corrects fb (frame i+1) to match fa's (frame i) local
        # photometry right at this boundary.
        ys, xs = np.where(in_band)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        fa_zone = warped_frames[i][y0:y1, x0:x1]
        fb_zone = warped_frames[i + 1][y0:y1, x0:x1]
        fb_corrected = _blocks_gain_compensate(fa_zone, fb_zone, block_size=32)

        src_i = warped_frames[i].astype(np.float32)
        src_next_local = warped_frames[i + 1].astype(np.float32)
        src_next_local[y0:y1, x0:x1] = fb_corrected.astype(np.float32)

        alpha = np.clip(0.5 + signed_dist / (2.0 * feather_px), 0.0, 1.0)
        alpha3 = alpha[:, :, None]
        blended = (
            alpha3 * src_i + (1.0 - alpha3) * src_next_local
        ).clip(0, 255).astype(np.uint8)
        out[in_band] = blended[in_band]
    return out


def _execute_graphcut_composite(
    warped_norm: List[np.ndarray],
    warped_bg: List[Optional[np.ndarray]],
    canvas: np.ndarray,
    H: int,
    W: int,
    N: int,
    boundaries: np.ndarray,
    seam_post_diffs: dict,
    seam_single_pose: dict,
    seam_meta_out: Optional[dict],
) -> np.ndarray:
    # Seam-estimation downscale (cv2.Stitcher runs GraphCut at seam_est_resol
    # ≈ 0.1 MPix; full-resolution min-cut over N frames is O(hours) on tall
    # canvases).  Find seams on a ≤ _GC_SEAM_EST_MPIX proxy, then upscale the
    # ownership masks back to canvas size with nearest-neighbour.
    _GC_SEAM_EST_MPIX = 0.4
    _scale = min(1.0, (_GC_SEAM_EST_MPIX * 1e6 / max(H * W, 1)) ** 0.5)
    if _scale < 1.0:
        _sw, _sh = max(8, int(W * _scale)), max(8, int(H * _scale))
        _gc_frames = [
            np.ascontiguousarray(
                cv2.resize(warped_norm[i], (_sw, _sh), interpolation=cv2.INTER_AREA)
            )
            for i in range(N)
        ]
    else:
        _gc_frames = [np.ascontiguousarray(warped_norm[i]) for i in range(N)]
    _gc_masks = [
        (f.max(axis=2) > 0).astype(np.uint8) * 255 for f in _gc_frames
    ]
    _gc_corners = [(0, 0)] * N
    print(
        f"[Stitch]   §4.2 GraphCut seam (global, {N} frames, "
        f"est scale={_scale:.2f})..."
    )
    _ownership = batch.seam.graphcut_seam_find(
        _gc_frames, _gc_masks, _gc_corners
    )
    if _scale < 1.0:
        _ownership = [
            cv2.resize(o, (W, H), interpolation=cv2.INTER_NEAREST)
            for o in _ownership
        ]
        _gc_frames = [np.ascontiguousarray(warped_norm[i]) for i in range(N)]
    result = canvas.copy()
    for i in range(N):
        own = _ownership[i] > 127
        src = warped_norm[i]
        has_content = src.max(axis=2) > 0
        _apply_gc = own & has_content
        if warped_bg[i] is not None:
            _apply_gc = _apply_gc & (~warped_bg[i])
        result[_apply_gc] = src[_apply_gc]

    _gc_black = result.max(axis=2) == 0
    if _gc_black.any():
        for _gcwn in warped_norm:
            _gc_fill = _gc_black & (_gcwn.max(axis=2) > 0)
            if _gc_fill.any():
                result[_gc_fill] = _gcwn[_gc_fill]
                _gc_black = result.max(axis=2) == 0
            if not _gc_black.any():
                break

    if _GC_FEATHER_PX > 0:
        result = _feather_gc_boundaries(result, _ownership, _gc_frames, feather_px=_GC_FEATHER_PX)

    if seam_meta_out is not None:
        seam_meta_out.update(
            {
                "boundaries": (
                    boundaries.tolist()
                    if hasattr(boundaries, "tolist")
                    else list(boundaries)
                ),
                "seam_post_diffs": dict(seam_post_diffs),
                "seam_single_pose": dict(seam_single_pose),
                "seam_crops": _extract_seam_crops(result, boundaries),
            }
        )
    print("[Stitch]   GraphCut composite done.")
    return result


def _try_global_seam_composite(
    warped_norm: List[np.ndarray],
    warped_bg: List[Optional[np.ndarray]],
    canvas: np.ndarray,
    H: int,
    W: int,
    N: int,
    boundaries: np.ndarray,
    seam_post_diffs: dict,
    seam_single_pose: dict,
    seam_meta_out: Optional[dict],
) -> Optional[np.ndarray]:
    if _GRAPHCUT_SEAM and BATCH_AVAILABLE and N >= 2:
        try:
            return _execute_graphcut_composite(
                warped_norm, warped_bg, canvas, H, W, N, boundaries,
                seam_post_diffs, seam_single_pose, seam_meta_out
            )
        except Exception as _gc_exc:
            print(
                f"[Stitch]   §4.2 GraphCut seam failed ({_gc_exc}), "
                "falling back to DP blend."
            )

    return None


__all__ = [
    "_feather_gc_boundaries",
    "_execute_graphcut_composite",
    "_try_global_seam_composite",
]
