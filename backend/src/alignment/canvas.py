"""
Canvas geometry, frame loading and normalization, and SCANS-mode fallback.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from asp_backend.core.stateless import _largest_valid_rect, _trim_dark_border
from backend.src.constants import CANVAS_MAX_DIM
from backend.src.errors import CanvasError
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import base as batch
    if (
        getattr(batch, "__file__", None) is None
        or not hasattr(batch, "canvas")
    ):
        raise ImportError("compiled base.canvas extension not available")
    _BATCH_CANVAS = True
except ImportError:
    batch = None
    _BATCH_CANVAS = False


def _load_frames(paths: list[str]) -> list[np.ndarray]:
    """Read frames from disk, trim broadcast dark borders, drop unreadables."""
    frames = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            logger.warning(f"[Stitch] Warning: could not read '{p}' — skipping.")
            continue
        img = _trim_dark_border(img)
        frames.append(img)
    return frames


def _normalise_widths(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Resize every frame to match the width of the first frame (Lanczos)."""
    target_w = frames[0].shape[1]
    out = []
    for img in frames:
        h, w = img.shape[:2]
        if w != target_w:
            nh = int(round(h * target_w / w))
            img = cv2.resize(img, (target_w, nh), interpolation=cv2.INTER_LANCZOS4)
        out.append(img)
    return out


def _compute_canvas(
    frames: list[np.ndarray],
    affines: list[np.ndarray],
) -> tuple[int, int, np.ndarray]:
    """
    Compute canvas dimensions and the global translation offset T_global
    that maps all warped corners into positive coordinates.

    Returns (canvas_h, canvas_w, T_global) where T_global is a (2,) array
    of (tx, ty) offsets to add to every affine's translation column.
    """
    if _BATCH_CANVAS:
        try:
            affines_f32 = [np.ascontiguousarray(a, dtype=np.float32) for a in affines]
            shapes = [f.shape[:2] for f in frames]
            canvas_h, canvas_w, sx, sy = batch.canvas.compute_canvas(affines_f32, shapes)
            canvas_w = min(canvas_w, CANVAS_MAX_DIM)
            canvas_h = min(canvas_h, CANVAS_MAX_DIM)
            return canvas_h, canvas_w, np.array([sx, sy], dtype=np.float32)
        except Exception as _e:
            logger.debug(f"[Stitch] batch.canvas.compute_canvas fallback: {_e}")

    _corners_list: list[np.ndarray] = []
    for i, img in enumerate(frames):
        h, w = img.shape[:2]
        M = affines[i]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        warped = (M[:2, :2] @ corners.T + M[:2, 2:3]).T
        _corners_list.append(warped)

    all_corners = np.vstack(_corners_list)
    min_xy = all_corners.min(axis=0)
    max_xy = all_corners.max(axis=0)

    T_global = -min_xy
    canvas_w = int(np.ceil(max_xy[0] - min_xy[0]))
    canvas_h = int(np.ceil(max_xy[1] - min_xy[1]))

    canvas_w = min(canvas_w, CANVAS_MAX_DIM)
    canvas_h = min(canvas_h, CANVAS_MAX_DIM)
    return canvas_h, canvas_w, T_global


def _detect_scroll_axis(affines: list[np.ndarray]) -> str:
    """
    Classify the scroll direction of a set of affines as:
    - 'vertical'    : ty_range >> tx_range (normal case)
    - 'horizontal'  : tx_range >> ty_range (test20 pattern)
    - 'diagonal'    : both ty and tx significant (test7 pattern)
    - 'none'        : all frames co-located
    """
    if _BATCH_CANVAS:
        try:
            affines_f32 = [np.ascontiguousarray(a, dtype=np.float32) for a in affines]
            return batch.canvas.detect_scroll_axis(affines_f32)
        except Exception as _e:
            logger.debug(f"[Stitch] batch.canvas.detect_scroll_axis fallback: {_e}")

    tys = np.array([float(a[1, 2]) for a in affines])
    txs = np.array([float(a[0, 2]) for a in affines])
    ty_range = float(tys.max() - tys.min())
    tx_range = float(txs.max() - txs.min())
    total = ty_range + tx_range
    if total < 1.0:
        return "none"
    if tx_range > 0 and ty_range / max(tx_range, 1.0) < 0.1:
        return "horizontal"
    if ty_range > 0 and tx_range / max(ty_range, 1.0) > 0.3:
        return "diagonal"
    return "vertical"


def _crop_to_valid(canvas: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Crop the canvas to remove empty borders.

    Adaptive strategy:
    - If ≥80% of the bounding-box area is valid (typical vertical scroll), use
      the bounding-box crop: tight but cheap, preserves sparse straggler rows at
      the bottom of the last frame (prevents historical test4 height-loss regression).
    - If <80% of the bounding-box area is valid (diagonal scroll whose valid region
      is a parallelogram), use _largest_valid_rect: finds the maximum inscribed
      rectangle, removing the black corner triangles without cropping real content.
    """
    if valid_mask.max() == 0:
        return canvas

    row_has_content = np.any(valid_mask > 0, axis=1)
    col_has_content = np.any(valid_mask > 0, axis=0)

    row_idx = np.where(row_has_content)[0]
    col_idx = np.where(col_has_content)[0]

    if row_idx.size == 0 or col_idx.size == 0:
        return canvas

    r0, r1 = int(row_idx[0]), int(row_idx[-1]) + 1
    c0, c1 = int(col_idx[0]), int(col_idx[-1]) + 1

    bb_sub = valid_mask[r0:r1, c0:c1] > 0
    valid_ratio = float(bb_sub.sum()) / max(float(bb_sub.size), 1.0)

    if valid_ratio < 0.80:
        # relocated: from asp_backend.core.stateless import _largest_valid_rect

        xv0, yv0, xv1, yv1 = _largest_valid_rect(valid_mask > 0)
        if (xv1 - xv0) * (yv1 - yv0) > 0:
            logger.debug(
                f"[Stitch]   crop_to_valid (inner-rect, ratio={valid_ratio:.2f}): "
                f"({xv0},{yv0}) → ({xv1},{yv1})"
            )
            return canvas[yv0:yv1, xv0:xv1]

    logger.info(f"[Stitch]   crop_to_valid: ({c0},{r0}) → ({c1},{r1})")
    return canvas[r0:r1, c0:c1]


def _telea_fill_gaps(canvas: np.ndarray, gap_mask: np.ndarray) -> np.ndarray:
    """§1.7B: Fill residual black border pixels with cv2.INPAINT_TELEA (S23).

    Designed as a fast fallback when diffusion inpainting fails or is unavailable.
    Reliable for gaps < 50px wide; larger regions may show visible smearing.
    """
    if not gap_mask.any():
        return canvas
    if _BATCH_CANVAS:
        try:
            return batch.canvas.telea_fill_gaps(
                np.ascontiguousarray(canvas),
                np.ascontiguousarray(gap_mask.astype(np.uint8)),
                3,
            )
        except Exception as _e:
            logger.debug(f"[Stitch] batch.canvas.telea_fill_gaps fallback: {_e}")
    return cv2.inpaint(
        canvas, gap_mask.astype(np.uint8), inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )


def _try_opencv_stitch(
    frames: list[np.ndarray],
    mode: int,
    registration_resol: float,
) -> tuple[int, np.ndarray | None]:
    """One OpenCV Stitcher attempt. status 0 is OK."""
    cv2.ocl.setUseOpenCL(False)
    try:
        stitcher = cv2.Stitcher_create(mode=mode)  # type: ignore[attr-defined]
    except AttributeError:
        stitcher = cv2.createStitcher(mode == 1)  # type: ignore[attr-defined]
    stitcher.setRegistrationResol(registration_resol)
    status, pano = stitcher.stitch(frames)
    return int(status), pano if status == cv2.Stitcher_OK else None


def _reuse_simple_stitch(output_path: str) -> Image.Image | None:
    """If the bench already wrote opencv_stitch.png, reuse it instead of crashing.

    ``run()`` writes to ``output/panorama_stages/run_output.png`` while the
    simple stitch lives at ``output/opencv_stitch.png`` — check this
    directory and its parent.
    """
    from pathlib import Path

    dest = Path(output_path).expanduser().resolve()
    candidates = [
        dest.parent / "opencv_stitch.png",
        dest.parent.parent / "opencv_stitch.png",
    ]
    for sibling in candidates:
        if not sibling.is_file() or sibling.resolve() == dest:
            continue
        import shutil

        shutil.copy2(sibling, dest)
        logger.warning(
            "[Stitch] SCANS stitcher failed; reused existing %s as %s",
            sibling,
            dest.name,
        )
        return Image.open(dest)
    return None


def _scan_stitch_fallback(
    frames: list[np.ndarray],
    output_path: str,
) -> Image.Image:
    """
    Fall back to OpenCV SCANS mode when the main pipeline cannot find
    enough edges.  Mirrors _merge_images_opencv(stitcher_mode=1).

    OpenCV's stitcher output usually has staircase-shaped black edges.
    After stitching we crop to the largest fully-covered inner rectangle so
    the final image contains no black boundary pixels.

    status=1 (ERR_NEED_MORE_IMGS) is common on hold-heavy / low-overlap
    smart-selected sets even when the bench's earlier simple stitch
    succeeded. Retry a few registrations, then reuse opencv_stitch.png
    rather than raising CanvasError and aborting the dataset.
    """
    logger.info("[Stitch] FALLBACK: using OpenCV SCANS mode.")
    usable = [f for f in frames if isinstance(f, np.ndarray) and f.size > 0]
    attempts: list[tuple[str, list[np.ndarray], int, float]] = []
    if len(usable) >= 2:
        attempts.append(("scans-0.8", usable, 1, 0.8))
        attempts.append(("scans-0.4", usable, 1, 0.4))
        if len(usable) > 8:
            idx = np.linspace(0, len(usable) - 1, 8, dtype=int)
            attempts.append(
                ("scans-sub8", [usable[int(i)] for i in idx], 1, 0.6)
            )
        if len(usable) >= 3:
            mid = len(usable) // 2
            attempts.append(
                ("scans-ends", [usable[0], usable[mid], usable[-1]], 1, 0.6)
            )
        attempts.append(("pano-0.6", usable[: min(12, len(usable))], 0, 0.6))

    last_status = -1
    for name, fr, mode, resol in attempts:
        last_status, pano = _try_opencv_stitch(fr, mode, resol)
        logger.info(
            "[Stitch]   SCANS attempt %s n=%d status=%s",
            name,
            len(fr),
            last_status,
        )
        if pano is None:
            continue
        valid_mask = pano.max(axis=2) > 0
        x0, y0, x1, y1 = _largest_valid_rect(valid_mask)
        if (x1 - x0) > 0 and (y1 - y0) > 0:
            pano = pano[y0:y1, x0:x1]
            logger.info(f"[Stitch] SCANS inner-rect crop: ({x0},{y0}) → ({x1},{y1})")
        rgb = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
        out = Image.fromarray(rgb)
        out.save(output_path)
        return out

    reused = _reuse_simple_stitch(output_path)
    if reused is not None:
        return reused
    raise CanvasError(f"SCANS fallback failed (status={last_status}).")


def _panorama_stitch_fallback(
    frames: list[np.ndarray],
    output_path: str,
) -> Image.Image:
    """§1.3B — PANORAMA stitcher attempted before SCANS for affine-validation failures.

    PANORAMA mode (mode=0) handles scale and rotation that the translation-only
    canvas model rejects.  Raises RuntimeError on failure so the caller can fall
    through to the SCANS path.
    """
    logger.info("[Stitch] §1.3B: Trying PANORAMA stitcher before SCANS.")

    pano = None
    if _BATCH_CANVAS:
        try:
            ok, result = batch.canvas.panorama_stitch_fallback(
                [np.ascontiguousarray(f) for f in frames]
            )
            if ok:
                pano = result
            else:
                raise CanvasError(
                    "PANORAMA stitcher failed (C++); caller should try SCANS."
                )
        except CanvasError:
            raise
        except Exception as _e:
            logger.debug(f"[Stitch] batch.canvas.panorama_stitch_fallback fallback: {_e}")

    if pano is None:
        # Python fallback path
        cv2.ocl.setUseOpenCL(False)
        try:
            stitcher = cv2.Stitcher_create(mode=0)  # type: ignore[attr-defined] # cv2-stub gap
        except AttributeError:
            stitcher = cv2.createStitcher(False)  # type: ignore[attr-defined] # cv2-stub gap
        stitcher.setRegistrationResol(0.8)
        status, pano = stitcher.stitch(frames)
        if status != cv2.Stitcher_OK:
            raise CanvasError(
                f"PANORAMA stitcher failed (status={status}); caller should try SCANS."
            )

    # relocated: from asp_backend.core.stateless import _largest_valid_rect

    valid_mask = pano.max(axis=2) > 0
    x0, y0, x1, y1 = _largest_valid_rect(valid_mask)
    if (x1 - x0) > 0 and (y1 - y0) > 0:
        pano = pano[y0:y1, x0:x1]
        logger.info(f"[Stitch] PANORAMA inner-rect crop: ({x0},{y0}) → ({x1},{y1})")

    rgb = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
    out = Image.fromarray(rgb)
    out.save(output_path)
    return out


def find_optimal_sequence(  # noqa: C901
    ref_path: str,
    candidates: list[str],
    min_inliers: int = 30,
    max_overlap: float = 0.85,
) -> list[str]:
    """Find the longest coherent panorama sequence while dropping redundant frames.

    Uses SIFT feature matching to build a directed overlap graph, then extracts
    the longest non-redundant path from *ref_path* outward. Frames with overlap
    exceeding *max_overlap* are considered redundant and pruned.

    Args:
        ref_path: Absolute path to the anchor (reference) frame. Always included
            as the first element of the returned sequence.
        candidates: Ordered list of candidate frame paths to evaluate. Paths that
            cannot be read (e.g. missing files) are silently skipped.
        min_inliers: Minimum RANSAC homography inliers required to consider two
            frames as overlapping. Pairs below this threshold are treated as
            non-adjacent.
        max_overlap: Maximum allowed overlap ratio [0, 1] between consecutive
            retained frames. Frames whose overlap with the previous kept frame
            exceeds this value are pruned as redundant.

    Returns:
        Ordered list of frame paths forming the longest coherent sequence,
        starting with *ref_path*. May be shorter than *candidates* if frames
        are pruned for redundancy or insufficient overlap.
    """
    # 1. Feature Extraction (SIFT)
    sift = cv2.SIFT_create(nfeatures=1200)  # type: ignore[attr-defined] # cv2-stub gap

    def get_feats(p):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None, None
        h, w = img.shape
        if w > 1024:
            img = cv2.resize(img, (1024, int(h * 1024 / w)))
        kp, des = sift.detectAndCompute(img, None)
        return kp, des, img.shape

    feats = {}
    for p in [ref_path] + list(candidates):
        if p not in feats:
            kp, des, shape = get_feats(p)
            if kp is not None:
                feats[p] = (kp, des, shape[0], shape[1])

    if ref_path not in feats:
        return []

    sequence = [ref_path]
    used = {ref_path}
    bf = cv2.BFMatcher()

    def find_optimal_extension(query_path, pool, direction="forward"):
        q_kp, q_des, q_h, q_w = feats[query_path]
        best_p = None
        max_dist = -1.0

        for p in pool:
            if p in used or p not in feats:
                continue
            t_kp, t_des, t_h, t_w = feats[p]

            matches = bf.knnMatch(q_des, t_des, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if len(good) < min_inliers:
                continue

            src_pts = np.float32([q_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([t_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None or mask is None:
                continue

            inliers = int(mask.sum())
            if inliers < min_inliers:
                continue

            dist = np.sqrt(M[0, 2] ** 2 + M[1, 2] ** 2)
            if dist < 0.15 * max(q_h, q_w):
                continue

            if dist > max_dist:
                max_dist = dist
                best_p = p

        return best_p

    # Expand Forward
    while True:
        nxt = find_optimal_extension(sequence[-1], candidates, "forward")
        if nxt:
            sequence.append(nxt)
            used.add(nxt)
        else:
            break

    # Expand Backward
    while True:
        prev = find_optimal_extension(sequence[0], candidates, "backward")
        if prev:
            sequence.insert(0, prev)
            used.add(prev)
        else:
            break

    return sequence


__all__ = [
    "_compute_canvas",
    "_crop_to_valid",
    "_detect_scroll_axis",
    "_load_frames",
    "_normalise_widths",
    "_panorama_stitch_fallback",
    "_scan_stitch_fallback",
    "_telea_fill_gaps",
    "find_optimal_sequence",
    "logger",
]
