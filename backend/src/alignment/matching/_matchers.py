"""Standalone single-strategy frame-pair matchers.

Each function independently estimates a pairwise transform between two
frames; ``_match_pair`` (in ``_pairwise.py``) tries them in a fallback
chain, from most to least precise.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from asp_backend.core.stateless import _highpass, _luma
from backend.src.constants import MIN_TEMPLATE_SCORE, PC_CONF_THRESHOLD


def _template_match(
    img_i: np.ndarray,
    img_j: np.ndarray,
    m_i: Optional[np.ndarray],
    m_j: Optional[np.ndarray],
    H: int,
    slice_h: int = 256,
    max_search_frac: float = 0.8,
    direction_sign: int = 0,
    max_dy_frac: float = 0.70,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Bidirectional template match: handles both upward and downward pans.

    direction_sign: +1 = only search downward (dy > 0),
                    -1 = only search upward (dy < 0),
                     0 = search both (default).
    max_dy_frac:    reject any dy whose |dy| > H * max_dy_frac; enforces a
                    minimum overlap fraction between adjacent frames so that
                    near-zero-overlap false matches (uniform background regions
                    found at the far edge of the search window) are discarded.
    """
    g_i = _luma(img_i)
    g_j = _luma(img_j)
    search_h = int(H * max_search_frac)
    max_dy = H * max_dy_frac

    best_dy = 0.0
    best_conf = 0.0

    # Config A: search i_bottom in j_top  -> gives dy > 0 (downward pan)
    # Config B: search i_top in j_bottom  -> gives dy < 0 (upward pan)
    all_configs = [
        {"strip_y": H - slice_h, "roi_y": 0, "sign": 1},
        {"strip_y": 0, "roi_y": H - search_h, "sign": -1},
    ]
    # Filter configs by requested direction
    test_configs = (
        [c for c in all_configs if c["sign"] == direction_sign]
        if direction_sign != 0
        else all_configs
    )

    for config in test_configs:
        strip_y = config["strip_y"]
        roi_y0 = max(0, config["roi_y"])
        roi_y1 = min(H, roi_y0 + search_h)

        tmpl = g_i[strip_y : strip_y + slice_h, :].copy()
        mask = m_i[strip_y : strip_y + slice_h, :] if m_i is not None else None
        roi = g_j[roi_y0:roi_y1, :]

        if roi.shape[0] < slice_h:
            continue
        if tmpl.std() < 2.0:
            continue

        try:
            res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCORR_NORMED, mask=mask)
            _, v, _, loc = cv2.minMaxLoc(res)

            if v > best_conf and v > 0.4:
                # i_strip_y matches j_y at (roi_y0 + loc[1])
                # dy = T_j - T_i = strip_y - (roi_y0 + loc[1])
                candidate_dy = float(strip_y - (roi_y0 + loc[1]))
                if abs(candidate_dy) > max_dy:
                    # Near-zero overlap — discard (likely false match on uniform bg)
                    continue
                best_conf = v
                best_dy = candidate_dy
        except Exception:
            continue

    if best_conf < MIN_TEMPLATE_SCORE:
        return None, 0.0

    M = np.array([[1, 0, 0], [0, 1, best_dy]], np.float32)
    return M, best_conf


def _phase_correlate(
    img_i: np.ndarray,
    img_j: np.ndarray,
    m_i: Optional[np.ndarray],
    m_j: Optional[np.ndarray],
    use_mask: bool = True,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Phase correlation on high-pass filtered Y' channels.

    use_mask: when True (default) zero out foreground pixels before
              correlating so that moving characters don't bias the shift.
              Set False for scenes where the background is so uniform that
              masking removes nearly all texture — the character itself then
              provides the dominant phase signal.
    """
    g_i = _highpass(_luma(img_i)).astype(np.float32)
    g_j = _highpass(_luma(img_j)).astype(np.float32)

    if use_mask:
        if m_i is not None:
            g_i[m_i == 0] = 0.0
        if m_j is not None:
            g_j[m_j == 0] = 0.0

    try:
        hann = cv2.createHanningWindow(g_i.shape[::-1], cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(g_i, g_j, hann)
    except Exception:
        return None, 0.0

    if response < PC_CONF_THRESHOLD:
        return None, 0.0

    dx, dy = shift[0], shift[1]
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return M, response


def _segment_guided_match(
    img_i: np.ndarray,
    img_j: np.ndarray,
    mask_i: Optional[np.ndarray] = None,
    mask_j: Optional[np.ndarray] = None,
    n_colors: int = 16,
    min_seg_px: int = 400,
    min_segs: int = 6,
) -> Tuple[Optional[np.ndarray], float]:
    """
    P2.9 — Segment-guided matching (AnimeInterp technique).

    Segments both frames into flat-color contiguous regions using mean-shift
    filtering + connected components.  For each background region in frame i,
    finds the closest color-and-position match in frame j, then computes the
    centroid displacement.  The median over all matched-region displacements
    gives a robust translation estimate even when LoFTR and phase correlation
    fail on uniform-background anime cells.

    Returns (M, confidence) or (None, 0.0).
    """
    h, w = img_i.shape[:2]

    def _segment(
        img: np.ndarray, mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Downscale for speed (mean-shift is O(N²))
        scale = min(1.0, 320.0 / max(h, w))
        img_s = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
        # Mean-shift segmentation into flat color regions
        ms = cv2.pyrMeanShiftFiltering(img_s, sp=8, sr=30)
        # Quantise colors to reduce fragmentation
        ms_flat = ms.reshape(-1, 3).astype(np.float32)
        _, labels_flat, centers = cv2.kmeans(
            ms_flat,
            n_colors,
            None,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        quantized = centers[labels_flat.flatten()].reshape(img_s.shape).astype(np.uint8)
        # Connected components on quantized image (one CC per flat region)
        gray_q = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
        _, cc_map = cv2.connectedComponents(gray_q, connectivity=8)
        # Scale CC map back to original size
        cc_full = cv2.resize(
            cc_map.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST
        )
        return cc_full, centers[labels_flat.reshape(img_s.shape[:2])]

    try:
        cc_i, _ = _segment(img_i, mask_i)
        cc_j, _ = _segment(img_j, mask_j)
    except Exception:
        return None, 0.0

    def _seg_stats(img: np.ndarray, cc: np.ndarray, mask: Optional[np.ndarray]):
        stats = {}
        for label in np.unique(cc):
            if label == 0:
                continue
            seg_px = cc == label
            if mask is not None:
                seg_px = seg_px & (mask > 127)
            count = int(seg_px.sum())
            if count < min_seg_px:
                continue
            ys, xs = np.where(seg_px)
            cy, cx = float(ys.mean()), float(xs.mean())
            color = img[seg_px].astype(np.float32).mean(axis=0)  # (3,)
            stats[label] = {"cy": cy, "cx": cx, "color": color, "count": count}
        return stats

    segs_i = _seg_stats(img_i, cc_i, mask_i)
    segs_j = _seg_stats(img_j, cc_j, mask_j)

    if len(segs_i) < min_segs or len(segs_j) < min_segs:
        return None, 0.0

    # Build color arrays for matching
    labels_j = list(segs_j.keys())
    colors_j = np.array([segs_j[l]["color"] for l in labels_j], dtype=np.float32)

    displacements = []
    for _li, si in segs_i.items():
        c_i = si["color"]
        # L2 color distance to all segments in j
        color_dists = np.linalg.norm(colors_j - c_i[np.newaxis], axis=1)
        # Position distance (normalised by image size)
        pos_dists = np.array(
            [
                np.sqrt(
                    ((segs_j[lj]["cy"] - si["cy"]) / h) ** 2
                    + ((segs_j[lj]["cx"] - si["cx"]) / w) ** 2
                )
                for lj in labels_j
            ],
            dtype=np.float32,
        )
        # Combined score: low color distance + nearby position
        score = color_dists / 256.0 + 2.0 * pos_dists
        best_idx = int(np.argmin(score))
        if score[best_idx] > 0.5:  # too dissimilar — skip
            continue
        sj = segs_j[labels_j[best_idx]]
        dy = sj["cy"] - si["cy"]
        dx = sj["cx"] - si["cx"]
        displacements.append((dx, dy))

    if len(displacements) < min_segs:
        return None, 0.0

    dxs = np.array([d[0] for d in displacements])
    dys = np.array([d[1] for d in displacements])
    dx_med = float(np.median(dxs))
    dy_med = float(np.median(dys))

    M = np.array([[1, 0, dx_med], [0, 1, dy_med]], dtype=np.float32)
    # Confidence: fraction of displacement pairs within 20px of the median
    residuals = np.sqrt((dxs - dx_med) ** 2 + (dys - dy_med) ** 2)
    conf = float((residuals < 20.0).mean()) * 0.5  # cap at 0.5 (lower than LoFTR)
    return M, max(conf, 0.15)


__all__ = ["_template_match", "_phase_correlate", "_segment_guided_match"]
