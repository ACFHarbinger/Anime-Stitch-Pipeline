"""§3.12A: Overmix-style hold-block sub-pixel averaging."""

from __future__ import annotations

import os
import tempfile
from typing import List, Tuple

import cv2
import numpy as np


def _hold_block_average(
    thumbs: List[np.ndarray],
    hold_ids: List[int],
    paths: List[str],
) -> Tuple[List[np.ndarray], List[str]]:
    """Compress hold blocks into one ECC-aligned average frame each.

    For MPEG-compressed sources, MPEG DCT block noise cancels out by √N when N
    full-resolution frames within the same animation hold are stack-averaged
    after sub-pixel ECC alignment.  The averaged image is written to a temp
    file so downstream stages (which read from ``paths``) see the denoised
    pixels rather than a single original frame — averaging that only touched
    the phase-correlation thumbnail would be invisible in the final composite.
    Singleton blocks are returned unchanged (no temp file, no re-encode).

    ``thumbs`` are the existing grayscale [0,1] thumbnails aligned 1:1 with
    ``paths``; the returned thumbnail list is recomputed from the averaged
    full-resolution image for blocks that were actually averaged.
    """
    from collections import OrderedDict

    blocks: OrderedDict[int, List[int]] = OrderedDict()
    for idx, hid in enumerate(hold_ids):
        blocks.setdefault(hid, []).append(idx)

    out_thumbs: List[np.ndarray] = []
    out_paths: List[str] = []

    for indices in blocks.values():
        if len(indices) == 1:
            out_thumbs.append(thumbs[indices[0]])
            out_paths.append(paths[indices[0]])
            continue

        full_frames = [cv2.imread(paths[i]) for i in indices]
        if any(f is None for f in full_frames):
            mid = indices[len(indices) // 2]
            out_thumbs.append(thumbs[mid])
            out_paths.append(paths[mid])
            continue

        ref = full_frames[0].astype(np.float32)
        ref_gray = cv2.cvtColor(full_frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        stack = [ref]
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1e-3)
        warp_init = np.eye(2, 3, dtype=np.float32)

        for f in full_frames[1:]:
            src_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            try:
                _, warp = cv2.findTransformECC(
                    ref_gray,
                    src_gray,
                    warp_init.copy(),
                    cv2.MOTION_TRANSLATION,
                    criteria,
                )
                aligned = cv2.warpAffine(
                    f.astype(np.float32),
                    warp,
                    (ref.shape[1], ref.shape[0]),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                )
                stack.append(aligned)
            except cv2.error:
                stack.append(f.astype(np.float32))

        avg = np.mean(stack, axis=0).clip(0, 255).astype(np.uint8)

        fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="asp_holdavg_")
        os.close(fd)
        cv2.imwrite(tmp_path, avg)

        th, tw = thumbs[indices[0]].shape[:2]
        avg_gray = cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY)
        avg_thumb = cv2.resize(avg_gray, (tw, th)).astype(np.float32) / 255.0

        out_thumbs.append(avg_thumb)
        out_paths.append(tmp_path)

    return out_thumbs, out_paths


__all__ = ["_hold_block_average"]
