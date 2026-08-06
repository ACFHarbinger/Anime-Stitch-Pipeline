"""Per-pixel FFT animation detection + phase clustering, for temporal-median re-render."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


def _cluster_animation_phases(
    frames: List[np.ndarray],
    affines: List[np.ndarray],
    H: int,
    W: int,
    target_w: int = 320,
    ac_threshold: float = 0.25,
    min_anim_pixels: int = 500,
):
    """
    Detect cyclic animation pixels via per-pixel FFT along the temporal axis,
    then cluster frames by animation phase.

    Returns
    -------
    anim_mask_full : (H, W) uint8 — 255 = animation pixel — or None.
    phase_groups   : list of frame-index lists, one per phase, or None.
    """
    N = len(frames)
    if N < 4:
        return None, None

    scale = target_w / max(W, 1)
    th = max(1, int(H * scale))
    tw = target_w

    small_stack = []
    for i in range(N):
        tx = float(affines[i][0, 2])
        ty = float(affines[i][1, 2])
        M_small = np.array(
            [[scale, 0.0, tx * scale], [0.0, scale, ty * scale]], np.float32
        )
        warped = cv2.warpAffine(
            frames[i],
            M_small,
            (tw, th),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        small_stack.append(gray)

    stack_arr = np.stack(small_stack, axis=0)  # (N, th, tw)

    # Per-pixel FFT along temporal axis
    F = np.fft.rfft(stack_arr, axis=0)
    power = np.abs(F) ** 2
    dc_power = power[0]
    ac_power = power[1:].sum(axis=0)
    ratio = ac_power / (dc_power + ac_power + 1e-8)

    anim_mask_small = (ratio > ac_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    anim_mask_small = cv2.morphologyEx(anim_mask_small, cv2.MORPH_OPEN, kernel)
    anim_mask_small = cv2.morphologyEx(anim_mask_small, cv2.MORPH_CLOSE, kernel)

    if int(anim_mask_small.sum()) // 255 < min_anim_pixels:
        return None, None

    anim_mask_full = cv2.resize(
        anim_mask_small, (W, H), interpolation=cv2.INTER_NEAREST
    )

    # Edge-signature KMeans clustering for phase assignment
    anim_ys, anim_xs = np.where(anim_mask_small > 0)
    sigs = []
    for gray in small_stack:
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
        sigs.append(edges[anim_ys, anim_xs].astype(np.float32))

    sig_matrix = np.stack(sigs, axis=0)
    n_clusters = max(2, min(8, N // 2))

    try:
        from sklearn.cluster import (
            KMeans,
        )  # §3.14 lazy — avoids sklearn load at pytest collection

        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
        labels = km.fit_predict(sig_matrix)
    except ImportError:
        labels = np.arange(N) % n_clusters

    phase_groups = [
        [idx for idx in range(N) if labels[idx] == k]
        for k in range(n_clusters)
        if any(labels == k)
    ]

    return anim_mask_full, phase_groups


__all__ = ["_cluster_animation_phases"]
