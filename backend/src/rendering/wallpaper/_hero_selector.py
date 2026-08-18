"""Hero-frame selection and foreground cel extraction (#426).

Implements the S_f appearance-scoring objective:
    S_f = w_area * Area_norm(mask)
        + w_sharpness * Sharpness(I ⊙ mask)
        + w_symmetry * Symmetry(mask)
        - w_border * BorderIntersection(mask)

Auto-selects the top-scoring candidate as the Hero Frame (or accepts a manual
user override) and extracts the master HeroCel + alpha matte.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class HeroCel:
    """Master foreground cel and metadata extracted from the hero frame."""

    frame_idx: int
    cel_rgba: np.ndarray  # (H, W, 4) uint8 (BGR + Alpha)
    alpha_mask: np.ndarray  # (H, W) uint8 in [0, 255]
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1) in frame space
    score: float
    score_breakdown: dict[str, float]
    all_scores: list[tuple[int, float]]  # (frame_idx, score) sorted descending
    multi_subject_count: int = 1


def _compute_area_norm(mask: np.ndarray) -> float:
    """Fraction of the frame occupied by foreground."""
    total_px = mask.size
    if total_px == 0:
        return 0.0
    fg_px = int(np.count_nonzero(mask > 0))
    return float(fg_px / total_px)


def _compute_sharpness_norm(frame: np.ndarray, mask: np.ndarray) -> float:
    """Normalized Laplacian variance of foreground pixels."""
    fg_indices = mask > 0
    if np.count_nonzero(fg_indices) < 20:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    fg_lap = lap[fg_indices]
    var = float(np.var(fg_lap))
    # Log-scale normalization: maps variance [0, ~1000] smoothly into [0, 1]
    return float(np.clip(np.log1p(var) / 7.0, 0.0, 1.0))


def _compute_symmetry_norm(mask: np.ndarray) -> float:
    """Horizontal symmetry score of the foreground mask within its bounding box."""
    ys, xs = np.where(mask > 0)
    if len(ys) < 20:
        return 0.0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = (mask[y0 : y1 + 1, x0 : x1 + 1] > 0).astype(np.uint8)
    h, w = crop.shape
    if w < 4 or h < 4:
        return 0.0
    mid = w // 2
    left = crop[:, :mid]
    right = crop[:, mid + (w % 2) :]
    right_flipped = np.fliplr(right)
    min_w = min(left.shape[1], right_flipped.shape[1])
    if min_w == 0:
        return 0.0
    l_sub = left[:, :min_w]
    r_sub = right_flipped[:, :min_w]
    intersection = np.logical_and(l_sub, r_sub).sum()
    union = np.logical_or(l_sub, r_sub).sum()
    return float(intersection / union) if union > 0 else 0.0


def _compute_border_intersection(mask: np.ndarray) -> float:
    """Fraction of outer frame perimeter intersected by foreground."""
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return 0.0
    top = mask[0, :] > 0
    bottom = mask[h - 1, :] > 0
    left = mask[:, 0] > 0
    right = mask[:, w - 1] > 0
    fg_border = top.sum() + bottom.sum() + left.sum() + right.sum()
    total_border = 2 * (h + w)
    return float(fg_border / total_border)


def score_candidate_frame(
    frame: np.ndarray,
    fg_mask: np.ndarray,
    *,
    w_area: float = 0.35,
    w_sharpness: float = 0.25,
    w_symmetry: float = 0.15,
    w_border_penalty: float = 0.75,
) -> tuple[float, dict[str, float]]:
    """Score a single candidate frame using the S_f objective."""
    area = _compute_area_norm(fg_mask)
    sharpness = _compute_sharpness_norm(frame, fg_mask)
    symmetry = _compute_symmetry_norm(fg_mask)
    border_pen = _compute_border_intersection(fg_mask)

    score = (
        w_area * area
        + w_sharpness * sharpness
        + w_symmetry * symmetry
        - w_border_penalty * border_pen
    )

    breakdown = {
        "area": area,
        "sharpness": sharpness,
        "symmetry": symmetry,
        "border_intersection": border_pen,
        "score": score,
    }
    return score, breakdown


def select_hero_cel(
    frames: Sequence[np.ndarray],
    fg_masks: Sequence[np.ndarray],
    *,
    override_frame_idx: int | None = None,
    w_area: float = 0.35,
    w_sharpness: float = 0.25,
    w_symmetry: float = 0.15,
    w_border_penalty: float = 0.75,
    min_component_area_ratio: float = 0.35,
) -> HeroCel:
    """Select the hero frame from candidates and extract the master HeroCel."""
    if len(frames) == 0 or len(fg_masks) == 0:
        raise ValueError("Cannot select hero cel from empty frames or masks.")
    if len(frames) != len(fg_masks):
        raise ValueError(
            f"Frames length ({len(frames)}) != fg_masks length ({len(fg_masks)})."
        )

    all_scores: list[tuple[int, float]] = []
    breakdowns: dict[int, dict[str, float]] = {}

    for idx, (frame, raw_mask) in enumerate(zip(frames, fg_masks)):
        mask = (raw_mask > 0).astype(np.uint8) * 255
        score, bd = score_candidate_frame(
            frame,
            mask,
            w_area=w_area,
            w_sharpness=w_sharpness,
            w_symmetry=w_symmetry,
            w_border_penalty=w_border_penalty,
        )
        all_scores.append((idx, score))
        breakdowns[idx] = bd

    all_scores.sort(key=lambda item: item[1], reverse=True)

    if override_frame_idx is not None:
        if not (0 <= override_frame_idx < len(frames)):
            raise IndexError(
                f"override_frame_idx {override_frame_idx} out of range [0, {len(frames)})."
            )
        hero_idx = override_frame_idx
    else:
        hero_idx = all_scores[0][0]

    hero_frame = frames[hero_idx]
    hero_mask = (fg_masks[hero_idx] > 0).astype(np.uint8) * 255

    # Connected component analysis for multi-subject handling
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        hero_mask, connectivity=8
    )
    multi_count = 1
    if num_labels > 2:
        # Exclude background label 0
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = float(np.max(areas)) if len(areas) > 0 else 1.0
        filtered_mask = np.zeros_like(hero_mask)
        active_cels = 0
        for lbl_idx, area in enumerate(areas, start=1):
            if (area / max_area) >= min_component_area_ratio:
                filtered_mask[labels == lbl_idx] = 255
                active_cels += 1
        if active_cels > 0:
            hero_mask = filtered_mask
            multi_count = active_cels

    # Compute bounding box
    ys, xs = np.where(hero_mask > 0)
    if len(ys) > 0:
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    else:
        h, w = hero_frame.shape[:2]
        bbox = (0, 0, w, h)

    # Build 4-channel BGRA cel
    b, g, r = cv2.split(hero_frame)
    cel_rgba = cv2.merge([b, g, r, hero_mask])

    return HeroCel(
        frame_idx=hero_idx,
        cel_rgba=cel_rgba,
        alpha_mask=hero_mask,
        bbox=bbox,
        score=breakdowns[hero_idx]["score"],
        score_breakdown=breakdowns[hero_idx],
        all_scores=all_scores,
        multi_subject_count=multi_count,
    )
