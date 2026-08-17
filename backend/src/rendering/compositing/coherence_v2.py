"""M3 first slice: ``coherence_v2`` region-to-single-pose assignment.

Critical Evaluation §9.2 Stage 2: where a pixel is foreground, take it from
exactly one pose. Do not median, feather, or seam-blend competing character
poses through the same region. When no background corridor exists across an
overlap, emit an explicit single-pose handoff instead of an infinite-cost
seam grid.

Assignment plus a pixel apply that copies each owned region from exactly
one warped frame. ``composite.py`` may take this path only when
``ASP_COHERENCE_V2=1``. The live HITL seam loop remains the default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np


def coherence_v2_enabled() -> bool:
    return os.environ.get("ASP_COHERENCE_V2", "0") == "1"


def _as_bool(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    return np.asarray(mask) > 0


def has_background_corridor(fg_a: np.ndarray, fg_b: np.ndarray) -> bool:
    """True if background pixels form a 4-connected left–right path.

    A vertical-scroll seam is a horizontal cut; a feasible corridor is a
    background path from x=0 to x=W-1. All-foreground overlap has none.
    """
    fg = _as_bool(fg_a) | _as_bool(fg_b)
    bg = ~fg
    h, w = bg.shape
    if w == 0 or h == 0 or not bg[:, 0].any() or not bg[:, -1].any():
        return False
    n, labels = cv2.connectedComponents(bg.astype(np.uint8), connectivity=4)
    left = set(int(v) for v in np.unique(labels[:, 0]) if v != 0)
    right = set(int(v) for v in np.unique(labels[:, -1]) if v != 0)
    return bool(left & right)


def _pick_owner(
    area_a: int,
    area_b: int,
    conf_a: float | None,
    conf_b: float | None,
) -> tuple[int, str]:
    if area_a == 0 and area_b == 0:
        return 0, "empty"
    if area_a == 0:
        return 1, "only_b"
    if area_b == 0:
        return 0, "only_a"
    if area_a != area_b:
        return (0, "coverage") if area_a > area_b else (1, "coverage")
    if conf_a is not None and conf_b is not None and conf_a != conf_b:
        return (0, "confidence") if conf_a > conf_b else (1, "confidence")
    return 0, "index_tiebreak"


@dataclass
class RegionOwnership:
    region_id: int
    owner: int
    area_a: int
    area_b: int
    reason: str


@dataclass
class CoherenceV2Plan:
    """Per-pixel owner map: -1 = background / unassigned, 0 = frame A, 1 = frame B."""

    ownership: np.ndarray
    regions: list[RegionOwnership] = field(default_factory=list)
    corridor: bool = True
    handoff: int | None = None


def plan_coherence_v2(
    fg_a: np.ndarray,
    fg_b: np.ndarray,
    *,
    conf_a: np.ndarray | None = None,
    conf_b: np.ndarray | None = None,
) -> CoherenceV2Plan:
    """Assign each foreground overlap region to exactly one source pose."""
    a = _as_bool(fg_a)
    b = _as_bool(fg_b)
    union = a | b
    h, w = union.shape
    ownership = np.full((h, w), -1, dtype=np.int8)
    corridor = has_background_corridor(a, b)

    if not corridor and union.any():
        mean_ca = float(conf_a[a].mean()) if conf_a is not None and a.any() else None
        mean_cb = float(conf_b[b].mean()) if conf_b is not None and b.any() else None
        owner, reason = _pick_owner(int(a.sum()), int(b.sum()), mean_ca, mean_cb)
        ownership[union] = owner
        return CoherenceV2Plan(
            ownership=ownership,
            regions=[
                RegionOwnership(
                    region_id=1,
                    owner=owner,
                    area_a=int(a.sum()),
                    area_b=int(b.sum()),
                    reason=f"handoff_{reason}",
                )
            ],
            corridor=False,
            handoff=owner,
        )

    if not union.any():
        return CoherenceV2Plan(ownership=ownership, corridor=corridor)

    n, labels = cv2.connectedComponents(union.astype(np.uint8), connectivity=4)
    regions: list[RegionOwnership] = []
    for rid in range(1, n):
        pix = labels == rid
        area_a = int((pix & a).sum())
        area_b = int((pix & b).sum())
        ca = float(conf_a[pix & a].mean()) if conf_a is not None and area_a else None
        cb = float(conf_b[pix & b].mean()) if conf_b is not None and area_b else None
        owner, reason = _pick_owner(area_a, area_b, ca, cb)
        ownership[pix] = owner
        regions.append(
            RegionOwnership(
                region_id=rid,
                owner=owner,
                area_a=area_a,
                area_b=area_b,
                reason=reason,
            )
        )
    return CoherenceV2Plan(
        ownership=ownership,
        regions=regions,
        corridor=corridor,
        handoff=None,
    )


def fg_mask_from_warped(
    warped: np.ndarray,
    bg_mask: np.ndarray | None,
) -> np.ndarray:
    """Foreground = on-canvas content that is not the BiRefNet background."""
    content = warped.max(axis=2) > 0 if warped.ndim == 3 else warped > 0
    if bg_mask is None:
        return content
    bg = np.asarray(bg_mask)
    if bg.ndim == 3:
        bg = bg.max(axis=2)
    if bg.dtype == bool:
        return content & ~bg
    return content & (bg <= 127)


def apply_coherence_v2(
    img_a: np.ndarray,
    img_b: np.ndarray,
    fg_a: np.ndarray,
    fg_b: np.ndarray,
    *,
    background: np.ndarray | None = None,
    conf_a: np.ndarray | None = None,
    conf_b: np.ndarray | None = None,
) -> tuple[np.ndarray, CoherenceV2Plan]:
    """Paint owned foreground from one source pose only. No blend."""
    plan = plan_coherence_v2(fg_a, fg_b, conf_a=conf_a, conf_b=conf_b)
    out = np.zeros_like(img_a) if background is None else background.copy()
    own0 = plan.ownership == 0
    own1 = plan.ownership == 1
    out[own0] = img_a[own0]
    out[own1] = img_b[own1]
    return out, plan


def composite_coherence_v2(
    warped: list[np.ndarray],
    fg_masks: list[np.ndarray],
    canvas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold adjacent pair plans onto a canvas. First pair to claim a pixel wins."""
    if len(warped) != len(fg_masks) or len(warped) < 2:
        raise ValueError("Need at least two warped frames and matching FG masks.")
    result = canvas.copy()
    claimed = np.full(canvas.shape[:2], -1, dtype=np.int16)
    for i in range(len(warped) - 1):
        plan = plan_coherence_v2(fg_masks[i], fg_masks[i + 1])
        for local, fi in ((0, i), (1, i + 1)):
            take = (plan.ownership == local) & (claimed < 0)
            result[take] = warped[fi][take]
            claimed[take] = fi
    return result, claimed


__all__ = [
    "CoherenceV2Plan",
    "RegionOwnership",
    "coherence_v2_enabled",
    "has_background_corridor",
    "plan_coherence_v2",
    "fg_mask_from_warped",
    "apply_coherence_v2",
    "composite_coherence_v2",
]
