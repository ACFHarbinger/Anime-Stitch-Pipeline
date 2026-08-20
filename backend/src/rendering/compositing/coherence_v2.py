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


def _region_visibility(pix: np.ndarray) -> float | None:
    """Compactness proxy: filled area / bounding-box area of *pix*.

    A pose fragmented by occlusion (holes, splits) fills less of its own
    bounding box than a fully visible one. ``None`` for an empty region.
    """
    ys, xs = np.where(pix)
    if ys.size == 0:
        return None
    bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    if bbox_area <= 0:
        return None
    return float(ys.size) / float(bbox_area)


def _region_boundary_truncation(pix: np.ndarray) -> float:
    """Fraction of *pix*'s own bounding box perimeter sitting on the frame
    edge — a pose cropped by the frame border scores higher (worse)."""
    ys, xs = np.where(pix)
    if ys.size == 0:
        return 0.0
    h, w = pix.shape
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    edges_touched = sum(
        (y0 == 0, y1 == h - 1, x0 == 0, x1 == w - 1)
    )
    return edges_touched / 4.0


def _region_frame_quality(frame: np.ndarray | None, pix: np.ndarray) -> float | None:
    """Normalized Laplacian-variance sharpness of *frame* restricted to
    *pix*. ``None`` when no source frame is available for this candidate."""
    if frame is None or not pix.any():
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    region = gray[pix]
    if region.size < 4:
        return None
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.abs(lap[pix]).mean())


def _weighted_score(
    visibility: float | None,
    boundary_trunc: float,
    quality: float | None,
    quality_ref: float,
    temporal_bonus: float,
) -> float:
    """Combine the M3 secondary ownership factors into one comparable score.

    Each term is already normalized to roughly [0, 1] before weighting;
    ``quality`` is scaled against ``quality_ref`` (the max of the two
    candidates' sharpness) so an absent frame doesn't zero out the score.
    """
    score = 0.0
    if visibility is not None:
        score += 0.35 * visibility
    score += 0.15 * (1.0 - boundary_trunc)
    if quality is not None and quality_ref > 0:
        score += 0.25 * min(quality / quality_ref, 1.0)
    score += 0.25 * temporal_bonus
    return score


def _pick_owner(
    area_a: int,
    area_b: int,
    conf_a: float | None,
    conf_b: float | None,
    *,
    pix: np.ndarray | None = None,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    frame_a: np.ndarray | None = None,
    frame_b: np.ndarray | None = None,
    prior_owner: np.ndarray | None = None,
) -> tuple[int, str]:
    """Deterministic ownership tie-break.

    Priority order: coverage -> confidence -> {visibility, boundary
    truncation, frame quality, temporal consistency} weighted score ->
    index. The first three stages are unchanged from the M3 first slice;
    the weighted stage only engages when a caller supplies the region mask
    (``pix``) and per-frame source data, so old call sites (area/confidence
    only) are bit-for-bit unaffected.
    """
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

    if pix is not None and a is not None and b is not None:
        # Visibility/boundary/quality are properties of each candidate's
        # *whole* pose mask in its own frame (how occluded/cropped/sharp
        # that pose is overall), not of the contested overlap pixels —
        # `pix` is by construction a subset of both `a` and `b`, so
        # `pix & a` and `pix & b` are the same set and could never
        # differentiate the two candidates.
        vis_a = _region_visibility(a)
        vis_b = _region_visibility(b)
        trunc_a = _region_boundary_truncation(a)
        trunc_b = _region_boundary_truncation(b)
        qual_a = _region_frame_quality(frame_a, a)
        qual_b = _region_frame_quality(frame_b, b)
        qual_ref = max(qual_a or 0.0, qual_b or 0.0)
        prior_a = prior_b = 0.0
        if prior_owner is not None:
            prior_a = float((prior_owner[pix] == 0).mean()) if pix.any() else 0.0
            prior_b = float((prior_owner[pix] == 1).mean()) if pix.any() else 0.0
        score_a = _weighted_score(vis_a, trunc_a, qual_a, qual_ref, prior_a)
        score_b = _weighted_score(vis_b, trunc_b, qual_b, qual_ref, prior_b)
        if abs(score_a - score_b) > 1e-6:
            return (0, "weighted") if score_a > score_b else (1, "weighted")

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
    frame_a: np.ndarray | None = None,
    frame_b: np.ndarray | None = None,
    prior_owner: np.ndarray | None = None,
) -> CoherenceV2Plan:
    """Assign each foreground overlap region to exactly one source pose."""
    a = _as_bool(fg_a)
    b = _as_bool(fg_b)
    union = a | b
    h, w = union.shape
    ownership = np.full((h, w), -1, dtype=np.int8)
    corridor = has_background_corridor(a, b)

    # Exclusive coverage always stays with its source. Only A∩B is contested.
    only_a = a & ~b
    only_b = b & ~a
    overlap = a & b
    ownership[only_a] = 0
    ownership[only_b] = 1

    if not union.any():
        return CoherenceV2Plan(ownership=ownership, corridor=corridor)

    regions: list[RegionOwnership] = []
    if only_a.any():
        regions.append(
            RegionOwnership(1, 0, int(only_a.sum()), 0, "only_a")
        )
    if only_b.any():
        regions.append(
            RegionOwnership(2, 1, 0, int(only_b.sum()), "only_b")
        )

    if not overlap.any():
        return CoherenceV2Plan(
            ownership=ownership, regions=regions, corridor=corridor
        )

    # No BG corridor: one owner for the whole contested overlap (not the union).
    if not corridor:
        mean_ca = float(conf_a[overlap].mean()) if conf_a is not None else None
        mean_cb = float(conf_b[overlap].mean()) if conf_b is not None else None
        owner, reason = _pick_owner(
            int((overlap & a).sum()),
            int((overlap & b).sum()),
            mean_ca,
            mean_cb,
            pix=overlap,
            a=a,
            b=b,
            frame_a=frame_a,
            frame_b=frame_b,
            prior_owner=prior_owner,
        )
        ownership[overlap] = owner
        regions.append(
            RegionOwnership(
                region_id=3,
                owner=owner,
                area_a=int((overlap & a).sum()),
                area_b=int((overlap & b).sum()),
                reason=f"handoff_{reason}",
            )
        )
        return CoherenceV2Plan(
            ownership=ownership,
            regions=regions,
            corridor=False,
            handoff=owner,
        )

    n, labels = cv2.connectedComponents(overlap.astype(np.uint8), connectivity=4)
    for rid in range(1, n):
        pix = labels == rid
        area_a = int((pix & a).sum())
        area_b = int((pix & b).sum())
        ca = float(conf_a[pix].mean()) if conf_a is not None else None
        cb = float(conf_b[pix].mean()) if conf_b is not None else None
        owner, reason = _pick_owner(
            area_a,
            area_b,
            ca,
            cb,
            pix=pix,
            a=a,
            b=b,
            frame_a=frame_a,
            frame_b=frame_b,
            prior_owner=prior_owner,
        )
        ownership[pix] = owner
        regions.append(
            RegionOwnership(
                region_id=10 + rid,
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
    prior_owner: np.ndarray | None = None,
) -> tuple[np.ndarray, CoherenceV2Plan]:
    """Paint owned foreground from one source pose only. No blend.

    ``img_a``/``img_b`` double as the frame-quality signal (M3 ownership
    factors) — no separate frame arguments needed since the sources being
    composited *are* the candidate frames.
    """
    plan = plan_coherence_v2(
        fg_a,
        fg_b,
        conf_a=conf_a,
        conf_b=conf_b,
        frame_a=img_a,
        frame_b=img_b,
        prior_owner=prior_owner,
    )
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
    """Fold adjacent pair plans onto a canvas. First pair to claim a pixel wins.

    Temporal consistency (an M3 ownership factor): each pair's plan is
    given the already-``claimed`` map from prior pairs as ``prior_owner``,
    so a contested region that a neighboring fold already assigned to one
    side is nudged toward staying with that side rather than flickering.
    """
    if len(warped) != len(fg_masks) or len(warped) < 2:
        raise ValueError("Need at least two warped frames and matching FG masks.")
    result = canvas.copy()
    claimed = np.full(canvas.shape[:2], -1, dtype=np.int16)
    ownership_log: list[dict] = []
    for i in range(len(warped) - 1):
        # claimed uses source-frame indices; plan_coherence_v2 wants local
        # 0/1 relative to this pair, so remap the two frame indices in play.
        prior_local = np.full(canvas.shape[:2], -1, dtype=np.int8)
        prior_local[claimed == i] = 0
        prior_local[claimed == i + 1] = 1
        plan = plan_coherence_v2(
            fg_masks[i],
            fg_masks[i + 1],
            frame_a=warped[i],
            frame_b=warped[i + 1],
            prior_owner=prior_local,
        )
        for local, fi in ((0, i), (1, i + 1)):
            take = (plan.ownership == local) & (claimed < 0)
            result[take] = warped[fi][take]
            claimed[take] = fi
        ownership_log.append(
            {
                "pair": [i, i + 1],
                "corridor": plan.corridor,
                "handoff": plan.handoff,
                "regions": [
                    {
                        "region_id": r.region_id,
                        "owner": i + r.owner,
                        "area_a": r.area_a,
                        "area_b": r.area_b,
                        "reason": r.reason,
                    }
                    for r in plan.regions
                ],
            }
        )
    claimed_meta = {"pairs": ownership_log}
    return result, claimed, claimed_meta


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
