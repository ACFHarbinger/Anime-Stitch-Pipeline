"""Pre-match provisional geometry for Stage 5-6 connectivity — the Hugin
``CalculateImageOverlap``/``ImageGraph`` analog for ASP.

ASP has no camera poses before matching, so Hugin's overlap graph cannot be
computed directly. The P2 experiment builds *provisional* strip coordinates
instead: cheap phase-correlated translations between adjacent frames on the
downsampled, background-masked luma (Stage-4 ``bg_masks``), chained from
frame 0 with confidence rejection — an unknown shift leaves the strip
position unknown (never interpolated). Anchored strips are then placed as
W×H rectangles at those provisional coordinates and pairwise rectangle
overlap builds graph components, mirroring Hugin's overlap-sampled image
graph.

The proposal is **additive only**: bridge pairs are appended to the temporal
backbone to connect components / strengthen high-overlap skips. Adjacent
pairs are never removed, and post-BA affines are never used to choose pairs.
Default-off behind ``ASP_OVERLAP_PROPOSAL``; if too few anchors are available
the module returns no bridges and reports a stop reason rather than forcing
extra matching (see `asp_connectivity_vendor_scope_2026-08-23.md` P2).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import cv2
import numpy as np

from ._matchers import _phase_correlate
from ._pairwise import TemporalPairProposal

_DEFAULT_SCALE = 0.25          # downsample factor for the cheap correlation
_DEFAULT_MIN_OVERLAP = 0.5     # overlap bridge threshold (fraction of min area)
_DEFAULT_CONF_THRESHOLD = 0.05  # phaseCorrelate response floor (PC_CONF_THRESHOLD)


def _downsample(img: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0.0 or scale >= 1.0:
        return img
    h, w = img.shape[:2]
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def estimate_provisional_positions(
    frames: list[np.ndarray],
    bg_masks: list[np.ndarray | None] | None = None,
    scale: float = _DEFAULT_SCALE,
    conf_threshold: float = _DEFAULT_CONF_THRESHOLD,
) -> tuple[list[tuple[float, float] | None], int, list[dict | None]]:
    """Return (positions, anchors, shifts).

    ``positions`` is one ``(x, y)`` per strip in full-resolution pixels, or
    ``None`` for strips whose chain to frame 0 contains an unreliable shift.
    ``anchors`` counts reliably-placed strips. ``shifts`` is one entry per
    adjacent pair: ``{"dx", "dy", "response"}`` or ``None`` for rejected.
    """
    N = len(frames)
    positions: list[tuple[float, float] | None] = [None] * N
    shifts: list[dict | None] = [None] * max(0, N - 1)
    if N == 0:
        return positions, 0, shifts
    positions[0] = (0.0, 0.0)
    anchored = 1
    if N == 1:
        return positions, anchored, shifts

    masks = bg_masks or [None] * N
    small = [_downsample(f, scale) for f in frames]
    small_masks = [_downsample(m, scale) if m is not None else None for m in masks]

    for k in range(N - 1):
        M, response = _phase_correlate(
            small[k], small[k + 1], small_masks[k], small_masks[k + 1]
        )
        if M is None or response < conf_threshold:
            continue  # shift rejected -> strip k+1 stays unknown
        dx = float(M[0, 2]) / scale
        dy = float(M[1, 2]) / scale
        shifts[k] = {"dx": dx, "dy": dy, "response": float(response)}
        # Chain only from a known anchor; unknown stays unknown (no interpolation).
        if positions[k] is not None:
            positions[k + 1] = (positions[k][0] - dx, positions[k][1] - dy)
            anchored += 1
    return positions, anchored, shifts


def _overlap_frac(rect_a: tuple[float, float, float, float], rect_b: tuple[float, float, float, float]) -> float:
    """Fraction of the smaller rectangle covered by the intersection (0..1)."""
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    area_a = aw * ah
    area_b = bw * bh
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    return inter / min(area_a, area_b)


def build_overlap_bridge_proposals(
    N: int,
    positions: list[tuple[float, float] | None],
    W: int,
    H: int,
    range_width: int = 3,
    min_overlap: float = _DEFAULT_MIN_OVERLAP,
    max_bridges: int | None = None,
) -> tuple[list[TemporalPairProposal], dict[str, Any]]:
    """Build bridge proposals from provisional anchors.

    Only strips with known positions participate. Components are the connected
    components of the anchored overlap graph (edge when overlap > 0). Returns
    ``(extra_proposals, telemetry)``; ``extra_proposals`` is empty and
    ``telemetry["stopped_reason"]`` set when anchoring is too sparse to be
    worth the match budget.
    """
    telemetry: dict[str, Any] = {
        "provisional_anchors": sum(1 for p in positions if p is not None),
        "provisional_unknown": sum(1 for p in positions if p is None),
        "overlap_bridge_added": 0,
        "component_bridge_added": 0,
        "components": 0,
        "stopped_reason": None,
    }
    anchored_idx = [i for i, p in enumerate(positions) if p is not None]
    if len(anchored_idx) < 2 or len(anchored_idx) < N / 2:
        telemetry["stopped_reason"] = "anchors_too_sparse"
        return [], telemetry

    # Overlap for every anchored pair not already in the temporal backbone.
    existing = {
        (i, i + span) for span in range(1, min(range_width, N - 1) + 1)
        for i in range(N - span)
    }
    rects = {
        i: (positions[i][0], positions[i][1], float(W), float(H))
        for i in anchored_idx
    }
    overlaps: dict[tuple[int, int], float] = {}
    for a_i, i in enumerate(anchored_idx):
        for j in anchored_idx[a_i + 1 :]:
            ov = _overlap_frac(rects[i], rects[j])
            if ov > 0.0:
                key = (i, j) if i < j else (j, i)
                overlaps[key] = max(overlaps.get(key, 0.0), ov)

    # Components of the *provisional chain*: consecutive strips with known
    # positions are edges (the reliable-shift graph, mirroring Hugin's ImageGraph
    # over the sampled overlap). Raw overlap is used only to score bridges.
    adj: dict[int, set[int]] = {i: set() for i in anchored_idx}
    known_positions = {i for i in anchored_idx}
    for k in range(N - 1):
        if k in known_positions and k + 1 in known_positions:
            adj[k].add(k + 1)
            adj[k + 1].add(k)
    comp_id: dict[int, int] = {}
    components: list[list[int]] = []
    for i in anchored_idx:
        if i in comp_id:
            continue
        cid = len(components)
        stack = [i]
        comp_id[i] = cid
        comp: list[int] = []
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in comp_id:
                    comp_id[y] = cid
                    stack.append(y)
        components.append(comp)
    telemetry["components"] = len(components)

    added: list[TemporalPairProposal] = []
    added_keys: set[tuple[int, int]] = set()
    # A handful of bridges, not a per-pair sweep: connectivity needs ~#components
    # bridges; a generous cap would blow the match budget (each bridge is a real
    # matcher run). Default = components-to-connect + 2 spare.
    budget = max_bridges if max_bridges is not None else max(1, len(components) + 1)

    # 1) Component bridges first: the highest-overlap pair across each split —
    #    these are what actually connect the provisional graph.
    if len(components) > 1:
        cross: list[tuple[float, int, int]] = []
        for (i, j), ov in overlaps.items():
            if comp_id.get(i) != comp_id.get(j):
                cross.append((ov, i, j))
        cross.sort(reverse=True)
        joined: set[int] = set()
        for ov, i, j in cross:
            if len(added) >= budget:
                break
            ci, cj = comp_id[i], comp_id[j]
            if ci in joined and cj in joined:
                continue
            if (i, j) in existing or (i, j) in added_keys:
                continue
            added_keys.add((i, j))
            joined.add(ci)
            joined.add(cj)
            added.append(TemporalPairProposal(i=i, j=j, span=j - i, reason="component_bridge"))
    telemetry["component_bridge_added"] = sum(1 for a in added if a.reason == "component_bridge")

    # 2) High-overlap skips beyond the temporal backbone, only if budget remains
    #    and only within the same component (redundant path, not a connect).
    for (i, j), ov in sorted(overlaps.items(), key=lambda kv: -kv[1]):
        if len(added) >= budget:
            break
        if (i, j) in existing or (i, j) in added_keys:
            continue
        if ov >= min_overlap and comp_id.get(i) == comp_id.get(j):
            added_keys.add((i, j))
            added.append(TemporalPairProposal(i=i, j=j, span=j - i, reason="overlap_bridge"))
    telemetry["overlap_bridge_added"] = sum(1 for a in added if a.reason == "overlap_bridge")

    return added, telemetry


def propose_overlap_bridge_pairs(
    frames: list[np.ndarray],
    bg_masks: list[np.ndarray | None] | None = None,
    range_width: int = 3,
    scale: float = _DEFAULT_SCALE,
    min_overlap: float = _DEFAULT_MIN_OVERLAP,
    max_bridges: int | None = None,
) -> tuple[list[TemporalPairProposal], dict[str, Any]]:
    """Top-level P2 entry: provisional geometry -> bridge proposals + telemetry."""
    N = len(frames)
    if N < 3:
        return [], {"provisional_anchors": N, "provisional_unknown": 0,
                    "overlap_bridge_added": 0, "component_bridge_added": 0,
                    "components": 1, "stopped_reason": "too_few_frames"}
    H, W = frames[0].shape[:2]
    positions, anchored, shifts = estimate_provisional_positions(
        frames, bg_masks, scale=scale
    )
    extra, telemetry = build_overlap_bridge_proposals(
        N, positions, W, H, range_width=range_width,
        min_overlap=min_overlap, max_bridges=max_bridges,
    )
    telemetry["adjacent_shifts_accepted"] = sum(1 for s in shifts if s is not None)
    telemetry["adjacent_shifts_rejected"] = sum(1 for s in shifts if s is None)
    return extra, telemetry


__all__ = [
    "estimate_provisional_positions",
    "build_overlap_bridge_proposals",
    "propose_overlap_bridge_pairs",
]
