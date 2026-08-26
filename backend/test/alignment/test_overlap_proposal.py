"""P2 connectivity: provisional-geometry overlap/component bridge proposals.

The pure overlap/component logic is tested deterministically from hand-built
positions; the phase-correlation anchoring is tested against synthetic
translated frames (a shifted stripe pattern that phase correlation recovers).
"""

from __future__ import annotations

import numpy as np
import pytest

from asp_backend.alignment.matching._overlap_proposal import (
    build_overlap_bridge_proposals,
    estimate_provisional_positions,
    propose_overlap_bridge_pairs,
)
from asp_backend.alignment.matching._pairwise import TemporalPairProposal, _pairwise_match


def _stripe_frames(n=5, dx=60, dy=8, size=(128, 192)):
    """Synthetic horizontal pan: frame k is frame 0 shifted by k*(dx, dy)."""
    base = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    base[:, 0::16] = 255  # vertical stripes -> horizontal-motion phase signal
    base[0::24, :] = 128  # a few horizontal lines -> vertical signal
    out = []
    for k in range(n):
        M = np.float32([[1, 0, k * dx], [0, 1, k * dy]])
        out.append(cv2.warpAffine(base, M, (size[1], size[0])))
    return out


import cv2  # noqa: E402


def test_estimate_positions_chains_reliable_shifts():
    frames = _stripe_frames(n=5)
    positions, anchored, shifts = estimate_provisional_positions(frames, None)
    assert anchored >= 4  # most adjacent shifts should be reliable
    assert positions[0] == (0.0, 0.0)
    # Relative offsets should recover the synthetic ~(dx, dy) per step.
    if positions[4] is not None and anchored == 5:
        assert abs(positions[4][0] + 4 * 60) < 8.0
        assert abs(positions[4][1] + 4 * 8) < 8.0


def test_unknown_stays_unknown_after_unreliable_shift(monkeypatch):
    frames = _stripe_frames(n=5)

    def fake_pc(img_i, img_j, m_i, m_j, use_mask=True):
        # Reject the 1->2 shift so strip 2 and everything after stays unknown.
        return (None, 0.0)

    import asp_backend.alignment.matching._overlap_proposal as mod

    real = mod._phase_correlate
    call = [0]

    def flaky(img_i, img_j, m_i, m_j, use_mask=True):
        call[0] += 1
        if call[0] == 2:  # pair (1,2)
            return (None, 0.0)
        return real(img_i, img_j, m_i, m_j, use_mask)

    monkeypatch.setattr(mod, "_phase_correlate", flaky)
    positions, anchored, _ = estimate_provisional_positions(frames, None)
    # strip 0 anchored; strip 1 chained; strip 2 unknown and never interpolated.
    assert positions[0] is not None
    assert positions[1] is not None
    assert positions[2] is None
    assert positions[3] is None
    assert anchored == 2


def test_build_bridge_proposals_high_overlap_skip():
    # 5 strips spaced 30px apart horizontally (heavy overlap at W=192), all in
    # one reliable chain -> high-overlap skip beyond the temporal backbone.
    positions = [(i * 30.0, 0.0) for i in range(5)]
    extra, telemetry = build_overlap_bridge_proposals(
        5, positions, W=192, H=128, range_width=3, min_overlap=0.3
    )
    assert telemetry["stopped_reason"] is None
    reasons = [p.reason for p in extra]
    assert "overlap_bridge" in reasons
    # The (0,4) pair spans 4, outside the temporal backbone.
    assert any(p.span > 3 for p in extra)


def test_build_bridge_proposals_connects_two_components():
    # Two reliable chains that geometrically overlap at the boundary: strips
    # 0..2 at 0/40/80 (chain), strip 3 unknown (chain gap), strip 4 at 100.
    positions = [(0.0, 0.0), (40.0, 0.0), (80.0, 0.0), None, (100.0, 0.0)]
    extra, telemetry = build_overlap_bridge_proposals(
        5, positions, W=192, H=128, range_width=3, min_overlap=0.1
    )
    assert telemetry["components"] == 2
    assert any(p.reason == "component_bridge" for p in extra)


def test_sparse_anchors_stop_without_proposals():
    positions = [(0.0, 0.0), None, None, (500.0, 0.0), None]
    extra, telemetry = build_overlap_bridge_proposals(
        5, positions, W=192, H=128
    )
    assert telemetry["stopped_reason"] == "anchors_too_sparse"
    assert extra == []


def test_propose_overlap_bridge_pairs_end_to_end():
    frames = _stripe_frames(n=5)
    extra, telemetry = propose_overlap_bridge_pairs(frames, None)
    assert "provisional_anchors" in telemetry
    assert telemetry["stopped_reason"] is None or extra == []


def test_pairwise_match_appends_extra_proposals_deduplicated():
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(5)]
    extra = [
        TemporalPairProposal(i=0, j=4, span=4, reason="component_bridge"),
        TemporalPairProposal(i=1, j=2, span=1, reason="duplicate_of_backbone"),
    ]
    calls: list[tuple[int, int]] = []

    def fake_match(frames, masks, i, j, H, W, **kwargs):
        calls.append((i, j))
        return {"i": i, "j": j, "M": np.eye(2, 3), "weight": 1.0}

    import asp_backend.alignment.matching._pairwise as pw

    orig = pw._match_pair
    pw._match_pair = fake_match
    try:
        telemetry: dict = {}
        edges = _pairwise_match(
            frames, [None] * 5, use_loftr=False, proposal_telemetry=telemetry,
            extra_proposals=extra,
        )
    finally:
        pw._match_pair = orig

    # The backbone (1,2) is matched exactly once; the duplicate extra is dropped;
    # the (0,4) bridge is appended.
    assert calls.count((1, 2)) == 1
    assert calls.count((0, 4)) == 1
    assert len(calls) == 10  # 9 backbone + 1 bridge
    assert telemetry["extra_proposals_added"] == 2  # counted before dedup
