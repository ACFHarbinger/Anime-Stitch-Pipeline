"""M3 first slice: coherence_v2 assigns each FG region to one pose."""

from __future__ import annotations

import numpy as np
from asp_backend.rendering.compositing.coherence_v2 import (
    coherence_v2_enabled,
    has_background_corridor,
    plan_coherence_v2,
)


def test_default_flag_is_off():
    assert coherence_v2_enabled() is False


def test_disjoint_blobs_keep_their_own_pose():
    a = np.zeros((20, 30), dtype=np.uint8)
    b = np.zeros((20, 30), dtype=np.uint8)
    a[2:8, 2:10] = 255
    b[12:18, 18:28] = 255
    plan = plan_coherence_v2(a, b)
    assert plan.corridor is True
    assert plan.handoff is None
    assert set(plan.ownership[a > 0]) == {0}
    assert set(plan.ownership[b > 0]) == {1}
    assert (plan.ownership[(a == 0) & (b == 0)] == -1).all()


def test_overlap_region_has_exactly_one_owner():
    a = np.zeros((16, 16), dtype=np.uint8)
    b = np.zeros((16, 16), dtype=np.uint8)
    a[4:12, 2:10] = 255
    b[4:12, 6:14] = 255  # overlap columns 6-9
    plan = plan_coherence_v2(a, b)
    mixed = (plan.ownership == 0) & (plan.ownership == 1)
    assert not mixed.any()
    overlap = (a > 0) & (b > 0)
    owners = set(plan.ownership[overlap].tolist())
    assert owners == {0}  # A covers more of the union? equal height; A starts earlier
    # Equal overlap area: A-only 4*4=16, B-only 16, overlap 4*4=16 → areas 32 vs 32
    # tie → index_tiebreak owner 0
    assert plan.regions
    assert all(r.owner in (0, 1) for r in plan.regions)


def test_coverage_picks_larger_blob():
    a = np.zeros((16, 16), dtype=np.uint8)
    b = np.zeros((16, 16), dtype=np.uint8)
    a[4:10, 4:10] = 255  # 36 px
    b[4:14, 4:14] = 255  # 100 px, contains A
    plan = plan_coherence_v2(a, b)
    assert set(plan.ownership[(a > 0) | (b > 0)]) == {1}
    assert plan.regions[0].reason == "coverage"


def test_all_foreground_overlap_is_handoff():
    a = np.full((8, 12), 255, dtype=np.uint8)
    b = np.full((8, 12), 255, dtype=np.uint8)
    assert has_background_corridor(a, b) is False
    plan = plan_coherence_v2(a, b)
    assert plan.corridor is False
    assert plan.handoff in (0, 1)
    assert plan.regions[0].reason.startswith("handoff_")
    assert set(np.unique(plan.ownership)) <= {-1, plan.handoff}


def test_background_strip_is_a_corridor():
    a = np.zeros((8, 12), dtype=np.uint8)
    b = np.zeros((8, 12), dtype=np.uint8)
    a[0:3, :] = 255
    b[5:8, :] = 255
    assert has_background_corridor(a, b) is True
    plan = plan_coherence_v2(a, b)
    assert plan.corridor is True
    assert plan.handoff is None
