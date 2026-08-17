"""M3 first slice: coherence_v2 assigns each FG region to one pose."""

from __future__ import annotations

import numpy as np
from asp_backend.rendering.compositing.coherence_v2 import (
    apply_coherence_v2,
    coherence_v2_enabled,
    composite_coherence_v2,
    fg_mask_from_warped,
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


def test_apply_does_not_blend_competing_poses():
    img_a = np.zeros((16, 16, 3), dtype=np.uint8)
    img_b = np.zeros((16, 16, 3), dtype=np.uint8)
    img_a[4:12, 2:10] = (0, 0, 200)
    img_b[4:12, 6:14] = (200, 0, 0)
    fg_a = img_a.max(axis=2) > 0
    fg_b = img_b.max(axis=2) > 0
    out, plan = apply_coherence_v2(img_a, img_b, fg_a, fg_b)
    overlap = fg_a & fg_b
    assert (plan.ownership[overlap] == 0).all()
    assert (out[overlap] == (0, 0, 200)).all()
    assert not np.any((out[..., 0] > 0) & (out[..., 2] > 0))


def test_nframe_first_claim_wins():
    h, w = 10, 12
    a = np.zeros((h, w, 3), dtype=np.uint8)
    b = np.zeros((h, w, 3), dtype=np.uint8)
    c = np.zeros((h, w, 3), dtype=np.uint8)
    a[1:4, 1:6] = 10
    b[1:4, 4:10] = 20
    c[1:4, 8:12] = 30
    canvas = np.full((h, w, 3), 1, dtype=np.uint8)
    out, claimed = composite_coherence_v2(
        [a, b, c],
        [a.max(axis=2) > 0, b.max(axis=2) > 0, c.max(axis=2) > 0],
        canvas,
    )
    assert (out[claimed < 0] == 1).all()
    vals = set(int(v) for v in np.unique(out[..., 0]))
    # Owner-take-all may write 0 where the winning pose has no color.
    # Never a blend of two sources (15/25).
    assert vals <= {0, 1, 10, 20, 30}
    assert 15 not in vals and 25 not in vals


def test_fg_mask_treats_birefnet_bg_as_background():
    img = np.full((4, 4, 3), 50, dtype=np.uint8)
    bg = np.full((4, 4), 200, dtype=np.uint8)
    bg[1:3, 1:3] = 0
    fg = fg_mask_from_warped(img, bg)
    assert bool(fg[0, 0]) is False
    assert bool(fg[1, 1]) is True


def test_fg_mask_inverts_boolean_background():
    img = np.full((3, 3, 3), 40, dtype=np.uint8)
    bg = np.zeros((3, 3), dtype=bool)
    bg[0, :] = True
    fg = fg_mask_from_warped(img, bg)
    assert bool(fg[0, 1]) is False
    assert bool(fg[1, 1]) is True


def test_flagged_composite_records_coherence_v2(monkeypatch):
    monkeypatch.setenv("ASP_COHERENCE_V2", "1")
    from asp_backend.rendering.compositing.composite import _composite_foreground

    h, w = 20, 24
    a = np.zeros((h, w, 3), dtype=np.uint8)
    b = np.zeros((h, w, 3), dtype=np.uint8)
    a[2:8, 2:10] = (0, 0, 180)
    b[10:16, 12:20] = (180, 0, 0)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    eye = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    ty = eye.copy()
    ty[1, 2] = 2.0
    meta: dict = {}
    out = _composite_foreground(
        [],
        [],
        canvas,
        h,
        w,
        [a, b],
        [eye, ty],
        [None, None],
        seam_meta_out=meta,
    )
    assert meta.get("coherence_v2") is True
    assert out.shape == canvas.shape
