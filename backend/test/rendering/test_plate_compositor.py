"""Unit tests for P1 canvas-aligned background plate + single foreground pose compositor."""

from __future__ import annotations

import numpy as np
from asp_backend.rendering.compositing._gain_compensation import _apply_joint_gain_solve
from asp_backend.rendering.compositing._normalization import _warp_inputs
from asp_backend.rendering.compositing._plate_compositor import (
    _blend_phase_plates,
    _build_aligned_background_plate,
    _laplacian_blend,
    composite_plate_multiphase,
    composite_plate_single_pose,
    plate_single_pose_enabled,
    plate_single_pose_safe_for_phases,
)
from asp_backend.rendering.compositing.composite import _composite_foreground


def test_plate_compositor_enabled_flag(monkeypatch):
    monkeypatch.delenv("ASP_PLATE_SINGLE_POSE", raising=False)
    assert not plate_single_pose_enabled()
    monkeypatch.setenv("ASP_PLATE_SINGLE_POSE", "1")
    assert plate_single_pose_enabled()


def test_plate_single_pose_rejects_multiple_phases():
    assert plate_single_pose_safe_for_phases(None, 3)
    assert plate_single_pose_safe_for_phases([0, 0, 0], 3)
    assert not plate_single_pose_safe_for_phases([0, 1, 1], 3)
    assert not plate_single_pose_safe_for_phases([0, 0, 0], 2)
    assert not plate_single_pose_safe_for_phases([0, 0, 0], 3, True)


def test_build_aligned_background_plate_reconstruction():
    """Verify plate builder recovers static background from masked inputs."""
    H, W = 100, 100
    # Background is a gradient
    bg_base = np.zeros((H, W, 3), dtype=np.uint8)
    for r in range(H):
        bg_base[r, :, :] = (r * 2) % 255

    # Frame 0: Character box at top (y: 10-30, x: 20-40)
    f0 = bg_base.copy()
    f0[10:30, 20:40] = [255, 0, 0]
    bg0 = np.ones((H, W), dtype=bool)
    bg0[10:30, 20:40] = False

    # Frame 1: Character box at bottom (y: 60-80, x: 20-40)
    f1 = bg_base.copy()
    f1[60:80, 20:40] = [0, 255, 0]
    bg1 = np.ones((H, W), dtype=bool)
    bg1[60:80, 20:40] = False

    plate, valid = _build_aligned_background_plate([f0, f1], [bg0, bg1], H, W)
    assert valid.all()
    # Plate should have cleanly reconstructed bg_base without the character boxes
    diff = np.abs(plate.astype(np.int16) - bg_base.astype(np.int16))
    assert diff.max() <= 1


def test_composite_plate_single_pose_one_character():
    """Verify that composite takes exactly ONE character pose per region."""
    H, W = 120, 120
    bg_base = np.full((H, W, 3), 100, dtype=np.uint8)

    # Frame 0: small character box (area 100)
    f0 = bg_base.copy()
    f0[40:50, 40:50] = 200
    bg0 = np.ones((H, W), dtype=bool)
    bg0[40:50, 40:50] = False

    # Frame 1: complete character box (area 400, strictly superior)
    f1 = bg_base.copy()
    f1[35:55, 35:55] = 250
    bg1 = np.ones((H, W), dtype=bool)
    bg1[35:55, 35:55] = False

    canvas = bg_base.copy()
    result, claimed, meta = composite_plate_single_pose(
        [f0, f1], [bg0, bg1], canvas, soft_edge_px=0
    )

    assert meta["n_claimed_pixels"] > 0
    assert len(meta["zones"]) == 1
    # Frame 1 has larger coverage so it must be chosen
    assert meta["zones"][0]["chosen_frame"] == 1
    assert np.all(result[35:55, 35:55] == 250)


def test_composite_preserves_canvas_where_warp_padding_has_no_content():
    """Zero-filled warp padding must not claim background plate ownership."""
    H, W = 80, 80
    canvas = np.full((H, W, 3), 41, dtype=np.uint8)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[20:60, 20:60] = 120
    bg = np.ones((H, W), dtype=bool)
    valid = np.zeros((H, W), dtype=bool)
    valid[20:60, 20:60] = True

    result, _claimed, _meta = composite_plate_single_pose(
        [frame], [bg], canvas, warped_valid=[valid]
    )

    assert np.all(result[:20] == 41)
    assert np.all(result[20:60, 20:60] == 120)


def test_composite_uses_canvas_for_single_sample_background():
    """P1 must not turn a one-frame plate contribution into a visible strip."""
    H, W = 40, 40
    canvas = np.full((H, W, 3), 41, dtype=np.uint8)
    frame0 = np.full((H, W, 3), 100, dtype=np.uint8)
    frame1 = frame0.copy()
    bg0 = np.ones((H, W), dtype=bool)
    bg1 = np.ones((H, W), dtype=bool)
    valid1 = np.ones((H, W), dtype=bool)
    valid1[10:30, :] = False

    result, _claimed, _meta = composite_plate_single_pose(
        [frame0, frame1], [bg0, bg1], canvas, warped_valid=[np.ones((H, W), bool), valid1],
        soft_edge_px=0,
    )

    assert np.all(result[:10] == 100)
    assert np.all(result[10:30] == 41)


def test_plate_compositor_p2_multiband_and_edge_preserve():
    """Verify P2 multiband blending runs cleanly over the background plate."""
    H, W = 128, 128
    bg0 = np.full((H, W, 3), 50, dtype=np.uint8)
    bg1 = np.full((H, W, 3), 150, dtype=np.uint8)
    mask0 = np.ones((H, W), dtype=bool)
    mask1 = np.ones((H, W), dtype=bool)

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    result, claimed, meta = composite_plate_single_pose(
        [bg0, bg1],
        [mask0, mask1],
        canvas,
        edge_preserve=True,
        multiband=True,
        multiband_levels=3,
    )
    assert meta.get("multiband_applied") is True
    # Background should be non-zero and blend smoothly
    assert result.mean() > 0


def test_plate_compositor_p4_ternary_uncertainty_exclusion():
    """Verify that uncertain pixels (value 128) are excluded from background plate generation."""
    H, W = 80, 80
    bg_clean = np.full((H, W, 3), 100, dtype=np.uint8)

    # Frame 0: Clean background everywhere
    f0 = bg_clean.copy()
    m0 = np.full((H, W), 255, dtype=np.uint8)

    # Frame 1: Contaminated leak in patch (20:40, 20:40)
    f1 = bg_clean.copy()
    f1[20:40, 20:40] = 255  # Contaminating color
    # Ternary mask: marks the contaminated patch as uncertain (128)
    m1 = np.full((H, W), 255, dtype=np.uint8)
    m1[20:40, 20:40] = 128

    plate, valid = _build_aligned_background_plate([f0, f1], [m0, m1], H, W)
    assert valid.all()
    # Frame 1 is excluded from the uncertain patch, so the plate matches Frame 0.
    assert np.all(plate[20:40, 20:40] == 100)


def test_warp_inputs_preserves_p4_uncertainty_for_plate_path():
    """The P1/P2 path must retain 128 rather than treating it as background."""
    frame = np.full((8, 8, 3), 100, dtype=np.uint8)
    mask = np.full((8, 8), 255, dtype=np.uint8)
    mask[2:6, 2:6] = 128
    affine = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    _frames, warped_bg, _valid = _warp_inputs(
        [frame], [affine], [mask], 8, 8, 1, include_valid=True, preserve_ternary=True
    )

    assert warped_bg[0].dtype == np.uint8
    assert np.all(warped_bg[0][2:6, 2:6] == 128)


def test_joint_gain_uses_boolean_selector_for_p4_masks():
    """Ternary P4 masks must not become uint8 advanced-index selectors."""
    frames = [
        np.full((8, 8, 3), 100, dtype=np.uint8),
        np.full((8, 8, 3), 120, dtype=np.uint8),
    ]
    masks = [np.full((8, 8), 255, dtype=np.uint8) for _ in frames]
    masks[0][2:6, 2:6] = 128

    out = _apply_joint_gain_solve(frames, masks)

    assert len(out) == 2
    assert out[0].shape == frames[0].shape


def test_laplacian_blend_matches_flat_color_feather_and_handles_one_row():
    """Flat plates should reduce to the mask feather, including a one-row band."""
    left = np.full((33, 17, 3), 24, dtype=np.uint8)
    right = np.full_like(left, 224)
    alpha = np.full(left.shape[:2], 0.5, dtype=np.float32)

    blended = _laplacian_blend(left, right, alpha)
    expected = np.rint(
        left * alpha[:, :, None] + right * (1.0 - alpha[:, :, None])
    ).astype(np.uint8)

    assert np.allclose(blended, expected, atol=1)

    one_row = _laplacian_blend(left[:1], right[:1], np.ones((1, left.shape[1]), np.float32))
    assert one_row.shape == (1, left.shape[1], 3)
    assert np.array_equal(one_row, left[:1])


def test_blend_phase_plates_feathers_background_and_keeps_canvas_voids():
    """The phase join follows vertical canvas order without creating a hard seam."""
    H, W = 80, 12
    canvas = np.full((H, W, 3), 17, dtype=np.uint8)
    rows = np.arange(H, dtype=np.uint8)[:, None, None]
    left = np.broadcast_to(30 + rows, canvas.shape).copy()
    right = np.broadcast_to(130 + rows, canvas.shape).copy()
    left_valid = np.zeros((H, W), dtype=bool)
    right_valid = np.zeros((H, W), dtype=bool)
    left_valid[8:57] = True
    right_valid[24:72] = True
    left[~left_valid] = canvas[~left_valid]
    right[~right_valid] = canvas[~right_valid]
    unclaimed = np.full((H, W), -1, dtype=np.int32)

    result, claimed = _blend_phase_plates(
        left, unclaimed, left_valid, right, unclaimed, right_valid, seam_y=40, blend_width=8
    )

    seam_values = result[32:49, W // 2, 0].astype(np.int16)
    assert np.max(np.diff(seam_values)) < 20
    assert result[32, W // 2, 0] < result[48, W // 2, 0]
    assert np.array_equal(result[~(left_valid | right_valid)], canvas[~(left_valid | right_valid)])
    assert np.all(claimed == -1)


def test_blend_phase_plates_preserves_claimed_cel_across_seam():
    """A hero cel crossing the join must not be replaced by a plate blend."""
    H, W = 72, 24
    left = np.full((H, W, 3), 40, dtype=np.uint8)
    right = np.full_like(left, 180)
    valid = np.ones((H, W), dtype=bool)
    left_claimed = np.full((H, W), -1, dtype=np.int32)
    right_claimed = np.full((H, W), -1, dtype=np.int32)
    cel = np.zeros((H, W), dtype=bool)
    cel[27:45, 6:18] = True
    left[cel] = (9, 201, 77)
    left_claimed[cel] = 3

    result, claimed = _blend_phase_plates(
        left, left_claimed, valid, right, right_claimed, valid, seam_y=36, blend_width=8
    )

    # The cel crosses the feather band, but its fully opaque core on the left
    # side must remain the original cel rather than the background blend.
    protected_core = cel & (np.arange(H)[:, None] < 28)
    assert np.array_equal(result[protected_core], left[protected_core])
    assert np.array_equal(claimed[protected_core], left_claimed[protected_core])


def test_composite_plate_multiphase_reverse_order_uses_canvas_order():
    """Physical order, rather than phase ids, determines which plate owns each side."""
    H, W = 72, 16
    canvas = np.full((H, W, 3), 9, dtype=np.uint8)
    phase_zero = np.full_like(canvas, 40)
    phase_one = np.full_like(canvas, 190)
    top_valid = np.zeros((H, W), dtype=bool)
    bottom_valid = np.zeros((H, W), dtype=bool)
    top_valid[:45] = True
    bottom_valid[27:] = True

    result, _claimed, meta = composite_plate_multiphase(
        [phase_zero, phase_one],
        [np.ones((H, W), dtype=bool), np.ones((H, W), dtype=bool)],
        canvas,
        [bottom_valid, top_valid],
        spans=[(0, 0, 0), (1, 1, 1)],
        physical_phase_order=[1, 0],
        edge_preserve=False,
        multiband=False,
    )

    assert meta["phase_order"] == [1, 0]
    assert result[10].mean() > result[62].mean()
    assert abs(result[10].mean() - 190) < abs(result[10].mean() - 40)
    assert abs(result[62].mean() - 40) < abs(result[62].mean() - 190)


def _multiphase_inputs(n_frames=4):
    """Small vertical-pan inputs with two frames per contiguous phase."""
    H, W = 96, 40
    canvas = np.full((H, W, 3), 35, dtype=np.uint8)
    frames = []
    masks = []
    affines = []
    for index, ty in enumerate(range(0, n_frames * 8, 8)):
        frame = np.full((64, W, 3), 80 + index * 10, dtype=np.uint8)
        mask = np.ones((64, W), dtype=bool)
        if index in (1, 2):
            frame[24:40, 12:28] = 220
            mask[24:40, 12:28] = False
        frames.append(frame)
        masks.append(mask)
        affines.append(np.array([[1, 0, 0], [0, 1, ty]], dtype=np.float32))
    return canvas, frames, masks, affines, H, W


def _run_multiphase(canvas, frames, masks, affines, H, W, phase_ids, meta):
    return _composite_foreground(
        frames, masks, canvas, H, W, frames, affines, masks,
        phase_ids=phase_ids, seam_meta_out=meta,
    )


def test_plate_multiphase_forward_uses_global_ownership(monkeypatch):
    monkeypatch.setenv("ASP_PLATE_SINGLE_POSE", "1")
    monkeypatch.setenv("ASP_PLATE_MULTIPHASE", "1")
    canvas, frames, masks, affines, H, W = _multiphase_inputs()
    meta = {}

    result = _run_multiphase(canvas, frames, masks, affines, H, W, [0, 0, 1, 1], meta)

    assert result.shape == canvas.shape
    assert meta["plate_multiphase"] is True
    assert meta["plate_multiphase_direction"] == "forward"
    ownership = meta["plate_ownership"]
    assert [span["phase"] for span in ownership["spans"]] == [0, 1]
    chosen = [zone["chosen_frame"] for span in ownership["spans"] for zone in span["zones"]]
    assert chosen and all(0 <= index < len(frames) for index in chosen)
    assert any(index >= 2 for index in chosen)


def test_plate_multiphase_reverse_pan_joins_canvas_order(monkeypatch):
    monkeypatch.setenv("ASP_PLATE_SINGLE_POSE", "1")
    monkeypatch.setenv("ASP_PLATE_MULTIPHASE", "1")
    canvas, frames, masks, affines, H, W = _multiphase_inputs()
    # Phase ids stay selection ordered; reversing ty is a reverse pan in canvas order.
    for index, affine in enumerate(affines):
        affine[1, 2] = (len(affines) - 1 - index) * 8
    meta = {}

    result = _run_multiphase(canvas, frames, masks, affines, H, W, [0, 0, 1, 1], meta)

    assert result.shape == canvas.shape
    assert meta["plate_multiphase_direction"] == "reverse"
    assert meta["plate_ownership"]["phase_order"] == [1, 0]


def test_plate_multiphase_interleaved_phases_fall_through(monkeypatch):
    monkeypatch.setenv("ASP_PLATE_SINGLE_POSE", "1")
    monkeypatch.setenv("ASP_PLATE_MULTIPHASE", "1")
    canvas, frames, masks, affines, H, W = _multiphase_inputs()
    meta = {}

    _run_multiphase(canvas, frames, masks, affines, H, W, [0, 1, 0, 1], meta)

    assert meta["plate_single_pose_skipped"] == "multiple_phases"
    assert "plate_multiphase" not in meta


def test_plate_multiphase_thin_span_falls_through_without_index_error(monkeypatch):
    monkeypatch.setenv("ASP_PLATE_SINGLE_POSE", "1")
    monkeypatch.setenv("ASP_PLATE_MULTIPHASE", "1")
    canvas, frames, masks, affines, H, W = _multiphase_inputs()
    meta = {}

    _run_multiphase(canvas, frames, masks, affines, H, W, [0, 0, 1, 2], meta)

    assert meta["plate_single_pose_skipped"] == "multiple_phases"
    assert "plate_multiphase" not in meta


def test_plate_multiphase_one_phase_is_single_pose_byte_identical(monkeypatch):
    canvas, frames, masks, affines, H, W = _multiphase_inputs()
    monkeypatch.setenv("ASP_PLATE_SINGLE_POSE", "1")
    monkeypatch.delenv("ASP_PLATE_MULTIPHASE", raising=False)
    single = _run_multiphase(canvas, frames, masks, affines, H, W, [0, 0, 0, 0], {})
    monkeypatch.setenv("ASP_PLATE_MULTIPHASE", "1")
    multi = _run_multiphase(canvas, frames, masks, affines, H, W, [0, 0, 0, 0], {})

    assert np.array_equal(single, multi)


def test_composite_plate_multiphase_single_span_matches_single_pose():
    """The direct one-span multiphase path remains byte-identical to P1."""
    canvas, frames, masks, affines, H, W = _multiphase_inputs()
    warped_frames, warped_bg, warped_valid = _warp_inputs(
        frames, affines, masks, H, W, len(frames), include_valid=True
    )

    single, _single_claimed, _single_meta = composite_plate_single_pose(
        warped_frames, warped_bg, canvas, warped_valid=warped_valid
    )
    multiphase, _multi_claimed, meta = composite_plate_multiphase(
        warped_frames,
        warped_bg,
        canvas,
        warped_valid,
        spans=[(0, 0, len(frames) - 1)],
        physical_phase_order=[0],
        edge_preserve=False,
        multiband=False,
    )

    assert meta["phase_order"] == [0]
    assert np.array_equal(multiphase, single)


def test_plate_multiphase_flag_off_retains_skip(monkeypatch):
    monkeypatch.setenv("ASP_PLATE_SINGLE_POSE", "1")
    monkeypatch.delenv("ASP_PLATE_MULTIPHASE", raising=False)
    canvas, frames, masks, affines, H, W = _multiphase_inputs()
    meta = {}

    _run_multiphase(canvas, frames, masks, affines, H, W, [0, 0, 1, 1], meta)

    assert meta["plate_single_pose_skipped"] == "multiple_phases"
    assert "plate_multiphase" not in meta
