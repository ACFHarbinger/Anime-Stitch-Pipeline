"""Unit tests for fallback routing gate and engine wrapper (#431)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from asp_backend.rendering.wallpaper import (
    RoutingDecision,
    evaluate_routing_gate,
)
from .wallpaper._synthetic_scene import H, W, IDENTITY, add_char, make_background


def test_evaluate_routing_gate_defaults_to_asp_on_planar_pan():
    f0 = make_background(level=100)
    f1 = make_background(level=105)
    f2 = make_background(level=110)

    # Clean linear translation affines (dx=20, dy=0)
    a0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    a1 = np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    a2 = np.array([[1.0, 0.0, 40.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    decision = evaluate_routing_gate([f0, f1, f2], [a0, a1, a2])

    assert isinstance(decision, RoutingDecision)
    assert decision.selected_engine == "asp"
    assert not decision.high_rotation
    assert not decision.wide_baseline
    assert decision.confidence > 0.90


def test_evaluate_routing_gate_detects_high_rotation():
    f0 = make_background(level=100)
    f1 = make_background(level=100)

    # Frame 1 has 15 degree rotation (cos=0.9659, sin=0.2588)
    theta = np.radians(15.0)
    c, s = np.cos(theta), np.sin(theta)
    a0 = IDENTITY
    a1 = np.array([[c, -s, 0.0], [s, c, 0.0]], dtype=np.float32)

    decision = evaluate_routing_gate([f0, f1], [a0, a1])

    assert decision.selected_engine == "hugin"
    assert decision.high_rotation
    assert "high rotation variance" in decision.reason


def test_evaluate_routing_gate_detects_wide_baseline_severe_gradient():
    f0 = make_background(level=50)
    f1 = make_background(level=180)  # Divergence 130 > 35

    # Large translation displacement > 45% of dimension
    a0 = IDENTITY
    a1 = np.array([[1.0, 0.0, 250.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    decision = evaluate_routing_gate([f0, f1], [a0, a1])

    assert decision.selected_engine == "hugin"
    assert decision.wide_baseline
    assert decision.severe_gradient


class TestGainNormalization:
    def test_gain_normalize_flat_clip_unchanged(self):
        from asp_backend.rendering.wallpaper._engine_router import _gain_normalize_frames

        f0 = np.full((64, 64, 3), 100, dtype=np.uint8)
        f1 = np.full((64, 64, 3), 110, dtype=np.uint8)
        out = _gain_normalize_frames([f0, f1])
        assert len(out) == 2
        # Both frames should now share (approximately) the median luminance.
        lums = [float(np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))) for f in out]
        assert abs(lums[0] - lums[1]) <= 1.0  # uint8 rounding tolerance

    def test_gain_normalize_single_frame_passthrough(self):
        from asp_backend.rendering.wallpaper._engine_router import _gain_normalize_frames

        f = np.full((32, 32, 3), 77, dtype=np.uint8)
        assert _gain_normalize_frames([f]) == [f]


class TestPipelineRouting:
    """#431: the engine gate is wired into run_wallpaper_pipeline — a clip
    with a severe lighting gradient routes to the Hugin branch (recorded in
    the session) rather than the native hero-cel branch."""

    def test_severe_gradient_clip_routes_to_hugin(self, tmp_path, monkeypatch):
        import cv2

        from asp_backend.rendering.wallpaper._aspect_framer import frame_wallpaper
        from asp_backend.rendering.wallpaper._engine_router import HuginRouteResult
        from asp_backend.rendering.wallpaper.wallpaper_pipeline import (
            WallpaperResult,
            run_wallpaper_pipeline,
        )

        # Stub the external Hugin toolchain: the test asserts the pipeline
        # *routes* correctly and records the decision, not the toolchain run.
        def _fake_hugin(frames, hero_cel=None, *, target_aspect="16:9", projection=0):
            h, w = frames[0].shape[:2]
            stitch = frames[0].copy()
            valid = np.any(stitch > 0, axis=2)
            framed = frame_wallpaper(
                stitch, valid, (0, 0, w, h), aspect=target_aspect,
            )
            return HuginRouteResult(
                framed=framed, stitch=stitch, composite=None,
                composite_mask=None, hero_bbox_canvas=(0, 0, w, h),
                n_normalized_frames=len(frames),
            )

        monkeypatch.setattr(
            "asp_backend.rendering.wallpaper._engine_router.run_hugin_with_asp_wrappers",
            _fake_hugin,
        )

        path = str(tmp_path / "grad.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(path, fourcc, 24, (320, 240))
        try:
            for i in range(120):
                # Luma 30 -> 70 across 120 frames (divergence 40 > 35);
                # identity camera (no rotation, no baseline) so only the
                # severe-gradient branch can fire.
                frame = np.full((240, 320, 3), 30 + (i // 2), dtype=np.uint8)
                cv2.rectangle(frame, (20, 60), (60, 180), (30, 30, 30), -1)
                vw.write(frame)
        finally:
            vw.release()

        result = run_wallpaper_pipeline(path, str(tmp_path / "o.png"), aspect="16:9")
        assert isinstance(result, WallpaperResult)
        assert result.session.artifacts["routing_engine"] == "hugin"
        assert result.routing_engine == "hugin"
        # The Hugin route produces no ASP plate/composite; framing still works.
        assert result.plate is None
        assert result.composite is None
        assert result.framed.target_aspect == "16:9"
        # Session records the anchored flag from the wrapper.
        assert result.session.artifacts["hugin_hero_anchored"] is False
