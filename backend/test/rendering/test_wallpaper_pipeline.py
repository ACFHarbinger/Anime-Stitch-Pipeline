"""Tests for the wallpaper orchestrator (#430).

Wires the four Slice-1 wallpaper modules into one PipelineSession-driven
pipeline (load -> hero_select -> plate -> composite -> crop -> save).
Synthetic clip + threshold-mask fallback only -- no GPU, no model deps.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import cv2

from asp_backend.rendering.wallpaper.wallpaper_pipeline import (
    STAGE_HERO_SELECT,
    STAGE_PLATE,
    WallpaperResult,
    estimate_wallpaper_time,
    run_wallpaper_pipeline,
)


def _make_clip(path: str, n_frames: int = 48) -> str:
    """Static background + a rectangle character walking horizontally."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, 24, (320, 240))
    bg = np.full((240, 320, 3), 130, dtype=np.uint8)
    try:
        for i in range(n_frames):
            frame = bg.copy()
            cv2.rectangle(frame, (20 + i * 4, 60), (60 + i * 4, 180), (30, 30, 30), -1)
            vw.write(frame)
    finally:
        vw.release()
    return path


class TestWallpaperPipeline:
    def test_full_pipeline_produces_wallpaper(self, tmp_path):
        clip = _make_clip(str(tmp_path / "walk.mp4"))
        out = tmp_path / "out.png"

        result = run_wallpaper_pipeline(clip, str(out), aspect="16:9", quality="balanced")

        assert isinstance(result, WallpaperResult)
        assert out.exists()
        img = cv2.imread(str(out))
        assert img is not None and img.shape[2] == 3
        # Session stage trace: load -> route -> hero -> plate -> composite
        # -> crop -> save (the route stage is the #431 engine gate).
        names = [s.name for s in result.session.stages]
        assert names == [
            "load",
            "wallpaper_route",
            STAGE_HERO_SELECT,
            STAGE_PLATE,
            "composite",
            "crop",
            "save",
        ]
        assert result.session.success is True
        assert result.routing_engine == "asp"
        assert result.session.artifacts["routing_engine"] == "asp"

    def test_hero_selected_and_recorded(self, tmp_path):
        clip = _make_clip(str(tmp_path / "walk.mp4"))
        result = run_wallpaper_pipeline(clip, str(tmp_path / "o.png"), aspect="9:16")
        assert 0 <= result.hero.frame_idx < 48
        assert result.hero.score > 0
        assert result.session.artifacts["hero_frame_idx"] == result.hero.frame_idx

    def test_threshold_mask_fallback_recorded(self, tmp_path):
        clip = _make_clip(str(tmp_path / "walk.mp4"))
        result = run_wallpaper_pipeline(clip, str(tmp_path / "o.png"))
        hero_stage = next(s for s in result.session.stages if s.name == STAGE_HERO_SELECT)
        assert hero_stage.fallback == "threshold-mask"

    def test_estimate_matches_parameter_scaling(self):
        est = estimate_wallpaper_time(n_frames=40, quality="balanced")
        assert est["total_s"] > 0
        fast = estimate_wallpaper_time(n_frames=40, quality="fast")
        assert fast["total_s"] < est["total_s"]
        # Stage sum equals total.
        assert abs(
            est["hero_select_s"] + est["plate_s"] + est["composite_s"] + est["aspect_frame_s"]
            - est["total_s"]
        ) < 1e-6

    def test_missing_clip_raises(self, tmp_path):
        with pytest.raises(Exception):
            run_wallpaper_pipeline(str(tmp_path / "nope.mp4"), str(tmp_path / "o.png"))
