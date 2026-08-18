"""Wallpaper-mode orchestrator (#430).

Wires the four Slice-1 wallpaper modules into one pipeline, registered
into the shared :class:`PipelineSession` stage protocol (M1a/b/c) rather
than a parallel runner:

    clip frames -> hero cel (select_hero_cel)
                 -> background plate (build_background_plate)
                 -> rigid composite (composite_hero_cel)
                 -> aspect framing (frame_wallpaper)

Stage ids reuse :class:`PipelineStage` where a wallpaper stage maps onto
an existing one (LOAD/CANVAS/COMPOSITE/CROP/INPAINT/SAVE) and introduce
wallpaper-specific ids (HERO_SELECT, PLATE) as plain strings, consistent
with the protocol's `start_stage(str)` tolerance.

Slice-1 scope notes (from the locked roadmap):
- Frames are sampled from the clip at a fixed interval; the static-camera
  model (asp_test97: camera static, character walks) uses identity affines
  on a shared canvas. Non-identity registration is a Slice-2 extension.
- FG masks are required input for hero selection. When a caller cannot
  provide them (no mask model wired yet), a fallback threshold mask is
  used and recorded as a stage fallback.
- `--estimate` prints a predicted wall-clock from per-stage measured
  costs and exits without running.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from asp_backend.core.pipeline.session import PipelineSession, PipelineStage
from asp_backend.rendering.wallpaper._aspect_framer import FramedWallpaper, frame_wallpaper
from asp_backend.rendering.wallpaper._cel_compositor import CelCompositeResult, composite_hero_cel
from asp_backend.rendering.wallpaper._hero_selector import HeroCel, select_hero_cel
from asp_backend.rendering.wallpaper._plate_builder import BackgroundPlate, build_background_plate

# Wallpaper-specific stage ids (beyond the existing PipelineStage set).
STAGE_HERO_SELECT = "wallpaper_hero_select"
STAGE_PLATE = "wallpaper_plate"

# Default canvas the wallpaper path registers frames into (identity affines).
DEFAULT_CANVAS_H = 1080
DEFAULT_CANVAS_W = 1920

# Frame sampling (Slice-1: static camera, character walks).
DEFAULT_FRAME_INTERVAL = 30  # every ~1s at 30fps
MAX_SAMPLED_FRAMES = 60

# Measured per-stage cost model for --estimate (seconds per frame / flat).
# Rough empirical constants; refined once #430's telemetry lands.
_ESTIMATE_COST = {
    STAGE_HERO_SELECT: 0.010,  # per frame
    STAGE_PLATE: 0.020,       # per frame
    "composite": 0.050,       # flat
    "crop": 0.010,            # flat
}


@dataclass
class WallpaperResult:
    """Final wallpaper output plus session observability."""

    wallpaper: np.ndarray  # (H, W, 3) uint8 BGR
    output_path: Path
    hero: HeroCel
    plate: BackgroundPlate
    composite: CelCompositeResult
    framed: FramedWallpaper
    session: PipelineSession


def _sample_frames(clip: str, *, interval: int = DEFAULT_FRAME_INTERVAL, max_frames: int = MAX_SAMPLED_FRAMES) -> list[np.ndarray]:
    """Read frames from the clip at a fixed interval (BGR uint8)."""
    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video clip: {clip}")
    frames: list[np.ndarray] = []
    idx = 0
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % interval == 0:
                frames.append(frame)
            idx += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"no frames extracted from clip: {clip}")
    return frames


def _fallback_fg_masks(frames: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Threshold-based foreground masks (Slice-1 fallback when no mask
    model is wired). Marks high-gradient regions as foreground."""
    masks: list[np.ndarray] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        # Dilate the edges to capture the cel body, not just the lineart.
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=2)
        masks.append((mask > 0).astype(np.uint8) * 255)
    return masks


def estimate_wallpaper_time(
    *,
    n_frames: int,
    aspect: str = "16:9",
    quality: str = "balanced",
) -> dict[str, float]:
    """Predicted wall-clock for the current parameters (per-frame costs)."""
    q_mult = {"fast": 0.6, "balanced": 1.0, "max": 2.0}[quality]
    hero = _ESTIMATE_COST[STAGE_HERO_SELECT] * n_frames
    plate = _ESTIMATE_COST[STAGE_PLATE] * n_frames
    comp = _ESTIMATE_COST["composite"]
    crop = _ESTIMATE_COST["crop"]
    return {
        "hero_select_s": round(hero * q_mult, 3),
        "plate_s": round(plate * q_mult, 3),
        "composite_s": round(comp * q_mult, 3),
        "aspect_frame_s": round(crop * q_mult, 3),
        "total_s": round((hero + plate + comp + crop) * q_mult, 3),
    }


def run_wallpaper_pipeline(
    clip: str,
    output_path: str | Path,
    *,
    aspect: str = "16:9",
    quality: str = "balanced",
    fg_masks: Optional[Sequence[np.ndarray]] = None,
    override_frame_idx: Optional[int] = None,
    canvas_h: int = DEFAULT_CANVAS_H,
    canvas_w: int = DEFAULT_CANVAS_W,
    pause_hook: Any = None,
) -> WallpaperResult:
    """Generate a wallpaper from a clip via Hero-Cel compositing (#430).

    Runs the four Slice-1 modules under one PipelineSession and returns the
    result plus observability. `fg_masks` may be supplied (mask model
    wired); otherwise a threshold fallback is used and recorded.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    session = PipelineSession.create(
        image_paths=[clip],
        output_path=str(out_path),
        config={"mode": "wallpaper", "aspect": aspect, "quality": quality},
        pause_hook=pause_hook,
    )

    # 1. Load frames.
    session.start_stage(PipelineStage.LOAD, clip=clip, quality=quality)
    frames = _sample_frames(clip)
    session.note_geometry(PipelineStage.LOAD, width=canvas_w, height=canvas_h, n_frames=len(frames))
    session.complete_stage(PipelineStage.LOAD, n_frames=len(frames))

    # 2. Hero selection.
    session.start_stage(STAGE_HERO_SELECT, n_frames=len(frames))
    used_fallback_masks = fg_masks is None
    if fg_masks is None:
        fg_masks = _fallback_fg_masks(frames)
    hero = select_hero_cel(frames, fg_masks, override_frame_idx=override_frame_idx)
    session.record_artifact("hero_frame_idx", hero.frame_idx)
    session.record_artifact("hero_score", hero.score)
    session.complete_stage(
        STAGE_HERO_SELECT,
        hero_frame=hero.frame_idx,
        score=hero.score,
        fallback="threshold-mask" if used_fallback_masks else None,
    )

    # 3. Background plate (identity affines; static-camera Slice-1 model).
    session.start_stage(STAGE_PLATE, canvas=(canvas_h, canvas_w))
    affines = [np.eye(3)[:2].astype(np.float32)] * len(frames)
    # Hero footprint: mask of the hero cel's canvas-space region (identity
    # affine => frame-space == canvas-space at the hero position).
    hero_bbox = hero.bbox
    footprint = np.zeros((canvas_h, canvas_w), dtype=bool)
    hx0, hy0, hx1, hy1 = hero_bbox
    hx1 = min(hx1, canvas_w)
    hy1 = min(hy1, canvas_h)
    if hx0 < canvas_w and hy0 < canvas_h:
        footprint[hy0:hy1, hx0:hx1] = hero.alpha_mask[hy0:hy1, hx0:hx1] > 0 if hero.alpha_mask.shape == (canvas_h, canvas_w) else True
    plate = build_background_plate(
        frames, affines, list(fg_masks), footprint, (canvas_h, canvas_w),
        hero_frame_idx=hero.frame_idx,
    )
    session.note_geometry(STAGE_PLATE, width=canvas_w, height=canvas_h, n_frames=len(frames))
    session.record_artifact("plate_void_ratio", plate.void_ratio)
    session.complete_stage(STAGE_PLATE, void_ratio=plate.void_ratio)

    # 4. Rigid composite at original position.
    session.start_stage(PipelineStage.COMPOSITE, blend="feather")
    hero_affine = affines[hero.frame_idx]
    composite = composite_hero_cel(plate.plate, hero, hero_affine, blend_mode="feather")
    session.record_artifact("composite_method", composite.blend_method)
    session.complete_stage(PipelineStage.COMPOSITE, blend_method=composite.blend_method)

    # 5. Aspect framing.
    session.start_stage(PipelineStage.CROP, aspect=aspect)
    framed = frame_wallpaper(
        composite.composite,
        plate.valid_mask,
        composite.hero_bbox_canvas,
        aspect=aspect,
    )
    session.note_geometry(PipelineStage.CROP, width=framed.wallpaper.shape[1], height=framed.wallpaper.shape[0])
    session.record_artifact("target_aspect", aspect)
    session.record_artifact("needs_generative_outpaint", framed.needs_generative_outpaint)
    session.complete_stage(PipelineStage.CROP, aspect=aspect, needs_outpaint=framed.needs_generative_outpaint)

    # 6. Save.
    session.start_stage(PipelineStage.SAVE, path=str(out_path))
    cv2.imwrite(str(out_path), framed.wallpaper)
    session.complete_stage(PipelineStage.SAVE, path=str(out_path))
    session.finished_at = time.perf_counter()
    session.success = True

    return WallpaperResult(
        wallpaper=framed.wallpaper,
        output_path=out_path,
        hero=hero,
        plate=plate,
        composite=composite,
        framed=framed,
        session=session,
    )


__all__ = [
    "STAGE_HERO_SELECT",
    "STAGE_PLATE",
    "WallpaperResult",
    "estimate_wallpaper_time",
    "run_wallpaper_pipeline",
]
