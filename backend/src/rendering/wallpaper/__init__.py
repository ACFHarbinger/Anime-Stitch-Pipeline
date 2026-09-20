"""Wallpaper-mode rendering (Slice 1: #426-#430; Slice 2: #431; #433 unwired)."""

from ._aspect_framer import FramedWallpaper, frame_wallpaper
from ._cel_compositor import CelCompositeResult, composite_hero_cel
from ._engine_router import (
    HuginRouteResult,
    RoutingDecision,
    evaluate_routing_gate,
    run_hugin_with_asp_wrappers,
)
from ._geometry import fit_window_containing_bbox, parse_aspect_ratio, union_bbox
from ._hero_selector import HeroCel, score_candidate_frame, select_hero_cel
from ._plate_builder import BackgroundPlate, build_background_plate
from ._plate_frame_cover import (
    PlateFrameSelection,
    maybe_select_plate_frames,
    plate_frame_cover_enabled,
    select_plate_frames,
)
from .wallpaper_pipeline import (
    STAGE_HERO_SELECT,
    STAGE_PLATE,
    WallpaperResult,
    estimate_wallpaper_time,
    run_wallpaper_pipeline,
)

__all__ = [
    "BackgroundPlate",
    "CelCompositeResult",
    "FramedWallpaper",
    "HeroCel",
    "HuginRouteResult",
    "PlateFrameSelection",
    "RoutingDecision",
    "STAGE_HERO_SELECT",
    "STAGE_PLATE",
    "WallpaperResult",
    "build_background_plate",
    "composite_hero_cel",
    "estimate_wallpaper_time",
    "evaluate_routing_gate",
    "fit_window_containing_bbox",
    "frame_wallpaper",
    "maybe_select_plate_frames",
    "parse_aspect_ratio",
    "plate_frame_cover_enabled",
    "run_hugin_with_asp_wrappers",
    "run_wallpaper_pipeline",
    "score_candidate_frame",
    "select_hero_cel",
    "select_plate_frames",
    "union_bbox",
]