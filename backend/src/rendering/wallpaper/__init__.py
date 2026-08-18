"""Wallpaper-mode rendering (Slice 1: #426, #427, #428, #429)."""

from ._aspect_framer import FramedWallpaper, frame_wallpaper
from ._cel_compositor import CelCompositeResult, composite_hero_cel
from ._geometry import fit_window_containing_bbox, parse_aspect_ratio, union_bbox
from ._hero_selector import HeroCel, score_candidate_frame, select_hero_cel
from ._plate_builder import BackgroundPlate, build_background_plate

__all__ = [
    "BackgroundPlate",
    "CelCompositeResult",
    "FramedWallpaper",
    "HeroCel",
    "build_background_plate",
    "composite_hero_cel",
    "fit_window_containing_bbox",
    "frame_wallpaper",
    "parse_aspect_ratio",
    "score_candidate_frame",
    "select_hero_cel",
    "union_bbox",
]