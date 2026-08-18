"""Wallpaper-mode rendering (Slice 1: background plate builder, #427)."""

from ._geometry import fit_window_containing_bbox, parse_aspect_ratio, union_bbox
from ._plate_builder import BackgroundPlate, build_background_plate

__all__ = [
    "BackgroundPlate",
    "build_background_plate",
    "parse_aspect_ratio",
    "union_bbox",
    "fit_window_containing_bbox",
]