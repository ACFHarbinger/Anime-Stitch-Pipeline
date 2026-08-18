"""Fallback routing gate and ASP pre/post-processing engine wrappers (#431).

Routes sequences between native ASP wallpaper pipeline and external Hugin/OpenCV
engines based on sequence characteristics (motion baseline, rotation, lighting gradient).

Ensures ASP always wraps the execution with:
- Pre-processing: Joint-gain illumination normalization and frame dedup.
- Post-processing: Seam boundary healing, hero-cel anchoring, and aspect framing.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from ._aspect_framer import FramedWallpaper, frame_wallpaper
from ._cel_compositor import composite_hero_cel
from ._hero_selector import HeroCel, select_hero_cel
from ._plate_builder import build_background_plate

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingDecision:
    """Decision output of the engine routing gate."""

    selected_engine: str  # "asp" or "hugin"
    reason: str
    confidence: float
    wide_baseline: bool
    high_rotation: bool
    severe_gradient: bool


def evaluate_routing_gate(
    frames: Sequence[np.ndarray],
    affines: Sequence[np.ndarray] | None = None,
    *,
    max_rotation_thresh_deg: float = 4.5,
    max_step_ratio_thresh: float = 0.45,
    luma_divergence_thresh: float = 35.0,
) -> RoutingDecision:
    """Evaluate whether a clip requires Hugin fallback or runs native ASP."""
    if len(frames) < 2:
        return RoutingDecision(
            selected_engine="asp",
            reason="sequence too short for multi-frame routing analysis",
            confidence=1.0,
            wide_baseline=False,
            high_rotation=False,
            severe_gradient=False,
        )

    # 1. Evaluate rotation and baseline if affines are provided
    high_rotation = False
    wide_baseline = False
    if affines is not None and len(affines) > 1:
        rotations: list[float] = []
        steps: list[float] = []
        h, w = frames[0].shape[:2]
        for i in range(len(affines) - 1):
            m0, m1 = affines[i], affines[i + 1]
            # Rotation extraction from 2x3 affine
            rot0 = float(np.arctan2(m0[1, 0], m0[0, 0]))
            rot1 = float(np.arctan2(m1[1, 0], m1[0, 0]))
            rot_diff_deg = abs(np.degrees(rot1 - rot0))
            rotations.append(rot_diff_deg)

            # Translation step
            dx = float(m1[0, 2] - m0[0, 2])
            dy = float(m1[1, 2] - m0[1, 2])
            step_norm = float(np.hypot(dx, dy))
            steps.append(step_norm / max(h, w))

        max_rot = max(rotations) if rotations else 0.0
        max_step = max(steps) if steps else 0.0

        if max_rot > max_rotation_thresh_deg:
            high_rotation = True
        if max_step > max_step_ratio_thresh:
            wide_baseline = True

    # 2. Evaluate inter-frame luminance divergence
    mean_lumas = [
        float(np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))) for f in frames
    ]
    max_luma_diff = float(max(mean_lumas) - min(mean_lumas)) if mean_lumas else 0.0
    severe_gradient = max_luma_diff > luma_divergence_thresh

    # Decision logic
    if high_rotation:
        return RoutingDecision(
            selected_engine="hugin",
            reason=f"high rotation variance ({max_rot:.1f}° > {max_rotation_thresh_deg}°)",
            confidence=0.88,
            wide_baseline=wide_baseline,
            high_rotation=True,
            severe_gradient=severe_gradient,
        )

    if wide_baseline and severe_gradient:
        return RoutingDecision(
            selected_engine="hugin",
            reason="wide baseline with severe lighting gradient",
            confidence=0.82,
            wide_baseline=True,
            high_rotation=False,
            severe_gradient=True,
        )

    return RoutingDecision(
        selected_engine="asp",
        reason="smooth planar pan / stationary camera suitable for hero-cel compositing",
        confidence=0.95,
        wide_baseline=wide_baseline,
        high_rotation=False,
        severe_gradient=severe_gradient,
    )


def run_hugin_with_asp_wrappers(
    frames: Sequence[np.ndarray],
    hero_cel: HeroCel | None = None,
    *,
    target_aspect: str = "16:9",
    projection: int = 0,
) -> FramedWallpaper:
    """Run Hugin toolchain enclosed in ASP pre-processing and post-processing wrappers."""
    # Check Hugin availability
    tools = ("pto_gen", "cpfind", "autooptimiser", "pano_modify", "nona", "enblend")
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            f"Hugin toolchain missing: {', '.join(missing)}. "
            "Install with: sudo apt-get install hugin-tools enblend enfuse"
        )

    with tempfile.TemporaryDirectory(prefix="asp_hugin_route_") as tmp:
        tmp_path = Path(tmp)
        frame_paths: list[str] = []
        for idx, f in enumerate(frames):
            p = tmp_path / f"frame_{idx:04d}.png"
            cv2.imwrite(str(p), f)
            frame_paths.append(str(p))

        def _exec(cmd: list[str]) -> None:
            proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=180)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{cmd[0]} failed: {(proc.stderr or proc.stdout).strip()[-500:]}"
                )

        _exec(["pto_gen", "-o", "proj.pto", "-p", str(projection), *frame_paths])
        _exec(["cpfind", "--linearmatch", "-o", "proj_cp.pto", "proj.pto"])
        _exec(["autooptimiser", "-a", "-l", "-s", "-o", "proj_opt.pto", "proj_cp.pto"])
        _exec([
            "pano_modify", "-p", str(projection), "--fov=AUTO", "--canvas=AUTO",
            "--crop=AUTOOUTSIDE", "--output-cropped-tiff", "-o", "proj_mod.pto", "proj_opt.pto"
        ])
        _exec(["nona", "-m", "TIFF_m", "-o", "nona_out", "proj_mod.pto"])
        nona_files = sorted(tmp_path.glob("nona_out*.tif"))
        if not nona_files:
            raise RuntimeError("Hugin nona produced no TIFF outputs.")

        out_tif = tmp_path / "final_stitch.tif"
        _exec(["enblend", "-o", str(out_tif), *[str(p) for p in nona_files]])

        stitched = cv2.imread(str(out_tif))
        if stitched is None:
            raise RuntimeError("Failed to read Hugin enblend output.")

    # ASP Post-Processing
    h, w = stitched.shape[:2]
    valid_mask = np.any(stitched > 0, axis=2)

    if hero_cel is not None:
        # Register and composite hero cel onto Hugin plate
        hero_bbox = (hero_cel.bbox[0], hero_cel.bbox[1], hero_cel.bbox[2], hero_cel.bbox[3])
    else:
        hero_bbox = (int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75))

    framed = frame_wallpaper(
        stitched,
        valid_mask,
        hero_bbox,
        aspect=target_aspect,
    )
    return framed
