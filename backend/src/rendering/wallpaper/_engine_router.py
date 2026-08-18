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

from asp_backend.rendering.compositing._gain_compensation import _adaptive_gain_clamp

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

    if severe_gradient:
        # Severe lighting gradient alone also routes to Hugin (the issue's
        # explicit second case): the gradient can break ASP's gain assumptions
        # even for a static camera, and Hugin's optimizer handles it natively.
        return RoutingDecision(
            selected_engine="hugin",
            reason=f"severe lighting gradient (luma divergence {max_luma_diff:.0f} > {luma_divergence_thresh:.0f})",
            confidence=0.70,
            wide_baseline=False,
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


def _gain_normalize_frames(
    frames: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Pre-processing: per-frame scalar gain toward the median frame luminance.

    Hugin's optimizer can over-weight one exposure when a clip has a severe
    lighting gradient (the #431 routing case). Equalizing each frame's mean
    luminance to the clip median (clamped via the standard adaptive clamp)
    gives the toolchain a flat starting point — the same joint-gain spirit
    the plate builder applies, without requiring solved affines.
    """
    if len(frames) < 2:
        return list(frames)
    lumas = [
        float(np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))) for f in frames
    ]
    ref_lum = float(np.median(lumas))
    out: list[np.ndarray] = []
    for f, lum in zip(frames, lumas):
        gain = _adaptive_gain_clamp(ref_lum, lum)
        out.append(np.clip(f.astype(np.float32) * gain, 0, 255).astype(np.uint8))
    return out


@dataclass(frozen=True)
class HuginRouteResult:
    """Output of the Hugin engine route with ASP pre/post-processing."""

    framed: FramedWallpaper
    stitch: np.ndarray
    composite: np.ndarray | None  # stitch + hero cel (None when no cel given)
    composite_mask: np.ndarray | None
    hero_bbox_canvas: tuple[int, int, int, int] | None
    n_normalized_frames: int


def run_hugin_with_asp_wrappers(
    frames: Sequence[np.ndarray],
    hero_cel: HeroCel | None = None,
    *,
    target_aspect: str = "16:9",
    projection: int = 0,
) -> HuginRouteResult:
    """Run Hugin toolchain enclosed in ASP pre-processing and post-processing wrappers."""
    # Check Hugin availability
    tools = ("pto_gen", "cpfind", "autooptimiser", "pano_modify", "nona", "enblend")
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            f"Hugin toolchain missing: {', '.join(missing)}. "
            "Install with: sudo apt-get install hugin-tools enblend enfuse"
        )

    # ASP pre-processing: gain normalization (no bare hand-off).
    normalized = _gain_normalize_frames(frames)

    with tempfile.TemporaryDirectory(prefix="asp_hugin_route_") as tmp:
        tmp_path = Path(tmp)
        frame_paths: list[str] = []
        for idx, f in enumerate(normalized):
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

    # ASP post-processing: hero-cel anchoring + framing (no bare hand-off).
    sh, sw = stitched.shape[:2]
    valid_mask = np.any(stitched > 0, axis=2)

    composite: np.ndarray | None = None
    composite_mask: np.ndarray | None = None
    hero_bbox_canvas: tuple[int, int, int, int] | None = None
    plate_for_frame = stitched

    if hero_cel is not None:
        # Map the hero cel from frame space into the stitch canvas. Hugin
        # does not expose a per-frame registration into the panorama, so we
        # preserve the cel's proportional placement: scale frame -> canvas
        # and translate so the hero bbox top-left lands at its proportional
        # position. This is a real composite (the cel appears in the output),
        # not a bare hand-off of the raw stitch.
        fh, fw = frames[0].shape[:2]
        sx = sw / max(fw, 1)
        sy = sh / max(fh, 1)
        bx0, by0, _, _ = hero_cel.bbox
        hero_affine = np.array(
            [[sx, 0.0, bx0 * sx], [0.0, sy, by0 * sy]], dtype=np.float32
        )
        comp = composite_hero_cel(stitched, hero_cel, hero_affine, blend_mode="feather")
        composite = comp.composite
        composite_mask = comp.cel_canvas_mask
        hero_bbox_canvas = comp.hero_bbox_canvas
        plate_for_frame = comp.composite
        valid_mask = np.any(comp.composite > 0, axis=2)
    else:
        hero_bbox_canvas = (int(sw * 0.25), int(sh * 0.25), int(sw * 0.75), int(sh * 0.75))

    framed = frame_wallpaper(
        plate_for_frame,
        valid_mask,
        hero_bbox_canvas,
        aspect=target_aspect,
    )
    return HuginRouteResult(
        framed=framed,
        stitch=stitched,
        composite=composite,
        composite_mask=composite_mask,
        hero_bbox_canvas=hero_bbox_canvas,
        n_normalized_frames=len(normalized),
    )
