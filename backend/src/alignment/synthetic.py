"""Procedural layered synthetic pan/hold fixture generator (M0 — issue #47).

Generates synthetic anime-style pan sequences comprising:
1. Static background plate (rich high-frequency textures / gradient scenery).
2. Configurable camera panning trajectory with known ground-truth velocity (dx, dy).
3. 2–3 held foreground character cels (RGBA layers) with distinct hold intervals.
4. Accurate ground-truth annotations: camera trajectory, inter-frame displacements,
   cel hold IDs, active frame spans, and ground-truth composite panorama.

Enables unit tests to verify translation recovery, phase correlation, and hold
segmentation without requiring external anime video files or GPU dependencies.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclasses.dataclass(frozen=True)
class HeldCel:
    """A foreground character cel with an alpha channel and hold duration."""

    cel_id: str
    hold_id: str
    active_frames: tuple[int, ...]
    world_pos: tuple[int, int]  # (x, y) on the static background plate
    rgba: np.ndarray  # (H, W, 4) uint8
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cel_id": self.cel_id,
            "hold_id": self.hold_id,
            "active_frames": list(self.active_frames),
            "world_pos": list(self.world_pos),
            "size": [int(self.rgba.shape[1]), int(self.rgba.shape[0])],
            "description": self.description,
        }


@dataclasses.dataclass
class SyntheticPanSequence:
    """A generated procedural sequence with frames and complete ground-truth metadata."""

    frames: list[np.ndarray]  # (H, W, 3) BGR uint8
    ground_truth_panorama: np.ndarray  # Full world composite
    background_plate: np.ndarray  # Pure background without cels
    camera_positions: list[tuple[int, int]]  # Top-left (x, y) per frame in world space
    ground_truth_displacements: list[tuple[float, float]]  # (dx, dy) from frame i to i+1
    held_cels: list[HeldCel]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "version": "1.0.0",
            "frame_count": len(self.frames),
            "frame_size": [
                int(self.frames[0].shape[1]) if self.frames else 0,
                int(self.frames[0].shape[0]) if self.frames else 0,
            ],
            "camera_positions": [list(pos) for pos in self.camera_positions],
            "ground_truth_displacements": [
                list(disp) for disp in self.ground_truth_displacements
            ],
            "held_cels": [cel.to_dict() for cel in self.held_cels],
            "metadata": dict(self.metadata),
        }


def _create_textured_background(
    width: int,
    height: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a high-frequency textured background plate for robust feature matching."""
    bg = np.zeros((height, width, 3), dtype=np.uint8)

    # Base vertical color gradient (sky to ground)
    y_coords = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    bg[:, :, 0] = (220 * (1.0 - y_coords * 0.7)).astype(np.uint8)  # Blue
    bg[:, :, 1] = (180 + 50 * np.sin(y_coords * math.pi * 2)).astype(np.uint8)  # Green
    bg[:, :, 2] = (100 + 120 * y_coords).astype(np.uint8)  # Red

    # High-frequency structural grid / architecture lines
    grid_spacing = 32
    for x in range(0, width, grid_spacing):
        cv2.line(bg, (x, 0), (x, height), (70, 70, 80), 1)
    for y in range(0, height, grid_spacing):
        cv2.line(bg, (0, y), (width, y), (70, 70, 80), 1)

    # Scenery elements: high-contrast geometric landmarks
    for i in range(12):
        cx = int(rng.uniform(40, width - 40))
        cy = int(rng.uniform(40, height - 40))
        radius = int(rng.uniform(15, 35))
        color = (
            int(rng.integers(50, 240)),
            int(rng.integers(50, 240)),
            int(rng.integers(50, 240)),
        )
        cv2.circle(bg, (cx, cy), radius, color, -1)
        cv2.circle(bg, (cx, cy), radius, (20, 20, 20), 2)  # Ink outline

    # Diagonal textured hatching for rich 2D gradients
    for diag in range(-height, width, 24):
        pt1 = (max(0, diag), max(0, -diag))
        pt2 = (min(width, diag + height), min(height, height - max(0, diag + height - width)))
        cv2.line(bg, pt1, pt2, (120, 130, 140), 1)

    return bg


def _create_character_cel(
    width: int,
    height: int,
    pose_name: str,
    color_accent: tuple[int, int, int],
) -> np.ndarray:
    """Render an RGBA character cel with solid cel shading and bold dark outlines."""
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    cx, cy = width // 2, height // 2

    # Draw body / torso
    torso_pts = np.array(
        [
            [cx - 30, cy + 50],
            [cx + 30, cy + 50],
            [cx + 20, cy - 20],
            [cx - 20, cy - 20],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(rgba, [torso_pts], (*color_accent, 255))
    cv2.polylines(rgba, [torso_pts], isClosed=True, color=(10, 10, 10, 255), thickness=2)

    # Draw head
    head_center = (cx, cy - 45)
    head_radius = 26
    cv2.circle(rgba, head_center, head_radius, (210, 225, 245, 255), -1)
    cv2.circle(rgba, head_center, head_radius, (10, 10, 10, 255), 2)

    # Draw pose-specific limbs
    if pose_name == "pose_a":
        # Left and right arms standard
        cv2.line(rgba, (cx - 20, cy - 10), (cx - 50, cy + 20), (10, 10, 10, 255), 3)
        cv2.line(rgba, (cx + 20, cy - 10), (cx + 50, cy + 20), (10, 10, 10, 255), 3)
    elif pose_name == "pose_b":
        # Raised right arm
        cv2.line(rgba, (cx - 20, cy - 10), (cx - 45, cy + 30), (10, 10, 10, 255), 3)
        cv2.line(rgba, (cx + 20, cy - 10), (cx + 55, cy - 40), (10, 10, 10, 255), 3)
    else:
        # Cross arms
        cv2.line(rgba, (cx - 20, cy - 10), (cx + 15, cy + 10), (10, 10, 10, 255), 3)
        cv2.line(rgba, (cx + 20, cy - 10), (cx - 15, cy + 10), (10, 10, 10, 255), 3)

    return rgba


def _alpha_blend(
    dst_bgr: np.ndarray,
    src_rgba: np.ndarray,
    offset_x: int,
    offset_y: int,
) -> None:
    """In-place alpha composite of src_rgba onto dst_bgr at (offset_x, offset_y)."""
    h_src, w_src = src_rgba.shape[:2]
    h_dst, w_dst = dst_bgr.shape[:2]

    # Compute intersection rect
    x1_dst = max(0, offset_x)
    y1_dst = max(0, offset_y)
    x2_dst = min(w_dst, offset_x + w_src)
    y2_dst = min(h_dst, offset_y + h_src)

    if x1_dst >= x2_dst or y1_dst >= y2_dst:
        return

    x1_src = x1_dst - offset_x
    y1_src = y1_dst - offset_y
    x2_src = x1_src + (x2_dst - x1_dst)
    y2_src = y1_src + (y2_dst - y1_dst)

    src_crop = src_rgba[y1_src:y2_src, x1_src:x2_src]
    dst_crop = dst_bgr[y1_dst:y2_dst, x1_dst:x2_dst]

    alpha = (src_crop[:, :, 3:4].astype(np.float32)) / 255.0
    src_rgb = src_crop[:, :, :3].astype(np.float32)
    dst_rgb = dst_crop.astype(np.float32)

    blended = src_rgb * alpha + dst_rgb * (1.0 - alpha)
    dst_bgr[y1_dst:y2_dst, x1_dst:x2_dst] = blended.astype(np.uint8)


def generate_layered_pan_sequence(
    num_frames: int = 6,
    frame_width: int = 400,
    frame_height: int = 300,
    pan_dx: float = 0.0,
    pan_dy: float = 40.0,
    seed: int = 42,
) -> SyntheticPanSequence:
    """Procedurally generate a layered synthetic pan sequence with held cels.

    Args:
        num_frames: Number of animation frames to render.
        frame_width: Width of each output frame in pixels.
        frame_height: Height of each output frame in pixels.
        pan_dx: Relative horizontal camera velocity per frame (pixels).
        pan_dy: Relative vertical camera velocity per frame (pixels).
        seed: PRNG seed for deterministic procedural generation.

    Returns:
        SyntheticPanSequence containing rendered frames, ground truth composite,
        camera trajectories, exact displacements, and held cel descriptors.
    """
    rng = np.random.default_rng(seed)

    # Compute required background plate dimensions
    total_pan_x = abs(pan_dx * (num_frames - 1))
    total_pan_y = abs(pan_dy * (num_frames - 1))
    margin = 80
    bg_width = int(frame_width + total_pan_x + 2 * margin)
    bg_height = int(frame_height + total_pan_y + 2 * margin)

    # 1. Generate Static Background Plate
    bg_plate = _create_textured_background(bg_width, bg_height, rng)

    # 2. Compute Camera Trajectory
    start_cam_x = margin if pan_dx >= 0 else int(bg_width - frame_width - margin)
    start_cam_y = margin if pan_dy >= 0 else int(bg_height - frame_height - margin)

    camera_positions: list[tuple[int, int]] = []
    displacements: list[tuple[float, float]] = []

    for i in range(num_frames):
        cam_x = int(round(start_cam_x + i * pan_dx))
        cam_y = int(round(start_cam_y + i * pan_dy))
        camera_positions.append((cam_x, cam_y))

    for i in range(num_frames - 1):
        dx = float(camera_positions[i + 1][0] - camera_positions[i][0])
        dy = float(camera_positions[i + 1][1] - camera_positions[i][1])
        displacements.append((dx, dy))

    # 3. Create Held Character Cels
    cel_w, cel_h = 140, 180
    cels: list[HeldCel] = []

    # Cel 1 (Hold A): Active on first half of frames
    half_pt = num_frames // 2
    cel_1_rgba = _create_character_cel(cel_w, cel_h, "pose_a", (40, 60, 220))
    cel_1_world_pos = (
        start_cam_x + (frame_width - cel_w) // 2,
        start_cam_y + (frame_height - cel_h) // 2,
    )
    cels.append(
        HeldCel(
            cel_id="cel_01",
            hold_id="hold_alpha",
            active_frames=tuple(range(0, half_pt)),
            world_pos=cel_1_world_pos,
            rgba=cel_1_rgba,
            description="Character in pose A (hold 1)",
        )
    )

    # Cel 2 (Hold B): Active on second half of frames
    cel_2_rgba = _create_character_cel(cel_w, cel_h, "pose_b", (200, 80, 40))
    # Slightly shifted along the pan axis to model a distinct held keyframe
    cel_2_world_pos = (
        int(start_cam_x + pan_dx * half_pt) + (frame_width - cel_w) // 2,
        int(start_cam_y + pan_dy * half_pt) + (frame_height - cel_h) // 2,
    )
    cels.append(
        HeldCel(
            cel_id="cel_02",
            hold_id="hold_beta",
            active_frames=tuple(range(half_pt, num_frames)),
            world_pos=cel_2_world_pos,
            rgba=cel_2_rgba,
            description="Character in pose B (hold 2)",
        )
    )

    # 4. Render Individual Camera Frames
    frames: list[np.ndarray] = []
    for frame_idx, (cam_x, cam_y) in enumerate(camera_positions):
        # Extract background crop under camera viewport
        frame = bg_plate[cam_y : cam_y + frame_height, cam_x : cam_x + frame_width].copy()

        # Composite active cels for this frame
        for cel in cels:
            if frame_idx in cel.active_frames:
                # Relative screen coordinates
                screen_x = cel.world_pos[0] - cam_x
                screen_y = cel.world_pos[1] - cam_y
                _alpha_blend(frame, cel.rgba, screen_x, screen_y)

        frames.append(frame)

    # 5. Build Ground-Truth Panorama (World composite of background + all rendered cels)
    gt_panorama = bg_plate.copy()
    for cel in cels:
        _alpha_blend(gt_panorama, cel.rgba, cel.world_pos[0], cel.world_pos[1])

    # Crop ground-truth panorama to the union of camera viewports
    min_x = min(pos[0] for pos in camera_positions)
    min_y = min(pos[1] for pos in camera_positions)
    max_x = max(pos[0] for pos in camera_positions) + frame_width
    max_y = max(pos[1] for pos in camera_positions) + frame_height
    cropped_gt = gt_panorama[min_y:max_y, min_x:max_x].copy()

    return SyntheticPanSequence(
        frames=frames,
        ground_truth_panorama=cropped_gt,
        background_plate=bg_plate,
        camera_positions=camera_positions,
        ground_truth_displacements=displacements,
        held_cels=cels,
        metadata={
            "num_frames": num_frames,
            "pan_dx": pan_dx,
            "pan_dy": pan_dy,
            "seed": seed,
        },
    )


def export_synthetic_sequence(
    sequence: SyntheticPanSequence,
    output_dir: Path | str,
) -> None:
    """Save all frames, ground truth panorama, and metadata manifest to disk."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(sequence.frames):
        cv2.imwrite(str(out_p / f"frame_{i:03d}.png"), frame)

    cv2.imwrite(str(out_p / "ground_truth_panorama.png"), sequence.ground_truth_panorama)
    cv2.imwrite(str(out_p / "background_plate.png"), sequence.background_plate)

    manifest_path = out_p / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(sequence.to_manifest(), f, indent=2)
