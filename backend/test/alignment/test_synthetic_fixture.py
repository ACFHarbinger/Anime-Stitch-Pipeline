"""Unit tests for procedural layered synthetic pan/hold fixture generator (M0c — issue #47)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

try:
    from asp_backend.alignment.synthetic import (
        HeldCel,
        SyntheticPanSequence,
        export_synthetic_sequence,
        generate_layered_pan_sequence,
    )
except ImportError:
    from backend.src.alignment.synthetic import (
        HeldCel,
        SyntheticPanSequence,
        export_synthetic_sequence,
        generate_layered_pan_sequence,
    )


def test_synthetic_pan_generation_dimensions_and_counts():
    num_frames = 6
    w, h = 320, 240
    seq = generate_layered_pan_sequence(
        num_frames=num_frames,
        frame_width=w,
        frame_height=h,
        pan_dx=0.0,
        pan_dy=35.0,
        seed=123,
    )

    assert len(seq.frames) == num_frames
    for frame in seq.frames:
        assert frame.shape == (h, w, 3)
        assert frame.dtype == np.uint8

    assert len(seq.camera_positions) == num_frames
    assert len(seq.ground_truth_displacements) == num_frames - 1
    assert len(seq.held_cels) == 2


def test_synthetic_camera_displacements_consistency():
    seq = generate_layered_pan_sequence(
        num_frames=5,
        frame_width=200,
        frame_height=150,
        pan_dx=10.0,
        pan_dy=25.0,
        seed=42,
    )

    for i, (dx, dy) in enumerate(seq.ground_truth_displacements):
        expected_dx = float(seq.camera_positions[i + 1][0] - seq.camera_positions[i][0])
        expected_dy = float(seq.camera_positions[i + 1][1] - seq.camera_positions[i][1])
        assert dx == pytest.approx(expected_dx)
        assert dy == pytest.approx(expected_dy)
        assert dy == pytest.approx(25.0)


def test_held_cel_attributes_and_hold_ids():
    seq = generate_layered_pan_sequence(num_frames=6, seed=99)
    cels = seq.held_cels

    assert len(cels) == 2
    cel_a, cel_b = cels[0], cels[1]

    assert cel_a.hold_id == "hold_alpha"
    assert cel_b.hold_id == "hold_beta"
    assert cel_a.active_frames == (0, 1, 2)
    assert cel_b.active_frames == (3, 4, 5)

    assert cel_a.rgba.shape[2] == 4
    assert cel_b.rgba.shape[2] == 4


def test_translation_recovery_against_known_velocity():
    """Verify that phase correlation recovers the known procedural displacement."""
    pan_dy = 30.0
    seq = generate_layered_pan_sequence(
        num_frames=4,
        frame_width=400,
        frame_height=300,
        pan_dx=0.0,
        pan_dy=pan_dy,
        seed=77,
    )

    # Use phase correlation on background luminance between consecutive frames
    for i in range(len(seq.frames) - 1):
        f1_gray = cv2.cvtColor(seq.frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        f2_gray = cv2.cvtColor(seq.frames[i + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Apply Hanning window to mitigate edge artifacts
        hanning = cv2.createHanningWindow((f1_gray.shape[1], f1_gray.shape[0]), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(f1_gray, f2_gray, window=hanning)

        recovered_dx, recovered_dy = shift
        # Expected displacement between frame i and frame i+1
        # In image coordinates, moving camera down by dy shifts image content up by -dy
        assert abs(recovered_dx - 0.0) < 1.5
        assert abs(abs(recovered_dy) - pan_dy) < 1.5


def test_export_and_manifest_serialization(tmp_path):
    seq = generate_layered_pan_sequence(num_frames=4, seed=55)
    out_dir = tmp_path / "synthetic_pan_run"

    export_synthetic_sequence(seq, out_dir)

    assert (out_dir / "frame_000.png").exists()
    assert (out_dir / "frame_003.png").exists()
    assert (out_dir / "ground_truth_panorama.png").exists()
    assert (out_dir / "background_plate.png").exists()

    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()

    with open(manifest_file, encoding="utf-8") as f:
        doc = json.load(f)

    assert doc["version"] == "1.0.0"
    assert doc["frame_count"] == 4
    assert len(doc["camera_positions"]) == 4
    assert len(doc["held_cels"]) == 2
    assert doc["held_cels"][0]["hold_id"] == "hold_alpha"
