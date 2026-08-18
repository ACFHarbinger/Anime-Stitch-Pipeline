"""Unit tests for fallback routing gate and engine wrapper (#431)."""

from __future__ import annotations

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
