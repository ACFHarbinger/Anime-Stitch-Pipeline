"""Unit tests for the 2D wave-correction candidate (M2 roadmap critique).

Verifies the trajectory-straightening contract: a bowed strip path is
straightened, along-axis spacing is preserved, and straight paths are
unchanged.
"""

from __future__ import annotations

import numpy as np
from asp_backend.core.pipeline._wave_correction import wave_correct_affines


def _affines(tx, ty):
    return [np.array([[1.0, 0.0, float(x)], [0.0, 1.0, float(y)]], dtype=np.float32)
            for x, y in zip(tx, ty, strict=True)]


def test_straight_trajectory_is_unchanged():
    tx = [float(i) * 100.0 for i in range(6)]
    ty = [5.0] * 6  # perfectly straight horizontal pan
    aff = _affines(tx, ty)
    out = wave_correct_affines(aff, kind="auto")
    np.testing.assert_allclose([a[0, 2] for a in out], tx)
    np.testing.assert_allclose([a[1, 2] for a in out], ty)


def _collinearity_residual(aff):
    pts = np.array([[a[0, 2], a[1, 2]] for a in aff], dtype=np.float64)
    # perpendicular distance of each point to the best-fit line
    d = np.diff(pts, axis=0)
    u = d.sum(axis=0)
    u /= np.linalg.norm(u)
    u_perp = np.array([-u[1], u[0]])
    return pts @ u_perp  # transverse component per point


def test_removes_bow_linear_and_arch():
    # Horizontal pan that bows vertically (arch): middle frames drift in y.
    tx = [float(i) * 100.0 for i in range(6)]
    ty = [0.0, 15.0, 20.0, 20.0, 15.0, 0.0]
    aff = _affines(tx, ty)
    out = wave_correct_affines(aff, kind="auto")
    across = _collinearity_residual(out)
    # After correction the trajectory must be collinear: residual transverse
    # component of the best-fit line is ~0.
    assert np.max(np.abs(across - across[0])) < 1e-3, across
    # Along-axis positions are preserved -> spacing never grows.
    np.testing.assert_allclose([a[0, 2] for a in out], tx)


def test_never_grows_adjacent_gaps():
    # The projection can only shrink transverse deviation, so adjacent
    # Euclidean gaps never increase (min_gap cannot be recovered by it).
    tx = [float(i) * 100.0 for i in range(6)]
    ty = [0.0, 15.0, 20.0, 20.0, 15.0, 0.0]
    aff = _affines(tx, ty)
    out = wave_correct_affines(aff, kind="auto")
    pts_b = np.array([[a[0, 2], a[1, 2]] for a in aff], dtype=np.float64)
    pts_a = np.array([[a[0, 2], a[1, 2]] for a in out], dtype=np.float64)
    gaps_b = np.linalg.norm(np.diff(pts_b, axis=0), axis=1)
    gaps_a = np.linalg.norm(np.diff(pts_a, axis=0), axis=1)
    assert np.all(gaps_a <= gaps_b + 1e-3)


def test_along_axis_spacing_preserved_for_diagonal():
    # Diagonal pan with a transverse bow; the projection must not grow any
    # adjacent gap and must straighten the trajectory.
    tx = [float(i) * 50.0 for i in range(7)]
    ty = [float(i) * 30.0 for i in range(7)]
    ty = [v + (8.0 if 2 <= i <= 4 else 0.0) for i, v in enumerate(ty)]
    aff = _affines(tx, ty)
    out = wave_correct_affines(aff, kind="auto")
    pts_b = np.array([[a[0, 2], a[1, 2]] for a in aff], dtype=np.float64)
    pts_a = np.array([[a[0, 2], a[1, 2]] for a in out], dtype=np.float64)
    gaps_b = np.linalg.norm(np.diff(pts_b, axis=0), axis=1)
    gaps_a = np.linalg.norm(np.diff(pts_a, axis=0), axis=1)
    assert np.all(gaps_a <= gaps_b + 1e-3)
    across = _collinearity_residual(out)
    assert np.max(np.abs(across - across[0])) < 1e-3, across


def test_handles_dict_affines():
    aff = [{"frame": i, "tx": float(i) * 100.0, "ty": (20.0 if i in (2, 3) else 0.0),
            "a": 1.0, "b": 0.0} for i in range(5)]
    out = wave_correct_affines(aff, kind="auto")
    across = _collinearity_residual(
        [np.array([[1, 0, a["tx"]], [0, 1, a["ty"]]], dtype=np.float32) for a in out]
    )
    assert np.max(np.abs(across - across[0])) < 1e-3, across


def test_single_or_empty_is_noop():
    aff = _affines([10.0], [20.0])
    assert wave_correct_affines(aff) is aff
    assert wave_correct_affines([]) == []
