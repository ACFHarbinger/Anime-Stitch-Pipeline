"""Stage 7b affine-validation recovery: Retry 0-3.

Extracted from ``AnimeStitchPipeline.run()`` as a standalone function -- pure
code motion, no logic change (see _photometric_stage.py's docstring). Unlike
most of run(), this block contains no early `return` -- every retry either
updates (affines, health) or leaves them unchanged, and the actual PANORAMA/
SCANS fallback bailout (which DOES return early) happens only after this
whole retry chain finishes, so it stays in run() itself.
"""

from __future__ import annotations

import numpy as np
from asp_backend.alignment.bundle_adjust import _bundle_adjust_affine
from asp_backend.core.validation import AffineHealth, _validate_affines

from ._edge_filters import _filter_high_conf_edges


def _recover_affine_health(  # noqa: C901
    edges: list[dict],
    N: int,
    affines: list[np.ndarray],
    health: AffineHealth,
    use_affine_ba: bool,
    adaptive_min_gap: float,
    adaptive_rot: float,
    adaptive_sc: float,
    logger,
) -> tuple[list[np.ndarray], AffineHealth]:
    """Attempt Retries 0-3 to recover a failed affine-validation ``health``.

    Returns ``(affines, health)`` -- either the original inputs unchanged
    (no retry improved on them) or the best-recovered affines/health found.
    Retry 0 (high-confidence-only re-solve) only runs for ratio failures.
    Retry 1 (adjacent-only bundle), Retry 2 (sequential+fill), and Retry 3
    (relaxed min_gap) always run in sequence until one succeeds.
    """
    logger.debug(
        f"[Stitch]   Affine health FAILED ({health.reason}); attempting recovery..."
    )
    # Retry 0: §2.9C — high-confidence-only re-solve (ratio failures only).
    # Low-confidence TM/PC fallback edges (weight 0.15–0.55) can corrupt BA
    # when a single bad edge pulls two frames together → inflated ratio.
    # Filter to LoFTR-quality edges (weight ≥ HIGH_CONF_EDGE_THRESH) and
    # re-solve if enough survive.  Falls through to Retry 1 if not.
    if health.reason.startswith("ratio="):
        _hc_edges = _filter_high_conf_edges(edges)
        if len(_hc_edges) >= N - 1:
            _affines_r0 = _bundle_adjust_affine(_hc_edges, N, use_affine=use_affine_ba)
            _health_r0 = _validate_affines(
                _affines_r0,
                min_step=adaptive_min_gap,
                max_rotation=adaptive_rot,
                max_scale_dev=adaptive_sc,
            )
            logger.debug(
                f"[Stitch]   Retry 0 (high-conf edges, {len(_hc_edges)} edges): "
                f"valid={_health_r0.valid}, {_health_r0.reason}"
            )
            if _health_r0.valid:
                affines, health = _affines_r0, _health_r0

    # Retry 1: consecutive-only bundle — skip edges sometimes corrupt the solution
    _adj_only = [e for e in edges if e["j"] == e["i"] + 1]
    if len(_adj_only) >= N - 1:
        affines_r1 = _bundle_adjust_affine(_adj_only, N, use_affine=use_affine_ba)
        health_r1 = _validate_affines(affines_r1)
        logger.debug(
            f"[Stitch]   Retry 1 (adj-only bundle): "
            f"valid={health_r1.valid}, {health_r1.reason}"
        )
        if health_r1.valid:
            affines, health = affines_r1, health_r1
    # Retry 2: smart sequential integration with gap-filling
    if not health.valid:
        _adj_only_r2 = [e for e in edges if e["j"] == e["i"] + 1]
        # Consensus step for interpolation/extrapolation of isolated frames
        _step_dx = (
            float(np.median([float(e["M"][0, 2]) for e in _adj_only_r2]))
            if _adj_only_r2
            else 0.0
        )
        _step_dy = (
            float(np.median([float(e["M"][1, 2]) for e in _adj_only_r2]))
            if _adj_only_r2
            else 0.0
        )
        # Frames that have an adj edge pointing to them
        _has_adj_src = {e["j"] for e in _adj_only_r2}

        _seq = [np.eye(2, 3, dtype=np.float32) for _ in range(N)]
        _anchored: set = {0}

        # Pass 1: greedy — for each frame use the shortest-span edge from an anchored frame
        for _f in range(1, N):
            _best_e, _best_span = None, float("inf")
            for _e in edges:
                if _e["j"] == _f and _e["i"] in _anchored and _f - _e["i"] < _best_span:
                    _best_span = _f - _e["i"]
                    _best_e = _e
            if _best_e is not None:
                _seq[_f][0, 2] = _seq[_best_e["i"]][0, 2] - float(_best_e["M"][0, 2])
                _seq[_f][1, 2] = _seq[_best_e["i"]][1, 2] - float(_best_e["M"][1, 2])
                _anchored.add(_f)

        # Pass 2: fill frames with no adj edge via interpolation or velocity extrapolation
        for _uf in sorted(i for i in range(N) if i not in _anchored):
            if _uf in _has_adj_src:
                continue  # will be chained in Pass 3
            _lft = max((a for a in _anchored if a < _uf), default=None)
            _rgt = min((a for a in _anchored if a > _uf), default=None)
            if _lft is not None and _rgt is not None:
                _t = (_uf - _lft) / (_rgt - _lft)
                _seq[_uf][0, 2] = _seq[_lft][0, 2] * (1 - _t) + _seq[_rgt][0, 2] * _t
                _seq[_uf][1, 2] = _seq[_lft][1, 2] * (1 - _t) + _seq[_rgt][1, 2] * _t
            elif _lft is not None:
                _n = _uf - _lft
                _seq[_uf][0, 2] = _seq[_lft][0, 2] - _n * _step_dx
                _seq[_uf][1, 2] = _seq[_lft][1, 2] - _n * _step_dy
            _anchored.add(_uf)

        # Pass 3: propagate through adj/skip edges from newly-anchored gap frames
        _chg = True
        while _chg:
            _chg = False
            for _f in range(1, N):
                if _f in _anchored:
                    continue
                _best_e, _best_span = None, float("inf")
                for _e in edges:
                    if _e["j"] == _f and _e["i"] in _anchored and _f - _e["i"] < _best_span:
                        _best_span = _f - _e["i"]
                        _best_e = _e
                if _best_e is not None:
                    _seq[_f][0, 2] = _seq[_best_e["i"]][0, 2] - float(_best_e["M"][0, 2])
                    _seq[_f][1, 2] = _seq[_best_e["i"]][1, 2] - float(_best_e["M"][1, 2])
                    _anchored.add(_f)
                    _chg = True

        health_r2 = _validate_affines(_seq)
        logger.debug(
            f"[Stitch]   Retry 2 (sequential+fill): "
            f"valid={health_r2.valid}, {health_r2.reason}"
        )
        if health_r2.valid:
            affines, health = _seq, health_r2
        else:
            # Retry 3: accept with relaxed min_gap when ratio is still healthy
            health_r3 = _validate_affines(_seq, min_step=20.0)
            if health_r3.valid:
                logger.debug(
                    f"[Stitch]   Retry 3 (relaxed min_gap=20px): "
                    f"valid={health_r3.valid}, {health_r3.reason}"
                )
                affines, health = _seq, health_r3

    return affines, health


__all__ = ["_recover_affine_health"]
