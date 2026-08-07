"""``AnimeStitchPipeline._filter_edges`` -- Stage 5-6 post-processing.

Applies geometric-consistency + direction-consensus filters to raw edges.
Separated from ``run()`` (in the pre-split file too) so the progress-aware
subclass can call it after its overridden ``_pairwise_match``.
"""

from __future__ import annotations

import logging
import os
import re

import numpy as np
from asp_backend.alignment.matching import _template_match
from backend.src.constants import MATCH_EDGE_CROP, MIN_EXPECTED_STEP

from ._edge_filters import _compute_adaptive_min_disp, _reject_static_edges
from ._probes import _HAS_BATCH, _batch

logger = logging.getLogger(__name__)


class _FilterEdgesMixin:
    """Provides ``_filter_edges`` for ``AnimeStitchPipeline``."""

    def _filter_edges(  # noqa: C901
        self,
        edges: list[dict],
        image_paths: list[str],
        H: int,
        W: int,
        frames: list[np.ndarray],
        bg_masks: list[np.ndarray | None],
    ) -> list[dict]:
        """
        Apply geometric-consistency + direction-consensus filters to raw edges.

        Separated from ``run()`` so the progress-aware subclass can call it
        after its overridden ``_pairwise_match``.
        """

        # ── §1.2A+C: Pre-filter static edges (adaptive threshold) ───────────
        # §1.2C: derive content-adaptive threshold before §1.2A rejection so
        # that high-resolution / fast-scroll sequences apply a proportionally
        # higher floor (10 % of median adjacent step, min STATIC_EDGE_MIN_DISP_PX).
        _min_disp = _compute_adaptive_min_disp(edges)
        edges = _reject_static_edges(edges, min_disp_px=_min_disp)

        # ── §2.14 + Geometric Consistency + Min-step (batch or Python) ──────
        # C++ batch.matching.filter_edge_graph covers all three classical gates
        # in a single pass; Python fallbacks run individually when batch is absent.
        _batch_filter_ok = False
        if _HAS_BATCH and hasattr(_batch, "matching"):
            try:
                edges = list(
                    _batch.matching.filter_edge_graph(
                        edges,
                        float(MIN_EXPECTED_STEP),
                        15.0,
                        0.0,
                    )
                )
                _batch_filter_ok = True
            except Exception:
                pass

        if not _batch_filter_ok:

            # ── Geometric Consistency Filter ──────────────────────────────────
            if len(edges) > 0:
                adj_map: dict[int, tuple[float, float]] = {}
                for e in edges:
                    if e["j"] == e["i"] + 1:
                        adj_map[e["i"]] = (e["M"][0, 2], e["M"][1, 2])

                filtered: list[dict] = []
                for e in edges:
                    i, j = e["i"], e["j"]
                    if j == i + 1:
                        filtered.append(e)
                        continue
                    can_verify = True
                    sum_dx, sum_dy = 0.0, 0.0
                    for k in range(i, j):
                        if k in adj_map:
                            sum_dx += adj_map[k][0]
                            sum_dy += adj_map[k][1]
                        else:
                            can_verify = False
                            break
                    if can_verify:
                        diff_x = abs(e["M"][0, 2] - sum_dx)
                        diff_y = abs(e["M"][1, 2] - sum_dy)
                        if diff_x < 15.0 and diff_y < 15.0:
                            filtered.append(e)
                        else:
                            logger.debug(
                                f"[Stitch]   Edge {i}→{j} rejected: inconsistency "
                                f"(dx={diff_x:.1f}, dy={diff_y:.1f})"
                            )
                    else:
                        filtered.append(e)
                edges = filtered

            # ── Min-step guard ─────────────────────────────────────────────────
            # Reject adjacent edges with near-zero displacement BEFORE the direction
            # consensus filter so the consensus median is not pulled toward zero.
            if len(edges) >= 3:
                adj_edges = [e for e in edges if e["j"] == e["i"] + 1]
                if len(adj_edges) > 0:
                    median_dx_abs = float(np.median([abs(e["M"][0, 2]) for e in adj_edges]))
                    median_dy_abs = float(np.median([abs(e["M"][1, 2]) for e in adj_edges]))
                    primary_axis = 0 if median_dx_abs > median_dy_abs else 1

                    adj_before = len(adj_edges)
                    edges = [
                        e
                        for e in edges
                        if e["j"] != e["i"] + 1
                        or abs(float(e["M"][primary_axis, 2])) >= MIN_EXPECTED_STEP
                    ]
                    adj_after = sum(1 for e in edges if e["j"] == e["i"] + 1)
                    n_rejected = adj_before - adj_after
                    if n_rejected > 0:
                        logger.debug(
                            f"[Stitch]   Min-step guard: rejected {n_rejected} near-zero "
                            f"edges (threshold={MIN_EXPECTED_STEP}px on axis {primary_axis})"
                        )

        # ── Direction Consensus Filter ────────────────────────────────────────
        if len(edges) >= 3:
            adj_edges = [e for e in edges if e["j"] == e["i"] + 1]
            if len(adj_edges) >= 3:
                median_dx_abs = float(np.median([abs(e["M"][0, 2]) for e in adj_edges]))
                median_dy_abs = float(np.median([abs(e["M"][1, 2]) for e in adj_edges]))
                primary_axis = 0 if median_dx_abs > median_dy_abs else 1

                adj_vals = [e["M"][primary_axis, 2] for e in adj_edges]
                median_val = float(np.median(adj_vals))
                consensus_sign = int(np.sign(median_val))

                # Drop skip edges (j > i+1) that scroll the wrong direction or are noise
                if consensus_sign != 0:
                    _pre_skip_n = len(edges)
                    edges = [
                        e
                        for e in edges
                        if e["j"] == e["i"] + 1
                        or abs(float(e["M"][primary_axis, 2])) < 20.0
                        or int(np.sign(float(e["M"][primary_axis, 2])))
                        == consensus_sign
                    ]
                    _n_skip_dropped = _pre_skip_n - len(edges)
                    if _n_skip_dropped:
                        logger.debug(
                            f"[Stitch]   Skip-edge sign filter: dropped "
                            f"{_n_skip_dropped} wrong-sign skip edges"
                        )

                _ts_pat = re.compile(r"_(\d+)ms", re.IGNORECASE)
                timestamps_ms: list[int | None] = []
                for p in image_paths:
                    m = _ts_pat.search(os.path.basename(p))
                    timestamps_ms.append(int(m.group(1)) if m else None)

                def _interval_ms(fi: int, fj: int) -> int | None:
                    t_i = timestamps_ms[fi] if fi < len(timestamps_ms) else None
                    t_j = timestamps_ms[fj] if fj < len(timestamps_ms) else None
                    if t_i is not None and t_j is not None and t_j != t_i:
                        return abs(t_j - t_i)
                    return None

                def _wrong_sign(val: float) -> bool:
                    return (
                        consensus_sign != 0
                        and np.sign(val) != 0
                        and int(np.sign(val)) != consensus_sign
                    )

                def _gross_outlier(val: float) -> bool:
                    return (
                        abs(val) > 2.0 * abs(median_val)
                        and abs(val - median_val) > 200.0
                    )

                vel_samples = []
                for e in edges:
                    if e["j"] != e["i"] + 1:
                        continue
                    v_e = float(e["M"][primary_axis, 2])
                    if _wrong_sign(v_e) or _gross_outlier(v_e):
                        continue
                    iv = _interval_ms(e["i"], e["j"])
                    if iv is not None:
                        vel_samples.append(v_e / iv)
                vel_px_per_ms: float | None = (
                    float(np.median(vel_samples)) if vel_samples else None
                )
                if vel_px_per_ms is not None:
                    logger.debug(
                        f"[Stitch]   Scroll velocity: {vel_px_per_ms:.4f} px/ms "
                        f"(from {len(vel_samples)} reliable edges)"
                    )

                def _is_outlier(val: float, fi: int, fj: int) -> tuple[bool, str]:
                    if _wrong_sign(val):
                        return True, "wrong sign"
                    if _gross_outlier(val):
                        return True, "gross outlier"
                    if vel_px_per_ms is not None:
                        iv = _interval_ms(fi, fj)
                        if iv is not None:
                            expected = abs(vel_px_per_ms) * iv
                            if abs(val - expected * consensus_sign) > max(
                                0.15 * expected, 15.0
                            ):
                                return (
                                    True,
                                    f"velocity outlier (expected {expected * consensus_sign:.1f})",
                                )
                    return False, ""

                def _apply_corrected_M(
                    edge: dict, new_M: np.ndarray, new_weight: float
                ) -> dict:
                    new_pts_j = edge["pts_i"] + new_M[:, 2].astype(np.float32)
                    return dict(edge, M=new_M, pts_j=new_pts_j, weight=new_weight)

                ec_h = int(H * MATCH_EDGE_CROP)
                ec_w = int(W * MATCH_EDGE_CROP)
                corrected: list[dict] = []
                for e in edges:
                    if e["j"] == e["i"] + 1:
                        fi, fj = e["i"], e["j"]
                        val = float(e["M"][primary_axis, 2])
                        outlier, reason = _is_outlier(val, fi, fj)
                        if outlier:
                            iv = _interval_ms(fi, fj)
                            replaced = False
                            if vel_px_per_ms is not None and iv is not None:
                                est_val = vel_px_per_ms * iv
                                logger.debug(
                                    f"[Stitch]   Edge {fi}→{fj}: val={val:.1f} ({reason}); "
                                    f"velocity → val={est_val:.1f}"
                                )
                                M_fix = np.eye(2, 3, dtype=np.float32)
                                M_fix[1 - primary_axis, 2] = e["M"][1 - primary_axis, 2]
                                M_fix[primary_axis, 2] = est_val
                                e = _apply_corrected_M(e, M_fix, 0.55)
                                replaced = True
                            if not replaced and primary_axis == 1:
                                img_i_c = frames[fi][ec_h:-ec_h, ec_w:-ec_w]
                                img_j_c = frames[fj][ec_h:-ec_h, ec_w:-ec_w]
                                m_i_c = (
                                    bg_masks[fi][ec_h:-ec_h, ec_w:-ec_w]
                                    if bg_masks[fi] is not None
                                    else None
                                )
                                M_dir, c_dir = _template_match(
                                    img_i_c,
                                    img_j_c,
                                    m_i_c,
                                    None,
                                    img_i_c.shape[0],
                                    direction_sign=consensus_sign,
                                )
                                if (
                                    M_dir is not None
                                    and int(np.sign(M_dir[1, 2])) == consensus_sign
                                ):
                                    new_val = float(M_dir[1, 2])
                                    logger.debug(
                                        f"[Stitch]   Edge {fi}→{fj}: directed TM → "
                                        f"val={new_val:.1f} conf={c_dir:.3f}"
                                    )
                                    M_new = np.array(
                                        [[1, 0, e["M"][0, 2]], [0, 1, new_val]],
                                        dtype=np.float32,
                                    )
                                    e = _apply_corrected_M(e, M_new, c_dir * 0.7)
                                    replaced = True
                            if not replaced:
                                logger.debug(
                                    f"[Stitch]   Edge {fi}→{fj}: val={val:.1f} ({reason}); "
                                    f"using median {median_val:.1f}"
                                )
                                M_fix = np.eye(2, 3, dtype=np.float32)
                                M_fix[1 - primary_axis, 2] = e["M"][1 - primary_axis, 2]
                                M_fix[primary_axis, 2] = median_val
                                e = _apply_corrected_M(
                                    e, M_fix, e.get("weight", 1.0) * 0.3
                                )
                        else:
                            logger.debug(
                                f"[Stitch]   Edge {fi}→{fj}: val={val:.1f} kept "
                                f"(consensus {median_val:.1f})"
                            )
                    corrected.append(e)
                edges = corrected

        return edges


__all__ = ["_FilterEdgesMixin"]
