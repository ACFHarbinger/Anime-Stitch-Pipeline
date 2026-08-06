"""
AnimeStitchPipeline — top-level orchestrator (§5.17 split).

Delegates each pipeline stage to its sibling module (matching, photometric,
masking, ECC, rendering, compositing, canvas, bundle adjustment).

Split by role:
  _probes.py             -- shared optional-dependency probes (torch, PIL,
                            base extension, model-availability *_OK flags)
  _edge_filters.py        -- edge-graph filtering (static/high-conf/connectivity)
  _frame_utils.py          -- frame ordering, dy_cv gate, spatial dedup, hires
                            keyframe swap-in, canvas row coverage
  _manual_edges.py        -- HITL manual-displacement/landmark edge builders
  _photometric_stage.py   -- run() Stage 4.5/4.5b, extracted (see its docstring)
  _content_trim.py         -- run() Stage 12.5, extracted
  _dedup_stage.py           -- run()'s pre-Stage-5 near-static frame dedup, extracted
  _affine_recovery.py      -- run()'s Stage 7b Retry 0-3 chain, extracted
  _matcher_selection.py    -- run()'s Stage 5-6 matcher-selection mixin method
  _filter_edges_mixin.py   -- AnimeStitchPipeline._filter_edges
  _thin_wrappers_mixin.py  -- delegate methods kept for external callers
  run_stage.py               -- AnimeStitchPipeline.run(), the 951-line entry
                              point (see its own module docstring for why
                              this is the one file left over 500 lines)
  manager.py                 -- AnimeStitchPipeline, composed from all mixins
"""

import logging

from ._edge_filters import (
    _check_edge_graph_connectivity,
    _compute_adaptive_min_disp,
    _filter_high_conf_edges,
    _reject_static_edges,
)
from ._frame_utils import (
    _apply_hires_keyframes,
    _compute_adaptive_dy_cv_max,
    _compute_dy_cv,
    _compute_row_coverage,
    _reload_scans_frames,
    _sort_frames_by_index,
    _spatial_dedup_frames,
)
from ._manual_edges import _build_landmark_affine, _build_manual_edge
from ._probes import _ALIKED_OK, _BIREFNET_OK, _DY_CV_MAX, _ELOFTR_OK, _LOFTR_OK, _USE_SAM2
from .manager import AnimeStitchPipeline

logger = logging.getLogger(__name__)

__all__ = [
    "AnimeStitchPipeline",
    "_ALIKED_OK",
    "_BIREFNET_OK",
    "_DY_CV_MAX",
    "_ELOFTR_OK",
    "_LOFTR_OK",
    "_USE_SAM2",
    "_apply_hires_keyframes",
    "_build_landmark_affine",
    "_build_manual_edge",
    "_check_edge_graph_connectivity",
    "_compute_adaptive_dy_cv_max",
    "_compute_adaptive_min_disp",
    "_compute_dy_cv",
    "_compute_row_coverage",
    "_filter_high_conf_edges",
    "_reject_static_edges",
    "_reload_scans_frames",
    "_sort_frames_by_index",
    "_spatial_dedup_frames",
    "logger",
]
