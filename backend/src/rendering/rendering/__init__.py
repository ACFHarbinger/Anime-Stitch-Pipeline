"""Temporal rendering: median, first-frame, and Laplacian-blend renderers (§5.17 split).

All functions are standalone — they take ``frames``, ``affines`` and (where
applicable) ``bg_masks`` as explicit arguments.

Split by role:
  _native.py            -- shared base.canvas C++ extension probe + GPU flag
                            (module-qualified access -- see its docstring)
  _gpu_median.py         -- §3.11A GPU-accelerated nanmedian
  _gain_correction.py    -- §1.40/§1.41 sequential photometric gain/bias
  _phase_clustering.py   -- per-pixel FFT animation-phase detection
  _animation_repaint.py  -- animation-phase re-render (shared by median's
                            fast + chunked paths -- deduplicated, see its
                            docstring)
  _median_fastpath.py    -- _render_median's C++/GPU fast path (Phase 5e)
  median.py                -- _render_median: chunked Python path + orchestration
  first.py                  -- _render_first
  laplacian.py              -- _render_laplacian
  dispatch.py                -- _render, the renderer-mode dispatcher
"""

from . import _native
from ._gain_correction import _adaptive_render_gain_clamp, _check_gain_chain_drift
from ._phase_clustering import _cluster_animation_phases
from .dispatch import _render
from .first import _render_first
from .laplacian import _render_laplacian
from .median import _render_median

__all__ = [
    "_native",
    "_render",
    "_render_median",
    "_render_first",
    "_render_laplacian",
    "_cluster_animation_phases",
    "_adaptive_render_gain_clamp",
    "_check_gain_chain_drift",
]
