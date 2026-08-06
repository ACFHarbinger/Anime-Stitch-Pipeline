"""Stage 5-6 matcher selection: JamMa (4K) -> EfficientLoFTR -> kornia LoFTR,
plus lazy ALIKED+LightGlue / RoMa wrapper construction.

Extracted from ``AnimeStitchPipeline.run()`` as its own mixin method -- pure
code motion, no logic change (see _photometric_stage.py's docstring). Kept
as an instance method (not a free function) since it mutates
self._eloftr/_loftr/_aliked/_roma and the use_* flags.
"""

from __future__ import annotations

import logging

from ._probes import RoMaWrapper

logger = logging.getLogger(__name__)


class _MatcherSelectionMixin:
    """Provides ``_select_matcher`` for ``AnimeStitchPipeline.run()``."""

    def _select_matcher(self, H: int, W: int):
        """Select and lazily construct the dense-matching backend, priority:
        JamMa (4K only) -> EfficientLoFTR -> kornia LoFTR -> None. Also lazily
        constructs the ALIKED+LightGlue and RoMa fallback wrappers when
        enabled. Returns the active LoFTR-family matcher (or None).
        """
        _is_4k = H * W > 3000 * 2000
        _active_loftr = None

        if self.use_jamma and _is_4k:
            try:
                from animation.models.wrappers.jamma_wrapper import JamMaWrapper  # §3.14 lazy

                _jamma_inst = JamMaWrapper()
                _jamma_inst.load_model()
                _active_loftr = _jamma_inst
                logger.info(f"[Stitch]   4K frame ({W}×{H}): using JamMa (O(N) Mamba).")
            except Exception as _jm_e:
                logger.info(
                    f"[Stitch]   JamMa unavailable ({_jm_e}); using EfficientLoFTR."
                )

        # P1.4 — Use EfficientLoFTR (2.5× faster) when available; fall back to
        # kornia LoFTR.  Both expose the same .match() interface.
        if _active_loftr is None and self.use_efficient_loftr:
            if self._eloftr is None:
                try:
                    from animation.models.wrappers.efficient_loftr_wrapper import (
                        EfficientLoFTRWrapper,
                    )  # §3.14 lazy

                    self._eloftr = EfficientLoFTRWrapper()
                    self._eloftr.load_model()
                    _active_loftr = self._eloftr
                    logger.info(
                        "[Stitch]   Using EfficientLoFTR (2.5× faster than LoFTR)."
                    )
                except Exception as _e:
                    logger.debug(
                        f"[Stitch]   EfficientLoFTR init failed ({_e}); falling back to LoFTR."
                    )
                    self.use_efficient_loftr = False
                    self._eloftr = None
            else:
                self._eloftr.load_model()
                _active_loftr = self._eloftr
        if _active_loftr is None and self.use_loftr:
            if self._loftr is None:
                from backend.src.models.wrappers.loftr_wrapper import LoFTRWrapper  # §3.14 lazy

                self._loftr = LoFTRWrapper()
            _active_loftr = self._loftr

        if self.use_aliked and self._aliked is None:
            try:
                from animation.models.wrappers.aliked_lg_wrapper import (
                    ALIKEDLightGlueWrapper,
                )  # §3.14 lazy

                self._aliked = ALIKEDLightGlueWrapper()
            except Exception as _e:
                logger.info(
                    f"[Stitch]   ALIKED+LightGlue init failed ({_e}); disabling."
                )
                self.use_aliked = False
                self._aliked = None
        if self.use_roma and self._roma is None:
            try:
                self._roma = RoMaWrapper()
            except Exception as _e:
                logger.info(f"[Stitch]   RoMa init failed ({_e}); disabling.")
                self.use_roma = False
                self._roma = None

        return _active_loftr


__all__ = ["_MatcherSelectionMixin"]
