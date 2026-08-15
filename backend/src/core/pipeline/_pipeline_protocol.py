"""Type-checking-only ``Protocol`` describing the ``AnimeStitchPipeline`` host.

``AnimeStitchPipeline`` (see ``manager.py``) is composed out of several
mixins (``_FilterEdgesMixin``, ``_MatcherSelectionMixin``, ``_RunStageMixin``,
``_ThinWrappersMixin``). Each mixin references attributes that are actually
set by ``AnimeStitchPipeline.__init__`` or provided by a *different* mixin
in the composing class -- invisible to mypy when a mixin file is type
-checked in isolation, since ``self`` is only known to be that mixin's own
class there.

This module exists purely to give mypy a complete picture of ``self`` for
those mixins. It has zero runtime effect: the Protocol is only imported
under ``TYPE_CHECKING``, and mixins only inherit from it in the same guard,
so at runtime the MRO/`__mro__` and attribute lookup behave exactly as
before.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from asp_backend.models.wrappers.aliked_lg_wrapper import ALIKEDLightGlueWrapper
    from asp_backend.models.wrappers.efficient_loftr_wrapper import EfficientLoFTRWrapper
    from asp_backend.models.wrappers.roma_wrapper import RoMaWrapper
    from backend.src.models.wrappers.basic_wrapper import BaSiCWrapper
    from backend.src.models.wrappers.birefnet_wrapper import BiRefNetWrapper
    from backend.src.models.wrappers.loftr_wrapper import LoFTRWrapper


class _PipelineHost(Protocol):
    """Attributes/methods a mixin expects to find on ``self``.

    Mirrors exactly what ``AnimeStitchPipeline.__init__`` sets (manager.py)
    plus the cross-mixin methods (``_select_matcher``, ``_filter_edges``)
    each mixin borrows from a sibling mixin in the composed class.
    """

    # ── flags set in __init__ ────────────────────────────────────────────
    kwargs: dict
    last_session: object | None
    pause_hook: object | None
    use_basic: bool
    use_birefnet: bool
    use_loftr: bool
    use_efficient_loftr: bool
    use_aliked: bool
    use_roma: bool
    use_sea_raft: bool
    use_jamma: bool
    use_ecc: bool
    renderer: str
    composite_fg: bool
    bands: int
    edge_crop: int
    motion_model: str

    # ── seam / exclusion state (§1.5D, Issue 10A3) ───────────────────────
    _seam_path_cache: dict
    exclusion_masks: list[np.ndarray] | None

    # ── live SAM-2 predictor state (Issue 10A2 S83) ──────────────────────
    _sam2_predictor: object | None
    _sam2_inference_state: object | None
    _sam2_tmp_dir: str | None
    _sam2_frame_h: int
    _sam2_frame_w: int

    # ── lazy-loaded model instances ──────────────────────────────────────
    _basic: BaSiCWrapper | None
    _baselines: list[float] | None
    _birefnet: BiRefNetWrapper | None
    _loftr: LoFTRWrapper | None
    _eloftr: EfficientLoFTRWrapper | None
    _aliked: ALIKEDLightGlueWrapper | None
    _roma: RoMaWrapper | None
    _sea_raft: object | None
    _stitch_net: object | None

    # ── methods provided by sibling mixins ───────────────────────────────
    def _select_matcher(self, H: int, W: int): ...

    def _filter_edges(
        self,
        edges: list[dict],
        image_paths: list[str],
        H: int,
        W: int,
        frames: list[np.ndarray],
        bg_masks: list[np.ndarray | None],
    ) -> list[dict]: ...


__all__ = ["_PipelineHost"]
