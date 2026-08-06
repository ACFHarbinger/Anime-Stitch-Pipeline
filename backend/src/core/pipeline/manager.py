"""``AnimeStitchPipeline`` -- composed from per-concern mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from backend.src.constants import LAPLACIAN_BANDS

from ._filter_edges_mixin import _FilterEdgesMixin
from ._matcher_selection import _MatcherSelectionMixin
from ._probes import (
    _ALIKED_OK,
    _BASIC_OK,
    _BIREFNET_OK,
    _ELOFTR_OK,
    _LOFTR_OK,
    _ROMA_OK,
    _SEA_RAFT_OK,
    BaSiCWrapper,
)
from ._thin_wrappers_mixin import _ThinWrappersMixin
from .run_stage import _RunStageMixin

if TYPE_CHECKING:
    from animation.models.stitch_net import AnimeStitchNet
    from animation.models.wrappers.aliked_lg_wrapper import ALIKEDLightGlueWrapper
    from backend.src.models.wrappers.birefnet_wrapper import BiRefNetWrapper
    from animation.models.wrappers.efficient_loftr_wrapper import EfficientLoFTRWrapper
    from backend.src.models.wrappers.loftr_wrapper import LoFTRWrapper


class AnimeStitchPipeline(
    _FilterEdgesMixin, _MatcherSelectionMixin, _RunStageMixin, _ThinWrappersMixin
):
    """
    Multi-stage anime frame stitching pipeline.

    Parameters
    ----------
    use_basic    : enable BaSiC photometric correction (broadcast dimming removal).
    use_birefnet : enable BiRefNet foreground masking (character exclusion).
    use_loftr    : enable LoFTR dense matching (falls back to template match if False).
    use_ecc      : enable ECC sub-pixel refinement after bundle adjustment.
    renderer     : 'median' — temporal Overmix-style median (suppresses noise);
                   'first'  — always use the first valid frame per canvas pixel;
                   'blend'  — sequential Laplacian blend (nearest to SCANS mode).
    composite_fg : paste the foreground character from the best single frame back
                   onto the median background.
    laplacian_bands : pyramid depth for multi-band blending.
    """

    def __init__(
        self,
        use_basic: bool = True,
        use_birefnet: bool = True,
        use_loftr: bool = True,
        use_efficient_loftr: bool = True,
        use_aliked: bool = True,
        use_roma: bool = True,
        use_sea_raft: bool = True,
        use_ecc: bool = True,
        renderer: str = "median",  # 'median' | 'first' | 'blend'
        composite_fg: bool = True,
        laplacian_bands: int = LAPLACIAN_BANDS,
        edge_crop: int = 30,
        motion_model: str = "translation",  # 'translation' or 'affine' (4-DOF)
        **kwargs,
    ):
        self.kwargs = kwargs
        self.use_basic = use_basic and _BASIC_OK
        self.use_birefnet = use_birefnet and _BIREFNET_OK
        self.use_loftr = use_loftr and _LOFTR_OK
        self.use_efficient_loftr = use_efficient_loftr and _ELOFTR_OK
        self.use_aliked = use_aliked and _ALIKED_OK
        self.use_roma = use_roma and _ROMA_OK
        self.use_sea_raft = use_sea_raft and _SEA_RAFT_OK
        self.use_jamma = kwargs.get("use_jamma", False)
        self.use_ecc = use_ecc
        self.renderer = renderer
        self.composite_fg = composite_fg
        self.bands = laplacian_bands
        self.edge_crop = edge_crop
        self.motion_model = motion_model

        # §1.5D: seam path cache shared across run() invocations on the same frame set
        self._seam_path_cache: Dict = {}

        # Issue 10A3: NL seam routing exclusion masks — set externally before run()
        # List of per-frame uint8 (H,W) masks where >127 forces seam cost=1e6.
        self.exclusion_masks: Optional[List[np.ndarray]] = None

        # Issue 10A2 S83: live SAM-2 predictor state preserved across HITL boundary.
        # Populated by _compute_fg_masks() when _USE_SAM2 is True; freed by
        # _cleanup_sam2_state() after checkpoint 1.5 mask review completes.
        self._sam2_predictor = None
        self._sam2_inference_state = None
        self._sam2_tmp_dir: Optional[str] = None
        self._sam2_frame_h: int = 0
        self._sam2_frame_w: int = 0

        # Lazy-loaded model instances (only allocated if the flag is True)
        self._basic: Optional["BaSiCWrapper"] = None
        self._baselines: Optional[List[float]] = None
        self._birefnet: Optional["BiRefNetWrapper"] = None
        self._loftr: Optional["LoFTRWrapper"] = None
        self._eloftr: Optional["EfficientLoFTRWrapper"] = None
        self._aliked: Optional["ALIKEDLightGlueWrapper"] = None
        self._roma = None
        self._sea_raft = None
        self._stitch_net: Optional["AnimeStitchNet"] = None


__all__ = ["AnimeStitchPipeline"]
