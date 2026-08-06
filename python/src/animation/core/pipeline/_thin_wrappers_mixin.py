"""Thin delegating wrapper methods on ``AnimeStitchPipeline``.

The original class exposed several stage methods (as bound or static). Kept
as thin wrappers so external callers (tests, helpers) still work.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from animation.alignment.canvas import (
    _compute_canvas,
    _crop_to_valid,
    _normalise_widths,
    _scan_stitch_fallback,
    find_optimal_sequence,
)
from animation.alignment.canvas import _load_frames as _load_frames_fn
from animation.alignment.ecc import _ecc_refine
from animation.alignment.matching import (
    _match_pair,
    _pairwise_match,
    _phase_correlate,
    _sample_bg_points,
    _template_match,
)
from animation.ingestion.masking import (
    _cleanup_sam2_state as _cleanup_sam2_state_fn,
)
from animation.ingestion.masking import (
    _compute_fg_masks,
    _compute_fg_masks_sam2_stateful,
)
from animation.rendering.compositing import _composite_foreground
from animation.rendering.photometric import (
    _apply_basic,
)
from animation.rendering.photometric import (
    _correct_vignetting as _correct_vignetting_fn,
)
from animation.rendering.rendering import (
    _cluster_animation_phases,
    _render,
    _render_first,
    _render_laplacian,
    _render_median,
)

from ._probes import _USE_SAM2, BaSiCWrapper, Image


class _ThinWrappersMixin:
    """Delegate methods preserved for external callers (tests, helpers)."""

    def _load_frames(self, paths: List[str]) -> List[np.ndarray]:
        return _load_frames_fn(paths)

    @staticmethod
    def _normalise_widths(frames: List[np.ndarray]) -> List[np.ndarray]:
        return _normalise_widths(frames)

    def _apply_basic(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        if self._basic is None:
            self._basic = BaSiCWrapper()
        corrected, baselines = _apply_basic(frames, self._basic)
        self._baselines = baselines
        return corrected

    def _correct_vignetting(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        return _correct_vignetting_fn(frames)

    def _compute_fg_masks(self, frames: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        if self.use_birefnet and self._birefnet is None:
            from backend.src.models.wrappers.birefnet_wrapper import (
                BiRefNetWrapper,
            )  # §3.14 lazy

            self._birefnet = BiRefNetWrapper()
        if _USE_SAM2:
            masks, pred, state, tmp, fh, fw = _compute_fg_masks_sam2_stateful(
                frames, self._birefnet, use_birefnet=self.use_birefnet
            )
            self._sam2_predictor = pred
            self._sam2_inference_state = state
            self._sam2_tmp_dir = tmp
            self._sam2_frame_h = fh
            self._sam2_frame_w = fw
            return masks
        return _compute_fg_masks(frames, self._birefnet, use_birefnet=self.use_birefnet)

    def _cleanup_sam2_state(self) -> None:
        """Free the live SAM-2 predictor state stored by _compute_fg_masks."""
        _cleanup_sam2_state_fn(
            self._sam2_predictor, self._sam2_inference_state, self._sam2_tmp_dir
        )
        self._sam2_predictor = None
        self._sam2_inference_state = None
        self._sam2_tmp_dir = None
        self._sam2_frame_h = 0
        self._sam2_frame_w = 0

    def _pairwise_match(
        self,
        frames: List[np.ndarray],
        bg_masks: List[Optional[np.ndarray]],
    ) -> List[Dict]:
        if self.use_loftr and self._loftr is None:
            from backend.src.models.wrappers.loftr_wrapper import LoFTRWrapper  # §3.14 lazy

            self._loftr = LoFTRWrapper()
        return _pairwise_match(
            frames,
            bg_masks,
            loftr_wrapper=self._loftr,
            use_loftr=self.use_loftr,
            motion_model=self.motion_model,
            aliked_wrapper=self._aliked if self.use_aliked else None,
        )

    def _match_pair(
        self,
        frames: List[np.ndarray],
        bg_masks: List[Optional[np.ndarray]],
        i: int,
        j: int,
        H: int,
        W: int,
    ) -> Optional[Dict]:
        return _match_pair(
            frames,
            bg_masks,
            i,
            j,
            H,
            W,
            loftr_wrapper=self._loftr,
            use_loftr=self.use_loftr,
            motion_model=self.motion_model,
            aliked_wrapper=self._aliked if self.use_aliked else None,
        )

    @staticmethod
    def _template_match(
        img_i: np.ndarray,
        img_j: np.ndarray,
        m_i: Optional[np.ndarray],
        m_j: Optional[np.ndarray],
        H: int,
        slice_h: int = 256,
        max_search_frac: float = 0.8,
        direction_sign: int = 0,
        max_dy_frac: float = 0.70,
    ) -> Tuple[Optional[np.ndarray], float]:
        return _template_match(
            img_i,
            img_j,
            m_i,
            m_j,
            H,
            slice_h=slice_h,
            max_search_frac=max_search_frac,
            direction_sign=direction_sign,
            max_dy_frac=max_dy_frac,
        )

    @staticmethod
    def _phase_correlate(
        img_i: np.ndarray,
        img_j: np.ndarray,
        m_i: Optional[np.ndarray],
        m_j: Optional[np.ndarray],
        use_mask: bool = True,
    ) -> Tuple[Optional[np.ndarray], float]:
        return _phase_correlate(img_i, img_j, m_i, m_j, use_mask=use_mask)

    @staticmethod
    def _sample_bg_points(
        mask: Optional[np.ndarray], H: int, W: int, n: int = 200
    ) -> np.ndarray:
        return _sample_bg_points(mask, H, W, n=n)

    def _ecc_refine(
        self,
        frames: List[np.ndarray],
        affines: List[np.ndarray],
        bg_masks: List[Optional[np.ndarray]],
    ) -> List[np.ndarray]:
        return _ecc_refine(frames, affines, bg_masks)

    @staticmethod
    def _compute_canvas(
        frames: List[np.ndarray],
        affines: List[np.ndarray],
    ) -> Tuple[int, int, np.ndarray]:
        return _compute_canvas(frames, affines)

    def _render(
        self,
        frames: List[np.ndarray],
        affines: List[np.ndarray],
        bg_masks: List[Optional[np.ndarray]],
        canvas_h: int,
        canvas_w: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        return _render(
            frames,
            affines,
            bg_masks,
            canvas_h,
            canvas_w,
            renderer=self.renderer,
            baselines=self._baselines,
        )

    def _render_median(self, *args, **kwargs):
        return _render_median(*args, **kwargs)

    def _render_first(self, frames, affines, H, W):
        return _render_first(frames, affines, H, W)

    def _render_laplacian(self, *args, **kwargs):
        return _render_laplacian(*args, **kwargs)

    @staticmethod
    def _cluster_animation_phases(
        frames: List[np.ndarray],
        affines: List[np.ndarray],
        H: int,
        W: int,
        target_w: int = 320,
        ac_threshold: float = 0.25,
        min_anim_pixels: int = 500,
    ):
        return _cluster_animation_phases(
            frames,
            affines,
            H,
            W,
            target_w=target_w,
            ac_threshold=ac_threshold,
            min_anim_pixels=min_anim_pixels,
        )

    def _composite_foreground(
        self,
        warped_corr: List[np.ndarray],
        warped_fgs: List[np.ndarray],
        canvas: np.ndarray,
        H: int,
        W: int,
        frames: List[np.ndarray],
        affines: List[np.ndarray],
        bg_masks: List[Optional[np.ndarray]],
        frame_keys: Optional[Tuple[str, ...]] = None,
        seam_path_cache: Optional[Dict] = None,
        exclusion_masks: Optional[List[np.ndarray]] = None,
        preset_boundaries: Optional[np.ndarray] = None,
        paint_mask: Optional[np.ndarray] = None,
        seam_meta_out: Optional[dict] = None,
        seam_overrides: Optional[dict] = None,
    ) -> np.ndarray:
        return _composite_foreground(
            warped_corr,
            warped_fgs,
            canvas,
            H,
            W,
            frames,
            affines,
            bg_masks,
            frame_keys=frame_keys,
            seam_path_cache=seam_path_cache,
            exclusion_masks=exclusion_masks,
            preset_boundaries=preset_boundaries,
            paint_mask=paint_mask,
            seam_meta_out=seam_meta_out,
            seam_overrides=seam_overrides,
        )

    def _crop_to_valid(self, canvas: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        return _crop_to_valid(canvas, valid_mask)

    @staticmethod
    def _scan_stitch_fallback(
        frames: List[np.ndarray],
        output_path: str,
    ) -> "Image.Image":
        return _scan_stitch_fallback(frames, output_path)

    @staticmethod
    def find_optimal_sequence(
        ref_path: str,
        candidates: List[str],
        min_inliers: int = 30,
        max_overlap: float = 0.85,
    ) -> List[str]:
        return find_optimal_sequence(
            ref_path,
            candidates,
            min_inliers=min_inliers,
            max_overlap=max_overlap,
        )


__all__ = ["_ThinWrappersMixin"]
