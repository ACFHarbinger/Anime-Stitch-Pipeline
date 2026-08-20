"""
backend/src/models/efficient_loftr_wrapper.py
=============================================
EfficientLoFTR feature matcher wrapper (P1.4).

Uses the HuggingFace transformers integration of EfficientLoFTR (CVPR 2024,
arXiv:2403.04765) — ``zju-community/efficientloftr``.  This exposes the same
interface as LoFTRWrapper so it can act as a drop-in replacement:

    pts1, pts2, conf = wrapper.match(img_i, img_j)
    M, mean_conf     = wrapper.get_affine_partial(img_i, img_j, mask1, mask2)

EfficientLoFTR replaces LoFTR's dense O(N²) coarse attention with a two-stage
adaptive-span correlation layer, giving 2.5× faster throughput at equal or
better AUC on standard benchmarks.

Falls back to kornia LoFTR automatically if the transformers package or the
model weights are unavailable.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
from backend.src.constants.models import WRAPPERS__HF_REPO, WRAPPERS__MIN_INLIERS
from backend.src.models.core.base import ModelWrapper, lazy_load

logger = logging.getLogger(__name__)

try:
    from PIL import Image as _PILImage
    from transformers import AutoImageProcessor, EfficientLoFTRForKeypointMatching
    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False


class EfficientLoFTRWrapper(ModelWrapper):
    """
    Wraps HuggingFace EfficientLoFTR for dense feature matching.

    Exposes the same interface as LoFTRWrapper so it can be swapped in
    without any changes to the calling code.
    """

    def __init__(self, device: str | None = None):
        if not _TRANSFORMERS_OK:
            raise ImportError(
                "transformers >= 4.52 is required for EfficientLoFTRWrapper. "
                "Install with: pip install transformers"
            )
        super().__init__(device)
        self._processor = None
        self._model = None

    @classmethod
    def is_available(cls) -> bool:
        return _TRANSFORMERS_OK

    # ------------------------------------------------------------------ lifecycle

    def unload(self) -> None:
        """Delete model and processor from VRAM/RAM, then flush CUDA cache."""
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        super().unload()

    def offload(self) -> None:
        """Move the model to CPU without synchronizing the CUDA allocator."""
        if self._model is not None:
            self._model.cpu()

    def load(self) -> None:
        """Load EfficientLoFTR weights and processor from HuggingFace."""
        if self._model is None:
            logger.debug("[ELoFTR] Loading EfficientLoFTR from HuggingFace …")
            self._processor = AutoImageProcessor.from_pretrained(
                WRAPPERS__HF_REPO, use_fast=True
            )
            self._model = (
                EfficientLoFTRForKeypointMatching.from_pretrained(WRAPPERS__HF_REPO)
                .eval()
                .to(self.device)
            )
        else:
            self._model.to(self.device).eval()

    # backward-compat alias
    load_model = load

    # ------------------------------------------------------------------ inference

    def _run(
        self,
        img_i: np.ndarray,
        img_j: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run EfficientLoFTR on a BGR uint8 image pair.

        Returns
        -------
        pts1, pts2 : (K, 2) float32 matched keypoints in pixel coords of the
                     *original* input images (before any model resizing).
        conf       : (K,) float32 match confidence.
        """
        self.load()
        # self.load() above guarantees self._processor/self._model are set;
        # mypy can't see that invariant across the method boundary.
        assert self._processor is not None
        assert self._model is not None

        h0, w0 = img_i.shape[:2]
        h1, w1 = img_j.shape[:2]

        # BGR → RGB PIL for the processor
        pil0 = _PILImage.fromarray(cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB))
        pil1 = _PILImage.fromarray(cv2.cvtColor(img_j, cv2.COLOR_BGR2RGB))

        inputs = self._processor(
            images=[[pil0, pil1]], return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Model resizes internally; track the model-input size to scale back.
        inputs["pixel_values"].shape[-2]
        inputs["pixel_values"].shape[-1]

        with torch.no_grad():
            outputs = self._model(**inputs)

        # outputs.keypoints : (1, 2, N, 2)  normalized coords in [0, 1]
        # outputs.matches   : (1, 2, N)     index of match in other image (-1 = unmatched)
        # outputs.matching_scores : (1, 2, N) confidence
        kps = outputs.keypoints[0].cpu().numpy()      # (2, N, 2)
        matches = outputs.matches[0].cpu().numpy()    # (2, N)
        scores = outputs.matching_scores[0].cpu().numpy()  # (2, N)

        valid = matches[0] >= 0
        if not valid.any():
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty, np.empty(0, dtype=np.float32)

        kp0_norm = kps[0, valid]                         # (K, 2) — [0,1] relative
        match_idx = matches[0, valid].astype(int)
        kp1_norm = kps[1, match_idx]                     # (K, 2)
        conf = scores[0, valid].astype(np.float32)

        # Scale normalised coords back to original image pixel coordinates.
        # The processor resizes images to (H_model, W_model); normalised coords
        # are relative to that size, so: px = norm * [W_orig, H_orig] directly
        # because norm = px_model / [W_model, H_model] and
        # px_orig = px_model * [W_orig/W_model, H_orig/H_model].
        pts1 = kp0_norm * np.array([w0, h0], dtype=np.float32)
        pts2 = kp1_norm * np.array([w1, h1], dtype=np.float32)

        return pts1.astype(np.float32), pts2.astype(np.float32), conf

    # ------------------------------------------------------------------ public API (LoFTRWrapper-compatible)

    @lazy_load
    def match(
        self,
        img_i: np.ndarray,
        img_j: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Basic EfficientLoFTR match without mask filtering.
        Returns: (pts1, pts2, confidence).
        """
        return self._run(img_i, img_j)

    def match_masked(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: np.ndarray | None = None,
        mask2: np.ndarray | None = None,
        conf_thresh: float = 0.4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match img1 ↔ img2, filter by confidence and optional background masks.
        """
        pts1, pts2, conf = self._run(img1, img2)
        if len(pts1) == 0:
            return pts1, pts2, conf

        keep = conf > conf_thresh

        if mask1 is not None and len(pts1) > 0:
            h1, w1 = mask1.shape[:2]
            ix = np.clip(pts1[:, 0].astype(int), 0, w1 - 1)
            iy = np.clip(pts1[:, 1].astype(int), 0, h1 - 1)
            keep &= mask1[iy, ix] > 0

        if mask2 is not None and len(pts2) > 0:
            h2, w2 = mask2.shape[:2]
            jx = np.clip(pts2[:, 0].astype(int), 0, w2 - 1)
            jy = np.clip(pts2[:, 1].astype(int), 0, h2 - 1)
            keep &= mask2[jy, jx] > 0

        return pts1[keep], pts2[keep], conf[keep]

    def get_affine_partial(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: np.ndarray | None = None,
        mask2: np.ndarray | None = None,
        min_inliers: int = WRAPPERS__MIN_INLIERS,
    ) -> tuple[np.ndarray | None, float]:
        """
        Estimate a (2, 3) translation-only affine from img1 to img2.
        Returns (M, mean_conf) — same interface as LoFTRWrapper.
        """
        pts1, pts2, conf = self.match_masked(img1, img2, mask1, mask2)
        if len(pts1) < min_inliers:
            return None, 0.0

        M_raw, inliers = cv2.estimateAffinePartial2D(
            pts1,
            pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
            confidence=0.999,
            maxIters=10_000,
        )
        if M_raw is None or inliers is None:
            return None, 0.0

        inl_mask = inliers.ravel().astype(bool)
        if inl_mask.sum() < min_inliers:
            return None, 0.0

        return M_raw.astype(np.float32), float(conf[inl_mask].mean())

    def get_transform(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> np.ndarray | None:
        """Legacy API — returns (3, 3) homography or None."""
        pts1, pts2, conf = self.match(img1, img2)
        if len(pts1) < 4:
            return None
        keep = conf > 0.5
        if keep.sum() < 4:
            return None
        H_mat, status = cv2.findHomography(pts1[keep], pts2[keep], cv2.RANSAC, 5.0)
        return H_mat if (status is not None and status.sum() > 4) else None
