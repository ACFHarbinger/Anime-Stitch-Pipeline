"""
backend/src/models/jamma_wrapper.py
=====================================
JamMa feature matcher wrapper (P3.2).

JamMa (CVPR 2025, arXiv:2503.03437) replaces LoFTR's O(N²) transformer
attention with O(N) Mamba state-space scans, enabling 4K frame matching at
< 2 s/pair vs ~ 5 s for EfficientLoFTR.

Requires:
  pip install mamba-ssm causal-conv1d
  (must be built against your installed PyTorch CUDA version)

The vendor checkout is at vendor/JamMa (cloned from leoluxxx/JamMa).
Pretrained outdoor weights: https://huggingface.co/leoluxxx/JamMa

Usage in pipeline
-----------------
JamMa activates automatically when the source frame resolution exceeds
3000 × 2000 and ``use_jamma=True`` (default).  For smaller frames
EfficientLoFTR is used instead.

Interface: identical to LoFTRWrapper and EfficientLoFTRWrapper:
    pts1, pts2, conf = wrapper.match(img_i, img_j)
    M, conf         = wrapper.get_affine_partial(img_i, img_j, m_i, m_j)
"""

from __future__ import annotations

import logging
import os
import sys
from copy import deepcopy

import cv2
import numpy as np
import torch
from backend.src.constants.models import _CKPT_FILE, _HF_REPO, _JAMMA_ERR, _JAMMA_OK, _MIN_INLIERS
from backend.src.models.core.base import ModelWrapper, lazy_load
from huggingface_hub import hf_hub_download

_VENDOR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "vendor", "JamMa"
)

try:
    if os.path.isdir(_VENDOR_PATH) and _VENDOR_PATH not in sys.path:
        sys.path.insert(0, _VENDOR_PATH)
    import mamba_ssm  # noqa: F401 — confirm CUDA extension is loadable

    from src.config.default import get_cfg_defaults  # type: ignore
    from src.jamma.jamma import JamMa  # type: ignore
    _JAMMA_OK = True
except Exception as _e:
    _JAMMA_ERR = str(_e)

logger = logging.getLogger(__name__)


class JamMaWrapper(ModelWrapper):
    """
    O(N) Mamba-based feature matcher — drop-in replacement for LoFTRWrapper.

    Activates automatically for frames > 3000 × 2000 px (4K tier).
    Falls back to EfficientLoFTR / LoFTR for smaller images.

    Raises ImportError at construction if mamba_ssm CUDA extensions are
    not available.  The pipeline catches this and falls back gracefully.
    """

    def __init__(self, device: str | None = None):
        if not _JAMMA_OK:
            raise ImportError(
                f"JamMa requires mamba_ssm with CUDA extensions. "
                f"Build error: {_JAMMA_ERR}\n"
                "Fix: pip install mamba-ssm causal-conv1d --no-build-isolation"
            )
        super().__init__(device)
        self._model: JamMa | None = None

    @classmethod
    def is_available(cls) -> bool:
        return _JAMMA_OK

    # ---------------------------------------------------------------- lifecycle

    def unload(self) -> None:
        """Delete model from VRAM/RAM, then flush CUDA cache."""
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
        super().unload()

    def offload(self) -> None:
        if self._model is not None:
            self._model.cpu()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self) -> None:
        """Load JamMa outdoor weights onto self.device."""
        if self._model is not None:
            self._model.to(self.device).eval()
            return

        logger.info("[JamMa] Loading outdoor model from HuggingFace …")

        ckpt_path = hf_hub_download(_HF_REPO, _CKPT_FILE)
        cfg = get_cfg_defaults()
        cfg.merge_from_file(
            os.path.join(_VENDOR_PATH, "configs", "jamma", "outdoor", "test.py")
        )
        cfg.freeze()

        model = JamMa(deepcopy(cfg.JAMMA))
        state = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        self._model = model.eval().to(self.device)
        logger.info("[JamMa] Model loaded.")

    # backward-compat alias
    load_model = load

    # ---------------------------------------------------------------- inference

    def _run(
        self,
        img_i: np.ndarray,
        img_j: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run JamMa matching, returns (pts1, pts2, conf) in original px coords."""
        self.load()

        h0, w0 = img_i.shape[:2]
        h1, w1 = img_j.shape[:2]

        # JamMa expects 832 × 832 or similar (divisible by 32)
        target_h, target_w = 832, 832

        def _prep(img: np.ndarray, th: int, tw: int) -> torch.Tensor:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (tw, th), interpolation=cv2.INTER_AREA)
            return torch.from_numpy(g).float()[None, None].to(self.device) / 255.0

        t0 = _prep(img_i, target_h, target_w)
        t1 = _prep(img_j, target_h, target_w)

        data = {
            "image0": t0, "image1": t1,
            "imagec_0": t0, "imagec_1": t1,
        }
        with torch.no_grad():
            self._model(data)

        if "mkpts0_f" not in data or len(data["mkpts0_f"]) == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty, np.empty(0, dtype=np.float32)

        pts1 = data["mkpts0_f"].cpu().numpy()  # (K, 2) in target_w × target_h
        pts2 = data["mkpts1_f"].cpu().numpy()
        conf = data["mconf"].cpu().numpy()

        # Scale back to original image coordinates
        pts1 = pts1 * np.array([w0 / target_w, h0 / target_h], dtype=np.float32)
        pts2 = pts2 * np.array([w1 / target_w, h1 / target_h], dtype=np.float32)
        return pts1, pts2, conf

    # ---------------------------------------------------------------- public API

    @lazy_load
    def match(
        self,
        img_i: np.ndarray,
        img_j: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._run(img_i, img_j)

    def match_masked(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: np.ndarray | None = None,
        mask2: np.ndarray | None = None,
        conf_thresh: float = 0.4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts1, pts2, conf = self._run(img1, img2)
        if len(pts1) == 0:
            return pts1, pts2, conf

        keep = conf > conf_thresh

        if mask1 is not None and len(pts1) > 0:
            h, w = mask1.shape[:2]
            ix = np.clip(pts1[:, 0].astype(int), 0, w - 1)
            iy = np.clip(pts1[:, 1].astype(int), 0, h - 1)
            keep &= mask1[iy, ix] > 0

        if mask2 is not None and len(pts2) > 0:
            h, w = mask2.shape[:2]
            jx = np.clip(pts2[:, 0].astype(int), 0, w - 1)
            jy = np.clip(pts2[:, 1].astype(int), 0, h - 1)
            keep &= mask2[jy, jx] > 0

        return pts1[keep], pts2[keep], conf[keep]

    def get_affine_partial(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask1: np.ndarray | None = None,
        mask2: np.ndarray | None = None,
        min_inliers: int = _MIN_INLIERS,
    ) -> tuple[np.ndarray | None, float]:
        pts1, pts2, conf = self.match_masked(img1, img2, mask1, mask2)
        if len(pts1) < min_inliers:
            return None, 0.0

        M_raw, inliers = cv2.estimateAffinePartial2D(
            pts1, pts2,
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
