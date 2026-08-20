"""
backend/src/models/roma_wrapper.py
===================================
RoMa v2 dense-warp last-resort matcher (P2.8).

RoMa (arXiv:2511.15706, installed as ``romatch``) uses frozen DINOv2 ViT
features (style-agnostic) to estimate a pixel-dense warp field between two
images.  DINOv2 features work on flat-shaded anime art where LoFTR and phase
correlation produce zero correspondences.

Used as Attempt 5 in _match_pair — the last resort before declaring failure.
Given the dense warp over background pixels, we take the trimmed-mean
translation as a 2-DOF estimate compatible with the BA anchor format.
"""

from __future__ import annotations

import contextlib
import logging

import cv2
import numpy as np
import torch
from backend.src.constants.models import _MAX_DRIFT_RATIO
from backend.src.models.core.base import ModelWrapper, lazy_load
from PIL import Image

logger = logging.getLogger(__name__)
try:
    from romatch import roma_outdoor

    _ROMA_OK = True
except ImportError:
    _ROMA_OK = False


class RoMaWrapper(ModelWrapper):
    """Wraps RoMa v2 for translation-only dense warp estimation."""

    def __init__(self, device: str | None = None):
        if not _ROMA_OK:
            raise ImportError(
                "romatch is required for RoMaWrapper. "
                "Install with: pip install git+https://github.com/Parskatt/RoMa.git"
            )
        super().__init__(device)
        self._model = None

    @classmethod
    def is_available(cls) -> bool:
        return _ROMA_OK

    def load(self) -> None:
        """Load the RoMa outdoor model onto self.device."""
        if self._model is None:
            logger.info("[RoMa] Loading RoMa outdoor model …")
            self._model = roma_outdoor(device=self.device)

    # backward-compat alias used by internal callers
    _load = load

    def unload(self) -> None:
        """Delete the model from VRAM/RAM."""
        if self._model is not None:
            del self._model
            self._model = None
        super().unload()

    def offload(self) -> None:
        """Move the model to CPU without synchronizing the CUDA allocator."""
        if self._model is not None:
            with contextlib.suppress(Exception):
                self._model.to("cpu")

    @lazy_load
    def match_translation(
        self,
        img_i: np.ndarray,
        img_j: np.ndarray,
        mask_i: np.ndarray | None = None,
        mask_j: np.ndarray | None = None,
        max_side: int = 512,
    ) -> tuple[np.ndarray | None, float]:
        """
        Estimate a 2-DOF translation from img_i → img_j via RoMa dense warp.

        Parameters
        ----------
        img_i, img_j : BGR uint8 images.
        mask_i, mask_j : uint8 background masks (255=background).
        max_side : resize input to at most this size to control VRAM.

        Returns
        -------
        M : (2, 3) float32 translation matrix, or None.
        conf : float confidence estimate (0.2–0.7 range).
        """
        h, w = img_i.shape[:2]

        # Resize for VRAM budget
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_i_rs = cv2.resize(img_i, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img_j_rs = cv2.resize(img_j, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask_i_rs = (
                cv2.resize(mask_i, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                if mask_i is not None else None
            )
        else:
            img_i_rs, img_j_rs, mask_i_rs = img_i, img_j, mask_i
            new_h, new_w = h, w
            scale = 1.0

        pil_i = Image.fromarray(cv2.cvtColor(img_i_rs, cv2.COLOR_BGR2RGB))
        pil_j = Image.fromarray(cv2.cvtColor(img_j_rs, cv2.COLOR_BGR2RGB))
        # @lazy_load guarantees self.load() ran (and thus self._model is set)
        # before this method body executes; mypy can't see through the decorator.
        assert self._model is not None
        try:
            with torch.no_grad():
                warp, certainty = self._model.match(pil_i, pil_j, device=self.device)
        except Exception as _e:
            logger.info(f"[RoMa] match failed: {_e}")
            return None, 0.0

        # warp: (H, W, 4) — (x_j, y_j, x_i, y_i) normalised to [-1,1]
        # certainty: (H, W) float
        if not isinstance(warp, np.ndarray):
            warp = warp.cpu().numpy()
        if not isinstance(certainty, np.ndarray):
            certainty = certainty.cpu().numpy()

        # Convert normalised coords to pixel offsets
        # RoMa norm: x_norm ∈ [-1,1] → x_px = (x_norm + 1) / 2 * (W - 1)
        W_m, H_m = new_w, new_h
        x_j = (warp[..., 0] + 1) / 2 * (W_m - 1)
        y_j = (warp[..., 1] + 1) / 2 * (H_m - 1)
        x_i = (warp[..., 2] + 1) / 2 * (W_m - 1)
        y_i = (warp[..., 3] + 1) / 2 * (H_m - 1)

        du = (x_j - x_i) / scale   # flow in original px coords
        dv = (y_j - y_i) / scale

        # Background mask filter
        bg_sel = np.ones(certainty.shape, dtype=bool)
        if mask_i_rs is not None and mask_i_rs.shape == certainty.shape:
            bg_sel &= mask_i_rs > 127
        # Certainty filter (RoMa certainty in [0, 1])
        cert_thresh = float(np.percentile(certainty, 60))
        bg_sel &= certainty > max(cert_thresh, 0.3)

        if bg_sel.sum() < 100:
            # Not enough background points — use all high-certainty points
            bg_sel = certainty > max(cert_thresh, 0.3)

        if bg_sel.sum() < 50:
            return None, 0.0

        du_bg = du[bg_sel]
        dv_bg = dv[bg_sel]

        # Trimmed mean (25–75 pct)
        p25_u, p75_u = np.percentile(du_bg, [25, 75])
        p25_v, p75_v = np.percentile(dv_bg, [25, 75])
        trim = (du_bg >= p25_u) & (du_bg <= p75_u) & (dv_bg >= p25_v) & (dv_bg <= p75_v)
        if trim.sum() < 20:
            return None, 0.0

        dx_est = float(du_bg[trim].mean())
        dy_est = float(dv_bg[trim].mean())

        # Reject if displacement is unreasonably large
        if abs(dx_est) > w * _MAX_DRIFT_RATIO:
            return None, 0.0

        M = np.array([[1, 0, dx_est], [0, 1, dy_est]], dtype=np.float32)
        conf = float(certainty[bg_sel].mean()) * 0.7  # cap lower than LoFTR
        return M, max(conf, 0.2)
