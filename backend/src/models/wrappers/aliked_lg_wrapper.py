"""
backend/src/models/aliked_lg_wrapper.py
=======================================
ALIKED + LightGlue sparse feature matcher wrapper (P2.3).

ALIKED (IEEE TIM 2023) uses a Sparse Deformable Descriptor Head that positions
sampling locations adaptively around each keypoint — significantly more robust
on anime line-art and gradient-sparse regions than fixed-grid SIFT/SuperPoint.
LightGlue (ICCV 2023) matches those descriptors with adaptive early exit,
stopping as soon as confidence is sufficient (40–60% compute reduction vs
SuperGlue on easy pairs).

Used as Attempt 1b in _match_pair when LoFTR returns < 20 background
keypoints, before falling back to template matching.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch

try:
    import kornia.feature as KF

    _KORNIA_OK = True
except ImportError:
    _KORNIA_OK = False

from backend.src.constants.models import WRAPPERS_ALIKED_LG_WRAPPER__MIN_INLIERS
from backend.src.models.core.base import ModelWrapper, lazy_load

logger = logging.getLogger(__name__)


class ALIKEDLightGlueWrapper(ModelWrapper):
    """
    ALIKED keypoint detector + LightGlue matcher via kornia.

    kornia >= 0.8 ships both:
      - ``KF.LightGlue(features='aliked')``   — the matcher
      - ``KF.LightGlueMatcher('aliked')``     — unified detector+matcher API

    The unified API requires LAFs (Local Affine Frames), so we use the
    lower-level KF.LightGlue + kornia's KeyNetAffNetHardNet-style extraction.
    Instead, we use ``KF.LightGlueMatcher`` which wraps the full pipeline.
    """

    def __init__(self, device: str | None = None):
        if not _KORNIA_OK:
            raise ImportError("kornia >= 0.8 is required for ALIKEDLightGlueWrapper.")
        self._matcher = None  # set before super().__init__ so loaded property is safe
        super().__init__(device)

    @classmethod
    def is_available(cls) -> bool:
        return _KORNIA_OK

    @property
    def loaded(self) -> bool:
        return getattr(self, "_matcher", None) is not None

    def load(self) -> None:
        """Load the ALIKED+LightGlue matcher onto self.device."""
        if self._matcher is None:
            logger.debug("[ALIKED+LG] Loading ALIKED+LightGlue matcher …")
            self._matcher = KF.LightGlueMatcher("aliked").eval().to(self.device)
        else:
            self._matcher.to(self.device).eval()

    # backward-compat alias used by internal callers
    _load = load

    def unload(self) -> None:
        """Delete matcher from VRAM/RAM, then flush CUDA cache."""
        if self._matcher is not None:
            self._matcher.cpu()
            del self._matcher
            self._matcher = None
        super().unload()

    def offload(self) -> None:
        if self._matcher is not None:
            self._matcher.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @lazy_load
    def match(
        self,
        img_i: np.ndarray,
        img_j: np.ndarray,
        mask_i: np.ndarray | None = None,
        mask_j: np.ndarray | None = None,
        conf_thresh: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect ALIKED keypoints and match with LightGlue.

        Parameters
        ----------
        img_i, img_j : BGR uint8 images (same size recommended).
        mask_i, mask_j : uint8 (H,W) background masks — 255=background.
        conf_thresh : minimum match confidence to retain.

        Returns
        -------
        pts_i, pts_j : (K,2) float32 in pixel coordinates of img_i / img_j.
        conf         : (K,) float32 match confidence.
        """

        h, w = img_i.shape[:2]

        # Convert to float32 grayscale tensors in [0,1] — kornia expects (1,1,H,W)
        def _to_tensor(img: np.ndarray) -> torch.Tensor:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            return torch.from_numpy(gray)[None, None].to(self.device)

        t_i = _to_tensor(img_i)
        t_j = _to_tensor(img_j)

        with torch.no_grad():
            # LightGlueMatcher.forward takes (desc1, desc2, lafs1, lafs2, hw1, hw2)
            # but the recommended high-level usage is via LocalFeatureMatcher.
            # Fallback: use the underlying KF.LightGlue via detect-then-match.
            matcher = self._matcher
            # kornia's LightGlueMatcher wraps a LocalFeatureMatcher internally.
            # Access it to run the full detect+match pipeline.
            if hasattr(matcher, "matcher") and hasattr(matcher.matcher, "forward"):
                # Detect keypoints with ALIKED via kornia's get_laf_descriptors
                try:
                    # Direct LightGlue pipeline (detector-free internal path not exposed).
                    # Use the lower-level path: LocalFeatureMatcher with ALIKED.
                    raise NotImplementedError("use lower-level path")
                except Exception:
                    pass

            # Lower-level path: KF.ALIKED (if available) + KF.LightGlue
            if hasattr(KF, "ALIKED"):
                aliked = KF.ALIKED(model_name="aliked-n16rot", max_num_keypoints=2048).eval().to(self.device)
                with torch.no_grad():
                    feats_i = aliked({"image": t_i})
                    feats_j = aliked({"image": t_j})
                lg = KF.LightGlue(features="aliked").eval().to(self.device)
                with torch.no_grad():
                    result = lg({"image0": feats_i, "image1": feats_j})
                kp_i = feats_i["keypoints"][0].cpu().numpy()  # (N, 2)
                kp_j = feats_j["keypoints"][0].cpu().numpy()
                m0 = result["matches0"][0].cpu().numpy()  # (N,) index into kp_j or -1
                valid = m0 >= 0
                pts_i = kp_i[valid]
                pts_j = kp_j[m0[valid]]
                mconf = result["matching_scores0"][0].cpu().numpy()[valid]
            else:
                # Kornia does not expose standalone ALIKED in this version;
                # fall back to DISK as a closely-related sparse detector.
                disk = KF.DISK.from_pretrained("depth").eval().to(self.device)
                with torch.no_grad():
                    feats_i = disk(t_i, n=2048, pad_if_not_divisible=True)
                    feats_j = disk(t_j, n=2048, pad_if_not_divisible=True)
                kp_i = feats_i[0].keypoints.cpu().numpy()   # (N, 2)
                kp_j = feats_j[0].keypoints.cpu().numpy()
                desc_i = feats_i[0].descriptors.cpu()        # (N, D)
                desc_j = feats_j[0].descriptors.cpu()

                laf_i = KF.laf_from_center_scale_ori(
                    torch.from_numpy(kp_i)[None].float(),
                    torch.ones(1, len(kp_i), 1, 1),
                    torch.zeros(1, len(kp_i), 1),
                ).to(self.device)
                laf_j = KF.laf_from_center_scale_ori(
                    torch.from_numpy(kp_j)[None].float(),
                    torch.ones(1, len(kp_j), 1, 1),
                    torch.zeros(1, len(kp_j), 1),
                ).to(self.device)

                lg = KF.LightGlue(features="disk").eval().to(self.device)
                with torch.no_grad():
                    result = lg({
                        "image0": {"keypoints": laf_i, "descriptors": desc_i[None].to(self.device)},
                        "image1": {"keypoints": laf_j, "descriptors": desc_j[None].to(self.device)},
                    })
                idxs = result["matches"][0].cpu().numpy()   # (M, 2)
                mconf = result["scores"][0].cpu().numpy()   # (M,)
                pts_i = kp_i[idxs[:, 0]]
                pts_j = kp_j[idxs[:, 1]]

        # Confidence filter
        keep = mconf >= conf_thresh
        pts_i, pts_j, mconf = pts_i[keep], pts_j[keep], mconf[keep]

        # Background mask filter
        if mask_i is not None and len(pts_i) > 0:
            hi, wi = mask_i.shape[:2]
            ix = np.clip(pts_i[:, 0].astype(int), 0, wi - 1)
            iy = np.clip(pts_i[:, 1].astype(int), 0, hi - 1)
            keep = mask_i[iy, ix] > 0
            pts_i, pts_j, mconf = pts_i[keep], pts_j[keep], mconf[keep]

        if mask_j is not None and len(pts_j) > 0:
            hj, wj = mask_j.shape[:2]
            jx = np.clip(pts_j[:, 0].astype(int), 0, wj - 1)
            jy = np.clip(pts_j[:, 1].astype(int), 0, hj - 1)
            keep = mask_j[jy, jx] > 0
            pts_i, pts_j, mconf = pts_i[keep], pts_j[keep], mconf[keep]

        return pts_i.astype(np.float32), pts_j.astype(np.float32), mconf.astype(np.float32)

    def get_translation(
        self,
        img_i: np.ndarray,
        img_j: np.ndarray,
        mask_i: np.ndarray | None = None,
        mask_j: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, float, np.ndarray, np.ndarray]:
        """
        Estimate a 2-DOF translation from img_i to img_j.

        Returns
        -------
        M         : (2,3) float32 translation matrix, or None.
        mean_conf : float confidence.
        pts_i     : (K,2) matched points in img_i coords.
        pts_j     : (K,2) matched points in img_j coords.
        """
        pts_i, pts_j, conf = self.match(img_i, img_j, mask_i, mask_j)
        if len(pts_i) < WRAPPERS_ALIKED_LG_WRAPPER__MIN_INLIERS:
            return None, 0.0, pts_i, pts_j

        dxs = pts_j[:, 0] - pts_i[:, 0]
        dys = pts_j[:, 1] - pts_i[:, 1]
        dx = float(np.median(dxs))
        dy = float(np.median(dys))

        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        return M, float(conf.mean()), pts_i, pts_j
