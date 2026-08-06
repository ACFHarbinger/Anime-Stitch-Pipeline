"""
backend/src/models/stitch_net/dataset.py
==========================================
SyntheticStitchDataset — generates training pairs from single anime images
by applying known random affine transforms, then supervising the network to
recover those transforms.  No labelled pairwise data is required.

Data pipeline (per sample)
--------------------------
1. Load a random image from disk → convert to Y′ grayscale float32 [0,1].
2. Sample a random 4-DoF affine transform (dx, dy, θ, log_s).
3. Extract an overlapping patch pair:
      frame_i  → random crop of the source image.
      frame_j  → apply the known affine warp to the source, then crop the
                 same spatial region.  This guarantees real overlap without
                 black-border padding.
4. Optionally apply:
      • MPEG DCT block noise  (simulates compression artefacts)
      • Broadcast dimming     (simulates inter-frame luminance shifts)
      • Gaussian motion blur  (fast panning shots)
      • Horizontal flip       (consistent across the pair → params unchanged)
5. Return {frame_i, frame_j, params, is_neg}.

Negative pairs
--------------
With probability `neg_pair_prob` the two patches come from entirely
different images (params ≈ 0).  This teaches the network to output near-
zero when frames do not overlap, improving robustness to scene cuts.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _luma(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → float32 Y′ in [0, 1]."""
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
    return y / 255.0

def _warp(img: np.ndarray, M: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    H, W = out_hw
    return cv2.warpAffine(
        img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

def _mpeg_noise(img: np.ndarray, strength: float = 0.03) -> np.ndarray:
    """Simulated MPEG DCT-block quantisation noise on a float32 grayscale image."""
    out = img.copy()
    block = 8
    H, W = out.shape
    for y in range(0, H - block + 1, block):
        for x in range(0, W - block + 1, block):
            q = random.randint(16, 64)
            pq = np.round(out[y : y + block, x : x + block] * q) / q
            n = np.random.uniform(-strength, strength, pq.shape).astype(np.float32)
            out[y : y + block, x : x + block] = np.clip(pq + n, 0.0, 1.0)
    return out

def _dimming(img: np.ndarray) -> np.ndarray:
    return np.clip(img * random.uniform(0.60, 1.00), 0.0, 1.0)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SyntheticStitchDataset(Dataset):
    """
    Parameters
    ----------
    image_dir       : directory of source anime images (PNG/JPG).
    patch_hw        : (H, W) crop size fed to the network.
    max_dx          : max |dx| as fraction of patch width.
    max_dy          : max |dy| as fraction of patch height.
    max_angle       : max |θ| in radians.
    max_log_s       : max |log(scale)|.
    mpeg_noise_prob : probability of adding MPEG noise to one / both patches.
    dimming_prob    : probability of applying broadcast dimming to frame_j.
    neg_pair_prob   : probability of producing a negative (non-overlapping) pair.
    dataset_size    : virtual epoch length (samples generated per epoch).
    augment         : enable on-the-fly augmentation.
    recursive       : search for images recursively inside image_dir.
    """

    def __init__(
        self,
        image_dir: str,
        patch_hw: Tuple[int, int] = (256, 256),
        max_dx: float = 0.50,
        max_dy: float = 0.50,
        max_angle: float = math.pi / 6,
        max_log_s: float = 0.25,
        mpeg_noise_prob: float = 0.30,
        dimming_prob: float = 0.40,
        neg_pair_prob: float = 0.10,
        dataset_size: int = 50_000,
        augment: bool = True,
        recursive: bool = True,
    ):
        super().__init__()
        self.patch_hw = patch_hw
        self.max_dx = max_dx
        self.max_dy = max_dy
        self.max_angle = max_angle
        self.max_log_s = max_log_s
        self.mpeg_noise_prob = mpeg_noise_prob
        self.dimming_prob = dimming_prob
        self.neg_pair_prob = neg_pair_prob
        self.dataset_size = dataset_size
        self.augment = augment

        base = Path(image_dir)
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        glob = base.rglob("*") if recursive else base.glob("*")
        self.paths: List[Path] = [
            p for p in glob if p.is_file() and p.suffix.lower() in exts
        ]
        if not self.paths:
            raise ValueError(f"No images found in '{image_dir}'.")
        print(f"[StitchDataset] {len(self.paths)} images in '{image_dir}'.")

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, _: int) -> dict:
        pH, pW = self.patch_hw

        # ── Sample transform params ───────────────────────────────────────
        dx = random.uniform(-self.max_dx, self.max_dx)
        dy = random.uniform(-self.max_dy, self.max_dy)
        theta = random.uniform(-self.max_angle, self.max_angle)
        log_s = random.uniform(-self.max_log_s, self.max_log_s)
        params = np.array([dx, dy, theta, log_s], dtype=np.float32)

        # ── Negative pair ─────────────────────────────────────────────────
        is_neg = (len(self.paths) >= 2) and (random.random() < self.neg_pair_prob)
        if is_neg:
            idx_i, idx_j = random.sample(range(len(self.paths)), 2)
            params = np.zeros(4, dtype=np.float32)
        else:
            idx_i = idx_j = random.randrange(len(self.paths))

        # ── Load images ───────────────────────────────────────────────────
        img_i = self._load(self.paths[idx_i], min_hw=(pH + 64, pW + 64))
        img_j = self._load(self.paths[idx_j], min_hw=(pH + 64, pW + 64))

        # ── Extract patch pair ────────────────────────────────────────────
        fi, fj = self._extract_pair(img_i, img_j, params)

        # ── Augmentation ─────────────────────────────────────────────────
        if self.augment:
            fi, fj = self._augment(fi, fj)

        return {
            "frame_i": torch.from_numpy(fi).unsqueeze(0),  # (1, H, W)
            "frame_j": torch.from_numpy(fj).unsqueeze(0),
            "params": torch.from_numpy(params),  # (4,)
            "is_neg": torch.tensor(is_neg, dtype=torch.bool),
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: Path, min_hw: Tuple[int, int]) -> np.ndarray:
        bgr = cv2.imread(str(path))
        if bgr is None:
            return np.ones(min_hw, dtype=np.float32)
        y = _luma(bgr)
        H, W = y.shape
        mH, mW = min_hw
        if mH > H or mW > W:
            scale = max(mH / H, mW / W) * 1.05
            y = cv2.resize(
                y, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_LANCZOS4
            )
        return y

    def _extract_pair(
        self,
        img_i: np.ndarray,
        img_j: np.ndarray,
        params: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pH, pW = self.patch_hw
        dx, dy, theta, log_s = params.tolist()
        s = math.exp(log_s)
        cos_t = s * math.cos(theta)
        sin_t = s * math.sin(theta)

        # frame_i: random crop from img_i
        iH, iW = img_i.shape
        y0 = random.randint(0, max(0, iH - pH))
        x0 = random.randint(0, max(0, iW - pW))
        fi = img_i[y0 : y0 + pH, x0 : x0 + pW].copy()

        # frame_j: warp img_j then take the same spatial crop
        M = np.array(
            [[cos_t, -sin_t, dx * pW], [sin_t, cos_t, dy * pH]], dtype=np.float32
        )
        jH, jW = img_j.shape
        warped = _warp(img_j, M, (jH, jW))
        y0j = min(max(y0, 0), jH - pH)
        x0j = min(max(x0, 0), jW - pW)
        fj = warped[y0j : y0j + pH, x0j : x0j + pW].copy()

        # Safety: guarantee exact patch size
        if fi.shape != (pH, pW):
            fi = cv2.resize(fi, (pW, pH))
        if fj.shape != (pH, pW):
            fj = cv2.resize(fj, (pW, pH))

        return fi.astype(np.float32), fj.astype(np.float32)

    def _augment(
        self,
        fi: np.ndarray,
        fj: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # MPEG noise (independent per patch)
        if random.random() < self.mpeg_noise_prob:
            tgt = random.choice(["i", "j", "both"])
            if tgt in ("i", "both"):
                fi = _mpeg_noise(fi)
            if tgt in ("j", "both"):
                fj = _mpeg_noise(fj)

        # Broadcast dimming on frame_j (inter-frame luminance shift)
        if random.random() < self.dimming_prob:
            fj = _dimming(fj)

        # Motion blur (fast panning)
        if random.random() < 0.15:
            k = random.choice([3, 5])
            fj = cv2.GaussianBlur(fj, (k, k), random.uniform(0.5, 1.5))

        # Horizontal flip (both patches → params unchanged)
        if random.random() < 0.30:
            fi = fi[:, ::-1].copy()
            fj = fj[:, ::-1].copy()

        return fi, fj

# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def stitch_collate_fn(batch: list) -> dict:
    return {
        "frame_i": torch.stack([b["frame_i"] for b in batch]),
        "frame_j": torch.stack([b["frame_j"] for b in batch]),
        "params": torch.stack([b["params"] for b in batch]),
        "is_neg": torch.stack([b["is_neg"] for b in batch]),
    }
