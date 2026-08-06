"""
backend/src/models/stitch_net/losses.py
=========================================
Loss functions for AnimeStitchNet training.

Three complementary objectives are combined:

1. _AffineParamLoss
   Supervised Huber regression on the 4-DoF parameter vector.
   Negative pairs contribute zero loss (both pred and GT ≈ 0).

2. _PhotometricConsistencyLoss
   Self-supervised ZNCC photometric loss.
   Warps frame_j with the predicted params and computes zero-mean
   normalised cross-correlation with frame_i.  ZNCC is invariant to
   additive and multiplicative intensity changes (broadcast dimming).
   Gradients flow back through F.grid_sample into the network.

3. _SymmetricConsistencyLoss
   Penalises |pred_ij + pred_ji| — for small transforms, the inverse of
   the forward transform is approximately the negative, so both should sum
   to near zero.  Prevents degenerate constant-offset solutions.

Combined:
    L = λ_param · L_param  +  λ_photo(t) · L_photo  +  λ_sym · L_sym
where λ_photo(t) is ramped from 0 → λ_photo over `warmup_steps` to give the
supervised loss time to initialise the network before adding the noisier
photometric signal.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Differentiable warp helper
# ---------------------------------------------------------------------------


def warp_with_params(
    frame: torch.Tensor,  # (B, 1, H, W)
    params: torch.Tensor,  # (B, 4): [dx, dy, theta, log_s]
    img_h: int,
    img_w: int,
) -> torch.Tensor:
    """
    Differentiably warp `frame` using predicted affine params.
    Uses F.affine_grid + F.grid_sample for end-to-end gradient flow.
    Returns: (B, 1, H, W) warped frame.
    """
    dx, dy, theta, log_s = (params[:, 0], params[:, 1], params[:, 2], params[:, 3])
    s = torch.exp(log_s)
    cos_t = s * torch.cos(theta)
    sin_t = s * torch.sin(theta)

    # F.affine_grid uses NDC coordinates: tx_ndc = 2*tx_px/W
    row0 = torch.stack([cos_t, -sin_t, 2.0 * dx], dim=1)
    row1 = torch.stack([sin_t, cos_t, 2.0 * dy], dim=1)
    theta_mat = torch.stack([row0, row1], dim=1)  # (B, 2, 3)

    grid = F.affine_grid(theta_mat, frame.shape, align_corners=False)
    warped = F.grid_sample(
        frame, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return warped


# ---------------------------------------------------------------------------
# ZNCC photometric loss
# ---------------------------------------------------------------------------


def zncc_loss(
    a: torch.Tensor,  # (B, 1, H, W)
    b: torch.Tensor,
    mask: Optional[torch.Tensor] = None,  # (B, 1, H, W) float 1=valid
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    1 - zero-mean normalised cross-correlation, averaged over the batch.
    Result in [0, 2]; minimising → maximising image similarity.
    Invariant to additive and multiplicative intensity changes.
    """
    if mask is not None:
        n = mask.sum(dim=[1, 2, 3]).clamp(min=1.0)
        ma = (a * mask).sum(dim=[1, 2, 3]) / n
        mb = (b * mask).sum(dim=[1, 2, 3]) / n
        da = (a - ma.view(-1, 1, 1, 1)) * mask
        db = (b - mb.view(-1, 1, 1, 1)) * mask
    else:
        ma = a.mean(dim=[1, 2, 3], keepdim=True)
        mb = b.mean(dim=[1, 2, 3], keepdim=True)
        da = a - ma
        db = b - mb

    norm_a = (da**2).sum(dim=[1, 2, 3]).sqrt().clamp(min=eps)
    norm_b = (db**2).sum(dim=[1, 2, 3]).sqrt().clamp(min=eps)
    zncc = (da * db).sum(dim=[1, 2, 3]) / (norm_a * norm_b)
    return (1.0 - zncc).mean()


# ---------------------------------------------------------------------------
# 1. _AffineParamLoss
# ---------------------------------------------------------------------------


class _AffineParamLoss(nn.Module):
    """Supervised Huber regression on the (dx, dy, θ, log_s) vector."""

    def __init__(self, delta: float = 0.1):
        super().__init__()
        self.delta = delta

    def forward(
        self,
        pred: torch.Tensor,  # (B, 4)
        target: torch.Tensor,  # (B, 4)
        is_neg: torch.Tensor,  # (B,) bool
    ) -> torch.Tensor:
        # Zero out negative pairs (GT is already all-zero, but be explicit)
        pos = (~is_neg).float().unsqueeze(1)
        return F.huber_loss(
            pred * pos, target * pos, delta=self.delta, reduction="mean"
        )


# ---------------------------------------------------------------------------
# 2. _PhotometricConsistencyLoss
# ---------------------------------------------------------------------------


class _PhotometricConsistencyLoss(nn.Module):
    """
    ZNCC photometric loss on the warped pair.
    Skipped entirely for negative pairs (they don't overlap by definition).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        frame_i: torch.Tensor,  # (B, 1, H, W)
        frame_j: torch.Tensor,
        pred: torch.Tensor,  # (B, 4)
        is_neg: torch.Tensor,  # (B,) bool
        fg_mask_i: Optional[torch.Tensor] = None,  # (B, 1, H, W)
    ) -> torch.Tensor:
        pos = ~is_neg
        if not pos.any():
            return torch.tensor(0.0, device=frame_i.device)

        fi, fj, p = frame_i[pos], frame_j[pos], pred[pos]
        _, _, H, W = fi.shape

        warped = warp_with_params(fj, p, H, W)  # (K, 1, H, W)

        # Valid mask: warped pixels not in the zero-padding region
        valid = (warped.abs().sum(dim=1, keepdim=True) > 1e-4).float()
        if fg_mask_i is not None:
            bg = (fg_mask_i[pos] < 0.5).float()
            valid = valid * bg

        if valid.sum() < 10:
            return torch.tensor(0.0, device=frame_i.device)

        return zncc_loss(fi, warped, mask=valid)


# ---------------------------------------------------------------------------
# 3. _SymmetricConsistencyLoss
# ---------------------------------------------------------------------------


class _SymmetricConsistencyLoss(nn.Module):
    """
    Penalises |pred_ij + pred_ji| for positive pairs.
    For small transforms, inv(T) ≈ -T, so forward + backward ≈ 0.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_ij: torch.Tensor,  # (B, 4)
        pred_ji: torch.Tensor,  # (B, 4)
        is_neg: torch.Tensor,  # (B,) bool
    ) -> torch.Tensor:
        pos = (~is_neg).float().unsqueeze(1)
        return ((pred_ij + pred_ji) * pos).pow(2).mean()


# ---------------------------------------------------------------------------
# Combined StitchNetLoss
# ---------------------------------------------------------------------------


class StitchNetLoss(nn.Module):
    """
    Weighted combination of all three losses.

    Parameters
    ----------
    lambda_param  : weight for _AffineParamLoss.        Default 1.0
    lambda_photo  : weight for PhotometricLoss.         Default 0.5
    lambda_sym    : weight for SymmetricConsistency.    Default 0.2
    huber_delta   : Huber δ for param loss.             Default 0.1
    warmup_steps  : steps over which λ_photo is ramped in.
    """

    def __init__(
        self,
        lambda_param: float = 1.0,
        lambda_photo: float = 0.5,
        lambda_sym: float = 0.2,
        huber_delta: float = 0.1,
        warmup_steps: int = 500,
    ):
        super().__init__()
        self.lambda_param = lambda_param
        self.lambda_photo = lambda_photo
        self.lambda_sym = lambda_sym
        self.warmup_steps = warmup_steps
        self._step = 0

        self.param_loss = _AffineParamLoss(delta=huber_delta)
        self.photo_loss = _PhotometricConsistencyLoss()
        self.sym_loss = _SymmetricConsistencyLoss()

    def step(self):
        """Advance internal step counter (call once per optimiser step)."""
        self._step += 1

    @property
    def photo_weight(self) -> float:
        if self._step < self.warmup_steps:
            return self.lambda_photo * (self._step / max(self.warmup_steps, 1))
        return self.lambda_photo

    def forward(
        self,
        pred_ij: torch.Tensor,  # (B, 4)
        pred_ji: torch.Tensor,  # (B, 4)
        target: torch.Tensor,  # (B, 4)
        frame_i: torch.Tensor,  # (B, 1, H, W)
        frame_j: torch.Tensor,
        is_neg: torch.Tensor,  # (B,) bool
        fg_mask_i: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Returns dict with keys: total, param, photo, sym.
        All values are scalar tensors.
        """
        l_param = self.param_loss(pred_ij, target, is_neg)
        l_photo = self.photo_loss(frame_i, frame_j, pred_ij, is_neg, fg_mask_i)
        l_sym = self.sym_loss(pred_ij, pred_ji, is_neg)

        total = (
            self.lambda_param * l_param
            + self.photo_weight * l_photo
            + self.lambda_sym * l_sym
        )
        return {"total": total, "param": l_param, "photo": l_photo, "sym": l_sym}
