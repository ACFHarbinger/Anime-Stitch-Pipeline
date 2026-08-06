"""
backend/src/models/stitch_net/model.py
========================================
AnimeStitchNet — lightweight Siamese network that regresses a 4-DoF
affine-partial transform (dx, dy, θ, log_s) from a pair of overlapping
anime frames.

Architecture
------------
                 ┌──────────────────┐
  frame_i ──────►│                  │──► feat_i  (B, C, Hf, Wf)
                 │  Shared Encoder  │
  frame_j ──────►│  (MobileNetV3-S) │──► feat_j  (B, C, Hf, Wf)
                 └──────────────────┘
                          │
                 ┌────────▼─────────┐
                 │  2-D Sinusoidal  │  positional encoding
                 │   Pos Encoding   │
                 └────────┬─────────┘
                          │
                 ┌────────▼─────────┐
                 │ Cross-Attention  │  feat_i queries feat_j
                 │  (× N layers)    │  mask-aware: FG tokens suppressed
                 └────────┬─────────┘
                          │
                 ┌────────▼─────────┐
                 │  Global AvgPool  │
                 │   + MLP Head     │──► (dx, dy, θ, log_s)
                 └──────────────────┘

Output units
------------
  dx, dy  ∈ [-1, 1]  fraction of image width / height
  θ       ∈ [-π/6, π/6]  rotation in radians
  log_s   ∈ [-0.3, 0.3]  log-scale (±26% scale change)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ---------------------------------------------------------------------------
# 2-D Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

class _SinusoidalPosEnc2D(nn.Module):
    """Adds 2-D sinusoidal position codes to a (B, C, H, W) feature map."""

    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0, "channels must be divisible by 4"
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device
        half = C // 4
        div = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * (-math.log(10_000.0) / half)
        )
        pos_h = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1)
        pos_w = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(1)
        enc_h = torch.cat([torch.sin(pos_h * div), torch.cos(pos_h * div)], dim=1)
        enc_w = torch.cat([torch.sin(pos_w * div), torch.cos(pos_w * div)], dim=1)
        # enc_h: (H, C//2)  enc_w: (W, C//2)
        enc = enc_h.unsqueeze(1).expand(H, W, C // 2) + enc_w.unsqueeze(0).expand(
            H, W, C // 2
        )  # (H, W, C//2)
        enc = torch.cat([enc, enc], dim=2)  # (H, W, C)
        enc = enc.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        return x + enc

# ---------------------------------------------------------------------------
# Shared Encoder  (MobileNetV3-Small backbone)
# ---------------------------------------------------------------------------

class _AnimeEncoder(nn.Module):
    """
    Shared convolutional encoder.
    Input:  (B, 1, H, W) grayscale Y′ in [0, 1]
    Output: (B, out_channels, H//16, W//16)
    """

    def __init__(self, out_channels: int = 256, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Patch first Conv2d: 3-ch → 1-ch (average pretrained weights)
        first_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        if pretrained:
            new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
        backbone.features[0][0] = new_conv

        # Take all feature layers up to output (96 ch at stride 16)
        self.features = backbone.features

        # Project to desired channel width
        self.proj = nn.Sequential(
            nn.Conv2d(96, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(x))  # (B, C, H//16, W//16)

# ---------------------------------------------------------------------------
# Cross-Attention Layer
# ---------------------------------------------------------------------------

class _CrossAttentionLayer(nn.Module):
    """Single cross-attention + feed-forward layer."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q_tok: torch.Tensor,  # (B, L, C)
        kv_tok: torch.Tensor,  # (B, L, C)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, L) bool
    ) -> torch.Tensor:
        out, _ = self.attn(q_tok, kv_tok, kv_tok, key_padding_mask=key_padding_mask)
        q_tok = self.norm1(q_tok + out)
        q_tok = self.norm2(q_tok + self.ff(q_tok))
        return q_tok

# ---------------------------------------------------------------------------
# Regression Head
# ---------------------------------------------------------------------------

class _AffineHead(nn.Module):
    """Maps a global feature vector → (dx, dy, θ, log_s)."""

    def __init__(
        self,
        in_features: int,
        hidden: int = 512,
        dx_scale: float = 1.0,
        dy_scale: float = 1.0,
        max_angle: float = math.pi / 6,  # ±30°
        max_log_s: float = 0.3,  # ±26% scale
    ):
        super().__init__()
        self.dx_scale = dx_scale
        self.dy_scale = dy_scale
        self.max_angle = max_angle
        self.max_log_s = max_log_s
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 4),
        )
        # Initialise final layer to zero → network starts from identity transform
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(feat)  # (B, 4)
        t = torch.tanh(raw)
        return torch.stack(
            [
                t[:, 0] * self.dx_scale,
                t[:, 1] * self.dy_scale,
                t[:, 2] * self.max_angle,
                t[:, 3] * self.max_log_s,
            ],
            dim=1,
        )  # (B, 4)

# ---------------------------------------------------------------------------
# AnimeStitchNet
# ---------------------------------------------------------------------------

class AnimeStitchNet(nn.Module):
    """
    End-to-end anime frame alignment network.

    Given two overlapping frames as Y′ grayscale tensors, predicts the
    4-DoF affine-partial transform that maps frame_i onto frame_j:

        M = [s·cos θ,  -s·sin θ,  dx·W]
            [s·sin θ,   s·cos θ,  dy·H]

    Parameters
    ----------
    enc_channels  : feature width in the shared encoder.
    num_heads     : multi-head attention heads.
    num_ca_layers : number of stacked cross-attention layers.
    pretrained    : use ImageNet-pretrained MobileNetV3 backbone.
    """

    def __init__(
        self,
        enc_channels: int = 256,
        num_heads: int = 8,
        num_ca_layers: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.encoder = _AnimeEncoder(out_channels=enc_channels, pretrained=pretrained)
        self.pos_enc = _SinusoidalPosEnc2D(channels=enc_channels)
        self.ca_layers = nn.ModuleList(
            [
                _CrossAttentionLayer(embed_dim=enc_channels, num_heads=num_heads)
                for _ in range(num_ca_layers)
            ]
        )
        self.head = _AffineHead(in_features=enc_channels)

    # ------------------------------------------------------------------

    def forward(
        self,
        frame_i: torch.Tensor,  # (B, 1, H, W) float32 [0,1]
        frame_j: torch.Tensor,  # (B, 1, H, W)
        mask_i: Optional[torch.Tensor] = None,  # (B, 1, H, W) 1=foreground
        mask_j: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns (B, 4): [dx, dy, theta, log_s]."""
        # 1. Encode + positional encoding
        feat_i = self.pos_enc(self.encoder(frame_i))  # (B, C, Hf, Wf)
        feat_j = self.pos_enc(self.encoder(frame_j))
        B, C, Hf, Wf = feat_i.shape
        Hf * Wf

        # 2. Flatten → token sequences (B, L, C)
        tok_i = feat_i.flatten(2).permute(0, 2, 1)
        tok_j = feat_j.flatten(2).permute(0, 2, 1)

        # 3. Build key_padding_mask from foreground mask on frame_j
        kpm_j = None
        if mask_j is not None:
            m_j = F.interpolate(mask_j.float(), size=(Hf, Wf), mode="nearest")
            kpm_j = m_j.squeeze(1).flatten(1) > 0.5  # True = ignore token

        # 4. Cross-attention: tok_i queries tok_j
        for layer in self.ca_layers:
            tok_i = layer(tok_i, tok_j, key_padding_mask=kpm_j)

        # 5. Global average pool (suppress foreground tokens in frame_i)
        if mask_i is not None:
            m_i = F.interpolate(mask_i.float(), size=(Hf, Wf), mode="nearest")
            bg_w = 1.0 - m_i.squeeze(1).flatten(1).unsqueeze(-1)  # (B,L,1)
            tok_i = tok_i * bg_w

        global_feat = tok_i.mean(dim=1)  # (B, C)

        # 6. Regress params
        return self.head(global_feat)  # (B, 4)

    # ------------------------------------------------------------------
    # Static utilities for converting between param vectors and matrices
    # ------------------------------------------------------------------

    @staticmethod
    def params_to_affine(
        params: torch.Tensor,  # (B, 4) or (4,)
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """
        (B, 4) → (B, 2, 3) OpenCV-compatible affine matrix.
        M = [s·cos θ,  -s·sin θ,  dx·W]
            [s·sin θ,   s·cos θ,  dy·H]
        """
        if params.dim() == 1:
            params = params.unsqueeze(0)
        dx, dy, theta, log_s = (params[:, 0], params[:, 1], params[:, 2], params[:, 3])
        s = torch.exp(log_s)
        cos_t = s * torch.cos(theta)
        sin_t = s * torch.sin(theta)
        row0 = torch.stack([cos_t, -sin_t, dx * img_w], dim=1)
        row1 = torch.stack([sin_t, cos_t, dy * img_h], dim=1)
        return torch.stack([row0, row1], dim=1)  # (B, 2, 3)

    @staticmethod
    def affine_to_params(
        M: torch.Tensor,  # (B, 2, 3) or (2, 3)
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """Inverse of params_to_affine. Used for LoFTR knowledge distillation."""
        if M.dim() == 2:
            M = M.unsqueeze(0)
        dx = M[:, 0, 2] / img_w
        dy = M[:, 1, 2] / img_h
        s = (M[:, 0, 0] ** 2 + M[:, 1, 0] ** 2).sqrt()
        theta = torch.atan2(M[:, 1, 0], M[:, 0, 0])
        log_s = torch.log(s.clamp(min=1e-4))
        return torch.stack([dx, dy, theta, log_s], dim=1)  # (B, 4)
