"""
backend/src/animation/flow_refine.py
================================
SEA-RAFT optical flow sub-pixel refinement (P2.1).

Replaces the ECC stage (Stage 8) with a learned flow estimator that works
correctly on flat cel-shaded anime regions where ECC diverges.

ECC fails because it minimises pixel-intensity correlation, requiring
non-zero image gradients to drive the optimiser.  Anime's large uniform
colour fields (sky, walls, costumes) produce near-zero gradients → the ECC
Hessian is near-singular → divergence.

SEA-RAFT uses a learned cost volume that remains informative even over flat
colour regions, and its rigid-motion pretraining is specifically optimised
for pure-translation scenes (exactly the anime pan case).

Implementation strategy
-----------------------
Only the *overlap zone* between adjacent frames is processed — a small
crop (≤ 512px) rather than the full frame, keeping VRAM usage low.  The
trimmed mean (25th–75th percentile) of background-pixel flow vectors gives
a robust per-pair sub-pixel translation estimate.
"""

from __future__ import annotations

from typing import List, Optional

# --------------------------------
import cv2
import numpy as np

# --- Relocated Nested Imports ---
import ptlflow
import torch

from backend.src.constants import FLOW_MAX_DRIFT, FLOW_PATCH_SIZE


def _load_sea_raft(device: str = "cpu"):
    """Load SEA-RAFT from ptlflow (cached after first call)."""
    try:
        # relocated: import ptlflow
        model = ptlflow.get_model("sea_raft")
        model = model.eval().to(device)
        return model
    except Exception as e:
        raise ImportError(f"ptlflow / sea_raft not available: {e}") from e


def _flow_refine(
    frames: List[np.ndarray],
    affines: List[np.ndarray],
    bg_masks: List[Optional[np.ndarray]],
    device: str = "cpu",
    raft_model=None,
) -> List[np.ndarray]:
    """
    Sub-pixel refinement of relative affines via SEA-RAFT optical flow.

    For each consecutive pair (i-1, i):
    1. Compute the canvas overlap zone from the two affines.
    2. Crop both frames to the overlap zone (≤ PATCH_SIZE px).
    3. Run SEA-RAFT to get dense flow in the overlap.
    4. Restrict to background pixels (bg_mask) to ignore character motion.
    5. Trimmed mean (25–75 pct) of (u, v) vectors → residual correction.
    6. Apply correction if |delta| < FLOW_MAX_DRIFT on each axis.

    Returns a new list of affines with sub-pixel corrections applied.
    """
    if raft_model is None:
        raft_model = _load_sea_raft(device)

    N = len(frames)
    refined = [affines[0].copy()]

    for i in range(1, N):
        M_prev = affines[i - 1]
        M_cur = affines[i]

        H_p, W_p = frames[i - 1].shape[:2]
        H_c, W_c = frames[i].shape[:2]

        ty_p = float(M_prev[1, 2])
        tx_p = float(M_prev[0, 2])
        ty_c = float(M_cur[1, 2])
        tx_c = float(M_cur[0, 2])

        # Canvas overlap bounding box
        ov_top = max(ty_p, ty_c)
        ov_bot = min(ty_p + H_p, ty_c + H_c)
        ov_left = max(tx_p, tx_c)
        ov_right = min(tx_p + W_p, tx_c + W_c)

        ov_h = int(ov_bot - ov_top)
        ov_w = int(ov_right - ov_left)

        if ov_h < 32 or ov_w < 32:
            refined.append(M_cur.copy())
            continue

        # Source-frame crop coordinates for the overlap zone
        r0_p = round(ov_top - ty_p)
        r1_p = r0_p + ov_h
        c0_p = round(ov_left - tx_p)
        c1_p = c0_p + ov_w
        r0_c = round(ov_top - ty_c)
        r1_c = r0_c + ov_h
        c0_c = round(ov_left - tx_c)
        c1_c = c0_c + ov_w

        # Clamp to valid frame bounds
        r0_p = max(0, r0_p)
        r1_p = min(H_p, r1_p)
        c0_p = max(0, c0_p)
        c1_p = min(W_p, c1_p)
        r0_c = max(0, r0_c)
        r1_c = min(H_c, r1_c)
        c0_c = max(0, c0_c)
        c1_c = min(W_c, c1_c)

        crop_p = frames[i - 1][r0_p:r1_p, c0_p:c1_p]
        crop_c = frames[i][r0_c:r1_c, c0_c:c1_c]

        if crop_p.size == 0 or crop_c.size == 0:
            refined.append(M_cur.copy())
            continue

        # Resize to at most FLOW_PATCH_SIZE for VRAM budget
        scale = min(1.0, FLOW_PATCH_SIZE / max(crop_p.shape[:2]))
        if scale < 1.0:
            new_h = int(crop_p.shape[0] * scale)
            new_w = int(crop_p.shape[1] * scale)
            crop_p = cv2.resize(crop_p, (new_w, new_h), interpolation=cv2.INTER_AREA)
            crop_c = cv2.resize(crop_c, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0

        # Background mask for the crop (restrict flow to background pixels)
        bg_p = bg_masks[i - 1]
        bm_crop = None
        if bg_p is not None:
            bm_c = bg_p[r0_p:r1_p, c0_p:c1_p]
            if scale < 1.0:
                bm_c = cv2.resize(
                    bm_c,
                    (crop_p.shape[1], crop_p.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            bm_crop = bm_c > 127

        # BGR → RGB float tensor (1, 3, H, W) in [0, 1]
        def _to_tensor(img: np.ndarray) -> torch.Tensor:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        t_p = _to_tensor(crop_p)
        t_c = _to_tensor(crop_c)

        try:
            with torch.no_grad():
                # ptlflow expects {'images': (B, 2, C, H, W)}
                out = raft_model({"images": torch.stack([t_p, t_c], dim=1)})
            # flows: (1, 1, 2, H, W) → (H, W, 2)
            flow = out["flows"][0, 0].permute(1, 2, 0).cpu().numpy()
        except Exception as _e:
            print(f"[FlowRefine]   SEA-RAFT failed on frame {i} ({_e}); keeping BA.")
            refined.append(M_cur.copy())
            continue

        u = flow[:, :, 0] / scale  # scale flow back to original px
        v = flow[:, :, 1] / scale

        # Background-only flow
        bg_sel = bm_crop if bm_crop is not None and bm_crop.shape == u.shape else np.ones(u.shape, dtype=bool)

        if bg_sel.sum() < 50:
            refined.append(M_cur.copy())
            continue

        u_bg = u[bg_sel]
        v_bg = v[bg_sel]

        # Trimmed mean (25–75 pct) for outlier robustness
        p25_u, p75_u = np.percentile(u_bg, [25, 75])
        p25_v, p75_v = np.percentile(v_bg, [25, 75])
        keep = (u_bg >= p25_u) & (u_bg <= p75_u) & (v_bg >= p25_v) & (v_bg <= p75_v)
        if keep.sum() < 20:
            refined.append(M_cur.copy())
            continue

        du = float(u_bg[keep].mean())  # residual correction in x
        dv = float(v_bg[keep].mean())  # residual correction in y

        if abs(du) > FLOW_MAX_DRIFT or abs(dv) > FLOW_MAX_DRIFT:
            print(
                f"[FlowRefine]   Frame {i}: SEA-RAFT correction clamped "
                f"(du={du:.1f}, dv={dv:.1f}); keeping BA."
            )
            refined.append(M_cur.copy())
            continue

        # flow is from frame_prev → frame_cur; residual correction to ty_cur:
        # corrected ty_c = ty_c + dv  (positive dv → cur is lower → ty increases)
        M_new = M_cur.copy()
        M_new[0, 2] = float(M_cur[0, 2]) + du
        M_new[1, 2] = float(M_cur[1, 2]) + dv
        refined.append(M_new)
        print(
            f"[FlowRefine]   Frame {i}: SEA-RAFT correction du={du:.2f} dv={dv:.2f} "
            f"(bg_pts={int(keep.sum())})"
        )

    return refined


__all__ = ["_flow_refine", "_load_sea_raft"]
