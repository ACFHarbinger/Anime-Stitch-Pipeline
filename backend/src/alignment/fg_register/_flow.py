"""Dense optical flow estimation for foreground registration.

SEA-RAFT (via ``ptlflow``) is preferred when installed: it uses learned cost
volumes that remain informative over flat cel-shaded regions where DIS's
gradient-based aperture problem produces chaotic / zero flow vectors. Falls
back to OpenCV ``DISOpticalFlow`` when ptlflow/torch are unavailable.
"""

from __future__ import annotations

import contextlib
import os

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import ptlflow
except ImportError:
    ptlflow = None

_DIS_SINGLETON = None
_SEARAFT_SINGLETON = None
_SEARAFT_DEVICE = None

# SEA-RAFT is preferred when ptlflow is installed: it uses learned cost volumes
# that remain informative over flat cel-shaded regions where DIS's gradient-
# based aperture problem produces chaotic / zero flow vectors. The model is
# loaded lazily on first call and cached for the benchmark run.
_FLOW_ENGINE = os.environ.get("ASP_FLOW_ENGINE", "searaft").lower()
_USE_SEARAFT = _FLOW_ENGINE == "searaft"


def _get_dis():
    """Lazily construct a reusable DISOpticalFlow instance (MEDIUM preset)."""
    global _DIS_SINGLETON
    if _DIS_SINGLETON is None:
        _DIS_SINGLETON = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        with contextlib.suppress(Exception):
            _DIS_SINGLETON.setUseSpatialPropagation(True)
    return _DIS_SINGLETON


def _get_searaft():
    """
    Lazily load a pretrained RAFT-class model (ptlflow required).

    Load order (first success wins):
      1. ``sea_raft`` with ``ckpt_path='things'`` — the actual SEA-RAFT pretrain.
      2. ``raft`` with ``ckpt_path='things'`` — classic RAFT, well-tested.
      3. ``raft_small`` with ``ckpt_path='things'`` — lighter fallback.

    Returns (model, device) or (None, None) when ptlflow is unavailable.
    """
    global _SEARAFT_SINGLETON, _SEARAFT_DEVICE
    if _SEARAFT_SINGLETON is not None or _SEARAFT_DEVICE == "FAILED":
        return _SEARAFT_SINGLETON, _SEARAFT_DEVICE
    try:
        if torch is None or ptlflow is None:
            raise RuntimeError("torch or ptlflow not installed")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded_name = None
        model = None
        for name, ckpt in [
            ("sea_raft", "things"),
            ("sea_raft_s", "things"),
            ("raft", "things"),
            ("raft_small", "things"),
        ]:
            try:
                model = ptlflow.get_model(name, ckpt_path=ckpt).eval().to(device)
                loaded_name = f"{name}@{ckpt}"
                break
            except Exception:
                continue
        if model is None:
            raise RuntimeError("no RAFT variant with pretrained weights found")
        _SEARAFT_SINGLETON = model
        _SEARAFT_DEVICE = device
        print(f"[FGReg] {loaded_name} loaded on {device}")
        return _SEARAFT_SINGLETON, _SEARAFT_DEVICE
    except Exception as e:
        print(f"[FGReg] RAFT (ptlflow) unavailable ({e}); using DIS fallback")
        _SEARAFT_SINGLETON = None
        _SEARAFT_DEVICE = "FAILED"
        return None, None


def _dense_flow_searaft(
    prev_bgr: np.ndarray,
    next_bgr: np.ndarray,
    fg_mask: np.ndarray | None = None,
    max_side: int = 640,
) -> np.ndarray | None:
    """
    Dense optical flow ``prev → next`` using RAFT (ptlflow pretrained).

    To stay within VRAM, computes flow on a downscaled version of the images
    (longest side ≤ ``max_side`` px) then upscales the flow field back.  This
    is identical to the overlap-zone-crop strategy in ``flow_refine.py``.

    Returns (H, W, 2) float32 or None if unavailable.
    """
    model, device = _get_searaft()
    if model is None:
        return None
    try:
        H, W = prev_bgr.shape[:2]
        scale = min(1.0, max_side / max(H, W, 1))
        th, tw = max(8, int(H * scale)), max(8, int(W * scale))

        prev_s = cv2.resize(prev_bgr, (tw, th))
        next_s = cv2.resize(next_bgr, (tw, th))

        def _to_t(img: np.ndarray) -> torch.Tensor:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model({"images": torch.stack([_to_t(prev_s), _to_t(next_s)], dim=1)})
        # out['flows']: (1, 1, 2, th, tw) → (th, tw, 2)
        flow_s = out["flows"][0, 0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

        # Scale flow vectors back to full resolution
        if scale < 1.0:
            flow_full_x = cv2.resize(flow_s[:, :, 0], (W, H)) / scale
            flow_full_y = cv2.resize(flow_s[:, :, 1], (W, H)) / scale
            flow = np.stack([flow_full_x, flow_full_y], axis=2)
        else:
            flow = flow_s
        return flow
    except Exception as e:
        print(f"[FGReg] RAFT inference failed ({e}); using DIS")
        return None


def _sparse_flow_to_dense(
    flow_arrows: list[tuple[float, float, float, float]],
    H: int,
    W: int,
) -> np.ndarray:
    """§2.10C: Interpolate a sparse set of user-drawn displacement arrows to a dense flow field.

    Each arrow is ``(x, y, dx, dy)`` in canvas-pixel space (origin at top-left).
    The sparse samples are interpolated across the whole (H, W) canvas using
    ``scipy.ndimage.gaussian_filter``-smoothed RBF interpolation backed by a
    nearest-neighbour fill so every pixel gets a value.

    Returns an (H, W, 2) float32 flow field ``flow[y, x] = (dx, dy)``.

    Raises ``ValueError`` when *flow_arrows* is empty.
    """
    from scipy.interpolate import RBFInterpolator  # lazy import

    if not flow_arrows:
        raise ValueError("flow_arrows must not be empty")

    pts = np.array(
        [[a[1], a[0]] for a in flow_arrows], dtype=np.float64
    )  # (N, 2) as (row, col)
    vals_x = np.array([a[2] for a in flow_arrows], dtype=np.float64)
    vals_y = np.array([a[3] for a in flow_arrows], dtype=np.float64)

    # Grid of query points
    rows, cols = np.mgrid[0:H, 0:W]
    query = np.column_stack(
        [rows.ravel().astype(np.float64), cols.ravel().astype(np.float64)]
    )

    try:
        rbf_x = RBFInterpolator(pts, vals_x, kernel="thin_plate_spline", smoothing=1.0)
        rbf_y = RBFInterpolator(pts, vals_y, kernel="thin_plate_spline", smoothing=1.0)
        flow_x = rbf_x(query).reshape(H, W)
        flow_y = rbf_y(query).reshape(H, W)
    except Exception:
        # Nearest-neighbour fallback when RBF fails (degenerate point set)
        flow_x = np.zeros((H, W), dtype=np.float32)
        flow_y = np.zeros((H, W), dtype=np.float32)
        for x, y, dx, dy in flow_arrows:
            iy = int(np.clip(round(y), 0, H - 1))
            ix = int(np.clip(round(x), 0, W - 1))
            flow_x[iy, ix] = dx
            flow_y[iy, ix] = dy

    return np.stack([flow_x.astype(np.float32), flow_y.astype(np.float32)], axis=2)


def _dense_flow(prev_bgr: np.ndarray, next_bgr: np.ndarray) -> np.ndarray:
    """
    Dense optical flow ``prev → next``.

    Uses RAFT (A1, pretrained on optical flow datasets) when available for
    robust flat-region flow; falls back to OpenCV DISOpticalFlow.

    The input is expected to be the SEAM BAND CROP (small strip around the
    seam boundary), so ``max_side=1280`` gives RAFT good resolution without
    VRAM pressure.

    Returns an (H, W, 2) float32 array ``flow`` where
    ``prev[y, x]`` corresponds to ``next[y + flow[y,x,1], x + flow[y,x,0]]``.
    """
    if _USE_SEARAFT:
        # Use 1280 max-side: seam band crops are ~440px tall × 1900px wide,
        # which downscales to ≈295×1280 — good resolution without OOM.
        flow = _dense_flow_searaft(prev_bgr, next_bgr, max_side=1280)
        if flow is not None:
            return flow
    # DIS fallback
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_bgr, cv2.COLOR_BGR2GRAY)
    dis = _get_dis()
    flow = dis.calc(prev_gray, next_gray, None)
    return flow.astype(np.float32)


__all__ = [
    "_get_dis",
    "_get_searaft",
    "_dense_flow_searaft",
    "_sparse_flow_to_dense",
    "_dense_flow",
    "_FLOW_ENGINE",
]
