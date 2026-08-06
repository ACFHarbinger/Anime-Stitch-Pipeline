"""
BiRefNet foreground / background masking for anime frames.
"""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
import warnings
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from backend.src.constants import FOREGROUND_DILATION, FOREGROUND_EROSION


def _compute_fg_masks(
    frames: List[np.ndarray],
    birefnet_wrapper,
    use_birefnet: bool = True,
) -> List[Optional[np.ndarray]]:
    """Returns list of background masks (255 = safe background, 0 = character)."""
    if not use_birefnet or birefnet_wrapper is None:
        return [None] * len(frames)

    # Detect which API version is loaded
    has_new_api = hasattr(birefnet_wrapper, "get_background_mask")

    masks: List[Optional[np.ndarray]] = []
    for i, img in enumerate(frames):
        try:
            if has_new_api:
                # New API: returns 255=background, 0=foreground, with dilation/erosion
                bg = birefnet_wrapper.get_background_mask(
                    img,
                    dilate_px=FOREGROUND_DILATION,
                    erode_px=FOREGROUND_EROSION,
                )
            else:
                # Legacy API: get_mask returns 255=foreground; invert + dilate manually
                fg = birefnet_wrapper.get_mask(img)
                bg = cv2.bitwise_not(fg)
                if FOREGROUND_DILATION > 0:
                    k = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (
                            2 * FOREGROUND_DILATION + 1,
                            2 * FOREGROUND_DILATION + 1,
                        ),
                    )
                    fg_dilated = cv2.dilate(fg, k)
                    bg = cv2.bitwise_not(fg_dilated)
            masks.append(bg)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[Stitch]   BiRefNet failed on frame {i}: {e}")
            masks.append(None)
    return masks


def _compute_fg_masks_sam2(  # noqa: C901
    frames: List[np.ndarray],
    birefnet_wrapper,
    use_birefnet: bool = True,
) -> List[Optional[np.ndarray]]:
    """
    SAM-2 video masking (§5.2) — temporally consistent foreground masks.

    Strategy:
      1. Run BiRefNet on frame 0 to get an initial fg bounding box prompt.
      2. Use SAM2VideoPredictor to propagate the bbox across all frames in one
         forward pass — hidden states keep mask boundaries stable across holds.
      3. Falls back to ``_compute_fg_masks`` (per-frame BiRefNet) on any error.

    Requires ``pip install sam2`` and checkpoint at ``$SAM2_CKPT``
    (default: ``~/.sam2/sam2_hiera_base_plus.pt``).

    Return convention: 255 = safe background, 0 = character foreground.
    """
    if not use_birefnet or birefnet_wrapper is None:
        return [None] * len(frames)

    has_new_api = hasattr(birefnet_wrapper, "get_background_mask")
    try:
        if has_new_api:
            _bg0 = birefnet_wrapper.get_background_mask(
                frames[0], dilate_px=FOREGROUND_DILATION, erode_px=FOREGROUND_EROSION
            )
        else:
            _bg0 = cv2.bitwise_not(birefnet_wrapper.get_mask(frames[0]))
        _fg_bin = (_bg0 < 128).astype(np.uint8) * 255
    except Exception:
        return _compute_fg_masks(frames, birefnet_wrapper, use_birefnet)

    _ys, _xs = np.where(_fg_bin > 127)
    if len(_ys) == 0:
        return _compute_fg_masks(frames, birefnet_wrapper, use_birefnet)

    _bbox = np.array(
        [[int(_xs.min()), int(_ys.min()), int(_xs.max()), int(_ys.max())]],
        dtype=np.float32,
    )

    try:
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore  # §3.14 lazy

        _ckpt = os.path.expanduser(
            os.environ.get("SAM2_CKPT", "~/.sam2/sam2_hiera_base_plus.pt")
        )
        _cfg = os.environ.get("SAM2_CFG", "sam2_hiera_b+.yaml")
        _device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

        predictor = build_sam2_video_predictor(_cfg, _ckpt, device=_device)
        _H, _W = frames[0].shape[:2]

        with tempfile.TemporaryDirectory() as _tmp:
            for _i, _f in enumerate(frames):
                cv2.imwrite(os.path.join(_tmp, f"{_i:06d}.jpg"), _f)

            _state = predictor.init_state(video_path=_tmp)
            predictor.add_new_points_or_box(
                inference_state=_state, frame_idx=0, obj_id=1, box=_bbox
            )

            masks_out: List[Optional[np.ndarray]] = [None] * len(frames)
            for _idx, _obj_ids, _logits in predictor.propagate_in_video(_state):
                if 1 in _obj_ids:
                    _li = list(_obj_ids).index(1)
                    _prob = torch.sigmoid(_logits[_li, 0]).cpu().numpy()
                    if _prob.shape != (_H, _W):
                        _prob = cv2.resize(_prob, (_W, _H), cv2.INTER_LINEAR)
                    _fg_i = (_prob > 0.5).astype(np.uint8) * 255
                    if FOREGROUND_DILATION > 0:
                        _k = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (2 * FOREGROUND_DILATION + 1, 2 * FOREGROUND_DILATION + 1),
                        )
                        _fg_i = cv2.dilate(_fg_i, _k)
                    masks_out[_idx] = cv2.bitwise_not(_fg_i)

            predictor.reset_state(_state)
            del predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Fill any missed frames with per-frame BiRefNet
        for _i, _m in enumerate(masks_out):
            if _m is None:
                try:
                    if has_new_api:
                        masks_out[_i] = birefnet_wrapper.get_background_mask(frames[_i])
                    else:
                        masks_out[_i] = cv2.bitwise_not(
                            birefnet_wrapper.get_mask(frames[_i])
                        )
                except Exception:
                    pass

        return masks_out

    except Exception as _e:
        print(f"[Stitch] SAM-2 unavailable ({_e}); falling back to per-frame BiRefNet.")
        return _compute_fg_masks(frames, birefnet_wrapper, use_birefnet)


def _compute_fg_masks_grounded_sam2(
    frames: List[np.ndarray],
    text_prompt: str,
    birefnet_wrapper,
    *,
    use_birefnet: bool = True,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
) -> List[Optional[np.ndarray]]:
    """
    Grounded SAM-2 masking — Issue 10A1.

    Flow:
      1. Run GroundingDINO on frame 0 with *text_prompt* to get a bounding box.
      2. Feed that bbox (instead of the BiRefNet-derived bbox) into SAM-2 video
         propagation across all frames.
      3. Falls back to ``_compute_fg_masks_sam2`` (BiRefNet bbox) if GroundingDINO
         is unavailable or produces no detection.
      4. Falls back to ``_compute_fg_masks`` (per-frame BiRefNet) as the final
         safety net.

    Parameters
    ----------
    frames        : list of uint8 BGR frames.
    text_prompt   : natural language description, e.g. "girl with blue hair".
    birefnet_wrapper : BiRefNet model wrapper (required for the fallback path).
    box_threshold : GroundingDINO box confidence threshold (default 0.35).
    text_threshold : GroundingDINO text confidence threshold (default 0.25).

    Return convention: 255 = safe background, 0 = character foreground.
    """
    if not text_prompt.strip():
        return _compute_fg_masks_sam2(
            frames, birefnet_wrapper, use_birefnet=use_birefnet
        )

    try:
        from backend.src.animation.hitl.grounding import (
            _detect_best_box,
        )  # §3.14 lazy — GroundingDINO is optional
    except ImportError:
        return _compute_fg_masks_sam2(
            frames, birefnet_wrapper, use_birefnet=use_birefnet
        )

    bbox = _detect_best_box(
        frames[0],
        text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    if bbox is None:
        print(
            f"[Stitch] GroundingDINO: no detection for '{text_prompt}'; falling back to BiRefNet bbox."
        )
        return _compute_fg_masks_sam2(
            frames, birefnet_wrapper, use_birefnet=use_birefnet
        )

    print(
        f"[Stitch] GroundingDINO: detected '{text_prompt}' bbox={bbox.astype(int).tolist()}"
    )

    # Try SAM-2 with the DINO-provided bbox
    try:
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore  # §3.14 lazy

        _ckpt = os.path.expanduser(
            os.environ.get("SAM2_CKPT", "~/.sam2/sam2_hiera_base_plus.pt")
        )
        _cfg = os.environ.get("SAM2_CFG", "sam2_hiera_b+.yaml")
        _device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        predictor = build_sam2_video_predictor(_cfg, _ckpt, device=_device)

        _H, _W = frames[0].shape[:2]
        _bbox_input = bbox.reshape(1, 4)

        with tempfile.TemporaryDirectory() as _tmp:
            for _i, _f in enumerate(frames):
                cv2.imwrite(os.path.join(_tmp, f"{_i:06d}.jpg"), _f)

            _state = predictor.init_state(video_path=_tmp)
            predictor.add_new_points_or_box(
                inference_state=_state, frame_idx=0, obj_id=1, box=_bbox_input
            )

            masks_out: List[Optional[np.ndarray]] = [None] * len(frames)
            for _idx, _obj_ids, _logits in predictor.propagate_in_video(_state):
                if 1 in _obj_ids:
                    _li = list(_obj_ids).index(1)
                    _prob = torch.sigmoid(_logits[_li, 0]).cpu().numpy()
                    if _prob.shape != (_H, _W):
                        _prob = cv2.resize(_prob, (_W, _H), cv2.INTER_LINEAR)
                    _fg_i = (_prob > 0.5).astype(np.uint8) * 255
                    if FOREGROUND_DILATION > 0:
                        _k = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (2 * FOREGROUND_DILATION + 1, 2 * FOREGROUND_DILATION + 1),
                        )
                        _fg_i = cv2.dilate(_fg_i, _k)
                    masks_out[_idx] = cv2.bitwise_not(_fg_i)

            predictor.reset_state(_state)
            del predictor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Fill missed frames with per-frame BiRefNet
        has_new_api = birefnet_wrapper is not None and hasattr(
            birefnet_wrapper, "get_background_mask"
        )
        for _i, _m in enumerate(masks_out):
            if _m is None and birefnet_wrapper is not None:
                try:
                    if has_new_api:
                        masks_out[_i] = birefnet_wrapper.get_background_mask(
                            frames[_i],
                            dilate_px=FOREGROUND_DILATION,
                            erode_px=FOREGROUND_EROSION,
                        )
                    else:
                        masks_out[_i] = cv2.bitwise_not(
                            birefnet_wrapper.get_mask(frames[_i])
                        )
                except Exception:
                    pass

        return masks_out

    except Exception as _e:
        print(
            f"[Stitch] Grounded SAM-2 failed ({_e}); falling back to per-frame BiRefNet."
        )
        return _compute_fg_masks(frames, birefnet_wrapper, use_birefnet)


def _compute_fg_masks_sam2_stateful(  # noqa: C901
    frames: List[np.ndarray],
    birefnet_wrapper,
    use_birefnet: bool = True,
) -> Tuple[List[Optional[np.ndarray]], Any, Any, Optional[str], int, int]:
    """
    SAM-2 video masking that keeps the predictor + inference_state alive — Issue 10A2 S83.

    Identical to ``_compute_fg_masks_sam2`` except:
    - Uses ``tempfile.mkdtemp()`` (caller owns the temp directory lifetime).
    - Does NOT call ``predictor.reset_state()`` or ``del predictor``.
    - Returns ``(masks, predictor, inference_state, tmp_dir, frame_h, frame_w)``
      so the caller can pass the live state to ``_refine_masks_with_clicks``.

    On any fallback path (SAM-2 absent, BiRefNet unavailable, any exception):
    - predictor, inference_state, tmp_dir are all ``None``.
    - masks come from the standard per-frame BiRefNet fallback.
    - Callers must call ``_cleanup_sam2_state`` with the returned values once done.
    """
    _H = frames[0].shape[0] if frames else 0
    _W = frames[0].shape[1] if frames else 0

    if not use_birefnet or birefnet_wrapper is None:
        return [None] * len(frames), None, None, None, _H, _W

    has_new_api = hasattr(birefnet_wrapper, "get_background_mask")
    try:
        if has_new_api:
            _bg0 = birefnet_wrapper.get_background_mask(
                frames[0], dilate_px=FOREGROUND_DILATION, erode_px=FOREGROUND_EROSION
            )
        else:
            _bg0 = cv2.bitwise_not(birefnet_wrapper.get_mask(frames[0]))
        _fg_bin = (_bg0 < 128).astype(np.uint8) * 255
    except Exception:
        fallback = _compute_fg_masks(frames, birefnet_wrapper, use_birefnet)
        return fallback, None, None, None, _H, _W

    _ys, _xs = np.where(_fg_bin > 127)
    if len(_ys) == 0:
        fallback = _compute_fg_masks(frames, birefnet_wrapper, use_birefnet)
        return fallback, None, None, None, _H, _W

    _bbox = np.array(
        [[int(_xs.min()), int(_ys.min()), int(_xs.max()), int(_ys.max())]],
        dtype=np.float32,
    )

    _tmp: Optional[str] = None
    try:
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore  # §3.14 lazy

        _ckpt = os.path.expanduser(
            os.environ.get("SAM2_CKPT", "~/.sam2/sam2_hiera_base_plus.pt")
        )
        _cfg = os.environ.get("SAM2_CFG", "sam2_hiera_b+.yaml")
        _device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

        predictor = build_sam2_video_predictor(_cfg, _ckpt, device=_device)

        _tmp = tempfile.mkdtemp(prefix="asp_sam2_")
        for _i, _f in enumerate(frames):
            cv2.imwrite(os.path.join(_tmp, f"{_i:06d}.jpg"), _f)

        _state = predictor.init_state(video_path=_tmp)
        predictor.add_new_points_or_box(
            inference_state=_state, frame_idx=0, obj_id=1, box=_bbox
        )

        masks_out: List[Optional[np.ndarray]] = [None] * len(frames)
        for _idx, _obj_ids, _logits in predictor.propagate_in_video(_state):
            if 1 in _obj_ids:
                _li = list(_obj_ids).index(1)
                _prob = torch.sigmoid(_logits[_li, 0]).cpu().numpy()
                if _prob.shape != (_H, _W):
                    _prob = cv2.resize(_prob, (_W, _H), cv2.INTER_LINEAR)
                _fg_i = (_prob > 0.5).astype(np.uint8) * 255
                if FOREGROUND_DILATION > 0:
                    _k = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (2 * FOREGROUND_DILATION + 1, 2 * FOREGROUND_DILATION + 1),
                    )
                    _fg_i = cv2.dilate(_fg_i, _k)
                masks_out[_idx] = cv2.bitwise_not(_fg_i)

        # Fill any missed frames with per-frame BiRefNet
        for _i, _m in enumerate(masks_out):
            if _m is None:
                try:
                    if has_new_api:
                        masks_out[_i] = birefnet_wrapper.get_background_mask(frames[_i])
                    else:
                        masks_out[_i] = cv2.bitwise_not(
                            birefnet_wrapper.get_mask(frames[_i])
                        )
                except Exception:
                    pass

        # Return live predictor + state — caller is responsible for cleanup
        return masks_out, predictor, _state, _tmp, _H, _W

    except Exception as _e:
        print(
            f"[Stitch] SAM-2 stateful unavailable ({_e}); falling back to per-frame BiRefNet."
        )
        if _tmp is not None:
            shutil.rmtree(_tmp, ignore_errors=True)
        fallback = _compute_fg_masks(frames, birefnet_wrapper, use_birefnet)
        return fallback, None, None, None, _H, _W


def _cleanup_sam2_state(
    predictor: Any,
    inference_state: Any,
    tmp_dir: Optional[str],
) -> None:
    """
    Release a live SAM-2 predictor state returned by ``_compute_fg_masks_sam2_stateful``.

    Safe to call with None values (no-op). Call this after HITL click refinement
    completes so GPU memory and the on-disk temp directory are freed.
    """
    try:
        if predictor is not None and inference_state is not None:
            predictor.reset_state(inference_state)
    except Exception:
        pass
    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _refine_masks_with_clicks(
    predictor,
    inference_state,
    *,
    pos_clicks: List[tuple],
    neg_clicks: List[tuple],
    frame_idx: int,
    frame_h: int,
    frame_w: int,
) -> List[Optional[np.ndarray]]:
    """
    Refine SAM-2 masks using positive and negative point prompts — Issue 10A2.

    Call this function from the MaskReviewDialog after the user has placed
    positive/negative clicks on the mask overlay. SAM-2 re-propagates the
    corrected segment across all frames starting from *frame_idx*.

    Parameters
    ----------
    predictor        : SAM-2 VideoPredictor (must already have an active state).
    inference_state  : the active SAM-2 inference state (from init_state).
    pos_clicks       : list of (x, y) pixel tuples for positive prompts.
    neg_clicks       : list of (x, y) pixel tuples for negative prompts.
    frame_idx        : the frame index to anchor the click refinement to.
    frame_h, frame_w : canvas dimensions for mask output sizing.

    Returns
    -------
    masks : list of uint8 (H, W) arrays — 255=background, 0=foreground.
            Returns empty list on error (caller should keep the previous masks).
    """
    if predictor is None or inference_state is None:
        return []
    if not pos_clicks and not neg_clicks:
        return []

    try:
        all_pts: List[List[float]] = []
        all_labels: List[int] = []
        for px, py in pos_clicks:
            all_pts.append([float(px), float(py)])
            all_labels.append(1)
        for px, py in neg_clicks:
            all_pts.append([float(px), float(py)])
            all_labels.append(0)

        pts_tensor = np.array(all_pts, dtype=np.float32)
        lbl_tensor = np.array(all_labels, dtype=np.int32)

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=1,
            points=pts_tensor,
            labels=lbl_tensor,
        )

        n_frames = inference_state.get("num_frames", 0) or len(
            inference_state.get("images", [])
        )
        masks_out: List[Optional[np.ndarray]] = [None] * n_frames
        for _idx, _obj_ids, _logits in predictor.propagate_in_video(inference_state):
            if 1 in _obj_ids:
                _li = list(_obj_ids).index(1)
                _prob = torch.sigmoid(_logits[_li, 0]).cpu().numpy()
                if _prob.shape != (frame_h, frame_w):
                    _prob = cv2.resize(_prob, (frame_w, frame_h), cv2.INTER_LINEAR)
                _fg_i = (_prob > 0.5).astype(np.uint8) * 255
                if FOREGROUND_DILATION > 0:
                    _k = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (2 * FOREGROUND_DILATION + 1, 2 * FOREGROUND_DILATION + 1),
                    )
                    _fg_i = cv2.dilate(_fg_i, _k)
                masks_out[_idx] = cv2.bitwise_not(_fg_i)

        return [m for m in masks_out if m is not None]

    except Exception as _e:
        warnings.warn(
            f"[ASP] _refine_masks_with_clicks failed: {_e}",
            RuntimeWarning,
            stacklevel=2,
        )
        return []


__all__ = [
    "_compute_fg_masks",
    "_compute_fg_masks_sam2",
    "_compute_fg_masks_sam2_stateful",
    "_cleanup_sam2_state",
    "_compute_fg_masks_grounded_sam2",
    "_refine_masks_with_clicks",
]
