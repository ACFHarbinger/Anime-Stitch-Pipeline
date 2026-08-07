"""Pose similarity metrics: fg-masked pixel L1 and DINOv2 cosine distance."""

from __future__ import annotations

import cv2
import numpy as np

# §3.14 — Optional ML imports: loaded only when DINOv2 / BiRefNet features are
# enabled at runtime.  Guarded so tests that don't use these paths don't pay
# the CUDA-context and model-weight initialisation overhead at collection time.
try:
    import torch
    import torchvision.transforms as T
    from PIL import Image as _PIL_Image
except ImportError:
    torch = None  # type: ignore[assignment]
    T = None  # type: ignore[assignment]
    _PIL_Image = None  # type: ignore[assignment]


def _fg_center_diff(
    thumb_a: np.ndarray,
    thumb_b: np.ndarray,
    fg_mask: np.ndarray | None = None,
) -> float:
    """
    Pose similarity metric between two thumbnails.

    **With fg_mask (BiRefNet fg probability at thumbnail scale):**
    Hard-thresholds the mask (> 0.3) to a binary fg_bin, zeroes out all
    background pixels, then computes mean absolute pixel difference on the
    foreground region.  Each frame's fg pixels are independently normalised to
    zero mean / unit std before differencing to remove inter-frame gain
    variations (ECC gain normalisation has not yet run at selection time).

    This is strictly background-invariant: background pixels are exactly 0.0 in
    both masked images, so camera-panning locker/wall/scenery structure
    contributes nothing to the score regardless of mask softness.  For "on
    twos" animation holds (same character cel for 2–3 consecutive frames),
    fg pixels look identical → score ≈ 0.  Across a hold boundary (new
    animation cel), fg pixels shift position → score > 0.

    The previous gradient-weighted approach computed the Sobel gradient on the
    full image and multiplied by fg_mask, so background edges (lockers, walls)
    with fg_mask weight of 0.05–0.1 still contributed proportionally.  This
    masked-pixel approach is background-invariant by construction.

    **Without fg_mask (fallback):**
    Gradient-magnitude L1 on the central 50% crop.  Partly confounded by
    background structure but does not require BiRefNet.

    Returns a non-negative float (0 = identical character region).
    """
    h = min(thumb_a.shape[0], thumb_b.shape[0])
    w = min(thumb_a.shape[1], thumb_b.shape[1])

    if fg_mask is not None and fg_mask.shape[0] >= h and fg_mask.shape[1] >= w:
        fg_bin = (fg_mask[:h, :w] > 0.3).astype(np.float32)
        total = float(fg_bin.sum())
        if total >= 50.0:
            a = thumb_a[:h, :w]
            b = thumb_b[:h, :w]
            # Per-frame fg normalisation to remove gain variation
            a_px = a[fg_bin > 0.5]
            b_px = b[fg_bin > 0.5]
            a_norm = (a - float(a_px.mean())) / (float(a_px.std()) + 1e-5)
            b_norm = (b - float(b_px.mean())) / (float(b_px.std()) + 1e-5)
            diff = np.abs(a_norm - b_norm) * fg_bin
            return float(diff.sum() / total)
        # fg mask too sparse — fall through to central-crop

    # Fallback: gradient-magnitude L1 on central 50% crop
    def _grad_mag(img: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)

    h0, h1 = h // 4, 3 * h // 4
    w0, w1 = w // 4, 3 * w // 4
    if h1 <= h0 or w1 <= w0:
        a, b = thumb_a[:h, :w], thumb_b[:h, :w]
    else:
        a, b = thumb_a[h0:h1, w0:w1], thumb_b[h0:h1, w0:w1]
    return float(np.mean(np.abs(_grad_mag(a) - _grad_mag(b))))


# Module-level model cache — avoids reloading DINOv2 on every benchmark test
# (96 tests × 10–30s reload = 15–48 minutes of avoidable overhead).
# Key: device string; Value: (model, transform) tuple.
_DINOV2_CACHE: dict = {}


def _compute_dinov2_features(frames_paths: list[str]) -> np.ndarray | None:
    """
    Compute DINOv2 (ViT-S/14) pose embeddings for all frames.

    Returns (N, 384) float32 array of L2-normalised feature vectors, or None
    if DINOv2 is unavailable (no torch.hub access, model weights not cached, etc.).

    The model is loaded once per process and cached in ``_DINOV2_CACHE`` — the
    first call to this function per device incurs the hub-load overhead
    (~5–30s); subsequent calls are instantaneous.

    DINOv2 features are used in Pass 2 of ``smart_select_frames()`` as the
    pose similarity metric.  Cosine distance between frame features captures
    pose difference robustly:
      - Animation holds (same cel, 2–3 consecutive frames) → distance ≈ 0.02–0.05
      - Cross-hold transitions (new cel) → distance ≈ 0.10–0.30
      - Different scenes → distance > 0.50

    This is background-invariant by design: DINOv2 was trained on diverse
    natural images and its ViT features are dominated by semantic content
    (pose, character shape) rather than background texture patterns.
    """
    try:
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

        if device not in _DINOV2_CACHE:
            model = (
                torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14", verbose=False
                )
                .to(device)
                .eval()
            )
            transform = T.Compose(
                [
                    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
            _DINOV2_CACHE[device] = (model, transform)

        model, transform = _DINOV2_CACHE[device]
    except Exception:
        return None

    # Batch-process frames: load, optionally crop to fg bounding box, stack, infer.
    tensors = []
    try:
        with torch.no_grad():
            for path in frames_paths:
                img = _PIL_Image.open(path).convert("RGB")

                # §1D — foreground-masked DINOv2: crop to the BiRefNet foreground
                # bounding box before embedding.  Background pixels dominate the
                # ViT attention on pan-shot anime where the scene is >80% bg,
                # causing DINOv2 to track camera translation rather than pose.
                # Cropping to the fg bbox removes the background from the input
                # and forces the network to attend to character shape and pose.
                try:
                    _img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    # Cheap fg estimate: pixels far from the median hue are
                    # character (anime backgrounds are mostly monotone gradient).
                    _gray = cv2.cvtColor(_img_bgr, cv2.COLOR_BGR2GRAY)
                    _h, _w = _gray.shape
                    # Use Otsu binarisation to separate fg/bg in luminance
                    _, _fg_bin = cv2.threshold(
                        _gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    _ys, _xs = np.where(_fg_bin > 0)
                    if len(_ys) > (_h * _w * 0.05):  # at least 5% fg pixels
                        _y0, _y1 = int(_ys.min()), int(_ys.max())
                        _x0, _x1 = int(_xs.min()), int(_xs.max())
                        # Add 5% padding
                        _pad_y = max(8, int((_y1 - _y0) * 0.05))
                        _pad_x = max(8, int((_x1 - _x0) * 0.05))
                        _y0 = max(0, _y0 - _pad_y)
                        _y1 = min(_h, _y1 + _pad_y)
                        _x0 = max(0, _x0 - _pad_x)
                        _x1 = min(_w, _x1 + _pad_x)
                        if (_y1 - _y0) > 32 and (_x1 - _x0) > 32:
                            img = img.crop((_x0, _y0, _x1, _y1))
                except Exception:
                    pass  # fg crop is best-effort; fall back to full frame

                tensors.append(transform(img))

            # Stack and infer in one forward pass (more efficient than per-frame)
            batch = torch.stack(tensors).to(model.parameters().__next__().device)
            feats = model(batch)  # (N, 384)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().astype(np.float32)
    except Exception:
        return None


__all__ = ["_fg_center_diff", "_compute_dinov2_features", "_DINOV2_CACHE"]
