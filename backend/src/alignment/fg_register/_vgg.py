"""VGG-19 feature extraction for AnimeInterp-style segment-guided matching.

Not currently called anywhere in the active registration path (SGM was
disabled — see the exception message below) but preserved as-is from the
pre-split module rather than removed, since that's a separate, deliberate
cleanup decision outside the scope of a mechanical file split.
"""

from __future__ import annotations

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from skimage.segmentation import slic as _slic_fn  # type: ignore
except ImportError:
    _slic_fn = None

try:
    import torch.nn as nn
except ImportError:
    nn = None  # type: ignore[assignment]

_VGG19_SINGLETON = None
_VGG19_DEVICE: str | None = None


def _get_vgg19_feat():
    """
    Lazily load VGG-19 up to conv3_4 (28×28 feature map for 224-px input).
    Returns (model_partial, device) or (None, None) if torch/torchvision missing.
    """
    global _VGG19_SINGLETON, _VGG19_DEVICE
    if _VGG19_SINGLETON is not None or _VGG19_DEVICE == "FAILED":
        return _VGG19_SINGLETON, _VGG19_DEVICE
    if torch is None or nn is None:
        raise RuntimeError("torch not installed")
    try:
        import torchvision.models as tvm  # §3.14 lazy

        device = "cuda" if torch.cuda.is_available() else "cpu"
        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1).features
        # conv3_4 is index 18 in VGG-19 features (0-indexed, after pool2 block)
        partial = nn.Sequential(*list(vgg.children())[:19]).to(device).eval()
        _VGG19_SINGLETON = partial
        _VGG19_DEVICE = device
        return _VGG19_SINGLETON, _VGG19_DEVICE
    except Exception as e:
        print(f"[FGReg] VGG-19 unavailable ({e}); AnimeInterp SGM disabled")
        _VGG19_SINGLETON = None
        _VGG19_DEVICE = "FAILED"
        return None, None


__all__ = ["_get_vgg19_feat"]
