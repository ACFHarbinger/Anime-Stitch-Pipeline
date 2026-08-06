"""Shared optional-dependency probes for the ASP pipeline.

§3.14 — Heavy model wrapper imports are deferred to first use. Each
module-level try/except was loading kornia/transformers/torchvision at
pytest collection time, contributing to the test-suite freeze (S140 root
causes). We probe availability cheaply with importlib.util.find_spec(); the
actual class is imported inside the method that instantiates it.

Centralised here (rather than duplicated per-file) since __init__.py,
run_stage.py, and _matcher_selection.py all need the same *_OK flags to
decide which matcher/model to construct.
"""

from __future__ import annotations

import importlib.util as _importlib_util_pipeline
import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

try:
    import base as _batch
    if getattr(_batch, "__file__", None) is None or not (
        hasattr(_batch, "frame_selection") or hasattr(_batch, "matching")
    ):
        raise ImportError("compiled base ASP extension not available")
    _HAS_BATCH: bool = True
except ImportError:
    _batch = None  # type: ignore[assignment]
    _HAS_BATCH: bool = False

# BaSiCWrapper only uses cv2/numpy/torch — safe to import at module level.
try:
    from backend.src.models.wrappers.basic_wrapper import BaSiCWrapper

    _BASIC_OK = True
except ImportError:
    BaSiCWrapper = None  # type: ignore[assignment]
    _BASIC_OK = False

# birefnet_wrapper → transformers; kornia wrappers → kornia+torchvision; EfficientLoFTR → transformers
_BIREFNET_OK: bool = _importlib_util_pipeline.find_spec("transformers") is not None
_LOFTR_OK: bool = _importlib_util_pipeline.find_spec("kornia") is not None
_ELOFTR_OK: bool = _importlib_util_pipeline.find_spec("transformers") is not None
_ALIKED_OK: bool = _importlib_util_pipeline.find_spec("kornia") is not None

# roma_wrapper: romatch library (not typically installed)
try:
    from animation.models.wrappers.roma_wrapper import RoMaWrapper

    _ROMA_OK = True
except ImportError:
    RoMaWrapper = None  # type: ignore[assignment]
    _ROMA_OK = False

try:
    from animation.flow.flow_refine import _flow_refine, _load_sea_raft

    _SEA_RAFT_OK = True
except ImportError:
    _flow_refine = None  # type: ignore[assignment]
    _load_sea_raft = None  # type: ignore[assignment]
    _SEA_RAFT_OK = False

# §4.7 — dy_cv pre-detection gate.
# Coefficient of variation of adjacent vertical frame steps.  When dy_cv ≥ threshold
# the pipeline immediately falls back to SCANS before expensive ARAP/BiRefNet work.
# 97-test benchmark: dy_cv ≥ 1.5 → catastrophic ASP failure (AlSSIM −22 to −37%,
# seam_vis 60–120 vs SCANS 2–3) while SCANS handles these sequences trivially.
# Default 1.5 (enabled). Set ASP_DY_CV_MAX=0 to disable.
_DY_CV_MAX: float = float(os.environ.get("ASP_DY_CV_MAX", "1.5"))
_USE_SAM2: bool = os.environ.get("ASP_USE_SAM2", "0") != "0"


__all__ = [
    "torch",
    "Image",
    "_batch",
    "_HAS_BATCH",
    "BaSiCWrapper",
    "_BASIC_OK",
    "_BIREFNET_OK",
    "_LOFTR_OK",
    "_ELOFTR_OK",
    "_ALIKED_OK",
    "RoMaWrapper",
    "_ROMA_OK",
    "_flow_refine",
    "_load_sea_raft",
    "_SEA_RAFT_OK",
    "_DY_CV_MAX",
    "_USE_SAM2",
]
