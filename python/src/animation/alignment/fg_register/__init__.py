"""Foreground pose registration (Stage 8.5), split by role (§5.17).

  _flow.py      -- dense optical flow estimation (SEA-RAFT/DIS, sparse->dense)
  _vgg.py       -- VGG-19 feature extraction (unused SGM leftover, preserved)
  _arap.py      -- ARAP Push + Regularise (Sýkora 2009)
  _geometry.py  -- seam taper weight + displacement remap
  register.py   -- register_foreground_at_seam(), the public entry point

This module is import-safe for headless use (no Qt, no torch import at module
load) and has no side effects.
"""

from ._arap import _arap_push, _arap_regularise
from ._flow import _FLOW_ENGINE, _dense_flow
from ._geometry import _seam_taper
from .register import register_foreground_at_seam

__all__ = [
    "register_foreground_at_seam",
    "_dense_flow",
    "_seam_taper",
    "_arap_regularise",
    "_arap_push",
    "_FLOW_ENGINE",
]
