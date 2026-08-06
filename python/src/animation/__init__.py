"""
animation — anime panorama stitching package.

Top-level orchestrator :class:`AnimeStitchPipeline` lives in ``pipeline.py`` and
delegates each stage to a sibling module (matching, photometric, masking, ecc,
rendering, compositing, canvas, bundle_adjust).
"""

# §1.8A: auto-load backend/config/asp_config.toml before any module-level env flags are read.
# Uses os.environ.setdefault → never overrides manually set env vars.
try:
    from .core.config import load_asp_config as _load_asp_config

    _load_asp_config()
except Exception:
    pass

from .core.pipeline import AnimeStitchPipeline

__all__ = ["AnimeStitchPipeline"]
