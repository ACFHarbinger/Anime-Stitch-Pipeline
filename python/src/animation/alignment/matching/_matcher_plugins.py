"""Concrete Matcher plugin implementations wrapping single-strategy matchers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ._matchers import _phase_correlate, _segment_guided_match, _template_match
from .matcher_base import Matcher, MatcherRegistry


class TemplateMatcher(Matcher):
    """Bidirectional template matcher plugin."""

    def __init__(self, priority: int = 10) -> None:
        super().__init__(name="template", priority=priority)

    def match(
        self,
        img_i: np.ndarray,
        img_j: np.ndarray,
        m_i: Optional[np.ndarray] = None,
        m_j: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[Optional[np.ndarray], float]:
        H = img_i.shape[0]
        return _template_match(img_i, img_j, m_i, m_j, H=H, **kwargs)

    def is_available(self) -> bool:
        return True


class PhaseCorrelateMatcher(Matcher):
    """Phase correlation matcher plugin."""

    def __init__(self, priority: int = 20) -> None:
        super().__init__(name="phase_correlate", priority=priority)

    def match(
        self,
        img_i: np.ndarray,
        img_j: np.ndarray,
        m_i: Optional[np.ndarray] = None,
        m_j: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[Optional[np.ndarray], float]:
        return _phase_correlate(img_i, img_j, m_i, m_j, **kwargs)

    def is_available(self) -> bool:
        return True


class SegmentGuidedMatcher(Matcher):
    """Segment-guided matcher plugin (AnimeInterp technique)."""

    def __init__(self, priority: int = 30) -> None:
        super().__init__(name="segment_guided", priority=priority)

    def match(
        self,
        img_i: np.ndarray,
        img_j: np.ndarray,
        m_i: Optional[np.ndarray] = None,
        m_j: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[Optional[np.ndarray], float]:
        return _segment_guided_match(img_i, img_j, mask_i=m_i, mask_j=m_j, **kwargs)

    def is_available(self) -> bool:
        return True


# Auto-register default matcher plugins into MatcherRegistry
MatcherRegistry.register(TemplateMatcher())
MatcherRegistry.register(PhaseCorrelateMatcher())
MatcherRegistry.register(SegmentGuidedMatcher())
