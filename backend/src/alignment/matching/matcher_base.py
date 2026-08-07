"""Abstract Matcher base class and MatcherRegistry (§5.3 Architecture)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Matcher(ABC):
    """
    Abstract base class for pairwise frame matchers.

    Parameters
    ----------
    name : str
        Unique identifier for the matcher (e.g., 'template', 'phase_correlate', 'segment_guided').
    priority : int
        Execution priority order (lower numbers run earlier in fallback chain).
    """

    def __init__(self, name: str, priority: int = 10) -> None:
        self.name = name
        self.priority = priority

    @abstractmethod
    def match(
        self,
        img_i: np.ndarray,
        img_j: np.ndarray,
        m_i: np.ndarray | None = None,
        m_j: np.ndarray | None = None,
        **kwargs,
    ) -> tuple[np.ndarray | None, float]:
        """
        Estimate 2D affine transform matrix M and confidence score between img_i and img_j.

        Parameters
        ----------
        img_i : np.ndarray
            First image frame.
        img_j : np.ndarray
            Second image frame.
        m_i : Optional[np.ndarray]
            Binary mask for img_i.
        m_j : Optional[np.ndarray]
            Binary mask for img_j.

        Returns
        -------
        Tuple[Optional[np.ndarray], float]
            Transform matrix M of shape (2, 3) or None, and float confidence in [0.0, 1.0].
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this matcher's dependencies and runtime requirements are met."""
        return True


class MatcherRegistry:
    """Registry for discovering and managing pairwise frame matchers."""

    _matchers: dict[str, Matcher] = {}

    @classmethod
    def register(cls, matcher: Matcher) -> None:
        """Register a Matcher instance."""
        cls._matchers[matcher.name] = matcher

    @classmethod
    def get(cls, name: str) -> Matcher | None:
        """Retrieve a registered Matcher by name."""
        return cls._matchers.get(name)

    @classmethod
    def list_available(cls) -> list[Matcher]:
        """List all registered and available matchers, sorted by priority."""
        available = [m for m in cls._matchers.values() if m.is_available()]
        available.sort(key=lambda m: m.priority)
        return available

    @classmethod
    def clear(cls) -> None:
        """Clear all registered matchers."""
        cls._matchers.clear()
