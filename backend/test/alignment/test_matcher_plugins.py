"""Unit tests for Matcher abstract base class, plugins, and MatcherRegistry (§5.3 Architecture)."""

import numpy as np
from asp_backend.alignment.matching import (
    Matcher,
    MatcherRegistry,
    PhaseCorrelateMatcher,
    SegmentGuidedMatcher,
    TemplateMatcher,
)


class DummyMatcher(Matcher):
    """Test subclass for Matcher contract testing."""

    def __init__(self, name: str = "dummy", priority: int = 5, available: bool = True):
        super().__init__(name=name, priority=priority)
        self._available = available

    def match(self, img_i, img_j, m_i=None, m_j=None, **kwargs):
        M = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]], dtype=np.float32)
        return M, 0.85

    def is_available(self) -> bool:
        return self._available


def test_matcher_abstract_contract():
    dummy = DummyMatcher()
    assert dummy.name == "dummy"
    assert dummy.priority == 5
    assert dummy.is_available() is True

    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    img2 = np.zeros((100, 100, 3), dtype=np.uint8)
    M, conf = dummy.match(img1, img2)
    assert M is not None
    assert M.shape == (2, 3)
    assert conf == 0.85


def test_matcher_registry():
    MatcherRegistry.clear()
    m1 = DummyMatcher("m1", priority=30)
    m2 = DummyMatcher("m2", priority=10)
    m3 = DummyMatcher("m3", priority=20, available=False)

    MatcherRegistry.register(m1)
    MatcherRegistry.register(m2)
    MatcherRegistry.register(m3)

    assert MatcherRegistry.get("m1") is m1
    assert MatcherRegistry.get("m2") is m2

    available = MatcherRegistry.list_available()
    assert len(available) == 2
    assert available[0].name == "m2"  # Priority 10
    assert available[1].name == "m1"  # Priority 30


def test_concrete_matcher_plugins():
    MatcherRegistry.clear()
    tmpl = TemplateMatcher(priority=1)
    pc = PhaseCorrelateMatcher(priority=2)
    sg = SegmentGuidedMatcher(priority=3)

    MatcherRegistry.register(tmpl)
    MatcherRegistry.register(pc)
    MatcherRegistry.register(sg)

    available = MatcherRegistry.list_available()
    assert [m.name for m in available] == ["template", "phase_correlate", "segment_guided"]

    img1 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    img2 = img1.copy()

    M_pc, conf_pc = pc.match(img1, img2)
    assert M_pc is not None or conf_pc == 0.0
