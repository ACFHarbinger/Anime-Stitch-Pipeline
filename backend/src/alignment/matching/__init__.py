"""Pairwise feature matching for anime stitching (§5.17 split).

Each function is standalone (no class state).  ``loftr_wrapper`` is passed in
explicitly when LoFTR matching is enabled.

Split by role: ``_math.py`` (geometry/statistics helpers), ``_sampling.py``
(background-point sampling), ``_matchers.py`` (single-strategy matchers),
``_pairwise.py`` (fallback-chain orchestration). All public names re-exported
here for backward compatibility.
"""

from ._matcher_plugins import PhaseCorrelateMatcher, SegmentGuidedMatcher, TemplateMatcher
from ._matchers import _phase_correlate, _segment_guided_match, _template_match
from ._math import _compute_bg_match_ratio, _compute_translation_spread, _extract_similarity
from ._pairwise import TemporalPairProposal, _match_pair, _pairwise_match, propose_temporal_pairs
from ._sampling import _sample_bg_points, _sample_bg_points_grid
from .matcher_base import Matcher, MatcherRegistry

__all__ = [
    "Matcher",
    "MatcherRegistry",
    "TemplateMatcher",
    "PhaseCorrelateMatcher",
    "SegmentGuidedMatcher",
    "_template_match",
    "_phase_correlate",
    "_sample_bg_points",
    "_sample_bg_points_grid",
    "_segment_guided_match",
    "_match_pair",
    "_pairwise_match",
    "TemporalPairProposal",
    "propose_temporal_pairs",
    "_extract_similarity",
    "_compute_translation_spread",
    "_compute_bg_match_ratio",
]
