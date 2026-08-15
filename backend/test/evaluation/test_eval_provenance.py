"""
Tests for the case-provenance schema (M0, issue #24; C0.5 dual-veto, issue #41).

Covers: round-trip serialization, the dual-veto OR-logic for exclusion vs.
AND-logic for inclusion, and that observations/adjudications are append-only.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _repo_root)

from asp_backend_evaluation.other.provenance import (  # noqa: E402
    SOURCE_HUMAN,
    CaseProvenance,
    SafetyAdjudication,
    SafetyObservation,
    load_provenance,
    save_provenance,
)


def test_round_trip_minimal():
    entry = CaseProvenance(case_id="asp_test04")
    restored = CaseProvenance.from_dict(json.loads(json.dumps(entry.to_dict())))
    assert restored.case_id == "asp_test04"
    assert restored.source_work_nsfw is None
    assert restored.content_tags == []
    assert restored.safety_tier is None


def test_round_trip_full():
    entry = CaseProvenance(
        case_id="asp_test50",
        corpus_id="sfw_q3",
        source_url="https://safebooru.org/index.php?page=post&s=view&id=1",
        source_board="safebooru",
        licence="unknown",
        web_redistribution_ok=True,
        source_work_nsfw=False,
        content_tags=["fanservice", "dark_themes"],
        safety_tier="tier_pg13",
        gt_known_defects=["color_shift"],
    )
    entry.add_observation(
        SafetyObservation(source=SOURCE_HUMAN, verdict="clear", evidence="reviewed 2026-08-15")
    )
    restored = CaseProvenance.from_dict(json.loads(json.dumps(entry.to_dict())))
    assert restored.corpus_id == "sfw_q3"
    assert restored.web_redistribution_ok is True
    assert restored.source_work_nsfw is False
    assert set(restored.content_tags) == {"fanservice", "dark_themes"}
    assert restored.safety_tier == "tier_pg13"
    assert restored.gt_known_defects == ["color_shift"]
    assert len(restored.safety_observations) == 1
    assert restored.safety_observations[0].source == SOURCE_HUMAN
    assert restored.safety_observations[0].verdict == "clear"


def test_source_work_nsfw_three_state_distinguishes_unknown_from_false():
    unknown = CaseProvenance(case_id="a")
    known_false = CaseProvenance(case_id="b", source_work_nsfw=False)
    assert unknown.source_work_nsfw is None
    assert known_false.source_work_nsfw is False
    assert unknown.source_work_nsfw is not known_false.source_work_nsfw


def test_invalid_safety_tier_rejected():
    with pytest.raises(ValueError):
        CaseProvenance(case_id="x", safety_tier="not_a_real_tier")


def test_invalid_observation_verdict_rejected():
    with pytest.raises(ValueError):
        SafetyObservation(source=SOURCE_HUMAN, verdict="definitely_fine")


class TestDualVetoGate:
    """The actual safety mechanism: OR-logic to exclude, AND-logic (human
    clear specifically) to include. See CaseProvenance.minor_presenting_*."""

    def test_no_observations_is_not_includable(self):
        entry = CaseProvenance(case_id="x")
        assert entry.minor_presenting_high_risk() is False
        assert entry.minor_presenting_includable() is False

    def test_human_clear_alone_is_includable(self):
        entry = CaseProvenance(case_id="x")
        entry.add_observation(SafetyObservation(source=SOURCE_HUMAN, verdict="clear"))
        assert entry.minor_presenting_includable() is True

    def test_automated_clear_alone_cannot_include(self):
        """Per C0.5's decision rule: an automated ensemble member can veto
        or abstain, but cannot clear a case on its own -- only a human
        `clear` observation satisfies the inclusion AND-logic."""
        entry = CaseProvenance(case_id="x")
        entry.add_observation(SafetyObservation(source="wd14", verdict="clear"))
        assert entry.minor_presenting_includable() is False

    def test_automated_high_risk_vetoes_despite_human_clear(self):
        """OR-logic for exclusion: either assessor's high_risk is a hard
        drop, even against a human clear -- this is the actual safety
        property, not a suggestion."""
        entry = CaseProvenance(case_id="x")
        entry.add_observation(SafetyObservation(source=SOURCE_HUMAN, verdict="clear"))
        entry.add_observation(SafetyObservation(source="commercial_endpoint", verdict="high_risk"))
        assert entry.minor_presenting_high_risk() is True
        assert entry.minor_presenting_includable() is False

    def test_uncertain_is_never_a_veto_and_never_clears(self):
        entry = CaseProvenance(case_id="x")
        entry.add_observation(SafetyObservation(source=SOURCE_HUMAN, verdict="uncertain"))
        entry.add_observation(SafetyObservation(source="clip_dinov2", verdict="uncertain"))
        assert entry.minor_presenting_high_risk() is False
        assert entry.minor_presenting_includable() is False  # no human `clear` yet

    def test_adjudication_is_appended_not_a_replacement(self):
        entry = CaseProvenance(case_id="x")
        entry.add_observation(SafetyObservation(source=SOURCE_HUMAN, verdict="uncertain"))
        entry.add_adjudication(
            SafetyAdjudication(decision="clear", reason="manual review resolved ambiguity")
        )
        # the original observation is untouched -- adjudication is a
        # separate record, never a rewrite
        assert len(entry.safety_observations) == 1
        assert entry.safety_observations[0].verdict == "uncertain"

    def test_controlled_one_sided_acceptance_with_reasoned_adjudication_includes(self):
        """The path C0.5's decision rule calls out explicitly: one assessor
        merely `uncertain` (not `clear`), but a reasoned adjudication with
        real supporting provenance (e.g. a PEGI-3 source rating) clears the
        case -- this is the gap Chat/Codex's review caught: adjudications
        existed as a data structure but were never consulted by the
        inclusion decision."""
        entry = CaseProvenance(case_id="x")
        entry.add_observation(SafetyObservation(source=SOURCE_HUMAN, verdict="uncertain"))
        entry.add_adjudication(
            SafetyAdjudication(
                decision="clear",
                reason="PEGI-3 official source rating on file, see provenance link",
                adjudicated_by="harbinger",
            )
        )
        assert entry.minor_presenting_includable() is True

    def test_one_sided_acceptance_requires_a_real_reason_not_a_bare_decision(self):
        """An adjudication with no reason is indistinguishable from a rubber
        stamp -- must not satisfy inclusion on its own."""
        entry = CaseProvenance(case_id="x")
        entry.add_observation(SafetyObservation(source=SOURCE_HUMAN, verdict="uncertain"))
        entry.add_adjudication(SafetyAdjudication(decision="clear", reason=""))
        assert entry.minor_presenting_includable() is False
        entry.add_adjudication(SafetyAdjudication(decision="clear", reason="   "))
        assert entry.minor_presenting_includable() is False  # whitespace-only doesn't count either

    def test_high_risk_still_vetoes_despite_a_one_sided_acceptance_adjudication(self):
        """The hard veto is checked before the one-sided-acceptance path --
        an adjudication cannot launder around a high_risk finding."""
        entry = CaseProvenance(case_id="x")
        entry.add_observation(SafetyObservation(source="commercial_endpoint", verdict="high_risk"))
        entry.add_adjudication(
            SafetyAdjudication(decision="clear", reason="reviewed and disagree with the flag")
        )
        assert entry.minor_presenting_includable() is False
        assert len(entry.safety_adjudications) == 1
        assert entry.safety_adjudications[0].decision == "clear"


def test_load_save_round_trip(tmp_path):
    path = str(tmp_path / "provenance.json")
    entries = {
        "asp_test04": CaseProvenance(case_id="asp_test04", corpus_id="nsfw_97"),
        "asp_test50": CaseProvenance(
            case_id="asp_test50", corpus_id="sfw_q3", safety_tier="tier_g"
        ),
    }
    save_provenance(path, entries)
    loaded = load_provenance(path)
    assert set(loaded.keys()) == {"asp_test04", "asp_test50"}
    assert loaded["asp_test50"].safety_tier == "tier_g"


def test_load_missing_file_returns_empty_dict(tmp_path):
    assert load_provenance(str(tmp_path / "does_not_exist.json")) == {}
