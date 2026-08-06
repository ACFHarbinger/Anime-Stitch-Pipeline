"""
Tests for the evaluation session's queue/history/persistence semantics.

Every test here pins a specific bug the old dashboard had (issue #123), because
those bugs were all in navigation and all invisible to a unit test that only
checked "does the window build":

  defect 2 — Skip didn't push history, so a skipped test was unreachable.
  defect 4 — `--redo` filtered candidates by dict membership, so it stopped
             after one test (in redo mode every name is already in the file).
  defect 5 — a merely-visited test was persisted with null scores and then
             excluded from the next session's queue forever.

``EvaluationSession`` has no Qt import precisely so this can be asserted
directly rather than through a window.
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

from asp_backend.evaluation.other.schema import (  # noqa: E402
    BoundingBox,
    RatingEntry,
    save_evaluations,
)
from asp_backend.evaluation.other.session import EvaluationSession  # noqa: E402

NAMES = [f"asp_test{i:02d}" for i in range(1, 7)]


@pytest.fixture()
def out_path(tmp_path):
    return str(tmp_path / "asp_evaluations_20260101.json")


def _rate(session, name, asp=3, simple=2):
    entry = session.entry(name)
    entry.set_score("asp", "coherence", asp)
    entry.set_score("simple", "coherence", simple)
    entry.reviewed = True
    session.commit(name)


def _rated_file(out_path, names=NAMES):
    evaluations = {}
    for name in names:
        entry = RatingEntry()
        entry.set_score("asp", "coherence", 3)
        entry.set_score("simple", "coherence", 2)
        entry.reviewed = True
        evaluations[name] = entry
    save_evaluations(out_path, evaluations)


# ---------------------------------------------------------------------------
# defect 2 — Back must work after any kind of move
# ---------------------------------------------------------------------------


def test_back_works_after_skip(out_path):
    session = EvaluationSession(NAMES, out_path)
    assert session.current == "asp_test01"
    session.advance(skip=True)
    assert session.current == "asp_test02"
    assert session.can_go_back() is True
    assert session.go_back() == "asp_test01"


def test_back_unwinds_the_full_history(out_path):
    session = EvaluationSession(NAMES, out_path, redo=True)
    for name in ("asp_test03", "asp_test06", "asp_test02"):
        session.go_to(name)
    assert [session.go_back() for _ in range(3)] == ["asp_test06", "asp_test03", "asp_test01"]
    assert session.can_go_back() is False
    assert session.go_back() is None


def test_back_at_the_start_is_a_no_op(out_path):
    session = EvaluationSession(NAMES, out_path)
    assert session.can_go_back() is False
    assert session.go_back() is None
    assert session.current == "asp_test01"


# ---------------------------------------------------------------------------
# defect 4 — redo mode must walk the whole corpus
# ---------------------------------------------------------------------------


def test_redo_advances_past_the_first_test(out_path):
    _rated_file(out_path)
    session = EvaluationSession(NAMES, out_path, redo=True)
    walked = [session.current]
    while True:
        nxt = session.advance(skip=False)
        if nxt is None:
            break
        walked.append(nxt)
    assert walked == NAMES


def test_normal_mode_starts_at_the_first_unrated(out_path):
    _rated_file(out_path, NAMES[:3])
    session = EvaluationSession(NAMES, out_path)
    assert session.current == "asp_test04"


def test_normal_mode_ends_when_everything_is_rated(out_path):
    _rated_file(out_path)
    session = EvaluationSession(NAMES, out_path)
    assert session.advance(skip=False) is None


def test_next_unrated_wraps_around(out_path):
    _rated_file(out_path, NAMES[1:])  # only asp_test01 unrated
    session = EvaluationSession(NAMES, out_path, redo=True)
    session.go_to("asp_test04")
    assert session.next_unrated() == "asp_test01"


# ---------------------------------------------------------------------------
# defect 5 — visiting a test must not record anything
# ---------------------------------------------------------------------------


def test_visiting_tests_writes_nothing(out_path):
    session = EvaluationSession(NAMES, out_path)
    session.go_to("asp_test03")
    session.go_to("asp_test05")
    session.save()
    with open(out_path) as f:
        assert not os.path.exists(out_path) or json.loads(f.read()) == {}


def test_a_visited_test_is_still_queued_next_session(out_path):
    first = EvaluationSession(NAMES, out_path)
    first.go_to("asp_test02")
    first.go_to("asp_test03")
    first.save()
    second = EvaluationSession(NAMES, out_path)
    assert second.current == "asp_test01"
    assert second.is_rated("asp_test02") is False


@pytest.mark.parametrize("mutate,expected_key", [
    (lambda e: e.set_score("asp", "coherence", 2), "asp"),
    (lambda e: setattr(e, "notes", "a note"), "notes"),
    (lambda e: setattr(e, "defects", ["banding"]), "defects"),
    (lambda e: setattr(e, "preference", "simple"), "preference"),
    (lambda e: setattr(e, "skipped", True), "skipped"),
    (lambda e: e.bboxes.append(BoundingBox(image="asp", x=0, y=0, w=0.1, h=0.1)), "bboxes"),
])
def test_any_real_input_is_persisted(out_path, mutate, expected_key):
    """The flip side of the above: an entry with *anything* in it is user input
    and must survive, or the tool loses work."""
    session = EvaluationSession(NAMES, out_path)
    mutate(session.entry("asp_test01"))
    session.commit("asp_test01")
    with open(out_path) as f:
        doc = json.loads(f.read())

    assert "asp_test01" in doc
    assert doc["asp_test01"].get(expected_key) not in (None, "", [], False)


# ---------------------------------------------------------------------------
# Skip / accept semantics
# ---------------------------------------------------------------------------


def test_skip_marks_deferred_but_keeps_it_findable(out_path):
    session = EvaluationSession(NAMES, out_path)
    session.advance(skip=True)
    assert session.is_skipped("asp_test01") is True
    assert session.is_rated("asp_test01") is False
    # "not now" must not mean "never": a later pass still offers it.
    assert EvaluationSession(NAMES, out_path).current == "asp_test01"


def test_skip_goes_to_the_next_test_not_the_next_unrated(out_path):
    _rated_file(out_path, ["asp_test02"])
    session = EvaluationSession(NAMES, out_path, redo=True)
    session.advance(skip=True)
    assert session.current == "asp_test02"


def test_accept_clears_a_previous_skip(out_path):
    session = EvaluationSession(NAMES, out_path)
    session.advance(skip=True)
    session.go_back()
    _rate(session, "asp_test01")
    session.accept()
    session.save()
    with open(out_path) as f:
        entry = json.loads(f.read())["asp_test01"]

    assert entry["asp"] == 3 and entry.get("skipped") is not True
    assert entry.get("reviewed") is True


def test_accept_advances_to_the_next_unrated(out_path):
    session = EvaluationSession(NAMES, out_path)
    _rate(session, "asp_test01")
    assert session.accept() == "asp_test02"


# ---------------------------------------------------------------------------
# Progress and free navigation
# ---------------------------------------------------------------------------


def test_progress_counts_rated_and_skipped_separately(out_path):
    session = EvaluationSession(NAMES, out_path)
    _rate(session, "asp_test01")
    session.go_to("asp_test02")
    session.advance(skip=True)
    progress = session.progress()
    assert progress.total == len(NAMES)
    assert progress.rated == 1
    assert progress.skipped == 1


def test_every_test_is_reachable_regardless_of_state(out_path):
    _rated_file(out_path, NAMES[:4])
    session = EvaluationSession(NAMES, out_path)
    for name in NAMES:
        assert session.go_to(name) == name


def test_go_to_an_unknown_name_is_ignored(out_path):
    session = EvaluationSession(NAMES, out_path)
    assert session.go_to("not_a_test") == "asp_test01"


def test_empty_corpus_does_not_crash(out_path):
    session = EvaluationSession([], out_path)
    assert session.current is None
    assert session.advance() is None
    assert session.progress().total == 0


def test_autosave_writes_on_every_commit(out_path):
    session = EvaluationSession(NAMES, out_path, autosave=True)
    _rate(session, "asp_test01")
    with open(out_path) as f:
        assert json.loads(f.read())["asp_test01"]["asp"] == 3


def test_autosave_off_defers_until_save(out_path):
    session = EvaluationSession(NAMES, out_path, autosave=False)
    _rate(session, "asp_test01")
    assert not os.path.exists(out_path)
    session.save()
    with open(out_path) as f:
        assert "asp_test01" in json.loads(f.read())
