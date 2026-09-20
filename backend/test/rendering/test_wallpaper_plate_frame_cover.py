"""Tests for the unwired plate-frame coverage selector (#433).

Synthetic coverage maps only — no GPU, no fixtures, no pipeline run.
Loaded from the module file so this does not pull wallpaper/__init__.py
(and the Image-Toolkit ``backend.src.constants`` import chain).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_WALLPAPER = Path(__file__).resolve().parents[2] / "src" / "rendering" / "wallpaper"
_MOD_PATH = _WALLPAPER / "_plate_frame_cover.py"


def _load_cover():
    spec = importlib.util.spec_from_file_location("_plate_frame_cover_under_test", _MOD_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cover = _load_cover()
select_plate_frames = cover.select_plate_frames
maybe_select_plate_frames = cover.maybe_select_plate_frames
plate_frame_cover_enabled = cover.plate_frame_cover_enabled
_QUALITY_PRESETS = cover._QUALITY_PRESETS


def _strip(h: int, w: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[:, x0:x1] = True
    return m


def test_disjoint_strips_require_both_frames() -> None:
    left = _strip(4, 8, 0, 4)
    right = _strip(4, 8, 4, 8)
    result = select_plate_frames(
        [left, right], min_samples=1, max_frames=8, cell_size=1
    )
    assert set(result.selected) == {0, 1}
    assert result.covered_fraction == pytest.approx(1.0)
    assert result.redundancy_floor_fraction == pytest.approx(1.0)
    assert result.n_universe_cells == 32
    assert result.n_candidates == 2


def test_redundant_duplicate_is_skipped() -> None:
    cover = _strip(4, 8, 0, 8)
    result = select_plate_frames(
        [cover, cover.copy(), cover.copy()],
        min_samples=1,
        max_frames=8,
        cell_size=1,
    )
    assert result.selected == (0,)
    assert result.covered_fraction == pytest.approx(1.0)


def test_tie_breaks_to_lower_index() -> None:
    a = _strip(2, 4, 0, 2)
    b = _strip(2, 4, 0, 2)
    result = select_plate_frames([a, b], min_samples=1, max_frames=1, cell_size=1)
    assert result.selected == (0,)


def test_hero_index_is_never_selected() -> None:
    left = _strip(4, 8, 0, 4)
    hero = _strip(4, 8, 2, 6)
    right = _strip(4, 8, 4, 8)
    result = select_plate_frames(
        [left, hero, right],
        hero_idx=1,
        min_samples=1,
        max_frames=8,
        cell_size=1,
    )
    assert 1 not in result.selected
    assert set(result.selected) == {0, 2}
    assert result.n_candidates == 2


def test_redundancy_floor_picks_overlap_not_just_cover() -> None:
    # Frame 0 covers the whole row; 1 and 2 each cover a half.  min_samples=2
    # needs a second hit on every cell, so 0 plus both halves.
    full = _strip(1, 4, 0, 4)
    left = _strip(1, 4, 0, 2)
    right = _strip(1, 4, 2, 4)
    result = select_plate_frames(
        [full, left, right], min_samples=2, max_frames=8, cell_size=1
    )
    assert set(result.selected) == {0, 1, 2}
    assert result.redundancy_floor_fraction == pytest.approx(1.0)

    one = select_plate_frames(
        [full, left, right], min_samples=1, max_frames=8, cell_size=1
    )
    assert one.selected == (0,)


def test_budget_cap_stops_before_full_cover() -> None:
    masks = [_strip(2, 6, x, x + 2) for x in (0, 2, 4)]
    result = select_plate_frames(
        masks, min_samples=1, max_frames=2, cell_size=1
    )
    assert len(result.selected) == 2
    assert result.covered_fraction == pytest.approx(2.0 / 3.0)
    assert result.redundancy_floor_fraction == pytest.approx(2.0 / 3.0)


def test_greedy_recovers_off_grid_unique_frame_even_decimation_misses() -> None:
    # 40-frame pan: frames 0..39 cover the left half; only frame 7 covers
    # the right half.  Even linspace(0, 39, 8) is {0,5,11,16,22,27,33,39}
    # and drops 7.  Greedy cover with budget 8 must keep 7.
    h, w = 4, 8
    left = _strip(h, w, 0, 4)
    right = _strip(h, w, 4, 8)
    masks = [left.copy() for _ in range(40)]
    masks[7] = right
    even = set(np.unique(np.linspace(0, 39, 8).astype(np.int64)).tolist())
    assert 7 not in even

    result = select_plate_frames(
        masks, min_samples=1, max_frames=8, cell_size=1
    )
    assert 7 in result.selected
    assert result.covered_fraction == pytest.approx(1.0)
    assert result.selected[0] in even  # first pick is the common left cover
    assert result.selected[0] != 7


def test_quality_presets_supply_budget_and_floor() -> None:
    cover = _strip(2, 4, 0, 4)
    fast = select_plate_frames([cover], quality="fast", cell_size=1)
    assert fast.min_samples == _QUALITY_PRESETS["fast"][0]
    assert fast.max_frames == _QUALITY_PRESETS["fast"][1]
    bal = select_plate_frames([cover], quality="balanced", cell_size=1)
    assert (bal.min_samples, bal.max_frames) == _QUALITY_PRESETS["balanced"]
    mx = select_plate_frames([cover], quality="max", cell_size=1)
    assert (mx.min_samples, mx.max_frames) == _QUALITY_PRESETS["max"]


def test_explicit_budget_overrides_quality_preset() -> None:
    cover = _strip(2, 4, 0, 4)
    result = select_plate_frames(
        [cover], quality="fast", min_samples=3, max_frames=4, cell_size=1
    )
    assert result.min_samples == 3
    assert result.max_frames == 4


def test_empty_universe_is_vacuously_covered() -> None:
    empty = np.zeros((4, 4), dtype=bool)
    result = select_plate_frames([empty, empty], min_samples=1, max_frames=4, cell_size=1)
    assert result.selected == ()
    assert result.covered_fraction == 1.0
    assert result.n_universe_cells == 0


def test_hero_only_sequence_selects_nothing() -> None:
    cover = _strip(2, 2, 0, 2)
    result = select_plate_frames([cover], hero_idx=0, min_samples=1, max_frames=4, cell_size=1)
    assert result.selected == ()
    assert result.n_candidates == 0


def test_uint8_masks_use_pipeline_threshold() -> None:
    low = np.full((2, 4), 127, dtype=np.uint8)
    high = np.full((2, 4), 128, dtype=np.uint8)
    result = select_plate_frames([low, high], min_samples=1, max_frames=4, cell_size=1)
    assert result.selected == (1,)
    assert result.n_universe_cells == 8


def test_or_pool_merges_partial_cells() -> None:
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    mask[2, 2] = True
    result = select_plate_frames([mask], min_samples=1, max_frames=4, cell_size=2)
    # 3x3 padded to 4x4 → 2x2 cells, two of them occupied.
    assert result.n_universe_cells == 2
    assert result.selected == (0,)


def test_api_validates_inputs() -> None:
    cover = _strip(2, 2, 0, 2)
    with pytest.raises(ValueError, match="non-empty"):
        select_plate_frames([])
    with pytest.raises(ValueError, match="2-D"):
        select_plate_frames([np.zeros(4, dtype=bool)])
    with pytest.raises(ValueError, match="shape"):
        select_plate_frames([cover, np.zeros((3, 3), dtype=bool)])
    with pytest.raises(ValueError, match="hero_idx"):
        select_plate_frames([cover], hero_idx=3)
    with pytest.raises(ValueError, match="unsupported quality"):
        select_plate_frames([cover], quality="turbo")
    with pytest.raises(ValueError, match="cell_size"):
        select_plate_frames([cover], cell_size=0)
    with pytest.raises(ValueError, match="min_samples"):
        select_plate_frames([cover], min_samples=0)


def test_flag_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASP_PLATE_FRAME_COVER", raising=False)
    assert plate_frame_cover_enabled() is False
    monkeypatch.setenv("ASP_PLATE_FRAME_COVER", "0")
    assert plate_frame_cover_enabled() is False
    monkeypatch.setenv("ASP_PLATE_FRAME_COVER", "1")
    assert plate_frame_cover_enabled() is True


def test_maybe_select_is_passthrough_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ASP_PLATE_FRAME_COVER", raising=False)
    left = _strip(2, 4, 0, 2)
    right = _strip(2, 4, 2, 4)
    dup = left.copy()
    off = maybe_select_plate_frames(
        [left, dup, right], hero_idx=1, min_samples=1, max_frames=1, cell_size=1
    )
    # Flag off: every non-hero index, even though greedy-with-budget-1
    # would have kept a single frame.
    assert off.selected == (0, 2)
    assert off.covered_fraction == pytest.approx(1.0)

    monkeypatch.setenv("ASP_PLATE_FRAME_COVER", "1")
    on = maybe_select_plate_frames(
        [left, dup, right], hero_idx=1, min_samples=1, max_frames=1, cell_size=1
    )
    assert on.selected == (0,)
    assert on.covered_fraction == pytest.approx(0.5)


def test_pipeline_and_plate_builder_do_not_call_selector() -> None:
    for name in ("wallpaper_pipeline.py", "_plate_builder.py"):
        src = (_WALLPAPER / name).read_text()
        assert "select_plate_frames" not in src
        assert "maybe_select_plate_frames" not in src
        assert "plate_frame_cover" not in src
        assert "ASP_PLATE_FRAME_COVER" not in src
