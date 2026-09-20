"""Plate-frame coverage selector (#433).

Which non-hero frames feed the wallpaper background plate is a coverage
problem, distinct from hero *selection* (a scoring function).  This module
is the unwired prototype: greedy set-cover / facility-location over
canvas-space coverage masks, with a blend-quality redundancy floor.

``ASP_PLATE_FRAME_COVER`` defaults **off**.  Nothing in
:mod:`wallpaper_pipeline` or :mod:`_plate_builder` calls this; Ground Rules
still require a benchmark before it is ever enabled.  Future call sites
should use :func:`maybe_select_plate_frames` so the default remains a
pass-through of every non-hero index.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Fast / Balanced / Max → (min_samples, max_frames).  ``max`` matches the
# plate builder's ``_MAX_TEMPORAL_SAMPLES`` cap so a future wire-up does not
# grow the median stack past what Slice 1 already budgets.
_QUALITY_PRESETS: dict[str, tuple[int, int]] = {
    "fast": (1, 8),
    "balanced": (2, 16),
    "max": (2, 40),
}

_DEFAULT_CELL_SIZE = 16


@dataclass(frozen=True)
class PlateFrameSelection:
    """Result of :func:`select_plate_frames`.

    ``selected`` is in greedy-pick order (highest marginal gain first),
    except for the flag-off pass-through which keeps original index order.
    """

    selected: tuple[int, ...]
    covered_fraction: float
    redundancy_floor_fraction: float
    n_candidates: int
    n_universe_cells: int
    min_samples: int
    max_frames: int | None
    quality: str


def plate_frame_cover_enabled() -> bool:
    """#433 A/B: ``ASP_PLATE_FRAME_COVER=1`` enables greedy cover.

    Default ``0`` — the prototype must not change plate inputs until a
    Harbinger-authorized benchmark says keep.
    """
    return os.environ.get("ASP_PLATE_FRAME_COVER", "0") == "1"


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"coverage mask must be 2-D, got shape {mask.shape}")
    if mask.dtype == np.bool_:
        return mask
    if mask.dtype == np.uint8:
        return mask > 127
    return mask.astype(bool)


def _or_pool(mask: np.ndarray, cell_size: int) -> np.ndarray:
    """OR-pool a 2-D bool mask into cells.  Remainder is padded False."""
    if cell_size == 1:
        return mask
    h, w = mask.shape
    ch = (h + cell_size - 1) // cell_size
    cw = (w + cell_size - 1) // cell_size
    pad_h = ch * cell_size - h
    pad_w = cw * cell_size - w
    if pad_h or pad_w:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=False)
    return mask.reshape(ch, cell_size, cw, cell_size).any(axis=(1, 3))


def _resolve_budget(
    quality: str,
    min_samples: int | None,
    max_frames: int | None,
) -> tuple[int, int | None]:
    if quality not in _QUALITY_PRESETS:
        raise ValueError(
            f"unsupported quality {quality!r}; expected one of "
            f"{sorted(_QUALITY_PRESETS)}"
        )
    preset_min, preset_max = _QUALITY_PRESETS[quality]
    resolved_min = preset_min if min_samples is None else int(min_samples)
    resolved_max = preset_max if max_frames is None else max_frames
    if resolved_min < 1:
        raise ValueError(f"min_samples must be >= 1, got {resolved_min}")
    if resolved_max is not None and resolved_max < 1:
        raise ValueError(f"max_frames must be >= 1 or None, got {resolved_max}")
    return resolved_min, resolved_max


def _eligible_indices(
    coverage: list[np.ndarray],
    hero_idx: int | None,
) -> list[int]:
    n = len(coverage)
    if hero_idx is not None and not 0 <= hero_idx < n:
        raise ValueError(f"hero_idx {hero_idx} out of range for {n} masks")
    return [i for i in range(n) if i != hero_idx]


def _flatten_cells(
    coverage: list[np.ndarray],
    eligible: Sequence[int],
    cell_size: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    cells = [_or_pool(coverage[i], cell_size).ravel() for i in eligible]
    if not cells:
        empty = np.zeros(0, dtype=bool)
        return [], empty
    universe = np.zeros(cells[0].shape[0], dtype=bool)
    for c in cells:
        universe |= c
    return cells, universe


def _fractions(
    count: np.ndarray,
    universe: np.ndarray,
    min_samples: int,
) -> tuple[float, float]:
    n_u = int(universe.sum())
    if n_u == 0:
        return 1.0, 1.0
    covered = int(np.count_nonzero((count > 0) & universe))
    floor = int(np.count_nonzero((count >= min_samples) & universe))
    return covered / n_u, floor / n_u


def select_plate_frames(
    coverage_masks: Sequence[np.ndarray],
    *,
    hero_idx: int | None = None,
    min_samples: int | None = None,
    max_frames: int | None = None,
    quality: str = "balanced",
    cell_size: int = _DEFAULT_CELL_SIZE,
) -> PlateFrameSelection:
    """Greedy set-cover with a facility-location redundancy floor.

    Each mask is canvas-space coverage (bool or uint8, True / >127 =
    this frame may contribute a background sample at that pixel).  The
    hero index is never selected.  At each step the remaining frame with
    the most still-under-covered cells is opened; ties take the lower
    index.  Stops when every universe cell has ``min_samples`` hits, no
    remaining frame has positive gain, or the budget is exhausted.

    Parameters
    ----------
    coverage_masks:
        Per-frame 2-D coverage, all the same shape.  Empty input is
        rejected.
    hero_idx:
        Index excluded from the candidate set (the hero cel's own frame).
    min_samples:
        Blend-quality floor.  ``None`` uses the quality preset.
    max_frames:
        Budget cap.  ``None`` uses the quality preset.
    quality:
        ``fast`` / ``balanced`` / ``max``.
    cell_size:
        OR-pool window for tractability.  ``1`` keeps exact pixels.
    """
    if not coverage_masks:
        raise ValueError("coverage_masks must be non-empty")
    if cell_size < 1:
        raise ValueError(f"cell_size must be >= 1, got {cell_size}")

    coverage = [_as_bool_mask(m) for m in coverage_masks]
    shape0 = coverage[0].shape
    for i, m in enumerate(coverage):
        if m.shape != shape0:
            raise ValueError(f"coverage mask {i} shape {m.shape} != {shape0}")

    resolved_min, resolved_max = _resolve_budget(quality, min_samples, max_frames)
    eligible = _eligible_indices(coverage, hero_idx)
    cells, universe = _flatten_cells(coverage, eligible, cell_size)

    n_universe = int(universe.sum())
    if not eligible or n_universe == 0:
        return PlateFrameSelection(
            selected=(),
            covered_fraction=1.0,
            redundancy_floor_fraction=1.0,
            n_candidates=len(eligible),
            n_universe_cells=n_universe,
            min_samples=resolved_min,
            max_frames=resolved_max,
            quality=quality,
        )

    count = np.zeros(universe.shape[0], dtype=np.int32)
    remaining = list(range(len(eligible)))
    picked: list[int] = []

    while remaining:
        if resolved_max is not None and len(picked) >= resolved_max:
            break
        need = universe & (count < resolved_min)
        if not need.any():
            break
        best_j = -1
        best_gain = 0
        for j in remaining:
            gain = int(np.count_nonzero(cells[j] & need))
            if gain > best_gain:
                best_gain = gain
                best_j = j
        if best_j < 0 or best_gain == 0:
            break
        picked.append(best_j)
        remaining.remove(best_j)
        count += cells[best_j].astype(np.int32)

    covered_frac, floor_frac = _fractions(count, universe, resolved_min)
    return PlateFrameSelection(
        selected=tuple(eligible[j] for j in picked),
        covered_fraction=covered_frac,
        redundancy_floor_fraction=floor_frac,
        n_candidates=len(eligible),
        n_universe_cells=n_universe,
        min_samples=resolved_min,
        max_frames=resolved_max,
        quality=quality,
    )


def maybe_select_plate_frames(
    coverage_masks: Sequence[np.ndarray],
    *,
    hero_idx: int | None = None,
    min_samples: int | None = None,
    max_frames: int | None = None,
    quality: str = "balanced",
    cell_size: int = _DEFAULT_CELL_SIZE,
) -> PlateFrameSelection:
    """Pass-through of every non-hero index unless the flag is on.

    Safe future call site: default-off means the plate still sees the
    full candidate list.  Stats are computed from that full set so a
    dry-run dump can still report coverage.
    """
    if plate_frame_cover_enabled():
        return select_plate_frames(
            coverage_masks,
            hero_idx=hero_idx,
            min_samples=min_samples,
            max_frames=max_frames,
            quality=quality,
            cell_size=cell_size,
        )

    if not coverage_masks:
        raise ValueError("coverage_masks must be non-empty")
    coverage = [_as_bool_mask(m) for m in coverage_masks]
    resolved_min, resolved_max = _resolve_budget(quality, min_samples, max_frames)
    eligible = _eligible_indices(coverage, hero_idx)
    cells, universe = _flatten_cells(coverage, eligible, cell_size)
    count = np.zeros(universe.shape[0], dtype=np.int32)
    for c in cells:
        count += c.astype(np.int32)
    covered_frac, floor_frac = _fractions(count, universe, resolved_min)
    return PlateFrameSelection(
        selected=tuple(eligible),
        covered_fraction=covered_frac,
        redundancy_floor_fraction=floor_frac,
        n_candidates=len(eligible),
        n_universe_cells=int(universe.sum()),
        min_samples=resolved_min,
        max_frames=resolved_max,
        quality=quality,
    )


__all__ = [
    "PlateFrameSelection",
    "maybe_select_plate_frames",
    "plate_frame_cover_enabled",
    "select_plate_frames",
]
