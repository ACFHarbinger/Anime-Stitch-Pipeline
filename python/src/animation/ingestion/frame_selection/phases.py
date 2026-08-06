"""Animation-phase clustering (§2.2 — measurement-only, no compositing effect)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ._hold_detection import _compute_dhash
from ._thumbs import _load_thumbs_parallel


def detect_animation_phases(
    frames_paths: List[str],
    z_thresh: float = 2.0,
) -> List[int]:
    """§2.2: cluster the *selected* frames into animation phases.

    A "phase" is coarser than a hold block: it groups a run of selected
    frames that continue the same pose/action, and marks a new phase where
    the character snaps to a substantially different configuration (a new
    animation loop, à la Overmix's ``AnimationSeparator``) rather than
    continuing the gradual per-frame drift that camera-step selection
    already expects between consecutive *selected* frames.

    Starting implementation (roadmap §2.2): pairwise dHash Hamming distance
    between consecutive selected frames, with a robust change-point rule —
    a boundary is declared where the distance is a statistical outlier
    relative to the rest of the sequence (median + ``z_thresh`` robust
    sigma, via MAD), rather than a fixed threshold. This is intentionally
    the same primitive as hold detection (§3.4A) applied one level up: hold
    detection groups near-identical consecutive frames into one cel, phase
    detection groups a run of *already-distinct* selected frames that still
    belong to the same pose/action.

    Measurement-only for now — the returned ``phase_ids`` are not yet
    consumed by compositing (Phase 2.3 wires that up). Callers use this to
    record phase-count/phase-span diagnostics and are free to ignore it.

    Parameters
    ----------
    frames_paths : the *selected* frame paths (post ``smart_select_frames``),
        not the raw ingested frame set.
    z_thresh : robust z-score multiplier for the change-point rule. Higher
        = fewer, larger phases. Default 2.0.

    Returns
    -------
    List[int] — phase id per frame, 0-indexed, monotonically non-decreasing.
    """
    N = len(frames_paths)
    if N <= 2:
        return list(range(N))

    thumbs = _load_thumbs_parallel(frames_paths)
    hashes = [_compute_dhash(t) for t in thumbs]
    return _phase_ids_from_hashes(hashes, z_thresh=z_thresh)


def _phase_ids_from_hashes(
    hashes: List[np.ndarray], z_thresh: float = 2.0
) -> List[int]:
    """Core §2.2 change-point clustering, factored out so callers that already
    have dHashes in hand (e.g. Pass 2 of ``smart_select_frames``, §2.4) don't
    pay for a redundant thumbnail reload. See ``detect_animation_phases`` for
    the algorithm description.
    """
    N = len(hashes)
    if N <= 2:
        return list(range(N))

    dists = [int(np.sum(hashes[i] != hashes[i - 1])) for i in range(1, N)]

    med = float(np.median(dists))
    abs_dev = np.abs(np.array(dists, dtype=np.float64) - med)
    mad = float(np.median(abs_dev))
    # 1.4826 = MAD→σ consistency constant for a normal distribution.
    robust_sigma = 1.4826 * mad if mad > 0.0 else 1.0
    thresh = med + z_thresh * robust_sigma

    phase_ids = [0] * N
    pid = 0
    for i, d in enumerate(dists):
        if d > thresh:
            pid += 1
        phase_ids[i + 1] = pid
    return phase_ids


def phase_spans(phase_ids: List[int]) -> List[Tuple[int, int, int]]:
    """§2.2 diagnostic helper: collapse ``phase_ids`` into ``(phase, start, end)``
    spans (inclusive frame indices) for reporting/visualization.
    """
    if not phase_ids:
        return []
    spans: List[Tuple[int, int, int]] = []
    start = 0
    cur = phase_ids[0]
    for i, pid in enumerate(phase_ids[1:], start=1):
        if pid != cur:
            spans.append((cur, start, i - 1))
            start = i
            cur = pid
    spans.append((cur, start, len(phase_ids) - 1))
    return spans


__all__ = ["detect_animation_phases", "_phase_ids_from_hashes", "phase_spans"]
