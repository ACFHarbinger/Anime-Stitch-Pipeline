"""Animation "on twos/threes" hold-block detection (§1.11, §3.4A, §1.11C, §1.64).

Anime animators draw a new character cel every 2-3 video frames.  Within a
hold block, consecutive frames are pixel-identical except for MPEG
compression noise and sub-pixel camera drift.  These detectors find the
block boundaries so ``smart_select_frames`` can skip redundant phase
correlation within a block and Pass 2 can prefer cross-block (different
pose) candidates.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from ._native import _BATCH_FSEL, _batch

# Animation hold detection — FD-Means preprocessing (§1.11 / §3.4).
# Default 0.025 corresponds to 2.5% mean absolute difference between
# consecutive thumbnails.  Within-hold frames typically score 0.003–0.010;
# cross-hold frames score 0.030–0.120.  Set ASP_HOLD_THRESHOLD=0 to disable.
try:
    _HOLD_THRESHOLD = float(os.environ.get("ASP_HOLD_THRESHOLD", "0.025"))
except ValueError:
    _HOLD_THRESHOLD = 0.025

# §1.11C: Post-hoc hold refinement using phase-correlation response.
# If phaseCorrelate returns response >= this threshold, the two frames are
# near-identical (same character cel; MAD-based detection missed them due to
# MPEG noise), so merge their hold blocks.  Default 0.85.
# Set ASP_HIGH_HOLD_RESPONSE=0.0 to disable.
try:
    _HIGH_HOLD_RESPONSE = float(os.environ.get("ASP_HIGH_HOLD_RESPONSE", "0.85"))
except ValueError:
    _HIGH_HOLD_RESPONSE = 0.85

# §3.4A: dHash hold detection — integer Hamming-distance threshold.
# 0 = disabled (use MAD-based detector).  Typical same-cel distance: 0–2;
# cross-cel: 5–20.  Enable with ASP_HOLD_DHASH_THRESH=4.
try:
    _HOLD_DHASH_THRESHOLD = int(os.environ.get("ASP_HOLD_DHASH_THRESH", "0"))
except ValueError:
    _HOLD_DHASH_THRESHOLD = 0

# §1.64 — Exact-duplicate pHash guard (S129).
# Drops consecutive frames whose dHash Hamming distance is exactly 0 — these are
# pixel-identical at the thumbnail level and carry zero new canvas information.
# Distinct from §3.4A hold detection (which groups them) and §1.2D temporal
# variance filter (which operates in float space and can miss MPEG-exact duplicates
# that upsampled to uint8 round to identical thumbnails).
# This guard fires in step 0 of smart_select_frames, before any other filter.
# Default OFF (ASP_DHASH_EXACT_DROP=0).  Set to 1 to enable.
try:
    _DHASH_EXACT_DROP: bool = os.environ.get("ASP_DHASH_EXACT_DROP", "0") != "0"
except Exception:
    _DHASH_EXACT_DROP = False

# Real animation holds are 2-6 frames (on twos/threes, occasionally slower).
# A "hold block" this large is a false positive from the MAD/dHash detector —
# e.g. a slow scroll whose per-frame MAD never trips the threshold — and must
# not be treated as one held cel: not for the phase-correlation skip (which
# would zero out real camera motion across the whole span) and not for
# hold-block averaging (which would blur dozens of distinct poses together).
# Exposed as ASP_MAX_SKIPPABLE_HOLD_SIZE (see core/config.py _CONFIG_SCHEMA).
try:
    _MAX_SKIPPABLE_HOLD_SIZE = int(os.environ.get("ASP_MAX_SKIPPABLE_HOLD_SIZE", "8"))
except ValueError:
    _MAX_SKIPPABLE_HOLD_SIZE = 8


def _detect_hold_blocks(
    thumbs: list[np.ndarray],
    hold_threshold: float = 0.025,
) -> list[int]:
    """
    Detect animation "on twos / on threes" hold blocks and return the index of
    the first frame of each block.

    Anime animators draw a new character cel every 2–3 video frames
    (occasionally every frame for action shots, or every 4–6 for slow scenes).
    Within a hold block, consecutive frames are pixel-identical except for MPEG
    compression noise and sub-pixel camera drift.  At a hold boundary, the
    character snaps to a new pose → large pixel MAD.

    The detector compares consecutive thumbnail mean absolute differences
    (normalised to [0,1]).  If the MAD is below ``hold_threshold``, the two
    frames belong to the same hold block.  The first frame of each block is the
    representative.

    Parameters
    ----------
    thumbs : list of (H, W) float32 thumbnails in [0, 1].
    hold_threshold : mean absolute difference (in [0,1]) below which two
        consecutive thumbnails are considered the same cel.  Default 0.025
        (2.5% of [0,1] range).  Typical within-hold MAD: 0.003–0.010.
        Typical cross-hold MAD: 0.030–0.120.

    Returns
    -------
    List[int] — indices of the first frame of each hold block.  Each block
    represents one unique animation cel.  Length ≤ len(thumbs).

    Notes
    -----
    - For ``hold_threshold=0`` or len(thumbs) ≤ 1, returns list(range(N)).
    - This function is pure NumPy — no GPU, ~1ms for 300 frames.
    - Hold boundaries are the natural pose-change points (Sýkora 2009 §3.1).
      They provide a principled frame universe for Pass 2 pose-consistent
      refinement: candidates that cross exactly one hold boundary are
      guaranteed to show a different character pose (needed for ARAP to have
      useful work to do); candidates that stay within one hold are wasted
      (identical pose → ARAP residual ≈ 0, good — but selection is redundant).
    """
    N = len(thumbs)
    if hold_threshold <= 0.0 or N <= 1:
        return list(range(N))

    if _BATCH_FSEL:
        try:
            # C++ expects uint8; convert float32 [0,1] grayscale thumbnails
            u8 = [np.ascontiguousarray(
                      np.clip(t * 255, 0, 255).astype(np.uint8)
                      if t.dtype != np.uint8 else t)
                  for t in thumbs]
            # C++ returns indices of hold frames (MAD < threshold w.r.t. previous)
            hold_set = set(_batch.frame_selection.detect_hold_blocks_mad(
                u8, hold_threshold))
            return [i for i in range(N) if i not in hold_set]
        except Exception:
            pass

    blocks: list[int] = [0]
    for i in range(1, N):
        h = min(thumbs[i].shape[0], thumbs[i - 1].shape[0])
        w = min(thumbs[i].shape[1], thumbs[i - 1].shape[1])
        mad = float(
            np.mean(
                np.abs(
                    thumbs[i][:h, :w].astype(np.float32)
                    - thumbs[i - 1][:h, :w].astype(np.float32)
                )
            )
        )
        if mad > hold_threshold:
            blocks.append(i)

    return blocks


def _compute_dhash(
    thumb: np.ndarray,
    hash_size: int = 8,
) -> np.ndarray:
    """§3.4A: Difference hash (dHash) of a grayscale thumbnail.

    Resizes *thumb* to (hash_size+1, hash_size) pixels, then binarises the
    horizontal luminance gradient: column j is set to True when it is brighter
    than column j-1.  Returns a flat boolean array of ``hash_size²`` bits.

    Accepts float32 thumbnails in [0, 1] or uint8 thumbnails.  Resize uses
    INTER_AREA which averages out MPEG DCT-block noise before the comparison —
    the key advantage over MAD (which sees the raw noise).

    Parameters
    ----------
    thumb:
        Grayscale or colour thumbnail array.
    hash_size:
        Side length of the hash grid (default 8 → 64-bit hash).

    Returns
    -------
    np.ndarray of dtype bool, shape (hash_size²,).
    """
    src = np.clip(thumb * 255, 0, 255).astype(np.uint8) if thumb.dtype != np.uint8 else thumb
    if len(src.shape) == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(src, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def _detect_hold_blocks_dhash(
    thumbs: list[np.ndarray],
    distance_threshold: int = 4,
) -> list[int]:
    """§3.4A: dHash-based animation hold detection.

    More robust to MPEG compression noise than the MAD detector
    (``_detect_hold_blocks``): the INTER_AREA resize averages DCT block
    artefacts before the directional comparison, so typical within-hold
    Hamming distance remains 0–2 even for aggressively-compressed sources
    where within-hold MAD can exceed the 0.025 default threshold.

    Parameters
    ----------
    thumbs:
        List of (H, W) or (H, W, C) thumbnail arrays.
    distance_threshold:
        Maximum Hamming distance (number of differing hash bits) for two
        consecutive frames to be considered the same animation hold.  When
        ``distance_threshold <= 0`` every frame starts a new block
        (equivalent to threshold = 0 for the MAD detector).

    Returns
    -------
    List[int] — indices of the first frame of each hold block.  Same return
    convention as ``_detect_hold_blocks``.
    """
    N = len(thumbs)
    if distance_threshold <= 0 or N <= 1:
        return list(range(N))

    if _BATCH_FSEL:
        try:
            u8 = [np.ascontiguousarray(
                      np.clip(t * 255, 0, 255).astype(np.uint8)
                      if t.dtype != np.uint8 else t)
                  for t in thumbs]
            hold_set = set(_batch.frame_selection.detect_hold_blocks_dhash(
                u8, 8, distance_threshold))
            return [i for i in range(N) if i not in hold_set]
        except Exception:
            pass

    hashes = [_compute_dhash(t) for t in thumbs]
    blocks: list[int] = [0]
    for i in range(1, N):
        dist = int(np.sum(hashes[i] != hashes[i - 1]))
        if dist > distance_threshold:
            blocks.append(i)
    return blocks


def _drop_exact_dhash_duplicates(
    thumbs: list[np.ndarray],
    paths: list[str],
) -> tuple[list[np.ndarray], list[str], int]:
    """§1.64: Drop consecutive frames that are pixel-identical at dHash scale (S129).

    Uses ``_compute_dhash`` (INTER_AREA resize, 64-bit hash) to detect
    exact duplicates: frames whose Hamming distance is **0** — every gradient
    bit matches.  When two consecutive frames have distance 0 the second frame
    is dropped (the first is kept as the canonical representative of that content).

    This is stricter than §3.4A hold detection (threshold ≤ 4) and earlier
    than §1.2D temporal variance — it eliminates true byte-level duplicates
    before any heavier processing runs.

    First and last frames are always retained, even if they are identical to
    their neighbours, to preserve canvas extent.

    Parameters
    ----------
    thumbs : list of (H, W) float32 thumbnails in [0, 1].  Length N.
    paths  : corresponding file paths.  Length N.

    Returns
    -------
    (filtered_thumbs, filtered_paths, n_dropped)
    """
    N = len(thumbs)
    if N < 3:
        return list(thumbs), list(paths), 0

    hashes = [_compute_dhash(t) for t in thumbs]
    keep = [True] * N
    for i in range(1, N - 1):
        if int(np.sum(hashes[i] != hashes[i - 1])) == 0:
            keep[i] = False

    n_dropped = keep.count(False)
    return (
        [t for t, k in zip(thumbs, keep, strict=False) if k],
        [p for p, k in zip(paths, keep, strict=False) if k],
        n_dropped,
    )


def _refine_hold_ids_by_response(
    hold_ids: list[int],
    responses: list[float],
    high_response_threshold: float = 0.85,
) -> tuple[list[int], int]:
    """§1.11C — Post-hoc hold refinement using phase-correlation response.

    After phaseCorrelate runs for all cross-hold pairs, any pair whose response
    exceeds ``high_response_threshold`` represents near-identical frames that the
    MAD-based detector split into separate blocks due to MPEG compression noise.
    This function merges those blocks so that Pass 2 does not treat them as
    distinct character poses.

    Parameters
    ----------
    hold_ids:
        Per-frame hold block IDs produced by ``_detect_hold_blocks``.
        Length N (one entry per frame).
    responses:
        Phase-correlation response values from step 3.  Length N-1.
        Within-hold pairs already have response=1.0 (synthetic).
    high_response_threshold:
        Pairs with response >= this value are treated as the same cel.

    Returns
    -------
    (refined_hold_ids, n_hold_blocks)
    """
    N = len(hold_ids)
    if N < 2 or not responses:
        return list(hold_ids), len(set(hold_ids))

    ids = list(hold_ids)
    for i, resp in enumerate(responses):
        if i + 1 >= N:
            break
        # Only merge blocks that are currently split and have a high response
        if resp >= high_response_threshold and ids[i] != ids[i + 1]:
            old_id = ids[i + 1]
            new_id = ids[i]
            ids = [new_id if h == old_id else h for h in ids]

    # Renumber consecutively preserving first-occurrence order
    seen: dict = {}
    counter = 0
    result: list[int] = []
    for h in ids:
        if h not in seen:
            seen[h] = counter
            counter += 1
        result.append(seen[h])

    return result, len(seen)


__all__ = [
    "_HOLD_THRESHOLD",
    "_HIGH_HOLD_RESPONSE",
    "_HOLD_DHASH_THRESHOLD",
    "_DHASH_EXACT_DROP",
    "_MAX_SKIPPABLE_HOLD_SIZE",
    "_detect_hold_blocks",
    "_compute_dhash",
    "_detect_hold_blocks_dhash",
    "_drop_exact_dhash_duplicates",
    "_refine_hold_ids_by_response",
]
