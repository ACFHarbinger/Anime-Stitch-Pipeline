"""Frame-list utilities: ordering, dy_cv scroll-irregularity gate, spatial
dedup, SCANS-fallback frame reload, canvas coverage, hires keyframe swap-in."""

from __future__ import annotations

import os
import re

import cv2
import numpy as np
from asp_backend.alignment.canvas import _load_frames, _normalise_widths

from ._probes import _HAS_BATCH, _batch


def _sort_frames_by_index(paths: list[str]) -> list[str]:
    """§1.63: Sort frame paths by numeric suffix extracted from the filename (S127).

    Frame file names produced by video extraction tools (FFmpeg, OpenCV) are
    typically ``frame_00001.png``, ``frame_00002.png``, etc.  When the caller
    discovers frames via ``glob()`` on some file systems (e.g. ext4 with dir_index)
    the OS-level directory order may not be numeric.  An out-of-order frame list
    causes the pipeline to treat consecutive file-system neighbours as adjacent
    camera positions, producing nonsensical phase-correlation displacements,
    reversed scroll direction, and incorrect BA edge graphs.

    This function re-sorts *paths* by the rightmost contiguous digit run in the
    stem (filename without extension).  When no digit run is found for a path,
    that path is sorted by its original index in *paths* (stable), placing it
    after all numerically-indexed paths.  This keeps the behaviour predictable
    for mixed-name directories while avoiding an import of ``natsort``.

    Parameters
    ----------
    paths : list of file paths to sort.

    Returns
    -------
    list[str]
        New list in ascending numeric-suffix order.  If all stems lack a digit
        suffix (e.g. user-supplied paths with descriptive names), the original
        order is returned unchanged.
    """

    def _key(p: str) -> tuple:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"(\d+)$", stem)
        return (0, int(m.group(1))) if m else (1, 0)

    sorted_paths = sorted(paths, key=_key)
    return sorted_paths


def _compute_dy_cv(affines: list[np.ndarray]) -> float:
    """§4.7: Coefficient of variation of adjacent vertical frame steps.

    Computes ``std(|Δty|) / mean(|Δty|)`` from the bundle-adjusted affines.
    A high dy_cv indicates an irregular scroll pattern (variable step sizes)
    where ASP's compositing assumptions break down.

    97-test benchmark (S160, 2026-06-23): dy_cv ≥ 1.5 → catastrophic ASP
    failure on every test in that regime (AlSSIM −22 to −37%, seam_vis
    60–120 vs SCANS 2–3).  SCANS handles these sequences trivially because
    it requires no frame-to-frame registration.

    Returns 0.0 when N < 2 (gate will not fire).

    Parameters
    ----------
    affines:
        List of N 2×3 float32 affine matrices from bundle adjustment.

    Returns
    -------
    float
        dy_cv ≥ 0.  Zero when N < 2.
    """
    N = len(affines)
    if N < 2:
        return 0.0
    dy_steps = [abs(float(affines[k][1, 2]) - float(affines[k - 1][1, 2])) for k in range(1, N)]
    mean_dy = float(np.mean(dy_steps))
    if mean_dy < 1.0:
        return 0.0
    return float(np.std(dy_steps)) / mean_dy


def _compute_adaptive_dy_cv_max(n_frames: int, base_max: float = 1.5) -> float:
    """§5.8: Lower dy_cv ceiling for sequences with many frames.

    With N≥8 frames, step irregularity compounds across more seams.
    Scale: max(base_max * 8 / max(n_frames, 8), 0.8)
    - N=8: base_max (no change, floor ≥0.8)
    - N=16: base_max * 0.5 = 0.75 (→ floor 0.8)
    - N=4: base_max (unchanged, below 8)
    """
    if n_frames < 8:
        return base_max
    return max(base_max * 8.0 / n_frames, 0.8)


def _spatial_dedup_frames(
    frames: list[np.ndarray],
    scans_frames: list[np.ndarray],
    bg_masks: list[np.ndarray],
    image_paths: list[str],
    edges: list[dict],
    min_displacement_px: float,
) -> tuple[
    list[np.ndarray], list[np.ndarray], list[np.ndarray], list[str], list[dict], int
]:
    """One pass of spatial near-static frame dedup (§1.9A).

    Identifies adjacent frames (j = i+1 in current edge list) whose
    measured displacement is below ``min_displacement_px`` on the dominant
    scroll axis and removes them.  ``scans_frames`` is kept synchronised
    with ``frames`` so every SCANS fallback path uses the same frame
    subset as the main compositing branch — eliminating the desync
    that previously caused the fallback to receive near-duplicate frames
    the compositor had already discarded.

    Returns ``(frames, scans_frames, bg_masks, image_paths, edges, n_dropped)``.
    When ``n_dropped == 0`` all lists are returned unchanged (no allocation).
    """
    adj_m: dict = {e["j"]: e for e in edges if e["j"] == e["i"] + 1}
    if not adj_m:
        return frames, scans_frames, bg_masks, image_paths, edges, 0

    if _HAS_BATCH and hasattr(_batch, "frame_selection"):
        try:
            # Convert M-affine edges to dx/dy format for C++ function
            dx_dy_edges = [
                {"i": e["i"], "j": e["j"], "dx": float(e["M"][0, 2]), "dy": float(e["M"][1, 2])}
                for e in edges
            ]
            keep_idx_raw = list(
                _batch.frame_selection.spatial_dedup_frames(
                    frames, scans_frames or [], bg_masks, image_paths,
                    dx_dy_edges, float(min_displacement_px),
                )
            )
            keep_idx = [int(i) for i in keep_idx_raw]
            if len(keep_idx) == len(frames):
                return frames, scans_frames, bg_masks, image_paths, edges, 0
            o2n: dict = {old: new for new, old in enumerate(keep_idx)}
            drop_set = set(range(len(frames))) - set(keep_idx)
            new_edges = [
                {**e, "i": o2n[e["i"]], "j": o2n[e["j"]]}
                for e in edges
                if e["i"] not in drop_set and e["j"] not in drop_set
            ]
            return (
                [frames[i] for i in keep_idx],
                [scans_frames[i] for i in keep_idx] if scans_frames else [],
                [bg_masks[i] for i in keep_idx],
                [image_paths[i] for i in keep_idx],
                new_edges,
                len(drop_set),
            )
        except Exception:
            pass

    adx = [abs(float(e["M"][0, 2])) for e in adj_m.values()]
    ady = [abs(float(e["M"][1, 2])) for e in adj_m.values()]
    spa_axis = 0 if float(np.median(adx)) > float(np.median(ady)) else 1

    drop: set = set()
    for jj in sorted(adj_m):
        ee = adj_m[jj]
        if ee["i"] in drop:
            continue
        if abs(float(ee["M"][spa_axis, 2])) < min_displacement_px:
            drop.add(jj)

    if not drop:
        return frames, scans_frames, bg_masks, image_paths, edges, 0

    N = len(frames)
    keep_idx = [i for i in range(N) if i not in drop]
    o2n: dict = {old: new for new, old in enumerate(keep_idx)}
    new_edges = [
        {**e, "i": o2n[e["i"]], "j": o2n[e["j"]]}
        for e in edges
        if e["i"] not in drop and e["j"] not in drop
    ]
    return (
        [frames[i] for i in keep_idx],
        [scans_frames[i] for i in keep_idx] if scans_frames else [],  # §1.9A/§1.9C
        [bg_masks[i] for i in keep_idx],
        [image_paths[i] for i in keep_idx],
        new_edges,
        len(drop),
    )


def _reload_scans_frames(paths: list[str]) -> list[np.ndarray]:
    """§1.9C: Reload and width-normalise original frames from disk on demand.

    Called only when a SCANS/PANORAMA fallback actually fires and
    ``_SCANS_RELOAD=True``, so the Stage-2 snapshot allocation is avoided for
    the common (success) path.  ``paths`` is already synchronised with the
    live frame list by §1.9A spatial dedup, so the reloaded set matches what
    the pipeline was working with when it failed.
    """
    loaded = _load_frames(paths)
    if not loaded:
        return []
    return _normalise_widths(loaded)


def _compute_row_coverage(
    affines: list,
    frames: list,
    canvas_h: int,
) -> tuple:
    """
    Compute per-row frame coverage for the multi-frame canvas coverage gate.

    Returns
    -------
    (row_cov, pct_multi, median_cov) where:
      row_cov    : (canvas_h,) int32 — number of frames covering each row
      pct_multi  : fraction of content rows with ≥2-frame coverage (0–1)
      median_cov : median coverage among content rows
    """
    row_cov = np.zeros(canvas_h, dtype=np.int32)
    for _aff, _frame in zip(affines, frames, strict=False):
        _r0 = max(0, round(float(_aff[1, 2])))
        _r1 = min(canvas_h, _r0 + _frame.shape[0])
        if _r1 > _r0:
            row_cov[_r0:_r1] += 1
    content_rows = row_cov > 0
    n_content = int(content_rows.sum())
    if n_content == 0:
        return row_cov, 0.0, 0.0
    n_multi = int((row_cov[content_rows] >= 2).sum())
    pct_multi = n_multi / n_content
    median_cov = float(np.median(row_cov[content_rows]))
    return row_cov, pct_multi, median_cov


def _apply_hires_keyframes(
    frames: list[np.ndarray],
    affines: list[np.ndarray],
    bg_masks: list[np.ndarray | None],
    hires_keyframes: dict[int, str],
) -> tuple[int, list[np.ndarray], list[np.ndarray], list[np.ndarray | None]]:
    """
    Replace proxy frames with hires counterparts and scale affines/masks.

    Issue 9C (Sprint 8) — Hybrid 4K/1080p compositing.

    All heavy computation (phases 1–8: photometric correction, masking, matching,
    BA, ECC) ran at proxy (1080p) resolution. This function:
    1. Loads hires frames for the indices listed in *hires_keyframes*.
    2. Determines the (scale_y, scale_x) factor from the first successfully
       loaded hires frame vs. its proxy counterpart.
    3. Scales affine translation components (tx, ty) by (scale_x, scale_y).
       The linear sub-matrix (rotation/scale/shear) is dimensionless and unchanged.
    4. For frame indices NOT in hires_keyframes, bicubic-upscales the proxy.
    5. Resizes all bg_masks to match the hires frame dimensions.

    Returns (n_loaded, frames_hires, affines_scaled, masks_resized).
    When n_loaded == 0 all inputs are returned unchanged.
    """
    hires_imgs: dict[int, np.ndarray] = {}
    for idx, path in hires_keyframes.items():
        if 0 <= idx < len(frames):
            img = cv2.imread(path)
            if img is not None:
                hires_imgs[idx] = img

    if not hires_imgs:
        return 0, frames, affines, bg_masks

    ref_idx = next(iter(hires_imgs))
    hires_h, hires_w = hires_imgs[ref_idx].shape[:2]
    proxy_h, proxy_w = frames[ref_idx].shape[:2]
    if proxy_h == 0 or proxy_w == 0:
        return 0, frames, affines, bg_masks

    scale_y = hires_h / proxy_h
    scale_x = hires_w / proxy_w

    affines_scaled = []
    for a in affines:
        a_new = a.copy().astype(np.float64)
        a_new[0, 2] *= scale_x
        a_new[1, 2] *= scale_y
        affines_scaled.append(a_new)

    frames_hires: list[np.ndarray] = []
    for i, f in enumerate(frames):
        if i in hires_imgs:
            frames_hires.append(hires_imgs[i])
        else:
            frames_hires.append(
                cv2.resize(f, (hires_w, hires_h), interpolation=cv2.INTER_LANCZOS4)
            )

    masks_resized: list[np.ndarray | None] = []
    for m in bg_masks:
        if m is None:
            masks_resized.append(None)
        else:
            masks_resized.append(
                cv2.resize(m, (hires_w, hires_h), interpolation=cv2.INTER_NEAREST)
            )

    return len(hires_imgs), frames_hires, affines_scaled, masks_resized


__all__ = [
    "_sort_frames_by_index",
    "_compute_dy_cv",
    "_compute_adaptive_dy_cv_max",
    "_spatial_dedup_frames",
    "_reload_scans_frames",
    "_compute_row_coverage",
    "_apply_hires_keyframes",
    "_HAS_BATCH",
    "_batch",
]
