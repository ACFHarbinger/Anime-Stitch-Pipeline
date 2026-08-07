"""Pixel-level comparison tools between two stitch outputs — the ImageJ/
DiffImg/ImageMagick-inspired half of the tool's spec. Every function is a pure
``np.ndarray -> np.ndarray`` (or tuple) transform with no Qt dependency, so each
is independently unit-testable and reusable from the inspector, the FiftyOne
surface, or a headless script.

Every knob a user can move in the inspector (blend alpha, swipe position,
checkerboard tile size, contour blur/threshold, amplification) is a parameter
here rather than a hardcoded constant, so the live sliders drive the same code
path the static exports do.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim

    _SSIM_OK = True
except ImportError:
    _SSIM_OK = False


def _match_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize b onto a's canvas — same convention bench_anime_stitch.py's own
    GT comparison metrics use."""
    h, w = a.shape[:2]
    if b.shape[:2] != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a, b


def shape_note(a: np.ndarray, b: np.ndarray) -> str | None:
    """A caveat string when the two images had different canvas sizes.

    Comparators legitimately produce different canvas dimensions (ASP
    1703x1704 vs Simple 1917x2050 on test01, a *framing* difference the
    objective explicitly cares about), and resampling one onto the other
    hides that. Any per-pixel comparison shown to a human needs to say so.
    """
    if a.shape[:2] == b.shape[:2]:
        return None
    return (
        f"Canvas sizes differ ({a.shape[1]}x{a.shape[0]} vs {b.shape[1]}x{b.shape[0]}); "
        "B was resampled onto A. Per-pixel differences below include that rescale."
    )


def abs_diff_inverted(a: np.ndarray, b: np.ndarray, amplify: float = 1.0) -> np.ndarray:
    """Absolute-difference map inverted so unchanged background reads white
    and any real difference pops as a dark/colored region.

    ``amplify`` scales the difference before inverting — anime cels are flat,
    so a genuine misalignment inside a colour field can be a handful of
    levels and invisible at 1x.
    """
    a, b = _match_shape(a, b)
    diff = cv2.absdiff(a, b)
    if amplify != 1.0:
        diff = cv2.convertScaleAbs(diff, alpha=amplify)
    return 255 - diff


@dataclasses.dataclass
class SsimResult:
    score: float
    heatmap: np.ndarray
    exact: bool  # False when skimage is missing and the abs-diff fallback ran


def ssim_heatmap(a: np.ndarray, b: np.ndarray) -> SsimResult:
    """Structural-similarity map plus its global score.

    Returns the score alongside the heatmap because the old dashboard
    computed it and threw it away (issue #123 defect 7) — the scalar is the
    number worth reading, the map only says *where*.
    """
    a, b = _match_shape(a, b)
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    if not _SSIM_OK:
        diff = cv2.absdiff(gray_a, gray_b).astype(np.float32) / 255.0
        heat = cv2.applyColorMap((diff * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        return SsimResult(score=float(1.0 - diff.mean()), heatmap=heat, exact=False)
    score, ssim_map = _ssim(gray_a, gray_b, full=True, data_range=255)
    normalized = np.clip((1.0 - ssim_map) * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    return SsimResult(score=float(score), heatmap=heat, exact=True)


def false_color_overlay(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Red = a, cyan = b, so any translation/rotation misalignment shows as
    red/cyan ghosting fringes rather than a flat blend."""
    a, b = _match_shape(a, b)
    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    out = np.zeros((*gray_a.shape, 3), dtype=np.uint8)
    out[..., 2] = gray_a  # R channel (BGR order)
    out[..., 1] = gray_b  # G
    out[..., 0] = gray_b  # B — G+B together = cyan
    return out


def alpha_blend(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    a, b = _match_shape(a, b)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(a, alpha, b, 1.0 - alpha, 0.0)


def swipe_composite(
    a: np.ndarray, b: np.ndarray, split: float = 0.5, vertical: bool = True
) -> tuple[np.ndarray, int]:
    """Classic before/after wipe: A on one side of a moving split, B on the
    other. Returns the composite and the split coordinate in pixels so the
    caller can draw the divider line in image space.

    ``vertical=True`` gives a vertical divider that slides horizontally; a
    horizontal divider is the useful one for scroll-axis seams in a tall
    panorama, hence the switch.
    """
    a, b = _match_shape(a, b)
    h, w = a.shape[:2]
    out = a.copy()
    if vertical:
        split_x = int(np.clip(split, 0.0, 1.0) * w)
        out[:, split_x:] = b[:, split_x:]
        return out, split_x
    split_y = int(np.clip(split, 0.0, 1.0) * h)
    out[split_y:, :] = b[split_y:, :]
    return out, split_y


def checkerboard_mosaic(a: np.ndarray, b: np.ndarray, tile: int = 64) -> np.ndarray:
    a, b = _match_shape(a, b)
    h, w = a.shape[:2]
    tile = max(4, int(tile))
    out = a.copy()
    # Vectorised tile mask — the old nested Python loop ran ~27x27 iterations
    # per redraw on a 1700px panorama, which a live tile-size slider makes
    # unusable.
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((yy // tile) + (xx // tile)) % 2) == 1
    out[mask] = b[mask]
    return out


def contour_bounding(
    a: np.ndarray, b: np.ndarray, blur_ksize: int = 15, thresh: int = 25, min_area: int = 64
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Gaussian-blur the diff map to suppress single-pixel noise, threshold
    it, then draw high-contrast boxes around the surviving changed regions on
    an alpha-blend base image — the "what actually moved" view."""
    a, b = _match_shape(a, b)
    diff = cv2.absdiff(a, b)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    k = max(1, int(blur_ksize)) | 1  # GaussianBlur requires an odd kernel size
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    _, mask = cv2.threshold(blurred, int(thresh), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    base = alpha_blend(a, b, 0.5)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:  # drop noise specks too small to be a real defect
            continue
        boxes.append((x, y, w, h))
        cv2.rectangle(base, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return base, boxes


def edge_overlay(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """B's Canny edges drawn over A in magenta.

    On flat anime cels the line art *is* the structure, so overlaying one
    output's edges on the other localises a misalignment far more precisely
    than an intensity difference, which a colour field swallows.
    """
    a, b = _match_shape(a, b)
    edges = cv2.Canny(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), 80, 160)
    out = (a.astype(np.float32) * 0.65).astype(np.uint8)
    out[edges > 0] = (255, 0, 255)
    return out


def difference_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Scalar summary of a pair, shown next to whichever map is on screen so
    the visual read is anchored to numbers."""
    a, b = _match_shape(a, b)
    diff = cv2.absdiff(a, b).astype(np.float32)
    gray = diff.mean(axis=2)
    total = gray.size
    return {
        "mean_abs_diff": float(gray.mean()),
        "max_abs_diff": float(gray.max()),
        "p99_abs_diff": float(np.percentile(gray, 99)),
        "changed_pct_gt2": 100.0 * float((gray > 2).sum()) / total,
        "changed_pct_gt10": 100.0 * float((gray > 10).sum()) / total,
    }


def pixel_value_grid(region_bgr: np.ndarray, max_cells: int = 24) -> str:
    """Text dump of raw RGB pixel values for a small selected region —
    Pixel Value Mode's numeric readout for a bbox-selected crop."""
    h, w = region_bgr.shape[:2]
    step_y = max(1, h // max_cells)
    step_x = max(1, w // max_cells)
    lines = []
    for y in range(0, h, step_y):
        row = " ".join(
            f"({region_bgr[y, x, 2]:3d},{region_bgr[y, x, 1]:3d},{region_bgr[y, x, 0]:3d})"
            for x in range(0, w, step_x)
        )
        lines.append(row)
    return "\n".join(lines)


def region_stats(region_bgr: np.ndarray) -> dict:
    """Per-channel statistics for a selected crop — the numeric companion to
    the pixel grid, useful for judging a flat-fill colour shift that the eye
    reads as "the same colour"."""
    channels = {}
    for i, name in enumerate(("B", "G", "R")):
        band = region_bgr[..., i].astype(np.float32)
        channels[name] = {
            "mean": float(band.mean()),
            "std": float(band.std()),
            "min": int(band.min()),
            "max": int(band.max()),
        }
    return {"height": region_bgr.shape[0], "width": region_bgr.shape[1], "channels": channels}
