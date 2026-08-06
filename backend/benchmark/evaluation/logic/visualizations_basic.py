"""Per-image pixel-data visualizations: histograms, scatter plots, heatmaps,
gradient maps, and FFT analysis. Each function takes one BGR image (or a
selected region crop) and returns a matplotlib ``Figure`` ready for
``PlotCanvas``. Gradient heatmap conventions (Sobel magnitude, ``inferno``
colormap) match ``bench_anime_stitch.py``'s ``_save_seam_heatmap`` so the tool
shows the same visual language as the static report.

Every raster is downsampled to ``VIZ_MAX_EDGE`` before plotting: a benchmark
panorama is ~1700-2000 px on its long edge, a matplotlib axes is a few hundred
device pixels wide, so full-res only costs time.
"""

from __future__ import annotations

import cv2
import numpy as np
from matplotlib.figure import Figure

from ..constants.logic import (
    _SCATTER_SAMPLE,
    FFT_PERCENTILE_HI,
    FFT_PERCENTILE_LO,
    FFT_RADIAL_BINS,
    VIZ_MAX_EDGE,
)
from .figure_theme import themed_figure, themed_legend


def _downsample(img: np.ndarray, max_edge: int = VIZ_MAX_EDGE) -> np.ndarray:
    """Area-downsample so the long edge is at most ``max_edge``."""
    h, w = img.shape[:2]
    scale = max(h, w) / float(max_edge)
    if scale <= 1.0:
        return img
    return cv2.resize(img, (max(1, int(w / scale)), max(1, int(h / scale))), interpolation=cv2.INTER_AREA)


def _gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _subsample_pixels(img: np.ndarray, n: int = _SCATTER_SAMPLE) -> np.ndarray:
    h, w = img.shape[:2]
    total = h * w
    idx = np.random.default_rng(0).choice(total, size=min(n, total), replace=False)
    flat = img.reshape(-1, img.shape[2]) if img.ndim == 3 else img.reshape(-1, 1)
    return flat[idx]


def color_channel_figure(img_bgr: np.ndarray) -> Figure:
    """Per-channel intensity distribution, one line per BGR channel.

    Named for what it actually is rather than "Color Histogram": a histogram
    implies bars, and a smooth per-value count curve read as broken/wrong at a
    glance next to the (properly blocky) cumulative plot below. The underlying
    data — ``cv2.calcHist`` per channel — is unchanged; only the honest name.
    """
    fig, ax = themed_figure()
    for i, hexcolor in enumerate(("#4ecdc4", "#6bff6b", "#ff6b6b")):  # B, G, R
        hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256]).flatten()
        ax.plot(hist, color=hexcolor, alpha=0.85, linewidth=1.2)
    ax.set_title("Colour Channel Distribution (BGR)")
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Count")
    return fig


def cumulative_histogram_figure(img_bgr: np.ndarray) -> Figure:
    """Cumulative luminance histogram, drawn as a filled step plot.

    A CDF is mathematically smooth, but rendering it as an interpolated line
    reads as "not actually a histogram" — the fix is the *visual convention*,
    not the data: a stepped, filled silhouette is what every other histogram
    tool draws for cumulative counts, so this one should too.
    """
    fig, ax = themed_figure()
    hist = cv2.calcHist([_gray(img_bgr)], [0], None, [256], [0, 256]).flatten()
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1] * 100.0
    bins = np.arange(257)  # 256 step edges + the closing right edge
    ax.fill_between(bins, np.append(cdf, cdf[-1]), step="post", color="#ffd93d", alpha=0.35)
    ax.step(bins, np.append(cdf, cdf[-1]), where="post", color="#ffd93d", linewidth=1.4)
    ax.set_title("Cumulative Luminance Histogram")
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Cumulative %")
    ax.set_xlim(0, 255)
    return fig


def scatter_2d_figure(img_bgr: np.ndarray) -> Figure:
    fig, ax = themed_figure()
    px = _subsample_pixels(img_bgr)
    ax.scatter(px[:, 2], px[:, 1], s=2, c=px[:, ::-1] / 255.0, alpha=0.5)
    ax.set_title("2D Scatter — Red vs Green channel")
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    return fig


def scatter_3d_figure(img_bgr: np.ndarray) -> Figure:
    fig, ax = themed_figure(projection="3d")
    px = _subsample_pixels(img_bgr, 1500)
    ax.scatter(px[:, 2], px[:, 1], px[:, 0], s=2, c=px[:, ::-1] / 255.0, alpha=0.6)
    ax.set_title("3D Scatter — RGB Color Space")
    ax.set_xlabel("R")
    ax.set_ylabel("G")
    ax.set_zlabel("B")
    return fig


def spatial_scatter_figure(img_bgr: np.ndarray) -> Figure:
    """Scatter of edge-pixel (x, y) locations — reveals where structural
    detail concentrates, distinct from the color-space scatter above."""
    fig, ax = themed_figure()
    edges = cv2.Canny(_gray(_downsample(img_bgr)), 80, 160)
    ys, xs = np.nonzero(edges)
    if len(xs) > _SCATTER_SAMPLE:
        idx = np.random.default_rng(0).choice(len(xs), _SCATTER_SAMPLE, replace=False)
        xs, ys = xs[idx], ys[idx]
    ax.scatter(xs, ys, s=1, c="#00e5ff", alpha=0.4)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("Spatial Scatter — Edge Pixel Locations")
    return fig


def intensity_heatmap_figure(img_bgr: np.ndarray) -> Figure:
    fig, ax = themed_figure()
    im = ax.imshow(_gray(_downsample(img_bgr)), cmap="viridis")
    ax.set_title("Intensity Heatmap")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.03)
    return fig


def gradient_heatmap_figure(img_bgr: np.ndarray) -> Figure:
    fig, ax = themed_figure()
    gray = _gray(_downsample(img_bgr)).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    im = ax.imshow(np.sqrt(gx**2 + gy**2), cmap="inferno")
    ax.set_title("Gradient Magnitude Heatmap")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.03)
    return fig


# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------


def fft_log_magnitude(img_bgr: np.ndarray) -> np.ndarray:
    """Centred log-magnitude spectrum of the luminance channel."""
    gray = _gray(_downsample(img_bgr)).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    return np.log1p(np.abs(spectrum))


def radial_power_profile(magnitude: np.ndarray, bins: int = FFT_RADIAL_BINS) -> tuple:
    """Mean log-magnitude per radial frequency bin, DC to Nyquist.

    This is the *discriminative* FFT read: a blurred or over-averaged
    composite loses its high-frequency tail, which shows up here as a
    steeper falloff, while the 2D spectrum image of any two natural images
    looks near-identical to the eye.

    Returns ``(normalized_frequency, mean_log_magnitude)``.
    """
    h, w = magnitude.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    # Normalize each axis by its own Nyquist so a non-square panorama isn't
    # measured as anisotropic purely because of its aspect ratio.
    radius = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2) / np.sqrt(2.0)
    idx = np.clip((radius * bins).astype(np.int32), 0, bins - 1)
    total = np.bincount(idx.ravel(), weights=magnitude.ravel(), minlength=bins)
    count = np.bincount(idx.ravel(), minlength=bins)
    valid = count > 0
    freqs = (np.arange(bins) + 0.5) / bins
    return freqs[valid], total[valid] / count[valid]


def fft_magnitude_figure(img_bgr: np.ndarray) -> Figure:
    """2D log-magnitude spectrum plus its radial power profile.

    Fixes issue #123 defect 3 ("FFT Spectrum is the same for all images").
    The old version handed the raw ``log1p(|F|)`` array to ``imshow`` on
    matplotlib's default linear autoscale; the DC spike is orders of
    magnitude above everything else, so every natural image rendered as the
    same near-uniform field with one bright centre dot. Two fixes: clip the
    colour range to the 1st-99.5th percentile of the log magnitudes so the
    actual spectral structure spans the full colormap, and plot the radial
    profile beside it, which is where a sharpness difference is legible as a
    number rather than a texture.
    """
    magnitude = fft_log_magnitude(img_bgr)
    lo, hi = np.percentile(magnitude, [FFT_PERCENTILE_LO, FFT_PERCENTILE_HI])
    if hi <= lo:  # a constant image has no spectrum to stretch
        hi = lo + 1e-6

    fig, axes = themed_figure(figsize=(9.5, 4.4), n_axes=2)
    ax_img, ax_profile = axes
    # aspect="equal" (imshow's default): the old aspect="auto" stretched the
    # spectrum to the axes box, which distorts the radial symmetry that is
    # the whole point of looking at it.
    im = ax_img.imshow(magnitude, cmap="magma", vmin=lo, vmax=hi)
    ax_img.set_title(f"FFT Log-Magnitude (p{FFT_PERCENTILE_LO:g}-p{FFT_PERCENTILE_HI:g} stretch)")
    ax_img.axis("off")
    fig.colorbar(im, ax=ax_img, fraction=0.046)

    freqs, profile = radial_power_profile(magnitude)
    ax_profile.plot(freqs, profile, color="#00e5ff", linewidth=1.6)
    ax_profile.set_title("Radial Power Profile")
    ax_profile.set_xlabel("Normalized spatial frequency (1.0 = Nyquist)")
    ax_profile.set_ylabel("Mean log magnitude")
    ax_profile.grid(True, color="#333", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return fig


def fft_profile_comparison_figure(images: dict) -> Figure:
    """Radial power profiles of several comparators on one axes.

    The direct answer to "are these spectra actually different?" — one curve
    per comparator, so a sharpness or blur difference is a visible gap
    rather than two pictures that look alike.
    """
    fig, ax = themed_figure(figsize=(7.5, 4.5))
    palette = ("#00e5ff", "#ffd93d", "#6bffb8", "#ff6b6b", "#c792ea")
    for i, (label, img) in enumerate(images.items()):
        if img is None:
            continue
        freqs, profile = radial_power_profile(fft_log_magnitude(img))
        ax.plot(freqs, profile, linewidth=1.6, alpha=0.9,
                color=palette[i % len(palette)], label=label)
    ax.set_title("Radial Power Profile — all comparators")
    ax.set_xlabel("Normalized spatial frequency (1.0 = Nyquist)")
    ax.set_ylabel("Mean log magnitude")
    ax.grid(True, color="#333", linewidth=0.5, alpha=0.6)
    themed_legend(ax)
    fig.tight_layout()
    return fig


def row_luminance_profile_figure(img_bgr: np.ndarray) -> Figure:
    """Per-row mean luminance — the same quantity ``_seam_coherence``
    reduces to a single std, so a banding artifact the metric only reports as
    a scalar is visible here as the periodic ripple that produced it."""
    fig, ax = themed_figure(figsize=(7.5, 4.0))
    gray = _gray(img_bgr).astype(np.float32)
    rows = gray.mean(axis=1)
    ax.plot(np.arange(len(rows)), rows, color="#ffd93d", linewidth=1.0)
    ax.axhline(float(rows.mean()), color="#00e5ff", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_title(f"Per-Row Mean Luminance (std={float(rows.std()):.2f} — the seam_coherence metric)")
    ax.set_xlabel("Row (y)")
    ax.set_ylabel("Mean luminance")
    ax.grid(True, color="#333", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return fig
