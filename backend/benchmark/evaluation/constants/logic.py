"""
Image-Toolkit Backend Evaluation Logic Constants

Contains constants used in the backend evaluation logic submodule.
"""

_SCATTER_SAMPLE = 4000  # subsampled point budget for scatter plots — full-res is unreadable and slow

# Heatmap/spectrum rasters are downsampled to this longest edge before
# plotting. A benchmark panorama is ~1700-2000 px on its long edge; a
# matplotlib axes is a few hundred device pixels, so plotting full-res only
# costs time (a full-res 1704x1703 FFT was ~1 s per click) without adding a
# single visible detail.
VIZ_MAX_EDGE = 720

# Percentile window used to contrast-stretch the FFT log-magnitude spectrum.
# The DC spike is orders of magnitude above everything else, so autoscaling
# to [min, max] renders every natural image as the same near-uniform field
# with one bright centre dot — the "FFT looks identical for every image" bug
# (issue #123 defect 3). Clipping to the 1st-99.5th percentile of the log
# magnitudes puts the actual spectral structure across the full colour range.
FFT_PERCENTILE_LO = 1.0
FFT_PERCENTILE_HI = 99.5

# Radial power-spectrum profile: number of frequency bins from DC to Nyquist.
# This is the discriminative FFT read (it separates a sharp composite from a
# blurred one by the slope of its high-frequency tail), so it is plotted
# alongside the 2D spectrum rather than instead of it.
FFT_RADIAL_BINS = 96

# Per-image figure cache size. One test's session touches at most 5
# comparator images x ~12 visualizations; 48 entries keeps a whole test's
# worth of already-computed figures warm without unbounded growth across a
# 97-test pass.
FIGURE_CACHE_SIZE = 48
