"""Pixel-data and diagnostics computation for the evaluation tooling.

Pure functions over numpy arrays / OpenCV images and already-flattened
benchmark metrics: histograms, scatter plots, heatmaps, FFT
(``visualizations_basic``), feature matching and optical flow
(``visualizations_matching``), comparison maps — diff, SSIM, overlay,
checkerboard, swipe (``comparison_maps``), per-test pipeline diagnostics
charts (``diagnostics``), and the shared matplotlib dark theme
(``figure_theme``).

No module here imports Qt or FiftyOne, so the same builders serve the
inspector, the FiftyOne surface, and headless scripts, and every one is
unit-testable without a display. ``ui/mpl_canvas.py`` imports
``figure_theme``, never the reverse — the pre-rebuild package had this
dependency inverted, with ``logic`` reaching into ``ui`` for its theming.
"""
