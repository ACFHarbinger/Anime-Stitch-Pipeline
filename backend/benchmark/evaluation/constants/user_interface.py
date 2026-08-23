"""
Image-Toolkit Backend Evaluation User Interface Constants

Contains constants used in the backend evaluation UI submodule.
"""

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Matplotlib figure backgrounds, kept identical to bench_anime_stitch.py's own
# plot styling so live visualizations read as the same tool family as the
# static report plots.
FIG_BG = "#12121f"
AX_BG = "#1a1a2e"

# The inspector's own chrome, derived from the two figure colours above so
# embedded plots sit flush in the surrounding panels instead of floating in a
# differently-tinted box. This is the DARK theme's palette; kept as the
# original bare names since most of the codebase (figure_theme.py,
# image_panel.py, panel_grid.py, matplotlib figure builders, ...) imports
# these directly as fixed overlay/annotation colours — that's deliberate, not
# an oversight: a link line or a defect-region outline is drawn *over*
# photographic/anime content, not over the app's own chrome, so keeping it
# constant regardless of the light/dark toggle is the right call, the same
# way most photo tools keep canvas-overlay colours theme-independent. Only
# the widget-chrome stylesheet itself (theme.py's DARK_STYLESHEET /
# LIGHT_STYLESHEET) actually switches with the toggle.
COL_BG = "#0e0e18"
COL_SURFACE = "#161624"
COL_SURFACE_HI = "#1f1f33"
COL_BORDER = "#2c2c44"
COL_TEXT = "#e6e6f0"
COL_TEXT_DIM = "#9a9ab4"
COL_ACCENT = "#00e5ff"
COL_ACCENT_DIM = "#0891b2"
COL_WARN = "#ffd93d"
COL_BAD = "#ff6b6b"
COL_GOOD = "#6bffb8"

# Annotation overlay colours
COL_BBOX = "#ffd93d"
COL_BBOX_ACTIVE = "#ff6b6b"
COL_POINT = "#6bffb8"
COL_EDGE = "#00e5ff"

# ---------------------------------------------------------------------------
# Themes (Settings dialog: dark/light chrome toggle)
# ---------------------------------------------------------------------------
THEME_DARK = "dark"
THEME_LIGHT = "light"
THEMES = ((THEME_DARK, "Dark"), (THEME_LIGHT, "Light"))

# The light theme's chrome palette — same key set as the dark constants above,
# bundled here (rather than as a second round of bare module constants) since
# nothing outside theme.py needs to pick a palette at runtime.
LIGHT_PALETTE = {
    "bg": "#eef0f5",
    "surface": "#ffffff",
    "surface_hi": "#e3e7f0",
    "border": "#c7cddb",
    "text": "#1a1c2b",
    "text_dim": "#5b6178",
    "accent": "#0e7490",
    "accent_dim": "#155e75",
    "warn": "#b45309",
    "bad": "#dc2626",
    "good": "#15803d",
}
DARK_PALETTE = {
    "bg": COL_BG,
    "surface": COL_SURFACE,
    "surface_hi": COL_SURFACE_HI,
    "border": COL_BORDER,
    "text": COL_TEXT,
    "text_dim": COL_TEXT_DIM,
    "accent": COL_ACCENT,
    "accent_dim": COL_ACCENT_DIM,
    "warn": COL_WARN,
    "bad": COL_BAD,
    "good": COL_GOOD,
}

# Score colour ramp, 0-4 — used by score buttons and by the per-test summary
# chips so a bad rating is visible without reading the number.
SCORE_COLORS = {
    0: "#ff6b6b",
    1: "#ff9f68",
    2: "#ffd93d",
    3: "#a8e05f",
    4: "#6bffb8",
}


# ---------------------------------------------------------------------------
# Canvas interaction
# ---------------------------------------------------------------------------
MODE_NAVIGATE = "navigate"
MODE_BBOX = "bbox"
MODE_POINT = "point"
MODE_PROBE = "probe"

TOOL_MODES = (
    (MODE_NAVIGATE, "Navigate", "Drag to pan, wheel to zoom"),
    (MODE_BBOX, "Draw region", "Drag a box around a defect, then tag it"),
    (MODE_POINT, "Link", "Click a point or drag a region in 2+ panels, then Enter to link them"),
    (MODE_PROBE, "Probe pixels", "Hover to read values; click to pin a readout"),
)

# Display modes. DISPLAY_PIXEL overlays a per-pixel grid with numeric RGB
# values once the zoom is high enough for the text to fit — the mode that was
# a complete no-op in the old dashboard (issue #123 defect 1).
DISPLAY_RAW = "display"
DISPLAY_PIXEL = "pixel"

# Zoom is expressed as a multiple of *native image pixels* (1.0 = 100%).
ZOOM_MIN = 0.02
ZOOM_MAX = 64.0
ZOOM_STEP = 1.15

# Above this zoom, Pixel Value Mode draws the pixel grid; the numeric RGB
# triples only fit once each pixel is at least ~26 device px across, so the
# text threshold is separate from (and higher than) the grid threshold.
PIXEL_GRID_ZOOM_THRESHOLD = 8.0
PIXEL_TEXT_ZOOM_THRESHOLD = 26.0
# Guard against painting text into a million cells when a huge region is on
# screen at a high zoom.
PIXEL_TEXT_MAX_CELLS = 900

# The hover magnifier is Pixel Value Mode's primary, always-on behaviour: at
# a test's default fit-to-view zoom (often well under 1x native on a 1700px+
# panorama), the in-image grid above never engages, which is why toggling the
# mode used to look like it "did nothing" (issue #123 followup). The magnifier
# reads directly from the source array, independent of the view's zoom/pan, so
# it works immediately on hover regardless of how far in the user has zoomed.
PIXEL_MAGNIFIER_RADIUS = 8  # neighbourhood half-width in source px (17x17 grid)
PIXEL_MAGNIFIER_CELL = 16  # on-screen px per cell
PIXEL_MAGNIFIER_MARGIN = 10  # inset from the viewport corner


# ---------------------------------------------------------------------------
# Panel layouts
# ---------------------------------------------------------------------------
LAYOUT_ROW = "row"
LAYOUT_GRID = "grid"
LAYOUT_COLUMN = "column"
LAYOUT_STACK = "stack"

LAYOUTS = (
    (LAYOUT_ROW, "Side by side", "One row, all visible comparators"),
    (LAYOUT_GRID, "Grid", "Wrapped 2-per-row grid — best for 3+ comparators"),
    (LAYOUT_COLUMN, "Stacked rows", "One column — best for wide panoramas"),
    (LAYOUT_STACK, "Flip (A/B)", "One panel at a time, Tab to flip between them"),
)


# ---------------------------------------------------------------------------
# Scoring / navigation
# ---------------------------------------------------------------------------
SCORE_SCALE_HINT = "4=keepable  3=minor flaw  2=flawed but parses  1=mostly broken  0=incoherent"

# Keyboard-first rating. §0.1 budgets ~45 min for the 97-test pass (~28 s per
# test), which is not reachable through a mouse-only scoring form.
KEY_HINTS = (
    ("0-4", "score the focused image's coherence"),
    ("A / S", "focus ASP / Simple for scoring"),
    ("Tab", "cycle focused panel"),
    ("[ / ]", "prefer ASP / prefer Simple"),
    ("=", "call it a tie"),
    ("F", "fit all panels"),
    ("Ctrl+0-9", "toggle defect tag N"),
    ("Space", "save and next"),
    ("Backspace", "previous test"),
    ("Ctrl+S", "save ratings (nothing is written until you do)"),
    ("Enter", "finish a multi-point link"),
    ("Esc", "cancel the in-progress link"),
)
