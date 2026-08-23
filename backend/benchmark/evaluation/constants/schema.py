"""
Image-Toolkit Backend Evaluation Schema Constants

The vocabulary the evaluation schema is written against: which comparator
images can be judged, which quality dimensions get a sub-score, and the
closed defect taxonomy. Kept out of ``other/schema.py`` so the UI, the
FiftyOne triage surface, and the tests all read the same lists without any
of them importing a Qt or FiftyOne module to get at them.
"""

# ---------------------------------------------------------------------------
# Comparator images
# ---------------------------------------------------------------------------
# Ordered as they should appear left-to-right in the inspector. "asp" and
# "simple" are load-bearing: bench_anime_stitch.py's _load_human_evaluations()
# reads exactly those two keys off the top level of every evaluation entry.
IMAGE_ASP = "asp"
IMAGE_SIMPLE = "simple"
IMAGE_OVERMIX = "overmix"
IMAGE_HUGIN = "hugin"
IMAGE_GROUND_TRUTH = "ground_truth"

COMPARATORS = (
    (IMAGE_ASP, "ASP"),
    # Display label only — "OpenCV (SCANS)" now that there's more than one
    # ASP alternative (Overmix, Hugin). The internal key stays "simple": it's
    # the literal JSON field bench_anime_stitch.py's _load_human_evaluations()
    # reads, and renaming it would break that contract for no user-visible gain.
    (IMAGE_SIMPLE, "OpenCV (SCANS)"),
    (IMAGE_OVERMIX, "Overmix"),
    (IMAGE_HUGIN, "Hugin"),
    (IMAGE_GROUND_TRUTH, "Ground Truth"),
)
COMPARATOR_KEYS = tuple(key for key, _ in COMPARATORS)
COMPARATOR_TITLES = dict(COMPARATORS)

# Ground truth is a reference, not a pipeline output — it is displayed and
# annotated but never scored, so it is excluded from the scoring UI.
SCORABLE_KEYS = (IMAGE_ASP, IMAGE_SIMPLE, IMAGE_OVERMIX, IMAGE_HUGIN)

# The two the ASP objective is actually defined against; the pairwise
# preference and the bench-facing top-level keys are about these two only.
PRIMARY_KEYS = (IMAGE_ASP, IMAGE_SIMPLE)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
SCORE_MIN = 0
SCORE_MAX = 4

# "coherence" is the headline dimension: its value *is* the top-level
# asp/simple score the benchmark's veto logic reads. The other four are
# additive diagnostics that let a rating disagree with a metric for a
# recorded reason rather than an unattributed gut call.
DIM_COHERENCE = "coherence"
DIMENSIONS = (
    (DIM_COHERENCE, "Coherence", "torn anatomy, duplicated strips, misordered content"),
    ("sharpness", "Sharpness", "sub-pixel detail retention vs blur/softening"),
    ("framing", "Framing", "coverage of the intended pan, crop loss at the edges"),
    ("seams", "Seams", "visible seam lines, banding, ghosting at boundaries"),
    ("color", "Colour", "photometric consistency across the composite"),
)
DIMENSION_KEYS = tuple(key for key, _, _ in DIMENSIONS)

SCORE_LABELS = {
    4: "keepable — no visible structural defects",
    3: "minor flaw — small artifact, still clearly usable",
    2: "flawed but parses — visible defect, anatomy/content still reads",
    1: "mostly broken — hard to parse, major tearing/duplication",
    0: "incoherent — torn anatomy, duplicated strips, misordered content",
}
SCORE_SCALE_HINT = "4=keepable  3=minor flaw  2=flawed but parses  1=mostly broken  0=incoherent"


# ---------------------------------------------------------------------------
# Pairwise preference
# ---------------------------------------------------------------------------
PREF_ASP = "asp"
PREF_SIMPLE = "simple"
PREF_TIE = "tie"
PREFERENCES = (
    (PREF_ASP, "ASP better"),
    (PREF_TIE, "Tie"),
    (PREF_SIMPLE, "Simple better"),
)
PREFERENCE_KEYS = tuple(key for key, _ in PREFERENCES)

CONFIDENCE_MIN = 1
CONFIDENCE_MAX = 3
CONFIDENCE_LABELS = {1: "low", 2: "medium", 3: "high"}


# ---------------------------------------------------------------------------
# Defect taxonomy
# ---------------------------------------------------------------------------
# The first three are the coherence failures the ASP objective names
# explicitly ("must never lose a test by producing torn anatomy, duplicated
# strips, or misordered content"); the rest cover the artifact classes the
# benchmark's automated metrics claim to measure, so a human tag can be
# rank-correlated against the metric that is supposed to catch it (§0.2's
# still-open metric-calibration item).
DEFECTS = (
    ("torn_anatomy", "Torn anatomy", "a character/object is cut and rejoined out of register"),
    ("duplicated_strip", "Duplicated strip", "the same content appears twice along the scroll axis"),
    ("misordered_content", "Misordered content", "regions appear out of their source order"),
    ("ghosting", "Ghosting", "semi-transparent doubled content from averaged poses"),
    ("banding", "Banding", "periodic horizontal/vertical intensity bands"),
    ("seam_line", "Seam line", "a visible hard boundary between contributions"),
    ("color_shift", "Colour shift", "luminance/hue discontinuity across the composite"),
    ("crop_loss", "Crop loss", "intended content missing at a canvas edge"),
    ("blur", "Blur", "softening or loss of line-art detail"),
    ("geometry_warp", "Geometry warp", "straight lines bent, aspect distorted"),
    ("other", "Other", "described in the notes"),
)
DEFECT_KEYS = tuple(key for key, _, _ in DEFECTS)
DEFECT_TITLES = {key: title for key, title, _ in DEFECTS}

SEVERITY_MIN = 1
SEVERITY_MAX = 3
SEVERITY_LABELS = {
    1: "trace",
    2: "noticeable",
    3: "severe / decision-driving",
}


# ---------------------------------------------------------------------------
# Result identities (M0 — asp_change_roadmap_2026q3.md §5 M0, issue #24)
# ---------------------------------------------------------------------------
# What a stored rating/artifact actually *is*. "asp" as previously written in
# RatingEntry was ambiguous between "the ungated compositor output" and "the
# policy-selected safe result" — this is exactly the M0 problem: the
# 2026-08-07 corpus is majority-fallback, and averaging fallback scores in
# with true-composite scores as if they were the same population overstated
# ASP's real quality. These three identities must always be distinguishable.
RESULT_RAW_ASP = "raw_asp"  # the ungated ASP compositor result, always an artifact
RESULT_SAFE_ASP = "safe_asp"  # the policy-selected result; may equal raw_asp or a fallback
RESULT_SCANS = "scans"  # the OpenCV stitcher comparator/fallback candidate
RESULT_IDENTITIES = (RESULT_RAW_ASP, RESULT_SAFE_ASP, RESULT_SCANS)

# Safe ASP is a *policy selection*, not a fourth compositor — record what it
# actually picked, not just that a fallback happened.
SAFE_ASP_SOURCE_RAW = "raw_asp"  # policy kept the raw compositor result
SAFE_ASP_SOURCE_SCANS = "scans"  # policy substituted SCANS
SAFE_ASP_SOURCE_NAMED_FALLBACK = "named_fallback"  # a non-SCANS named fallback (e.g. panorama)
SAFE_ASP_SOURCES = (
    SAFE_ASP_SOURCE_RAW,
    SAFE_ASP_SOURCE_SCANS,
    SAFE_ASP_SOURCE_NAMED_FALLBACK,
)


# ---------------------------------------------------------------------------
# Case-level content safety (C0.5 — asp_sfw_corpus_roadmap_2026q3.md, issue #41)
# ---------------------------------------------------------------------------
# Objective content description — not a policy judgment. Multi-valued per
# case; extend, don't repurpose an existing tag for something it doesn't mean.
CONTENT_TAG_VIOLENCE = "violence"
CONTENT_TAG_GORE = "gore"
CONTENT_TAG_NUDITY_IMPLICIT = "nudity_implicit"
CONTENT_TAG_NUDITY_EXPLICIT = "nudity_explicit"
CONTENT_TAG_FANSERVICE = "fanservice"
CONTENT_TAG_DARK_THEMES = "dark_themes"  # mature/gray-morality content
CONTENT_TAGS = (
    CONTENT_TAG_VIOLENCE,
    CONTENT_TAG_GORE,
    CONTENT_TAG_NUDITY_IMPLICIT,
    CONTENT_TAG_NUDITY_EXPLICIT,
    CONTENT_TAG_FANSERVICE,
    CONTENT_TAG_DARK_THEMES,
)
# Deliberately open-ended (roadmap: "and others added as needed") — this is
# the known starting set, not an enforced closed enum. Don't reject an
# unrecognized tag on load; a human may add one before this list catches up.

# Named tiers, not a numeric score — an unlabelled composite number invites
# false precision the same way an unvalidated automated site score would.
SAFETY_TIER_G = "tier_g"  # no tags beyond fully benign
SAFETY_TIER_PG13 = "tier_pg13"  # mild fanservice/violence, no explicit content
SAFETY_TIER_MATURE_SFW = "tier_mature_sfw"  # suggestive, dark themes, not explicit
SAFETY_TIER_NSFW = "tier_nsfw"  # explicit
SAFETY_TIERS = (
    SAFETY_TIER_G,
    SAFETY_TIER_PG13,
    SAFETY_TIER_MATURE_SFW,
    SAFETY_TIER_NSFW,
)

# The minor-presenting hard floor's per-assessor verdict (human or
# automated). "high_risk" is never emitted by an unvalidated automated
# ensemble member — see C0.5's decision rule — but the verdict enum itself
# doesn't encode that restriction; the caller does.
MINOR_RISK_HIGH_RISK = "high_risk"  # permanent hard drop regardless of source
MINOR_RISK_CLEAR = "clear"
MINOR_RISK_UNCERTAIN = "uncertain"  # never a veto, never stored as high_risk
MINOR_RISK_VERDICTS = (MINOR_RISK_HIGH_RISK, MINOR_RISK_CLEAR, MINOR_RISK_UNCERTAIN)
