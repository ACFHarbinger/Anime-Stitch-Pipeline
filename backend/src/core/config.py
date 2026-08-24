"""
§1.8A: TOML-based ASP pipeline configuration loader.

Loads ``backend/config/asp_config.toml`` (or a path supplied by the caller).
Any key present in the TOML file is exported as an environment variable (via
``os.environ.setdefault``), so all downstream ``os.environ.get`` calls in
pipeline modules pick up the value automatically.
Existing env-var values always win over the config file.

Usage::

    from asp_backend.core.config import load_asp_config
    load_asp_config()  # reads backend/config/asp_config.toml

Example ``asp_config.toml``::

    [frame_selection]
    ASP_NEAR_DUP_LUMA = 5.0
    ASP_HOLD_THRESHOLD = 0.03

    [compositing]
    ASP_SP_SOFT_PX = 6
    ASP_GRAPHCUT_SEAM = 1

    [pipeline]
    ASP_COV_MIN_MULTI_PCT = 0.30

All keys are optional; unrecognised keys are silently accepted (forwarded as
env vars).  Values must be numeric (int/float) or boolean — TOML strings are
forwarded as-is.
"""

from __future__ import annotations

import os
import tomllib
import warnings
from pathlib import Path
from typing import Any

from backend.src.errors import ConfigError

__all__ = ["load_asp_config", "validate_asp_config", "dump_asp_config", "get_asp", "asp_schema"]

# Resolved relative to this file: backend/src/animation/core/ → up 4 → backend/config/
_DEFAULT_CONFIG_NAME = Path(__file__).resolve().parent.parent.parent.parent / "config" / "asp_config.toml"

# §1.8B — Schema for known ASP env-var keys.
# Tuple: (expected_type, min_val, max_val, description)
# min_val / max_val are None when the bound is open.
_CONFIG_SCHEMA: dict[str, tuple] = {
    # ── Reproducibility ─────────────────────────────────────────────────
    "ASP_DETERMINISTIC": (
        int,
        0,
        1,
        "Pin Python/NumPy/OpenCV/Torch RNGs and deterministic runtime kernels",
    ),
    "ASP_REPRO_SEED": (int, 0, 2**63 - 1, "Seed used when ASP_DETERMINISTIC=1"),
    # ── Frame selection ─────────────────────────────────────────────────
    "ASP_HOLD_THRESHOLD": (float, 0.0, 1.0, "MAD hold-detection threshold [0, 1]"),
    "ASP_HOLD_DHASH_THRESH": (int, 0, 64, "dHash Hamming floor for hold detection (0=off)"),
    "ASP_DHASH_EXACT_DROP": (int, 0, 1, "Drop exact dHash duplicates before selection"),
    "ASP_HIGH_HOLD_RESPONSE": (float, 0.0, 1.0, "phaseCorrelate response floor for hold merge"),
    "ASP_HOLD_AVERAGE": (int, 0, 1, "Overmix-style ECC sub-pixel averaging within hold blocks"),
    "ASP_HOLD_BG_SUB": (int, 0, 1, "EXPERIMENTAL, default-off: unaligned-median background plate for hold detection; not validated under camera pan (M4 owns keep/delete)"),
    "ASP_BLUR_REJECT_THRESH": (float, 0.0, None, "Laplacian-variance floor for blurry-frame rejection (0=off)"),
    "ASP_CONTRAST_THRESH": (float, 0.0, None, "Pixel-std floor for low-contrast frame rejection (0=off)"),
    "ASP_NEAR_DUP_LUMA": (float, 0.0, 255.0, "Near-dup luma dedup ceiling (0=off)"),
    "ASP_TEMPORAL_VAR_THRESH": (float, 0.0, 1.0, "Static-frame temporal variance floor (0=off)"),
    "ASP_SHOT_BOUNDARY_THRESH": (float, 0.0, 1.0, "Stage-1 histogram shot-boundary threshold (0=off)"),
    "ASP_OTSU_BG_CORR": (int, 0, 1, "Per-pair Otsu bg mask for phase correlation"),
    "ASP_TWO_CHANNEL_SELECT": (int, 0, 1, "BiRefNet two-channel camera/animation selection (experimental)"),
    "ASP_POSE_WINDOW_PX": (int, 0, None, "DINOv2 pose-consistent selection window (0=off, experimental)"),
    "ASP_PHASE_AWARE_SELECT": (int, 0, 1, "§2.4 Pass-2 bias toward same-phase candidates (default OFF pending A/B)"),
    "ASP_PHASE_CROSS_PENALTY": (float, 0.0, 1.0, "§2.4 tie-break penalty applied to a cross-phase Pass-2 candidate"),
    # ── Video ingestion ─────────────────────────────────────────────────
    "ASP_VIDEO_MAX_FRAMES": (int, 1, None, "Max frames decoded from a video input"),
    "ASP_VIDEO_PROXY_SCALE": (float, 0.05, 1.0, "Proxy decode scale for the selection pass"),
    "ASP_VIDEO_TELECINE_MAD": (float, 0.0, None, "Telecine duplicate MAD threshold"),
    "ASP_VIDEO_KEYFRAMES_ONLY": (int, 0, 1, "Decode only keyframes in the proxy pass"),
    # ── Masking ─────────────────────────────────────────────────────────
    "ASP_USE_SAM2": (int, 0, 1, "Use SAM-2 video predictor instead of BiRefNet"),
    "ASP_TRAPPED_BALL": (int, 0, 1, "Classical deterministic trapped-ball line-art segmentation (Zhang 2009) for background masking"),
    "ASP_TRAPPED_BALL_RADIUS": (int, 1, 32, "Trapped-ball structuring element radius (px) for closing line-art gaps"),
    # ── Matching / alignment ────────────────────────────────────────────
    "ASP_MATCH_SPREAD_CEIL": (float, 0.0, None, "Max MAD of per-match displacements (0=off)"),
    "ASP_LOFTR_BG_RATIO_MIN": (float, 0.0, 1.0, "Min fraction of LoFTR matches on background (0=off)"),
    "ASP_SIMILARITY_MODE": (int, 0, 1, "4-DOF similarity constraint for per-pair affines"),
    "ASP_ALIGN_GATE_DX": (float, 0.0, None, "75th-pct |dx| gate for vertical-scroll alignment"),
    "ASP_CLEANCP_RESOLVE": (int, 0, 1, "Default-off CleanCP local edge recovery after filtering disconnects the graph"),
    "ASP_BA_F_SCALE": (float, 0.0, None, "Cauchy loss f_scale (px) in bundle adjustment"),
    "ASP_GNC_OUTER": (int, 1, 32, "GNC outer continuation iterations in BA"),
    "ASP_DY_CV_MAX": (float, 0.0, None, "dy_cv gate: SCANS fallback above this step-CV (0=off)"),
    # ── Foreground registration (Stage 8.5) ─────────────────────────────
    "ASP_FG_REGISTER": (int, 0, 1, "Enable Stage 8.5 foreground pose registration"),
    "ASP_FLOW_ENGINE": (str, None, None, "Dense flow engine: searaft | dis"),
    "ASP_ARAP_PUSH": (int, 0, 1, "ARAP Push phase before Regularise"),
    "ASP_FG_MAX_RESIDUAL": (float, 0.0, None, "Max animation residual (px) to warp; above → single-pose"),
    # ── Rendering (Stage 10) ─────────────────────────────────────────────
    "ASP_FG_EXCLUDE_MEDIAN": (int, 0, 1, "Foreground-excluded temporal median (A5)"),
    "ASP_BG_AVERAGE": (int, 0, 1, "§2.5 Overmix-style mean/median blend for confirmed-bg samples (default OFF pending A/B; fixed 2026-07-27, was measured harmful with an abrupt switch)"),
    "ASP_BG_AVERAGE_FULL_AT": (int, 3, None, "§2.5 sample count at which the mean/median blend reaches full mean weight (default 5)"),
    "ASP_MASKED_MEDIAN": (int, 0, 1, "Leave always-fg pixels black instead of ghost-averaging"),
    "ASP_ADAPTIVE_RENDER_GAIN": (int, 0, 1, "Adaptive gain clamp in sequential render normalisation"),
    "ASP_GAIN_DRIFT_MAX": (float, 0.0, None, "Max cumulative gain fold-change before reset (0=off)"),
    "ASP_GPU_MEDIAN": (int, 0, 1, "GPU temporal median via base (UMat)"),
    "ASP_COV_MIN_MULTI_PCT": (float, 0.0, 1.0, "Min multi-frame canvas coverage before SCANS fallback"),
    # ── Compositing (Stage 11) ───────────────────────────────────────────
    "ASP_PHASE_COMPOSITE": (int, 0, 1, "§2.3 phase-consistent compositing: never midpoint-warp across an animation-phase boundary, escalate to single-pose instead (default OFF pending A/B)"),
    "ASP_COHERENCE_V2": (int, 0, 1, "M3 default-off §9.2 Stage 2 compositor candidate: one pose per foreground overlap region; not wired into the live seam loop"),
    "ASP_PLATE_SINGLE_POSE": (int, 0, 1, "P1 default-off candidate: canvas-aligned background plate + single foreground pose per connected region"),
    "ASP_PLATE_MULTIBAND": (int, 0, 1, "P2 default-off candidate: multi-band pyramid blending on background plate transitions"),
    "ASP_PLATE_EDGE_PRESERVE": (int, 0, 1, "P2 edge-preserving sharp background source selection in multi-sample zones"),
    "ASP_GRAPHCUT_SEAM": (int, 0, 1, "GraphCut global multi-image seam (§4.2; default OFF — measured worse seam_visibility than DP path)"),
    "ASP_GC_FEATHER_PX": (int, 0, None, "Feather width at GraphCut ownership boundaries"),
    "ASP_BLOCKS_GAIN_COMP": (int, 0, 1, "32×32 blocks BGR gain compensation in blend zones (§4.1)"),
    "ASP_BLOCKS_LUM_COMP": (int, 0, 1, "LAB-L blocks gain compensation in blend zones (§4.4)"),
    "ASP_GLOBAL_GAIN_COMP": (int, 0, 1, "Pre-seam sequential global gain equalization (§4.10)"),
    "ASP_JOINT_GAIN_SOLVE": (int, 0, 1, "§3.1 Brown-Lowe joint gain solve, replaces §4.10's sequential chain (default OFF pending A/B)"),
    "ASP_JOINT_GAIN_SIGMA_N": (float, 0.01, None, "§3.1 joint gain solve noise sigma (default 10.0)"),
    "ASP_JOINT_GAIN_SIGMA_G": (float, 0.001, None, "§3.1 joint gain solve gain-prior sigma (default 0.1)"),
    "ASP_JOINT_GAIN_ROBUST": (int, 0, 1, "§3.1 reject isolated overlap luminance ratios before joint gain solve"),
    "ASP_SP_SOFT_PX": (int, 0, None, "Single-pose soft-edge half-width (px)"),
    "ASP_BG_NORM_MIN_PX": (int, 0, None, "Min bg pixels for normalisation gain estimate (0=200)"),
    "ASP_POST_SEAM_WARN_THRESH": (float, 0.0, None, "Post-composite seam lum-step warning threshold"),
    # ── C++ acceleration ─────────────────────────────────────────────────
    "ASP_BATCH_GPU": (int, 0, 1, "GPU dispatch for C++ base kernels"),
    # ── Bundle adjustment (previously hardcoded module constants) ────────
    "ASP_ST_INLIER_THRESHOLD": (float, 0.0, None, "§1.1B max allowed disagreement (px) vs spanning-tree reference"),
    # ── Affine validation (previously hardcoded module constants) ────────
    "ASP_ROT_TIGHT": (float, 0.0, None, "§0.5D tight rotation threshold (high frame-to-frame variance)"),
    "ASP_ROT_LOOSE": (float, 0.0, None, "§0.5D loose rotation threshold (near-identical camera rotation)"),
    "ASP_SC_TIGHT": (float, 0.0, None, "§0.5D tight scale threshold (high frame-to-frame variance)"),
    "ASP_SC_LOOSE": (float, 0.0, None, "§0.5D loose scale threshold (near-identical camera scale)"),
    "ASP_MONO_TAU_MIN": (float, 0.0, 1.0, "§1.12 min |Kendall tau| for translation-monotonicity check"),
    "ASP_ROT_SCALE_CONSISTENCY_THRESH": (float, 0.0, None, "Rotation/scale consistency threshold for adaptive tight/loose selection"),
    # ── Frame selection: hold detection / pose refinement (previously
    # hardcoded module constants) ─────────────────────────────────────────
    "ASP_MAX_SKIPPABLE_HOLD_SIZE": (int, 1, None, "Max hold-block size (frames) treated as a real animation hold, not a false positive"),
    "ASP_POSE_REFINE_LOOK_RANGE": (int, 0, None, "Pass-2 pose refinement: +-N slot search window"),
    "ASP_POSE_REFINE_MIN_GAIN": (float, 0.0, 1.0, "Pass-2 pose refinement: min pose-similarity improvement to swap candidates"),
    "ASP_POSE_REFINE_MIN_ADV_FRAC": (float, 0.0, None, "Pass-2 pose refinement: min frame-advance fraction constraint"),
    "ASP_POSE_REFINE_MAX_ADV_FRAC": (float, 0.0, None, "Pass-2 pose refinement: max frame-advance fraction constraint"),
    "ASP_POSE_REFINE_SAME_HOLD_PENALTY": (float, 0.0, 1.0, "Pass-2 pose refinement: similarity penalty for staying within the same hold block"),
    "ASP_POSE_PATH_SELECT": (int, 0, 1, "Experimental global dynamic-programming pose path selection"),
    "ASP_POSE_PATH_SAFE": (int, 0, 1, "Reject experimental pose paths with structural-risk diagnostics"),
}


def validate_asp_config(
    config: dict[str, Any],
    *,
    strict: bool = False,
) -> list[str]:
    """§1.8B: Validate a flat ASP config dict against ``_CONFIG_SCHEMA``.

    Args:
        config: Flat mapping of ASP key → value (as returned by `load_asp_config`).
        strict: When True, raises `ConfigError` listing all violations instead of
            returning them. Use in CI/scripting contexts where a misconfigured
            experiment should abort immediately.

    Returns:
        Violation messages as a list of strings. Empty list means the config is valid.

    Note:
        Unknown keys (not in ``_CONFIG_SCHEMA``) emit a `UserWarning` but are not
        counted as violations — forward-compatibility is preserved so that configs
        written for a newer pipeline version still load on an older one. TOML
        integers are accepted where float is expected.

    Example:
        >>> cfg = {"ASP_HOLD_THRESHOLD": 0.03, "ASP_SP_SOFT_PX": 6}
        >>> validate_asp_config(cfg)
        []
    """
    violations: list[str] = []

    for key, val in config.items():
        if key not in _CONFIG_SCHEMA:
            warnings.warn(
                f"[ASP config] Unknown key {key!r} — not in schema; forwarded as-is.",
                UserWarning,
                stacklevel=2,
            )
            continue

        expected_type, lo, hi, desc = _CONFIG_SCHEMA[key]

        # Allow int where float is expected (TOML integers → Python int)
        is_type_ok = isinstance(val, expected_type) or (
            expected_type is float and isinstance(val, int)
        )
        if not is_type_ok:
            violations.append(
                f"{key}: expected {expected_type.__name__}, "
                f"got {type(val).__name__} ({val!r}). Hint: {desc}"
            )
            continue

        numeric_val = float(val) if expected_type is float else int(val)
        if lo is not None and numeric_val < lo:
            violations.append(f"{key}={val!r} is below minimum {lo}. Hint: {desc}")
        if hi is not None and numeric_val > hi:
            violations.append(f"{key}={val!r} exceeds maximum {hi}. Hint: {desc}")

    if strict and violations:
        raise ConfigError(
            "ASP config validation failed:\n"
            + "\n".join(f"  • {v}" for v in violations)
        )

    return violations


def load_asp_config(
    path: str | None = None,
    *,
    override_env: bool = True,
    validate: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Load ASP pipeline configuration from a TOML file.

    Reads ``asp_config.toml`` (or the file at *path*) and merges all sections
    into a flat dict. Each key is written to ``os.environ`` via ``setdefault``
    so that all downstream ``os.environ.get("ASP_*")`` calls pick it up
    automatically. Existing environment variables always take precedence.

    Args:
        path: Path to the TOML file. Defaults to ``backend/config/asp_config.toml``
            relative to the package root. Returns an empty dict silently if the
            file does not exist.
        override_env: When True (default), write each loaded key to
            ``os.environ`` via ``setdefault``. Set to False to dry-run the
            load without touching the environment (useful for testing).
        validate: When True, run `validate_asp_config` on the loaded dict
            before writing env vars. Invalid keys emit warnings (or raise if
            *strict* is also True).
        strict: Passed to `validate_asp_config` when *validate* is True.
            Raises `ConfigError` on the first batch of violations.

    Returns:
        Flat mapping of all keys found in the TOML file (sections merged).
        Empty dict if the file is absent or contains no section data.

    Example:
        >>> import os, tempfile, pathlib
        >>> toml = b"[frame_selection]\\nASP_HOLD_THRESHOLD = 0.03\\n"
        >>> with tempfile.NamedTemporaryFile(suffix='.toml', delete=False) as f:
        ...     _ = f.write(toml); name = f.name
        >>> cfg = load_asp_config(name, override_env=False)
        >>> cfg['ASP_HOLD_THRESHOLD']
        0.03
    """
    config_path = Path(path) if path is not None else Path(_DEFAULT_CONFIG_NAME)
    if not config_path.exists():
        return {}

    with open(config_path, "rb") as fh:
        raw: dict[str, Any] = tomllib.load(fh)

    flat: dict[str, Any] = {}
    for value in raw.values():
        if isinstance(value, dict):
            flat.update(value)

    if not flat:
        return {}

    if validate:
        validate_asp_config(flat, strict=strict)

    if override_env:
        for key, val in flat.items():
            if isinstance(val, bool):
                os.environ.setdefault(key, "1" if val else "0")
            else:
                os.environ.setdefault(key, str(val))

    return flat


def asp_schema() -> dict[str, tuple]:
    """Return a copy of the ASP config schema (§1.8B).

    Exposes ``_CONFIG_SCHEMA`` publicly so callers outside this module (e.g.
    the unified ``AppConfig`` snapshot, §5.14C) can enumerate every known
    ``ASP_*`` key, its expected type, and its valid range without reaching
    into a private module attribute. Each value is
    ``(expected_type, min_val, max_val, description)``.
    """
    return dict(_CONFIG_SCHEMA)


def get_asp(key: str, default: str = "") -> str:
    """Return an ASP pipeline env-var, falling back to *default*.

    Prefer this over bare ``os.environ.get("ASP_*")`` calls because it
    guarantees the default is consistent with the schema and makes call-sites
    greppable via a single name.

    Args:
        key: The ``ASP_*`` environment variable name (e.g. ``"ASP_HOLD_THRESHOLD"``).
        default: String default returned when the key is absent from the environment.
            Callers that need a non-string type should cast the return value::

                threshold = float(get_asp("ASP_HOLD_THRESHOLD", "0.025"))
                enabled   = get_asp("ASP_POISSON_SEAM", "0") != "0"

    Returns:
        The env-var value or *default*.
    """
    return os.environ.get(key, default)


# Logical section groupings for the TOML dump (§1.8C).
# Maps section header → list of ASP env-var key prefixes belonging to it.
_DUMP_SECTIONS: dict[str, list[str]] = {
    "frame_selection": [
        "ASP_HOLD_THRESHOLD", "ASP_HOLD_DHASH_THRESH", "ASP_DHASH_EXACT_DROP",
        "ASP_HIGH_HOLD_RESPONSE", "ASP_HOLD_AVERAGE", "ASP_HOLD_BG_SUB", "ASP_BLUR_REJECT_THRESH",
        "ASP_CONTRAST_THRESH", "ASP_NEAR_DUP_LUMA", "ASP_TEMPORAL_VAR_THRESH",
        "ASP_OTSU_BG_CORR", "ASP_TWO_CHANNEL_SELECT", "ASP_POSE_WINDOW_PX",
        "ASP_PHASE_AWARE_SELECT", "ASP_PHASE_CROSS_PENALTY",
    ],
    "video": [
        "ASP_VIDEO_MAX_FRAMES", "ASP_VIDEO_PROXY_SCALE",
        "ASP_VIDEO_TELECINE_MAD", "ASP_VIDEO_KEYFRAMES_ONLY",
    ],
    "masking": ["ASP_USE_SAM2"],
    "alignment": [
        "ASP_MATCH_SPREAD_CEIL", "ASP_LOFTR_BG_RATIO_MIN", "ASP_SIMILARITY_MODE",
        "ASP_ALIGN_GATE_DX", "ASP_CLEANCP_RESOLVE", "ASP_BA_F_SCALE", "ASP_GNC_OUTER", "ASP_DY_CV_MAX",
    ],
    "fg_register": [
        "ASP_FG_REGISTER", "ASP_FLOW_ENGINE", "ASP_ARAP_PUSH", "ASP_FG_MAX_RESIDUAL",
    ],
    "rendering": [
        "ASP_FG_EXCLUDE_MEDIAN", "ASP_BG_AVERAGE", "ASP_BG_AVERAGE_FULL_AT", "ASP_MASKED_MEDIAN", "ASP_ADAPTIVE_RENDER_GAIN",
        "ASP_GAIN_DRIFT_MAX", "ASP_GPU_MEDIAN", "ASP_COV_MIN_MULTI_PCT",
    ],
    "compositing": [
        "ASP_PHASE_COMPOSITE", "ASP_COHERENCE_V2", "ASP_GRAPHCUT_SEAM", "ASP_GC_FEATHER_PX", "ASP_BLOCKS_GAIN_COMP",
        "ASP_BLOCKS_LUM_COMP", "ASP_GLOBAL_GAIN_COMP", "ASP_JOINT_GAIN_SOLVE",
        "ASP_JOINT_GAIN_SIGMA_N", "ASP_JOINT_GAIN_SIGMA_G", "ASP_JOINT_GAIN_ROBUST", "ASP_SP_SOFT_PX",
        "ASP_BG_NORM_MIN_PX", "ASP_POST_SEAM_WARN_THRESH",
    ],
    "acceleration": ["ASP_BATCH_GPU"],
    "bundle_adjust": ["ASP_ST_INLIER_THRESHOLD"],
    "validation": [
        "ASP_ROT_TIGHT", "ASP_ROT_LOOSE", "ASP_SC_TIGHT", "ASP_SC_LOOSE",
        "ASP_MONO_TAU_MIN", "ASP_ROT_SCALE_CONSISTENCY_THRESH",
    ],
    "pose_refine": [
        "ASP_MAX_SKIPPABLE_HOLD_SIZE", "ASP_POSE_REFINE_LOOK_RANGE",
        "ASP_POSE_REFINE_MIN_GAIN", "ASP_POSE_REFINE_MIN_ADV_FRAC",
        "ASP_POSE_REFINE_MAX_ADV_FRAC", "ASP_POSE_REFINE_SAME_HOLD_PENALTY",
        "ASP_POSE_PATH_SELECT",
        "ASP_POSE_PATH_SAFE",
    ],
}


def dump_asp_config(
    path: str | None = None,
    *,
    include_defaults: bool = False,
) -> str:
    """§1.8C/D: Serialize the current ASP env-var state to a TOML file (S126/S131).

    Reads all known ``ASP_*`` env-var keys from ``os.environ`` and writes them
    to a TOML file grouped into logical sections (frame_selection, compositing,
    bundle_adjust, pipeline).  Only keys that are currently set in the environment
    are written by default; pass ``include_defaults=True`` to emit all schema keys
    with their default values (``"0"`` for most flags, or the built-in default).

    §1.8D enhancement (S131): each key is preceded by two comment lines:
    *   ``# type: <typename>  range: [min, max]`` — machine-readable constraint
        annotation (``min``/``max`` are ``None`` when unbounded).
    *   ``# <description>`` — human-readable explanation from ``_CONFIG_SCHEMA``.

    These comments survive round-trip through a TOML editor and let tools
    validate the file against the schema without having to import the Python module.

    This is the inverse of :func:`load_asp_config`: it lets you capture the
    current tuning state of a successful run and save it as a reproducible config
    file for future experiments.

    Args:
        path: Destination TOML file path. Defaults to ``backend/config/asp_config.toml``
            relative to the package root. Parent directories are created if needed.
        include_defaults: When True, all schema keys are emitted (with ``"0"`` as
            the fallback default for unset keys). When False (default), only keys
            that are explicitly set in the environment are written.

    Returns:
        Absolute path to the written file.
    """
    out_path = Path(path) if path is not None else Path(_DEFAULT_CONFIG_NAME)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# ASP pipeline configuration — generated by dump_asp_config()",
        f"# Generated at: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    written_keys: set = set()

    for section, keys in _DUMP_SECTIONS.items():
        section_lines: list[str] = []
        for key in keys:
            env_val = os.environ.get(key)
            if env_val is None and not include_defaults:
                continue
            val_str = env_val if env_val is not None else "0"
            # Preserve numeric type: if the value looks like a float with decimal
            # point keep it; if it looks like a plain integer, omit quotes.
            try:
                toml_val = str(float(val_str)) if "." in val_str else str(int(val_str))
            except ValueError:
                toml_val = f'"{val_str}"'
            schema_entry = _CONFIG_SCHEMA.get(key)
            if schema_entry is not None:
                _typ, _lo, _hi, hint = schema_entry
                # §1.8D: emit machine-readable type/range annotation first.
                _type_name = (
                    getattr(_typ, "__name__", str(_typ)) if _typ is not None else "str"
                )
                _range_str = f"[{_lo}, {_hi}]"
                section_lines.append(f"# type: {_type_name}  range: {_range_str}")
                if hint:
                    section_lines.append(f"# {hint}")
            section_lines.append(f"{key} = {toml_val}")
            written_keys.add(key)

        if section_lines:
            lines.append(f"[{section}]")
            lines.extend(section_lines)
            lines.append("")

    # Emit any env-set ASP_* keys not covered by _DUMP_SECTIONS under [extra].
    extra_lines: list[str] = []
    for key, val in os.environ.items():
        if key.startswith("ASP_") and key not in written_keys:
            try:
                toml_val = str(float(val)) if "." in val else str(int(val))
            except ValueError:
                toml_val = f'"{val}"'
            extra_lines.append(f"{key} = {toml_val}")
    if extra_lines:
        lines.append("[extra]")
        lines.extend(extra_lines)
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path.resolve())
