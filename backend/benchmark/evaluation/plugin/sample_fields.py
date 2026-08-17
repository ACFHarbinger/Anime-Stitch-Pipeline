"""Maps one benchmark dataset + its human evaluation into flat FiftyOne sample
payloads — one payload per comparator image.

Kept free of any ``fiftyone`` import so the mapping is unit-testable without a
running MongoDB, and so a machine without FiftyOne installed can still exercise
the data layer. ``ingest.py`` turns these dicts into ``fo.Sample`` objects.

Every field here exists to be *filtered and sorted on* in the App sidebar: the
point of the triage surface is questions like "show me every test where the
metric verdict says asp_better but the human rated ASP below Simple", or "every
seam_vis_gate fallback with GT-SSIM under 0.65", which is precisely what the old
tool made impossible by showing one test at a time with no metrics at all.
"""

from __future__ import annotations

from ..constants.schema import (
    DIMENSION_KEYS,
    IMAGE_ASP,
    PRIMARY_KEYS,
)
from ..other import metrics_view as mv
from ..other.schema import RatingEntry

# Metrics copied onto each comparator's own sample, so sorting the grid by
# `sharpness` sorts within the currently-selected slice.
_PER_IMAGE_METRICS = (
    "sharpness",
    "ghosting_siqe",
    "seam_visibility",
    "seam_coherence",
    "seam_gradient",
    "edge_energy_score",
    "coverage",
    "color_entropy",
    "ghost_seam_max",
    "width",
    "height",
)
_GT_METRICS = ("ssim_vs_gt", "aligned_ssim_vs_gt", "psnr_vs_gt")


def _num(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_fields(entry: dict) -> dict:
    """Fields shared by every comparator of one test."""
    comparison = entry.get("comparison") or {}
    gt = entry.get("ground_truth") or {}
    frames = entry.get("frames") or {}
    phases = entry.get("phases") or {}
    time = entry.get("time") or {}
    alignment = entry.get("alignment") or {}
    selection = entry.get("frame_selection") or {}
    health = entry.get("affine_health") or {}
    photometric = entry.get("photometric") or {}
    matching = entry.get("matching") or {}

    original = _num(selection.get("original_count"))
    final = _num(selection.get("final_count"))
    dropped_pct = (
        100.0 * (original - final) / original if original and final is not None else None
    )
    fallback_reason = entry.get("fallback_reason") or ""
    return {
        "verdict": comparison.get("verdict"),
        "verdict_source": comparison.get("verdict_source"),
        "gt_verdict": gt.get("verdict"),
        "has_ground_truth": bool(gt.get("available")),
        "used_fallback": bool(entry.get("used_fallback")),
        "fallback_reason": fallback_reason,
        # The gate name alone, so the sidebar offers ~6 discrete values to
        # filter on instead of 97 unique reason strings with embedded numbers.
        "fallback_gate": fallback_reason.split(":", 1)[0] if fallback_reason else "",
        "pair_ssim": _num(comparison.get("ssim")),
        "pair_psnr_db": _num(comparison.get("psnr_db")),
        "frames_count": _num(frames.get("count")),
        "source_width": _num(frames.get("source_w")),
        "source_height": _num(frames.get("source_h")),
        "phases_count": _num(phases.get("count")),
        "total_sec": _num(time.get("total_sec")),
        "mean_post_warp_diff": _num(entry.get("mean_post_warp_diff")),
        "dy_cv": _num(alignment.get("dy_cv")),
        "dx_cv": _num(alignment.get("dx_cv")),
        "affine_valid": bool(health.get("valid")) if health else None,
        "affine_reason": health.get("reason"),
        "selection_mode": selection.get("selection_mode"),
        "frames_dropped_pct": dropped_pct,
        "frames_corrected": _num(photometric.get("frames_corrected")),
        "raw_edges": _num(matching.get("raw_edges")),
        "filtered_edges": _num(matching.get("filtered_edges")),
    }


def human_fields(evaluation: RatingEntry | None) -> dict:
    """Human judgment as flat fields, prefixed so they group together in the
    sidebar and can never collide with a metric name."""
    if evaluation is None:
        return {
            "human_rated": False,
            "human_reviewed": False,
            "human_skipped": False,
            "human_asp": None,
            "human_simple": None,
            "human_preference": None,
            "human_confidence": None,
            "human_notes": "",
            "human_defects": [],
            "human_disagrees_with_metric": None,
        }
    fields = {
        "human_rated": evaluation.is_rated(),
        "human_reviewed": evaluation.reviewed,
        "human_skipped": evaluation.skipped,
        "human_asp": evaluation.asp,
        "human_simple": evaluation.simple,
        "human_preference": evaluation.preference,
        "human_confidence": evaluation.confidence,
        "human_notes": evaluation.notes,
        "human_defects": list(evaluation.defects),
    }
    for image in PRIMARY_KEYS:
        for dimension in DIMENSION_KEYS:
            value = evaluation.score(image, dimension)
            if value is not None:
                fields[f"human_{image}_{dimension}"] = value
    return fields


def disagreement(entry: dict, evaluation: RatingEntry | None) -> bool | None:
    """Whether the human's preference contradicts the recorded verdict.

    This is the single most valuable derived field on the triage surface: §0.2's
    open metric-calibration item is exactly "which automated metrics disagree
    with humans", and the coherence veto only fires on the ``asp_better``
    direction, so the disagreements it does *not* veto are otherwise invisible.
    """
    if evaluation is None or not evaluation.is_rated():
        return None
    verdict = ((entry.get("comparison") or {}).get("verdict")) or ""
    if evaluation.asp > evaluation.simple:
        human = "asp_better"
    elif evaluation.simple > evaluation.asp:
        human = "simple_better"
    else:
        human = "comparable"
    if verdict not in ("asp_better", "simple_better", "comparable"):
        return None
    return human != verdict


def sample_tags(entry: dict, evaluation: RatingEntry | None) -> list[str]:
    """Tags are FiftyOne's fastest filter, so the handful of facts a triage pass
    slices on most go here as well as into fields."""
    tags: list[str] = []
    comparison = entry.get("comparison") or {}
    if comparison.get("verdict"):
        tags.append(f"verdict:{comparison['verdict']}")
    if entry.get("used_fallback"):
        tags.append("fallback")
        gate = (entry.get("fallback_reason") or "").split(":", 1)[0]
        if gate:
            tags.append(f"gate:{gate}")
    else:
        tags.append("true_composite")
    if (entry.get("ground_truth") or {}).get("available"):
        tags.append("has_gt")
    if evaluation is None or not evaluation.is_rated():
        tags.append("unrated")
    else:
        tags.append("rated")
        tags.append(f"human_asp:{evaluation.asp}")
        tags.append(f"human_simple:{evaluation.simple}")
        if evaluation.preference:
            tags.append(f"prefers:{evaluation.preference}")
    if evaluation is not None:
        # Outside the rated branch on purpose: a defect tag is real information
        # whether or not the test has been scored yet, and tagging in the App
        # then pushing back is exactly how an unscored test acquires one. Keeping
        # this inside the `else` above silently dropped those tags on the next
        # push.
        for defect in evaluation.defects:
            tags.append(f"defect:{defect}")
    if evaluation is not None and evaluation.skipped and not evaluation.is_rated():
        tags.append("skipped")
    if disagreement(entry, evaluation):
        tags.append("human_disagrees")
    return tags


def image_metric_fields(entry: dict, image_key: str) -> dict:
    """The comparator's own no-reference metrics, plus its GT metrics when it is
    one of the two the pipeline scores against ground truth."""
    block = entry.get(mv.METRICS_BLOCK.get(image_key, ""), {}) or {}
    fields = {name: _num(block.get(name)) for name in _PER_IMAGE_METRICS}
    if image_key in PRIMARY_KEYS:
        gt = entry.get("ground_truth") or {}
        gt_block = gt.get("metrics_asp" if image_key == IMAGE_ASP else "metrics_simple") or {}
        for name in _GT_METRICS:
            fields[name] = _num(gt_block.get(name))
    return fields


def build_payloads(
    name: str,
    entry: dict,
    paths: dict[str, str],
    evaluation: RatingEntry | None,
) -> list[tuple[str, dict]]:
    """One ``(comparator_key, fields)`` pair per available comparator image.

    ``fields`` carries no ``filepath`` — the caller owns that, since it also
    owns the FiftyOne group element the sample belongs to.
    """
    shared = run_fields(entry)
    shared.update(human_fields(evaluation))
    shared["human_disagrees_with_metric"] = disagreement(entry, evaluation)
    shared["dataset_name"] = name
    tags = sample_tags(entry, evaluation)

    payloads = []
    for image_key, path in paths.items():
        fields = dict(shared)
        fields.update(image_metric_fields(entry, image_key))
        fields["comparator_key"] = image_key
        fields["source_path"] = path
        # Only the scored comparators carry a human score of their own; ground
        # truth is a reference and never rated.
        if evaluation is not None:
            own = evaluation.score(image_key)
            fields["human_score"] = own
        else:
            fields["human_score"] = None
        fields["_tags"] = list(tags)
        payloads.append((image_key, fields))
    return payloads


def bbox_detections(evaluation: RatingEntry | None, image_key: str) -> list[dict]:
    """The user's tagged defect regions for one comparator, in FiftyOne's
    ``[x, y, w, h]`` normalized bounding-box convention — which is already the
    convention ``BoundingBox`` stores, so no conversion is needed.

    Round-tripping these means a region drawn in the inspector is visible and
    filterable in the FiftyOne grid, which is how the two surfaces stay one
    tool rather than two.
    """
    if evaluation is None:
        return []
    detections = []
    for box in evaluation.bboxes:
        if box.image != image_key:
            continue
        detections.append({
            "label": box.defect or "untagged",
            "bounding_box": [box.x, box.y, box.w, box.h],
            "severity": box.severity,
            "note": box.label,
        })
    return detections


NUMERIC_FIELDS = tuple(_PER_IMAGE_METRICS) + _GT_METRICS + (
    "pair_ssim", "pair_psnr_db", "frames_count", "source_width", "source_height",
    "phases_count", "total_sec", "mean_post_warp_diff", "dy_cv", "dx_cv",
    "frames_dropped_pct", "frames_corrected", "raw_edges", "filtered_edges",
)
_INT_FIELDS = ("human_asp", "human_simple", "human_confidence", "human_score") + tuple(
    f"human_{image}_{dimension}"
    for image in PRIMARY_KEYS
    for dimension in DIMENSION_KEYS
)
_BOOL_FIELDS = (
    "has_ground_truth", "used_fallback", "affine_valid",
    "human_rated", "human_reviewed", "human_skipped", "human_disagrees_with_metric",
)
_STR_FIELDS = (
    "dataset_name", "comparator_key", "verdict", "verdict_source", "gt_verdict",
    "fallback_reason", "fallback_gate", "affine_reason", "selection_mode",
    "human_preference", "human_notes",
)
_STR_LIST_FIELDS = ("human_defects",)

# (field name, one of float/int/bool/str/strlist). Declared explicitly at ingest
# rather than left to FiftyOne's implied-type inference, for two reasons: a field
# whose every value happens to be None on the first ingest would otherwise not
# exist at all (so it's missing from the sidebar, and a later `sync.push` writing
# None to it raises "Cannot infer an appropriate field type for value 'None'"),
# and an int-valued score would be inferred as int on one dataset and float on
# another depending on which test got ingested first.
FIELD_SCHEMA = (
    tuple((name, "float") for name in NUMERIC_FIELDS)
    + tuple((name, "int") for name in _INT_FIELDS)
    + tuple((name, "bool") for name in _BOOL_FIELDS)
    + tuple((name, "str") for name in _STR_FIELDS)
    + tuple((name, "strlist") for name in _STR_LIST_FIELDS)
)
