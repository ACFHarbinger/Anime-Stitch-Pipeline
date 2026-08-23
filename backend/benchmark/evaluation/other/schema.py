"""Evaluation data schema: bounding boxes, edges, per-dimension scores, and
the per-test evaluation entry.

**Backward compatibility is load-bearing here.** ``bench_anime_stitch.py``'s
``_load_human_evaluations()`` reads the newest ``asp_evaluations_*.json`` and
only ever does ``.get("asp")`` / ``.get("simple")`` on each entry to drive its
one-directional coherence veto. So the original
``{test: {asp, simple, notes, bboxes, edges}}`` keys are written exactly as
before; every richer field this module adds is *additive*, and
``RatingEntry.from_dict`` reads a file written by the old tool without
special-casing.

The one behavioural change: an entry now carries an explicit ``reviewed``
flag. The old tool inferred "rated" from mere presence of the dict key, but
its ``_current_entry()`` used ``setdefault`` and ``closeEvent`` persisted, so
*visiting* a test wrote ``{"asp": null, "simple": null}`` and permanently
excluded it from the next session's queue (issue #123 defect 5; already
present in live data for ``asp_test02``). ``is_rated()`` is now a real
predicate over the scores instead of a dict-membership test, so a legacy
all-null entry correctly reads as unrated.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os

from ..constants.schema import (
    COMPARATOR_KEYS,
    DEFECT_KEYS,
    DIM_COHERENCE,
    DIMENSION_KEYS,
    IMAGE_ASP,
    IMAGE_SIMPLE,
    PRIMARY_KEYS,
    SCORABLE_KEYS,
    SCORE_MAX,
    SCORE_MIN,
)


def _clamp_score(value: int | None) -> int | None:
    """Coerce anything read off disk into a valid score or ``None`` — a
    hand-edited or truncated file must not crash the tool on load."""
    if value is None:
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    if ivalue < SCORE_MIN or ivalue > SCORE_MAX:
        return None
    return ivalue


@dataclasses.dataclass
class BoundingBox:
    """A user-drawn failure-mode region, in image-normalized [0, 1] coords."""

    image: str  # one of constants.schema.COMPARATOR_KEYS
    x: float
    y: float
    w: float
    h: float
    label: str = ""
    defect: str = ""  # a constants.schema.DEFECT_KEYS member, or "" if untagged
    severity: int | None = None  # 1-3, see constants.schema.SEVERITY_LABELS

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> BoundingBox:
        return BoundingBox(
            image=d["image"], x=d["x"], y=d["y"], w=d["w"], h=d["h"],
            label=d.get("label", ""),
            defect=d.get("defect", ""),
            severity=d.get("severity"),
        )


@dataclasses.dataclass
class EdgePoint:
    """One endpoint of a link: a point, or — when ``w``/``h`` are non-zero —
    a region, drawn as a box with the line meeting its centre. The fields are
    additive over the original point-only shape (default 0 == a plain point),
    so a file written before regions existed loads unchanged."""

    image: str
    x: float
    y: float
    w: float = 0.0
    h: float = 0.0

    @property
    def is_region(self) -> bool:
        return self.w > 0.0 and self.h > 0.0


@dataclasses.dataclass
class Edge:
    """A misalignment/comparison link across 2 or more images — e.g. "this
    seam in ASP corresponds to this clean region in Overmix and this point in
    ground truth." Each endpoint is independently a point or a region.
    """

    points: list[EdgePoint]
    label: str = ""

    def to_dict(self) -> dict:
        return {
            "points": [dataclasses.asdict(p) for p in self.points],
            "label": self.label,
        }

    @staticmethod
    def from_dict(d: dict) -> Edge:
        # Pre-2026-07-30 shape: exactly two endpoints, {"a": ..., "b": ...}.
        # EdgePoint(**d["a"]) works unchanged since w/h default to 0.0.
        points = [EdgePoint(**p) for p in d["points"]] if "points" in d else [EdgePoint(**d["a"]), EdgePoint(**d["b"])]
        return Edge(points=points, label=d.get("label", ""))


@dataclasses.dataclass
class RatingEntry:
    # -- the original, bench-facing fields (never rename or retype these) ---
    asp: int | None = None
    simple: int | None = None
    notes: str = ""
    bboxes: list[BoundingBox] = dataclasses.field(default_factory=list)
    edges: list[Edge] = dataclasses.field(default_factory=list)

    # -- additive fields ----------------------------------------------------
    # {image_key: {dimension_key: 0-4}}. The "coherence" dimension of "asp"
    # and "simple" mirrors the two fields above; keeping both is deliberate
    # redundancy so the file stays readable by the old consumer.
    dimensions: dict[str, dict[str, int | None]] = dataclasses.field(default_factory=dict)
    preference: str | None = None  # constants.schema.PREFERENCE_KEYS
    confidence: int | None = None  # 1-3
    defects: list[str] = dataclasses.field(default_factory=list)  # DEFECT_KEYS
    # {output: {defect: 1..3}}. ``defects`` remains the compatible binary
    # projection: a defect is present whenever any output has severity > 0.
    defect_severity: dict[str, dict[str, int]] = dataclasses.field(default_factory=dict)
    reviewed: bool = False
    skipped: bool = False
    updated_at: str = ""

    # -- score accessors ----------------------------------------------------

    def score(self, image: str, dimension: str = DIM_COHERENCE) -> int | None:
        if dimension == DIM_COHERENCE and image in PRIMARY_KEYS:
            # The top-level field is authoritative for the two primary
            # images so a legacy file (no `dimensions` block at all) still
            # reports its coherence scores.
            return self.asp if image == IMAGE_ASP else self.simple
        return self.dimensions.get(image, {}).get(dimension)

    def set_score(self, image: str, dimension: str, value: int | None) -> None:
        self.dimensions.setdefault(image, {})[dimension] = value
        if dimension == DIM_COHERENCE:
            if image == IMAGE_ASP:
                self.asp = value
            elif image == IMAGE_SIMPLE:
                self.simple = value

    def is_rated(self) -> bool:
        """True once the entry carries the judgment the benchmark needs.

        Deliberately *not* "the key exists in the file" — that was the old
        tool's test, and it let an all-null visited entry masquerade as rated.
        """
        return self.asp is not None and self.simple is not None

    def touch(self) -> None:
        self.updated_at = datetime.datetime.now().isoformat(timespec="seconds")

    def severity(self, image: str, defect: str) -> int:
        """Return the ordinal severity; absent and malformed values read as 0."""
        value = self.defect_severity.get(image, {}).get(defect, 0)
        return value if isinstance(value, int) and 1 <= value <= 3 else 0

    def set_severity(self, image: str, defect: str, severity: int) -> None:
        """Set one output's defect severity and keep the legacy tag in sync."""
        if image not in COMPARATOR_KEYS:
            raise ValueError(f"unknown comparator: {image}")
        if defect not in DEFECT_KEYS:
            raise ValueError(f"unknown defect: {defect}")
        if severity not in (0, 1, 2, 3):
            raise ValueError(f"severity must be 0..3, got {severity}")
        if severity:
            self.defect_severity.setdefault(image, {})[defect] = severity
            self.defects = sorted(set(self.defects) | {defect})
            return
        values = self.defect_severity.get(image)
        if values is not None:
            values.pop(defect, None)
            if not values:
                self.defect_severity.pop(image, None)
        if not any(defect in values for values in self.defect_severity.values()):
            self.defects = [tag for tag in self.defects if tag != defect]

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        doc = {
            "asp": self.asp,
            "simple": self.simple,
            "notes": self.notes,
            "bboxes": [b.to_dict() for b in self.bboxes],
            "edges": [e.to_dict() for e in self.edges],
        }
        # Only emit additive fields that carry information, so a file written
        # by a plain 0-4 pass stays as small and readable as the old tool's.
        if self.dimensions:
            doc["dimensions"] = {
                img: {dim: val for dim, val in dims.items() if val is not None}
                for img, dims in self.dimensions.items()
                if any(val is not None for val in dims.values())
            }
            if not doc["dimensions"]:
                del doc["dimensions"]
        if self.preference is not None:
            doc["preference"] = self.preference
        if self.confidence is not None:
            doc["confidence"] = self.confidence
        if self.defects:
            doc["defects"] = sorted(set(self.defects))
        if self.defect_severity:
            doc["defect_severity"] = {
                image: dict(sorted(values.items()))
                for image, values in sorted(self.defect_severity.items())
                if values
            }
        if self.reviewed:
            doc["reviewed"] = True
        if self.skipped:
            doc["skipped"] = True
        if self.updated_at:
            doc["updated_at"] = self.updated_at
        return doc

    @staticmethod
    def from_dict(d: dict) -> RatingEntry:
        asp = _clamp_score(d.get("asp"))
        simple = _clamp_score(d.get("simple"))
        dimensions: dict[str, dict[str, int | None]] = {}
        for image, dims in (d.get("dimensions") or {}).items():
            if image not in SCORABLE_KEYS or not isinstance(dims, dict):
                continue
            kept = {
                dim: _clamp_score(val)
                for dim, val in dims.items()
                if dim in DIMENSION_KEYS
            }
            if kept:
                dimensions[image] = kept
        entry = RatingEntry(
            asp=asp,
            simple=simple,
            notes=d.get("notes", ""),
            bboxes=[BoundingBox.from_dict(b) for b in d.get("bboxes", [])],
            edges=[Edge.from_dict(e) for e in d.get("edges", [])],
            dimensions=dimensions,
            preference=d.get("preference"),
            confidence=_clamp_score(d.get("confidence")),
            defects=[t for t in d.get("defects", []) if isinstance(t, str)],
            skipped=bool(d.get("skipped", False)),
            updated_at=d.get("updated_at", ""),
        )
        for image, values in (d.get("defect_severity") or {}).items():
            if image not in COMPARATOR_KEYS or not isinstance(values, dict):
                continue
            for defect, severity in values.items():
                if defect not in DEFECT_KEYS:
                    continue
                try:
                    parsed = int(severity)
                except (TypeError, ValueError):
                    continue
                if 1 <= parsed <= 3:
                    entry.set_severity(image, defect, parsed)
        # A legacy file has no `reviewed` key; infer it from the scores so a
        # genuinely-rated old entry isn't re-queued, while an all-null
        # visited entry correctly reads as never reviewed.
        entry.reviewed = bool(d.get("reviewed", entry.is_rated()))
        # Keep the mirrored coherence scores consistent in both directions,
        # whichever side of the file they were written on.
        for image, top in ((IMAGE_ASP, asp), (IMAGE_SIMPLE, simple)):
            if top is not None:
                entry.dimensions.setdefault(image, {})[DIM_COHERENCE] = top
        return entry


def load_evaluations(path: str) -> dict[str, RatingEntry]:
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        raw = json.load(fh)
    return {name: RatingEntry.from_dict(entry) for name, entry in raw.items()}


def save_evaluations(path: str, evaluations: dict[str, RatingEntry]) -> None:
    doc = {name: entry.to_dict() for name, entry in evaluations.items()}
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    os.replace(tmp_path, path)  # atomic — a mid-write crash never corrupts the real file


def rated_names(evaluations: dict[str, RatingEntry]) -> list[str]:
    """Names carrying a real judgment, in file order — the queue filter the
    old tool got wrong by testing dict membership instead."""
    return [name for name, entry in evaluations.items() if entry.is_rated()]
