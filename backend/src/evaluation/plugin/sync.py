"""Syncs human judgment between the FiftyOne dataset and the evaluations JSON.

``data/benchmarks/asp_evaluations_*.json`` is the single source of truth — it is
what ``bench_anime_stitch.py`` reads, and it survives a dropped FiftyOne
database. So:

- **JSON → FiftyOne** (``push``) refreshes the human fields/tags/regions of an
  existing dataset without a full rebuild, which is what you want after a
  session in the inspector.
- **FiftyOne → JSON** (``pull``) folds back what can be edited in the App:
  sample tags. FiftyOne's App has no numeric-score editor and no label drawing
  (annotation is delegated to CVAT/Label Studio), so scores and regions are
  inspector-owned by design — see issue #123. Tagging in the App *is* useful
  though, so ``defect:<key>`` tags round-trip into ``RatingEntry.defects``.

``pull`` is deliberately conservative: it only ever adds or removes defect tags
and the skipped flag, and it never invents or clears a score. A surface that
can't author scores must not be able to destroy them.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Set

from ..constants.schema import DEFECT_KEYS
from ..other.schema import RatingEntry, load_evaluations, save_evaluations
from . import sample_fields as sf
from .preflight import require

_DEFECT_TAG_PREFIX = "defect:"


@dataclasses.dataclass
class SyncReport:
    tests_touched: int
    defects_added: int
    defects_removed: int
    skipped_changed: int
    unknown_tags: List[str]

    def summary(self) -> str:
        parts = [
            f"{self.tests_touched} test(s) updated",
            f"+{self.defects_added} / -{self.defects_removed} defect tags",
        ]
        if self.skipped_changed:
            parts.append(f"{self.skipped_changed} skip flag(s) changed")
        if self.unknown_tags:
            parts.append(f"ignored unknown tags: {', '.join(sorted(set(self.unknown_tags)))}")
        return "; ".join(parts)


def _group_tags(dataset) -> Dict[str, Set[str]]:
    """Union of every slice's tags, keyed by test name.

    Tags are per-sample in FiftyOne, and a user tagging in the App tags whatever
    slice is on screen. A defect is a property of the *test*, so the union is the
    right reading — and it means tagging from any slice works.
    """
    view = dataset.select_group_slices(_allow_mixed=True)
    tags: Dict[str, Set[str]] = {}
    for name, sample_tags in zip(*_iter_values(view, ("dataset_name", "tags")), strict=False):
        if not name:
            continue
        tags.setdefault(name, set()).update(sample_tags or [])
    return tags


def _iter_values(view, fields):
    return [view.values(field) for field in fields]


def pull(
    evaluations_path: str,
    dataset_name: Optional[str] = None,
    dry_run: bool = False,
) -> SyncReport:
    """Fold App-side tagging back into the evaluations file."""
    require(require_db=True)
    import fiftyone as fo

    from .ingest import DATASET_NAME

    dataset = fo.load_dataset(dataset_name or DATASET_NAME)
    evaluations = load_evaluations(evaluations_path)
    tags_by_test = _group_tags(dataset)

    touched = added = removed = skip_changed = 0
    unknown: List[str] = []
    for name, tags in tags_by_test.items():
        tagged_defects = set()
        for tag in tags:
            if not tag.startswith(_DEFECT_TAG_PREFIX):
                continue
            key = tag[len(_DEFECT_TAG_PREFIX):]
            if key in DEFECT_KEYS:
                tagged_defects.add(key)
            else:
                unknown.append(tag)
        skipped = "skipped" in tags

        entry = evaluations.get(name)
        if entry is None:
            if not tagged_defects and not skipped:
                continue  # nothing worth creating an entry for
            entry = RatingEntry()
            evaluations[name] = entry

        existing = set(entry.defects)
        if existing != tagged_defects:
            added += len(tagged_defects - existing)
            removed += len(existing - tagged_defects)
            entry.defects = sorted(tagged_defects)
            touched += 1
        if entry.skipped != skipped and not entry.is_rated():
            entry.skipped = skipped
            skip_changed += 1
            touched += 1
        if touched:
            entry.touch()

    if not dry_run and touched:
        save_evaluations(evaluations_path, evaluations)
    return SyncReport(touched, added, removed, skip_changed, unknown)


def push(
    evaluations_path: str,
    dataset_name: Optional[str] = None,
) -> int:
    """Refresh the dataset's human fields, tags and defect regions from the JSON.

    Cheaper than a rebuild and the normal move after an inspector session: the
    images and metrics haven't changed, only the judgment has.
    """
    require(require_db=True)
    import fiftyone as fo

    from .ingest import DATASET_NAME

    dataset = fo.load_dataset(dataset_name or DATASET_NAME)
    # A dataset built before a field was added to the schema still needs it to
    # exist before a None can be written to it.
    from .ingest import declare_schema

    declare_schema(dataset)
    evaluations = load_evaluations(evaluations_path)

    updated = 0
    view = dataset.select_group_slices(_allow_mixed=True)
    for sample in view.iter_samples(autosave=True, progress=False):
        name = sample["dataset_name"]
        evaluation = evaluations.get(name)
        image_key = sample["comparator_key"]

        for key, value in sf.human_fields(evaluation).items():
            sample[key] = value
        sample["human_score"] = evaluation.score(image_key) if evaluation else None

        # Rebuild only the tag classes this surface owns, so a tag a user added
        # by hand in the App on some other axis survives a push.
        managed_prefixes = ("rated", "unrated", "skipped", "human_asp:", "human_simple:",
                            "prefers:", _DEFECT_TAG_PREFIX, "human_disagrees")
        kept = [
            t for t in (sample.tags or [])
            if not any(t == p or t.startswith(p) for p in managed_prefixes)
        ]
        fresh = [
            t for t in sf.sample_tags(_metrics_stub(sample), evaluation)
            if any(t == p or t.startswith(p) for p in managed_prefixes)
        ]
        sample.tags = sorted(set(kept) | set(fresh))

        detections = sf.bbox_detections(evaluation, image_key)
        sample["defect_regions"] = (
            fo.Detections(detections=[
                fo.Detection(
                    label=d["label"], bounding_box=d["bounding_box"],
                    severity=d["severity"], note=d["note"],
                )
                for d in detections
            ])
            if detections else None
        )
        updated += 1
    return updated


def _metrics_stub(sample) -> Dict:
    """Reconstruct just enough of a benchmark entry for ``sample_tags`` to
    recompute the verdict/fallback/GT tags from fields already on the sample —
    avoids re-reading the results JSON during a push."""
    return {
        "comparison": {"verdict": sample["verdict"]},
        "used_fallback": bool(sample["used_fallback"]),
        "fallback_reason": sample["fallback_reason"] or "",
        "ground_truth": {"available": bool(sample["has_ground_truth"])},
    }
