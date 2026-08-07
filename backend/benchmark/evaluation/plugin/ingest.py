"""Builds the FiftyOne grouped dataset for a benchmark corpus.

One *group* per benchmark test, one *slice* per comparator image (asp / simple /
overmix / hugin / ground_truth). FiftyOne's grouped-dataset model
(``add_group_field`` + ``group_slice``) is what makes the N-way comparison native
here: the App shows one slice in the grid with a slice switcher, and the sample
modal can flip between all of a group's slices.

Every benchmark metric and every human judgment lands as a flat sample field, so
the App sidebar becomes the corpus-level query surface the old tool never had —
"metric says asp_better but the human preferred Simple", "every seam_vis_gate
fallback with aligned GT-SSIM below 0.65", "every test tagged torn_anatomy".

Saved views ship with the dataset so those queries are one click rather than
re-derived each session.
"""

from __future__ import annotations

import dataclasses

from ..constants.schema import COMPARATOR_TITLES
from ..other import discovery
from ..other.schema import RatingEntry, load_evaluations
from . import sample_fields as sf
from .preflight import require

DATASET_NAME = "asp_benchmark_evaluation"

# (view name, description, builder taking the dataset and returning a view).
# Built lazily in `save_default_views` so importing this module needs no
# FiftyOne import.
_VIEW_SPECS = (
    ("unrated", "Tests with no human judgment yet"),
    ("human_disagrees", "Human preference contradicts the recorded verdict"),
    ("fallbacks", "Tests that fell back to the SCANS simple stitch"),
    ("gt_asp_behind", "Has ground truth and ASP's aligned SSIM trails Simple's"),
    ("rated_bad_asp", "Human scored ASP coherence 0-1"),
)


@dataclasses.dataclass
class IngestResult:
    dataset_name: str
    groups: int
    samples: int
    slices: list[str]
    rated: int


def build_dataset(
    base_dir: str,
    repo_root: str,
    evaluations_path: str | None = None,
    dataset_name: str = DATASET_NAME,
    results_path: str | None = None,
    overwrite: bool = True,
    persistent: bool = True,
) -> IngestResult:
    """(Re)build the grouped dataset from what's on disk.

    Cheap enough to re-run rather than incrementally update — the whole corpus is
    97 groups of at most 5 image references, and re-running is the only way to
    pick up a fresh benchmark run's metrics anyway.
    """
    require(require_db=True)
    import fiftyone as fo

    evaluations: dict[str, RatingEntry] = (
        load_evaluations(evaluations_path) if evaluations_path else {}
    )
    names = discovery.discover_datasets(base_dir)

    dataset = fo.Dataset(name=dataset_name, overwrite=overwrite, persistent=persistent)
    dataset.add_group_field("comparator", default="asp")
    declare_schema(dataset)
    dataset.description = (
        "ASP benchmark corpus — one group per test, one slice per comparator. "
        "Rate and annotate in the PySide6 inspector (just asp-benchmark-assess); "
        "this surface is for corpus-level triage."
    )

    samples = []
    for name in names:
        assets = discovery.load_test_assets(base_dir, name, repo_root, results_path)
        evaluation = evaluations.get(name)
        group = fo.Group()
        for image_key, fields in sf.build_payloads(
            name, assets.metrics, assets.paths, evaluation
        ):
            tags = fields.pop("_tags", [])
            path = fields.pop("source_path")
            sample = fo.Sample(filepath=path, comparator=group.element(image_key))
            sample.tags = tags
            for key, value in fields.items():
                sample[key] = value
            detections = sf.bbox_detections(evaluation, image_key)
            if detections:
                sample["defect_regions"] = fo.Detections(detections=[
                    fo.Detection(
                        label=d["label"],
                        bounding_box=d["bounding_box"],
                        severity=d["severity"],
                        note=d["note"],
                    )
                    for d in detections
                ])
            samples.append(sample)

    dataset.add_samples(samples)
    save_default_views(dataset)
    rated = sum(1 for e in evaluations.values() if e.is_rated())
    return IngestResult(
        dataset_name=dataset_name,
        groups=len(names),
        samples=len(samples),
        slices=list(dataset.group_slices),
        rated=rated,
    )


def declare_schema(dataset) -> None:
    """Create every sample field up front, with an explicit type.

    Without this, FiftyOne infers a field's type from the first non-None value
    it sees — so a field that is None across the whole first ingest never exists
    (invisible in the sidebar), and a later ``sync.push`` writing None to it
    fails with "Cannot infer an appropriate field type for value 'None'". It also
    pins score fields to int rather than letting the type depend on which test
    happened to be ingested first.
    """
    import fiftyone as fo

    kinds = {
        "float": fo.FloatField,
        "int": fo.IntField,
        "bool": fo.BooleanField,
        "str": fo.StringField,
    }
    for name, kind in sf.FIELD_SCHEMA:
        if dataset.has_sample_field(name):
            continue
        if kind == "strlist":
            dataset.add_sample_field(
                name, fo.ListField, subfield=fo.StringField
            )
        else:
            dataset.add_sample_field(name, kinds[kind])
    if not dataset.has_sample_field("defect_regions"):
        dataset.add_sample_field(
            "defect_regions",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
        )


def save_default_views(dataset) -> list[str]:
    """Persist the triage queries this workflow actually asks."""
    import fiftyone as fo
    from fiftyone import ViewField as F

    builders = {
        "unrated": lambda: dataset.match(F("human_rated") == False),  # noqa: E712
        "human_disagrees": lambda: dataset.match(F("human_disagrees_with_metric") == True),  # noqa: E712
        "fallbacks": lambda: dataset.match(F("used_fallback") == True),  # noqa: E712
        "gt_asp_behind": lambda: dataset.match(
            (F("has_ground_truth") == True)  # noqa: E712
            & (F("aligned_ssim_vs_gt") != None)  # noqa: E711
        ).sort_by("aligned_ssim_vs_gt"),
        "rated_bad_asp": lambda: dataset.match(
            (F("human_asp") != None) & (F("human_asp") <= 1)  # noqa: E711
        ),
    }
    saved = []
    for name, description in _VIEW_SPECS:
        builder = builders.get(name)
        if builder is None:
            continue
        try:
            view = builder()
            if dataset.has_saved_view(name):
                dataset.delete_saved_view(name)
            dataset.save_view(name, view, description=description)
            saved.append(name)
        except Exception:
            # A view that can't be built (a field absent because no run ever
            # emitted it) must not abort the whole ingest.
            continue
    _ = fo
    return saved


def launch(dataset_name: str = DATASET_NAME, port: int | None = None, wait: bool = True):
    """Open the FiftyOne App on the dataset."""
    require(require_db=True)
    import fiftyone as fo

    dataset = fo.load_dataset(dataset_name)
    session = fo.launch_app(dataset, port=port)
    if wait:
        session.wait()
    return session


def slice_titles() -> dict[str, str]:
    """Human-readable comparator names, for anything rendering slice labels."""
    return dict(COMPARATOR_TITLES)
