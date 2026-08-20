"""Versioned development slices for fast benchmarking and regression gating (M0 — issue #48).

Defines canonical development subsets for M0-M6:
1. 5-case Smoke Set (asp_test04, 08, 09, 27, 57) for rapid smoke checks.
2. Structural Red Set covering:
   - Crop loss (asp_test07, asp_test97)
   - Torn anatomy (asp_test04, asp_test06, asp_test12, asp_test15)
   - Duplicated strips (asp_test04, asp_test08)
   - Misordered content (asp_test12, asp_test41)
   - Banding / color break (asp_test11, asp_test26)
   - Known-good controls (asp_test28, asp_test58)
   - Manual-selection oracle (asp_test14)
   - Structural alignment with GT (asp_test96)

See ``submodules/ASP/docs/moon/asp_change_roadmap_2026q3.md`` §5 (M0d).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class DevelopmentSlice:
    """A versioned benchmark subset with explicit role and defect coverage."""

    name: str
    version: str
    description: str
    case_ids: tuple[str, ...]
    target_failure_modes: dict[str, tuple[str, ...]] = dataclasses.field(
        default_factory=dict
    )
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "case_ids": list(self.case_ids),
            "target_failure_modes": {
                k: list(v) for k, v in self.target_failure_modes.items()
            },
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> DevelopmentSlice:
        return DevelopmentSlice(
            name=d["name"],
            version=d.get("version", "v1"),
            description=d.get("description", ""),
            case_ids=tuple(d.get("case_ids", ())),
            target_failure_modes={
                k: tuple(v) for k, v in d.get("target_failure_modes", {}).items()
            },
            metadata=dict(d.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Canonical Slices (V1)
# ---------------------------------------------------------------------------

SMOKE_SET_V1 = DevelopmentSlice(
    name="smoke_v1",
    version="1.0.0",
    description="Canonical 5-case fast iteration smoke set (M0d).",
    case_ids=(
        "asp_test04",
        "asp_test08",
        "asp_test09",
        "asp_test27",
        "asp_test57",
    ),
    target_failure_modes={
        "torn_anatomy": ("asp_test04",),
        "duplicated_strip": ("asp_test04", "asp_test08"),
        "ghosting": ("asp_test08",),
        "affine_stress": ("asp_test09", "asp_test27"),
        "multi_strip": ("asp_test57",),
    },
    metadata={
        "purpose": "fast_smoke",
        "target_execution_seconds": 15,
    },
)

STRUCTURAL_RED_SET_V1 = DevelopmentSlice(
    name="structural_red_v1",
    version="1.0.0",
    description=(
        "Structural red set covering crop loss, torn anatomy, duplicated strips, "
        "misordering, banding, known-good controls, and test-14 oracle (M0d)."
    ),
    case_ids=(
        "asp_test04",
        "asp_test06",
        "asp_test07",
        "asp_test08",
        "asp_test11",
        "asp_test12",
        "asp_test14",
        "asp_test15",
        "asp_test26",
        "asp_test28",
        "asp_test41",
        "asp_test58",
        "asp_test96",
        "asp_test97",
    ),
    target_failure_modes={
        "crop_loss": ("asp_test07", "asp_test97"),
        "torn_anatomy": ("asp_test04", "asp_test06", "asp_test12", "asp_test15"),
        "duplicated_strip": ("asp_test04", "asp_test08"),
        "misordered_content": ("asp_test12", "asp_test41"),
        "banding": ("asp_test11", "asp_test26"),
        "known_good": ("asp_test28", "asp_test58"),
        "test14_oracle": ("asp_test14",),
        "structural_alignment_gt": ("asp_test96",),
    },
    metadata={
        "purpose": "structural_regression_gate",
        "required_coverage_categories": [
            "crop_loss",
            "torn_anatomy",
            "duplicated_strip",
            "misordered_content",
            "banding",
            "known_good",
            "test14_oracle",
        ],
    },
)

CANONICAL_SLICES: dict[str, DevelopmentSlice] = {
    SMOKE_SET_V1.name: SMOKE_SET_V1,
    "smoke": SMOKE_SET_V1,
    STRUCTURAL_RED_SET_V1.name: STRUCTURAL_RED_SET_V1,
    "structural_red": STRUCTURAL_RED_SET_V1,
    "red_set": STRUCTURAL_RED_SET_V1,
}


def get_slice(name: str) -> DevelopmentSlice:
    """Retrieve a versioned slice by name or alias."""
    key = name.strip().lower()
    if key not in CANONICAL_SLICES:
        available = ", ".join(sorted(CANONICAL_SLICES.keys()))
        raise KeyError(f"Unknown benchmark slice {name!r}. Available: {available}")
    return CANONICAL_SLICES[key]


def list_slices() -> list[DevelopmentSlice]:
    """Return all unique canonical development slices."""
    seen: set[str] = set()
    result: list[DevelopmentSlice] = []
    for s in CANONICAL_SLICES.values():
        if s.name not in seen:
            seen.add(s.name)
            result.append(s)
    return result


def get_slice_cases(name: str) -> list[str]:
    """Return list of case names for a slice."""
    return list(get_slice(name).case_ids)


def get_cases_for_failure_mode(slice_name: str, failure_mode: str) -> list[str]:
    """Return cases tagged with a specific failure mode in a slice."""
    s = get_slice(slice_name)
    return list(s.target_failure_modes.get(failure_mode, ()))


def verify_slice_coverage(slice_name: str) -> dict[str, list[str]]:
    """Verify and return the category mapping for a slice."""
    s = get_slice(slice_name)
    required = s.metadata.get("required_coverage_categories", [])
    missing = [cat for cat in required if cat not in s.target_failure_modes or not s.target_failure_modes[cat]]
    if missing:
        raise ValueError(
            f"Slice {s.name!r} missing required coverage categories: {missing}"
        )
    return {k: list(v) for k, v in s.target_failure_modes.items()}


def export_slices_manifest(output_path: Path | str) -> None:
    """Export all canonical slices to a JSON manifest."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "1.0.0",
        "slices": {s.name: s.to_dict() for s in list_slices()},
    }
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
