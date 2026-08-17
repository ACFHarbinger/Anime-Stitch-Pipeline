"""M3 red-set compositor A/B: current seam loop vs ``coherence_v2``.

Same warped inputs, two compositors. Does **not** rematch or claim a human
rating of the v2 output — existing human labels describe the published
default path. The M3 exit asks whether the structural red set improves
without increasing crop loss; this screen reports coverage/area and
diagnostic structural scores for that decision.

Frame set: sorted dump frames, evenly subsampled to ``--max-frames``.
Affines: median ``dy_steps`` from the saved run JSON (vertical pan).
This is a compositor-only proxy, not a full-pipeline restitch.

Usage::

    .venv/bin/python backend/benchmark/screen_coherence_v2.py \\
        --run backend/benchmark/output/anime_stitch_20260807_045552.json \\
        --data-dir dump --max-frames 6 --scale 0.25
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ASP_ROOT = Path(__file__).resolve().parents[2]


def _load_asp_backend() -> None:
    src = ASP_ROOT / "backend" / "src"
    if "asp_backend" in sys.modules or not src.is_dir():
        return
    spec = importlib.util.spec_from_file_location(
        "asp_backend",
        src / "__init__.py",
        submodule_search_locations=[str(src)],
    )
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    sys.modules["asp_backend"] = mod
    spec.loader.exec_module(mod)


_load_asp_backend()

from asp_backend.alignment.canvas import _compute_canvas, _load_frames, _normalise_widths
from asp_backend.core.pipeline._frame_utils import _sort_frames_by_index
from asp_backend.rendering.compositing.composite import _composite_foreground
from asp_backend.rendering.compositing.coherence_v2 import coherence_v2_enabled

RED_SET = (
    "asp_test04",
    "asp_test06",
    "asp_test07",
    "asp_test12",
    "asp_test14",
    "asp_test15",
    "asp_test96",
)
CATASTROPHES = set(RED_SET) - {"asp_test96"}


def content_coverage(img: np.ndarray) -> float:
    return float((img.max(axis=2) > 10).mean()) if img.size else 0.0


def content_area(img: np.ndarray) -> int:
    ys, xs = np.where(img.max(axis=2) > 10)
    if len(xs) == 0:
        return 0
    return int((int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1))


def crop_loss_increased(base: np.ndarray, cand: np.ndarray, *, eps: float = 0.02) -> bool:
    """True if candidate covers less of the canvas or a smaller content box."""
    if content_coverage(cand) + 1e-6 < content_coverage(base) - eps:
        return True
    ba, ca = content_area(base), content_area(cand)
    return ba > 0 and ca < ba * (1.0 - eps)


def _even_sample(paths: list[str], k: int) -> list[str]:
    if len(paths) <= k:
        return list(paths)
    idx = np.linspace(0, len(paths) - 1, k, dtype=int)
    return [paths[int(i)] for i in idx]


def _median_dy(dataset: dict[str, Any], scale: float) -> float:
    steps = (dataset.get("alignment") or {}).get("dy_steps") or []
    if not steps:
        src_h = float((dataset.get("frames") or {}).get("source_h") or 1080)
        return max(40.0, src_h * 0.08) * scale
    return float(np.median(np.abs(np.asarray(steps, dtype=np.float64)))) * scale


def _affines_vertical(n: int, dy: float) -> list[np.ndarray]:
    out = []
    for i in range(n):
        m = np.eye(2, 3, dtype=np.float32)
        m[1, 2] = float(i) * dy
        out.append(m)
    return out


def _discover_frames(case_dir: Path) -> list[str]:
    files = [
        str(p)
        for p in case_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.is_file()
    ]
    return _sort_frames_by_index(files)


def _score(img: np.ndarray) -> dict[str, float]:
    row = {
        "coverage": round(content_coverage(img), 4),
        "content_area": float(content_area(img)),
        "width": float(img.shape[1]),
        "height": float(img.shape[0]),
    }
    try:
        from asp_backend.core.pipeline.anime_metrics import line_art_fracture_score
        from asp_backend.core.pipeline.safety_metrics import seam_visibility_score

        row["line_art_fracture"] = round(float(line_art_fracture_score(img)), 3)
        row["seam_visibility"] = round(float(seam_visibility_score(img)), 3)
    except Exception:
        pass
    return row


def evaluate_case(
    name: str,
    dataset: dict[str, Any],
    data_dir: Path,
    *,
    max_frames: int,
    scale: float,
) -> dict[str, Any]:
    case_dir = data_dir / name
    paths = _even_sample(_discover_frames(case_dir), max_frames)
    frames = _normalise_widths(_load_frames(paths))
    if scale != 1.0:
        frames = [
            cv2.resize(
                f,
                (max(8, int(f.shape[1] * scale)), max(8, int(f.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
            for f in frames
        ]
    if len(frames) < 2:
        return {"name": name, "error": "need_two_frames"}
    dy = _median_dy(dataset, scale)
    affines = _affines_vertical(len(frames), dy)
    canvas_h, canvas_w, t_global = _compute_canvas(frames, affines)
    for a in affines:
        a[0, 2] += t_global[0]
        a[1, 2] += t_global[1]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    bg = [None] * len(frames)

    os.environ["ASP_COHERENCE_V2"] = "0"
    assert coherence_v2_enabled() is False
    default = _composite_foreground(
        [], [], canvas, canvas_h, canvas_w, frames, affines, bg
    )
    os.environ["ASP_COHERENCE_V2"] = "1"
    assert coherence_v2_enabled() is True
    v2 = _composite_foreground(
        [], [], canvas, canvas_h, canvas_w, frames, affines, bg
    )
    os.environ["ASP_COHERENCE_V2"] = "0"

    return {
        "name": name,
        "role": "known_good" if name == "asp_test96" else "catastrophe",
        "n_frames": len(frames),
        "scale": scale,
        "dy": round(dy, 2),
        "default": _score(default),
        "coherence_v2": _score(v2),
        "crop_loss_increased": crop_loss_increased(default, v2),
        "human_labels_apply_to": "published_default_path_only",
    }


def screen(
    run_path: Path,
    data_dir: Path,
    *,
    max_frames: int = 6,
    scale: float = 0.25,
    names: tuple[str, ...] = RED_SET,
) -> dict[str, Any]:
    doc = json.loads(run_path.read_text(encoding="utf-8"))
    by_name = {d["name"]: d for d in doc.get("datasets") or []}
    rows = []
    for name in names:
        if name not in by_name:
            rows.append({"name": name, "error": "missing_in_run"})
            continue
        rows.append(
            evaluate_case(
                name,
                by_name[name],
                data_dir,
                max_frames=max_frames,
                scale=scale,
            )
        )
    scored = [r for r in rows if "crop_loss_increased" in r]
    n_crop = sum(1 for r in scored if r["crop_loss_increased"])
    good = next((r for r in scored if r["name"] == "asp_test96"), None)
    return {
        "n": len(scored),
        "n_crop_loss_increased": n_crop,
        "known_good_crop_ok": bool(good) and not good["crop_loss_increased"],
        "passes_crop_gate": n_crop == 0 and bool(scored),
        "note": (
            "Compositor-only A/B on subsampled frames + median-dy affines. "
            "Human ratings still describe the published default path, not v2."
        ),
        "cases": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=ASP_ROOT / "dump")
    ap.add_argument("--max-frames", type=int, default=6)
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument(
        "--out",
        type=Path,
        default=ASP_ROOT / "backend" / "benchmark" / "output" / "coherence_v2_redset.json",
    )
    args = ap.parse_args()
    result = screen(
        args.run,
        args.data_dir,
        max_frames=args.max_frames,
        scale=args.scale,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(
        f"crop_gate={result['passes_crop_gate']} "
        f"n_crop_loss={result['n_crop_loss_increased']}/{result['n']} "
        f"known_good_crop_ok={result['known_good_crop_ok']}"
    )
    for row in result["cases"]:
        if "error" in row:
            print(f"  {row['name']:14s} ERROR {row['error']}")
            continue
        d, v = row["default"], row["coherence_v2"]
        flag = " CROP_LOSS" if row["crop_loss_increased"] else ""
        print(
            f"  {row['name']:14s} cov {d['coverage']:.3f}->{v['coverage']:.3f} "
            f"area {d['content_area']:.0f}->{v['content_area']:.0f}{flag}"
        )
    website = (
        ASP_ROOT.parent.parent / "docs" / "website" / "public" / "data" / "coherence_v2_redset.json"
    )
    if website.parent.is_dir():
        website.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {website}")
    raise SystemExit(0 if result["passes_crop_gate"] else 5)


if __name__ == "__main__":
    main()
