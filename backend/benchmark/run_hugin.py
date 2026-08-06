"""
Generates Hugin reference-comparator stitches for the ASP benchmark corpus
(roadmap docs/moon/roadmaps/asp.md §0.5, GitHub issue #20).

Uses the system `hugin-tools`/`enblend` packages (apt) as the CLI toolchain —
pto_gen -> cpfind -> autooptimiser -> pano_modify -> nona -> enblend. This is
run as an external tool only, same posture as run_overmix.py.

For each dataset directory this writes:
  output/hugin_stitch.png      — Hugin's toolchain on the *smart-selected*
                                 frames (same input the ASP gets).
                                 bench_anime_stitch.py picks this up
                                 automatically as a reference column.
  output/hugin_full_stitch.png — Hugin on the *full* raw frame set (--full).
  output/hugin_variant.json    — per-stage settings, frame counts, wall time,
                                 success/failure for both variants.

Usage:
  python3 backend/benchmark/run_hugin.py --tests asp_test04 asp_test08
  python3 backend/benchmark/run_hugin.py --tests asp_test04 --full
  python3 backend/benchmark/run_hugin.py --first 20
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


def _load_package(alias: str, src_dir: Path) -> None:
    """Register ``src_dir`` under ``alias`` in ``sys.modules``, the same
    collision-avoiding pattern ``backend/test/conftest.py`` uses for
    ``asp_backend`` (see issue #3: this repo's and Image-Toolkit's
    top-level ``backend`` packages share a name, so a raw absolute
    ``backend.X`` import resolves inconsistently depending on which repo's
    package Python's import system finds first — this script needs to be
    run by file path (``python backend/benchmark/run_hugin.py``), not
    ``-m backend.benchmark.run_hugin``, for the same reason). A no-op when
    the alias is already registered (e.g. by Image-Toolkit's own
    bootstrap)."""
    import importlib.util

    if alias in sys.modules or not src_dir.is_dir():
        return
    spec = importlib.util.spec_from_file_location(
        alias, src_dir / "__init__.py", submodule_search_locations=[str(src_dir)]
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)


_load_package("asp_backend", Path(__file__).resolve().parents[1] / "src")

from asp_backend.ingestion.frame_selection import smart_select_frames

# All the required CLI tools ship in Debian/Ubuntu's `hugin-tools` + `enblend`
# packages — no vendor/Hugin submodule build needed (that fork's CMake only
# builds align_image_stack; pto_gen/cpfind/autooptimiser/nona aren't wired
# into any target, and cpfind's own subdirectory is commented out). See the
# field notes for the full reasoning.
_TOOLS = ("pto_gen", "cpfind", "autooptimiser", "pano_modify", "nona", "enblend")

_PROJECTION = "0"  # rectilinear — matches a scrolling/panning camera model
_OMP_THREADS = os.environ.get("ASP_BENCH_THREAD_CAP", "4")
_TIMEOUT_SEC = 300
_MAX_CANVAS_DIM = 20000


def _read_pto_canvas_size(pto_path: str) -> Tuple[int, int]:
    """Parse the `p` (panorama) line's w/h fields from a .pto file."""
    with open(pto_path, "r") as fh:
        for line in fh:
            if line.startswith("p "):
                w_match = re.search(r"\bw(\d+)", line)
                h_match = re.search(r"\bh(\d+)", line)
                if w_match and h_match:
                    return int(w_match.group(1)), int(h_match.group(1))
    raise RuntimeError(f"could not parse canvas size from {pto_path}")


def _tools_available() -> Optional[str]:
    missing = [t for t in _TOOLS if shutil.which(t) is None]
    if missing:
        return f"missing tools: {', '.join(missing)} (apt install hugin-tools enblend)"
    return None


def _smart_select_frames(frames_paths: List[str]) -> List[str]:
    # Mirrors bench_anime_stitch.py's own wrapper exactly.
    return smart_select_frames(frames_paths, min_step_px=50.0)


def _collect_frames(dataset_dir: str) -> List[str]:
    all_pngs = sorted(
        glob.glob(os.path.join(dataset_dir, "*.png"))
        + glob.glob(os.path.join(dataset_dir, "*.jpg"))
    )
    return [
        p
        for p in all_pngs
        if "panorama" not in os.path.basename(p)
        and "test_" not in os.path.basename(p)
        and "stage" not in os.path.basename(p)
    ]


def _run(cmd: List[str], cwd: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=_TIMEOUT_SEC
    )


def _run_hugin_toolchain(frames: List[str], out_path: str) -> Dict:
    """Run the full pto_gen->cpfind->autooptimiser->pano_modify->nona->enblend
    chain on `frames`, saving the final blend to `out_path`. Returns a result dict.
    """
    missing = _tools_available()
    if missing:
        return {"ok": False, "error": missing}
    if len(frames) < 2:
        return {"ok": False, "error": f"only {len(frames)} frame(s), need >=2"}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = _OMP_THREADS

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="hugin_") as tmp:
        try:
            steps = [
                ["pto_gen", "-o", "project.pto", "-p", _PROJECTION, *frames],
                ["cpfind", "--linearmatch", "-o", "project_cp.pto", "project.pto"],
                ["autooptimiser", "-a", "-l", "-s", "-o", "project_opt.pto", "project_cp.pto"],
                [
                    "pano_modify", "-p", _PROJECTION, "--fov=AUTO", "--canvas=AUTO",
                    "--crop=AUTOOUTSIDE", "--output-cropped-tiff",
                    "-o", "project_mod.pto", "project_opt.pto",
                ],
            ]
            for cmd in steps:
                proc = _run(cmd, cwd=tmp, env=env)
                if proc.returncode != 0:
                    return {
                        "ok": False,
                        "error": f"{cmd[0]} failed: {(proc.stderr or proc.stdout).strip()[-500:]}",
                        "frame_count": len(frames),
                        "wall_sec": round(time.perf_counter() - t0, 3),
                    }

            # §0.5 field-note finding: rectilinear/cylindrical FOV
            # optimization degenerates for long planar-scroll sequences — the
            # implied FOV needed to cover many frames' worth of pure
            # translation approaches Hugin's 180° projection singularity,
            # producing a canvas hundreds of thousands of pixels wide/tall
            # that then hangs or OOMs nona. Fail fast instead.
            canvas_w, canvas_h = _read_pto_canvas_size(os.path.join(tmp, "project_mod.pto"))
            if canvas_w > _MAX_CANVAS_DIM or canvas_h > _MAX_CANVAS_DIM:
                return {
                    "ok": False,
                    "error": (
                        f"degenerate canvas size {canvas_w}x{canvas_h} — pan "
                        "sequence too long for Hugin's FOV model (180° "
                        "projection singularity)"
                    ),
                    "frame_count": len(frames),
                    "wall_sec": round(time.perf_counter() - t0, 3),
                }

            proc = _run(["nona", "-m", "TIFF_m", "-o", "nona_", "project_mod.pto"], cwd=tmp, env=env)
            if proc.returncode != 0:
                return {
                    "ok": False,
                    "error": f"nona failed: {(proc.stderr or proc.stdout).strip()[-500:]}",
                    "frame_count": len(frames),
                    "wall_sec": round(time.perf_counter() - t0, 3),
                }

            layers = sorted(glob.glob(os.path.join(tmp, "nona_*.tif")))
            if len(layers) < 2:
                return {
                    "ok": False,
                    "error": f"nona produced {len(layers)} layer(s), need >=2",
                    "frame_count": len(frames),
                    "wall_sec": round(time.perf_counter() - t0, 3),
                }

            # §0.5 field-note finding: enblend's overlap-check safety guard
            # (designed for photography with partial overlap) always trips on
            # anime pan frames, which overlap almost entirely by design.
            # overlap-check-threshold=0 disables that specific check; the
            # blend itself is unaffected.
            result_tif = os.path.join(tmp, "result.tif")
            proc = _run(
                [
                    "enblend", "--parameter=overlap-check-threshold=0",
                    "-o", result_tif, *layers,
                ],
                cwd=tmp, env=env,
            )
            if proc.returncode != 0 or not os.path.exists(result_tif):
                return {
                    "ok": False,
                    "error": f"enblend failed: {(proc.stderr or proc.stdout).strip()[-500:]}",
                    "frame_count": len(frames),
                    "wall_sec": round(time.perf_counter() - t0, 3),
                }

            with Image.open(result_tif) as im:
                im.convert("RGB").save(out_path)
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "error": f"{exc.cmd[0] if exc.cmd else 'step'} timed out after {_TIMEOUT_SEC}s",
                "frame_count": len(frames),
            }

    wall_sec = round(time.perf_counter() - t0, 3)
    if not os.path.exists(out_path):
        return {"ok": False, "error": "no output produced", "frame_count": len(frames), "wall_sec": wall_sec}
    return {
        "ok": True,
        "frame_count": len(frames),
        "wall_sec": wall_sec,
        "projection": _PROJECTION,
        "matching": "linearmatch",
        "out_path": out_path,
    }


def process_dataset(dataset_dir: str, run_full: bool) -> Optional[Dict]:
    dataset_name = os.path.basename(dataset_dir)
    frames_paths = _collect_frames(dataset_dir)
    if len(frames_paths) < 2:
        print(f"Skipping {dataset_name}: not enough frames.")
        return None

    out_dir = os.path.join(dataset_dir, "output")
    variant_log: Dict = {"dataset": dataset_name}

    smart_frames = _smart_select_frames(frames_paths)
    print(f"\n=== {dataset_name}: smart variant ({len(smart_frames)}/{len(frames_paths)} frames) ===")
    smart_out = os.path.join(out_dir, "hugin_stitch.png")
    smart_result = _run_hugin_toolchain(smart_frames, smart_out)
    variant_log["smart"] = smart_result
    if smart_result["ok"]:
        print(f"  OK in {smart_result['wall_sec']}s -> {smart_out}")
    else:
        print(f"  FAILED: {smart_result['error']}")

    if run_full:
        print(f"=== {dataset_name}: full variant ({len(frames_paths)} frames) ===")
        full_out = os.path.join(out_dir, "hugin_full_stitch.png")
        full_result = _run_hugin_toolchain(frames_paths, full_out)
        variant_log["full"] = full_result
        if full_result["ok"]:
            print(f"  OK in {full_result['wall_sec']}s -> {full_out}")
        else:
            print(f"  FAILED: {full_result['error']}")

    with open(os.path.join(out_dir, "hugin_variant.json"), "w") as fh:
        json.dump(variant_log, fh, indent=2)

    return variant_log


def _resolve_datasets(base_dir: str, args) -> List[str]:
    # Mirrors bench_anime_stitch.py's _resolve_datasets exactly (kept
    # standalone here so this script has no heavy torch/cv2-chain import).
    all_dirs = sorted(
        d for d in glob.glob(os.path.join(base_dir, "asp_test*")) if os.path.isdir(d)
    )
    if args.tests:
        name_set = set(args.tests)
        selected = [d for d in all_dirs if os.path.basename(d) in name_set]
        order = {n: i for i, n in enumerate(args.tests)}
        selected.sort(key=lambda d: order.get(os.path.basename(d), 999))
    elif args.range:
        nums: set = set()
        for part in args.range.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                nums.update(range(int(lo), int(hi) + 1))
            else:
                nums.add(int(part))
        selected = [
            d
            for d in all_dirs
            if any(
                os.path.basename(d) == f"asp_test{n:02d}" or os.path.basename(d) == f"asp_test{n}"
                for n in nums
            )
        ]
    elif args.first:
        selected = all_dirs[: args.first]
    else:
        selected = all_dirs
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tests", nargs="+", metavar="NAME", help="Specific dataset names (e.g. asp_test04 asp_test27)")
    parser.add_argument("--range", metavar="SPEC", help='Numeric range "1-10" or comma list "1,3,5"')
    parser.add_argument("--first", type=int, metavar="N", help="Run only the first N datasets in sorted order")
    parser.add_argument("--full", action="store_true", help="Also generate the full-raw-frame-set variant (slower)")
    parser.add_argument(
        "--data-dir",
        default=os.path.expanduser("~/Downloads/Data/Dump"),
        metavar="DIR",
        help="Root data directory containing asp_testXX subdirectories",
    )
    args = parser.parse_args()

    missing = _tools_available()
    if missing:
        print(f"ERROR: {missing}")
        return

    datasets = _resolve_datasets(args.data_dir, args)
    if not datasets:
        print("No datasets matched.")
        return

    print(f"Processing {len(datasets)} dataset(s) with the Hugin toolchain…")
    results = []
    for d in datasets:
        r = process_dataset(d, run_full=args.full)
        if r is not None:
            results.append(r)

    n_ok = sum(1 for r in results if r.get("smart", {}).get("ok"))
    print(f"\nDone. {n_ok}/{len(results)} smart-variant stitches succeeded.")


if __name__ == "__main__":
    main()
