"""
Generates Overmix reference-comparator stitches for the ASP benchmark corpus
(roadmap docs/moon/roadmaps/asp.md §0.3, GitHub issue #18).

Overmix (github.com/spillerrec/Overmix, GPL-3.0) is built and run as an
external tool — never linked into our own binaries. Run
``desktop/linux/scripts/setup_overmix.sh`` once first to build ``OvermixCli``.

For each dataset directory this writes:
  output/overmix_stitch.png       — Overmix on the *smart-selected* frames,
                                     the same input the ASP pipeline gets.
                                     bench_anime_stitch.py picks this up
                                     automatically as a reference column.
  output/overmix_full_stitch.png  — Overmix on the *full* raw frame set
                                     (--full flag only; Overmix's own
                                     "maximal ingestion" philosophy wants
                                     every frame, not a pre-thinned subset).
  output/overmix_variant.json     — aligner/comparator/render settings, frame
                                     counts, wall time, and success/failure
                                     for both variants (feeds the
                                     .agent/cache/overmix_field_notes.md
                                     write-up).

Usage:
  python -m backend.benchmark.run_overmix --tests asp_test04 asp_test08
  python -m backend.benchmark.run_overmix --tests asp_test04 --full
  python -m backend.benchmark.run_overmix --first 20
"""

import argparse
import glob
import json
import os
import subprocess
import time
from typing import Dict, List, Optional

from backend.src.animation.ingestion.frame_selection import smart_select_frames

_TOOLKIT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OVERMIX_BIN = os.path.join(_TOOLKIT_ROOT, "vendor", "Overmix", "build", "OvermixCli")

# Same settings the setup smoke-test validated: Gradient's coarse-to-fine
# pyramid search is dramatically faster than BruteForce at full frame
# resolution (a 6-frame BruteForce run didn't finish in 90s; Gradient did the
# same job in ~1.2s). Values match GradientComparator's own C++ defaults
# except AlignMethod, made explicit as "both" since these are free-camera pans,
# not a fixed scroll axis.
_COMPARATOR = "Gradient:1/false/0:both:0.75:1:6:1638"
_ALIGNER = "Recursive"
# average render (not statistics:median) — deliberately mirrors ASP's default
# temporal-median renderer with Overmix's own preferred statistic, per
# roadmap §1.2(b): "does its average-render on our bg regions beat our
# temporal median visually?"
_RENDER = "average:false:false"
_OMP_THREADS = os.environ.get("ASP_BENCH_THREAD_CAP", "4")
_TIMEOUT_SEC = 300


def _smart_select_frames(frames_paths: List[str]) -> List[str]:
    # Mirrors bench_anime_stitch.py's own wrapper exactly, so the "smart"
    # variant here really is "the same input the ASP gets."
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


def _run_overmix(frames: List[str], out_path: str) -> Dict:
    """Invoke OvermixCli on `frames`, saving to `out_path`. Returns a result dict."""
    if not os.path.exists(_OVERMIX_BIN):
        return {"ok": False, "error": f"OvermixCli not built at {_OVERMIX_BIN}; run setup_overmix.sh"}
    if len(frames) < 2:
        return {"ok": False, "error": f"only {len(frames)} frame(s), need >=2"}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        _OVERMIX_BIN,
        *frames,
        f"--comparator={_COMPARATOR}",
        f"--align={_ALIGNER}",
        f"--render={_RENDER}",
        f"--save=0:{out_path}",
    ]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = _OMP_THREADS

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=_TIMEOUT_SEC
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {_TIMEOUT_SEC}s", "frame_count": len(frames)}
    wall_sec = round(time.perf_counter() - t0, 3)

    if proc.returncode != 0 or not os.path.exists(out_path):
        return {
            "ok": False,
            "error": (proc.stderr or proc.stdout or "unknown failure").strip()[-500:],
            "frame_count": len(frames),
            "wall_sec": wall_sec,
        }
    return {
        "ok": True,
        "frame_count": len(frames),
        "wall_sec": wall_sec,
        "comparator": _COMPARATOR,
        "aligner": _ALIGNER,
        "render": _RENDER,
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
    smart_out = os.path.join(out_dir, "overmix_stitch.png")
    smart_result = _run_overmix(smart_frames, smart_out)
    variant_log["smart"] = smart_result
    if smart_result["ok"]:
        print(f"  OK in {smart_result['wall_sec']}s -> {smart_out}")
    else:
        print(f"  FAILED: {smart_result['error']}")

    if run_full:
        print(f"=== {dataset_name}: full variant ({len(frames_paths)} frames) ===")
        full_out = os.path.join(out_dir, "overmix_full_stitch.png")
        full_result = _run_overmix(frames_paths, full_out)
        variant_log["full"] = full_result
        if full_result["ok"]:
            print(f"  OK in {full_result['wall_sec']}s -> {full_out}")
        else:
            print(f"  FAILED: {full_result['error']}")

    with open(os.path.join(out_dir, "overmix_variant.json"), "w") as fh:
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

    datasets = _resolve_datasets(args.data_dir, args)
    if not datasets:
        print("No datasets matched.")
        return

    print(f"Processing {len(datasets)} dataset(s) with Overmix ({_OVERMIX_BIN})…")
    results = []
    for d in datasets:
        r = process_dataset(d, run_full=args.full)
        if r is not None:
            results.append(r)

    n_ok = sum(1 for r in results if r.get("smart", {}).get("ok"))
    print(f"\nDone. {n_ok}/{len(results)} smart-variant stitches succeeded.")


if __name__ == "__main__":
    main()
