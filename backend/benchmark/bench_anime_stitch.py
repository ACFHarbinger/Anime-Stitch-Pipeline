#!/usr/bin/env python3
"""
Anime Stitch Pipeline Benchmark
================================
Runs both the Anime Stitch Pipeline (ASP) and OpenCV SCANS Simple Stitch on
every asp_testX dataset in Data/, then generates a comprehensive markdown
report with side-by-side comparisons, CV metrics, intermediate-output
analysis (2-D and 3-D visualizations), and structured feedback blocks for
human review and LLM-assisted iteration.
"""

import contextlib
import datetime
import gc
import glob
import json
import logging
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch

# Same NVIDIA GPU as PyTorch CUDA. Must be off before BaSiC/BiRefNet (#49).
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


def _load_package(alias: str, src_dir: Path) -> None:
    """Register ``src_dir`` under ``alias`` in ``sys.modules``, the same
    collision-avoiding pattern ``backend/test/conftest.py`` uses for
    ``asp_backend`` (see issue #3: this repo's and Image-Toolkit's
    top-level ``backend`` packages share a name, so a raw absolute
    ``backend.X`` import resolves inconsistently depending on which repo's
    package Python's import system finds first — this script needs to be
    run by file path (``python backend/benchmark/bench_anime_stitch.py``),
    not ``-m backend.benchmark.bench_anime_stitch``, for the same reason).
    A no-op when the alias is already registered (e.g. by Image-Toolkit's
    own bootstrap)."""
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

from asp_backend.alignment.bundle_adjust import _bundle_adjust_affine  # noqa: E402
from asp_backend.alignment.canvas import (  # noqa: E402
    _compute_canvas,
    _crop_to_valid,
    _load_frames,
    _normalise_widths,
    _scan_stitch_fallback,
)
from asp_backend.alignment.ecc import _ecc_refine  # noqa: E402
from asp_backend.alignment.matching import _pairwise_match  # noqa: E402
from asp_backend.core.pipeline import AnimeStitchPipeline  # noqa: E402
from asp_backend.core.pipeline._probes import _USE_SAM2  # noqa: E402
from asp_backend.core.pipeline.safety_metrics import (  # noqa: E402
    ghosting_score_v2 as _ghosting_score_v2,
    seam_coherence as _seam_coherence,
    seam_visibility_score as _seam_visibility_score,
    strip_banding_score as _strip_banding_score,
)
from asp_backend.core.pipeline.bench_adapter import (  # noqa: E402
    apply_ungated_gate_env,
    bench_legacy_enabled,
    run_canonical_asp,
)
from asp_backend.core.pipeline.safety_policy import (  # noqa: E402
    default_benchmark_policy,
)
from asp_backend.core.validation import _validate_affines  # noqa: E402
from asp_backend.ingestion.frame_selection import (  # noqa: E402
    detect_animation_phases,
    phase_spans,
    smart_select_frames,
)
from asp_backend.ingestion.masking import (  # noqa: E402
    _cleanup_sam2_state,
    _compute_fg_masks,
    _compute_fg_masks_sam2_stateful,
)
from asp_backend.rendering.compositing import _composite_foreground  # noqa: E402
from asp_backend.rendering.rendering import _render_median  # noqa: E402

# §2.6 (2026-07-27): repeated benchmark runs have frozen the host hard enough
# to force a restart, and the user independently observed the benchmark
# spawning many concurrent processes/threads in htop. Nothing in this backend
# ever capped OpenMP/BLAS/OpenCV/PyTorch thread pools, so each library
# independently defaults to spawning one thread per logical CPU core — on a
# high-core-count machine, several such uncoordinated pools (BLAS for numpy,
# OpenCV's parallel_for_, PyTorch's intraop pool) stack multiplicatively.
# This doesn't explain a leak by itself, but bounds peak concurrent resource
# usage regardless of what the deeper cause turns out to be, and makes any
# per-thread cost far less severe. These MUST be set before numpy/cv2/torch
# are imported — they read these env vars once at native library load time.
_THREAD_CAP = os.environ.get("ASP_BENCH_THREAD_CAP", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREAD_CAP)

cv2.setNumThreads(int(_THREAD_CAP))
try:
    # torch is imported above so the OpenMP environment variables are too late
    # for its pools; enforce the benchmark cap through its runtime API too.
    torch.set_num_threads(int(_THREAD_CAP))
    torch.set_num_interop_threads(1)
except RuntimeError:
    # An embedding/test host can have already configured the inter-op pool.
    # The process still keeps its existing safe setting in that case.
    pass
torch.set_num_threads(int(_THREAD_CAP))

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# Lazy-import heavy plotting deps so the benchmark still runs without them
# ---------------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D projection)

    _MPL_OK = True
except ImportError:
    _MPL_OK = False

try:
    from skimage.metrics import structural_similarity as ssim

    _SSIM_OK = True
except ImportError:
    _SSIM_OK = False

logger = logging.getLogger(__name__)


# ============================================================================
# SYSTEM INFO
# ============================================================================


def _system_info() -> dict:
    info: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count() or 0,
        "cpu_threads": 0,
        "ram_gb": 0.0,
        "gpu": "N/A",
        "cuda_version": "N/A",
        "vram_gb": 0.0,
    }
    try:
        import psutil as ps  # pyrefly: ignore [untyped-import]

        info["cpu_threads"] = ps.cpu_count(logical=True) or 0
        info["ram_gb"] = round(ps.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda or "N/A"
        props = torch.cuda.get_device_properties(0)
        info["vram_gb"] = round(props.total_memory / 1024**3, 1)
    return info


# ── Resource guardrail (added 2026-07-27) ───────────────────────────────────
# Benchmark runs have frozen the host badly enough to require a hard restart,
# on a machine with 128GB RAM / 24GB VRAM — meaning uncontrolled growth, not
# "the box is slow". Root cause not yet found. Until it is, the batch loop
# below checks system RAM and GPU VRAM after every dataset and aborts
# gracefully (writing whatever results already exist) rather than continuing
# toward the point where the OS itself becomes unresponsive.
_RAM_ABORT_PCT = float(os.environ.get("ASP_BENCH_RAM_ABORT_PCT", "80"))
_VRAM_ABORT_PCT = float(os.environ.get("ASP_BENCH_VRAM_ABORT_PCT", "85"))


def _resource_snapshot() -> dict:
    """RSS/VRAM snapshot for per-dataset instrumentation and the abort guardrail."""
    proc = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()
    snap: dict = {
        "rss_gb": round(proc.memory_info().rss / 1024**3, 2),
        "sys_ram_used_pct": vm.percent,
        "sys_ram_available_gb": round(vm.available / 1024**3, 2),
        "vram_allocated_gb": None,
        "vram_reserved_gb": None,
        "vram_used_pct": None,
    }
    if torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            used_b = total_b - free_b
            snap["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
            snap["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
            snap["vram_used_pct"] = round(100.0 * used_b / total_b, 1)
        except Exception:
            pass
    return snap


def _resource_danger(snap: dict) -> str | None:
    """Reason string if the snapshot crosses an abort threshold, else None."""
    if snap["sys_ram_used_pct"] >= _RAM_ABORT_PCT:
        return (
            f"system RAM at {snap['sys_ram_used_pct']:.0f}% "
            f"(abort >= {_RAM_ABORT_PCT:.0f}%)"
        )
    if snap["vram_used_pct"] is not None and snap["vram_used_pct"] >= _VRAM_ABORT_PCT:
        return (
            f"GPU VRAM at {snap['vram_used_pct']:.0f}% "
            f"(abort >= {_VRAM_ABORT_PCT:.0f}%)"
        )
    return None


# §2.6 diagnostic instrumentation (2026-07-27): per-stage checkpoints inside
# process_dataset itself, not just per-dataset in the outer loop, so a single
# dataset already reveals which *stage* — not just which dataset — leaves
# memory elevated. Prints one compact line per call; cheap enough to leave in
# permanently. CUDA allocator flushing is deliberately opt-in because
# ``empty_cache()`` synchronizes the device and was masking a third stall after
# matching (set ``ASP_RESOURCE_FLUSH_CUDA=1`` when collecting allocator data).
def _log_resource(tag: str, store: dict[str, float] | None = None) -> dict:
    gc.collect()
    if os.environ.get("ASP_RESOURCE_FLUSH_CUDA", "0") == "1" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    snap = _resource_snapshot()
    print(
        f"    [Res/{tag}] RSS={snap['rss_gb']}GB  sys_ram={snap['sys_ram_used_pct']}%  "
        f"vram_alloc={snap['vram_allocated_gb']}GB  vram_reserved={snap['vram_reserved_gb']}GB  "
        f"vram_used={snap['vram_used_pct']}%"
    )
    # §11.6 (issue #69): feed the same snapshot this line already prints into
    # the per-dataset accumulator so the benchmark JSON carries a
    # stage_memory_rss_mb series, not just a console log — the waterfall
    # chart in _report_stage_memory_waterfall() reads this.
    if store is not None:
        store[tag] = round(snap["rss_gb"] * 1024, 1)
    return snap


# Fixed stage order for the waterfall chart — mirrors the call sequence in
# process_dataset() (dataset_start ... dataset_end). Not every dataset hits
# every tag (e.g. a SCANS fallback skips before/after_render_median), so the
# waterfall renderer only plots tags actually present in a given result.
STAGE_MEMORY_ORDER: tuple[str, ...] = (
    "dataset_start",
    "before_birefnet",
    "after_birefnet_offload",
    "before_loftr",
    "after_loftr_offload",
    "before_render_median",
    "after_render_median",
    "after_composite",
    "dataset_end",
)


# ============================================================================
# CV METRIC HELPERS
# ============================================================================


def _sharpness(img: np.ndarray) -> float:
    """Laplacian-variance sharpness (higher = sharper)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(lap.var())


def _coverage(img: np.ndarray) -> float:
    """Fraction of non-black pixels (proxy for crop completeness)."""
    mask = img.max(axis=2) > 8 if img.ndim == 3 else img > 8
    return float(mask.sum()) / max(mask.size, 1)


def _mean_seam_gradient(
    img: np.ndarray, affines: list[np.ndarray] | None = None
) -> float:
    """
    Average gradient magnitude along horizontal seam boundaries.
    Without affines, samples the whole image and returns mean gradient.
    With affines, evaluates only the seam transition rows.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    if affines is None:
        return float(np.abs(gy).mean())
    H, W = gray.shape
    seam_rows = set()
    for a in affines:
        row = round(float(a[1, 2]))
        for dr in range(-5, 6):
            r = row + dr
            if 0 <= r < H:
                seam_rows.add(r)
    if not seam_rows:
        return float(np.abs(gy).mean())
    rows = np.array(sorted(seam_rows))
    return float(np.abs(gy[rows]).mean())


def _color_entropy(img: np.ndarray) -> float:
    """Shannon entropy of luma histogram (higher = more diverse colours)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / max(hist.sum(), 1.0)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def _ssim_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """SSIM between two images (resized to min dims if needed)."""
    if not _SSIM_OK:
        return float("nan")
    h = min(img_a.shape[0], img_b.shape[0])
    w = min(img_a.shape[1], img_b.shape[1])
    a = cv2.resize(img_a, (w, h))
    b = cv2.resize(img_b, (w, h))
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(ga, gb, full=True, data_range=255)
    return float(score)


def _psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """PSNR (dB) between two images after resizing to common dims."""
    h = min(img_a.shape[0], img_b.shape[0])
    w = min(img_a.shape[1], img_b.shape[1])
    a = cv2.resize(img_a, (w, h)).astype(np.float32)
    b = cv2.resize(img_b, (w, h)).astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-8:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def _edge_energy_score(img: np.ndarray) -> float:
    """§3.32: Edge energy proxy (formerly mislabelled as ghosting).

    Computes mean absolute value of the second-order vertical derivative
    (double-Sobel Y) — this measures *edge energy / sharpness*, NOT ghosting.
    Kept for backward compatibility with ASP_GATE_GHOST and historical JSON keys;
    emitted as ``edge_energy_score`` in all new benchmark output.

    For true ghosting detection use ``_ghosting_score_v2`` (``ghosting_siqe``).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = gray.astype(np.float32)
    gy2 = cv2.Sobel(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3), cv2.CV_32F, 0, 1, ksize=3)
    return float(np.abs(gy2).mean())


def _compute_per_seam_ghost_scores(
    img: np.ndarray,
    n_strips: int,
    band_px: int = 100,
) -> list[float]:
    """§3.8B: Per-seam SIQE ghost scores for a vertically-assembled panorama.

    Divides the output image into *n_strips* equal-height zones and evaluates
    ``_ghosting_score_v2`` in a ±*band_px* horizontal band centred at each of
    the ``n_strips − 1`` inter-strip seam boundaries.  This localises ghosting
    to specific seams rather than averaging over the whole image (as
    ``ghosting_siqe`` does), making it actionable for per-seam quality triage.

    Parameters
    ----------
    img : (H, W, 3) or (H, W) uint8 image.
    n_strips : number of equal-height zones.  Must be ≥ 2 to produce any
               scores; returns ``[]`` for *n_strips* ≤ 1.
    band_px : half-height of the analysis window around each boundary (px).
              Clipped to image bounds automatically.

    Returns
    -------
    List of ``n_strips − 1`` float scores (same units as ``_ghosting_score_v2``:
    0–100).  Empty list when *n_strips* ≤ 1.
    """
    if n_strips <= 1:
        return []
    H = img.shape[0]
    zone_h = H / n_strips
    scores: list[float] = []
    for k in range(1, n_strips):
        boundary_y = int(round(zone_h * k))
        y0 = max(0, boundary_y - band_px)
        y1 = min(H, boundary_y + band_px)
        band = img[y0:y1]
        scores.append(_ghosting_score_v2(band) if (y1 - y0) >= 20 else 0.0)
    return scores


def _compute_cqas_v1_legacy(metrics: dict) -> float | None:
    """Legacy CQAS v1 diagnostic; not a quality verdict or ranking signal.

    The 2026-08-17 human-label audit found no association with human
    ASP-vs-SCANS preference (rho=-0.09, non-significant). Preserve the
    historic formula only for backwards-compatible diagnostics; it must not
    drive automated verdicts, sorting, or "X wins" claims. A replacement is
    M2.5a work and must be versioned and validated on a held-out split.

    Components (all normalized to [0, 1]):
      ghosting_siqe        : 0=clean → 1.0; ≥60=ghost → 0.0  (weight 0.35)
      seam_visibility      : 0=invisible → 1.0; ≥25=hard-cut → 0.0  (weight 0.30)
      seam_coherence       : 0=coherent → 1.0; ≥50=incoherent → 0.0  (weight 0.20)
      sharpness            : corpus ref ~100; clamped to [0, 1]  (weight 0.15)
      canvas_gain_uniformity: 0=uniform → 1.0; ≥0.40=banded → 0.0  (weight 0.15)

    Returns None only when all five metrics are None.
    total_w normalization handles missing components gracefully.
    """
    g = metrics.get("ghosting_siqe")
    g_score = float(np.clip(1.0 - g / 60.0, 0.0, 1.0)) if g is not None else None

    sv = metrics.get("seam_visibility")
    sv_score = float(np.clip(1.0 - sv / 25.0, 0.0, 1.0)) if sv is not None else None

    sc = metrics.get("seam_coherence")
    sc_score = float(np.clip(1.0 - sc / 50.0, 0.0, 1.0)) if sc is not None else None

    sh = metrics.get("sharpness")
    sh_score = float(np.clip(sh / 100.0, 0.0, 1.0)) if sh is not None else None

    cgu = metrics.get("canvas_gain_uniformity")
    cgu_score = float(np.clip(1.0 - cgu / 0.40, 0.0, 1.0)) if cgu is not None else None

    components = [
        (g_score, 0.35),
        (sv_score, 0.30),
        (sc_score, 0.20),
        (sh_score, 0.15),
        (cgu_score, 0.15),
    ]
    available = [(s, w) for s, w in components if s is not None]
    if not available:
        return None
    total_w = sum(w for _, w in available)
    return round(sum(s * w for s, w in available) / total_w, 4)


def _compute_all_metrics(
    img: np.ndarray,
    affines: list | None = None,
    n_strips: int = 1,
) -> dict:
    """Core no-reference metric set for one output image.

    2026-07 trim: reduced from ~40 metrics to the validated core.  Metric
    semantics (per the 2026-07-08 critical evaluation):
      edge_energy_score — double-Sobel Y energy; a SHARPNESS proxy, not ghosting.
      ghosting_siqe     — FFT autocorrelation double-edge score; the TRUE
                          ghosting metric (higher = more periodic ghosting).
      seam_coherence    — std of per-row mean luminance (banding proxy).
      seam_visibility   — worst adjacent-row luminance jump (dominant ASP
                          failure mode vs simple stitch: 25.8 vs 4.2 at S160).
      strip_banding_score — max luminance jump between adjacent frame-strip
                          entry bands (CompositeGate input). 0.0 without
                          affines, so SCANS/simple is 0 by construction.
      cqas_v1_legacy    — historic aggregate, diagnostic-only; not a verdict.
    """
    seam_scores = _compute_per_seam_ghost_scores(img, n_strips)
    metrics = {
        "sharpness": round(_sharpness(img), 2),
        "coverage": round(_coverage(img), 4),
        "seam_gradient": round(_mean_seam_gradient(img, affines), 3),
        "color_entropy": round(_color_entropy(img), 4),
        "edge_energy_score": round(_edge_energy_score(img), 4),
        "ghosting_siqe": round(_ghosting_score_v2(img), 2),
        "seam_coherence": round(_seam_coherence(img), 2),
        "seam_visibility": round(_seam_visibility_score(img, affines), 2),
        "strip_banding_score": round(_strip_banding_score(img, affines), 2),
        "ghost_seam_scores": [round(x, 2) for x in seam_scores],
        "ghost_seam_max": round(max(seam_scores), 2) if seam_scores else None,
        "width": img.shape[1],
        "height": img.shape[0],
    }
    metrics["cqas_v1_legacy"] = _compute_cqas_v1_legacy(metrics)
    return metrics


def _load_ground_truth(dataset_name: str, gt_dir: str) -> np.ndarray | None:
    """
    Load the ground truth reference image for a dataset, if one exists.

    Tries .png, .jpg, .jpeg extensions in that order.  Returns None when no
    ground truth is available for the given dataset.
    """
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(gt_dir, f"{dataset_name}{ext}")
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return img
    return None


def _compute_aligned_ssim(output_img: np.ndarray, gt_img: np.ndarray) -> float:
    """
    SSIM after ECC Euclidean alignment of output_img to gt_img (S8 metric, S25 dedup).

    Eliminates framing/translation bias from GT-coupling — frame substitutions that
    diverge from the GT's temporal reference shift the output, penalising raw SSIM
    even when pose quality is identical. MOTION_EUCLIDEAN handles both translation
    and small rotation residuals from the panorama assembly.

    gaussFiltSize=5 pre-smooths ECC input for robustness on noisy/low-texture crops.
    GT dimensions are used as the canonical reference space. Falls back to
    non-aligned SSIM if ECC diverges (e.g. featureless input).

    §0.4(b): the SSIM mean is restricted to pixels that are real content in
    *both* images — the naive whole-canvas mean previously included the
    warpAffine's border-replicated padding (wherever the ECC alignment
    shifted content off one edge) and any genuinely non-overlapping frame
    coverage, both of which measure framing/coverage differences rather than
    the pose/sharpness quality this metric exists to isolate. A test whose
    ASP output is more tightly cropped than its GT (a coverage difference,
    already scored elsewhere) no longer gets an extra unrelated SSIM penalty
    from comparing against replicated-edge filler.
    """
    if not _SSIM_OK:
        return float("nan")

    h, w = gt_img.shape[:2]
    resized_out = cv2.resize(output_img, (w, h))

    gray_gt = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
    gray_out = cv2.cvtColor(resized_out, cv2.COLOR_BGR2GRAY)

    # Real-content masks in the pre-warp/pre-alignment frame — anything a
    # cropped or letterboxed source leaves black on either side.
    valid_out = resized_out.max(axis=2) > 10
    valid_gt = gt_img.max(axis=2) > 10

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-4)

    def _overlap_mean(ssim_map: np.ndarray, valid: np.ndarray, fallback: float) -> float:
        if valid.sum() < 500:  # too little real overlap to trust a windowed mean
            return fallback
        return float(ssim_map[valid].mean())

    try:
        # pyrefly: ignore [no-matching-overload]
        _, warp_matrix = cv2.findTransformECC(
            gray_out,
            gray_gt,
            warp_matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
        aligned_out = cv2.warpAffine(
            resized_out,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        # Warp the validity mask with the *same* transform but zero-fill
        # borders (not replicate) — replicated padding must never count as
        # "real" content regardless of how the image itself was padded.
        aligned_valid = cv2.warpAffine(
            valid_out.astype(np.uint8) * 255,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 127
        overlap = aligned_valid & valid_gt

        gray_aligned = cv2.cvtColor(aligned_out, cv2.COLOR_BGR2GRAY)
        score, ssim_map = ssim(gray_aligned, gray_gt, full=True, data_range=255)
        return _overlap_mean(ssim_map, overlap, float(score))
    except Exception:
        overlap = valid_out & valid_gt
        score, ssim_map = ssim(gray_out, gray_gt, full=True, data_range=255)
        return _overlap_mean(ssim_map, overlap, float(score))


def _compute_gt_metrics(
    output_img: np.ndarray | None,
    gt_img: np.ndarray,
) -> dict:
    """
    Compute SSIM and PSNR between a pipeline output and the ground truth.

    Both images are resized to the smaller of the two dimensions before
    comparison, matching the existing _ssim_score / _psnr helpers.  Returns an
    empty dict when output_img is None.
    """
    if output_img is None:
        return {}
    ssim_val = _ssim_score(output_img, gt_img)
    aligned_ssim = _compute_aligned_ssim(output_img, gt_img)
    psnr_val = _psnr(output_img, gt_img)
    sc_val = _seam_coherence(output_img)
    return {
        "ssim_vs_gt": round(ssim_val, 4) if not math.isnan(ssim_val) else None,
        "aligned_ssim_vs_gt": round(aligned_ssim, 4)
        if not math.isnan(aligned_ssim)
        else None,
        "psnr_vs_gt": round(psnr_val, 2) if not math.isnan(psnr_val) else None,
        "seam_coherence": round(sc_val, 2),
    }


_HUMAN_RATINGS_CACHE: dict | None = None


def _load_human_evaluations() -> dict:
    """§0.1/0.2: load the most recent human coherence evaluations file (from
    ``evaluation_manager.py``), if any exist. Cached per-process — evaluations don't
    change mid-run. Schema: {test_name: {"asp": 0-4, "simple": 0-4, "notes": str}}.
    """
    global _HUMAN_RATINGS_CACHE
    if _HUMAN_RATINGS_CACHE is not None:
        return _HUMAN_RATINGS_CACHE
    evaluations_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "benchmarks",
    )
    files = sorted(glob.glob(os.path.join(evaluations_dir, "asp_evaluations_*.json")))
    if not files:
        _HUMAN_RATINGS_CACHE = {}
        return _HUMAN_RATINGS_CACHE
    with open(files[-1]) as fh:
        _HUMAN_RATINGS_CACHE = json.load(fh)
    return _HUMAN_RATINGS_CACHE


def _gt_verdict(
    asp_gt: dict,
    sim_gt: dict,
) -> str | None:
    """
    Quality verdict derived from ground truth SSIM comparison.

    Returns 'asp_better', 'simple_better', or 'comparable' when both outputs
    have GT SSIM scores.  Returns None when ground truth is unavailable.

    SSIM-vs-GT is a far more reliable signal than Laplacian sharpness because
    it measures structural similarity to the *intended* output, not the presence
    of high-frequency edge artifacts introduced by banding or misalignment.
    """
    asp_ssim = asp_gt.get("aligned_ssim_vs_gt", asp_gt.get("ssim_vs_gt"))
    sim_ssim = sim_gt.get("aligned_ssim_vs_gt", sim_gt.get("ssim_vs_gt"))
    if asp_ssim is None or sim_ssim is None:
        return None
    if asp_ssim > sim_ssim * 1.03:  # 3 % margin to avoid noise-driven flips
        return "asp_better"
    if sim_ssim > asp_ssim * 1.03:
        return "simple_better"
    return "comparable"


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================


def _save_affine_path_plot(
    affines: list[np.ndarray],
    canvas_h: int,
    canvas_w: int,
    frame_h: int,
    frame_w: int,
    out_path: str,
) -> None:
    """2-D plot of frame placement on the canvas."""
    if not _MPL_OK:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-50, canvas_w + 50)
    ax.set_ylim(canvas_h + 50, -50)
    ax.set_aspect("equal")
    ax.set_title("Frame Placement on Canvas (2D)", fontsize=11)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    colors = plt.cm.plasma(np.linspace(0, 1, len(affines)))  # pyrefly: ignore [missing-attribute]
    for idx, (M, color) in enumerate(zip(affines, colors, strict=False)):
        tx = float(M[0, 2])
        ty = float(M[1, 2])
        rect = plt.Rectangle(
            (tx, ty),
            frame_w,
            frame_h,
            linewidth=1.5,
            edgecolor=color,
            facecolor=(*color[:3], 0.08),
        )
        ax.add_patch(rect)
        ax.text(
            tx + frame_w / 2,
            ty + frame_h / 2,
            str(idx),
            ha="center",
            va="center",
            fontsize=7,
            color=color,
        )
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#12121f")
    ax.title.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, len(affines) - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Frame index", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


_PHASE_PALETTE = [
    (78, 205, 196), (107, 107, 255), (102, 209, 255), (160, 214, 6),
    (178, 138, 17), (111, 71, 239), (76, 59, 7), (150, 96, 131),
]


def _save_phase_strip_plot(
    frames_paths: list[str],
    phase_ids: list[int],
    out_path: str,
    thumb_h: int = 100,
) -> None:
    """§2.2: horizontal strip of selected-frame thumbnails with a colored bar
    under each tile marking its detected animation phase."""
    if not frames_paths:
        return
    tiles = []
    for path, pid in zip(frames_paths, phase_ids, strict=False):
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_h / max(h, 1)
        tile = cv2.resize(img, (max(1, int(w * scale)), thumb_h))
        color = _PHASE_PALETTE[pid % len(_PHASE_PALETTE)]
        bar = np.full((10, tile.shape[1], 3), color, dtype=np.uint8)
        tiles.append(np.vstack([tile, bar]))
    if not tiles:
        return
    gap = np.full((tiles[0].shape[0], 4, 3), 18, dtype=np.uint8)
    strip_parts: list[np.ndarray] = []
    for i, t in enumerate(tiles):
        if i > 0:
            strip_parts.append(gap)
        strip_parts.append(t)
    cv2.imwrite(out_path, np.hstack(strip_parts))


def _save_translation_plot(
    affines: list[np.ndarray],
    out_path: str,
    title: str = "Translation Vectors per Frame",
) -> None:
    """2-D plot of tx/ty translation per frame."""
    if not _MPL_OK:
        return
    N = len(affines)
    txs = [float(M[0, 2]) for M in affines]
    tys = [float(M[1, 2]) for M in affines]
    frames = list(range(N))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, label, color in zip(
        axes, [txs, tys], ["tx (horizontal)", "ty (vertical)"], ["#4ecdc4", "#ff6b6b"], strict=False
    ):
        ax.plot(frames, vals, marker="o", color=color, linewidth=2, markersize=5)
        ax.set_xlabel("Frame index")
        ax.set_ylabel(f"{label} (px)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#1a1a2e")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
    fig.suptitle(title, color="white")
    fig.patch.set_facecolor("#12121f")
    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_gains_plot(
    frame_lums: list[float | None],
    gains: list[float],
    out_path: str,
) -> None:
    """Bar chart of per-frame luminance gain corrections."""
    if not _MPL_OK:
        return
    N = len(gains)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    valid = [lum if lum is not None else 0.0 for lum in frame_lums]

    ax0, ax1 = axes
    ax0.bar(range(N), valid, color="#4ecdc4", alpha=0.8)
    ax0.axhline(
        float(np.median([v for v in valid if v > 0]))
        if any(v > 0 for v in valid)
        else 0,
        color="#ff6b6b",
        linestyle="--",
        label="median",
    )
    ax0.set_title("Background Luminance per Frame")
    ax0.set_xlabel("Frame index")
    ax0.set_ylabel("Mean luminance")
    ax0.legend(facecolor="#2a2a3e", labelcolor="white")

    ax1.bar(range(N), gains, color="#ff6b6b", alpha=0.8)
    ax1.axhline(1.0, color="#4ecdc4", linestyle="--", label="gain=1.0")
    ax1.set_title("Applied Luminance Gain per Frame")
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel("Gain multiplier")
    ax1.legend(facecolor="#2a2a3e", labelcolor="white")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
    fig.patch.set_facecolor("#12121f")
    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_seam_heatmap(img: np.ndarray, out_path: str, title: str = "") -> None:
    """2-D heatmap of gradient magnitude — highlights seam artefacts."""
    if not _MPL_OK:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # Compute magnitude of gradient
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    # Downsample for plotting
    scale = max(1, max(mag.shape) // 512)
    if scale > 1:
        mag = cv2.resize(mag, (mag.shape[1] // scale, mag.shape[0] // scale))
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mag, cmap="inferno", aspect="auto")
    ax.set_title(title or "Gradient Magnitude Heatmap", color="white")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03)
    cbar.set_label("Gradient magnitude", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    fig.patch.set_facecolor("#12121f")
    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_3d_surface(img: np.ndarray, out_path: str, title: str = "") -> None:
    """3-D surface plot of pixel luminance — reveals exposure ridges/valleys."""
    if not _MPL_OK:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # Aggressively downsample to keep rendering fast
    target = 96
    h, w = gray.shape
    sh = max(1, h // target)
    sw = max(1, w // target)
    small = gray[::sh, ::sw].astype(np.float32)
    # Smooth to reduce noise
    small = cv2.GaussianBlur(small, (5, 5), 0)
    Y, X = np.mgrid[0 : small.shape[0], 0 : small.shape[1]]
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X,
        Y,
        small,
        cmap="viridis",
        linewidth=0,
        antialiased=False,
        rstride=1,
        cstride=1,
        alpha=0.9,
    )
    ax.set_title(title or "Luminance Surface (3D)", color="white", pad=8)
    ax.set_xlabel("X (px ÷ " + str(sw) + ")", color="white", fontsize=7)
    ax.set_ylabel("Y (px ÷ " + str(sh) + ")", color="white", fontsize=7)
    ax.set_zlabel("Luma", color="white", fontsize=7)
    ax.tick_params(colors="white", labelsize=6)
    ax.xaxis.pane.fill = False  # pyrefly: ignore [missing-attribute]
    ax.yaxis.pane.fill = False  # pyrefly: ignore [missing-attribute]
    ax.zaxis.pane.fill = False  # pyrefly: ignore [missing-attribute]
    fig.patch.set_facecolor("#12121f")
    ax.set_facecolor("#1a1a2e")
    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_overlap_map(
    affines: list[np.ndarray],
    canvas_h: int,
    canvas_w: int,
    frame_h: int,
    frame_w: int,
    out_path: str,
) -> None:
    """2-D heatmap counting how many frames contribute to each canvas pixel."""
    if not _MPL_OK:
        return
    scale = max(1, max(canvas_h, canvas_w) // 512)
    ch = max(1, canvas_h // scale)
    cw = max(1, canvas_w // scale)
    acc = np.zeros((ch, cw), dtype=np.float32)
    for M in affines:
        tx = int(float(M[0, 2]) / scale)
        ty = int(float(M[1, 2]) / scale)
        fh = max(1, frame_h // scale)
        fw = max(1, frame_w // scale)
        r0, r1 = max(0, ty), min(ch, ty + fh)
        c0, c1 = max(0, tx), min(cw, tx + fw)
        if r1 > r0 and c1 > c0:
            acc[r0:r1, c0:c1] += 1.0
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(acc, cmap="hot", aspect="auto")
    ax.set_title("Frame Overlap Count Map (2D)", color="white")
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03)
    cbar.set_label("# overlapping frames", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    fig.patch.set_facecolor("#12121f")
    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray | None,
    out_path: str,
    title: str = "",
) -> None:
    """Visualize a foreground mask overlaid on the source frame."""
    if not _MPL_OK:
        return
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    overlay = rgb.copy().astype(np.float32)
    if mask is not None:
        fg = mask < 128  # fg pixels (BiRefNet: 0=foreground)
        overlay[fg, 0] = np.clip(overlay[fg, 0] * 0.4 + 200, 0, 255)
        overlay[fg, 1] = np.clip(overlay[fg, 1] * 0.4, 0, 255)
        overlay[fg, 2] = np.clip(overlay[fg, 2] * 0.4, 0, 255)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(overlay.astype(np.uint8), aspect="auto")
    ax.set_title(title or "FG mask overlay (red=foreground)", color="white")
    ax.axis("off")
    fig.patch.set_facecolor("#12121f")
    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_metrics_bar(metrics_asp: dict, metrics_simple: dict, out_path: str) -> None:
    """Side-by-side bar chart comparing key CV metrics for ASP vs simple."""
    if not _MPL_OK:
        return
    keys = ["sharpness", "coverage", "seam_gradient", "color_entropy", "ghosting_siqe"]
    labels = [
        "Sharpness",
        "Coverage",
        "Seam\nGradient",
        "Color\nEntropy",
        "Ghosting\n(SIQE)",
    ]
    asp_vals = [metrics_asp.get(k, 0) for k in keys]
    sim_vals = [metrics_simple.get(k, 0) for k in keys]
    # Normalize each metric to [0,1] for display
    maxes = [max(a, b, 1e-9) for a, b in zip(asp_vals, sim_vals, strict=False)]
    asp_n = [v / m for v, m in zip(asp_vals, maxes, strict=False)]
    sim_n = [v / m for v, m in zip(sim_vals, maxes, strict=False)]
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    b1 = ax.bar(x - width / 2, asp_n, width, label="ASP", color="#4ecdc4", alpha=0.85)
    b2 = ax.bar(
        x + width / 2, sim_n, width, label="Simple", color="#ff6b6b", alpha=0.85
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=9)
    ax.set_ylabel("Normalised value", color="white")
    ax.set_title("CV Metrics: ASP vs Simple Stitch (normalised)", color="white")
    ax.legend(facecolor="#2a2a3e", labelcolor="white")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    # Raw value annotations
    for bar, val in zip(b1, asp_vals, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#4ecdc4",
        )
    for bar, val in zip(b2, sim_vals, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#ff6b6b",
        )
    fig.patch.set_facecolor("#12121f")
    plt.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ============================================================================
# SMART FRAME SELECTOR
# ============================================================================
# The benchmark used to carry its own reimplementation of frame selection,
# separate from the production GUI pipeline (backend/src/animation/ingestion/
# frame_selection.py). That divergence meant flags like ASP_HOLD_AVERAGE could
# be toggled here without ever affecting the benchmarked code. Consolidated:
# the benchmark now calls the same smart_select_frames() the GUI stitch tab
# uses, so every future frame-selection flag is measured against this suite.


def _smart_select_frames(frames_paths: list[str]) -> list[str]:
    return smart_select_frames(frames_paths, min_step_px=50.0)


# ============================================================================
# SIMPLE STITCH (OpenCV SCANS)
# ============================================================================


def _run_simple_stitch(frames_paths: list[str], out_path: str) -> bool:
    """Generate OpenCV SCANS simple stitch and save. Returns True on success."""
    raw = [cv2.imread(p) for p in frames_paths]
    raw = [f for f in raw if f is not None]
    if len(raw) < 2:
        return False
    raw = _normalise_widths(raw)
    try:
        _scan_stitch_fallback(raw, out_path)
        return True
    except Exception as exc:
        print(f"  [Simple stitch] FAILED: {exc}")
        return False


# ============================================================================
# MAIN DATASET PROCESSOR
# ============================================================================


def _process_dataset_canonical(
    *,
    dataset_dir: str,
    dataset_name: str,
    frames_paths: list[str],
    out_path: str,
    simple_stitch_path: str,
    simple_ok: bool,
    central_anime_path: str,
    central_simple_path: str,
    stage_dir: str,
    plots_dir: str,
    timings: dict,
    stage_memory_rss_mb: dict[str, float],
    t_total_start: float,
    orig_frame_count: int,
    smart_select_count: int,
    gt_img,
    overmix_img,
    overmix_path: str,
    hugin_img,
    hugin_path: str,
) -> dict:
    """M1b: selected frames → product ``run()`` → Safe ASP policy → report."""
    apply_ungated_gate_env()
    raw_asp_path = os.path.join(stage_dir, "raw_asp.png")
    scans_path = simple_stitch_path if simple_ok else None
    _log_resource("before_canonical_run", store=stage_memory_rss_mb)
    t0 = time.perf_counter()
    result = run_canonical_asp(
        frames_paths,
        raw_asp_path=raw_asp_path,
        safe_asp_path=out_path,
        scans_path=scans_path,
        policy=default_benchmark_policy(),
    )
    timings["canonical_run_sec"] = round(time.perf_counter() - t0, 3)
    timings["total_sec"] = round(time.perf_counter() - t_total_start, 3)
    _log_resource("after_canonical_run", store=stage_memory_rss_mb)

    shutil.copy2(result.safe_asp_path, central_anime_path)
    central_raw = os.path.join(
        os.path.dirname(central_anime_path), f"{dataset_name}_raw_asp.png"
    )
    if result.raw_asp_available and result.raw_asp_path and os.path.isfile(
        result.raw_asp_path
    ):
        shutil.copy2(result.raw_asp_path, central_raw)

    session = result.session
    affines = []
    raw_aff = session.artifacts.get("affines")
    if isinstance(raw_aff, list):
        affines = [np.asarray(a, dtype=np.float32) for a in raw_aff]

    asp_img = cv2.imread(result.safe_asp_path)
    sim_img = cv2.imread(central_simple_path) if simple_ok else None
    canvas = session.artifacts.get("canvas_size") or [None, None]
    frame_count = int(session.artifacts.get("frame_count") or len(frames_paths))
    n_edges = int(session.artifacts.get("n_edges") or 0)

    probe = cv2.imread(frames_paths[0]) if frames_paths else None
    frame_h, frame_w = (probe.shape[0], probe.shape[1]) if probe is not None else (0, 0)

    try:
        phase_ids = detect_animation_phases(frames_paths)
        spans = phase_spans(phase_ids)
        phase_count = len(spans)
    except Exception:
        spans, phase_count = [], 0

    print(
        f"\nFinished ({result.identity}): {dataset_dir} -> {result.safe_asp_path}"
        f"{'  [fallback: ' + result.fallback_reason + ']' if result.used_fallback else ''}"
    )
    built = _build_result(
        dataset_name,
        central_anime_path,
        central_simple_path,
        asp_img,
        sim_img,
        affines,
        [],
        [],
        None,
        plots_dir,
        stage_dir,
        canvas_h=canvas[1] if isinstance(canvas, list) and len(canvas) == 2 else None,
        canvas_w=canvas[0] if isinstance(canvas, list) and len(canvas) == 2 else None,
        used_fallback=result.used_fallback,
        timings=timings,
        frame_count=frame_count,
        frame_h=frame_h,
        frame_w=frame_w,
        raw_edge_count=n_edges,
        filtered_edge_count=n_edges,
        birefnet_ok=bool(session.config.get("use_birefnet")),
        loftr_ok=bool(session.config.get("use_loftr")),
        gt_img=gt_img,
        fallback_reason=result.fallback_reason,
        orig_frame_count=orig_frame_count,
        smart_select_count=smart_select_count,
        spatial_dedup_count=frame_count,
        phase_count=phase_count,
        phase_spans_list=spans,
        overmix_img=overmix_img,
        overmix_path=(overmix_path if overmix_img is not None else None),
        hugin_img=hugin_img,
        hugin_path=(hugin_path if hugin_img is not None else None),
        stage_memory_rss_mb=stage_memory_rss_mb,
        photometric_telemetry=session.gain_telemetry,
    )
    built["raw_asp_path"] = result.raw_asp_path
    built["safe_asp_path"] = result.safe_asp_path
    built["result_identity"] = result.identity
    built["session_digest"] = result.session.digest()
    built["raw_asp_available"] = result.raw_asp_available
    built["safe_asp_counterfactual"] = result.extra.get("safe_asp_counterfactual")
    built["ungated_gate_config"] = result.extra.get("ungated_gate_config")
    built["observability"] = result.extra.get("observability") or session.observability()
    return built


def process_dataset(dataset_dir: str) -> dict | None:  # noqa: C901
    """
    Run both pipelines on a single dataset directory.

    Returns a dict of per-dataset results for the global report, or None if
    the dataset is skipped.
    """
    t_total_start = time.perf_counter()
    timings: dict[str, float] = {}
    stage_memory_rss_mb: dict[str, float] = {}  # §11.6 (issue #69)

    print(f"\n{'=' * 60}\nProcessing dataset: {dataset_dir}\n{'=' * 60}")
    _log_resource("dataset_start", store=stage_memory_rss_mb)

    dataset_name = os.path.basename(dataset_dir)
    stage_dir = os.path.join(dataset_dir, "output", "panorama_stages")
    out_path = os.path.join(dataset_dir, "output", "panorama.png")
    # "opencv_stitch", not "simple_stitch": there's now more than one ASP
    # alternative (Overmix, Hugin), so the OpenCV SCANS baseline needs a name
    # that says what it actually is. Only the filename changed — the
    # simple_stitch_path/central_simple_path *variable* names and every
    # downstream dict key (metrics_simple, simple_path, etc.) are untouched,
    # since renaming those has no user-visible benefit and a much larger blast
    # radius (they're read by discovery.py, the report, and every human_coherence
    # veto check). evaluation/other/discovery.py falls back to the old
    # "_simple_stitch.png" name for datasets already generated under it.
    simple_stitch_path = os.path.join(dataset_dir, "output", "opencv_stitch.png")
    plots_dir = os.path.join(dataset_dir, "output", "plots")

    # Central output
    central_out_dir = os.path.join(os.path.dirname(dataset_dir), "output")
    os.makedirs(central_out_dir, exist_ok=True)
    central_anime_path = os.path.join(
        central_out_dir, f"{dataset_name}_anime_stitch.png"
    )
    central_simple_path = os.path.join(
        central_out_dir, f"{dataset_name}_opencv_stitch.png"
    )

    # Ground truth (if available)
    gt_dir = os.path.join(os.path.dirname(dataset_dir), "ground_truth")
    gt_img = _load_ground_truth(dataset_name, gt_dir)
    if gt_img is not None:
        print(f"  [GT] Ground truth found for {dataset_name}: {gt_img.shape}")

    # §0.3 — Overmix reference comparator (external tool, generated ahead of
    # time by backend/benchmark/run_overmix.py, not invoked from here).
    # Picked up if present; a reference column only, never a gate/verdict input.
    _overmix_path = os.path.join(dataset_dir, "output", "overmix_stitch.png")
    overmix_img = cv2.imread(_overmix_path) if os.path.exists(_overmix_path) else None
    if overmix_img is not None:
        print(f"  [Overmix] Comparator output found for {dataset_name}: {overmix_img.shape}")

    # §0.5 — Hugin reference comparator (external tool via system
    # hugin-tools/enblend, generated ahead of time by
    # backend/benchmark/run_hugin.py, not invoked from here). Picked up if
    # present; a reference column only, never a gate/verdict input.
    _hugin_path = os.path.join(dataset_dir, "output", "hugin_stitch.png")
    hugin_img = cv2.imread(_hugin_path) if os.path.exists(_hugin_path) else None
    if hugin_img is not None:
        print(f"  [Hugin] Comparator output found for {dataset_name}: {hugin_img.shape}")

    # Clean old outputs
    if os.path.exists(out_path):
        os.remove(out_path)
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Collect frames
    all_pngs = sorted(
        glob.glob(os.path.join(dataset_dir, "*.png"))
        + glob.glob(os.path.join(dataset_dir, "*.jpg"))
    )
    frames_paths = [
        p
        for p in all_pngs
        if "panorama" not in os.path.basename(p)
        and "test_" not in os.path.basename(p)
        and "stage" not in os.path.basename(p)
    ]
    if len(frames_paths) < 2:
        print(f"Skipping {dataset_dir}: not enough frames.")
        return None

    # Smart frame selection: drop near-duplicates and backward-direction frames
    # before any GPU processing.  Large datasets can have 50-330 consecutive
    # video frames; naive stride subsampling would miss character-animation
    # conflicts where the character returns to the same pose as a previous
    # selected frame but the camera is now in a different position.
    _orig_frame_count = len(frames_paths)
    frames_paths = _smart_select_frames(frames_paths)
    _smart_select_count = len(frames_paths)
    if _smart_select_count < _orig_frame_count:
        print(
            f"  Smart selection: {_orig_frame_count} → {_smart_select_count} frames "
            f"({_orig_frame_count - _smart_select_count} dropped)."
        )

    print(f"Source frames ({len(frames_paths)}):")
    for p in frames_paths:
        print(f"  {os.path.basename(p)}")

    # ------------------------------------------------------------------
    # STEP 0: Generate simple stitch (always regenerate for consistency)
    # ------------------------------------------------------------------
    print("\n[0] Running OpenCV SCANS simple stitch …")
    t0 = time.perf_counter()
    simple_ok = _run_simple_stitch(frames_paths, simple_stitch_path)
    timings["simple_stitch_sec"] = round(time.perf_counter() - t0, 3)
    if simple_ok:
        shutil.copy2(simple_stitch_path, central_simple_path)
        print(f"  Saved: {simple_stitch_path}")
    else:
        print(f"  Warning: simple stitch failed for {dataset_name}")

    if not bench_legacy_enabled():
        print("\n[M1b] Canonical AnimeStitchPipeline.run() + Safe ASP policy")
        built = _process_dataset_canonical(
            dataset_dir=dataset_dir,
            dataset_name=dataset_name,
            frames_paths=frames_paths,
            out_path=out_path,
            simple_stitch_path=simple_stitch_path,
            simple_ok=simple_ok,
            central_anime_path=central_anime_path,
            central_simple_path=central_simple_path,
            stage_dir=stage_dir,
            plots_dir=plots_dir,
            timings=timings,
            stage_memory_rss_mb=stage_memory_rss_mb,
            t_total_start=t_total_start,
            orig_frame_count=_orig_frame_count,
            smart_select_count=_smart_select_count,
            gt_img=gt_img,
            overmix_img=overmix_img,
            overmix_path=_overmix_path,
            hugin_img=hugin_img,
            hugin_path=_hugin_path,
        )
        return built

    # ------------------------------------------------------------------
    # STEP 1-2: Load and normalise  (ASP_BENCH_LEGACY=1)
    # ------------------------------------------------------------------
    frames = _load_frames(frames_paths)
    N = len(frames)
    frames = _normalise_widths(frames)
    H, W = frames[0].shape[:2]
    scans_frames = list(frames)  # pre-ML snapshot for SCANS fallback
    _fallback_reason: str | None = None  # set by whichever gate triggers SCANS
    _mean_post_warp_diff: float | None = None  # §0.4, set after Stage 11 composite
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(stage_dir, f"stage02_normalised_frame{i:02d}.png"), f)

    # ------------------------------------------------------------------
    # STEP 3: BiRefNet foreground masks
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    birefnet_ok = False
    _log_resource("before_birefnet", store=stage_memory_rss_mb)
    try:
        from backend.src.models.wrappers.birefnet_wrapper import BiRefNetWrapper

        birefnet = BiRefNetWrapper()
        if _USE_SAM2:
            # §10: route through the pipeline's own ASP_USE_SAM2-aware
            # masking, mirroring AnimeStitchPipeline._compute_fg_masks --
            # this benchmark previously called the raw BiRefNet-only
            # _compute_fg_masks() unconditionally, silently ignoring the
            # flag (see docs/moon/ROADMAP.md's 2026-08-07 entry, issue #10).
            bg_masks, _sam2_pred, _sam2_state, _sam2_tmp, _, _ = (
                _compute_fg_masks_sam2_stateful(frames, birefnet, use_birefnet=True)
            )
            _cleanup_sam2_state(_sam2_pred, _sam2_state, _sam2_tmp)
        else:
            bg_masks = _compute_fg_masks(frames, birefnet)
        birefnet_ok = True
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                birefnet.offload()
        del birefnet
        gc.collect()
    except Exception as e:
        print(f"  BiRefNet failed ({e}), using None masks")
        bg_masks = [None] * N
    timings["birefnet_sec"] = round(time.perf_counter() - t0, 3)
    _log_resource("after_birefnet_offload", store=stage_memory_rss_mb)

    for i, m in enumerate(bg_masks):
        img = m if m is not None else np.ones((H, W), dtype=np.uint8) * 255
        cv2.imwrite(os.path.join(stage_dir, f"stage04_bgmask_frame{i:02d}.png"), img)

    # Visualise mask overlays for first 3 frames
    for i in range(min(3, N)):
        _save_mask_overlay(
            frames[i],
            bg_masks[i],
            os.path.join(plots_dir, f"mask_overlay_frame{i:02d}.png"),
            title=f"FG Mask Overlay — Frame {i}",
        )

    # ------------------------------------------------------------------
    # STEP 4: Background photometric normalisation (luminance scalar gain)
    # ------------------------------------------------------------------
    _LUM_W = np.array([0.114, 0.587, 0.299], dtype=np.float32)
    bg_frame_lums: list[float | None] = []
    for frame, mask in zip(frames, bg_masks, strict=False):
        if mask is not None:
            bg_px = frame[mask > 127].astype(np.float32)
            if len(bg_px) >= 1000:
                bg_frame_lums.append(float(bg_px.dot(_LUM_W).mean()))
                continue
        bg_frame_lums.append(None)

    valid_lums = [lum for lum in bg_frame_lums if lum is not None]
    applied_gains = [1.0] * N
    if len(valid_lums) >= 3:
        ref_lum = float(np.median(valid_lums))
        _gain_lo, _gain_hi = (0.80, 1.25) if ref_lum < 80.0 else (0.88, 1.14)
        for i in range(N):
            if bg_frame_lums[i] is None:
                continue
            gain = float(
                np.clip(ref_lum / max(bg_frame_lums[i], 1.0), _gain_lo, _gain_hi) # pyrefly: ignore [bad-specialization, unsupported-operation]
            )
            applied_gains[i] = gain
            if abs(gain - 1.0) > 0.01:
                hsv = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * gain, 0, 255)
                frames[i] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Save stage 3 corrected frames
    for i, f in enumerate(frames):
        cv2.imwrite(
            os.path.join(stage_dir, f"stage03_basic_corrected_frame{i:02d}.png"), f
        )

    # Gains plot
    _save_gains_plot(
        bg_frame_lums,
        applied_gains,
        os.path.join(plots_dir, "gains.png"),
    )

    # ------------------------------------------------------------------
    # STEP 5-7: Match → filter → bundle-adjust → ECC
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    loftr_ok = False
    _log_resource("before_loftr", store=stage_memory_rss_mb)
    try:
        from backend.src.models.wrappers.loftr_wrapper import LoFTRWrapper

        loftr = LoFTRWrapper()
        loftr_ok = True
    except Exception:
        loftr = None

    edges = _pairwise_match(frames, bg_masks, loftr_wrapper=loftr) # pyrefly: ignore [bad-argument-type]
    if loftr is not None:
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                loftr.offload()
        del loftr
        gc.collect()
    timings["matching_sec"] = round(time.perf_counter() - t0, 3)
    _log_resource("after_loftr_offload", store=stage_memory_rss_mb)

    # Collect edge metadata before filtering
    raw_edge_count = len(edges)
    edge_methods: dict[str, int] = {}
    for e in edges:
        m = e.get("method", "unknown")
        edge_methods[m] = edge_methods.get(m, 0) + 1

    # ── Post-match: Spatial dedup of near-static consecutive frames ──────────
    _SPATIAL_DEDUP_PX = 25
    _spa_changed = True
    _total_spa_dropped = 0
    while _spa_changed:
        _spa_changed = False
        _adj_m = {e["j"]: e for e in edges if e["j"] == e["i"] + 1}
        if not _adj_m:
            break
        _adx = [abs(float(e["M"][0, 2])) for e in _adj_m.values()]
        _ady = [abs(float(e["M"][1, 2])) for e in _adj_m.values()]
        _spa_axis = 0 if float(np.median(_adx)) > float(np.median(_ady)) else 1
        _drop: set = set()
        for _jj in sorted(_adj_m):
            _ee = _adj_m[_jj]
            if _ee["i"] in _drop:
                continue
            if abs(float(_ee["M"][_spa_axis, 2])) < _SPATIAL_DEDUP_PX:
                _drop.add(_jj)
                _spa_changed = True
                print(
                    f"  Spatial dedup: frame {_jj} ≈ frame {_ee['i']} "
                    f"(d{'x' if _spa_axis == 0 else 'y'}="
                    f"{float(_ee['M'][_spa_axis, 2]):.1f}px) — dropped."
                )
        if _drop:
            _total_spa_dropped += len(_drop)
            _keep_idx = [i for i in range(N) if i not in _drop]
            frames = [frames[i] for i in _keep_idx]
            bg_masks = [bg_masks[i] for i in _keep_idx]
            frames_paths = [frames_paths[i] for i in _keep_idx]
            _o2n = {old: new for new, old in enumerate(_keep_idx)}
            edges = [
                {**e, "i": _o2n[e["i"]], "j": _o2n[e["j"]]}
                for e in edges
                if e["i"] not in _drop and e["j"] not in _drop
            ]
            N = len(frames)
            H, W = frames[0].shape[:2]
            if N < 2:
                print(
                    f"  Spatial dedup removed too many frames; skipping {dataset_dir}."
                )
                return None
    if _total_spa_dropped:
        print(
            f"  Spatial dedup complete: {_total_spa_dropped} frames removed, {N} remain."
        )

    # §2.2: animation-phase clustering — measurement-only unless
    # ASP_PHASE_COMPOSITE=1 (compositing.py reads that flag itself). Computed
    # here, after spatial dedup, so phase_ids indices stay aligned with the
    # final frames/frames_paths/affines that compositing actually uses —
    # spatial dedup above can drop frames by index, which would otherwise
    # desync a phase_ids list computed on the pre-dedup selection.
    _phase_ids = detect_animation_phases(frames_paths)
    _phase_spans = phase_spans(_phase_ids)
    _phase_count = len(_phase_spans)
    print(
        f"  [PhaseDetect] {_phase_count} animation phase(s) across "
        f"{len(frames_paths)} selected frames "
        f"(spans={[(p, a, b) for p, a, b in _phase_spans]})"
    )
    _save_phase_strip_plot(
        frames_paths, _phase_ids, os.path.join(plots_dir, "animation_phases.png")
    )

    t0 = time.perf_counter()
    pipe = AnimeStitchPipeline(
        use_basic=False, use_birefnet=False, use_loftr=False, use_ecc=False
    )
    print(f"  [PostMatch] filtering {len(edges)} edges and starting bundle adjustment...")
    edges = pipe._filter_edges(edges, frames_paths, H, W, frames, bg_masks) # pyrefly: ignore [bad-argument-type]
    affines = _bundle_adjust_affine(edges, N)
    timings["bundle_adjust_sec"] = round(time.perf_counter() - t0, 3)
    print(f"  [PostMatch] bundle adjustment complete ({timings['bundle_adjust_sec']:.1f}s).")

    filtered_edge_count = len(edges)
    edge_stats = [
        {
            "i": int(e["i"]),
            "j": int(e["j"]),
            "method": e.get("method", "unknown"),
            "weight": round(float(e.get("weight", 0.0)), 4),
            "n_pts": len(e.get("pts_i", [])),
            "tx": round(float(e["M"][0, 2]), 2),
            "ty": round(float(e["M"][1, 2]), 2),
        }
        for e in edges
    ]

    # Validate affines
    health = _validate_affines(affines)
    print(
        f"  Affine health: valid={health.valid}, reason={health.reason}, "
        f"ratio={health.ratio:.2f}, min_gap={health.min_gap:.1f}px"
    )

    if not health.valid:
        print(f"  Validation FAILED ({health.reason}); attempting recovery...")
        # Retry 1: consecutive-only bundle
        _adj_only = [e for e in edges if e["j"] == e["i"] + 1]
        if len(_adj_only) >= N - 1:
            affines_r1 = _bundle_adjust_affine(_adj_only, N)
            health_r1 = _validate_affines(affines_r1)
            if health_r1.valid:
                affines, health = affines_r1, health_r1
                print(f"  Recovery Retry 1 succeeded: {health.reason}")

        # Retry 2: smart sequential + fill
        if not health.valid:
            _adj_only_r2 = [e for e in edges if e["j"] == e["i"] + 1]
            _step_dx = (
                float(np.median([float(e["M"][0, 2]) for e in _adj_only_r2]))
                if _adj_only_r2
                else 0.0
            )
            _step_dy = (
                float(np.median([float(e["M"][1, 2]) for e in _adj_only_r2]))
                if _adj_only_r2
                else 0.0
            )
            _has_adj_src = {e["j"] for e in _adj_only_r2}
            _seq = [np.eye(2, 3, dtype=np.float32) for _ in range(N)]
            _anchored: set = {0}
            for _f in range(1, N):
                _best_e, _best_span = None, float("inf")
                for _e in edges:
                    if _e["j"] == _f and _e["i"] in _anchored and _f - _e["i"] < _best_span:
                        _best_span = _f - _e["i"]
                        _best_e = _e
                if _best_e is not None:
                    _seq[_f][0, 2] = _seq[_best_e["i"]][0, 2] - float(
                        _best_e["M"][0, 2]
                    )
                    _seq[_f][1, 2] = _seq[_best_e["i"]][1, 2] - float(
                        _best_e["M"][1, 2]
                    )
                    _anchored.add(_f)
            for _uf in sorted(i for i in range(N) if i not in _anchored):
                if _uf in _has_adj_src:
                    continue
                _lft = max((a for a in _anchored if a < _uf), default=None)
                _rgt = min((a for a in _anchored if a > _uf), default=None)
                if _lft is not None and _rgt is not None:
                    _t = (_uf - _lft) / (_rgt - _lft)
                    _seq[_uf][0, 2] = (
                        _seq[_lft][0, 2] * (1 - _t) + _seq[_rgt][0, 2] * _t
                    )
                    _seq[_uf][1, 2] = (
                        _seq[_lft][1, 2] * (1 - _t) + _seq[_rgt][1, 2] * _t
                    )
                elif _lft is not None:
                    _n = _uf - _lft
                    _seq[_uf][0, 2] = _seq[_lft][0, 2] - _n * _step_dx
                    _seq[_uf][1, 2] = _seq[_lft][1, 2] - _n * _step_dy
                _anchored.add(_uf)
            _chg = True
            while _chg:
                _chg = False
                for _f in range(1, N):
                    if _f in _anchored:
                        continue
                    _best_e, _best_span = None, float("inf")
                    for _e in edges:
                        if _e["j"] == _f and _e["i"] in _anchored and _f - _e["i"] < _best_span:
                            _best_span = _f - _e["i"]
                            _best_e = _e
                    if _best_e is not None:
                        _seq[_f][0, 2] = _seq[_best_e["i"]][0, 2] - float(
                            _best_e["M"][0, 2]
                        )
                        _seq[_f][1, 2] = _seq[_best_e["i"]][1, 2] - float(
                            _best_e["M"][1, 2]
                        )
                        _anchored.add(_f)
                        _chg = True
            health_r2 = _validate_affines(_seq)
            if health_r2.valid:
                affines, health = _seq, health_r2
                print(f"  Recovery Retry 2 succeeded: {health.reason}")
            else:
                health_r3 = _validate_affines(_seq, min_step=20.0)
                if health_r3.valid:
                    affines, health = _seq, health_r3
                    print(f"  Recovery Retry 3 (relaxed) succeeded: {health.reason}")
                else:
                    # Retry 4: very permissive — only reject truly co-located frames
                    # (min_gap < 3px) or extreme clustering (ratio > 10x).
                    # Needed for slow-pan sequences with many fine-grained frames.
                    health_r4 = _validate_affines(
                        _seq,
                        min_step=3.0,
                        max_ratio=10.0,
                        max_rotation=0.3,
                        max_scale_dev=0.3,
                    )
                    if health_r4.valid:
                        affines, health = _seq, health_r4
                        print(
                            f"  Recovery Retry 4 (permissive) succeeded: {health.reason}"
                        )
                    else:
                        print(
                            f"  Recovery Retry 4 failed: {health_r4.reason} "
                            f"(ratio={health_r4.ratio:.2f} min_gap={health_r4.min_gap:.1f}px)"
                        )
                        # Retry 5: final attempt — accept any _seq with non-zero gaps
                        health_r5 = _validate_affines(
                            _seq,
                            min_step=0.5,
                            max_ratio=50.0,
                            max_rotation=0.5,
                            max_scale_dev=0.5,
                        )
                        if health_r5.valid:
                            affines, health = _seq, health_r5
                            print(
                                f"  Recovery Retry 5 (final) succeeded: {health.reason}"
                            )

    if not health.valid:
        _fallback_reason = f"alignment_failed:{health.reason}"
        print("  Validation FAILED → SCANS fallback.")
        t0 = time.perf_counter()
        _scan_stitch_fallback(scans_frames, out_path)
        timings["scans_fallback_sec"] = round(time.perf_counter() - t0, 3)
        timings["total_sec"] = round(time.perf_counter() - t_total_start, 3)
        shutil.copy2(out_path, central_anime_path)
        print(f"\nFinished (SCANS): {dataset_dir} -> {out_path}")
        asp_img = cv2.imread(central_anime_path)
        sim_img = cv2.imread(central_simple_path) if simple_ok else None
        return _build_result(
            dataset_name,
            central_anime_path,
            central_simple_path,
            asp_img,
            sim_img,
            affines,
            bg_frame_lums,
            applied_gains,
            health,
            plots_dir,
            stage_dir,
            canvas_h=None,
            canvas_w=None,
            used_fallback=True,
            timings=timings,
            frame_count=N,
            frame_h=H,
            frame_w=W,
            raw_edge_count=raw_edge_count,
            filtered_edge_count=filtered_edge_count,
            edge_methods=edge_methods,
            edge_stats=edge_stats,
            birefnet_ok=birefnet_ok,
            loftr_ok=loftr_ok,
            gt_img=gt_img,
            fallback_reason=_fallback_reason,
            orig_frame_count=_orig_frame_count,
            smart_select_count=_smart_select_count,
            spatial_dedup_count=N,
            phase_count=_phase_count,
            phase_spans_list=_phase_spans,
            overmix_img=overmix_img,
            overmix_path=(_overmix_path if overmix_img is not None else None),
            hugin_img=hugin_img,
            hugin_path=(_hugin_path if hugin_img is not None else None),
            stage_memory_rss_mb=stage_memory_rss_mb,
        )

    try:
        # ECC refinement
        t0 = time.perf_counter()
        print("  [PostMatch] starting ECC refinement...")
        affines = _ecc_refine(frames, affines, bg_masks) # pyrefly: ignore [bad-argument-type]
        timings["ecc_sec"] = round(time.perf_counter() - t0, 3)
        print(f"  [PostMatch] ECC refinement complete ({timings['ecc_sec']:.1f}s).")

        # Canvas construction
        canvas_h, canvas_w, T_global = _compute_canvas(frames, affines)
        for i in range(N):
            affines[i][0, 2] += T_global[0]
            affines[i][1, 2] += T_global[1]

        canvas_info = {
            "canvas_h": canvas_h,
            "canvas_w": canvas_w,
            "affines_final": [a.tolist() for a in affines],
        }
        with open(os.path.join(stage_dir, "stage08_canvas_info.json"), "w") as fh:
            json.dump(canvas_info, fh)

        # Canvas visualisations
        _save_affine_path_plot(
            affines,
            canvas_h,
            canvas_w,
            H,
            W,
            os.path.join(plots_dir, "canvas_frame_placement.png"),
        )
        _save_translation_plot(
            affines,
            os.path.join(plots_dir, "translation_vectors.png"),
            title=f"{dataset_name} — Translation Vectors",
        )
        _save_overlap_map(
            affines,
            canvas_h,
            canvas_w,
            H,
            W,
            os.path.join(plots_dir, "overlap_map.png"),
        )

        # ── Alignment stability gate (advisory) ──────────────────────────
        # Log the horizontal drift but do NOT abort — the composite render
        # gate below uses a SCANS-relative quality comparison and will catch
        # any genuinely degraded output regardless of the motion pattern.
        # The old hard-abort was over-triggering on scenes where ASP quality
        # is actually comparable to or better than SCANS despite diagonal motion.
        # Override: ASP_ALIGN_GATE_DX=99 to suppress the log entirely.
        try:
            _ALIGN_DX_LIMIT = float(os.environ.get("ASP_ALIGN_GATE_DX", "50"))
        except ValueError:
            _ALIGN_DX_LIMIT = 50.0
        _txs_raw = [float(affines[i][0, 2]) for i in range(N)]
        _dx_raw = [abs(_txs_raw[i + 1] - _txs_raw[i]) for i in range(N - 1)]
        if _dx_raw:
            _dx_p75 = float(np.percentile(_dx_raw, 75))
            _align_flag = _dx_p75 > _ALIGN_DX_LIMIT
            print(
                f"  [AlignGate] 75th-pct |dx|={_dx_p75:.1f}px  "
                f"limit={_ALIGN_DX_LIMIT:.0f}px  {'⚠ high drift' if _align_flag else 'ok'}"
            )

        # ------------------------------------------------------------------
        # STEP 8-10: Render → quality gate → composite → crop
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        _log_resource("before_render_median", store=stage_memory_rss_mb)
        print("  [PostMatch] starting temporal median render...")
        canvas, valid_mask, _, _ = _render_median(
            frames, affines, bg_masks, canvas_h, canvas_w # pyrefly: ignore [bad-argument-type]
        )
        timings["render_sec"] = round(time.perf_counter() - t0, 3)
        print(f"  [PostMatch] temporal median render complete ({timings['render_sec']:.1f}s).")
        _log_resource("after_render_median", store=stage_memory_rss_mb)
        cv2.imwrite(os.path.join(stage_dir, "stage09_temporal_render.png"), canvas)

        # Run the full foreground-assembly composite (Stage 11) — this applies
        # the bg-only scalar gain correction AND the flow-guided foreground
        # re-posing (Stage 8.5) + single-pose fallback.
        t0 = time.perf_counter()
        _seam_meta: dict = {}
        print("  [PostMatch] starting foreground composite...")
        canvas = _composite_foreground(
            [], [], canvas, canvas_h, canvas_w, frames, affines, bg_masks, # pyrefly: ignore [bad-argument-type]
            phase_ids=_phase_ids, seam_meta_out=_seam_meta,
        )
        timings["composite_sec"] = round(time.perf_counter() - t0, 3)
        print(f"  [PostMatch] foreground composite complete ({timings['composite_sec']:.1f}s).")
        _log_resource("after_composite", store=stage_memory_rss_mb)

        # §0.4 — seam-band pose-residual stats: lower mean post_warp_diff
        # across seams means the frame selection handed compositing an
        # easier job (less pose gap to bridge). Sentinel values (98/99) mark
        # single-pose escalations (phase boundary / user override) rather
        # than a measured warp residual — excluded from the mean so it
        # reflects genuine re-posing difficulty, not escalation counts.
        _seam_diffs = _seam_meta.get("seam_post_diffs", {}) or {}
        _real_diffs = [v for v in _seam_diffs.values() if v < 90.0]
        _mean_post_warp_diff = (
            float(np.mean(_real_diffs)) if _real_diffs else None
        )
        cv2.imwrite(os.path.join(stage_dir, "stage11_fg_composite.png"), canvas)

        # ── Composite / Ghost / SeamVis gates (M1b injectable policy) ─────
        # Formulae and reason strings are unchanged; they now live in
        # asp_backend.core.pipeline.safety_policy so M2 can inject the same
        # object into the canonical runner. Raw ASP is still written above
        # (stage11_fg_composite.png) before any fallback.
        _safe_policy = default_benchmark_policy()
        _scans_img_gate = cv2.imread(simple_stitch_path) if simple_ok else None
        _cg = _safe_policy.evaluate_composite(canvas, _scans_img_gate, affines)
        if _cg.log_line:
            print(_cg.log_line)
        if not _cg.accept:
            _fallback_reason = _cg.reason
            if _cg.fail_log_line:
                print(_cg.fail_log_line)
            timings["render_gate_fallback"] = _cg.fallback_code
            raise RuntimeError(_cg.runtime_message)

        timings["render_gate_fallback"] = 0

        canvas_out = _crop_to_valid(canvas, valid_mask)
        ec = 30
        if ec * 2 < canvas_out.shape[0] and ec * 2 < canvas_out.shape[1]:
            canvas_out = canvas_out[ec:-ec, ec:-ec]

        # Note: content-aware crop was removed — cropping based on fg union
        # across a vertical pan incorrectly cuts horizontal extent (the lockers
        # background) rather than trimming excess top/bottom panning extent.
        # The scale mismatch for test27 (2× larger than GT) is a fundamental
        # frame-selection issue, not a post-processing crop problem.

        # GhostGate + SeamVisGate (post-crop). Same env knobs as before.
        _simple_img_gate = cv2.imread(central_simple_path) if simple_ok else None
        _gg = _safe_policy.evaluate_ghost(canvas_out, _simple_img_gate)
        if _gg.log_line:
            print(_gg.log_line)
        if not _gg.accept:
            _fallback_reason = _gg.reason
            if _gg.fail_log_line:
                print(_gg.fail_log_line)
            timings["render_gate_fallback"] = _gg.fallback_code
            raise RuntimeError(_gg.runtime_message)

        _svg = _safe_policy.evaluate_seam_vis(canvas_out, _simple_img_gate)
        if _svg.log_line:
            print(_svg.log_line)
        if not _svg.accept:
            _fallback_reason = _svg.reason
            if _svg.fail_log_line:
                print(_svg.fail_log_line)
            timings["render_gate_fallback"] = _svg.fallback_code
            raise RuntimeError(_svg.runtime_message)
        from PIL import Image

        rgb = cv2.cvtColor(canvas_out, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(out_path)
        shutil.copy2(out_path, central_anime_path)
        print(f"\nFinished: {dataset_dir} -> {out_path}")

    except Exception as _render_exc:
        gc.collect()
        print(f"  ASP render/ECC failed ({_render_exc}); falling back to SCANS.")
        if _fallback_reason is None:
            _fallback_reason = f"render_exception:{type(_render_exc).__name__}"
        t0 = time.perf_counter()
        _scan_stitch_fallback(scans_frames, out_path)
        timings["scans_fallback_sec"] = round(time.perf_counter() - t0, 3)
        timings["total_sec"] = round(time.perf_counter() - t_total_start, 3)
        shutil.copy2(out_path, central_anime_path)
        print(f"\nFinished (SCANS): {dataset_dir} -> {out_path}")
        asp_img = cv2.imread(central_anime_path)
        sim_img = cv2.imread(central_simple_path) if simple_ok else None
        return _build_result(
            dataset_name,
            central_anime_path,
            central_simple_path,
            asp_img,
            sim_img,
            affines,
            bg_frame_lums,
            applied_gains,
            health,
            plots_dir,
            stage_dir,
            canvas_h=None,
            canvas_w=None,
            used_fallback=True,
            timings=timings,
            frame_count=N,
            frame_h=H,
            frame_w=W,
            raw_edge_count=raw_edge_count,
            filtered_edge_count=filtered_edge_count,
            edge_methods=edge_methods,
            edge_stats=edge_stats,
            birefnet_ok=birefnet_ok,
            loftr_ok=loftr_ok,
            gt_img=gt_img,
            fallback_reason=_fallback_reason,
            orig_frame_count=_orig_frame_count,
            smart_select_count=_smart_select_count,
            spatial_dedup_count=N,
            phase_count=_phase_count,
            phase_spans_list=_phase_spans,
            mean_post_warp_diff=_mean_post_warp_diff,
            overmix_img=overmix_img,
            overmix_path=(_overmix_path if overmix_img is not None else None),
            hugin_img=hugin_img,
            hugin_path=(_hugin_path if hugin_img is not None else None),
            stage_memory_rss_mb=stage_memory_rss_mb,
        )

    # ------------------------------------------------------------------
    # Visualisations on final images
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    asp_img = cv2.imread(central_anime_path)
    sim_img = cv2.imread(central_simple_path) if simple_ok else None

    if asp_img is not None:
        _save_seam_heatmap(
            asp_img,
            os.path.join(plots_dir, "asp_seam_heatmap.png"),
            title="ASP — Gradient Magnitude Heatmap",
        )
        _save_3d_surface(
            asp_img,
            os.path.join(plots_dir, "asp_3d_surface.png"),
            title="ASP — Luminance Surface (3D)",
        )
    if sim_img is not None:
        _save_seam_heatmap(
            sim_img,
            os.path.join(plots_dir, "simple_seam_heatmap.png"),
            title="Simple Stitch — Gradient Magnitude Heatmap",
        )
        _save_3d_surface(
            sim_img,
            os.path.join(plots_dir, "simple_3d_surface.png"),
            title="Simple Stitch — Luminance Surface (3D)",
        )

    # Temporal render visualisation
    render_img = cv2.imread(os.path.join(stage_dir, "stage09_temporal_render.png"))
    if render_img is not None:
        _save_3d_surface(
            render_img,
            os.path.join(plots_dir, "temporal_render_3d.png"),
            title="Stage 9 — Temporal Render Luminance (3D)",
        )

    # Metrics comparison bar
    if asp_img is not None and sim_img is not None:
        _save_metrics_bar(
            _compute_all_metrics(asp_img, affines),
            _compute_all_metrics(sim_img),
            os.path.join(plots_dir, "metrics_comparison.png"),
        )

    timings["visualisations_sec"] = round(time.perf_counter() - t0, 3)
    timings["total_sec"] = round(time.perf_counter() - t_total_start, 3)
    _log_resource("dataset_end", store=stage_memory_rss_mb)

    return _build_result(
        dataset_name,
        central_anime_path,
        central_simple_path,
        asp_img,
        sim_img,
        affines,
        bg_frame_lums,
        applied_gains,
        health,
        plots_dir,
        stage_dir,
        canvas_h,
        canvas_w,
        used_fallback=False,
        timings=timings,
        frame_count=N,
        frame_h=H,
        frame_w=W,
        raw_edge_count=raw_edge_count,
        filtered_edge_count=filtered_edge_count,
        edge_methods=edge_methods,
        edge_stats=edge_stats,
        birefnet_ok=birefnet_ok,
        loftr_ok=loftr_ok,
        gt_img=gt_img,
        fallback_reason=None,
        orig_frame_count=_orig_frame_count,
        smart_select_count=_smart_select_count,
        spatial_dedup_count=N,
        phase_count=_phase_count,
        phase_spans_list=_phase_spans,
        mean_post_warp_diff=_mean_post_warp_diff,
        overmix_img=overmix_img,
        overmix_path=(_overmix_path if overmix_img is not None else None),
        hugin_img=hugin_img,
        hugin_path=(_hugin_path if hugin_img is not None else None),
        stage_memory_rss_mb=stage_memory_rss_mb,
    )


# ============================================================================
# RESULT BUILDER
# ============================================================================


def _production_photometric_result(payload: dict) -> dict:
    """Serialize the canonical pipeline's Stage 4.5 telemetry verbatim enough
    for existing benchmark readers while retaining the per-frame production
    record. No gain is recomputed in the benchmark adapter."""
    rows = list(payload.get("frames") or [])
    scalar_gains = [
        round(float(np.mean(row["gain_bgr"])), 4)
        for row in rows
        if isinstance(row.get("gain_bgr"), list)
    ]
    return {
        "source": "production_stage",
        "ref_lum": payload.get("reference_luminance"),
        "bg_lums": [row.get("background_luminance") for row in rows],
        "applied_gains": scalar_gains,
        "frames_corrected": int(payload.get("applied_gain_count") or 0),
        "gain_range": [min(scalar_gains), max(scalar_gains)] if scalar_gains else None,
        "eligible_mask_count": int(payload.get("eligible_mask_count") or 0),
        "n_clamped": int(payload.get("n_clamped") or 0),
        "mean_residual": payload.get("mean_residual"),
        "frames": rows,
    }


def _build_result(
    dataset_name: str,
    anime_path: str,
    simple_path: str,
    asp_img: np.ndarray | None,
    sim_img: np.ndarray | None,
    affines: list[np.ndarray],
    bg_frame_lums: list[float | None],
    applied_gains: list[float],
    health,
    plots_dir: str,
    stage_dir: str,
    canvas_h: int | None,
    canvas_w: int | None,
    used_fallback: bool,
    timings: dict | None = None,
    frame_count: int = 0,
    frame_h: int = 0,
    frame_w: int = 0,
    raw_edge_count: int = 0,
    filtered_edge_count: int = 0,
    edge_methods: dict | None = None,
    edge_stats: list | None = None,
    birefnet_ok: bool = False,
    loftr_ok: bool = False,
    gt_img: np.ndarray | None = None,
    fallback_reason: str | None = None,
    orig_frame_count: int = 0,
    smart_select_count: int = 0,
    spatial_dedup_count: int = 0,
    phase_count: int = 0,
    phase_spans_list: list | None = None,
    mean_post_warp_diff: float | None = None,
    overmix_img: np.ndarray | None = None,
    overmix_path: str | None = None,
    hugin_img: np.ndarray | None = None,
    hugin_path: str | None = None,
    stage_memory_rss_mb: dict[str, float] | None = None,
    photometric_telemetry: dict | None = None,
) -> dict:
    asp_metrics = _compute_all_metrics(asp_img, affines) if asp_img is not None else {}
    sim_metrics = _compute_all_metrics(sim_img) if sim_img is not None else {}
    # §0.3/§0.5 — Overmix and Hugin are reference comparator columns, not
    # gates: neither participates in the asp-vs-simple verdict logic below.
    overmix_metrics = _compute_all_metrics(overmix_img) if overmix_img is not None else {}
    hugin_metrics = _compute_all_metrics(hugin_img) if hugin_img is not None else {}

    ssim_val = float("nan")
    psnr_val = float("nan")
    if asp_img is not None and sim_img is not None:
        ssim_val = _ssim_score(asp_img, sim_img)
        psnr_val = _psnr(asp_img, sim_img)

    # Ground truth comparison
    gt_metrics_asp: dict = (
        _compute_gt_metrics(asp_img, gt_img) if gt_img is not None else {}
    )
    gt_metrics_sim: dict = (
        _compute_gt_metrics(sim_img, gt_img) if gt_img is not None else {}
    )
    gt_ver = _gt_verdict(gt_metrics_asp, gt_metrics_sim)
    has_gt = gt_img is not None

    if not has_gt:
        try:
            from evaluation.si_fid import compute_si_fid
            if asp_img is not None:
                gt_metrics_asp["si_fid"] = compute_si_fid(asp_img)
            if sim_img is not None:
                gt_metrics_sim["si_fid"] = compute_si_fid(sim_img)
        except ImportError:
            pass

    # §0.2 — human-coherence-aware verdict: no automated metric measures
    # structural coherence, so when a evaluation exists, it may veto a false
    # "asp_better" that a metric-only read would otherwise report (the
    # test84/test53/test07 class of failure the critical evaluation names).
    # One-directional by design (matches the roadmap spec literally) — a
    # human "asp better" preference does not force-upgrade a metric verdict,
    # only a metric "asp_better" that the human disagreed with gets vetoed.
    human_evaluations = _load_human_evaluations()
    human_coherence = human_evaluations.get(dataset_name)
    verdict = gt_ver if gt_ver is not None else _auto_verdict(asp_metrics, sim_metrics)
    verdict_source = "ground_truth" if gt_ver is not None else "cv_metrics"
    if human_coherence is not None:
        h_asp = human_coherence.get("asp")
        h_sim = human_coherence.get("simple")
        if (
            h_asp is not None
            and h_sim is not None
            and h_asp < h_sim
            and verdict == "asp_better"
        ):
            verdict = "simple_better"
            verdict_source = "human_coherence_veto"

    # Affine translation summary for JSON
    affine_translations = [
        {
            "frame": i,
            "tx": round(float(M[0, 2]), 2),
            "ty": round(float(M[1, 2]), 2),
            "a": round(float(M[0, 0]), 5),
            "b": round(float(M[0, 1]), 5),
        }
        for i, M in enumerate(affines)
    ]

    # Inter-frame deltas
    tys = [float(M[1, 2]) for M in affines]
    txs = [float(M[0, 2]) for M in affines]
    dy_steps = [round(tys[i + 1] - tys[i], 2) for i in range(len(tys) - 1)]
    dx_steps = [round(txs[i + 1] - txs[i], 2) for i in range(len(txs) - 1)]
    dy_cv = (
        float(np.std(dy_steps) / (abs(np.mean(dy_steps)) + 1e-6)) if dy_steps else 0.0
    )
    dx_cv = (
        float(np.std(dx_steps) / (abs(np.mean(dx_steps)) + 1e-6)) if dx_steps else 0.0
    )

    # Background luminance stats
    valid_lums = [lum for lum in bg_frame_lums if lum is not None]
    ref_lum = round(float(np.median(valid_lums)), 2) if valid_lums else None
    non_trivial_gains = sum(1 for g in applied_gains if abs(g - 1.0) > 0.01)

    photometric = {
        "source": "legacy_harness",
        "ref_lum": ref_lum,
        "bg_lums": [
            round(lum, 2) if lum is not None else None for lum in bg_frame_lums
        ],
        "applied_gains": [round(g, 4) for g in applied_gains],
        "frames_corrected": non_trivial_gains,
        "gain_range": (
            [round(min(applied_gains), 4), round(max(applied_gains), 4)]
            if applied_gains
            else None
        ),
    }
    if photometric_telemetry is not None:
        photometric = _production_photometric_result(photometric_telemetry)

    return {
        "name": dataset_name,
        "anime_path": anime_path,
        "simple_path": simple_path,
        # --- timing ---
        "time": timings or {},
        # --- frame / canvas geometry ---
        "frames": {
            "count": frame_count,
            "source_h": frame_h,
            "source_w": frame_w,
        },
        # --- §2.2 animation-phase diagnostics (measurement-only) ---
        "phases": {
            "count": phase_count,
            "spans": [
                {"phase": p, "start": a, "end": b}
                for p, a, b in (phase_spans_list or [])
            ],
        },
        # --- §0.4 seam-band pose-residual (mean post_warp_diff, excludes
        # single-pose-escalation sentinels) — lower means an easier
        # compositing job was handed down from frame selection ---
        "mean_post_warp_diff": mean_post_warp_diff,
        "canvas": {
            "width": canvas_w,
            "height": canvas_h,
        },
        # --- pipeline config ---
        "pipeline_config": {
            "use_birefnet": birefnet_ok,
            "use_loftr": loftr_ok,
            "use_basic": False,
            "use_ecc": True,
            "renderer": "median",
            "edge_erosion_px": 30,
        },
        # --- matching ---
        "matching": {
            "raw_edges": raw_edge_count,
            "filtered_edges": filtered_edge_count,
            "methods": edge_methods or {},
            "edges": edge_stats or [],
        },
        # --- alignment ---
        "alignment": {
            "affines": affine_translations,
            "dy_steps": dy_steps,
            "dx_steps": dx_steps,
            "dy_cv": round(dy_cv, 4),
            "dx_cv": round(dx_cv, 4),
        },
        "affine_health": {
            "valid": getattr(health, "valid", None),
            "ratio": round(float(getattr(health, "ratio", 0.0) or 0.0), 3),
            "min_gap_px": round(float(getattr(health, "min_gap", 0.0) or 0.0), 1),
            "max_rotation": round(float(getattr(health, "max_rotation", 0.0) or 0.0), 4),
            "max_scale_dev": round(
                float(getattr(health, "max_scale_dev", 0.0) or 0.0), 4
            ),
            "reason": getattr(health, "reason", "canonical_adapter"),
        },
        # --- photometric correction ---
        "photometric": photometric,
        # --- quality metrics ---
        "metrics_asp": asp_metrics,
        "metrics_simple": sim_metrics,
        # --- §0.3 Overmix reference comparator (external tool, GPL-3.0, run
        # via backend/benchmark/run_overmix.py — never a gate/verdict input) ---
        "metrics_overmix": overmix_metrics,
        "overmix_path": overmix_path,
        # --- §0.5 Hugin reference comparator (external tool via system
        # hugin-tools/enblend, run via backend/benchmark/run_hugin.py —
        # never a gate/verdict input) ---
        "metrics_hugin": hugin_metrics,
        "hugin_path": hugin_path,
        "comparison": {
            "ssim": round(ssim_val, 4) if not math.isnan(ssim_val) else None,
            "psnr_db": round(psnr_val, 2) if not math.isnan(psnr_val) else None,
            # GT-based verdict when available (most reliable); CV-based
            # otherwise; §0.2 human-coherence veto applied on top of either.
            "verdict": verdict,
            "verdict_source": verdict_source,
        },
        # --- ground truth comparison ---
        "ground_truth": {
            "available": has_gt,
            "metrics_asp": gt_metrics_asp,
            "metrics_simple": gt_metrics_sim,
            "verdict": gt_ver,
        },
        # --- §0.1/0.2 human coherence evaluations (None if this dataset hasn't
        # been rated — see backend/controllers/bench_eval_dispatch.py) ---
        "human_coherence": human_coherence,
        # --- status ---
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        # --- frame selection telemetry ---
        "frame_selection": {
            "original_count": orig_frame_count,
            "smart_select_count": smart_select_count,
            "spatial_dedup_count": spatial_dedup_count,
            "final_count": frame_count,
            "frames_dropped_smart": max(0, orig_frame_count - smart_select_count),
            "frames_dropped_dedup": max(0, smart_select_count - spatial_dedup_count),
            "selection_mode": (
                "dinov2"
                if os.environ.get("ASP_POSE_WINDOW_PX", "0") != "0"
                else "phase_correlation"
            ),
        },
        # --- paths (for the notebook to locate files) ---
        "paths": {
            "plots_dir": plots_dir,
            "stage_dir": stage_dir,
            "anime_stitch": anime_path,
            "simple_stitch": simple_path,
        },
        # --- §11.6 (issue #69) stage-level RSS, keyed by _log_resource() tag,
        # in the order the pipeline actually visited them ---
        "stage_memory_rss_mb": stage_memory_rss_mb or {},
        # --- §11.10 (issue #69) experiment tag for this run, e.g.
        # "S44-seam-cache" — set ASP_EXPERIMENT_LABEL before running to tag a
        # batch for the comparison table in _report_experiment_comparison().
        # Unset by default, so untagged runs don't clutter that section.
        "experiment_label": os.environ.get("ASP_EXPERIMENT_LABEL") or None,
    }


# ============================================================================
# JSON RESULTS FILE
# ============================================================================


def generate_json_results(results: list[dict], suite_start_time: float) -> str:
    """
    Write a structured JSON results file to backend/benchmark/output/ and
    return the path.  Schema mirrors the existing benchmark JSON files.
    """
    total_sec = round(time.perf_counter() - suite_start_time, 3)
    ts = datetime.datetime.now()
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    ts_iso = ts.isoformat()

    results_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"anime_stitch_{ts_str}.json")

    # Aggregate summary stats
    asp_sharpness = [
        r["metrics_asp"].get("sharpness", 0.0) for r in results if r["metrics_asp"]
    ]
    sim_sharpness = [
        r["metrics_simple"].get("sharpness", 0.0)
        for r in results
        if r["metrics_simple"]
    ]
    asp_ghosting = [
        r["metrics_asp"].get("ghosting_siqe", 0.0) for r in results if r["metrics_asp"]
    ]
    sim_ghosting = [
        r["metrics_simple"].get("ghosting_siqe", 0.0)
        for r in results
        if r["metrics_simple"]
    ]
    asp_coverage = [
        r["metrics_asp"].get("coverage", 0.0) for r in results if r["metrics_asp"]
    ]
    ssim_vals = [
        r["comparison"]["ssim"]
        for r in results
        if r["comparison"].get("ssim") is not None
    ]
    dataset_times = [r["time"].get("total_sec", 0.0) for r in results]
    fallback_count = sum(1 for r in results if r["used_fallback"])
    verdicts = [r["comparison"]["verdict"] for r in results]

    # Performance insights
    def _rank_by(key_fn, results_list, top=True):
        valid = [(r["name"], key_fn(r)) for r in results_list if key_fn(r) is not None]
        if not valid:
            return None
        ranked = sorted(valid, key=lambda x: x[1], reverse=top)
        return {"name": ranked[0][0], "value": round(ranked[0][1], 4)}

    best_asp = _rank_by(lambda r: r["metrics_asp"].get("sharpness"), results)
    worst_asp = _rank_by(
        lambda r: r["metrics_asp"].get("sharpness"), results, top=False
    )
    slowest = _rank_by(lambda r: r["time"].get("total_sec"), results)
    fastest = _rank_by(
        lambda r: r["time"].get("total_sec")
        if r["time"].get("total_sec", 0) > 0
        else None,
        results,
        top=False,
    )
    most_ghosting = _rank_by(lambda r: r["metrics_asp"].get("ghosting_siqe"), results)
    least_ghosting = _rank_by(
        lambda r: r["metrics_asp"].get("ghosting_siqe"), results, top=False
    )

    doc = {
        "metadata": {
            "suite_name": "Anime Stitch Pipeline",
            "timestamp": ts_iso,
            "total_datasets": len(results),
            "total_time_sec": total_sec,
            "format_version": "1.0",
            "experiment_label": os.environ.get("ASP_EXPERIMENT_LABEL") or None,
        },
        "system": _system_info(),
        "summary": {
            "total_datasets": len(results),
            "datasets_passed": len(results) - fallback_count,
            "datasets_fallback": fallback_count,
            "total_time_sec": total_sec,
            "avg_time_per_dataset_sec": round(
                sum(dataset_times) / max(len(dataset_times), 1), 3
            ),
            "avg_sharpness_asp": round(float(np.mean(asp_sharpness)), 3)
            if asp_sharpness
            else None,
            "avg_sharpness_simple": round(float(np.mean(sim_sharpness)), 3)
            if sim_sharpness
            else None,
            "avg_ghosting_asp": round(float(np.mean(asp_ghosting)), 4)
            if asp_ghosting
            else None,
            "avg_ghosting_simple": round(float(np.mean(sim_ghosting)), 4)
            if sim_ghosting
            else None,
            "avg_coverage_asp": round(float(np.mean(asp_coverage)), 4)
            if asp_coverage
            else None,
            "avg_ssim": round(float(np.mean(ssim_vals)), 4) if ssim_vals else None,
            "verdict_counts": {
                "asp_better": verdicts.count("asp_better"),
                "simple_better": verdicts.count("simple_better"),
                "comparable": verdicts.count("comparable"),
                "insufficient_data": verdicts.count("insufficient_data"),
            },
            # Ground truth summary
            "datasets_with_ground_truth": sum(
                1 for r in results if r.get("ground_truth", {}).get("available")
            ),
            "gt_verdict_counts": {
                "asp_better": sum(
                    1
                    for r in results
                    if r.get("ground_truth", {}).get("verdict") == "asp_better"
                ),
                "simple_better": sum(
                    1
                    for r in results
                    if r.get("ground_truth", {}).get("verdict") == "simple_better"
                ),
                "comparable": sum(
                    1
                    for r in results
                    if r.get("ground_truth", {}).get("verdict") == "comparable"
                ),
            },
            "avg_ssim_asp_vs_gt": round(
                float(
                    np.mean(
                        [
                            r["ground_truth"]["metrics_asp"]["ssim_vs_gt"]
                            for r in results
                            if r.get("ground_truth", {}).get("available")
                            and r["ground_truth"]["metrics_asp"].get("ssim_vs_gt")
                            is not None
                        ]
                    )
                ),
                4,
            )
            if any(r.get("ground_truth", {}).get("available") for r in results)
            else None,
            "avg_ssim_simple_vs_gt": round(
                float(
                    np.mean(
                        [
                            r["ground_truth"]["metrics_simple"]["ssim_vs_gt"]
                            for r in results
                            if r.get("ground_truth", {}).get("available")
                            and r["ground_truth"]["metrics_simple"].get("ssim_vs_gt")
                            is not None
                        ]
                    )
                ),
                4,
            )
            if any(r.get("ground_truth", {}).get("available") for r in results)
            else None,
            # §0.1/0.2 — human coherence evaluation coverage for this run
            "human_coherence_rated": sum(
                1 for r in results if r.get("human_coherence") is not None
            ),
            "human_coherence_veto_count": sum(
                1
                for r in results
                if r.get("comparison", {}).get("verdict_source")
                == "human_coherence_veto"
            ),
        },
        "datasets": results,
        "performance_insights": {
            "slowest_dataset": slowest,
            "fastest_dataset": fastest,
            "best_asp_sharpness": best_asp,
            "worst_asp_sharpness": worst_asp,
            "most_asp_ghosting": most_ghosting,
            "least_asp_ghosting": least_ghosting,
            "datasets_asp_better_than_simple": [
                r["name"] for r in results if r["comparison"]["verdict"] == "asp_better"
            ],
            "datasets_simple_better_than_asp": [
                r["name"]
                for r in results
                if r["comparison"]["verdict"] == "simple_better"
            ],
            "datasets_alignment_failed": [
                r["name"] for r in results if not r["affine_health"]["valid"]
            ],
        },
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)

    print(f"\n[JSON] Results written to {out_path}")
    try:
        from merge_run_json import maybe_write_consolidated

        merged = maybe_write_consolidated(Path(out_path))
        if merged is not None:
            print(f"[JSON] Consolidated range merge written to {merged}")
    except Exception as exc:
        print(f"[JSON] Consolidated merge skipped: {exc!r}")
    return out_path


# ============================================================================
# MARKDOWN REPORT GENERATOR
# ============================================================================

_REPORT_HEADER = """\
---
report_version: "1.0"
generated: "{date}"
pipeline: "AnimeStitchPipeline"
datasets: {num_datasets}
---

# Anime Stitch Pipeline — Benchmark Report

> **How to use this report**
>
> Each test section contains:
> - Side-by-side outputs (ASP vs Simple/OpenCV)
> - CV metric table
> - Intermediate output visualizations (2D and 3D)
> - A structured `<!-- FEEDBACK -->` block
>
> To review/correct feedback, edit the YAML inside each `<!-- FEEDBACK -->…<!-- /FEEDBACK -->`
> block. Valid `status` values: `pending`, `correct`, `incomplete`, `incorrect`.
> Add your corrections in the `human_notes` field.
> Machine-readable fields (`asp_issues`, `simple_issues`, `verdict`) are pre-filled
> and updated automatically on re-runs.

"""

_GLOBAL_SUMMARY_HEADER = """\
---

## Global Summary

"""

_GLOBAL_FEEDBACK_BLOCK = """\

---

## Global Feedback and Human Notes

<!-- GLOBAL_FEEDBACK
status: pending
overall_asp_evaluation: null
overall_simple_evaluation: null
most_common_asp_failure: null
most_common_simple_failure: null
priority_fixes:
  - null
human_notes: |
  (Your analysis here)
/GLOBAL_FEEDBACK -->

"""

_PER_TEST_HUMAN_SECTION = """\

### My Feedback

<!-- FEEDBACK
status: pending
asp_issues:
{asp_issues}
simple_issues:
{simple_issues}
verdict: "{verdict}"
human_notes: |
  (Edit this section — confirm, correct, or extend the CV analysis above)
/FEEDBACK -->

---
"""


def _auto_verdict(asp_m: dict, sim_m: dict) -> str:
    """
    Quality verdict using seam_coherence as the primary discriminator.

    ``cqas_v1_legacy`` / historical ``cqas`` are intentionally not read here:
    the aggregate failed the completed human-label audit and is diagnostic-only.

    Laplacian sharpness is NOT used as a primary signal because hard seam edges
    inflate it, making catastrophically banded ASP outputs appear "sharper" than
    clean simple-stitch results.  Instead:

      - seam_coherence (row-mean luminance std): lower = more coherent.
        If ASP seam_coherence > 28 (severe banding), simple_better.
        If both are low, use coverage and ghosting as tiebreaker.
      - seam_gradient (gradient at seam rows): lower = smoother seams.
      - coverage: higher = more useful canvas area.
      - ghosting_siqe: lower = fewer periodic double-edge (ghost) artifacts.
    """
    if not asp_m or not sim_m:
        return "insufficient_data"

    asp_sc = asp_m.get("seam_coherence", 0.0)
    sim_sc = sim_m.get("seam_coherence", 0.0)

    # If ASP has severe banding (high seam_coherence) → simple is better
    if asp_sc > 28.0 and asp_sc > sim_sc * 1.5:
        return "simple_better"

    # §5.5: seam_visibility penalty (strip banding term)
    asp_sv = asp_m.get("seam_visibility") or 0.0
    sim_sv = sim_m.get("seam_visibility") or 0.0
    asp_sv_score = float(np.clip(1.0 - asp_sv / 25.0, 0.0, 1.0))
    sim_sv_score = float(np.clip(1.0 - sim_sv / 25.0, 0.0, 1.0))

    # Composite quality score: penalise banding and ghosting, reward coverage
    asp_score = (
        asp_m.get("coverage", 0) * 100 * 0.4
        - asp_sc * 0.3
        - asp_m.get("seam_gradient", 0) * 0.15
        - asp_m.get("ghosting_siqe", 0) * 0.15
        + asp_sv_score * 0.10
    )
    sim_score = (
        sim_m.get("coverage", 0) * 100 * 0.4
        - sim_sc * 0.3
        - sim_m.get("seam_gradient", 0) * 0.15
        - sim_m.get("ghosting_siqe", 0) * 0.15
        + sim_sv_score * 0.10
    )
    if asp_score > sim_score * 1.1:
        return "asp_better"
    if sim_score > asp_score * 1.1:
        return "simple_better"
    return "comparable"


def _auto_issues(metrics: dict, is_asp: bool) -> list[str]:
    """Generate a list of detected issues from metrics."""
    issues = []
    if not metrics:
        return ["- no_image"]
    cov = metrics.get("coverage", 1.0)
    if cov < 0.70:
        issues.append(
            f"  - low_coverage: {cov:.2%} (image heavily cropped or malformed)"
        )
    ghost = metrics.get("ghosting_siqe", 0)
    if ghost > 30:
        issues.append(f"  - high_ghosting: siqe={ghost:.2f} (periodic double-edges)")
    seam = metrics.get("seam_gradient", 0)
    if seam > 20:
        issues.append(
            f"  - seam_discontinuity: gradient={seam:.2f} (abrupt transitions)"
        )
    sc = metrics.get("seam_coherence", 0.0)
    if sc > 28.0:
        issues.append(
            f"  - color_banding: seam_coherence={sc:.1f} (severe horizontal strip color mismatch)"
        )
    elif sc > 18.0:
        issues.append(
            f"  - mild_banding: seam_coherence={sc:.1f} (visible color variation between strips)"
        )
    if not issues:
        issues.append("  - none_detected")
    return issues


def _rel_path(path: str, report_dir: str) -> str:
    """Return path relative to report_dir for markdown embedding."""
    try:
        return os.path.relpath(path, report_dir)
    except ValueError:
        return path


def _plot_exists(plots_dir: str, name: str) -> bool:
    return os.path.exists(os.path.join(plots_dir, name))


def _report_header_and_summary(results: list[dict], lines: list[str]) -> None:
    # Header
    lines.append(
        _REPORT_HEADER.format(
            date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            num_datasets=len(results),
        )
    )

    # Global summary table
    lines.append(_GLOBAL_SUMMARY_HEADER)
    lines.append(
        "| Test | SC ASP | SC Sim | SC OM | SC HG | GT SSIM ASP | GT SSIM Sim | Align SSIM ASP | Align SSIM Sim | Verdict | Src | FB |\n"
    )
    lines.append(
        "|------|-------:|-------:|------:|------:|------------:|------------:|---------------:|---------------:|---------|-----|----|\n"
    )
    for r in results:
        am, sm = r["metrics_asp"], r["metrics_simple"]
        om = r.get("metrics_overmix") or {}
        hg = r.get("metrics_hugin") or {}
        sc_a = f"{am.get('seam_coherence', 0):.1f}" if am else "—"
        sc_s = f"{sm.get('seam_coherence', 0):.1f}" if sm else "—"
        sc_om = f"{om.get('seam_coherence', 0):.1f}" if om else "—"
        sc_hg = f"{hg.get('seam_coherence', 0):.1f}" if hg else "—"
        gt = r.get("ground_truth", {})
        gt_ssim_a = gt.get("metrics_asp", {}).get("ssim_vs_gt")
        gt_ssim_s = gt.get("metrics_simple", {}).get("ssim_vs_gt")
        gt_ssim_a_s = f"{gt_ssim_a:.3f}" if gt_ssim_a is not None else "—"
        gt_ssim_s_s = f"{gt_ssim_s:.3f}" if gt_ssim_s is not None else "—"

        align_ssim_a = gt.get("metrics_asp", {}).get("aligned_ssim_vs_gt")
        align_ssim_s = gt.get("metrics_simple", {}).get("aligned_ssim_vs_gt")
        align_ssim_a_s = f"{align_ssim_a:.3f}" if align_ssim_a is not None else "—"
        align_ssim_s_s = f"{align_ssim_s:.3f}" if align_ssim_s is not None else "—"

        verdict = r["comparison"]["verdict"]
        vsrc = r["comparison"].get("verdict_source", "cv")[:2].upper()
        fallback = "✓" if r["used_fallback"] else ""
        lines.append(
            f"| [{r['name']}](#{r['name']}) | {sc_a} | {sc_s} | {sc_om} | {sc_hg} | {gt_ssim_a_s} | {gt_ssim_s_s} | "
            f"{align_ssim_a_s} | {align_ssim_s_s} | {verdict} | {vsrc} | {fallback} |\n"
        )
    lines.append("\n")
    lines.append(
        "*SC = seam_coherence (lower is better); GT SSIM = raw SSIM; Align SSIM = ECC-aligned SSIM (no framing bias)*\n\n"
    )


def _report_fail_breakdown(results: list[dict], lines: list[str]) -> None:
    # Global ASP failure breakdown
    lines.append("### Failure Mode Counts (ASP)\n\n")
    fail_counts: dict[str, int] = {}
    for r in results:
        for issue in _auto_issues(r["metrics_asp"], is_asp=True):
            key = issue.strip().lstrip("- ").split(":")[0]
            fail_counts[key] = fail_counts.get(key, 0) + 1
    lines.append("| Issue | Count |\n|-------|-------|\n")
    for k, v in sorted(fail_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| `{k}` | {v} |\n")
    lines.append("\n")


# §11.7 — roadmap-defined threshold: flag any dataset whose smart-select
# stage alone drops more than this fraction of its original frame count
# (indicates extreme frame redundancy or a selection bug).
_FRAME_SELECT_DROP_FLAG_PCT = 40.0


def _report_frame_selection_telemetry(
    results: list[dict], output_dir: str, lines: list[str]
) -> None:
    """
    §11.7 dashboard — Frame Selection Telemetry.

    Shows, per dataset, frames kept vs dropped at each reduction stage
    (original -> smart-select -> spatial-dedup -> final), aggregated once
    near the top of the report (not per-test). Flags datasets where smart
    selection alone drops >40% of frames per the roadmap's own threshold.
    """
    lines.append("### Frame Selection Telemetry (§11.7)\n\n")

    rows = []
    for r in results:
        fs = r.get("frame_selection") or {}
        orig = fs.get("original_count", 0) or 0
        smart = fs.get("smart_select_count", 0) or 0
        dedup = fs.get("spatial_dedup_count", 0) or 0
        final = fs.get("final_count", 0) or 0
        drop_smart = fs.get("frames_dropped_smart", max(0, orig - smart))
        drop_dedup = fs.get("frames_dropped_dedup", max(0, smart - dedup))
        smart_drop_pct = (drop_smart / orig * 100.0) if orig else 0.0
        rows.append(
            {
                "name": r["name"],
                "orig": orig,
                "smart": smart,
                "dedup": dedup,
                "final": final,
                "drop_smart": drop_smart,
                "drop_dedup": drop_dedup,
                "smart_drop_pct": smart_drop_pct,
                "mode": fs.get("selection_mode", "—"),
            }
        )

    # Optional stacked-bar PNG, precedent: per-test gains.png/animation_phases.png.
    # Categorical order/colors follow the dataviz palette's validated adjacent
    # ordering (blue/orange/aqua/yellow); the accompanying table below is the
    # "relief" for the two low-contrast fills (aqua, yellow).
    chart_rel = None
    if _MPL_OK and rows:
        try:
            chart_path = os.path.join(output_dir, "frame_selection_telemetry.png")
            names = [row["name"] for row in rows]
            kept_smart = [row["smart"] for row in rows]
            dropped_smart = [row["drop_smart"] for row in rows]
            kept_dedup = [row["dedup"] for row in rows]
            dropped_dedup = [row["drop_dedup"] for row in rows]

            fig, ax = plt.subplots(figsize=(max(6.0, len(rows) * 0.7), 5))
            x = np.arange(len(rows))
            width = 0.35
            ax.bar(
                x - width / 2, kept_smart, width,
                label="kept (smart-select)", color="#2a78d6",
                edgecolor="white", linewidth=1,
            )
            ax.bar(
                x - width / 2, dropped_smart, width, bottom=kept_smart,
                label="dropped (smart-select)", color="#eb6834",
                edgecolor="white", linewidth=1,
            )
            ax.bar(
                x + width / 2, kept_dedup, width,
                label="kept (spatial-dedup)", color="#1baf7a",
                edgecolor="white", linewidth=1,
            )
            ax.bar(
                x + width / 2, dropped_dedup, width, bottom=kept_dedup,
                label="dropped (spatial-dedup)", color="#eda100",
                edgecolor="white", linewidth=1,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("Frame count")
            ax.set_title("Frame Selection Telemetry — kept vs dropped per stage")
            ax.legend(fontsize=7, loc="upper right")
            plt.tight_layout()
            fig.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            chart_rel = os.path.basename(chart_path)
        except Exception:
            chart_rel = None

    if chart_rel:
        lines.append(f"![Frame Selection Telemetry]({chart_rel})\n\n")

    lines.append(
        "| Test | Original | Smart-Select | Spatial-Dedup | Final | "
        "Dropped (smart) | Dropped (dedup) | Smart-drop % | Mode |\n"
    )
    lines.append(
        "|------|---------:|-------------:|---------------:|------:|"
        "-----------------:|------------------:|-------------:|------|\n"
    )
    flagged = []
    for row in rows:
        pct_str = f"{row['smart_drop_pct']:.1f}%"
        if row["smart_drop_pct"] > _FRAME_SELECT_DROP_FLAG_PCT:
            flagged.append(row["name"])
            pct_str = f"**{pct_str}**"
        lines.append(
            f"| [{row['name']}](#{row['name']}) | {row['orig']} | {row['smart']} | "
            f"{row['dedup']} | {row['final']} | {row['drop_smart']} | "
            f"{row['drop_dedup']} | {pct_str} | {row['mode']} |\n"
        )
    lines.append("\n")
    if flagged:
        lines.append(
            f"> ⚠️ **{len(flagged)} dataset(s) drop more than "
            f"{_FRAME_SELECT_DROP_FLAG_PCT:.0f}% of frames at the smart-select "
            "stage** (indicates extreme frame redundancy or a selection bug): "
            + ", ".join(f"[{n}](#{n})" for n in flagged)
            + "\n\n"
        )
    else:
        lines.append(
            f"> No dataset drops more than {_FRAME_SELECT_DROP_FLAG_PCT:.0f}% "
            "of frames at the smart-select stage.\n\n"
        )


# §11.8 — the exact gate-classification prefixes emitted by the pipeline's
# `_fallback_reason = ...` assignments (see fallback trigger sites in the
# main dataset loop above). Listed first, in this order, so the count table
# always shows all five even when a given run triggers zero of them.
_KNOWN_FALLBACK_GATES = [
    "alignment_failed",
    "composite_gate_sc",
    "composite_gate_sb",
    "ghost_gate_siqe",
    "render_exception",
]


def _report_fallback_breakdown(results: list[dict], lines: list[str]) -> None:
    """
    §11.8 dashboard — Fallback Root Cause Breakdown.

    Classifies every SCANS fallback in the run by its trigger gate (the
    `fallback_reason` string's prefix before the first ':'), shows an
    aggregate count table, then lists which specific datasets hit each gate
    type, linked to their per-test `#asp_testNN` anchor.
    """
    lines.append("### Fallback Root Cause Breakdown (§11.8)\n\n")

    by_gate: dict[str, list[str]] = {}
    total_fallbacks = 0
    for r in results:
        if not r.get("used_fallback"):
            continue
        total_fallbacks += 1
        reason = r.get("fallback_reason") or "unknown"
        gate = reason.split(":", 1)[0]
        by_gate.setdefault(gate, []).append(r["name"])

    all_gates = list(_KNOWN_FALLBACK_GATES)
    for gate in sorted(by_gate):
        if gate not in all_gates:
            all_gates.append(gate)

    lines.append(
        f"Total fallbacks in this run: **{total_fallbacks}** / {len(results)} datasets.\n\n"
    )
    lines.append("| Gate | Count |\n|------|------:|\n")
    for gate in all_gates:
        lines.append(f"| `{gate}` | {len(by_gate.get(gate, []))} |\n")
    lines.append("\n")

    if not by_gate:
        lines.append("_No fallbacks triggered in this run._\n\n")
        return

    lines.append("**Datasets by gate:**\n\n")
    for gate in all_gates:
        names = by_gate.get(gate)
        if not names:
            continue
        links = ", ".join(f"[{n}](#{n})" for n in names)
        lines.append(f"- `{gate}` ({len(names)}): {links}\n")
    lines.append("\n")


def _report_stage_memory_waterfall(
    results: list[dict], output_dir: str, lines: list[str]
) -> None:
    """
    §11.6 dashboard — Stage-Level Memory Profiling.

    Averages each dataset's `stage_memory_rss_mb` (populated by
    `_log_resource(tag, store=...)` inside `process_dataset`) across every
    tag in `STAGE_MEMORY_ORDER`, then renders a waterfall: each bar's base
    sits at the running RSS and its height is the delta from the previous
    stage, so a stage that leaks stands out as an oversized step rather than
    just a tall absolute bar (which every later stage would also look like,
    since RSS only trends up across a run).
    """
    lines.append("### Stage-Level Memory Profiling (§11.6)\n\n")

    # Average RSS per tag across datasets that reported it (a SCANS fallback
    # skips before/after_render_median and after_composite, so not every tag
    # has full coverage — only average over datasets that actually hit it).
    per_tag_values: dict[str, list[float]] = {tag: [] for tag in STAGE_MEMORY_ORDER}
    for r in results:
        smem = r.get("stage_memory_rss_mb") or {}
        for tag, val in smem.items():
            if tag in per_tag_values and val is not None:
                per_tag_values[tag].append(val)

    tags_present = [t for t in STAGE_MEMORY_ORDER if per_tag_values[t]]
    if not tags_present:
        lines.append(
            "_No stage_memory_rss_mb data in this run (older results predate "
            "issue #69's §11.6 instrumentation)._\n\n"
        )
        return

    avg_rss = [sum(per_tag_values[t]) / len(per_tag_values[t]) for t in tags_present]

    chart_rel = None
    if _MPL_OK:
        try:
            chart_path = os.path.join(output_dir, "stage_memory_waterfall.png")
            fig, ax = plt.subplots(figsize=(max(6.0, len(tags_present) * 1.1), 5))
            deltas = [avg_rss[0]] + [
                avg_rss[i] - avg_rss[i - 1] for i in range(1, len(avg_rss))
            ]
            bottoms = [0.0] + avg_rss[:-1]
            colors = ["#2a78d6" if d >= 0 else "#eb6834" for d in deltas]
            ax.bar(
                range(len(tags_present)), deltas, bottom=bottoms,
                color=colors, edgecolor="white", linewidth=1,
            )
            for i, (b, d) in enumerate(zip(bottoms, deltas, strict=False)):
                ax.text(
                    i, b + d + (3 if d >= 0 else -3), f"{d:+.0f}",
                    ha="center", va="bottom" if d >= 0 else "top", fontsize=7,
                )
            ax.set_xticks(range(len(tags_present)))
            ax.set_xticklabels(tags_present, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("RSS (MB)")
            ax.set_title(
                f"Stage-Level Memory Waterfall — averaged across {len(results)} dataset(s)"
            )
            plt.tight_layout()
            fig.savefig(chart_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            chart_rel = os.path.basename(chart_path)
        except Exception:
            chart_rel = None

    if chart_rel:
        lines.append(f"![Stage-Level Memory Waterfall]({chart_rel})\n\n")

    lines.append("| Stage | Avg RSS (MB) | Δ vs previous stage |\n|---|---:|---:|\n")
    prev = None
    worst_stage, worst_delta = None, 0.0
    for tag, rss in zip(tags_present, avg_rss, strict=False):
        delta = rss if prev is None else rss - prev
        if prev is not None and delta > worst_delta:
            worst_delta, worst_stage = delta, tag
        lines.append(f"| `{tag}` | {rss:.1f} | {delta:+.1f} |\n")
        prev = rss
    lines.append("\n")
    if worst_stage:
        lines.append(
            f"> Largest single-stage growth: **`{worst_stage}`** "
            f"(+{worst_delta:.1f} MB avg) — the likeliest stage to investigate "
            "for a leak if overall RSS is trending up across runs.\n\n"
        )


def _find_latest_baseline(results_dir: str | None = None) -> list[dict] | None:
    """
    §11.9 (issue #69) — load the most recent prior `anime_stitch_*.json`
    run's `datasets` list from `backend/benchmark/output/` (the fixed,
    script-relative location `generate_json_results()` writes to — not the
    per-corpus `output_dir` passed to `generate_report()`).

    Called before the current run's own JSON is written (see the
    generate_report()/generate_json_results() call order in `main()`), so
    the most recent file found here is always a genuinely prior run, never
    the one currently being generated. Returns None on a first-ever run
    (nothing in output/ yet) or if the newest file fails to parse.
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), "output")
    candidates = sorted(glob.glob(os.path.join(results_dir, "anime_stitch_*.json")))
    if not candidates:
        return None
    try:
        with open(candidates[-1]) as fh:
            doc = json.load(fh)
        return doc.get("datasets")
    except Exception:
        return None


def detect_regressions(
    current: list[dict],
    baseline: list[dict],
    quality_drop_thr: float = 0.05,
    ghosting_increase_thr: float = 0.10,
    time_increase_thr: float = 0.20,
) -> list[dict]:
    """
    §11.9 (issue #69) — per-dataset regression detection between two runs.

    Python port of `detectRegressions()` in `frontend/src/math/benchmark.ts`
    (same 3-metric design, same default thresholds), but keyed by dataset
    name and reading the ASP-pipeline-specific metric names this benchmark
    actually emits (`metrics_asp.composite_quality`, `metrics_asp.ghosting_siqe`,
    `time.total_sec`) rather than the generic `GeneralBenchmark` schema the
    TS function operates on — the two aren't a drop-in match, so this is a
    reimplementation of the same threshold logic, not a call-through.

    A dataset only present in one of the two runs is skipped (nothing to
    diff against). Returns one entry per dataset that regressed on at least
    one dimension, each with a ``reasons`` list naming which metric(s) did.
    """
    baseline_by_name = {r["name"]: r for r in baseline if "name" in r}
    regressions: list[dict] = []
    for r in current:
        base = baseline_by_name.get(r.get("name"))
        if base is None:
            continue
        reasons: list[str] = []
        deltas: dict[str, float] = {}

        cur_q = (r.get("metrics_asp") or {}).get("composite_quality")
        base_q = (base.get("metrics_asp") or {}).get("composite_quality")
        if cur_q is not None and base_q:
            q_delta = (cur_q - base_q) / base_q
            deltas["composite_quality_pct"] = round(q_delta * 100, 1)
            if q_delta < -quality_drop_thr:
                reasons.append("composite_quality")

        cur_g = (r.get("metrics_asp") or {}).get("ghosting_siqe")
        base_g = (base.get("metrics_asp") or {}).get("ghosting_siqe")
        if cur_g is not None and base_g:
            g_delta = (cur_g - base_g) / base_g
            deltas["ghosting_siqe_pct"] = round(g_delta * 100, 1)
            if g_delta > ghosting_increase_thr:
                reasons.append("ghosting_siqe")

        cur_t = (r.get("time") or {}).get("total_sec")
        base_t = (base.get("time") or {}).get("total_sec")
        if cur_t is not None and base_t:
            t_delta = (cur_t - base_t) / base_t
            deltas["total_sec_pct"] = round(t_delta * 100, 1)
            if t_delta > time_increase_thr:
                reasons.append("total_sec")

        if reasons:
            regressions.append({"name": r["name"], "reasons": reasons, "deltas": deltas})
    return regressions


def _report_regression_dashboard(
    current: list[dict], baseline: list[dict] | None, lines: list[str]
) -> None:
    """
    §11.9 dashboard — Cross-Run Regression Dashboard.

    Compares this run against the most recent prior `anime_stitch_*.json`
    (see `_find_latest_baseline`), flagging composite_quality drops >5%,
    ghosting_siqe increases >10%, and total_sec increases >20% (the exact
    thresholds this item's spec names). One row per dataset with a
    red/green indicator, so a regression is visible without cross-checking
    the two JSON files by hand.
    """
    lines.append("### Cross-Run Regression Dashboard (§11.9)\n\n")

    if baseline is None:
        lines.append(
            "_No prior `anime_stitch_*.json` run found in `backend/benchmark/output/` "
            "— no baseline to compare against (this may be the first run)._\n\n"
        )
        return

    regressions_by_name = {r["name"]: r for r in detect_regressions(current, baseline)}
    baseline_names = {r["name"] for r in baseline if "name" in r}

    lines.append(
        "| Test | composite_quality Δ | ghosting_siqe Δ | total_sec Δ | Status |\n"
        "|------|---------------------:|-----------------:|------------:|:------:|\n"
    )
    any_row = False
    for r in current:
        name = r.get("name")
        if name not in baseline_names:
            continue
        any_row = True
        reg = regressions_by_name.get(name)
        deltas = reg["deltas"] if reg else {}
        reasons = set(reg["reasons"]) if reg else set()

        def _cell(key: str, metric: str, deltas=deltas, reasons=reasons) -> str:
            if key not in deltas:
                return "—"
            val = deltas[key]
            marker = " 🔴" if metric in reasons else ""
            return f"{val:+.1f}%{marker}"

        status = "🔴 regression" if reasons else "🟢 clean"
        lines.append(
            f"| [{name}](#{name}) | {_cell('composite_quality_pct', 'composite_quality')} | "
            f"{_cell('ghosting_siqe_pct', 'ghosting_siqe')} | "
            f"{_cell('total_sec_pct', 'total_sec')} | {status} |\n"
        )

    if not any_row:
        lines.append(
            "_No dataset names overlap between this run and the baseline "
            "— nothing to compare._\n\n"
        )
        return
    lines.append("\n")

    if regressions_by_name:
        lines.append(
            f"> 🔴 **{len(regressions_by_name)} dataset(s) regressed**: "
            + ", ".join(f"[{n}](#{n})" for n in regressions_by_name)
            + "\n\n"
        )
    else:
        lines.append("> 🟢 No regressions against the baseline run.\n\n")


def _report_experiment_comparison(results: list[dict], lines: list[str]) -> None:
    """
    §11.10 dashboard — Comparative Seam-Configuration Experiment Tracker.

    Groups datasets by `experiment_label` (set via the `ASP_EXPERIMENT_LABEL`
    env var before a run — see `_build_result`) and shows, per label, the
    mean composite_quality and mean total_sec across the datasets tagged
    with it — a side-by-side view of which configuration change actually
    moved which metric, per this item's spec. Untagged runs (the common
    case — most runs aren't A/B experiments) render a one-line note instead
    of an empty table.
    """
    lines.append("### Comparative Seam-Configuration Experiment Tracker (§11.10)\n\n")

    by_label: dict[str, list[dict]] = {}
    for r in results:
        label = r.get("experiment_label")
        if label:
            by_label.setdefault(label, []).append(r)

    if not by_label:
        lines.append(
            "_No experiment label set on any dataset in this run — set "
            "`ASP_EXPERIMENT_LABEL=<tag>` before running to tag a batch for "
            "comparison (e.g. `S44-seam-cache`, `S45-spanning-tree`)._\n\n"
        )
        return

    lines.append(
        "| Experiment | Datasets | Avg composite_quality (ASP) | Avg total_sec |\n"
        "|------------|---------:|-----------------------------:|---------------:|\n"
    )
    summary: list[tuple[str, int, float | None, float | None]] = []
    for label in sorted(by_label):
        rows = by_label[label]
        quals = [
            (r.get("metrics_asp") or {}).get("composite_quality")
            for r in rows
            if (r.get("metrics_asp") or {}).get("composite_quality") is not None
        ]
        times = [
            (r.get("time") or {}).get("total_sec")
            for r in rows
            if (r.get("time") or {}).get("total_sec") is not None
        ]
        avg_q = sum(quals) / len(quals) if quals else None
        avg_t = sum(times) / len(times) if times else None
        summary.append((label, len(rows), avg_q, avg_t))
        q_str = f"{avg_q:.2f}" if avg_q is not None else "—"
        t_str = f"{avg_t:.1f}" if avg_t is not None else "—"
        lines.append(f"| `{label}` | {len(rows)} | {q_str} | {t_str} |\n")
    lines.append("\n")

    valid = [row for row in summary if row[2] is not None and row[3] is not None]
    if len(valid) >= 2:
        best_q = max(valid, key=lambda row: row[2])
        best_t = min(valid, key=lambda row: row[3])
        lines.append(
            f"> Best `composite_quality`: **`{best_q[0]}`** ({best_q[2]:.2f}). "
            f"Fastest: **`{best_t[0]}`** ({best_t[3]:.1f}s).\n\n"
        )


def _report_single_test_outputs(
    r: dict,
    anime_rel: str | None,
    simple_rel: str | None,
    overmix_rel: str | None,
    hugin_rel: str | None,
    lines: list[str],
) -> None:
    lines.append("### Final Outputs\n\n")
    lines.append("| Anime Stitch Pipeline | OpenCV Simple Stitch | Overmix (reference) | Hugin (reference) |\n")
    lines.append("|:---------------------:|:--------------------:|:--------------------:|:--------------------:|\n")
    asp_cell = (
        f"![ASP]({anime_rel})"
        if os.path.exists(r["anime_path"])
        else "_not generated_"
    )
    simple_cell = (
        f"![Simple]({simple_rel})"
        if simple_rel and os.path.exists(r["simple_path"])
        else "_not generated_"
    )
    overmix_path = r.get("overmix_path")
    overmix_cell = (
        f"![Overmix]({overmix_rel})"
        if overmix_rel and overmix_path and os.path.exists(overmix_path)
        else "_not generated_"
    )
    hugin_path = r.get("hugin_path")
    hugin_cell = (
        f"![Hugin]({hugin_rel})"
        if hugin_rel and hugin_path and os.path.exists(hugin_path)
        else "_not generated_"
    )
    lines.append(f"| {asp_cell} | {simple_cell} | {overmix_cell} | {hugin_cell} |\n\n")


def _report_single_test_cv_metrics(
    r: dict,
    am: dict | None,
    sm: dict | None,
    om: dict | None,
    hg: dict | None,
    lines: list[str],
) -> None:
    lines.append("### CV Metrics\n\n")
    lines.append("| Metric | ASP | Simple | Overmix | Hugin | Notes |\n")
    lines.append("|--------|-----|--------|---------|-------|-------|\n")
    metric_defs = [
        ("sharpness", "Laplacian variance — higher = sharper edges"),
        ("coverage", "Fraction of non-black pixels — lower = heavy crop"),
        (
            "seam_gradient",
            "Mean gradient magnitude at seam rows — higher = abrupt transitions",
        ),
        ("color_entropy", "Shannon entropy of luma histogram — lower = washed out"),
        (
            "edge_energy_score",
            "2nd-order vertical gradient — a sharpness proxy, NOT ghosting",
        ),
        (
            "ghosting_siqe",
            "FFT autocorr double-edge score [0–100] — the true ghosting metric",
        ),
        ("seam_visibility", "Worst adjacent-row luminance jump — lower = smoother"),
        ("seam_coherence", "Row-mean luminance std — lower = less banding"),
        ("width", "Output width (px)"),
        ("height", "Output height (px)"),
    ]
    for key, note in metric_defs:
        a_val = f"{am.get(key, '—')}" if am else "—"
        s_val = f"{sm.get(key, '—')}" if sm else "—"
        om_val = f"{om.get(key, '—')}" if om else "—"
        hg_val = f"{hg.get(key, '—')}" if hg else "—"
        lines.append(f"| `{key}` | {a_val} | {s_val} | {om_val} | {hg_val} | {note} |\n")
    ssim_v = (
        f"{r['comparison']['ssim']:.3f}"
        if r["comparison"]["ssim"] is not None
        else "—"
    )
    psnr_v = (
        f"{r['comparison']['psnr_db']:.1f} dB"
        if r["comparison"]["psnr_db"] is not None
        else "—"
    )
    lines.append(
        f"| `ssim (asp vs simple)` | {ssim_v} | — | — | — | Structural similarity between the two outputs |\n"
    )
    lines.append(
        f"| `psnr (asp vs simple)` | {psnr_v} | — | — | — | Peak SNR between the two outputs |\n"
    )
    lines.append(
        f"| `seam_coherence` | {am.get('seam_coherence', '—') if am else '—'} | "
        f"{sm.get('seam_coherence', '—') if sm else '—'} | "
        f"{om.get('seam_coherence', '—') if om else '—'} | "
        f"{hg.get('seam_coherence', '—') if hg else '—'} | "
        f"Row-mean lum std — lower = less color banding (≤18 good, 18–28 moderate, >28 severe) |\n"
    )
    lines.append("\n")


def _report_single_test_gt(r: dict, lines: list[str]) -> None:
    gt = r.get("ground_truth", {})
    if gt.get("available"):
        lines.append("### Ground Truth Comparison\n\n")
        gt_am = gt.get("metrics_asp", {})
        gt_sm = gt.get("metrics_simple", {})
        asp_ssim_gt = gt_am.get("ssim_vs_gt")
        sim_ssim_gt = gt_sm.get("ssim_vs_gt")
        asp_psnr_gt = gt_am.get("psnr_vs_gt")
        sim_psnr_gt = gt_sm.get("psnr_vs_gt")
        gt_ver = gt.get("verdict", "—")
        lines.append("| Metric | ASP | Simple | Notes |\n")
        lines.append("|--------|-----|--------|-------|\n")
        lines.append(
            f"| SSIM vs Ground Truth | "
            f"{f'{asp_ssim_gt:.4f}' if asp_ssim_gt is not None else '—'} | "
            f"{f'{sim_ssim_gt:.4f}' if sim_ssim_gt is not None else '—'} | "
            f"Higher = closer to reference |\n"
        )
        lines.append(
            f"| PSNR vs Ground Truth | "
            f"{f'{asp_psnr_gt:.1f} dB' if asp_psnr_gt is not None else '—'} | "
            f"{f'{sim_psnr_gt:.1f} dB' if sim_psnr_gt is not None else '—'} | "
            f"Higher = closer to reference |\n"
        )
        lines.append(
            f"| **GT-based verdict** | **{gt_ver}** | — | Most reliable quality signal |\n"
        )
        lines.append("\n")


def _report_single_test_align(r: dict, lines: list[str]) -> None:
    ah = r["affine_health"]
    lines.append("### Alignment Health\n\n")
    lines.append("```yaml\n")
    lines.append(f"valid: {ah['valid']}\n")
    lines.append(f"reason: {ah['reason']}\n")
    lines.append(f"spacing_ratio: {ah['ratio']}\n")
    lines.append(f"min_gap_px: {ah['min_gap_px']}\n")
    lines.append(f"max_rotation: {ah['max_rotation']}\n")
    lines.append(f"max_scale_deviation: {ah['max_scale_dev']}\n")
    lines.append(f"used_scans_fallback: {r['used_fallback']}\n")
    if r["canvas"]["height"] is not None:
        lines.append(f"canvas: {r['canvas']['width']}×{r['canvas']['height']}\n")
    lines.append("```\n\n")

    phases = r.get("phases") or {}
    if phases.get("spans"):
        lines.append("### Animation Phases (§2.2, measurement-only)\n\n")
        lines.append(f"- Detected **{phases['count']}** phase(s) across "
                      f"{r['frames']['count']} selected frames.\n")
        for span in phases["spans"]:
            lines.append(
                f"  - Phase {span['phase']}: frames {span['start']}–{span['end']}\n"
            )
        lines.append("\n")

    _mpwd = r.get("mean_post_warp_diff")
    if _mpwd is not None:
        lines.append(
            f"- **§0.4 mean seam post_warp_diff**: {_mpwd:.2f} "
            "(lower = easier compositing job from frame selection; "
            "excludes single-pose-escalation sentinels)\n\n"
        )


def _report_single_test_photo(r: dict, lines: list[str]) -> None:
    gains = r["photometric"]["applied_gains"]
    non_trivial = [g for g in gains if abs(g - 1.0) > 0.01]
    lines.append("### Photometric Correction\n\n")
    lines.append(f"- Frames: **{len(gains)}**  \n")
    lines.append(
        f"- Frames corrected (|gain − 1| > 0.01): **{len(non_trivial)}**  \n"
    )
    if non_trivial:
        lines.append(
            f"- Gain range: [{min(non_trivial):.4f}, {max(non_trivial):.4f}]  \n"
        )
    lines.append("\n")


def _report_single_test_visualizations_plots(pd: str, rd: str, lines: list[str]) -> None:
    def _img_row(label, fname, alt=""):
        p = os.path.join(pd, fname)
        if os.path.exists(p):
            rel = _rel_path(p, rd)
            return f"**{label}**  \n![{alt or label}]({rel})\n\n"
        return ""

    # Metrics comparison bar
    bar_path = os.path.join(pd, "metrics_comparison.png")
    if os.path.exists(bar_path):
        lines.append(
            _img_row("CV Metrics Comparison (normalised)", "metrics_comparison.png")
        )

    # Gains
    gains_path = os.path.join(pd, "gains.png")
    if os.path.exists(gains_path):
        lines.append(_img_row("Per-Frame Luminance Gains", "gains.png"))

    # §2.2 animation-phase strip
    phases_path = os.path.join(pd, "animation_phases.png")
    if os.path.exists(phases_path):
        lines.append(
            _img_row(
                "Animation Phases (selected frames, colored by phase)",
                "animation_phases.png",
            )
        )

    # 2D canvas and overlap
    cp = os.path.join(pd, "canvas_frame_placement.png")
    if os.path.exists(cp):
        lines.append(
            _img_row("Canvas Frame Placement (2D)", "canvas_frame_placement.png")
        )

    tv = os.path.join(pd, "translation_vectors.png")
    if os.path.exists(tv):
        lines.append(
            _img_row("Translation Vectors (2D)", "translation_vectors.png")
        )

    om = os.path.join(pd, "overlap_map.png")
    if os.path.exists(om):
        lines.append(_img_row("Frame Overlap Count Map (2D)", "overlap_map.png"))

    # Seam heatmaps
    for img_type in ["asp", "simple"]:
        hm = os.path.join(pd, f"{img_type}_seam_heatmap.png")
        if os.path.exists(hm):
            label = "ASP" if img_type == "asp" else "Simple Stitch"
            lines.append(
                _img_row(
                    f"{label} — Seam Gradient Heatmap (2D)",
                    f"{img_type}_seam_heatmap.png",
                )
            )

    # 3D surface plots
    for fname, label in [
        ("asp_3d_surface.png", "ASP — Luminance Surface (3D)"),
        ("simple_3d_surface.png", "Simple Stitch — Luminance Surface (3D)"),
        (
            "temporal_render_3d.png",
            "Stage 9 Temporal Render — Luminance Surface (3D)",
        ),
    ]:
        p = os.path.join(pd, fname)
        if os.path.exists(p):
            lines.append(_img_row(label, fname))


def _report_single_test_visualizations_masks(pd: str, rd: str, lines: list[str]) -> None:
    mask_any = False
    for i in range(3):
        mp = os.path.join(pd, f"mask_overlay_frame{i:02d}.png")
        if os.path.exists(mp) and not mask_any:
            lines.append(
                "**BiRefNet Foreground Mask Overlays (first 3 frames)**\n\n"
            )
            lines.append(
                "| Frame 0 | Frame 1 | Frame 2 |\n|:---:|:---:|:---:|\n| "
            )
            mask_any = True
    if mask_any:
        cells = []
        for i in range(3):
            mp = os.path.join(pd, f"mask_overlay_frame{i:02d}.png")
            if os.path.exists(mp):
                cells.append(f"![mask f{i}]({_rel_path(mp, rd)})")
            else:
                cells.append("—")
        lines.append(" | ".join(cells) + " |\n\n")


def _report_single_test_visualizations_stages(sd: str, rd: str, r: dict, lines: list[str]) -> None:
    # Stage images
    lines.append("#### Stage Intermediate Outputs\n\n")
    _n_frames = min(r.get("frames", {}).get("count", 4), 4)
    stage_imgs = {
        "Stage 2 Normalised Frames": [
            os.path.join(sd, f"stage02_normalised_frame{i:02d}.png")
            for i in range(_n_frames)
        ],
        "Stage 3 Corrected Frames": [
            os.path.join(sd, f"stage03_basic_corrected_frame{i:02d}.png")
            for i in range(_n_frames)
        ],
        "Stage 4 BG Masks": [
            os.path.join(sd, f"stage04_bgmask_frame{i:02d}.png")
            for i in range(_n_frames)
        ],
    }
    for stage_label, paths in stage_imgs.items():
        existing = [p for p in paths if os.path.exists(p)]
        if not existing:
            continue
        lines.append(f"**{stage_label}**\n\n")
        cols = min(4, len(existing))
        header = "| " + " | ".join([f"Frame {i}" for i in range(cols)]) + " |\n"
        sep = "|" + "---|" * cols + "\n"
        row = (
            "| "
            + " | ".join(
                [
                    f"![f{i}]({_rel_path(p, rd)})"
                    for i, p in enumerate(existing[:cols])
                ]
            )
            + " |\n\n"
        )
        lines.append(header + sep + row)

    # Temporal render and composite
    for fname, label in [
        ("stage09_temporal_render.png", "Stage 9 — Temporal Median Render"),
        ("stage11_fg_composite.png", "Stage 11 — FG Composite"),
    ]:
        sp = os.path.join(sd, fname)
        if os.path.exists(sp):
            rel = _rel_path(sp, rd)
            lines.append(f"**{label}**  \n![{label}]({rel})\n\n")


def _report_single_test_visualizations(
    r: dict, pd: str, sd: str, rd: str, lines: list[str]
) -> None:
    lines.append("### Intermediate Output Visualizations\n\n")
    _report_single_test_visualizations_plots(pd, rd, lines)
    _report_single_test_visualizations_masks(pd, rd, lines)
    _report_single_test_visualizations_stages(sd, rd, r, lines)


def _report_single_test_analysis(
    r: dict, am: dict | None, sm: dict | None, lines: list[str]
) -> None:
    lines.append("### Automated Analysis\n\n")
    verdict = _auto_verdict(am, sm)
    verdict_map = {
        "asp_better": "ASP produces a **higher-quality** output by CV metrics.",
        "simple_better": "Simple/OpenCV produces a **higher-quality** output by CV metrics.",
        "comparable": "Both pipelines produce **comparable** quality by CV metrics.",
        "insufficient_data": "Insufficient data to determine a verdict.",
    }
    lines.append(f"> **CV Verdict:** {verdict_map.get(verdict, verdict)}\n\n")

    lines.append("**Detected issues — ASP:**\n")
    for issue in _auto_issues(am, is_asp=True):
        lines.append(f"{issue}\n")
    lines.append("\n**Detected issues — Simple Stitch:**\n")
    for issue in _auto_issues(sm, is_asp=False):
        lines.append(f"{issue}\n")
    lines.append("\n")

    if r["used_fallback"]:
        lines.append(
            "> ⚠️ **SCANS Fallback used** — Alignment failed, ASP result is identical to Simple Stitch.\n\n"
        )

    # Human feedback block
    asp_issues_yaml = "\n".join(_auto_issues(am, True))
    simple_issues_yaml = "\n".join(_auto_issues(sm, False))
    lines.append(
        _PER_TEST_HUMAN_SECTION.format(
            asp_issues=asp_issues_yaml,
            simple_issues=simple_issues_yaml,
            verdict=verdict,
        )
    )


def _report_per_test_details(results: list[dict], rd: str, lines: list[str]) -> None:
    for r in results:
        name = r["name"]
        anime_rel = _rel_path(r["anime_path"], rd)
        simple_rel = (
            _rel_path(r["simple_path"], rd)
            if os.path.exists(r["simple_path"])
            else None
        )
        pd = r["paths"]["plots_dir"]
        sd = r["paths"]["stage_dir"]
        am, sm = r["metrics_asp"], r["metrics_simple"]
        om = r.get("metrics_overmix") or {}
        overmix_path = r.get("overmix_path")
        overmix_rel = (
            _rel_path(overmix_path, rd)
            if overmix_path and os.path.exists(overmix_path)
            else None
        )
        hg = r.get("metrics_hugin") or {}
        hugin_path = r.get("hugin_path")
        hugin_rel = (
            _rel_path(hugin_path, rd)
            if hugin_path and os.path.exists(hugin_path)
            else None
        )

        lines.append(f"---\n\n## {name}\n\n")

        _report_single_test_outputs(r, anime_rel, simple_rel, overmix_rel, hugin_rel, lines)
        _report_single_test_cv_metrics(r, am, sm, om, hg, lines)
        _report_single_test_gt(r, lines)
        _report_single_test_align(r, lines)
        _report_single_test_photo(r, lines)
        _report_single_test_visualizations(r, pd, sd, rd, lines)
        _report_single_test_analysis(r, am, sm, lines)


def generate_report(results: list[dict], output_dir: str) -> str:
    """
    Write benchmark_report.md inside output_dir.
    Returns the path to the written file.
    """
    report_path = os.path.join(output_dir, "benchmark_report.md")
    rd = output_dir  # report dir = base for relative paths

    lines = []

    _report_header_and_summary(results, lines)
    _report_fail_breakdown(results, lines)
    _report_frame_selection_telemetry(results, output_dir, lines)
    _report_fallback_breakdown(results, lines)
    _report_stage_memory_waterfall(results, output_dir, lines)
    _report_regression_dashboard(results, _find_latest_baseline(), lines)
    _report_experiment_comparison(results, lines)
    _report_per_test_details(results, rd, lines)

    # Global feedback section
    lines.append(_GLOBAL_FEEDBACK_BLOCK)

    # Appendix: raw metrics JSON
    lines.append("---\n\n## Appendix — Raw Metrics JSON\n\n")
    lines.append("```json\n")
    summary = {
        "generated": datetime.datetime.now().isoformat(),
        "datasets": [
            {
                "name": r["name"],
                "asp_metrics": r["metrics_asp"],
                "sim_metrics": r["metrics_simple"],
                "overmix_metrics": r.get("metrics_overmix"),
                "hugin_metrics": r.get("metrics_hugin"),
                "ssim": r["comparison"]["ssim"],
                "psnr": r["comparison"]["psnr_db"],
                "affine_health": r["affine_health"],
                "used_fallback": r["used_fallback"],
            }
            for r in results
        ],
    }
    lines.append(json.dumps(summary, indent=2))
    lines.append("\n```\n")

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)

    print(f"\n[Report] Written to {report_path}")
    return report_path


# ============================================================================
# ENTRY POINT
# ============================================================================


def default_checkpoint_path() -> str:
    return os.path.join(os.path.dirname(__file__), "output", "_checkpoint.json")


def _checkpoint_done_names() -> set[str]:
    """Dataset names already persisted in the incremental bench checkpoint."""
    path = default_checkpoint_path()
    if not os.path.isfile(path):
        return set()
    try:
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(rows, list):
        return set()
    return {
        str(row["name"])
        for row in rows
        if isinstance(row, dict) and row.get("name")
    }


def _resolve_datasets(base_dir: str, args) -> list[str]:
    """
    Return an ordered list of dataset directories to process based on CLI args.

    Selection flags (mutually exclusive, first match wins):
      --tests asp_test04 asp_test27    specific dataset names
      --range 1-10                     inclusive numeric range (zero-padded)
      --range 1,3,5,27                 explicit comma-separated numbers
      --first N                        first N datasets in sorted order
      (none)                           all datasets

    Additional filter:
      --skip-done   skip any dataset whose output panorama.png already exists
      --resume-checkpoint   skip names already recorded in _checkpoint.json
    """
    all_dirs = sorted(
        d for d in glob.glob(os.path.join(base_dir, "asp_test*")) if os.path.isdir(d)
    )

    if args.tests:
        # Explicit names, e.g. asp_test04 asp_test27
        name_set = set(args.tests)
        selected = [d for d in all_dirs if os.path.basename(d) in name_set]
        # Preserve CLI order for exact names
        order = {n: i for i, n in enumerate(args.tests)}
        selected.sort(key=lambda d: order.get(os.path.basename(d), 999))
    elif args.range:
        # Numeric range "1-10" or comma list "1,3,27"
        spec = args.range
        nums: set = set()
        for part in spec.split(","):
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
                os.path.basename(d) == f"asp_test{n:02d}"
                or os.path.basename(d) == f"asp_test{n}"
                for n in nums
            )
        ]
    elif args.first:
        selected = all_dirs[: args.first]
    else:
        selected = all_dirs

    if args.skip_done:

        def _is_done(d: str) -> bool:
            return os.path.exists(os.path.join(d, "output", "panorama.png"))

        before = len(selected)
        selected = [d for d in selected if not _is_done(d)]
        print(
            f"[skip-done] Skipped {before - len(selected)} already-processed datasets."
        )

    if getattr(args, "resume_checkpoint", False):
        done = _checkpoint_done_names()
        if done:
            before = len(selected)
            selected = [d for d in selected if os.path.basename(d) not in done]
            print(
                f"[resume-checkpoint] Skipped {before - len(selected)} "
                f"datasets already in _checkpoint.json ({len(done)} named)."
            )

    return selected


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Anime Stitch Pipeline Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 94 tests (default)
  python3 backend/benchmark/bench_anime_stitch.py

  # Run specific tests by name
  python3 backend/benchmark/bench_anime_stitch.py --tests asp_test04 asp_test27

  # Run a numeric range (zero-padded names)
  python3 backend/benchmark/bench_anime_stitch.py --range 1-10

  # Mix: explicit comma list of numbers
  python3 backend/benchmark/bench_anime_stitch.py --range 1,4,8,27,57

  # First N tests only
  python3 backend/benchmark/bench_anime_stitch.py --first 5

  # Skip tests already processed (panorama.png exists)
  python3 backend/benchmark/bench_anime_stitch.py --skip-done

  # Resume a killed long run from incremental _checkpoint.json
  python3 backend/benchmark/bench_anime_stitch.py --range 2-97 --resume-checkpoint

  # Combine: first 20 tests, skip done
  python3 backend/benchmark/bench_anime_stitch.py --first 20 --skip-done
""",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        metavar="NAME",
        help="Specific dataset names to run (e.g. asp_test04 asp_test27)",
    )
    parser.add_argument(
        "--range",
        metavar="SPEC",
        help='Numeric range "1-10" or comma list "1,3,5" of test numbers',
    )
    parser.add_argument(
        "--first",
        type=int,
        metavar="N",
        help="Run only the first N datasets in sorted order",
    )
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Skip datasets whose output/panorama.png already exists",
    )
    parser.add_argument(
        "--resume-checkpoint",
        action="store_true",
        help="Skip datasets already listed in backend/benchmark/output/_checkpoint.json",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.expanduser("~/Downloads/Data/Dump"),
        metavar="DIR",
        help="Root data directory containing asp_testXX subdirectories",
    )
    args = parser.parse_args()

    base_dir = args.data_dir
    datasets = _resolve_datasets(base_dir, args)

    if not datasets:
        print("No datasets matched the selection criteria.")
        raise SystemExit(0)

    print(f"[Benchmark] Running {len(datasets)} dataset(s):")
    for d in datasets:
        print(f"  {os.path.basename(d)}")
    print()

    suite_start = time.perf_counter()
    _checkpoint_path = default_checkpoint_path()
    _baseline_snap = _resource_snapshot()
    print(
        f"[Resources] Baseline before any dataset: RSS={_baseline_snap['rss_gb']}GB  "
        f"sys_ram={_baseline_snap['sys_ram_used_pct']}%  "
        f"vram_used={_baseline_snap['vram_used_pct']}%  "
        f"(abort thresholds: RAM>={_RAM_ABORT_PCT:.0f}%, VRAM>={_VRAM_ABORT_PCT:.0f}%)"
    )
    results = []
    for ds in datasets:
        try:
            result = process_dataset(ds)
        except Exception as _ds_exc:
            # A single dataset's uncaught exception (e.g. SCANS fallback
            # failure inside process_dataset's own fallback path) must not
            # take the whole multi-hour batch down with it — every already-
            # accumulated result would be lost with nothing written to JSON.
            print(f"  [FATAL] {os.path.basename(ds)} crashed: {_ds_exc!r} — skipping.")
            result = None
        if result is not None:
            results.append(result)
            # Incremental checkpoint: a multi-hour batch can still be killed
            # by something outside this process (host sleep, OOM, external
            # signal) even with the try/except above catching in-process
            # crashes. Persist progress after every dataset so a killed run
            # loses at most the in-flight dataset, not the whole batch.
            try:
                os.makedirs(os.path.dirname(_checkpoint_path), exist_ok=True)
                with open(_checkpoint_path, "w") as _cp_fh:
                    json.dump(results, _cp_fh)
            except Exception:
                pass
        gc.collect()
        # Unconditional (unlike _log_resource's internal per-stage flush,
        # still gated behind ASP_RESOURCE_FLUSH_CUDA): this fires once per
        # dataset, not inside a hot per-pair/per-frame loop, so it doesn't
        # reintroduce #49's stream-serialization stalls. Without it, the
        # allocator's reserved-but-unallocated blocks accumulate across
        # datasets (vram_reserved stays ~15+GB even as vram_alloc drops to
        # ~0 between datasets) and spuriously trip the VRAM abort guardrail.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _snap = _resource_snapshot()
        print(
            f"  [Resources] RSS={_snap['rss_gb']}GB (Δ{_snap['rss_gb'] - _baseline_snap['rss_gb']:+.2f}GB)  "
            f"sys_ram={_snap['sys_ram_used_pct']}%  "
            f"vram_alloc={_snap['vram_allocated_gb']}GB  "
            f"vram_reserved={_snap['vram_reserved_gb']}GB  "
            f"vram_used={_snap['vram_used_pct']}%"
        )
        _danger = _resource_danger(_snap)
        if _danger:
            print(
                f"  [ABORT] Resource guardrail tripped: {_danger}. "
                "Stopping the batch here to protect the host — "
                f"{len(results)} dataset(s) already completed are still written out below."
            )
            break

    if results:
        output_dir = os.path.join(base_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        generate_report(results, output_dir)
        generate_json_results(results, suite_start)
        print(f"\nAll done. {len(results)} datasets processed.")
    else:
        print("No results to report.")
