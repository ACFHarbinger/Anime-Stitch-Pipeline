"""M0b experiment manifest: git / profile / config / hashes / resources.

Two runs with the same manifest must produce equivalent stage traces.
``compare_traces`` flags note/fallback divergence as nondeterministic and
treats duration-only deltas as expected timing noise.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .session import json_safe

if TYPE_CHECKING:
    from .session import PipelineSession

SCHEMA_NAME = "asp.experiment_manifest"
SCHEMA_VERSION = "m0b.1"

_REPO_ROOT = Path(__file__).resolve().parents[4]
_HASH_CHUNK = 1 << 20
_GIT_CACHE: dict[str, Any] | None = None
_MODEL_CACHE: dict[str, Any] | None = None
_REPRO_ENV_KEYS = (
    "ASP_DETERMINISTIC",
    "ASP_REPRO_SEED",
    "ASP_BENCH_THREAD_CAP",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
)


def _repro_seed() -> int:
    try:
        return int(os.environ.get("ASP_REPRO_SEED", "1729"))
    except ValueError:
        return 1729


def configure_reproducibility() -> dict[str, Any]:
    """Pin process-local RNGs and runtime kernels when explicitly requested.

    Native BLAS/OpenMP environment variables are intentionally not modified
    here: those libraries consume them at process start. The caller must set
    them before importing NumPy, OpenCV, or Torch; the resulting values are
    recorded by :func:`reproducibility_snapshot` for auditability.
    """
    requested = os.environ.get("ASP_DETERMINISTIC", "0") != "0"
    seed = _repro_seed()
    applied = False
    if requested:
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass
        try:
            import cv2

            cv2.setRNGSeed(seed)
            cv2.setNumThreads(1)
        except Exception:
            pass
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        except Exception:
            pass
        applied = True
    return {"requested": requested, "seed": seed, "applied": applied}


def reproducibility_snapshot() -> dict[str, Any]:
    """Return the execution settings that can change routing decisions."""
    requested = os.environ.get("ASP_DETERMINISTIC", "0") != "0"
    seed = _repro_seed()
    payload: dict[str, Any] = {
        "requested": requested,
        "seed": seed,
        "environment": {key: os.environ.get(key) for key in _REPRO_ENV_KEYS},
        "python_random_seeded": requested,
        "numpy_random_seeded": requested,
        "opencv_rng_seed": seed if requested else None,
        "torch_rng_seed": seed if requested else None,
        "cuda_rng_seed": seed if requested else None,
    }
    try:
        import cv2

        payload["opencv_threads"] = cv2.getNumThreads()
    except Exception:
        payload["opencv_threads"] = None
    try:
        import torch

        payload.update(
            {
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "torch_device": "cuda" if torch.cuda.is_available() else "cpu",
                "cuda_device_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            }
        )
    except Exception:
        payload.update(
            {
                "torch_threads": None,
                "torch_interop_threads": None,
                "torch_device": None,
                "cuda_device_name": None,
                "cudnn_deterministic": None,
                "cudnn_benchmark": None,
                "deterministic_algorithms": None,
            }
        )
    return payload


def git_identity(repo_root: Path | None = None) -> dict[str, Any]:
    global _GIT_CACHE
    if _GIT_CACHE is not None and repo_root is None:
        return dict(_GIT_CACHE)
    root = str(repo_root or _REPO_ROOT)

    def _run(args: list[str]) -> str:
        try:
            proc = subprocess.run(
                args,
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if proc.returncode != 0:
            return ""
        return proc.stdout.strip()

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    porcelain = _run(["git", "status", "--porcelain"])
    identity = {
        "commit": commit or "unknown",
        "branch": branch or "unknown",
        "dirty": bool(porcelain),
    }
    if repo_root is None:
        _GIT_CACHE = identity
    return dict(identity)


def hash_file(path: str | os.PathLike[str]) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as fh:
            while True:
                chunk = fh.read(_HASH_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def hash_paths(paths: list[str]) -> dict[str, str | None]:
    return {path: hash_file(path) for path in paths}


def model_versions() -> dict[str, Any]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return dict(_MODEL_CACHE)
    versions: dict[str, Any] = {}
    for name in ("numpy", "cv2", "torch", "PIL"):
        try:
            mod = __import__(name)
        except Exception:
            versions[name] = None
            continue
        versions[name] = getattr(mod, "__version__", None)
    try:
        import torch

        versions["cuda"] = bool(torch.cuda.is_available())
        versions["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    except Exception:
        versions["cuda"] = False
        versions["torch_cuda"] = None
    try:
        from . import _probes

        versions["birefnet_ok"] = bool(_probes._BIREFNET_OK)
        versions["loftr_ok"] = bool(_probes._LOFTR_OK)
        versions["eloftr_ok"] = bool(_probes._ELOFTR_OK)
        versions["aliked_ok"] = bool(_probes._ALIKED_OK)
        versions["roma_ok"] = bool(_probes._ROMA_OK)
        versions["sea_raft_ok"] = bool(_probes._SEA_RAFT_OK)
        versions["batch_ok"] = bool(_probes._HAS_BATCH)
    except Exception:
        pass
    _MODEL_CACHE = versions
    return dict(versions)


def effective_asp_env() -> dict[str, str]:
    try:
        from asp_backend.core.config import asp_schema

        keys = set(asp_schema())
    except Exception:
        keys = {key for key in os.environ if key.startswith("ASP_")}
    return {
        key: os.environ[key]
        for key in sorted(keys)
        if key in os.environ
    }


def current_profile() -> str:
    return os.environ.get("ASP_PROFILE", "laptop_balanced") or "laptop_balanced"


@dataclass
class TraceDiff:
    """Result of comparing two session traces for the M0b exit criterion."""

    equivalent: bool
    digest_a: str
    digest_b: str
    extra_stages_a: list[str] = field(default_factory=list)
    extra_stages_b: list[str] = field(default_factory=list)
    divergent_stages: list[str] = field(default_factory=list)
    timing_only: list[str] = field(default_factory=list)
    order_mismatch: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "equivalent": self.equivalent,
            "digest_a": self.digest_a,
            "digest_b": self.digest_b,
            "extra_stages_a": list(self.extra_stages_a),
            "extra_stages_b": list(self.extra_stages_b),
            "divergent_stages": list(self.divergent_stages),
            "timing_only": list(self.timing_only),
            "order_mismatch": self.order_mismatch,
        }


def _stage_signature(record: Any) -> dict[str, Any]:
    return {
        "name": record.name,
        "skipped": bool(record.skipped),
        "fallback": record.fallback,
        "notes": json_safe(record.notes),
    }


def compare_traces(session_a: PipelineSession, session_b: PipelineSession) -> TraceDiff:
    """Identify equivalent traces vs. stages whose notes/fallbacks diverged.

    Duration differences alone are ``timing_only`` — not nondeterminism.
    """
    names_a = session_a.stage_names()
    names_b = session_b.stage_names()
    extra_a = [name for name in names_a if name not in names_b]
    extra_b = [name for name in names_b if name not in names_a]
    divergent: list[str] = []
    timing_only: list[str] = []
    by_a = {record.name: record for record in session_a.stages}
    by_b = {record.name: record for record in session_b.stages}
    for name in names_a:
        left = by_a.get(name)
        right = by_b.get(name)
        if left is None or right is None:
            continue
        if _stage_signature(left) != _stage_signature(right):
            divergent.append(name)
        elif left.duration_s != right.duration_s:
            timing_only.append(name)
    digest_a = session_a.digest()
    digest_b = session_b.digest()
    order_mismatch = names_a != names_b
    equivalent = (
        digest_a == digest_b
        and not extra_a
        and not extra_b
        and not divergent
        and not order_mismatch
        and session_a.identity == session_b.identity
    )
    return TraceDiff(
        equivalent=equivalent,
        digest_a=digest_a,
        digest_b=digest_b,
        extra_stages_a=extra_a,
        extra_stages_b=extra_b,
        divergent_stages=divergent,
        timing_only=timing_only,
        order_mismatch=order_mismatch,
    )


def build_experiment_manifest(session: PipelineSession) -> dict[str, Any]:
    resources = getattr(session, "resources", None)
    resource_dict = resources.as_dict() if resources is not None else {
        "peak_rss_bytes": 0,
        "peak_vram_bytes": None,
    }
    input_hashes = getattr(session, "input_hashes", None) or hash_paths(
        list(session.inputs.image_paths)
    )
    output_hashes = getattr(session, "output_hashes", None) or {}
    wall_s = None
    if session.finished_at is not None:
        wall_s = round(session.finished_at - session.started_at, 6)
    identity = session.identity
    if hasattr(identity, "value"):
        identity = identity.value
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "git": getattr(session, "git", None) or git_identity(),
        "profile": getattr(session, "profile", None) or current_profile(),
        "config": json_safe(session.config),
        "effective_env": getattr(session, "effective_env", None) or effective_asp_env(),
        "reproducibility": getattr(session, "reproducibility", None) or reproducibility_snapshot(),
        "model_versions": getattr(session, "model_versions", None) or model_versions(),
        "inputs": {
            "image_paths": list(session.inputs.image_paths),
            "hashes": input_hashes,
            "hires_keyframes": session.inputs.hires_keyframes,
        },
        "outputs": {
            "path": session.inputs.output_path,
            "hashes": output_hashes,
            "identity": identity,
        },
        "timings": {
            "wall_s": wall_s,
            "stages": [
                {"name": record.name, "duration_s": record.duration_s}
                for record in session.stages
            ],
        },
        "resources": resource_dict,
        "trace": {
            "stages": session.stage_names(),
            "digest": session.digest(),
            "fallbacks": list(session.fallbacks),
            "success": session.success,
            "error": session.error,
        },
    }


def write_manifest(payload: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "TraceDiff",
    "build_experiment_manifest",
    "compare_traces",
    "configure_reproducibility",
    "current_profile",
    "effective_asp_env",
    "git_identity",
    "hash_file",
    "hash_paths",
    "model_versions",
    "reproducibility_snapshot",
    "write_manifest",
]
