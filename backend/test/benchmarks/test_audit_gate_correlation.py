"""Focused tests for the M2 gate-signal audit (strip_banding recompute)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

_AUDIT_PATH = (
    Path(__file__).resolve().parents[3]
    / "backend"
    / "benchmark"
    / "audit_gate_correlation.py"
)


def _audit_mod():
    spec = importlib.util.spec_from_file_location(
        "audit_gate_correlation", _AUDIT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_AUDIT = _audit_mod()
affines_from_alignment = _AUDIT.affines_from_alignment
audit = _AUDIT.audit
resolve_panorama = _AUDIT.resolve_panorama


def _write_png(path: Path, top: int, bot: int, h: int = 80, w: int = 60) -> None:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2] = top
    img[h // 2 :] = bot
    cv2.imwrite(str(path), img)


def test_affines_from_alignment_uses_ty():
    ds = {
        "alignment": {
            "affines": [
                {"frame": 0, "tx": 1.0, "ty": 10.0, "a": 1.0, "b": 0.0},
                {"frame": 1, "tx": 2.0, "ty": 40.0, "a": 1.0, "b": 0.0},
            ]
        }
    }
    aff = affines_from_alignment(ds)
    assert len(aff) == 2
    assert float(aff[0][1, 2]) == 10.0
    assert float(aff[1][1, 2]) == 40.0


def test_resolve_panorama_prefers_raw(tmp_path: Path):
    ds = {
        "name": "asp_test01",
        "anime_path": "dump/output/asp_test01_anime_stitch.png",
        "simple_path": "dump/output/asp_test01_opencv_stitch.png",
        "paths": {},
    }
    raw = tmp_path / "asp_test01_raw_asp.png"
    pub = tmp_path / "asp_test01_anime_stitch.png"
    _write_png(raw, 10, 200)
    _write_png(pub, 80, 90)
    assert resolve_panorama(ds, images_root=tmp_path, kind="asp", prefer_raw=True) == raw
    assert resolve_panorama(ds, images_root=tmp_path, kind="asp", prefer_raw=False) == pub


def test_audit_recomputes_strip_banding(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    run = {"datasets": []}
    labels = {}
    for i in range(6):
        name = f"asp_test{i:02d}"
        run["datasets"].append(
            {
                "name": name,
                "metrics_asp": {
                    "sharpness": 10.0 + i,
                    "edge_energy_score": 1.0,
                    "ghosting_siqe": 10.0 + i * 5,
                    "seam_coherence": 5.0,
                    "seam_visibility": 4.0 + i,
                    "cqas": 0.5,
                    "coverage": 0.9,
                    "color_entropy": 1.0,
                    "seam_gradient": 2.0,
                },
                "metrics_simple": {
                    "sharpness": 8.0,
                    "edge_energy_score": 0.8,
                    "ghosting_siqe": 8.0,
                    "seam_coherence": 4.0,
                    "seam_visibility": 3.0,
                    "cqas": 0.6,
                    "coverage": 0.9,
                    "color_entropy": 1.0,
                    "seam_gradient": 1.0,
                },
                "alignment": {
                    "affines": [
                        {"frame": 0, "tx": 0.0, "ty": 0.0, "a": 1.0, "b": 0.0},
                        {"frame": 1, "tx": 0.0, "ty": 40.0, "a": 1.0, "b": 0.0},
                    ]
                },
                "anime_path": f"{name}_anime_stitch.png",
                "simple_path": f"{name}_opencv_stitch.png",
                "paths": {},
            }
        )
        # Increasing inter-strip jump; humans prefer ASP less as banding grows.
        _write_png(images / f"{name}_raw_asp.png", 80 - i * 10, 80 + i * 20)
        labels[name] = {"reviewed": True, "asp": 4 - i // 2, "simple": 2}

    run_path = tmp_path / "run.json"
    labels_path = tmp_path / "labels.json"
    run_path.write_text(json.dumps(run))
    labels_path.write_text(json.dumps(labels))

    results = audit(
        run_path,
        labels_path,
        recompute_missing=True,
        images_root=images,
        prefer_raw=True,
    )
    by_name = {k: (rho, n) for k, rho, _p, n in results}
    assert "strip_banding_score" in by_name
    rho, n = by_name["strip_banding_score"]
    assert n == 6
    assert rho > 0.2
