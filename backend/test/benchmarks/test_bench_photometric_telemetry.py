"""Canonical benchmark photometric records come from the production stage."""

import importlib.util
from pathlib import Path

_BENCH = (
    Path(__file__).resolve().parents[3]
    / "backend"
    / "benchmark"
    / "bench_anime_stitch.py"
)


def _bench_module():
    spec = importlib.util.spec_from_file_location("bench_photo_telemetry", _BENCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_photometric_record_preserves_production_payload():
    payload = {
        "eligible_mask_count": 2,
        "reference_luminance": 101.25,
        "applied_gain_count": 1,
        "n_clamped": 1,
        "mean_residual": 0.073,
        "frames": [
            {
                "frame": 0,
                "background_pixels": 1200,
                "eligible": True,
                "background_luminance": 101.25,
                "gain_bgr": [1.0, 1.0, 1.0],
                "clamped": False,
            },
            {
                "frame": 1,
                "background_pixels": 900,
                "eligible": False,
                "background_luminance": None,
            },
            {
                "frame": 2,
                "background_pixels": 1600,
                "eligible": True,
                "background_luminance": 80.0,
                "gain_bgr": [1.14, 1.14, 1.14],
                "clamped": True,
            },
        ],
    }

    record = _bench_module()._production_photometric_result(payload)

    assert record["source"] == "production_stage"
    assert record["frames"] == payload["frames"]
    assert record["eligible_mask_count"] == 2
    assert record["bg_lums"] == [101.25, None, 80.0]
    assert record["applied_gains"] == [1.0, 1.14]
    assert record["gain_range"] == [1.0, 1.14]
    assert record["n_clamped"] == 1
