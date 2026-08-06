"""
backend/test/animation/batch/test_batch_imports.py
====================================================
Pybind11 binding smoke tests.

Validates that the compiled ``batch`` extension exposes every expected
Python symbol through its pybind11 interface.  These tests are the only
batch tests that must be in Python, because they exercise the Python-facing
API layer (pybind11 argument parsing, docstrings, default args) rather than
the underlying C++ algorithm.

All tests are skipped automatically when ``batch`` has not been built.
"""

import pytest

try:
    import base as batch
    if (
        getattr(batch, "__file__", None) is None
        or not hasattr(batch, "bundle_adjust")
    ):
        raise ImportError("compiled base ASP extension not available")
    HAS_BATCH = True
except ImportError:
    HAS_BATCH = False

pytestmark = pytest.mark.skipif(not HAS_BATCH, reason="batch not built")


# ---------------------------------------------------------------------------
# Module-level metadata
# ---------------------------------------------------------------------------


class TestModuleMetadata:
    def test_module_has_non_empty_docstring(self):
        assert batch.__doc__ is not None
        assert len(batch.__doc__.strip()) > 0

    def test_all_submodules_present(self):
        expected = [
            "matching",
            "bundle_adjust",
            "validation",
            "canvas",
            "seam",
            "compositing",
            "exposure",
            "frame_selection",
            "fg_register",
        ]
        for name in expected:
            assert hasattr(batch, name), f"batch.{name} submodule missing"

    def test_submodules_have_docstrings(self):
        for attr_name in dir(batch):
            attr = getattr(batch, attr_name)
            if not attr_name.startswith("_") and hasattr(attr, "__doc__"):
                assert attr.__doc__ is not None, f"batch.{attr_name} has no docstring"


# ---------------------------------------------------------------------------
# Per-submodule API surface checks
# ---------------------------------------------------------------------------

SUBMODULE_APIS = {
    "matching": [
        "phase_correlate_masked",
        "build_edge_graph",
        "reject_static_edges",
        "compute_adaptive_min_disp",
        "spatial_dedup_frames",
        # "filter_edge_graph",  # add after next C++ build (Phase 3b)
    ],
    "bundle_adjust": [
        "bundle_adjust_affine",
        "spanning_tree_inlier_filter",
        "compute_adaptive_f_scale",
    ],
    "validation": [
        "validate_affines",
        "compute_adaptive_min_gap",
        "compute_adaptive_rot_scale",
    ],
    "canvas": [
        "compute_canvas",
        "warp_frames_to_canvas",
        "render_median",
        "crop_to_valid",
        "telea_fill_gaps",
        "detect_scroll_axis",
        "panorama_stitch_fallback",
        # "gpu_device_count",  # add after next C++ build (Phase 6)
    ],
    "seam": [
        "seam_cut",
        "build_seam_cost_map",
        "graphcut_seam_find",
    ],
    "compositing": [
        "laplacian_blend",
        "single_pose_soft_edge",
        "seam_color_match",
        "normalize_warped_frames",
        "find_optimal_boundaries",
        "blocks_gain_compensate_pair",
        "blocks_lum_compensate_pair",
    ],
    "exposure": [
        "blocks_gain_compensate",
        "correct_vignetting",
    ],
    "frame_selection": [
        "detect_hold_blocks_mad",
        "detect_hold_blocks_dhash",
        "temporal_variance_filter",
        "near_dup_luma_filter",
        "spatial_dedup_frames",
    ],
    "fg_register": [
        "arap_push_regularise",
        "ecc_refine",
    ],
}


@pytest.mark.parametrize("submodule_name,symbols", SUBMODULE_APIS.items())
class TestSubmoduleAPISurface:
    def test_all_symbols_exist_and_callable(self, submodule_name, symbols):
        submodule = getattr(batch, submodule_name)
        for fn_name in symbols:
            assert hasattr(submodule, fn_name), (
                f"batch.{submodule_name}.{fn_name} is missing from the pybind11 binding"
            )
            assert callable(getattr(submodule, fn_name)), (
                f"batch.{submodule_name}.{fn_name} is not callable"
            )

    def test_all_symbols_have_docstrings(self, submodule_name, symbols):
        submodule = getattr(batch, submodule_name)
        for fn_name in symbols:
            fn = getattr(submodule, fn_name)
            assert fn.__doc__ is not None and len(fn.__doc__.strip()) > 0, (
                f"batch.{submodule_name}.{fn_name} has no docstring"
            )


# ---------------------------------------------------------------------------
# pybind11 default argument parsing — validate Python calling conventions
# ---------------------------------------------------------------------------


class TestDefaultArguments:
    """
    Verify that pybind11 default argument declarations are correct by
    checking that calling each function with only required args does not
    raise TypeError (wrong number of args).

    Functions raise RuntimeError("not implemented") in Phase 1 — that is
    expected and caught.  TypeError would indicate a binding mistake.
    """

    def _call_tolerating_not_impl(self, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except RuntimeError:
            pass  # expected stub response
        except TypeError as e:
            pytest.fail(f"TypeError (bad binding): {e}")

    def test_seam_cut_minimal_args(self):
        import numpy as np

        fa = np.zeros((20, 30, 3), dtype=np.uint8)
        fb = np.zeros((20, 30, 3), dtype=np.uint8)
        self._call_tolerating_not_impl(batch.seam.seam_cut, fa, fb)

    def test_bundle_adjust_affine_minimal_args(self):
        edges = [{"i": 0, "j": 1, "dx": 0.0, "dy": 100.0, "weight": 0.9, "type": "adjacent"}]
        self._call_tolerating_not_impl(batch.bundle_adjust.bundle_adjust_affine, edges, 2)

    def test_blocks_gain_compensate_minimal_args(self):
        import numpy as np

        frames  = [np.zeros((60, 80, 3), dtype=np.uint8)]
        masks   = [np.zeros((60, 80), dtype=np.uint8)]
        corners = [(0, 0)]
        self._call_tolerating_not_impl(
            batch.exposure.blocks_gain_compensate, frames, masks, corners
        )

    def test_correct_vignetting_minimal_args(self):
        import numpy as np

        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        vmap  = np.ones((40, 60), dtype=np.float32)
        self._call_tolerating_not_impl(batch.exposure.correct_vignetting, frame, vmap)
