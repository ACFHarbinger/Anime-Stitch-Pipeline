// ---------------------------------------------------------------------------
// batch/src/validation.cpp
//
// Affine sequence validation: monotone step, ratio, gap checks,
// rotation/scale deviation, and Kendall-τ monotonicity.
//
// Replaces:
//   core/validation.py  :: _validate_affines, _compute_adaptive_min_gap,
//                          _compute_adaptive_rot_scale
//
// Implementation roadmap: Phase 3.
// See docs/moon/roadmaps/asp_cpp_migration.md §base::validation
// ---------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "common.hpp"
#include "affine_types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Constants (mirror backend/src/animation/core/validation.py)
// ---------------------------------------------------------------------------
static constexpr float ROT_TIGHT                 = 0.10f;
static constexpr float ROT_LOOSE                 = 0.15f;
static constexpr float SC_TIGHT                  = 0.10f;
static constexpr float SC_LOOSE                  = 0.15f;
static constexpr float ROT_SCALE_CONSISTENCY_THRESH = 0.02f;
static constexpr float MONO_TAU_MIN              = 0.40f;  // §1.12 Kendall τ floor

// ---------------------------------------------------------------------------
// ValidationResult — internal result type
// ---------------------------------------------------------------------------
struct ValidationResult {
    bool        valid         = true;
    float       ratio         = 1.0f;
    float       min_gap       = 0.0f;
    float       max_rotation  = 0.0f;
    float       max_scale_dev = 0.0f;
    std::string reason;
};

// ---------------------------------------------------------------------------
// detect_scroll_axis_impl (pure C++)
//
// Mirrors alignment/canvas.py::_detect_scroll_axis.
// Returns "vertical" | "horizontal" | "diagonal" | "none".
// ---------------------------------------------------------------------------
static std::string detect_scroll_axis_impl(
    const std::vector<float>& txs,
    const std::vector<float>& tys)
{
    float ty_min = *std::min_element(tys.begin(), tys.end());
    float ty_max = *std::max_element(tys.begin(), tys.end());
    float tx_min = *std::min_element(txs.begin(), txs.end());
    float tx_max = *std::max_element(txs.begin(), txs.end());
    float ty_range = ty_max - ty_min;
    float tx_range = tx_max - tx_min;
    float total    = ty_range + tx_range;

    if (total < 1.0f)    return "none";
    if (tx_range > 0.0f &&
        ty_range / std::max(tx_range, 1.0f) < 0.1f) return "horizontal";
    if (ty_range > 0.0f &&
        tx_range / std::max(ty_range, 1.0f) > 0.3f) return "diagonal";
    return "vertical";
}

// ---------------------------------------------------------------------------
// kendall_tau_monotone_impl (pure C++)
//
// §1.12 — Verify that the ty/tx sequence is spatially monotone.
// Returns (is_monotone, tau_abs).  Requires N ≥ 4; shorter sequences return
// (true, 1.0).
// ---------------------------------------------------------------------------
static std::pair<bool, float> kendall_tau_monotone_impl(
    const std::vector<float>& vals,
    float                     min_tau_abs)
{
    int N = static_cast<int>(vals.size());
    if (N < 4) return {true, 1.0f};

    int concordant = 0, discordant = 0;
    for (int i = 0; i < N - 1; ++i) {
        for (int j = i + 1; j < N; ++j) {
            float d = vals[j] - vals[i];
            if (d > 0.0f)       ++concordant;
            else if (d < 0.0f)  ++discordant;
        }
    }
    int n_pairs = concordant + discordant;
    if (n_pairs == 0) return {true, 1.0f};

    float tau     = static_cast<float>(concordant - discordant) / n_pairs;
    float tau_abs = std::abs(tau);
    return {tau_abs >= min_tau_abs, tau_abs};
}

// ---------------------------------------------------------------------------
// validate_affines_impl (pure C++)
//
// Parameters are parallel vectors extracted from the 2×3 affine matrices.
// Returns a ValidationResult whose fields mirror Python's AffineHealth.
// ---------------------------------------------------------------------------
ValidationResult validate_affines_impl(
    const std::vector<float>& txs,
    const std::vector<float>& tys,
    const std::vector<float>& rots,    ///< max(|M[0,1]|, |M[1,0]|) per frame
    const std::vector<float>& scales,  ///< max(|M[0,0]-1|, |M[1,1]-1|) per frame
    float min_step    = 25.0f,
    float max_ratio   = 3.0f,
    float max_rotation   = 0.10f,
    float max_scale_dev  = 0.10f)
{
    ValidationResult ok;
    int N = static_cast<int>(txs.size());

    if (N < 2) {
        ok.reason = "single frame";
        return ok;
    }

    std::string scroll_axis = detect_scroll_axis_impl(txs, tys);

    // Sort frame positions along the dominant axis, compute Euclidean gaps
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);

    if (scroll_axis == "horizontal") {
        std::sort(order.begin(), order.end(),
                  [&txs](int a, int b) { return txs[a] < txs[b]; });
    } else if (scroll_axis == "diagonal") {
        // Sort by cumulative distance from frame 0
        std::vector<float> dists(N);
        for (int i = 0; i < N; ++i)
            dists[i] = std::sqrt((txs[i] - txs[0]) * (txs[i] - txs[0]) +
                                 (tys[i] - tys[0]) * (tys[i] - tys[0]));
        std::sort(order.begin(), order.end(),
                  [&dists](int a, int b) { return dists[a] < dists[b]; });
    } else {
        // Vertical (default)
        std::sort(order.begin(), order.end(),
                  [&tys](int a, int b) { return tys[a] < tys[b]; });
    }

    // Euclidean gaps between consecutive sorted frames
    std::vector<float> gaps;
    gaps.reserve(N - 1);
    for (int k = 0; k < N - 1; ++k) {
        int i = order[k], j = order[k + 1];
        float dx = txs[j] - txs[i];
        float dy = tys[j] - tys[i];
        gaps.push_back(std::sqrt(dx * dx + dy * dy));
    }

    if (gaps.empty()) {
        ok.valid  = false;
        ok.reason = "all frames at same position";
        return ok;
    }

    std::vector<float> sorted_gaps = gaps;
    std::sort(sorted_gaps.begin(), sorted_gaps.end());
    float median_gap = (sorted_gaps.size() % 2 == 0)
        ? (sorted_gaps[sorted_gaps.size() / 2 - 1] +
           sorted_gaps[sorted_gaps.size() / 2]) * 0.5f
        : sorted_gaps[sorted_gaps.size() / 2];
    float max_gap = *std::max_element(gaps.begin(), gaps.end());
    float min_gap = *std::min_element(gaps.begin(), gaps.end());
    float ratio   = max_gap / std::max(median_gap, 1.0f);

    float max_rot = *std::max_element(rots.begin(), rots.end());
    float max_sc  = *std::max_element(scales.begin(), scales.end());

    ok.ratio         = ratio;
    ok.min_gap       = min_gap;
    ok.max_rotation  = max_rot;
    ok.max_scale_dev = max_sc;

    if (ratio > max_ratio) {
        ok.valid  = false;
        std::ostringstream oss;
        oss << "ratio=" << ratio << " > " << max_ratio;
        ok.reason = oss.str();
        return ok;
    }
    if (min_gap < min_step) {
        ok.valid  = false;
        std::ostringstream oss;
        oss << "min_gap=" << min_gap << "px < " << min_step << "px";
        ok.reason = oss.str();
        return ok;
    }
    if (max_rot > max_rotation) {
        ok.valid  = false;
        std::ostringstream oss;
        oss << "rotation=" << max_rot << " > " << max_rotation;
        ok.reason = oss.str();
        return ok;
    }
    if (max_sc > max_scale_dev) {
        ok.valid  = false;
        std::ostringstream oss;
        oss << "scale_dev=" << max_sc << " > " << max_scale_dev;
        ok.reason = oss.str();
        return ok;
    }

    // §1.12 — Kendall τ monotonicity (vertical / horizontal only)
    if (scroll_axis == "vertical" || scroll_axis == "horizontal") {
        const std::vector<float>& prim_vals =
            (scroll_axis == "horizontal") ? txs : tys;
        auto [is_mono, tau_abs] = kendall_tau_monotone_impl(prim_vals, MONO_TAU_MIN);
        if (!is_mono) {
            ok.valid  = false;
            std::ostringstream oss;
            oss << "monotonicity=" << tau_abs << " < " << MONO_TAU_MIN;
            ok.reason = oss.str();
            return ok;
        }
    }

    ok.reason = "ok";
    return ok;
}

// ---------------------------------------------------------------------------
// compute_adaptive_min_gap_impl (pure C++)
//
// §0.5C — Returns max(20.0, canvas_span / (N × 3)).
// ---------------------------------------------------------------------------
float compute_adaptive_min_gap_impl(
    const std::vector<float>& txs,
    const std::vector<float>& tys)
{
    int N = static_cast<int>(txs.size());
    if (N < 2) return 20.0f;

    float ty_min = *std::min_element(tys.begin(), tys.end());
    float ty_max = *std::max_element(tys.begin(), tys.end());
    float tx_min = *std::min_element(txs.begin(), txs.end());
    float tx_max = *std::max_element(txs.begin(), txs.end());
    float dy_span = ty_max - ty_min;
    float dx_span = tx_max - tx_min;

    std::string scroll_axis = detect_scroll_axis_impl(txs, tys);

    float canvas_span;
    if (scroll_axis == "horizontal") {
        canvas_span = dx_span;
    } else if (scroll_axis == "diagonal") {
        canvas_span = std::sqrt(dy_span * dy_span + dx_span * dx_span);
    } else {
        canvas_span = dy_span;
    }

    if (canvas_span < 1.0f) return 20.0f;
    return std::max(20.0f, canvas_span / (static_cast<float>(N) * 3.0f));
}

// ---------------------------------------------------------------------------
// compute_adaptive_rot_scale_impl (pure C++)
//
// §0.5D — Returns (max_rotation, max_scale_dev) adaptive thresholds.
// ---------------------------------------------------------------------------
std::pair<float, float> compute_adaptive_rot_scale_impl(
    const std::vector<float>& rots,
    const std::vector<float>& scales)
{
    if (rots.size() < 2)
        return {ROT_TIGHT, SC_TIGHT};

    // Compute std dev of rots
    float rot_mean = 0.0f, sc_mean = 0.0f;
    for (float v : rots)   rot_mean += v;
    for (float v : scales) sc_mean  += v;
    rot_mean /= rots.size();
    sc_mean  /= scales.size();

    float rot_var = 0.0f, sc_var = 0.0f;
    for (float v : rots)   rot_var += (v - rot_mean) * (v - rot_mean);
    for (float v : scales) sc_var  += (v - sc_mean)  * (v - sc_mean);
    float rot_std = std::sqrt(rot_var / rots.size());
    float sc_std  = std::sqrt(sc_var  / scales.size());

    float max_rot = (rot_std < ROT_SCALE_CONSISTENCY_THRESH) ? ROT_LOOSE : ROT_TIGHT;
    float max_sc  = (sc_std  < ROT_SCALE_CONSISTENCY_THRESH) ? SC_LOOSE  : SC_TIGHT;

    return {max_rot, max_sc};
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------
#ifndef BATCH_TESTS

// Helper: extract affine data from a Python numpy 2×3 array
static void extract_affine_data_from_py(
    py::handle item,
    float& tx, float& ty, float& rot, float& sc)
{
    auto arr = item.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    BATCH_CHECK(arr.ndim() == 2 && arr.shape(0) >= 2 && arr.shape(1) >= 3,
                "validate_affines: each affine must be a (2,3) float numpy array");
    auto a = arr.unchecked<2>();
    tx  = a(0, 2);
    ty  = a(1, 2);
    rot = std::max(std::abs(a(0, 1)), std::abs(a(1, 0)));
    sc  = std::max(std::abs(a(0, 0) - 1.0f), std::abs(a(1, 1) - 1.0f));
}

// ---------------------------------------------------------------------------
// validate_affines — Python wrapper
// ---------------------------------------------------------------------------
static py::tuple validate_affines(
    py::list affines_py,
    float    min_step    = 25.0f,
    float    max_ratio   = 3.0f,
    float    max_rotation   = 0.10f,
    float    max_scale_dev  = 0.10f)
{
    int N = static_cast<int>(py::len(affines_py));
    std::vector<float> txs, tys, rots, scales;
    txs.reserve(N); tys.reserve(N); rots.reserve(N); scales.reserve(N);

    for (auto item : affines_py) {
        float tx, ty, rot, sc;
        extract_affine_data_from_py(item, tx, ty, rot, sc);
        txs.push_back(tx); tys.push_back(ty);
        rots.push_back(rot); scales.push_back(sc);
    }

    auto res = validate_affines_impl(txs, tys, rots, scales,
                                     min_step, max_ratio,
                                     max_rotation, max_scale_dev);
    return py::make_tuple(res.valid, res.reason,
                          res.ratio, res.min_gap,
                          res.max_rotation, res.max_scale_dev);
}

// ---------------------------------------------------------------------------
// compute_adaptive_min_gap — Python wrapper
// ---------------------------------------------------------------------------
static float compute_adaptive_min_gap(py::list affines_py)
{
    std::vector<float> txs, tys;
    for (auto item : affines_py) {
        float tx, ty, rot, sc;
        extract_affine_data_from_py(item, tx, ty, rot, sc);
        txs.push_back(tx); tys.push_back(ty);
    }
    return compute_adaptive_min_gap_impl(txs, tys);
}

// ---------------------------------------------------------------------------
// compute_adaptive_rot_scale — Python wrapper
// ---------------------------------------------------------------------------
static py::tuple compute_adaptive_rot_scale(py::list affines_py)
{
    std::vector<float> rots, scales;
    for (auto item : affines_py) {
        float tx, ty, rot, sc;
        extract_affine_data_from_py(item, tx, ty, rot, sc);
        rots.push_back(rot); scales.push_back(sc);
    }
    auto [max_rot, max_sc] = compute_adaptive_rot_scale_impl(rots, scales);
    return py::make_tuple(max_rot, max_sc);
}

// ---------------------------------------------------------------------------
// register_validation — called from bindings.cpp
// ---------------------------------------------------------------------------
void register_validation(py::module_& m) {
    m.doc() = R"doc(
        batch.validation — Affine sequence health checks.

        Functions
        ---------
        validate_affines(affines, min_step, max_ratio, max_rotation, max_scale_dev)
            -> (valid, reason, ratio, min_gap, max_rot, max_sc)
        compute_adaptive_min_gap(affines) -> float
        compute_adaptive_rot_scale(affines) -> (max_rotation, max_scale_dev)

        All functions accept a list of (2, 3) float32 numpy affine matrices.
    )doc";

    m.def("validate_affines", &validate_affines,
        py::arg("affines"),
        py::arg("min_step")       = 25.0f,
        py::arg("max_ratio")      = 3.0f,
        py::arg("max_rotation")   = 0.10f,
        py::arg("max_scale_dev")  = 0.10f,
        R"doc(
            Full affine sequence health check.

            Checks (in order):
              1. max_gap / median_gap < max_ratio
              2. min_gap >= min_step
              3. max off-diagonal element <= max_rotation
              4. max diagonal deviation from 1.0 <= max_scale_dev
              5. §1.12 Kendall-τ monotonicity (vertical/horizontal only)

            Returns (valid, reason, ratio, min_gap, max_rotation, max_scale_dev).
            reason = "ok" when valid is True.
        )doc");

    m.def("compute_adaptive_min_gap", &compute_adaptive_min_gap,
        py::arg("affines"),
        R"doc(
            §0.5C — Adaptive min-gap threshold: max(20.0, canvas_span / (N × 3)).

            Returns float (pixels).
        )doc");

    m.def("compute_adaptive_rot_scale", &compute_adaptive_rot_scale,
        py::arg("affines"),
        R"doc(
            §0.5D — Adaptive rotation/scale thresholds.

            Returns (max_rotation, max_scale_dev) using loose thresholds
            when the sequence shows consistent camera properties.
        )doc");
}

#endif // BATCH_TESTS
