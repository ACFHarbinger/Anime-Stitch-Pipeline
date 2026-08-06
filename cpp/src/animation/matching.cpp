// ---------------------------------------------------------------------------
// batch/src/matching.cpp
//
// Phase correlation, static edge rejection, adaptive min-disp threshold.
//
// Replaces:
//   flow/cam_flow.py       :: bg_masked_phase_correlate
//   alignment/matching.py  :: _reject_static_edges, _compute_adaptive_min_disp
//   core/pipeline.py       :: _reject_static_edges, _compute_adaptive_min_disp,
//                             _spatial_dedup_frames
//
// Implementation roadmap: Phase 3 (alignment hot path).
// See docs/moon/roadmaps/asp_cpp_migration.md §base::matching
// ---------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "common.hpp"
#include "affine_types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Constants (mirror backend/src/constants/animation.py)
// ---------------------------------------------------------------------------
static constexpr float STATIC_EDGE_MIN_DISP_PX  = 50.0f;  // §1.2A
static constexpr float ADAPTIVE_MIN_DISP_FRAC    = 0.10f;  // §1.2C
static constexpr int   CAM_FLOW_MIN_BG_PIXELS    = 500;    // cam_flow.py

// ---------------------------------------------------------------------------
// PhaseResult — return type for phase_correlate_masked_impl
// ---------------------------------------------------------------------------
struct PhaseResult { float dx, dy, response; };

// ---------------------------------------------------------------------------
// phase_correlate_masked_impl (pure C++, callable from Catch2 tests)
//
// Computes sub-pixel camera displacement via cv::phaseCorrelate.  When masks
// are provided, pixels where at least one frame is foreground are zeroed before
// the FFT to restrict correlation to background texture.
//
// mask_a / mask_b : uint8 Mat, nonzero = background.  May be empty.
// Returns {dx, dy, response} matching Python bg_masked_phase_correlate().
// ---------------------------------------------------------------------------
PhaseResult phase_correlate_masked_impl(
    const cv::Mat& fa_gray,
    const cv::Mat& fb_gray,
    const cv::Mat& mask_a,
    const cv::Mat& mask_b)
{
    if (fa_gray.size() != fb_gray.size())
        throw std::invalid_argument(
            "phase_correlate_masked: frame size mismatch");

    // Convert to float32 (cv::phaseCorrelate requires CV_32F or CV_64F)
    cv::Mat fa_f, fb_f;
    fa_gray.convertTo(fa_f, CV_32F);
    fb_gray.convertTo(fb_f, CV_32F);

    // Apply background masking when masks are supplied and large enough
    if (!mask_a.empty() && !mask_b.empty()) {
        cv::Mat bg_a, bg_b, combined_bg, fg;
        cv::compare(mask_a, 0, bg_a, cv::CMP_GT);  // nonzero → background
        cv::compare(mask_b, 0, bg_b, cv::CMP_GT);
        cv::bitwise_and(bg_a, bg_b, combined_bg);   // both-background pixels

        int n_bg = cv::countNonZero(combined_bg);
        if (n_bg >= CAM_FLOW_MIN_BG_PIXELS) {
            cv::bitwise_not(combined_bg, fg);        // foreground: not both-bg
            fa_f.setTo(0.0f, fg);
            fb_f.setTo(0.0f, fg);
        }
    }

    cv::Point2d shift;
    double response = 0.0;
    shift = cv::phaseCorrelate(fa_f, fb_f, cv::Mat(), &response);

    return {static_cast<float>(shift.x),
            static_cast<float>(shift.y),
            static_cast<float>(response)};
}

// ---------------------------------------------------------------------------
// reject_static_edges_impl (pure C++)
//
// §1.2A — Drop edges where |dx| < min_disp_px AND |dy| < min_disp_px.
// Keeps an edge if EITHER axis displacement meets the threshold.
// ---------------------------------------------------------------------------
std::vector<base::Edge> reject_static_edges_impl(
    const std::vector<base::Edge>& edges,
    float min_disp_px)
{
    std::vector<base::Edge> out;
    out.reserve(edges.size());
    for (const auto& e : edges) {
        if (std::abs(e.dx) >= min_disp_px || std::abs(e.dy) >= min_disp_px)
            out.push_back(e);
    }
    return out;
}

// ---------------------------------------------------------------------------
// compute_adaptive_min_disp_impl (pure C++)
//
// §1.2C — Content-adaptive minimum displacement threshold.
// Returns max(STATIC_EDGE_MIN_DISP_PX, ADAPTIVE_MIN_DISP_FRAC × median_step).
// Adjacent edges (dst == src + 1) define the expected inter-frame step.
// ---------------------------------------------------------------------------
float compute_adaptive_min_disp_impl(const std::vector<base::Edge>& edges)
{
    std::vector<float> adx, ady;
    for (const auto& e : edges) {
        if (e.dst == e.src + 1) {
            adx.push_back(std::abs(e.dx));
            ady.push_back(std::abs(e.dy));
        }
    }

    if (adx.empty())
        return STATIC_EDGE_MIN_DISP_PX;

    auto median_of = [](std::vector<float> v) -> float {
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        return (n % 2 == 0)
            ? (v[n / 2 - 1] + v[n / 2]) * 0.5f
            : v[n / 2];
    };

    float med_x = median_of(adx);
    float med_y = median_of(ady);
    float expected_step = (med_x >= med_y) ? median_of(adx) : median_of(ady);

    return std::max(STATIC_EDGE_MIN_DISP_PX,
                    ADAPTIVE_MIN_DISP_FRAC * expected_step);
}

// ---------------------------------------------------------------------------
// filter_edge_graph_impl (pure C++)
//
// Phase 3b gate chain: §2.14 triangular consistency, geometric consistency,
// min-step guard.  Operates on "M"-format edge dicts (keys: i, j, M, weight).
//
// Parameters
// ----------
// edges_py         : Python list of edge dicts
// min_step_px      : min-step guard threshold on dominant adjacent axis
// consistency_tol  : skip-edge chain-sum tolerance (default 15 px)
// max_tri_residual : §2.14 L2 residual ceiling (0 = disabled)
//
// Returns filtered list[dict] with §2.14 weight penalties applied.
// ---------------------------------------------------------------------------

// Parse tx (M[0,2]) and ty (M[1,2]) from an edge dict.
static std::pair<float, float> _get_m_txty(const py::dict& d)
{
    if (!d.contains("M")) return {0.0f, 0.0f};
    auto M = d["M"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto buf = M.request();
    if (buf.size < 6) return {0.0f, 0.0f};
    float* ptr = static_cast<float*>(buf.ptr);
    return {ptr[2], ptr[5]};  // M[0,2], M[1,2]
}

static py::list filter_edge_graph_impl(
    py::list edges_py,
    float    min_step_px,
    float    consistency_tol_px,
    float    max_tri_residual_px)
{
    struct E { int src, dst; float dx, dy, weight; };
    size_t N = edges_py.size();
    if (N == 0) return py::list();

    std::vector<E> es;
    es.reserve(N);
    for (size_t k = 0; k < N; ++k) {
        auto d = edges_py[k].cast<py::dict>();
        auto [tx, ty] = _get_m_txty(d);
        int src = d["i"].cast<int>();
        int dst = d["j"].cast<int>();
        float w = d.contains("weight") ? d["weight"].cast<float>() : 1.0f;
        es.push_back({src, dst, tx, ty, w});
    }

    // ── §2.14 Triangular Consistency (weight halving) ─────────────────────
    std::vector<float> penalty(N, 1.0f);
    if (max_tri_residual_px > 0.0f && N >= 3) {
        std::map<std::pair<int,int>, size_t> edge_map;
        for (size_t k = 0; k < N; ++k)
            edge_map[{es[k].src, es[k].dst}] = k;

        std::set<int> ids_set;
        for (auto& e : es) { ids_set.insert(e.src); ids_set.insert(e.dst); }
        std::vector<int> ids(ids_set.begin(), ids_set.end());
        size_t F = ids.size();

        for (size_t ai = 0; ai < F; ++ai) {
            int fi = ids[ai];
            for (size_t bi = ai + 1; bi < F; ++bi) {
                int fj = ids[bi];
                for (size_t ci = bi + 1; ci < F; ++ci) {
                    int fk = ids[ci];
                    auto it_ij = edge_map.find({fi, fj});
                    auto it_jk = edge_map.find({fj, fk});
                    auto it_ik = edge_map.find({fi, fk});
                    if (it_ij == edge_map.end() || it_jk == edge_map.end()
                        || it_ik == edge_map.end()) continue;
                    size_t idx_ij = it_ij->second;
                    size_t idx_jk = it_jk->second;
                    size_t idx_ik = it_ik->second;
                    float pred_x = es[idx_ij].dx + es[idx_jk].dx;
                    float pred_y = es[idx_ij].dy + es[idx_jk].dy;
                    float dx = es[idx_ik].dx - pred_x;
                    float dy = es[idx_ik].dy - pred_y;
                    float res = std::sqrt(dx * dx + dy * dy);
                    if (res <= max_tri_residual_px) continue;
                    float w[3] = {es[idx_ij].weight, es[idx_jk].weight, es[idx_ik].weight};
                    int weakest = 0;
                    if (w[1] < w[weakest]) weakest = 1;
                    if (w[2] < w[weakest]) weakest = 2;
                    std::array<size_t, 3> idx_arr = {idx_ij, idx_jk, idx_ik};
                    penalty[idx_arr[weakest]] *= 0.5f;
                }
            }
        }
    }

    // ── Geometric Consistency (skip-edge vs adjacent chain sum) ──────────
    std::map<int, std::pair<float, float>> adj_map;
    for (size_t k = 0; k < N; ++k)
        if (es[k].dst == es[k].src + 1)
            adj_map[es[k].src] = {es[k].dx, es[k].dy};

    std::vector<bool> keep(N, true);
    for (size_t k = 0; k < N; ++k) {
        int src = es[k].src, dst = es[k].dst;
        if (dst == src + 1) continue;
        float sum_dx = 0.0f, sum_dy = 0.0f;
        bool can_verify = true;
        for (int m = src; m < dst; ++m) {
            auto it = adj_map.find(m);
            if (it == adj_map.end()) { can_verify = false; break; }
            sum_dx += it->second.first;
            sum_dy += it->second.second;
        }
        if (can_verify) {
            if (std::abs(es[k].dx - sum_dx) >= consistency_tol_px ||
                std::abs(es[k].dy - sum_dy) >= consistency_tol_px)
                keep[k] = false;
        }
    }

    // ── Min-step guard (adjacent edges on dominant axis) ──────────────────
    if (min_step_px > 0.0f) {
        std::vector<float> adx_kept, ady_kept;
        for (size_t k = 0; k < N; ++k) {
            if (!keep[k] || es[k].dst != es[k].src + 1) continue;
            adx_kept.push_back(std::abs(es[k].dx));
            ady_kept.push_back(std::abs(es[k].dy));
        }
        if (adx_kept.size() >= 2) {
            auto median_f = [](std::vector<float> v) -> float {
                std::sort(v.begin(), v.end());
                size_t n = v.size();
                return (n % 2 == 0) ? (v[n/2-1] + v[n/2]) * 0.5f : v[n/2];
            };
            int primary = (median_f(adx_kept) >= median_f(ady_kept)) ? 0 : 1;
            for (size_t k = 0; k < N; ++k) {
                if (!keep[k] || es[k].dst != es[k].src + 1) continue;
                float val = (primary == 0) ? std::abs(es[k].dx) : std::abs(es[k].dy);
                if (val < min_step_px) keep[k] = false;
            }
        }
    }

    // ── Build output ──────────────────────────────────────────────────────
    py::list out;
    for (size_t k = 0; k < N; ++k) {
        if (!keep[k]) continue;
        if (penalty[k] == 1.0f) {
            out.append(edges_py[k]);
        } else {
            py::dict old_d = edges_py[k].cast<py::dict>();
            py::dict new_d;
            for (auto item : old_d)
                new_d[item.first] = item.second;
            float w = old_d.contains("weight") ? old_d["weight"].cast<float>() : 1.0f;
            new_d["weight"] = w * penalty[k];
            out.append(new_d);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------
#ifndef BATCH_TESTS

// ---------------------------------------------------------------------------
// phase_correlate_masked
//
// Python API:
//   result = batch.matching.phase_correlate_masked(frame_a, frame_b)
//   result = batch.matching.phase_correlate_masked(frame_a, frame_b, mask_a, mask_b)
//   # result is a dict with "dx", "dy", "response"
// ---------------------------------------------------------------------------
static py::dict phase_correlate_masked(
    py::array_t<uint8_t> frame_a_gray,
    py::array_t<uint8_t> frame_b_gray,
    py::object           bg_mask_a,
    py::object           bg_mask_b)
{
    cv::Mat fa = base::mat_from_array(frame_a_gray);
    cv::Mat fb = base::mat_from_array(frame_b_gray);

    cv::Mat ma, mb;
    if (!bg_mask_a.is_none())
        ma = base::mat_from_array(bg_mask_a.cast<py::array_t<uint8_t>>());
    if (!bg_mask_b.is_none())
        mb = base::mat_from_array(bg_mask_b.cast<py::array_t<uint8_t>>());

    auto r = phase_correlate_masked_impl(fa, fb, ma, mb);

    py::dict out;
    out["dx"]       = r.dx;
    out["dy"]       = r.dy;
    out["response"] = r.response;
    return out;
}

// ---------------------------------------------------------------------------
// reject_static_edges — Python wrapper
// ---------------------------------------------------------------------------
static py::list reject_static_edges(py::list edges_py, float min_disp_px)
{
    std::vector<base::Edge> edges;
    edges.reserve(py::len(edges_py));
    for (auto item : edges_py)
        edges.push_back(base::edge_from_dict(item.cast<py::dict>()));

    auto filtered = reject_static_edges_impl(edges, min_disp_px);
    return base::edges_to_list(filtered);
}

// ---------------------------------------------------------------------------
// compute_adaptive_min_disp — Python wrapper
// ---------------------------------------------------------------------------
static float compute_adaptive_min_disp(py::list edges_py)
{
    std::vector<base::Edge> edges;
    edges.reserve(py::len(edges_py));
    for (auto item : edges_py)
        edges.push_back(base::edge_from_dict(item.cast<py::dict>()));
    return compute_adaptive_min_disp_impl(edges);
}

// ---------------------------------------------------------------------------
// filter_edge_graph — Phase 3b: classical post-match gate chain
// ---------------------------------------------------------------------------
static py::list filter_edge_graph(
    py::list edges_py,
    float    min_step_px         = 10.0f,
    float    consistency_tol_px  = 15.0f,
    float    max_tri_residual_px = 0.0f)
{
    return filter_edge_graph_impl(edges_py, min_step_px,
                                  consistency_tol_px, max_tri_residual_px);
}

// ---------------------------------------------------------------------------
// build_edge_graph — stub (Phase 3 in-progress; complex gate chain)
// ---------------------------------------------------------------------------
static py::list build_edge_graph(py::list, py::list, int)
{
    BATCH_NOT_IMPLEMENTED("matching.build_edge_graph");
}

// ---------------------------------------------------------------------------
// spatial_dedup_frames — stub (Phase 5; bounding-box overlap check)
// ---------------------------------------------------------------------------
static py::list spatial_dedup_frames(py::list, py::list, py::list,
                                     py::list, py::list, float)
{
    BATCH_NOT_IMPLEMENTED("matching.spatial_dedup_frames");
}

// ---------------------------------------------------------------------------
// register_matching — called from bindings.cpp
// ---------------------------------------------------------------------------
void register_matching(py::module_& m) {
    m.doc() = R"doc(
        batch.matching — Phase correlation and edge graph functions.

        Implemented
        -----------
        phase_correlate_masked(frame_a, frame_b, mask_a=None, mask_b=None) -> dict
        reject_static_edges(edges, min_disp_px=50.0) -> list[dict]
        compute_adaptive_min_disp(edges) -> float

        Phase 3b
        --------
        filter_edge_graph(edges, min_step_px, consistency_tol_px, max_tri_residual_px)
            -> list[dict]

        Stubs (Phase 3 / Phase 5)
        -------------------------
        build_edge_graph(raw_matches, bg_masks, N) -> list[dict]
        spatial_dedup_frames(frames, scans, masks, paths, edges, min_disp) -> list[int]
    )doc";

    m.def("phase_correlate_masked", &phase_correlate_masked,
        py::arg("frame_a_gray"),
        py::arg("frame_b_gray"),
        py::arg("bg_mask_a") = py::none(),
        py::arg("bg_mask_b") = py::none(),
        R"doc(
            Compute phase-correlation shift between two luma frames.

            Background pixels (where both masks are nonzero) are used for
            correlation; foreground pixels are zeroed.  Falls back to
            whole-frame correlation when fewer than 500 bg pixels are available.

            Args
            ----
            frame_a_gray, frame_b_gray : uint8 (H, W) luma planes
            bg_mask_a, bg_mask_b       : optional uint8 (H, W) masks,
                                         nonzero = background

            Returns dict with "dx" (float), "dy" (float), "response" (float).
        )doc");

    m.def("reject_static_edges", &reject_static_edges,
        py::arg("edges"),
        py::arg("min_disp_px") = STATIC_EDGE_MIN_DISP_PX,
        R"doc(
            §1.2A — Drop edges where |dx| < min_disp_px AND |dy| < min_disp_px.

            An edge is kept if EITHER axis displacement meets the threshold
            (preserves valid diagonal-scroll edges).

            Returns filtered list[dict].
        )doc");

    m.def("compute_adaptive_min_disp", &compute_adaptive_min_disp,
        py::arg("edges"),
        R"doc(
            §1.2C — Content-adaptive minimum displacement threshold.

            Estimates the expected inter-frame step from adjacent-edge medians
            and returns max(STATIC_EDGE_MIN_DISP_PX, 0.10 × expected_step).

            Returns float (pixels).
        )doc");

    m.def("filter_edge_graph", &filter_edge_graph,
        py::arg("edges"),
        py::arg("min_step_px")          = 10.0f,
        py::arg("consistency_tol_px")   = 15.0f,
        py::arg("max_tri_residual_px")  = 0.0f,
        R"doc(
            Phase 3b classical gate chain on "M"-format edge dicts.

            Applies in order:
              1. §2.14 Triangular Consistency — halves weight of weakest edge
                 in every inconsistent triangle (L2 residual > max_tri_residual_px).
                 Disabled when max_tri_residual_px == 0.
              2. Geometric Consistency — rejects skip edges whose measured
                 displacement disagrees with the adjacent-edge chain sum by more
                 than consistency_tol_px on either axis.
              3. Min-step guard — drops adjacent edges whose displacement on the
                 dominant axis is below min_step_px.

            Args
            ----
            edges                : list of edge dicts with keys "i", "j", "M", "weight"
            min_step_px          : min-step guard threshold (default 10 px)
            consistency_tol_px   : geometric consistency tolerance (default 15 px)
            max_tri_residual_px  : §2.14 L2 residual ceiling; 0 = disabled (default)

            Returns filtered list[dict] with weight penalties applied.
        )doc");

    m.def("build_edge_graph", &build_edge_graph,
        py::arg("raw_matches"),
        py::arg("bg_masks"),
        py::arg("N"),
        R"doc(
            Filter raw match dicts through post-match gates (stub — Phase 3).
        )doc");

    m.def("spatial_dedup_frames", &spatial_dedup_frames,
        py::arg("frames"),
        py::arg("scans_frames"),
        py::arg("bg_masks"),
        py::arg("image_paths"),
        py::arg("edges"),
        py::arg("min_displacement_px"),
        R"doc(
            Bounding-box overlap spatial deduplication (stub — Phase 5).
        )doc");
}

#endif // BATCH_TESTS
