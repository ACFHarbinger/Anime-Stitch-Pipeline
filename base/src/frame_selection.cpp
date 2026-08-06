// ---------------------------------------------------------------------------
// batch/src/frame_selection.cpp
//
// Classical frame filtering: hold detection, temporal variance, dedup.
//
// Replaces (non-DINOv2 paths):
//   ingestion/frame_selection.py  :: _detect_hold_blocks, _detect_hold_blocks_dhash,
//     _temporal_variance_filter, _near_dup_luma_filter, _spatial_dedup_frames
//
// Phase 5 implementation.
// See docs/moon/roadmaps/asp_cpp_migration.md §base::frame_selection
// ---------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "common.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

inline cv::Mat as_mat(py::array_t<uint8_t> arr) {
    auto req = arr.request();
    int type = (req.ndim == 3 && req.shape[2] == 3) ? CV_8UC3
             : (req.ndim == 3 && req.shape[2] == 4) ? CV_8UC4
             : CV_8UC1;
    return cv::Mat(
        static_cast<int>(req.shape[0]),
        static_cast<int>(req.shape[1]),
        type, req.ptr).clone();
}

/// Convert BGR or grey thumbnail to float32 luma [0,1], resized to (h, w).
static cv::Mat to_luma_f32(const cv::Mat& src, int h = 0, int w = 0)
{
    cv::Mat gray;
    if (src.channels() == 1)
        gray = src.clone();
    else
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    if (h > 0 && w > 0 && (gray.rows != h || gray.cols != w))
        cv::resize(gray, gray, cv::Size(w, h), 0, 0, cv::INTER_AREA);

    cv::Mat f;
    gray.convertTo(f, CV_32F, 1.0 / 255.0);
    return f;
}

// ---------------------------------------------------------------------------
// detect_hold_blocks_mad
//
// Per-pair mean absolute difference (MAD) of thumbnail luma planes.
// A frame is a "hold" if its MAD against the previous frame < threshold.
// Frame 0 is never a hold.
//
// OpenMP: each pair computed independently.
// Returns list[int] — indices of hold frames (duplicates of previous frame).
// ---------------------------------------------------------------------------
static py::list detect_hold_blocks_mad(
    py::list thumbs,
    float    threshold = 0.025f)
{
    size_t N = thumbs.size();
    std::vector<cv::Mat> lumas(N);
    for (size_t i = 0; i < N; ++i)
        lumas[i] = to_luma_f32(as_mat(thumbs[i].cast<py::array_t<uint8_t>>()));

    std::vector<float> mads(N, 1.0f);  // frame 0 never hold
    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < static_cast<int>(N); ++i) {
            const cv::Mat& a = lumas[i - 1];
            const cv::Mat& b = lumas[i];
            // Resize b to match a if needed
            cv::Mat b_rs = b;
            if (b.rows != a.rows || b.cols != a.cols)
                cv::resize(b, b_rs, a.size(), 0, 0, cv::INTER_AREA);
            cv::Mat diff;
            cv::absdiff(a, b_rs, diff);
            mads[i] = static_cast<float>(cv::mean(diff)[0]);
        }
    }

    py::list result;
    for (size_t i = 1; i < N; ++i)
        if (mads[i] < threshold)
            result.append(static_cast<int>(i));
    return result;
}

// ---------------------------------------------------------------------------
// detect_hold_blocks_dhash
//
// Perceptual dHash hold detection:
//   1. INTER_AREA resize to (hash_size, hash_size*2)
//   2. Horizontal gradient binarisation: bit[y][x] = (row[x+1] > row[x])
//   3. Hamming distance (XOR popcount)
//
// Returns list[int] — hold frame indices where hamming_dist <= hamming_thresh.
// ---------------------------------------------------------------------------
static py::list detect_hold_blocks_dhash(
    py::list thumbs,
    int      hash_size      = 8,
    int      hamming_thresh = 4)
{
    size_t N = thumbs.size();
    int hash_bits = hash_size * hash_size;

    // Build hashes for all frames
    std::vector<std::vector<uint8_t>> hashes(N, std::vector<uint8_t>(hash_bits));
    for (size_t i = 0; i < N; ++i) {
        cv::Mat src = as_mat(thumbs[i].cast<py::array_t<uint8_t>>());
        cv::Mat gray;
        if (src.channels() == 1)
            gray = src.clone();
        else
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        // INTER_AREA resize to (hash_size, hash_size+1) wide to allow horizontal diff
        cv::Mat resized;
        cv::resize(gray, resized, cv::Size(hash_size + 1, hash_size),
                   0, 0, cv::INTER_AREA);
        // Compute horizontal gradient binarisation
        for (int y = 0; y < hash_size; ++y) {
            const uint8_t* row = resized.ptr<uint8_t>(y);
            for (int x = 0; x < hash_size; ++x)
                hashes[i][y * hash_size + x] = (row[x + 1] > row[x]) ? 1u : 0u;
        }
    }

    // Compute pairwise Hamming distances
    py::list result;
    for (size_t i = 1; i < N; ++i) {
        int dist = 0;
        for (int b = 0; b < hash_bits; ++b)
            dist += (hashes[i][b] != hashes[i - 1][b]) ? 1 : 0;
        if (dist <= hamming_thresh)
            result.append(static_cast<int>(i));
    }
    return result;
}

// ---------------------------------------------------------------------------
// temporal_variance_filter
//
// For each interior frame i (1 .. N-2): compute mean per-pixel variance
// across the triplet (i-1, i, i+1) in [0,1]² float space.
// Drop frame if mean variance < sigma_threshold.
// First and last frames are always kept.
//
// Returns (kept_thumbs: list[ndarray], kept_paths: list[str]).
// ---------------------------------------------------------------------------
static py::tuple temporal_variance_filter(
    py::list thumbs,
    py::list paths,
    float    sigma_threshold = 1e-3f)
{
    size_t N = thumbs.size();
    if (N < 3 || sigma_threshold <= 0.0f) {
        // Nothing to filter
        return py::make_tuple(thumbs, paths);
    }

    std::vector<cv::Mat> lumas(N);
    for (size_t i = 0; i < N; ++i)
        lumas[i] = to_luma_f32(as_mat(thumbs[i].cast<py::array_t<uint8_t>>()));

    std::vector<bool> keep(N, true);
    {
        py::gil_scoped_release release;
        for (size_t i = 1; i + 1 < N; ++i) {
            const cv::Mat& a = lumas[i - 1];
            const cv::Mat& b = lumas[i];
            const cv::Mat& c = lumas[i + 1];
            // Resize b, c to a's size if needed
            cv::Mat b_rs = b, c_rs = c;
            if (b.size() != a.size()) cv::resize(b, b_rs, a.size(), 0, 0, cv::INTER_AREA);
            if (c.size() != a.size()) cv::resize(c, c_rs, a.size(), 0, 0, cv::INTER_AREA);

            // Per-pixel variance: Var = E[x²] - E[x]²
            cv::Mat mean_mat = (a + b_rs + c_rs) / 3.0f;
            cv::Mat var = (a.mul(a) + b_rs.mul(b_rs) + c_rs.mul(c_rs)) / 3.0f
                          - mean_mat.mul(mean_mat);
            float mean_var = static_cast<float>(cv::mean(var)[0]);
            if (mean_var < sigma_threshold)
                keep[i] = false;
        }
    }

    py::list out_thumbs, out_paths;
    for (size_t i = 0; i < N; ++i) {
        if (keep[i]) {
            out_thumbs.append(thumbs[i]);
            out_paths.append(paths[i]);
        }
    }
    return py::make_tuple(out_thumbs, out_paths);
}

// ---------------------------------------------------------------------------
// near_dup_luma_filter
//
// Per-pair mean abs grayscale diff at thumbnail scale.
// Drop frame if diff against previous kept frame < threshold.
// First and last frames are always kept.
//
// Returns (kept_thumbs: list[ndarray], kept_paths: list[str]).
// ---------------------------------------------------------------------------
static py::tuple near_dup_luma_filter(
    py::list thumbs,
    py::list paths,
    float    threshold = 3.0f)
{
    size_t N = thumbs.size();
    if (N <= 2 || threshold <= 0.0f)
        return py::make_tuple(thumbs, paths);

    // Convert threshold from [0,255] scale to [0,1] float
    float thr_f = threshold / 255.0f;

    std::vector<cv::Mat> lumas(N);
    for (size_t i = 0; i < N; ++i)
        lumas[i] = to_luma_f32(as_mat(thumbs[i].cast<py::array_t<uint8_t>>()));

    py::list out_thumbs, out_paths;
    out_thumbs.append(thumbs[0]);
    out_paths.append(paths[0]);

    size_t prev = 0;
    for (size_t i = 1; i + 1 < N; ++i) {
        const cv::Mat& a = lumas[prev];
        const cv::Mat& b = lumas[i];
        cv::Mat b_rs = b;
        if (b.size() != a.size()) cv::resize(b, b_rs, a.size(), 0, 0, cv::INTER_AREA);
        cv::Mat diff;
        cv::absdiff(a, b_rs, diff);
        float mad = static_cast<float>(cv::mean(diff)[0]);
        if (mad >= thr_f) {
            out_thumbs.append(thumbs[i]);
            out_paths.append(paths[i]);
            prev = i;
        }
    }
    // Always keep last
    out_thumbs.append(thumbs[N - 1]);
    out_paths.append(paths[N - 1]);
    return py::make_tuple(out_thumbs, out_paths);
}

// ---------------------------------------------------------------------------
// spatial_dedup_frames
//
// Drop frames that are spatially redundant with an already-accepted frame.
// Uses per-frame bounding boxes derived from displacement edges.
//
// Simplified: frame i is a dup of frame j if |dy(i) - dy(j)| < min_displacement_px
// AND |dx(i) - dx(j)| < min_displacement_px (both axes too close).
// Edge dict format: {"i": src, "j": dst, "dx": float, "dy": float, ...}.
//
// Returns list[int] — indices of frames to KEEP.
// ---------------------------------------------------------------------------
static py::list spatial_dedup_frames(
    py::list frames,
    py::list scans_frames,
    py::list bg_masks,
    py::list image_paths,
    py::list edges,
    float    min_displacement_px)
{
    size_t N = frames.size();
    if (N == 0) {
        return py::list();
    }

    // Build cumulative displacement per frame from edges (sorted by src index)
    std::vector<float> cum_dx(N, 0.0f), cum_dy(N, 0.0f);
    size_t E = edges.size();
    for (size_t e = 0; e < E; ++e) {
        auto d = edges[e].cast<py::dict>();
        // Accept both "i"/"j" and "src"/"dst" key conventions
        int src = -1, dst = -1;
        if (d.contains("i"))   src = d["i"].cast<int>();
        if (d.contains("src")) src = d["src"].cast<int>();
        if (d.contains("j"))   dst = d["j"].cast<int>();
        if (d.contains("dst")) dst = d["dst"].cast<int>();
        if (src < 0 || dst < 0) continue;
        if (src >= static_cast<int>(N) || dst >= static_cast<int>(N)) continue;
        // Use the displacement as-is; cumulative just propagates
        if (d.contains("dx")) cum_dx[dst] = cum_dx[src] + d["dx"].cast<float>();
        if (d.contains("dy")) cum_dy[dst] = cum_dy[src] + d["dy"].cast<float>();
    }

    // Greedy keep: accept frame if it is far enough from all accepted frames
    std::vector<int> keep_indices;
    keep_indices.push_back(0);  // always keep first

    for (size_t i = 1; i < N; ++i) {
        bool is_dup = false;
        for (int k : keep_indices) {
            float ddx = std::abs(cum_dx[i] - cum_dx[k]);
            float ddy = std::abs(cum_dy[i] - cum_dy[k]);
            if (ddx < min_displacement_px && ddy < min_displacement_px) {
                is_dup = true;
                break;
            }
        }
        if (!is_dup)
            keep_indices.push_back(static_cast<int>(i));
    }

    // Always ensure last frame is kept (Python impl keeps it)
    if (keep_indices.empty() ||
        keep_indices.back() != static_cast<int>(N) - 1)
        keep_indices.push_back(static_cast<int>(N) - 1);

    py::list result;
    for (int idx : keep_indices)
        result.append(idx);
    return result;
}

// ---------------------------------------------------------------------------
// register_frame_selection — called from bindings.cpp
// ---------------------------------------------------------------------------
void register_frame_selection(py::module_& m) {
    m.doc() = R"doc(
        batch.frame_selection — Classical frame filtering.

        Functions
        ---------
        detect_hold_blocks_mad(thumbs, threshold) -> list[int]
        detect_hold_blocks_dhash(thumbs, hash_size, hamming_thresh) -> list[int]
        temporal_variance_filter(thumbs, paths, sigma_threshold) -> (list, list)
        near_dup_luma_filter(thumbs, paths, threshold) -> (list, list)
        spatial_dedup_frames(frames, scans, masks, paths, edges, min_disp) -> list[int]
    )doc";

    m.def("detect_hold_blocks_mad", &detect_hold_blocks_mad,
        py::arg("thumbs"),
        py::arg("threshold") = 0.025f,
        "Detect hold (duplicate) frames via per-pair MAD on thumbnail luma (OpenMP).");

    m.def("detect_hold_blocks_dhash", &detect_hold_blocks_dhash,
        py::arg("thumbs"),
        py::arg("hash_size")      = 8,
        py::arg("hamming_thresh") = 4,
        "Detect hold frames via dHash (INTER_AREA resize + Hamming distance).");

    m.def("temporal_variance_filter", &temporal_variance_filter,
        py::arg("thumbs"),
        py::arg("paths"),
        py::arg("sigma_threshold") = 1e-3f,
        "Drop interior frames with per-triplet variance < sigma_threshold.");

    m.def("near_dup_luma_filter", &near_dup_luma_filter,
        py::arg("thumbs"),
        py::arg("paths"),
        py::arg("threshold") = 3.0f,
        "Drop near-duplicate frames by mean abs luma diff (first/last always kept).");

    m.def("spatial_dedup_frames", &spatial_dedup_frames,
        py::arg("frames"),
        py::arg("scans_frames"),
        py::arg("bg_masks"),
        py::arg("image_paths"),
        py::arg("edges"),
        py::arg("min_displacement_px"),
        "Drop frames whose cumulative displacement overlaps an accepted frame's position.");
}
