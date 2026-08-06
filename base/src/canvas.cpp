// ---------------------------------------------------------------------------
// batch/src/canvas.cpp
//
// Warp frames to canvas, per-pixel median render, crop, fill, scroll detect,
// panorama stitch fallback.
//
// Replaces:
//   alignment/canvas.py  :: _compute_canvas, _crop_to_valid, _telea_fill_gaps,
//                           _detect_scroll_axis, _panorama_stitch_fallback
//   rendering/rendering.py :: _render_median (basic hot-path)
//
// Phase 5 implementation.
// See docs/moon/roadmaps/asp_cpp_migration.md §base::canvas
// ---------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/photo.hpp>

#ifdef HAVE_CUDA
#include <opencv2/cuda.hpp>
#endif

#include "common.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Local helpers (same pattern as seam.cpp)
// ---------------------------------------------------------------------------

/// Zero-copy read; clone because warpAffine needs contiguous owned memory.
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

/// Extract the (2,3) float32 affine matrix from a numpy array.
inline cv::Mat affine_from_arr(py::array_t<float> arr) {
    auto req = arr.request();
    if (req.ndim != 2 || req.shape[0] != 2 || req.shape[1] != 3)
        throw std::runtime_error("affine must be shape (2,3) float32");
    cv::Mat M(2, 3, CV_32F);
    std::memcpy(M.data, req.ptr, 6 * sizeof(float));
    return M;
}

// ---------------------------------------------------------------------------
// compute_canvas
//
// Compute bounding box from warped corners of all frames.
// Returns (canvas_h: int, canvas_w: int, shift_x: float, shift_y: float)
// where shift_{x,y} is the offset to add to all affine translations so the
// minimum warped corner falls at (0, 0).
// ---------------------------------------------------------------------------
static py::tuple compute_canvas(
    py::list affines,
    py::list frame_shapes)   // list of (H, W) or (H, W, C) tuples
{
    size_t N = affines.size();
    if (N == 0) throw std::runtime_error("compute_canvas: empty input");
    if (frame_shapes.size() != N)
        throw std::runtime_error("compute_canvas: affines and frame_shapes must match");

    float min_x =  std::numeric_limits<float>::infinity();
    float min_y =  std::numeric_limits<float>::infinity();
    float max_x = -std::numeric_limits<float>::infinity();
    float max_y = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < N; ++i) {
        cv::Mat M = affine_from_arr(affines[i].cast<py::array_t<float>>());
        const float* m = M.ptr<float>();  // row-major 2x3: m00 m01 m02 m10 m11 m12
        auto sh = frame_shapes[i].cast<py::tuple>();
        float fH = sh[0].cast<float>(), fW = sh[1].cast<float>();

        // Four corners in (x, y) image order
        float cx[4] = {0.f, fW, fW, 0.f};
        float cy[4] = {0.f, 0.f, fH, fH};

        for (int k = 0; k < 4; ++k) {
            float wx = m[0]*cx[k] + m[1]*cy[k] + m[2];
            float wy = m[3]*cx[k] + m[4]*cy[k] + m[5];
            min_x = std::min(min_x, wx); max_x = std::max(max_x, wx);
            min_y = std::min(min_y, wy); max_y = std::max(max_y, wy);
        }
    }

    int canvas_w = static_cast<int>(std::ceil(max_x - min_x));
    int canvas_h = static_cast<int>(std::ceil(max_y - min_y));
    // Python caller applies CANVAS_MAX_DIM guard after this call.
    return py::make_tuple(canvas_h, canvas_w, -min_x, -min_y);
}

// ---------------------------------------------------------------------------
// warp_frames_to_canvas
//
// Apply cv::warpAffine to all N frames in parallel via OpenMP.
// Affines are (2,3) float32 ndarrays (tx/ty already shifted by caller).
//
// try_gpu: when true, attempt OpenCL-accelerated warpAffine via cv::UMat;
//          silently falls back to OpenMP CPU path on any failure.
//
// Returns list of N warped uint8 ndarrays of shape (canvas_h, canvas_w, C).
// ---------------------------------------------------------------------------
static py::list warp_frames_to_canvas(
    py::list frames,
    py::list affines,
    int      canvas_h,
    int      canvas_w,
    bool     try_gpu = false)
{
    size_t N = frames.size();
    if (N == 0) return py::list();
    if (affines.size() != N)
        throw std::runtime_error(
            "warp_frames_to_canvas: frames and affines must have same length");

    std::vector<cv::Mat> f_mats(N), Ms(N);
    for (size_t i = 0; i < N; ++i) {
        f_mats[i] = as_mat(frames[i].cast<py::array_t<uint8_t>>());
        Ms[i] = affine_from_arr(affines[i].cast<py::array_t<float>>());
    }

    cv::Size dst_size(canvas_w, canvas_h);
    std::vector<cv::Mat> warped(N);

    // OpenCL/UMat fast path — each frame is uploaded to GPU memory for warpAffine.
    // Sequential (no OpenMP) because OpenCL commands are already async-queued.
    if (try_gpu) {
        bool gpu_ok = true;
        {
            py::gil_scoped_release release;
            for (int i = 0; i < static_cast<int>(N) && gpu_ok; ++i) {
                try {
                    cv::UMat src_u, dst_u;
                    f_mats[i].copyTo(src_u);
                    cv::warpAffine(src_u, dst_u, Ms[i], dst_size,
                                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
                    dst_u.copyTo(warped[i]);
                } catch (...) {
                    gpu_ok = false;
                }
            }
        }
        if (gpu_ok) {
            py::list result;
            for (size_t i = 0; i < N; ++i)
                result.append(base::array_from_mat(warped[i]));
            return result;
        }
        // fall through to CPU path
    }

    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(N); ++i) {
            warped[i] = cv::Mat::zeros(dst_size, f_mats[i].type());
            cv::warpAffine(f_mats[i], warped[i], Ms[i], dst_size,
                           cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        }
    }

    py::list result;
    for (size_t i = 0; i < N; ++i)
        result.append(base::array_from_mat(warped[i]));
    return result;
}

// ---------------------------------------------------------------------------
// render_median
//
// Per-pixel temporal median across N warped frames (OpenMP parallel over rows).
// Content-aware: pixels where all channels == 0 are excluded from the sample.
//
// try_gpu: reserved for future CUDA nth_element kernel; currently no-op
//          (nth_element is CPU-only in OpenCV; UMat provides no benefit here).
//
// Returns uint8 ndarray (canvas_h, canvas_w, C).
// ---------------------------------------------------------------------------
static py::array_t<uint8_t> render_median(py::list warped_frames, bool try_gpu = false)
{
    (void)try_gpu;  // GPU nth_element not yet implemented — CPU path always used
    size_t N = warped_frames.size();
    if (N == 0) throw std::runtime_error("render_median: empty frame list");

    std::vector<cv::Mat> mats(N);
    for (size_t i = 0; i < N; ++i)
        mats[i] = as_mat(warped_frames[i].cast<py::array_t<uint8_t>>());

    int H = mats[0].rows, W = mats[0].cols, C = mats[0].channels();
    for (size_t i = 1; i < N; ++i) {
        if (mats[i].rows != H || mats[i].cols != W || mats[i].channels() != C)
            throw std::runtime_error(
                "render_median: all frames must have the same shape");
    }

    cv::Mat out = cv::Mat::zeros(H, W, mats[0].type());
    {
        py::gil_scoped_release release;
        #pragma omp parallel for schedule(dynamic, 4)
        for (int y = 0; y < H; ++y) {
            std::vector<uint8_t> col_vals(N);
            for (int x = 0; x < W; ++x) {
                for (int c = 0; c < C; ++c) {
                    int n_valid = 0;
                    for (size_t fi = 0; fi < N; ++fi) {
                        const uint8_t* row = mats[fi].ptr<uint8_t>(y);
                        bool has_content = false;
                        for (int cc = 0; cc < C; ++cc) {
                            if (row[x * C + cc] > 0) { has_content = true; break; }
                        }
                        if (has_content)
                            col_vals[n_valid++] = row[x * C + c];
                    }
                    if (n_valid == 0) continue;
                    int mi = n_valid / 2;
                    std::nth_element(col_vals.begin(),
                                     col_vals.begin() + mi,
                                     col_vals.begin() + n_valid);
                    out.ptr<uint8_t>(y)[x * C + c] = col_vals[mi];
                }
            }
        }
    }
    return base::array_from_mat(out);
}

// ---------------------------------------------------------------------------
// crop_to_valid
//
// Tight bounding box of non-zero content pixels.
// valid_fraction = minimum non-zero fraction in a row/col to count as content.
//
// Returns (y0, y1, x0, x1) — y1/x1 are exclusive (Python slice style).
// ---------------------------------------------------------------------------
static py::tuple crop_to_valid(
    py::array_t<uint8_t> canvas_arr,
    float                valid_fraction = 0.0f)
{
    cv::Mat canvas = as_mat(canvas_arr);
    int H = canvas.rows, W = canvas.cols;

    cv::Mat gray;
    if (canvas.channels() == 1)
        gray = canvas.clone();
    else
        cv::cvtColor(canvas, gray, cv::COLOR_BGR2GRAY);

    int y0 = H, y1 = -1;
    for (int y = 0; y < H; ++y) {
        const uint8_t* row = gray.ptr<uint8_t>(y);
        int cnt = 0;
        for (int x = 0; x < W; ++x)
            if (row[x] > 0) ++cnt;
        if (float(cnt) / float(W) > valid_fraction) {
            if (y < y0) y0 = y;
            y1 = y;
        }
    }
    if (y1 < 0) { y0 = 0; y1 = H - 1; }

    int x0 = W, x1 = -1;
    for (int x = 0; x < W; ++x) {
        int cnt = 0;
        for (int y = y0; y <= y1; ++y)
            if (gray.ptr<uint8_t>(y)[x] > 0) ++cnt;
        if (float(cnt) / float(y1 - y0 + 1) > valid_fraction) {
            if (x < x0) x0 = x;
            x1 = x;
        }
    }
    if (x1 < 0) { x0 = 0; x1 = W - 1; }

    return py::make_tuple(y0, y1 + 1, x0, x1 + 1);
}

// ---------------------------------------------------------------------------
// telea_fill_gaps
//
// Inpaint gap_mask regions using cv::inpaint(INPAINT_TELEA, radius).
// Returns a new filled ndarray (deep copy).
// ---------------------------------------------------------------------------
static py::array_t<uint8_t> telea_fill_gaps(
    py::array_t<uint8_t> canvas_arr,
    py::array_t<uint8_t> gap_mask_arr,
    int                  inpaint_radius = 3)
{
    cv::Mat canvas   = as_mat(canvas_arr);
    cv::Mat gap_mask = as_mat(gap_mask_arr);

    if (gap_mask.channels() > 1)
        cv::cvtColor(gap_mask, gap_mask, cv::COLOR_BGR2GRAY);

    if (cv::countNonZero(gap_mask) == 0)
        return base::array_from_mat(canvas);

    cv::Mat result;
    {
        py::gil_scoped_release release;
        cv::inpaint(canvas, gap_mask, result, inpaint_radius, cv::INPAINT_TELEA);
    }
    return base::array_from_mat(result);
}

// ---------------------------------------------------------------------------
// detect_scroll_axis
//
// Classify scroll direction from list of (2,3) float32 affine matrices.
// M[1,2] = ty, M[0,2] = tx.
//
// Returns "vertical" | "horizontal" | "diagonal" | "none".
// ---------------------------------------------------------------------------
static std::string detect_scroll_axis(py::list affines)
{
    size_t N = affines.size();
    if (N < 2) return "none";

    float tx_min =  std::numeric_limits<float>::infinity();
    float tx_max = -std::numeric_limits<float>::infinity();
    float ty_min =  std::numeric_limits<float>::infinity();
    float ty_max = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < N; ++i) {
        cv::Mat M = affine_from_arr(affines[i].cast<py::array_t<float>>());
        float tx = M.at<float>(0, 2);
        float ty = M.at<float>(1, 2);
        tx_min = std::min(tx_min, tx); tx_max = std::max(tx_max, tx);
        ty_min = std::min(ty_min, ty); ty_max = std::max(ty_max, ty);
    }

    float tx_range = tx_max - tx_min;
    float ty_range = ty_max - ty_min;
    float total = tx_range + ty_range;

    if (total < 1.0f) return "none";
    if (tx_range > 0.0f && ty_range / std::max(tx_range, 1.0f) < 0.1f)
        return "horizontal";
    if (ty_range > 0.0f && tx_range / std::max(ty_range, 1.0f) > 0.3f)
        return "diagonal";
    return "vertical";
}

// ---------------------------------------------------------------------------
// panorama_stitch_fallback
//
// Wrap cv::Stitcher_create(PANORAMA).stitch() with error-code handling.
// Returns (ok: bool, stitched: ndarray | None).
// ---------------------------------------------------------------------------
static py::tuple panorama_stitch_fallback(py::list frames)
{
    size_t N = frames.size();
    if (N == 0) return py::make_tuple(false, py::none());

    std::vector<cv::Mat> imgs(N);
    for (size_t i = 0; i < N; ++i)
        imgs[i] = as_mat(frames[i].cast<py::array_t<uint8_t>>());

    cv::Mat pano;
    cv::Stitcher::Status status;
    {
        py::gil_scoped_release release;
        auto stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);
        status = stitcher->stitch(imgs, pano);
    }

    if (status != cv::Stitcher::OK)
        return py::make_tuple(false, py::none());

    return py::make_tuple(true, base::array_from_mat(pano));
}

// ---------------------------------------------------------------------------
// gpu_device_count
//
// Returns the number of CUDA-capable devices visible to OpenCV.
// Returns 0 when built without CUDA support.
// ---------------------------------------------------------------------------
static int gpu_device_count()
{
#ifdef HAVE_CUDA
    return cv::cuda::getCudaEnabledDeviceCount();
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// register_canvas — called from bindings.cpp
// ---------------------------------------------------------------------------
void register_canvas(py::module_& m) {
    m.doc() = R"doc(
        batch.canvas — Warp, crop, fill, and median render.

        Functions
        ---------
        compute_canvas(affines, frame_shapes) -> (canvas_h, canvas_w, shift_x, shift_y)
        warp_frames_to_canvas(frames, affines, canvas_h, canvas_w, try_gpu) -> list[ndarray]
        render_median(warped_frames, try_gpu) -> ndarray
        crop_to_valid(canvas, valid_fraction) -> (y0, y1, x0, x1)
        telea_fill_gaps(canvas, gap_mask, inpaint_radius) -> ndarray
        detect_scroll_axis(affines) -> str
        panorama_stitch_fallback(frames) -> (bool, ndarray | None)
        gpu_device_count() -> int
    )doc";

    m.def("compute_canvas", &compute_canvas,
        py::arg("affines"),
        py::arg("frame_shapes"),
        "Compute bounding box of warped frame corners; returns (H, W, shift_x, shift_y).");

    m.def("warp_frames_to_canvas", &warp_frames_to_canvas,
        py::arg("frames"),
        py::arg("affines"),
        py::arg("canvas_h"),
        py::arg("canvas_w"),
        py::arg("try_gpu") = false,
        "Apply cv::warpAffine to all frames; UMat OpenCL path when try_gpu=True.");

    m.def("render_median", &render_median,
        py::arg("warped_frames"),
        py::arg("try_gpu") = false,
        "Per-pixel content-aware median across N warped frames (OpenMP nth_element).");

    m.def("crop_to_valid", &crop_to_valid,
        py::arg("canvas"),
        py::arg("valid_fraction") = 0.0f,
        "Tight bounding box of non-zero content; returns (y0, y1, x0, x1) exclusive.");

    m.def("telea_fill_gaps", &telea_fill_gaps,
        py::arg("canvas"),
        py::arg("gap_mask"),
        py::arg("inpaint_radius") = 3,
        "Inpaint gap regions via cv::inpaint(INPAINT_TELEA).");

    m.def("detect_scroll_axis", &detect_scroll_axis,
        py::arg("affines"),
        "Classify scroll axis from affine tx/ty ranges: vertical/horizontal/diagonal/none.");

    m.def("panorama_stitch_fallback", &panorama_stitch_fallback,
        py::arg("frames"),
        "Attempt cv::Stitcher_create(PANORAMA).stitch(); returns (ok, result_or_None).");

    m.def("gpu_device_count", &gpu_device_count,
        "Number of CUDA-capable devices visible to OpenCV (0 when built without CUDA).");
}
