// ---------------------------------------------------------------------------
// batch/src/exposure.cpp
//
// Block-based gain compensation and vignetting correction.
//
// Replaces:
//   rendering/photometric.py         :: _apply_basic, _correct_vignetting
//   rendering/compositing.py         :: _blocks_gain_compensate,
//                                       _blocks_lum_compensate
//
// Wraps:
//   cv::detail::BlocksGainCompensator   (per-block gain)
//   cv::detail::BlocksChannelsCompensator (per-block per-channel, Phase 4)
//
// Implementation roadmap: Phase 2.
// See docs/moon/roadmaps/asp_cpp_migration.md §base::exposure
// ---------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "common.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>

namespace py = pybind11;

using namespace base;

// ---------------------------------------------------------------------------
// Helpers shared across implementations
// ---------------------------------------------------------------------------

/// Convert a list of Python ndarrays to cv::UMat via cv::Mat.
static std::vector<cv::UMat> list_to_umats_bgr(const py::list& lst)
{
    std::vector<cv::UMat> out;
    out.reserve(lst.size());
    for (auto item : lst) {
        cv::Mat m = mat_from_array(item.cast<py::array_t<uint8_t>>()).clone();
        cv::UMat u;
        m.copyTo(u);
        out.push_back(u);
    }
    return out;
}

/// Convert a list of Python mask ndarrays to cv::UMat (single-channel).
static std::vector<cv::UMat> list_to_umats_mask(const py::list& lst)
{
    std::vector<cv::UMat> out;
    out.reserve(lst.size());
    for (auto item : lst) {
        cv::Mat m = mat_from_array(item.cast<py::array_t<uint8_t>>()).clone();
        if (m.channels() > 1) cv::cvtColor(m, m, cv::COLOR_BGR2GRAY);
        cv::UMat u;
        m.copyTo(u);
        out.push_back(u);
    }
    return out;
}

/// Parse a Python list of (x,y) tuples to cv::Point.
static std::vector<cv::Point> list_to_points(const py::list& lst)
{
    std::vector<cv::Point> pts;
    pts.reserve(lst.size());
    for (auto item : lst) {
        auto t = item.cast<py::tuple>();
        pts.emplace_back(t[0].cast<int>(), t[1].cast<int>());
    }
    return pts;
}


// ---------------------------------------------------------------------------
// correct_vignetting
//
// Apply a pre-computed vignette gain map (float32, H×W) to a single
// uint8 BGR frame.  Each pixel channel is multiplied by gain_map[y][x]
// and the result is clamped to [0, 255].
//
// Args
// ----
// frame        : uint8 ndarray (H, W, 3) BGR
// vignette_map : float32 ndarray (H, W), values > 1.0 brighten
//
// Returns uint8 ndarray (H, W, 3).
// ---------------------------------------------------------------------------
static py::array_t<uint8_t> correct_vignetting(
    py::array_t<uint8_t> frame_arr,
    py::array_t<float>   vmap_arr)
{
    cv::Mat frame = mat_from_array(frame_arr).clone();
    cv::Mat vmap  = mat_from_f32(vmap_arr).clone();

    if (frame.empty() || vmap.empty())
        return frame_arr;

    // Resize gain map if sizes differ (e.g. scaled-down gain map)
    if (vmap.size() != frame.size())
        cv::resize(vmap, vmap, frame.size(), 0, 0, cv::INTER_LINEAR);

    cv::Mat frame_f32;
    frame.convertTo(frame_f32, CV_32F);

    std::vector<cv::Mat> channels(3);
    cv::split(frame_f32, channels);
    for (auto& ch : channels) {
        cv::multiply(ch, vmap, ch);
    }
    cv::Mat result_f32;
    cv::merge(channels, result_f32);

    cv::Mat result;
    result_f32.convertTo(result, CV_8UC3);
    return array_from_mat(result);
}


// ---------------------------------------------------------------------------
// blocks_gain_compensate
//
// Wraps cv::detail::BlocksGainCompensator.
// Per 32×32 block: builds overlap matrix A, solve A × gain = b (Eigen Chol),
// apply gain map via bilinear interpolation.
// nr_iterations rounds of 5×5 Gaussian smoothing on the block gain grid.
//
// Args
// ----
// warped_frames : list[ndarray uint8 (H,W,C)]
// warped_masks  : list[ndarray uint8 (H,W)]
// corners       : list[(x,y)] — top-left corner of each frame in canvas
// bl_width, bl_height : block dimensions (pixels)
// nr_feeds      : int — number of compensator feed rounds
// nr_iterations : int — Gaussian smoothing rounds on gain grid
//
// Returns list[ndarray uint8 (H,W,C)] — compensated frames.
// ---------------------------------------------------------------------------
static py::list blocks_gain_compensate(
    py::list warped_frames,
    py::list warped_masks,
    py::list corners,
    int      bl_width      = 32,
    int      bl_height     = 32,
    int      nr_feeds      = 1,
    int      nr_iterations = 2)
{
    size_t N = warped_frames.size();
    if (N == 0) return py::list();

    std::vector<cv::UMat> imgs  = list_to_umats_bgr(warped_frames);
    std::vector<cv::UMat> masks = list_to_umats_mask(warped_masks);
    std::vector<cv::Point> pts  = list_to_points(corners);

    // BlocksGainCompensator.feed() takes vector<pair<UMat, uchar>>
    // where the uchar is an overlap flag (255 = all pixels overlap)
    std::vector<std::pair<cv::UMat, uchar>> mask_pairs(N);
    for (size_t i = 0; i < N; ++i) {
        mask_pairs[i] = {masks[i], uchar(255)};
    }

    cv::detail::BlocksGainCompensator compensator(bl_width, bl_height, nr_feeds);
    compensator.setNrGainsFilteringIterations(nr_iterations);

    {
        py::gil_scoped_release release;
        compensator.feed(pts, imgs, mask_pairs);
    }

    py::list result;
    for (size_t i = 0; i < N; ++i) {
        cv::UMat img_out;
        imgs[i].copyTo(img_out);
        compensator.apply(static_cast<int>(i), pts[i], img_out, masks[i]);
        cv::Mat out = img_out.getMat(cv::ACCESS_READ).clone();
        result.append(array_from_mat(out));
    }
    return result;
}

// ---------------------------------------------------------------------------
// register_exposure — called from bindings.cpp
// ---------------------------------------------------------------------------
void register_exposure(py::module_& m) {
    m.doc() = R"doc(
        batch.exposure — Block gain compensation and vignetting correction.

        Functions
        ---------
        blocks_gain_compensate(frames, masks, corners, bl_w, bl_h, nr_feeds, nr_iters)
            -> list[ndarray]
        correct_vignetting(frame, vignette_map) -> ndarray
    )doc";

    m.def("blocks_gain_compensate", &blocks_gain_compensate,
        py::arg("warped_frames"),
        py::arg("warped_masks"),
        py::arg("corners"),
        py::arg("bl_width")      = 32,
        py::arg("bl_height")     = 32,
        py::arg("nr_feeds")      = 1,
        py::arg("nr_iterations") = 2,
        R"doc(
            Per-block gain compensation via cv::detail::BlocksGainCompensator.

            Args
            ----
            warped_frames : list[uint8 ndarray (H,W,C)]
            warped_masks  : list[uint8 ndarray (H,W)]
            corners       : list[(x,y)] top-left canvas corners
            bl_width, bl_height : block size (pixels), default 32×32
            nr_feeds      : compensator feed rounds
            nr_iterations : Gaussian smoothing rounds on gain grid

            Returns list[uint8 ndarray].
        )doc");


    m.def("correct_vignetting", &correct_vignetting,
        py::arg("frame"),
        py::arg("vignette_map"),
        R"doc(
            Multiply each pixel by vignette_map[y][x] and clip to [0,255].
            Handles mismatched sizes via bilinear resize of the gain map.

            Args
            ----
            frame        : uint8 ndarray (H, W, 3) BGR
            vignette_map : float32 ndarray (H, W), values > 1.0 brighten

            Returns uint8 ndarray (H, W, 3).
        )doc");
}

std::vector<cv::Mat> blocks_gain_compensate_impl(
    const std::vector<cv::Mat>& /*frames*/,
    const std::vector<cv::Mat>& /*masks*/,
    const std::vector<cv::Point>& /*corners*/,
    int /*bl_width*/, int /*bl_height*/,
    int /*nr_feeds*/, int /*nr_iterations*/)
{
    throw std::runtime_error("blocks_gain_compensate_impl: not implemented");
}
cv::Mat correct_vignetting_impl(
    const cv::Mat& /*frame*/,
    const cv::Mat& /*vignette_map*/)
{
    throw std::runtime_error("correct_vignetting_impl: not implemented");
}
