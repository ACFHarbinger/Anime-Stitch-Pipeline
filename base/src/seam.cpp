// ---------------------------------------------------------------------------
// batch/src/seam.cpp
//
// Seam finding: DP (_seam_cut), cost map builder, GraphCut seam finder,
// parallel seam batch.
//
// Replaces (hot path — ~40% of total pipeline time):
//   rendering/compositing.py  :: _seam_cut, _build_seam_cost_map,
//                                _find_optimal_boundaries
//
// New algorithms (Phase 4):
//   cv::detail::GraphCutSeamFinder — global multi-image seam
//
// Implementation roadmap: Phase 2 (seam DP + cost map) + Phase 4 (GraphCut).
// See docs/moon/roadmaps/asp_cpp_migration.md §base::seam
// ---------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "common.hpp"
#include "affine_types.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>

namespace py = pybind11;

inline cv::Mat as_mat(py::array_t<uint8_t> arr) {
    auto req = arr.request();
    int type = (req.ndim == 3 && req.shape[2] == 3) ? CV_8UC3 : CV_8UC1;
    return cv::Mat(req.shape[0], req.shape[1], type, req.ptr).clone();
}
std::vector<int> seam_cut_impl(const cv::Mat& fa_zone, const cv::Mat& fb_zone, const cv::Mat& sem_cost, const std::vector<int>& waypoints, float transition_penalty, float edge_weight) {
    int H = fa_zone.rows, W = fa_zone.cols;
    cv::Mat diff;
    cv::absdiff(fa_zone, fb_zone, diff);
    cv::Mat luma_diff(H, W, CV_32F);
    cv::Mat diff_f32;
    diff.convertTo(diff_f32, CV_32F);
    cv::transform(diff_f32, luma_diff, cv::Matx13f(0.114f, 0.587f, 0.299f));
    
    cv::Mat grad_diff;
    cv::Sobel(luma_diff, grad_diff, CV_32F, 1, 0, 3);
    cv::Mat grad_diff_y;
    cv::Sobel(luma_diff, grad_diff_y, CV_32F, 0, 1, 3);
    grad_diff = cv::abs(grad_diff) + cv::abs(grad_diff_y);
    
    cv::Mat luma_a, luma_b;
    cv::Mat fa_f32, fb_f32;
    fa_zone.convertTo(fa_f32, CV_32F);
    fb_zone.convertTo(fb_f32, CV_32F);
    cv::transform(fa_f32, luma_a, cv::Matx13f(0.114f, 0.587f, 0.299f));
    cv::transform(fb_f32, luma_b, cv::Matx13f(0.114f, 0.587f, 0.299f));

    cv::Mat g1x, g1y, g2x, g2y;
    cv::Sobel(luma_a, g1x, CV_32F, 1, 0, 3);
    cv::Sobel(luma_a, g1y, CV_32F, 0, 1, 3);
    cv::Sobel(luma_b, g2x, CV_32F, 1, 0, 3);
    cv::Sobel(luma_b, g2y, CV_32F, 0, 1, 3);
    cv::Mat g1 = cv::abs(g1x) + cv::abs(g1y);
    cv::Mat g2 = cv::abs(g2x) + cv::abs(g2y);
    
    cv::Mat energy = luma_diff + 0.5f * grad_diff + edge_weight * (g1 + g2);
    if (!sem_cost.empty()) energy += sem_cost;
    
    if (transition_penalty > 0.0f) {
        int mid_row = H / 2;
        for (int y = 0; y < H; y++) {
            float dist = std::abs(y - mid_row) / float(std::max(mid_row, 1));
            float* erow = energy.ptr<float>(y);
            for (int x = 0; x < W; x++) erow[x] += dist * transition_penalty;
        }
    }

    if (!waypoints.empty() && waypoints.size() == static_cast<size_t>(W)) {
        for (int x = 0; x < W; x++) {
            int y_pin = waypoints[x];
            if (y_pin >= 0 && y_pin < H) {
                for (int y = 0; y < H; y++) {
                    if (y != y_pin) energy.at<float>(y, x) += 1e6f;
                }
            }
        }
    }
    
    cv::Mat dp(H, W, CV_32F, cv::Scalar(std::numeric_limits<float>::infinity()));
    energy.col(0).copyTo(dp.col(0));
    for (int x = 1; x < W; x++) {
        for (int y = 0; y < H; y++) {
            float prev_min = dp.at<float>(y, x-1);
            if (y > 0) prev_min = std::min(prev_min, dp.at<float>(y-1, x-1));
            if (y < H-1) prev_min = std::min(prev_min, dp.at<float>(y+1, x-1));
            dp.at<float>(y, x) = energy.at<float>(y, x) + prev_min;
        }
    }
    
    // §2.11A: build waypoint hard-force map (column → pinned row) for traceback
    auto wp_pinned = [&](int x) -> int {
        if (static_cast<int>(waypoints.size()) == W && waypoints[x] >= 0 && waypoints[x] < H)
            return waypoints[x];
        return -1;
    };

    std::vector<int> path(W);
    {
        int wp = wp_pinned(W - 1);
        path[W-1] = wp >= 0 ? wp : 0;
        if (wp < 0) {
            for (int y = 0; y < H; y++)
                if (dp.at<float>(y, W-1) < dp.at<float>(path[W-1], W-1))
                    path[W-1] = y;
        }
    }
    for (int x = W-2; x >= 0; x--) {
        int wp = wp_pinned(x);
        if (wp >= 0) {
            path[x] = wp;  // §2.11A: hard-force traceback through waypoint
            continue;
        }
        int prev_y = path[x+1];
        int best_y = prev_y;
        float best_v = dp.at<float>(prev_y, x);
        for (int dy : {-1, 0, 1}) {
            int ny = prev_y + dy;
            if (ny >= 0 && ny < H && dp.at<float>(ny, x) < best_v) {
                best_v = dp.at<float>(ny, x); best_y = ny;
            }
        }
        path[x] = best_y;
    }
    return path;
}

cv::Mat build_seam_cost_map_impl(
    const cv::Mat& fa_zone,
    const cv::Mat& bg_mask_a,
    const cv::Mat& bg_mask_b,
    float cost_map_blur_sigma,
    float cost_col_smooth_sigma,
    bool cost_map_norm,
    float scatter_cost_weight,
    const std::vector<int>& pinned_rows,
    bool try_gpu = false,
    const std::vector<cv::Mat>& exclusion_masks = {})
{
    int H = fa_zone.rows, W = fa_zone.cols;
    cv::Mat cost(H, W, CV_32F, cv::Scalar(0.0f));

    int dilate_px = 15;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * dilate_px + 1, 2 * dilate_px + 1));
    cv::Mat edge_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

    auto process_mask = [&](const cv::Mat& bm) {
        if (bm.empty()) return;
        cv::Mat fg = (bm < 127);
        cv::Mat tier1;
        fg.convertTo(tier1, CV_32F, 1.0/255.0);
        cost = cv::max(cost, tier1);

        cv::Mat edge;
        cv::morphologyEx(fg, edge, cv::MORPH_GRADIENT, edge_kernel);
        cv::Mat dilated;
        cv::dilate(edge, dilated, kernel);
        cv::Mat tier2;
        dilated.convertTo(tier2, CV_32F, 0.5/255.0);
        cost = cv::max(cost, tier2);
    };

    process_mask(bg_mask_a);
    process_mask(bg_mask_b);

    float FG_MAJORITY_FLOOR = 1.5f;
    float zone_fg_frac = static_cast<float>(cv::mean(cost >= 1.0f)[0] / 255.0);
    if (zone_fg_frac > 0.60f) {
        cv::Mat col_fg_frac;
        cv::reduce(cost >= 1.0f, col_fg_frac, 0, cv::REDUCE_AVG, CV_32F);
        col_fg_frac /= 255.0f;
        
        bool any_heavy = false, all_heavy = true;
        for (int x = 0; x < W; x++) {
            if (col_fg_frac.at<float>(0, x) > 0.8001f) any_heavy = true;
            else all_heavy = false;
        }
        if (any_heavy && !all_heavy) {
            for (int x = 0; x < W; x++) {
                if (col_fg_frac.at<float>(0, x) > 0.8001f) {
                    for (int y = 0; y < H; y++) {
                        cost.at<float>(y, x) = std::max(cost.at<float>(y, x), FG_MAJORITY_FLOOR);
                    }
                }
            }
        }
    }

    float barrier_cost = 2.0f;
    cv::Mat fg_col_frac;
    cv::reduce(cost >= 1.0f, fg_col_frac, 0, cv::REDUCE_AVG, CV_32F);
    fg_col_frac /= 255.0f;
    
    bool any_dominated = false, all_dominated = true;
    for (int x = 0; x < W; x++) {
        if (fg_col_frac.at<float>(0, x) > 0.50f) any_dominated = true;
        else all_dominated = false;
    }
    if (any_dominated && !all_dominated) {
        for (int x = 0; x < W; x++) {
            if (fg_col_frac.at<float>(0, x) > 0.50f) {
                for (int y = 0; y < H; y++) {
                    cost.at<float>(y, x) = std::max(cost.at<float>(y, x), barrier_cost);
                }
            }
        }
    }

    if (cost_map_norm) {
        cv::Mat soft_mask = (cost < 1e5f);
        double max_val;
        cv::minMaxLoc(cost, nullptr, &max_val, nullptr, nullptr, soft_mask);
        if (max_val > 1e-6) {
            for (int y = 0; y < H; ++y) {
                float* cptr = cost.ptr<float>(y);
                const uint8_t* mptr = soft_mask.ptr<uint8_t>(y);
                for (int x = 0; x < W; ++x) {
                    if (mptr[x]) cptr[x] /= static_cast<float>(max_val);
                }
            }
        }
    }

    if (cost_map_blur_sigma > 0.0f) {
        cv::Mat soft_mask = (cost < 1e5f);
        cv::Mat soft_only;
        cost.copyTo(soft_only, soft_mask);
        cv::Mat blurred;
        if (try_gpu) {
            cv::UMat src_u, dst_u;
            soft_only.copyTo(src_u);
            cv::GaussianBlur(src_u, dst_u, cv::Size(0, 0), cost_map_blur_sigma, cost_map_blur_sigma);
            dst_u.copyTo(blurred);
        } else {
            cv::GaussianBlur(soft_only, blurred, cv::Size(0, 0), cost_map_blur_sigma, cost_map_blur_sigma);
        }
        for (int y = 0; y < H; ++y) {
            float* cptr = cost.ptr<float>(y);
            const float* bptr = blurred.ptr<float>(y);
            const uint8_t* mptr = soft_mask.ptr<uint8_t>(y);
            for (int x = 0; x < W; ++x) {
                if (mptr[x]) cptr[x] = bptr[x];
            }
        }
    }

    if (cost_col_smooth_sigma > 0.0f) {
        cv::Mat soft_mask = (cost < 1e5f);
        cv::Mat soft_only;
        cost.copyTo(soft_only, soft_mask);
        cv::Mat smoothed;
        if (try_gpu) {
            cv::UMat src_u, dst_u;
            soft_only.copyTo(src_u);
            cv::GaussianBlur(src_u, dst_u, cv::Size(1, 0), 0.0, cost_col_smooth_sigma);
            dst_u.copyTo(smoothed);
        } else {
            cv::GaussianBlur(soft_only, smoothed, cv::Size(1, 0), 0.0, cost_col_smooth_sigma);
        }
        for (int y = 0; y < H; ++y) {
            float* cptr = cost.ptr<float>(y);
            const float* sptr = smoothed.ptr<float>(y);
            const uint8_t* mptr = soft_mask.ptr<uint8_t>(y);
            for (int x = 0; x < W; ++x) {
                if (mptr[x]) cptr[x] = sptr[x];
            }
        }
    }

    if (scatter_cost_weight > 0.0f && !fa_zone.empty()) {
        cv::Mat gray_sc;
        cv::cvtColor(fa_zone, gray_sc, cv::COLOR_BGR2GRAY);
        gray_sc.convertTo(gray_sc, CV_32F);
        cv::Mat mean_sc, mean_sq_sc;
        cv::boxFilter(gray_sc, mean_sc, CV_32F, cv::Size(3, 3));
        cv::boxFilter(gray_sc.mul(gray_sc), mean_sq_sc, CV_32F, cv::Size(3, 3));
        cv::Mat var_sc = cv::max(0.0f, mean_sq_sc - mean_sc.mul(mean_sc));
        
        cv::Mat soft_mask = (cost < 1e5f);
        double var_max;
        cv::minMaxLoc(var_sc, nullptr, &var_max, nullptr, nullptr, soft_mask);
        if (var_max > 1e-6) {
            cv::Mat scatter = (var_sc / static_cast<float>(var_max)) * scatter_cost_weight;
            for (int y = 0; y < H; ++y) {
                float* cptr = cost.ptr<float>(y);
                const float* scptr = scatter.ptr<float>(y);
                const uint8_t* mptr = soft_mask.ptr<uint8_t>(y);
                for (int x = 0; x < W; ++x) {
                    if (mptr[x]) cptr[x] += scptr[x];
                }
            }
        }
    }

    for (int y : pinned_rows) {
        if (y >= 0 && y < H) {
            for (int x = 0; x < W; x++) cost.at<float>(y, x) = 1e6f;
        }
    }

    for (const auto& em : exclusion_masks) {
        if (em.empty()) continue;
        cv::Mat em_resized = em;
        if (em.rows != H || em.cols != W) {
            cv::resize(em, em_resized, cv::Size(W, H), 0, 0, cv::INTER_NEAREST);
        }
        for (int y = 0; y < H; ++y) {
            const uint8_t* mptr = em_resized.ptr<uint8_t>(y);
            float* cptr = cost.ptr<float>(y);
            for (int x = 0; x < W; ++x) {
                if (mptr[x] > 127) {
                    cptr[x] = 1e9f;
                }
            }
        }
    }

    return cost;
}

// ---------------------------------------------------------------------------
// seam_cut
//
// Forward-DP seam finder operating column-wise on two overlapping zone crops.
//
// Energy per pixel:
//   E[y][x] = diff + 0.5*|∇diff| + edge_weight*(|∇img1|+|∇img2|) + sem_cost
//   plus optional §1.125 midline transition penalty.
//
// DP forward pass:
//   dp[y][x] = E[y][x] + min(dp[y-1][x-1], dp[y][x-1], dp[y+1][x-1])
//
// Traceback: column-wise argmin.
//
// Returns list[int] of length W (seam row per column).
// ---------------------------------------------------------------------------
static py::array_t<int32_t> seam_cut(
    py::array_t<uint8_t> fa_zone,
    py::array_t<uint8_t> fb_zone,
    py::object           sem_cost           = py::none(),  // float32 (H,W) or None
    py::object           waypoints          = py::none(),  // list[int] or None
    float                transition_penalty = 0.0f,
    float                edge_weight        = 15.0f)
{
    cv::Mat fa = as_mat(fa_zone);
    cv::Mat fb = as_mat(fb_zone);
    cv::Mat cost;
    if (!sem_cost.is_none()) {
        auto arr = sem_cost.cast<py::array_t<float>>();
        auto req = arr.request();
        cost = cv::Mat(req.shape[0], req.shape[1], CV_32F, req.ptr).clone();
    }
    std::vector<int> wp;
    if (!waypoints.is_none()) {
        auto w_list = waypoints.cast<py::list>();
        for (auto item : w_list) wp.push_back(item.cast<int>());
    }
    std::vector<int> result = seam_cut_impl(fa, fb, cost, wp, transition_penalty, edge_weight);
    std::vector<ssize_t> shape   = {static_cast<ssize_t>(result.size())};
    std::vector<ssize_t> strides = {static_cast<ssize_t>(sizeof(int32_t))};
    py::array_t<int32_t> out(shape, strides);
    std::copy(result.begin(), result.end(), out.mutable_data());
    return out;
}

// ---------------------------------------------------------------------------
// build_seam_cost_map
//
// Six-tier cost map from foreground masks:
//   Tier 0:   background         = 0.0
//   Tier 0.3: outer fg ring      = 0.3  (§3.20 EXTRA_FG_DILATION)
//   Tier 0.5: edge buffer        = 0.5  (§3.Tier-2 buffer)
//   Tier 1.0: fg interior        = 1.0
//   Tier 1.5: fg-heavy columns   = 1.5  (§1.126 FG_MAJORITY_FLOOR)
//   Tier 2.0: dominated columns  = 2.0  (§3.15A column barrier)
//   Tier 1e6: pinned rows        (hard barrier)
//   Tier 1e9: character mask     (maximal barrier)
//
// Additional modifiers:
//   §1.110 COST_MAP_BLUR_SIGMA  : cv::GaussianBlur on soft cost
//   §1.113 COST_COL_SMOOTH_SIGMA: 1D Gaussian on per-column mean
//   §1.109 COST_MAP_NORM        : renormalize barriers after blur
//   §1.123 SCATTER_COST         : local 3×3 variance via cv::boxFilter
//
// Returns float32 ndarray (H, W).
// ---------------------------------------------------------------------------
static py::array_t<float> build_seam_cost_map(
    py::array_t<uint8_t> fa_zone,
    py::array_t<uint8_t> bg_mask_a,
    py::array_t<uint8_t> bg_mask_b,
    float                cost_map_blur_sigma   = 0.0f,
    float                cost_col_smooth_sigma = 0.0f,
    bool                 cost_map_norm         = true,
    float                scatter_cost_weight   = 0.0f,
    py::object           pinned_rows           = py::none(),
    bool                 try_gpu               = false,
    py::object           exclusion_masks       = py::none())
{
    cv::Mat fa = as_mat(fa_zone);
    cv::Mat bg_a, bg_b;
    if (!bg_mask_a.is_none()) bg_a = as_mat(bg_mask_a.cast<py::array_t<uint8_t>>());
    if (!bg_mask_b.is_none()) bg_b = as_mat(bg_mask_b.cast<py::array_t<uint8_t>>());
    std::vector<int> pinned;
    if (!pinned_rows.is_none()) {
        auto w_list = pinned_rows.cast<py::list>();
        for (auto item : w_list) pinned.push_back(item.cast<int>());
    }
    std::vector<cv::Mat> ex_mats;
    if (!exclusion_masks.is_none()) {
        if (py::isinstance<py::list>(exclusion_masks)) {
            auto m_list = exclusion_masks.cast<py::list>();
            for (auto item : m_list) {
                if (!item.is_none()) ex_mats.push_back(as_mat(item.cast<py::array_t<uint8_t>>()));
            }
        } else if (py::isinstance<py::array>(exclusion_masks)) {
            ex_mats.push_back(as_mat(exclusion_masks.cast<py::array_t<uint8_t>>()));
        }
    }
    cv::Mat cost = build_seam_cost_map_impl(fa, bg_a, bg_b, cost_map_blur_sigma, cost_col_smooth_sigma, cost_map_norm, scatter_cost_weight, pinned, try_gpu, ex_mats);
    
    std::vector<ssize_t> shape = {cost.rows, cost.cols};
    std::vector<ssize_t> strides = {static_cast<ssize_t>(cost.step[0]), 4};
    auto result = py::array_t<float>(shape, strides);
    auto req = result.request();
    std::memcpy(req.ptr, cost.data, cost.total() * cost.elemSize());
    return result;
}

// ---------------------------------------------------------------------------
// graphcut_seam_find  (Phase 4)
//
// Wraps cv::detail::GraphCutSeamFinder("COST_COLOR_GRAD").
// Global multi-image seam optimisation — eliminates pairwise DP conflicts.
//
// Args
// ----
// warped_frames : list[ndarray uint8 (H,W,C)] — N frames in canvas space
// warped_masks  : list[ndarray uint8 (H,W)]   — N binary masks
// corners       : list[(x,y)]                 — top-left corner per frame
//
// Returns list[ndarray uint8 (H,W)] — N updated ownership masks
// ---------------------------------------------------------------------------
static py::list graphcut_seam_find(
    py::list warped_frames,
    py::list warped_masks,
    py::list corners)
{
    size_t N = warped_frames.size();
    if (N == 0) throw std::runtime_error("graphcut_seam_find: empty frame list");
    if (warped_masks.size() != N || corners.size() != N)
        throw std::runtime_error("graphcut_seam_find: mismatched input lengths");

    std::vector<cv::UMat> imgs(N), masks(N);
    std::vector<cv::Point> pts(N);

    for (size_t i = 0; i < N; ++i) {
        cv::Mat frame = as_mat(warped_frames[i].cast<py::array_t<uint8_t>>());
        cv::Mat mask  = as_mat(warped_masks[i].cast<py::array_t<uint8_t>>());

        // GraphCutSeamFinder expects float32 BGR
        cv::Mat frame_f;
        frame.convertTo(frame_f, CV_32F);
        frame_f.copyTo(imgs[i]);

        // Mask: 255 where valid
        cv::Mat mask8 = (mask > 0);
        mask8.copyTo(masks[i]);

        auto pt_tuple = corners[i].cast<py::tuple>();
        pts[i] = cv::Point(pt_tuple[0].cast<int>(), pt_tuple[1].cast<int>());
    }

    {
        py::gil_scoped_release release;
        cv::detail::GraphCutSeamFinder finder(cv::detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
        finder.find(imgs, pts, masks);
    }

    py::list result;
    for (size_t i = 0; i < N; ++i) {
        cv::Mat mask_out;
        masks[i].getMat(cv::ACCESS_READ).copyTo(mask_out);
        // Return as uint8 ndarray (H, W)
        std::vector<ssize_t> shape   = {mask_out.rows, mask_out.cols};
        std::vector<ssize_t> strides = {static_cast<ssize_t>(mask_out.step[0]), 1};
        auto arr = py::array_t<uint8_t>(shape, strides);
        auto req = arr.request();
        std::memcpy(req.ptr, mask_out.data, mask_out.total());
        result.append(arr);
    }
    return result;
}
// ---------------------------------------------------------------------------
// register_seam — called from bindings.cpp
// ---------------------------------------------------------------------------
void register_seam(py::module_& m) {
    m.doc() = R"doc(
        batch.seam — Seam DP, cost map, GraphCut seam.

        Functions
        ---------
        seam_cut(fa_zone, fb_zone, sem_cost, waypoints, transition_penalty, edge_weight)
            -> ndarray[int32, shape=(W,)]
        build_seam_cost_map(fa_zone, bg_mask_a, bg_mask_b, ...) -> ndarray[float32]
        graphcut_seam_find(warped_frames, warped_masks, corners) -> list[ndarray]
    )doc";

    m.def("seam_cut", &seam_cut,
        py::arg("fa_zone"),
        py::arg("fb_zone"),
        py::arg("sem_cost")           = py::none(),
        py::arg("waypoints")          = py::none(),
        py::arg("transition_penalty") = 0.0f,
        py::arg("edge_weight")        = 15.0f,
        R"doc(
            Column-wise DP seam cutter.

            Energy E[y][x] = |diff| + 0.5*|∇diff| + edge_w*(|∇img1|+|∇img2|)
                           + sem_cost + transition_penalty * dist_from_midline.
            DP: dp[y][x] = E[y][x] + min(dp[y±1][x-1], dp[y][x-1]).

            Args
            ----
            fa_zone, fb_zone      : uint8 (H, W, 3) BGR zone crops
            sem_cost              : float32 (H, W) or None
            waypoints             : list[int] y-pin rows or None
            transition_penalty    : float ≥ 0 — §1.125 midline prior weight
            edge_weight           : float — image gradient weight

            Returns
            -------
            int32 ndarray of shape (W,) — seam row index per column
        )doc");

    m.def("build_seam_cost_map", &build_seam_cost_map,
        py::arg("fa_zone"),
        py::arg("bg_mask_a"),
        py::arg("bg_mask_b"),
        py::arg("cost_map_blur_sigma")   = 0.0f,
        py::arg("cost_col_smooth_sigma") = 0.0f,
        py::arg("cost_map_norm")         = true,
        py::arg("scatter_cost_weight")   = 0.0f,
        py::arg("pinned_rows")           = py::none(),
        py::arg("try_gpu")               = false,
        py::arg("exclusion_masks")       = py::none(),
        R"doc(
            Build a seven-tier seam cost map from foreground masks and character barriers.

            Tiers: 0.0 (bg), 0.3 (outer ring), 0.5 (buffer), 1.0 (fg),
                   1.5 (fg-heavy columns), 2.0 (dominated columns), 1e6 (hard barrier),
                   1e9 (character cel mask barrier).

            try_gpu: use cv::UMat for GaussianBlur/boxFilter (OpenCL path).
            exclusion_masks: list of uint8 masks (255 = exclude) carrying 1e9 cost barrier.
            Returns float32 ndarray (H, W).
        )doc");

    m.def("graphcut_seam_find", &graphcut_seam_find,
        py::arg("warped_frames"),
        py::arg("warped_masks"),
        py::arg("corners"),
        R"doc(
            Global multi-image seam via cv::detail::GraphCutSeamFinder.

            Input: N frames + N masks + N (x,y) corners in canvas space.
            Output: N updated ownership masks (255 = owned by this frame).

            Gate: ASP_GRAPHCUT_SEAM=1 in Python wrapper (Phase 4).
        )doc");

}
