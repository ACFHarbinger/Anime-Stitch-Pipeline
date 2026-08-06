// ---------------------------------------------------------------------------
// batch/src/compositing.cpp
//
// Zone normalization chain, Laplacian blend, gain loops, single-pose helpers.
//
// Replaces (compute bodies only — Python wrappers remain):
//   rendering/compositing.py  :: _laplacian_blend,
//     _single_pose_soft_edge, _seam_color_match, _poisson_seam_blend,
//     _smooth_gain_array, _normalize_warped_frames, _blocks_lum_compensate,
//     _blocks_gain_compensate, _blocks_lum_compensate,
//     gain normalization loops, all single-pose escalation gates
//
// Implementation roadmap: Phase 2 (zone norms wired) + Phase 4 (MultiBandBlender).
// See docs/moon/roadmaps/asp_cpp_migration.md §base::compositing
// ---------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "common.hpp"
#include "image_utils.hpp"
#include <opencv2/stitching/detail/blenders.hpp>

namespace py = pybind11;

inline cv::Mat as_mat(py::array_t<uint8_t> arr) {
    auto req = arr.request();
    int type = (req.ndim == 3 && req.shape[2] == 3) ? CV_8UC3 : CV_8UC1;
    return cv::Mat(req.shape[0], req.shape[1], type, req.ptr).clone();
}

inline py::array_t<uint8_t> as_ndarray(const cv::Mat& mat) {
    if (mat.empty()) return py::array_t<uint8_t>();
    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;
    if (mat.channels() == 3) {
        shape = {mat.rows, mat.cols, 3};
        strides = {mat.step[0], mat.step[1], 1};
    } else {
        shape = {mat.rows, mat.cols};
        strides = {mat.step[0], 1};
    }
    auto result = py::array_t<uint8_t>(shape, strides);
    auto req = result.request();
    std::memcpy(req.ptr, mat.data, mat.total() * mat.elemSize());
    return result;
}

cv::Mat laplacian_blend_impl(const cv::Mat& fa, const cv::Mat& fb, const std::vector<int>& path, int band_px, int bands, float alpha_schedule) {
    int H = fa.rows, W = fa.cols;
    cv::Mat mask_float(H, W, CV_32F, cv::Scalar(0));
    for (int x = 0; x < W; ++x) {
        int seam_y = path[x];
        for (int y = 0; y < H; ++y) {
            float dist = static_cast<float>(y - seam_y);
            float weight = 0.5f - dist / (2.0f * band_px);
            mask_float.at<float>(y, x) = std::clamp(weight, 0.0f, 1.0f);
        }
    }

    std::vector<cv::Mat> ga, gb, gm;
    cv::Mat fa_f, fb_f;
    fa.convertTo(fa_f, CV_32F);
    fb.convertTo(fb_f, CV_32F);
    ga.push_back(fa_f);
    gb.push_back(fb_f);
    
    cv::Mat m;
    cv::cvtColor(mask_float, m, cv::COLOR_GRAY2BGR);
    gm.push_back(m);

    for (int i = 0; i < bands - 1; ++i) {
        cv::Mat down_a, down_b, down_m;
        cv::pyrDown(ga[i], down_a);
        cv::pyrDown(gb[i], down_b);
        cv::pyrDown(gm[i], down_m);
        ga.push_back(down_a);
        gb.push_back(down_b);
        gm.push_back(down_m);
    }

    std::vector<cv::Mat> la, lb;
    la.push_back(ga.back());
    lb.push_back(gb.back());
    for (int i = bands - 1; i > 0; --i) {
        cv::Mat up_a, up_b;
        cv::pyrUp(ga[i], up_a, ga[i - 1].size());
        cv::pyrUp(gb[i], up_b, gb[i - 1].size());
        la.push_back(ga[i - 1] - up_a);
        lb.push_back(gb[i - 1] - up_b);
    }
    std::reverse(la.begin(), la.end());
    std::reverse(lb.begin(), lb.end());

    std::vector<cv::Mat> blended(bands);
    for (int i = 0; i < bands; ++i) {
        cv::Mat cur_m = gm[i];
        if (cur_m.size() != la[i].size()) {
            cv::resize(cur_m, cur_m, la[i].size());
        }
        cv::Mat inv_m = cv::Scalar(1.0f, 1.0f, 1.0f) - cur_m;
        blended[i] = la[i].mul(cur_m) + lb[i].mul(inv_m);
    }

    cv::Mat result = blended.back();
    for (int i = bands - 2; i >= 0; --i) {
        cv::Mat up_res;
        cv::pyrUp(result, up_res, blended[i].size());
        result = up_res + blended[i];
    }

    if (alpha_schedule > 0.0f) {
        cv::Mat sharp_mask = mask_float.mul(mask_float);
        cv::Mat m3;
        cv::cvtColor(sharp_mask, m3, cv::COLOR_GRAY2BGR);
        cv::Mat sharp_blend = fa_f.mul(m3) + fb_f.mul(cv::Scalar(1.0f, 1.0f, 1.0f) - m3);
        result = alpha_schedule * sharp_blend + (1.0f - alpha_schedule) * result;
    }

    cv::Mat out;
    result.convertTo(out, CV_8U);
    return out;
}

cv::Mat single_pose_soft_edge_impl(const cv::Mat& fa, const cv::Mat& fb, const std::vector<int>& path, int soft_px) {
    int H = fa.rows, W = fa.cols;
    cv::Mat out = fb.clone();
    for (int x = 0; x < W; ++x) {
        int seam_y = path[x];
        for (int y = 0; y < H; ++y) {
            float dist = static_cast<float>(std::abs(y - seam_y));
            float alpha = std::max(0.0f, 1.0f - dist / soft_px) * 0.5f;
            if (alpha > 0.0f) {
                const cv::Vec3b& cA = fa.at<cv::Vec3b>(y, x);
                const cv::Vec3b& cB = fb.at<cv::Vec3b>(y, x);
                out.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    std::clamp(static_cast<int>(cA[0] * alpha + cB[0] * (1.0f - alpha)), 0, 255),
                    std::clamp(static_cast<int>(cA[1] * alpha + cB[1] * (1.0f - alpha)), 0, 255),
                    std::clamp(static_cast<int>(cA[2] * alpha + cB[2] * (1.0f - alpha)), 0, 255)
                );
            }
        }
    }
    return out;
}

cv::Mat seam_color_match_impl(const cv::Mat& dom_zone, const cv::Mat& oth_zone, const std::vector<int>& path, int band_px) {
    int H = dom_zone.rows, W = dom_zone.cols;
    if (band_px <= 0) return oth_zone.clone();

    long long count = 0;
    cv::Vec3f dom_sum(0, 0, 0), oth_sum(0, 0, 0);

    for (int x = 0; x < W; ++x) {
        int seam_y = path[x];
        for (int y = std::max(0, seam_y - band_px); y <= std::min(H - 1, seam_y + band_px); ++y) {
            const cv::Vec3b& d = dom_zone.at<cv::Vec3b>(y, x);
            const cv::Vec3b& o = oth_zone.at<cv::Vec3b>(y, x);
            if (std::max({d[0], d[1], d[2]}) > 0 && std::max({o[0], o[1], o[2]}) > 0) {
                dom_sum += cv::Vec3f(d[0], d[1], d[2]);
                oth_sum += cv::Vec3f(o[0], o[1], o[2]);
                count++;
            }
        }
    }

    if (count < 10) return oth_zone.clone();

    cv::Vec3f delta = (dom_sum - oth_sum) / static_cast<float>(count);
    cv::Mat out = oth_zone.clone();
    for (int x = 0; x < W; ++x) {
        int seam_y = path[x];
        for (int y = std::max(0, seam_y - band_px); y <= std::min(H - 1, seam_y + band_px); ++y) {
            const cv::Vec3b& o = oth_zone.at<cv::Vec3b>(y, x);
            if (std::max({o[0], o[1], o[2]}) > 0) {
                out.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    std::clamp(static_cast<int>(o[0] + delta[0]), 0, 255),
                    std::clamp(static_cast<int>(o[1] + delta[1]), 0, 255),
                    std::clamp(static_cast<int>(o[2] + delta[2]), 0, 255)
                );
            }
        }
    }
    return out;
}

std::vector<cv::Mat> normalize_warped_frames_impl(const std::vector<cv::Mat>& frames, const std::vector<cv::Mat>& bgs, int radius, bool do_lum, float gain_clamp) {
    if (frames.empty()) return {};
    int N = frames.size();
    std::vector<float> lums(N, 0.0f);
    
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        if (bgs[i].empty()) {
            lums[i] = 1.0f;
            continue;
        }
        cv::Mat f_f32;
        frames[i].convertTo(f_f32, CV_32F);
        cv::Mat gray;
        cv::transform(f_f32, gray, cv::Matx13f(0.114f, 0.587f, 0.299f));
        cv::Mat mask = (bgs[i] > 127);
        if (cv::countNonZero(mask) > 0) {
            lums[i] = static_cast<float>(cv::mean(gray, mask)[0]);
        } else {
            lums[i] = static_cast<float>(cv::mean(gray)[0]);
        }
    }

    std::vector<float> smooth_lums = lums;
    if (radius > 0) {
        for (int i = 0; i < N; ++i) {
            float sum = 0;
            int count = 0;
            for (int j = std::max(0, i - radius); j <= std::min(N - 1, i + radius); ++j) {
                sum += lums[j];
                count++;
            }
            smooth_lums[i] = sum / count;
        }
    }

    std::vector<cv::Mat> result(N);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        if (!do_lum || lums[i] < 1.0f || smooth_lums[i] < 1.0f) {
            result[i] = frames[i].clone();
            continue;
        }
        float gain = std::clamp(smooth_lums[i] / lums[i], 1.0f / gain_clamp, gain_clamp);
        cv::Mat f_f32;
        frames[i].convertTo(f_f32, CV_32F);
        f_f32 *= gain;
        cv::Mat out;
        f_f32.convertTo(out, CV_8UC3);
        result[i] = out;
    }
    return result;
}
// ---------------------------------------------------------------------------
// laplacian_blend
//
// Multi-band Laplacian pyramid blend guided by a DP seam path.
// Builds a per-pixel soft weight mask from the path (linear ramp ±feather_px).
// Blends n_bands Laplacian pyramid levels, then reconstructs.
//
// Optionally wraps cv::detail::MultiBandBlender (Phase 4: ASP_MULTIBAND_BLEND=1).
//
// Returns uint8 ndarray (H, W, 3) BGR.
// ---------------------------------------------------------------------------
static py::array_t<uint8_t> laplacian_blend(
    py::array_t<uint8_t>  fa_zone,
    py::array_t<uint8_t>  fb_zone,
    py::array_t<int32_t>  path,
    int                   feather_px          = 12,
    int                   n_bands             = 5,
    float                 alpha_fine_weight   = 0.3f)
{
    cv::Mat fa = as_mat(fa_zone);
    cv::Mat fb = as_mat(fb_zone);
    const int* p_ptr = reinterpret_cast<const int*>(path.request().ptr);
    std::vector<int> p_vec(p_ptr, p_ptr + path.size());
    cv::Mat out = laplacian_blend_impl(fa, fb, p_vec, feather_px, n_bands, alpha_fine_weight);
    return as_ndarray(out);
}

// ---------------------------------------------------------------------------
// single_pose_soft_edge
//
// Linear ramp ±soft_px around the DP seam path.
// alpha[y][x] = max(0, 1 - |y - path[x]| / soft_px) × 0.5
// Applied in OpenMP parallel over columns.
//
// Returns blended uint8 ndarray (H, W, 3).
// ---------------------------------------------------------------------------
static py::array_t<uint8_t> single_pose_soft_edge(
    py::array_t<uint8_t> fa_zone,
    py::array_t<uint8_t> fb_zone,
    py::array_t<int32_t> path,
    int                  soft_px = 6)
{
    cv::Mat fa = as_mat(fa_zone);
    cv::Mat fb = as_mat(fb_zone);
    const int* p_ptr = reinterpret_cast<const int*>(path.request().ptr);
    std::vector<int> p_vec(p_ptr, p_ptr + path.size());
    cv::Mat out = single_pose_soft_edge_impl(fa, fb, p_vec, soft_px);
    return as_ndarray(out);
}

// ---------------------------------------------------------------------------
// seam_color_match
//
// Per-channel mean shift in the blend band between dominant and other zone.
//
// Returns uint8 ndarray (H, W, 3) — the corrected other-zone.
// ---------------------------------------------------------------------------
static py::array_t<uint8_t> seam_color_match(
    py::array_t<uint8_t> dom_zone,
    py::array_t<uint8_t> oth_zone,
    py::array_t<int32_t> path,
    int                  band_half_px = 8)
{
    cv::Mat dom = as_mat(dom_zone);
    cv::Mat oth = as_mat(oth_zone);
    const int* p_ptr = reinterpret_cast<const int*>(path.request().ptr);
    std::vector<int> p_vec(p_ptr, p_ptr + path.size());
    cv::Mat out = seam_color_match_impl(dom, oth, p_vec, band_half_px);
    return as_ndarray(out);
}

// ---------------------------------------------------------------------------
// normalize_warped_frames
//
// Apply per-frame scalar gain corrections with Gaussian smoothing across
// the sequence. Coherence gate (§1.18) skips frames where gain delta > limit.
//
// Returns list of uint8 ndarrays.
// ---------------------------------------------------------------------------
static py::list normalize_warped_frames(
    py::list warped_frames,
    py::list bg_masks,
    int      ref_frame_idx,
    bool     adaptive_gain_clamp = true,
    float    coherence_limit     = 20.0f)
{
    std::vector<cv::Mat> w_frames;
    for (auto item : warped_frames) {
        w_frames.push_back(as_mat(item.cast<py::array_t<uint8_t>>()));
    }
    std::vector<cv::Mat> b_masks;
    for (auto item : bg_masks) {
        if (!item.is_none()) {
            b_masks.push_back(as_mat(item.cast<py::array_t<uint8_t>>()));
        } else {
            b_masks.push_back(cv::Mat());
        }
    }
    std::vector<cv::Mat> out = normalize_warped_frames_impl(w_frames, b_masks, ref_frame_idx, adaptive_gain_clamp, coherence_limit);
    py::list ret;
    for (const auto& mat : out) {
        ret.append(as_ndarray(mat));
    }
    return ret;
}


// ---------------------------------------------------------------------------
// register_compositing — called from bindings.cpp
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// blocks_gain_compensate_pair  (§4.1)
//
// Per-block per-channel BGR gain compensation between two zones.
// Divides zones into block_size×block_size blocks, computes mean(fa)/mean(fb)
// per channel per block, bilinear-resizes the gain grid, clamps to [0.5, 2.0],
// and applies to fb_zone.  Blocks where fb-channel mean < 1.0 use gain=1.0.
//
// Returns uint8 (H, W, 3) BGR.
// ---------------------------------------------------------------------------
static py::array_t<uint8_t> blocks_gain_compensate_pair(
    py::array_t<uint8_t> fa_arr,
    py::array_t<uint8_t> fb_arr,
    int block_size = 32)
{
    cv::Mat fa = as_mat(fa_arr), fb = as_mat(fb_arr);
    if (fa.empty() || fb.empty()) return fb_arr;
    int H = fb.rows, W = fb.cols;
    int bs = std::max(1, block_size);
    int n_rows = std::max(1, (H + bs - 1) / bs);
    int n_cols = std::max(1, (W + bs - 1) / bs);

    cv::Mat gain_grid(n_rows, n_cols, CV_32FC3, cv::Scalar(1.f, 1.f, 1.f));

    for (int ri = 0; ri < n_rows; ++ri) {
        int r0 = ri * bs, r1 = std::min(r0 + bs, H);
        for (int ci = 0; ci < n_cols; ++ci) {
            int c0 = ci * bs, c1 = std::min(c0 + bs, W);
            cv::Rect rect(c0, r0, c1 - c0, r1 - r0);
            cv::Mat fa_b_f, fb_b_f;
            fa(rect).convertTo(fa_b_f, CV_32F);
            fb(rect).convertTo(fb_b_f, CV_32F);
            cv::Scalar m_fa = cv::mean(fa_b_f);
            cv::Scalar m_fb = cv::mean(fb_b_f);
            float g0 = (m_fb[0] >= 1.f) ? float(m_fa[0] / m_fb[0]) : 1.f;
            float g1 = (m_fb[1] >= 1.f) ? float(m_fa[1] / m_fb[1]) : 1.f;
            float g2 = (m_fb[2] >= 1.f) ? float(m_fa[2] / m_fb[2]) : 1.f;
            gain_grid.at<cv::Vec3f>(ri, ci) = cv::Vec3f(g0, g1, g2);
        }
    }

    cv::Mat gain_map;
    cv::resize(gain_grid, gain_map, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);

    std::vector<cv::Mat> gain_chans(3);
    cv::split(gain_map, gain_chans);
    for (auto& gc : gain_chans) {
        cv::threshold(gc, gc, 2.0, 2.0, cv::THRESH_TRUNC);
        gc.setTo(0.5f, gc < 0.5f);
    }
    cv::merge(gain_chans, gain_map);

    cv::Mat fb_f32;
    fb.convertTo(fb_f32, CV_32F);
    cv::Mat result_f32;
    cv::multiply(fb_f32, gain_map, result_f32);
    cv::Mat result;
    result_f32.convertTo(result, CV_8UC3);
    return as_ndarray(result);
}


// ---------------------------------------------------------------------------
// blocks_lum_compensate_pair  (§4.4)
//
// LAB L-channel blocks gain compensation.  Computes scalar L gain per block
// (mean(fa_L) / max(1.0, mean(fb_L))), bilinear-resizes, clamps to [0.5, 2.0],
// and applies uniformly to all BGR channels.  Avoids colour cast from near-zero
// channels that BGR gain can produce.
//
// Returns uint8 (H, W, 3) BGR.
// ---------------------------------------------------------------------------
static py::array_t<uint8_t> blocks_lum_compensate_pair(
    py::array_t<uint8_t> fa_arr,
    py::array_t<uint8_t> fb_arr,
    int block_size = 32)
{
    cv::Mat fa = as_mat(fa_arr), fb = as_mat(fb_arr);
    if (fa.empty() || fb.empty()) return fb_arr;
    int H = fb.rows, W = fb.cols;

    cv::Mat fa_lab, fb_lab;
    cv::cvtColor(fa, fa_lab, cv::COLOR_BGR2Lab);
    cv::cvtColor(fb, fb_lab, cv::COLOR_BGR2Lab);
    cv::Mat fa_lab_f, fb_lab_f;
    fa_lab.convertTo(fa_lab_f, CV_32F);
    fb_lab.convertTo(fb_lab_f, CV_32F);

    int bs = std::max(1, block_size);
    int n_rows = std::max(1, (H + bs - 1) / bs);
    int n_cols = std::max(1, (W + bs - 1) / bs);

    cv::Mat gain_grid(n_rows, n_cols, CV_32F, cv::Scalar(1.f));

    for (int ri = 0; ri < n_rows; ++ri) {
        int r0 = ri * bs, r1 = std::min(r0 + bs, H);
        for (int ci = 0; ci < n_cols; ++ci) {
            int c0 = ci * bs, c1 = std::min(c0 + bs, W);
            cv::Rect rect(c0, r0, c1 - c0, r1 - r0);
            cv::Mat fa_l, fb_l;
            cv::extractChannel(fa_lab_f(rect), fa_l, 0);
            cv::extractChannel(fb_lab_f(rect), fb_l, 0);
            double m_fa_l = cv::mean(fa_l)[0];
            double m_fb_l = cv::mean(fb_l)[0];
            gain_grid.at<float>(ri, ci) = float(m_fa_l / std::max(1.0, m_fb_l));
        }
    }

    cv::Mat gain_map;
    cv::resize(gain_grid, gain_map, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
    cv::threshold(gain_map, gain_map, 2.0, 2.0, cv::THRESH_TRUNC);
    gain_map.setTo(0.5f, gain_map < 0.5f);

    cv::Mat fb_f32;
    fb.convertTo(fb_f32, CV_32F);
    std::vector<cv::Mat> fb_chans(3);
    cv::split(fb_f32, fb_chans);
    for (auto& ch : fb_chans) {
        cv::multiply(ch, gain_map, ch);
    }
    cv::Mat result_f32;
    cv::merge(fb_chans, result_f32);
    cv::Mat result;
    result_f32.convertTo(result, CV_8UC3);
    return as_ndarray(result);
}


// ---------------------------------------------------------------------------
// find_optimal_boundaries  (§S17 adaptive range + §S34 bg-pixel scoring)
//
// For each adjacent frame pair in `order`, scans ±search_range rows from
// initial_boundaries[k] to find the y-position that minimises the mean
// absolute per-pixel BGR difference in a search_slab-row band.
//
// When bg_masks (List[Optional[ndarray]]) and affines (List[ndarray 2×3])
// are provided, the bg masks are warped into canvas space and a 40% bg /
// 60% all-pixel blended score is used (background is unaffected by pose).
//
// Returns (boundaries_array, diffs_array) — both float64, shape (N-1,).
// Equivalent to Python _find_optimal_boundaries in compositing.py.
// ---------------------------------------------------------------------------
static py::tuple find_optimal_boundaries(
    const py::list&                                                  warped_frames,    // N_total×(H,W,3) uint8, by frame-idx
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> order_arr,        // (N_ordered,) frame indices
    py::array_t<double,  py::array::c_style | py::array::forcecast> init_bounds,      // (N_ordered-1,) initial boundary y
    int H, int W,
    int search_range = 250,
    int search_slab  = 20,
    py::object bg_masks_obj = py::none(),           // None or List[Optional[ndarray]]
    py::object affines_obj  = py::none()            // None or List[ndarray 2×3 float32]
) {
    int N        = (int)order_arr.size();
    int n_bounds = (int)init_bounds.size();

    const int64_t* p_order = order_arr.data();
    const double* p_init   = init_bounds.data();

    std::vector<int64_t> order(N);
    for (int i = 0; i < N; ++i) {
        order[i] = p_order[i];
    }

    std::vector<double> init(n_bounds);
    for (int i = 0; i < n_bounds; ++i) {
        init[i] = p_init[i];
    }

    // Load frames by frame index
    int n_total = (int)warped_frames.size();
    std::vector<cv::Mat> frames(n_total);
    for (int i = 0; i < n_total; ++i)
        frames[i] = as_mat(warped_frames[i].cast<py::array_t<uint8_t>>());

    // Adaptive effective range: if tx spread < 5px use narrow ±100px window
    int effective_range = search_range;
    bool has_affines = !affines_obj.is_none();
    py::list affines_list;
    if (has_affines) {
        affines_list = affines_obj.cast<py::list>();
        if ((int)affines_list.size() >= N) {
            float tx_min = std::numeric_limits<float>::max();
            float tx_max = std::numeric_limits<float>::lowest();
            for (int i = 0; i < N; ++i) {
                int fi = (int)order[i];
                if (fi < (int)affines_list.size()) {
                    auto aff = affines_list[fi].cast<py::array_t<float>>().unchecked<2>();
                    float tx = aff(0, 2);
                    tx_min = std::min(tx_min, tx);
                    tx_max = std::max(tx_max, tx);
                }
            }
            if (tx_max - tx_min < 5.0f) effective_range = 100;
        }
    }

    // Warp bg masks into canvas space (same as Python cv2.warpAffine call)
    bool has_bg = !bg_masks_obj.is_none() && has_affines;
    std::vector<cv::Mat> warped_bgs;
    if (has_bg) {
        py::list bg_list = bg_masks_obj.cast<py::list>();
        int max_fi = 0;
        for (int i = 0; i < N; ++i) max_fi = std::max(max_fi, (int)order[i]);
        warped_bgs.resize(max_fi + 1);
        for (int i = 0; i < N; ++i) {
            int fi = (int)order[i];
            if (fi >= (int)bg_list.size() || fi >= (int)affines_list.size()) continue;
            py::object bg_obj = bg_list[fi];
            if (bg_obj.is_none()) continue;
            cv::Mat bg_mat = as_mat(bg_obj.cast<py::array_t<uint8_t>>());
            if (bg_mat.empty()) continue;
            auto aff = affines_list[fi].cast<py::array_t<float>>().unchecked<2>();
            cv::Mat M = (cv::Mat_<double>(2, 3) <<
                (double)aff(0,0), (double)aff(0,1), (double)aff(0,2),
                (double)aff(1,0), (double)aff(1,1), (double)aff(1,2));
            cv::warpAffine(bg_mat, warped_bgs[fi], M, cv::Size(W, H),
                           cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
        }
    }

    // Main boundary search — released GIL (pure C++ pixel loops)
    std::vector<double> optimised(n_bounds);
    std::vector<double> result_diffs(n_bounds, std::numeric_limits<double>::infinity());
    {
        py::gil_scoped_release release;
        double prev_optimised = 0.0;
        for (int k = 0; k < n_bounds; ++k) {
            int fi_a = (int)order[k];
            int fi_b = (int)order[k + 1];
            double by = init[k];

            int lo_limit = (k > 0)
                ? (int)prev_optimised + 2*search_slab + 1
                : search_slab;
            int hi_limit = (k < n_bounds - 1)
                ? (int)init[k+1] - 2*search_slab - 1
                : H - 2*search_slab;

            int y_lo = std::max(lo_limit, (int)by - effective_range);
            int y_hi = std::min(hi_limit, (int)by + effective_range);

            int    best_y     = (int)by;
            double best_score = std::numeric_limits<double>::infinity();
            double best_diff  = std::numeric_limits<double>::infinity();

            if (fi_a >= n_total || fi_b >= n_total) {
                optimised[k]    = by;
                prev_optimised  = by;
                result_diffs[k] = 0.0;
                continue;
            }
            const cv::Mat& fa = frames[fi_a];
            const cv::Mat& fb = frames[fi_b];
            bool has_bg_ab = has_bg
                && fi_a < (int)warped_bgs.size() && !warped_bgs[fi_a].empty()
                && fi_b < (int)warped_bgs.size() && !warped_bgs[fi_b].empty();
            const cv::Mat* bga_ptr = has_bg_ab ? &warped_bgs[fi_a] : nullptr;
            const cv::Mat* bgb_ptr = has_bg_ab ? &warped_bgs[fi_b] : nullptr;

            for (int y_cand = y_lo; y_cand < std::min(y_hi, H - search_slab); ++y_cand) {
                int valid_count = 0, bg_count = 0;
                double sum_all = 0.0, sum_bg = 0.0;
                for (int y = y_cand; y < y_cand + search_slab && y < H; ++y) {
                    const uchar* pa  = fa.ptr<uchar>(y);
                    const uchar* pb  = fb.ptr<uchar>(y);
                    const uchar* bga = (bga_ptr && y < bga_ptr->rows) ? bga_ptr->ptr<uchar>(y) : nullptr;
                    const uchar* bgb = (bgb_ptr && y < bgb_ptr->rows) ? bgb_ptr->ptr<uchar>(y) : nullptr;
                    for (int x = 0; x < W; ++x) {
                        int a0=pa[x*3], a1=pa[x*3+1], a2=pa[x*3+2];
                        int b0=pb[x*3], b1=pb[x*3+1], b2=pb[x*3+2];
                        if (!((a0|a1|a2) && (b0|b1|b2))) continue;
                        double d = (std::abs(a0-b0) + std::abs(a1-b1) + std::abs(a2-b2)) / 3.0;
                        sum_all += d;
                        ++valid_count;
                        if (bga && bgb && bga[x] > 0 && bgb[x] > 0) {
                            sum_bg += d;
                            ++bg_count;
                        }
                    }
                }
                if (valid_count < 50) continue;
                double all_d = sum_all / valid_count;
                double score, cand_diff;
                if (bg_count >= 50) {
                    double bg_d = sum_bg / bg_count;
                    score = 0.4*bg_d + 0.6*all_d;
                    cand_diff = bg_d;
                } else {
                    score = all_d;
                    cand_diff = all_d;
                }
                if (score < best_score) {
                    best_score = score;
                    best_diff  = cand_diff;
                    best_y = y_cand + search_slab / 2;
                }
            }

            // Final measurement at best_y ±half for feather metric
            int half = search_slab / 2;
            int y0_f = std::max(0, best_y - half);
            int y1_f = std::min(H - 1, best_y + half);
            int valid_f = 0; double sum_f = 0.0;
            for (int y = y0_f; y < y1_f; ++y) {
                const uchar* pa = fa.ptr<uchar>(y);
                const uchar* pb = fb.ptr<uchar>(y);
                for (int x = 0; x < W; ++x) {
                    int a0=pa[x*3], a1=pa[x*3+1], a2=pa[x*3+2];
                    int b0=pb[x*3], b1=pb[x*3+1], b2=pb[x*3+2];
                    if (!((a0|a1|a2) && (b0|b1|b2))) continue;
                    sum_f += (std::abs(a0-b0)+std::abs(a1-b1)+std::abs(a2-b2)) / 3.0;
                    ++valid_f;
                }
            }
            double total_diff = (valid_f >= 10) ? sum_f / valid_f : best_diff;
            double feather_metric = (best_diff < 20.0 && total_diff < 20.0)
                                    ? best_diff : total_diff;

            optimised[k]    = (double)best_y;
            prev_optimised  = (double)best_y;
            result_diffs[k] = feather_metric;
        }
    }

    py::array_t<double> out_bounds(n_bounds);
    py::array_t<double> out_diffs(n_bounds);
    auto ob = out_bounds.mutable_unchecked<1>();
    auto od = out_diffs.mutable_unchecked<1>();
    for (int i = 0; i < n_bounds; ++i) { ob(i) = optimised[i]; od(i) = result_diffs[i]; }
    return py::make_tuple(out_bounds, out_diffs);
}


void register_compositing(py::module_& m) {
    m.doc() = R"doc(
        batch.compositing — Zone normalization chain, blending, gain loops.

        Functions
        ---------
        laplacian_blend(fa, fb, path, feather_px, n_bands, alpha_fine_weight) -> ndarray
        single_pose_soft_edge(fa, fb, path, soft_px) -> ndarray
        seam_color_match(dom_zone, oth_zone, path, band_half_px) -> ndarray
        normalize_warped_frames(frames, masks, ref, adaptive, coherence_limit) -> list[ndarray]
        blocks_gain_compensate_pair(fa, fb, block_size) -> ndarray
        blocks_lum_compensate_pair(fa, fb, block_size) -> ndarray
        find_optimal_boundaries(warped_frames, order, init_bounds, H, W, ...) -> (bounds, diffs)
    )doc";






    m.def("laplacian_blend", &laplacian_blend,
        py::arg("fa_zone"), py::arg("fb_zone"),
        py::arg("path"),
        py::arg("feather_px")        = 12,
        py::arg("n_bands")           = 5,
        py::arg("alpha_fine_weight") = 0.3f,
        "Multi-band Laplacian pyramid blend guided by DP seam path.");

    m.def("single_pose_soft_edge", &single_pose_soft_edge,
        py::arg("fa_zone"), py::arg("fb_zone"),
        py::arg("path"),
        py::arg("soft_px") = 6,
        "Linear ramp ±soft_px around DP seam for single-pose escalation.");

    m.def("seam_color_match", &seam_color_match,
        py::arg("dom_zone"), py::arg("oth_zone"),
        py::arg("path"),
        py::arg("band_half_px") = 8,
        "Per-channel mean shift in blend band to reduce seam color discontinuity.");

    m.def("normalize_warped_frames", &normalize_warped_frames,
        py::arg("warped_frames"),
        py::arg("bg_masks"),
        py::arg("ref_frame_idx"),
        py::arg("adaptive_gain_clamp") = true,
        py::arg("coherence_limit")     = 20.0f,
        "Apply per-frame scalar gains with Gaussian smooth + §1.18 coherence gate.");

    m.def("blocks_gain_compensate_pair", &blocks_gain_compensate_pair,
        py::arg("fa_zone"), py::arg("fb_zone"),
        py::arg("block_size") = 32,
        "§4.1 Per-block BGR gain compensation: mean(fa)/mean(fb) per block, bilinear gain map, clamp [0.5,2.0].");

    m.def("blocks_lum_compensate_pair", &blocks_lum_compensate_pair,
        py::arg("fa_zone"), py::arg("fb_zone"),
        py::arg("block_size") = 32,
        "§4.4 LAB L-channel blocks gain: scalar L ratio per block, avoids BGR channel-cast, clamp [0.5,2.0].");

#ifndef BATCH_TESTS
#endif

    m.def("find_optimal_boundaries", &find_optimal_boundaries,
        py::arg("warped_frames"),
        py::arg("order"),
        py::arg("init_bounds"),
        py::arg("H"),
        py::arg("W"),
        py::arg("search_range")  = 250,
        py::arg("search_slab")   = 20,
        py::arg("bg_masks")      = py::none(),
        py::arg("affines")       = py::none(),
        R"doc(
            Scan ±search_range rows per boundary and return the y-position that
            minimises mean-abs BGR diff in a search_slab-row band.

            warped_frames : list of (H,W,3) uint8 arrays indexed by frame index
            order         : int64 array of N frame indices (compositing order)
            init_bounds   : float64 array of N-1 initial boundary y positions
            bg_masks      : optional list of per-frame uint8 background masks
            affines       : optional list of 2×3 float32 affines (for bg warp)

            Returns (boundaries, diffs) both float64 shape (N-1,).
            GIL released during pixel-loop computation.
        )doc");
}
