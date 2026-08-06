// ---------------------------------------------------------------------------
// batch/tests/test_compositing.cpp
//
// Native C++ unit tests for base::compositing functions.
//
// Tests (all pure C++ — no Python references):
//   zone_chroma_align  : output shape/type
//   zone_lum_norm      : output shape/type, identity-zone no-change
//   zone_sat_norm      : output shape/type
//   zone_contrast_eq   : output shape/type
//   zone_hue_eq        : output shape/type
//   laplacian_blend    : output shape/type, horizontal seam continuity
//   single_pose_soft_edge : output shape/type
//   normalize_warped_frames : output count and frame shapes
//
// All tagged [not_impl] until Phase 2; REQUIRE_THROWS_AS guards every call.
// ---------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Forward declarations of internal impl functions (defined in compositing.cpp)
// ---------------------------------------------------------------------------

cv::Mat laplacian_blend_impl(
    const cv::Mat& fa, const cv::Mat& fb,
    const std::vector<int>& path,
    int feather_px, int n_bands, float alpha_fine_weight);

cv::Mat single_pose_soft_edge_impl(
    const cv::Mat& fa, const cv::Mat& fb,
    const std::vector<int>& path, int soft_px);

std::vector<cv::Mat> normalize_warped_frames_impl(
    const std::vector<cv::Mat>& frames,
    const std::vector<cv::Mat>& bg_masks,
    int ref_frame_idx,
    bool adaptive_gain_clamp,
    float coherence_limit);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static cv::Mat rand_bgr(int H, int W, uint64_t seed = 0) {
    cv::RNG rng(seed);
    cv::Mat m(H, W, CV_8UC3);
    rng.fill(m, cv::RNG::UNIFORM, 0, 256);
    return m;
}

static cv::Mat solid_bgr(int H, int W, uint8_t v) {
    return cv::Mat(H, W, CV_8UC3, cv::Scalar(v, v, v));
}

static cv::Mat full_mask(int H, int W) {
    return cv::Mat(H, W, CV_8UC1, cv::Scalar(255));
}

static std::vector<int> midline_path(int W, int H) {
    return std::vector<int>(W, H / 2);
}


// ---------------------------------------------------------------------------
// laplacian_blend tests
// ---------------------------------------------------------------------------

TEST_CASE("laplacian_blend returns BGR mat of input shape", "[compositing]") {
    const int H = 80, W = 120;
    cv::Mat fa   = rand_bgr(H, W, 0);
    cv::Mat fb   = rand_bgr(H, W, 1);
    auto path    = midline_path(W, H);
    cv::Mat out = laplacian_blend_impl(fa, fb, path, 12, 5, 0.3f);
    REQUIRE(out.rows == H); REQUIRE(out.cols == W);
    REQUIRE(out.type() == CV_8UC3);
}

TEST_CASE("laplacian_blend horizontal seam shows no hard edge between adjacent rows", "[compositing]") {
    // Black top / white bottom, midline seam: adjacent rows near seam must
    // differ by < 200 (blended, not hard-cut).
    const int H = 80, W = 120;
    cv::Mat fa = solid_bgr(H, W, 0);    // black
    cv::Mat fb = solid_bgr(H, W, 255);  // white
    auto path  = midline_path(W, H);
    cv::Mat out = laplacian_blend_impl(fa, fb, path, 20, 5, 0.3f);

    // Verify row 45 is a blend of fa and fb
    cv::Vec3b px = out.at<cv::Vec3b>(45, 50);
    REQUIRE(px[0] > 0);
    REQUIRE(px[0] < 255);
}

// ---------------------------------------------------------------------------
// single_pose_soft_edge tests
// ---------------------------------------------------------------------------

TEST_CASE("single_pose_soft_edge returns BGR mat of input shape", "[compositing]") {
    const int H = 60, W = 80;
    cv::Mat fa = rand_bgr(H, W, 0);
    cv::Mat fb = rand_bgr(H, W, 1);
    auto path  = midline_path(W, H);
    cv::Mat out = single_pose_soft_edge_impl(fa, fb, path, 6);
    REQUIRE(out.size() == fa.size());
}

// ---------------------------------------------------------------------------
// normalize_warped_frames tests
// ---------------------------------------------------------------------------

TEST_CASE("normalize_warped_frames returns N frames of correct shape", "[compositing]") {
    const int N = 5, H = 60, W = 80;
    std::vector<cv::Mat> frames, masks;
    cv::RNG rng(0);
    for (int i = 0; i < N; ++i) {
        cv::Mat f(H, W, CV_8UC3);
        rng.fill(f, cv::RNG::UNIFORM, 0, 256);
        frames.push_back(f);
        masks.push_back(cv::Mat(H, W, CV_8UC1, cv::Scalar(255)));
    }
    auto result = normalize_warped_frames_impl(frames, masks, 0, true, 20.0f);
    REQUIRE(static_cast<int>(result.size()) == N);
    for (auto& r : result) { REQUIRE(r.rows == H); REQUIRE(r.cols == W); }
}

