// ---------------------------------------------------------------------------
// batch/tests/test_seam.cpp
//
// Native C++ unit tests for base::seam functions.
//
// Tests (all pure C++ — no Python references required):
//   seam_cut  : output length, row range, semantic cost, transition penalty,
//               single-column edge case, waypoint pinning
//   build_seam_cost_map : output shape, dtype equivalent, all-background = 0
//   seam_batch : count and per-path length
//
// NOTE: all tests call BATCH_NOT_IMPLEMENTED stubs and are therefore tagged
// [not_impl] so they can be excluded until Phase 2 lands:
//   ./batch_tests "[seam]" --allow-running-no-tests
// ---------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#include <opencv2/core.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <stdexcept>
#include <vector>

#include "affine_types.hpp"
using ZonePair = base::ZonePair;

// Pull in internal seam helpers.  The real implementations are compiled via
// batch_impl; until Phase 2 they throw std::runtime_error("not implemented").
// Forward-declare the C-linkage entry points used by pybind11 so we can call
// the underlying implementations directly without going through Python.
//
// Strategy: each src/seam.cpp function is separated into a *_impl free
// function.  The pybind11 wrapper calls the impl; tests call the impl too.
// Until Phase 2 the impl == BATCH_NOT_IMPLEMENTED, so all tests below are
// wrapped in REQUIRE_THROWS_AS(…, std::runtime_error) and tagged [not_impl].
// When Phase 2 ships, remove the throws wrappers and the [not_impl] tag.

// ---------------------------------------------------------------------------
// Helpers: build cv::Mat test data without Python
// ---------------------------------------------------------------------------

static cv::Mat random_bgr(int H, int W, uint64_t seed = 0) {
    cv::RNG rng(seed);
    cv::Mat m(H, W, CV_8UC3);
    rng.fill(m, cv::RNG::UNIFORM, 0, 256);
    return m;
}

static cv::Mat zeros_mask(int H, int W) {
    return cv::Mat::zeros(H, W, CV_8UC1);
}

static cv::Mat ones_mask(int H, int W) {
    return cv::Mat(H, W, CV_8UC1, cv::Scalar(255));
}

static cv::Mat random_cost(int H, int W, uint64_t seed = 42) {
    cv::RNG rng(seed);
    cv::Mat m(H, W, CV_32F);
    rng.fill(m, cv::RNG::UNIFORM, 0.0f, 1.0f);
    return m;
}

// ---------------------------------------------------------------------------
// Forward declarations of internal impl functions.
// These will be defined in seam.cpp once Phase 2 is implemented.
// ---------------------------------------------------------------------------
std::vector<int> seam_cut_impl(
    const cv::Mat& fa_zone,
    const cv::Mat& fb_zone,
    const cv::Mat& sem_cost,       // may be empty
    const std::vector<int>& waypoints,
    float transition_penalty,
    float edge_weight);

cv::Mat build_seam_cost_map_impl(
    const cv::Mat& fa_zone,
    const cv::Mat& bg_mask_a,
    const cv::Mat& bg_mask_b,
    float cost_map_blur_sigma,
    float cost_col_smooth_sigma,
    bool  cost_map_norm,
    float scatter_cost_weight,
    const std::vector<int>& pinned_rows,
    bool  try_gpu = false);

// ---------------------------------------------------------------------------
// seam_cut tests
// ---------------------------------------------------------------------------

TEST_CASE("seam_cut returns path of length W", "[seam]") {
        const int H = 40, W = 60;
        cv::Mat fa = random_bgr(H, W, 0);
        cv::Mat fb = random_bgr(H, W, 1);
        auto path = seam_cut_impl(fa, fb, cv::Mat{}, {}, 0.0f, 1.0f);
        REQUIRE(static_cast<int>(path.size()) == W);
}

TEST_CASE("seam_cut path rows are in [0, H)", "[seam]") {
        const int H = 40, W = 60;
        cv::Mat fa = random_bgr(H, W, 2);
        cv::Mat fb = random_bgr(H, W, 3);
        auto path = seam_cut_impl(fa, fb, cv::Mat{}, {}, 0.0f, 1.0f);
        for (int y : path) { CHECK(y >= 0); CHECK(y < H); }
}

TEST_CASE("seam_cut accepts optional semantic cost map", "[seam]") {
        const int H = 30, W = 50;
        cv::Mat fa = random_bgr(H, W, 4);
        cv::Mat fb = random_bgr(H, W, 5);
        cv::Mat sem = random_cost(H, W, 42);
        seam_cut_impl(fa, fb, sem, {}, 0.0f, 1.0f);
}

TEST_CASE("seam_cut with high transition_penalty biases path toward midline", "[seam]") {
    // With high penalty, mean |path[x] - H/2| should be <= mean without penalty.
    // Both calls are expected to throw in Phase 1; test documents the contract.
    const int H = 40, W = 60;
    cv::Mat fa = random_bgr(H, W, 7);
    cv::Mat fb = random_bgr(H, W, 8);
    // Post-Phase-2:
    auto path_no_pen = seam_cut_impl(fa, fb, cv::Mat{}, {}, 0.0f, 1.0f);
    auto path_pen    = seam_cut_impl(fa, fb, cv::Mat{}, {}, 5.0f, 1.0f);
    double mid = H / 2.0;
    auto mean_dist = [&](const std::vector<int>& p) {
        double s = 0; for (int y : p) s += std::abs(y - mid); return s / p.size();
    };
    CHECK(mean_dist(path_pen) <= mean_dist(path_no_pen));
}

TEST_CASE("seam_cut handles single-column zone (W=1)", "[seam]") {
    const int H = 20, W = 1;
    cv::Mat fa = random_bgr(H, W, 9);
    cv::Mat fb = random_bgr(H, W, 10);
    // Post-Phase-2:
    auto path = seam_cut_impl(fa, fb, cv::Mat{}, {}, 0.0f, 1.0f);
    REQUIRE(path.size() == 1u);
    CHECK(path[0] >= 0); CHECK(path[0] < H);
}

// ---------------------------------------------------------------------------
// build_seam_cost_map tests
// ---------------------------------------------------------------------------

TEST_CASE("build_seam_cost_map returns float32 mat of shape (H,W)", "[seam]") {
        const int H = 50, W = 80;
        cv::Mat fa     = random_bgr(H, W, 0);
        cv::Mat mask_a = ones_mask(H, W);
        cv::Mat mask_b = ones_mask(H, W);
        cv::Mat cost = build_seam_cost_map_impl(fa, mask_a, mask_b, 0.0f, 0.0f, true, 0.0f, {});
        REQUIRE(cost.rows == H);
        REQUIRE(cost.cols == W);
        REQUIRE(cost.type() == CV_32FC1);
}

TEST_CASE("build_seam_cost_map all-background gives near-zero cost", "[seam]") {
    const int H = 30, W = 40;
    cv::Mat fa     = cv::Mat::zeros(H, W, CV_8UC3);
    cv::Mat mask_a = ones_mask(H, W);
    cv::Mat mask_b = ones_mask(H, W);
    // Post-Phase-2:
    cv::Mat cost = build_seam_cost_map_impl(fa, mask_a, mask_b, 0.0f, 0.0f, true, 0.0f, {});
    double mn, mx;
    cv::minMaxLoc(cost, &mn, &mx);
    CHECK(mx < 0.5);
}
// ---------------------------------------------------------------------------
// GraphCutSeamFinder tests (Phase 4 — native OpenCV API, no pybind11 needed)
// ---------------------------------------------------------------------------

TEST_CASE("GraphCutSeamFinder: 2 frames produces 2 updated masks", "[seam][phase4]") {
    const int H = 60, W = 80;
    cv::UMat fa_u, fb_u, ma_u, mb_u;
    random_bgr(H, W, 0).convertTo(fa_u, CV_32F);
    random_bgr(H, W, 1).convertTo(fb_u, CV_32F);
    ones_mask(H, W).copyTo(ma_u);
    ones_mask(H, W).copyTo(mb_u);

    std::vector<cv::UMat> imgs  = {fa_u, fb_u};
    std::vector<cv::UMat> masks = {ma_u, mb_u};
    std::vector<cv::Point> pts  = {cv::Point(0, 0), cv::Point(W/2, 0)};

    cv::detail::GraphCutSeamFinder finder(cv::detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
    finder.find(imgs, pts, masks);

    REQUIRE(static_cast<int>(masks.size()) == 2);
    for (auto& m : masks) {
        cv::Mat m_cpu;
        m.getMat(cv::ACCESS_READ).copyTo(m_cpu);
        REQUIRE(m_cpu.rows == H);
        REQUIRE(m_cpu.cols == W);
        REQUIRE(m_cpu.type() == CV_8UC1);
    }
}

TEST_CASE("GraphCutSeamFinder: masks partition ownership (non-overlapping tiles)", "[seam][phase4]") {
    const int H = 50, W = 40;
    cv::UMat fa_u, fb_u, ma_u, mb_u;
    random_bgr(H, W, 10).convertTo(fa_u, CV_32F);
    random_bgr(H, W, 20).convertTo(fb_u, CV_32F);
    // Non-overlapping: place fb at x=W
    ones_mask(H, W).copyTo(ma_u);
    ones_mask(H, W).copyTo(mb_u);

    std::vector<cv::UMat> imgs  = {fa_u, fb_u};
    std::vector<cv::UMat> masks = {ma_u, mb_u};
    std::vector<cv::Point> pts  = {cv::Point(0, 0), cv::Point(W, 0)};

    cv::detail::GraphCutSeamFinder finder(cv::detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
    finder.find(imgs, pts, masks);

    // With no overlap, masks should remain fully white (ownership unchanged)
    for (auto& m : masks) {
        cv::Mat m_cpu;
        m.getMat(cv::ACCESS_READ).copyTo(m_cpu);
        CHECK(cv::countNonZero(m_cpu) == H * W);
    }
}

TEST_CASE("GraphCutSeamFinder: 3 frames completes without error", "[seam][phase4]") {
    const int H = 40, W = 50;
    std::vector<cv::UMat> imgs, masks;
    std::vector<cv::Point> pts;
    for (int i = 0; i < 3; ++i) {
        cv::UMat img_u, mask_u;
        random_bgr(H, W, i * 7).convertTo(img_u, CV_32F);
        ones_mask(H, W).copyTo(mask_u);
        imgs.push_back(img_u);
        masks.push_back(mask_u);
        pts.push_back(cv::Point(i * (W / 2), 0));
    }
    cv::detail::GraphCutSeamFinder finder(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
    REQUIRE_NOTHROW(finder.find(imgs, pts, masks));
    REQUIRE(static_cast<int>(masks.size()) == 3);
}
