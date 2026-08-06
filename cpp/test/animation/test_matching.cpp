// ---------------------------------------------------------------------------
// batch/tests/test_matching.cpp
//
// Native C++ unit tests for base::matching functions.
//
// Tests:
//   phase_correlate_masked : zero-shift for identical frames, known-shift
//                            detection (within 2px), response in [0,1]
//   reject_static_edges    : removes small-disp edges, keeps large-disp
//   compute_adaptive_min_disp : positive, proportional to edge size
// ---------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>
#include <cmath>

#include "affine_types.hpp"
using Edge = base::Edge;

// ---------------------------------------------------------------------------
// Forward declarations of impl functions (matching.cpp)
// ---------------------------------------------------------------------------
struct PhaseResult { float dx, dy, response; };

PhaseResult phase_correlate_masked_impl(
    const cv::Mat& frame_a_gray,
    const cv::Mat& frame_b_gray,
    const cv::Mat& mask_a,
    const cv::Mat& mask_b);

std::vector<Edge> reject_static_edges_impl(
    const std::vector<Edge>& edges,
    float min_disp_px);

float compute_adaptive_min_disp_impl(const std::vector<Edge>& edges);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static cv::Mat rand_gray(int H, int W, uint64_t seed = 0) {
    cv::RNG rng(seed);
    cv::Mat m(H, W, CV_8UC1);
    rng.fill(m, cv::RNG::UNIFORM, 0, 256);
    return m;
}

static cv::Mat bright_block_gray(int H, int W, int y0, int x0, int bh, int bw) {
    cv::Mat m = cv::Mat::zeros(H, W, CV_8UC1);
    cv::Rect roi(x0, y0, bw, bh);
    m(roi).setTo(cv::Scalar(200));
    return m;
}

// ---------------------------------------------------------------------------
// phase_correlate_masked tests
// ---------------------------------------------------------------------------

TEST_CASE("phase_correlate_masked: response is in [0, 1]", "[matching]") {
    cv::Mat a = rand_gray(64, 64, 2);
    cv::Mat b = rand_gray(64, 64, 3);
    auto r = phase_correlate_masked_impl(a, b, cv::Mat{}, cv::Mat{});
    CHECK(r.response >= 0.0f);
    CHECK(r.response <= 1.0f);
}

TEST_CASE("phase_correlate_masked: identical frames give |dx|<1 and |dy|<1", "[matching]") {
    cv::Mat a = rand_gray(64, 64, 1);
    auto r = phase_correlate_masked_impl(a, a.clone(), cv::Mat{}, cv::Mat{});
    CHECK(std::abs(r.dx) < 1.0f);
    CHECK(std::abs(r.dy) < 1.0f);
}

TEST_CASE("phase_correlate_masked: bright block shifted by (10,10) detected within 2px", "[matching]") {
    const int H = 128, W = 128;
    cv::Mat a = bright_block_gray(H, W, 20, 20, 60, 60);
    cv::Mat b = bright_block_gray(H, W, 30, 30, 60, 60);  // shifted by (+10, +10) in y,x
    auto r = phase_correlate_masked_impl(a, b, cv::Mat{}, cv::Mat{});
    // cv::phaseCorrelate returns (dx, dy) = shift of b relative to a
    // a has block at y=20..80, b at y=30..90 → b is shifted down by 10 in y
    CHECK(std::abs(r.dy - 10.0f) < 2.0f);
}

TEST_CASE("phase_correlate_masked: mismatched sizes throw", "[matching]") {
    cv::Mat a = rand_gray(64, 64, 0);
    cv::Mat b = rand_gray(128, 64, 1);
    REQUIRE_THROWS_AS(
        phase_correlate_masked_impl(a, b, cv::Mat{}, cv::Mat{}),
        std::invalid_argument);
}

// ---------------------------------------------------------------------------
// reject_static_edges tests
// ---------------------------------------------------------------------------

TEST_CASE("reject_static_edges: removes edges where both |dx|<min and |dy|<min", "[matching]") {
    std::vector<Edge> edges = {
        {0, 1,   0.5f,   0.2f, 0.8f},   // |dx|=0.5 < 50, |dy|=0.2 < 50 → remove
        {1, 2, 100.0f, 100.0f, 0.9f},   // |dx|=100 >= 50 → keep
        {2, 3,  -0.1f,   0.3f, 0.7f},   // both < 50 → remove
    };
    auto filtered = reject_static_edges_impl(edges, 50.0f);
    REQUIRE(filtered.size() == 1u);
    CHECK(filtered[0].src == 1);
}

TEST_CASE("reject_static_edges: edge with large |dx| only is kept", "[matching]") {
    std::vector<Edge> edges = {
        {0, 1, 200.0f,   0.0f, 0.9f},   // |dx|=200 >= 50 → keep
        {1, 2,   0.0f, 200.0f, 0.9f},   // |dy|=200 >= 50 → keep
    };
    auto filtered = reject_static_edges_impl(edges, 50.0f);
    CHECK(filtered.size() == 2u);
}

TEST_CASE("reject_static_edges: empty input returns empty output", "[matching]") {
    std::vector<Edge> edges;
    auto filtered = reject_static_edges_impl(edges, 50.0f);
    CHECK(filtered.empty());
}

TEST_CASE("reject_static_edges: negative dx/dy use abs", "[matching]") {
    std::vector<Edge> edges = {
        {0, 1, -200.0f, -200.0f, 0.9f},  // |dx|=200 → keep
        {1, 2,   -5.0f,   -5.0f, 0.8f},  // both < 50 → remove
    };
    auto filtered = reject_static_edges_impl(edges, 50.0f);
    REQUIRE(filtered.size() == 1u);
    CHECK(filtered[0].src == 0);
}

// ---------------------------------------------------------------------------
// compute_adaptive_min_disp tests
// ---------------------------------------------------------------------------

TEST_CASE("compute_adaptive_min_disp: returns positive value", "[matching]") {
    std::vector<Edge> edges = {
        {0, 1, 80.0f, 90.0f, 0.9f},
        {1, 2, 70.0f, 100.0f, 0.85f},
    };
    float t = compute_adaptive_min_disp_impl(edges);
    CHECK(t > 0.0f);
}

TEST_CASE("compute_adaptive_min_disp: no adjacent edges returns floor", "[matching]") {
    // Skip-frame edges only (dst != src+1)
    std::vector<Edge> edges = {{0, 2, 150.0f, 0.0f, 0.8f}};
    float t = compute_adaptive_min_disp_impl(edges);
    CHECK(t == Catch::Approx(50.0f));  // STATIC_EDGE_MIN_DISP_PX
}

TEST_CASE("compute_adaptive_min_disp: larger step gives larger threshold", "[matching]") {
    // Large adjacent steps → adaptive threshold exceeds floor
    std::vector<Edge> large_edges = {{0, 1, 1000.0f, 0.0f, 0.9f},
                                     {1, 2, 1000.0f, 0.0f, 0.9f}};
    std::vector<Edge> small_edges = {{0, 1,   50.0f, 0.0f, 0.9f},
                                     {1, 2,   50.0f, 0.0f, 0.9f}};
    float t_large = compute_adaptive_min_disp_impl(large_edges);
    float t_small = compute_adaptive_min_disp_impl(small_edges);
    CHECK(t_large > t_small);
}

TEST_CASE("compute_adaptive_min_disp: uses dominant axis (max of median dx vs dy)", "[matching]") {
    // dy is larger → dominant axis → threshold uses dy median
    std::vector<Edge> edges = {
        {0, 1, 100.0f, 800.0f, 0.9f},
        {1, 2, 100.0f, 800.0f, 0.9f},
    };
    float t = compute_adaptive_min_disp_impl(edges);
    // ADAPTIVE_MIN_DISP_FRAC * 800 = 80 > STATIC_EDGE_MIN_DISP_PX=50
    CHECK(t == Catch::Approx(80.0f).epsilon(0.01));
}
