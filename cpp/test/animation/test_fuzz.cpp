// ---------------------------------------------------------------------------
// batch/tests/test_fuzz.cpp
//
// Fuzz tests for batch C++ functions: verify no crash/abort on random
// or boundary inputs.  These tests call the impl functions directly and
// assert only that either:
//   (a) the function returns without throwing, OR
//   (b) it throws std::runtime_error (not_impl stub)
//
// They do NOT assert on output values — just crash-safety.
//
// Tags: [fuzz][not_impl]
//
// When Phase N ships: remove REQUIRE_THROWS_AS guards and let the tests
// become plain crash-safety assertions.
// ---------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>

#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>
#include <random>
#include <functional>

// ---------------------------------------------------------------------------
// Forward declarations (one per impl function being fuzzed)
// ---------------------------------------------------------------------------
struct PhaseResult { float dx, dy, response; };
struct Edge        { int i, j; float dx, dy, weight; };

PhaseResult phase_correlate_masked_impl(
    const cv::Mat&, const cv::Mat&, const cv::Mat&, const cv::Mat&);

std::vector<int> seam_cut_impl(
    const cv::Mat&, const cv::Mat&, const cv::Mat&,
    const std::vector<int>&, float, float);


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Call fn; succeed if it returns normally OR throws std::runtime_error.
// Any other exception (segfault causes SIGABRT, abort, etc.) is a bug.
template<typename Fn>
static void require_no_unexpected_throw(Fn&& fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        // Expected: stub not yet implemented
    }
    // If another exception escapes, Catch2 will report it as FAILED
}

static cv::Mat rand_gray(int H, int W, uint64_t seed) {
    cv::RNG rng(seed);
    cv::Mat m(H, W, CV_8UC1);
    rng.fill(m, cv::RNG::UNIFORM, 0, 256);
    return m;
}

static cv::Mat rand_bgr(int H, int W, uint64_t seed) {
    cv::RNG rng(seed);
    cv::Mat m(H, W, CV_8UC3);
    rng.fill(m, cv::RNG::UNIFORM, 0, 256);
    return m;
}

static cv::Mat rand_f32(int H, int W, uint64_t seed) {
    cv::RNG rng(seed);
    cv::Mat m(H, W, CV_32F);
    rng.fill(m, cv::RNG::UNIFORM, 0.0f, 1.0f);
    return m;
}

// ---------------------------------------------------------------------------
// Fuzz: phase_correlate_masked — 20 random sizes
// ---------------------------------------------------------------------------

TEST_CASE("FUZZ phase_correlate_masked: no crash on random (H, W)", "[fuzz][matching]") {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> size_dist(16, 256);
    for (int trial = 0; trial < 20; ++trial) {
        int H = size_dist(rng), W = size_dist(rng);
        cv::Mat a = rand_gray(H, W, trial);
        cv::Mat b = rand_gray(H, W, trial + 100);
        require_no_unexpected_throw([&] {
            phase_correlate_masked_impl(a, b, cv::Mat{}, cv::Mat{});
        });
    }
}

// ---------------------------------------------------------------------------
// Fuzz: seam_cut — 20 random (H, W) dimensions
// ---------------------------------------------------------------------------

TEST_CASE("FUZZ seam_cut: no crash on random (H, W)", "[fuzz][seam]") {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> size_dist(5, 100);
    for (int trial = 0; trial < 20; ++trial) {
        int H = size_dist(rng), W = size_dist(rng);
        cv::Mat fa = rand_bgr(H, W, trial);
        cv::Mat fb = rand_bgr(H, W, trial + 200);
        require_no_unexpected_throw([&] {
            seam_cut_impl(fa, fb, cv::Mat{}, {}, 0.0f, 1.0f);
        });
    }
}

TEST_CASE("FUZZ seam_cut: no crash with optional semantic cost map", "[fuzz][seam]") {
    std::mt19937 rng(13);
    std::uniform_int_distribution<int> size_dist(5, 60);
    for (int trial = 0; trial < 5; ++trial) {
        int H = size_dist(rng), W = size_dist(rng);
        cv::Mat fa  = rand_bgr(H, W, trial);
        cv::Mat fb  = rand_bgr(H, W, trial + 50);
        cv::Mat sem = rand_f32(H, W, trial + 300);
        require_no_unexpected_throw([&] {
            seam_cut_impl(fa, fb, sem, {}, 0.0f, 1.0f);
        });
    }
}

TEST_CASE("FUZZ seam_cut: no crash with high transition penalty", "[fuzz][seam]") {
    std::mt19937 rng(17);
    std::uniform_int_distribution<int> size_dist(5, 80);
    for (int trial = 0; trial < 5; ++trial) {
        int H = size_dist(rng), W = size_dist(rng);
        cv::Mat fa = rand_bgr(H, W, trial);
        cv::Mat fb = rand_bgr(H, W, trial + 400);
        require_no_unexpected_throw([&] {
            seam_cut_impl(fa, fb, cv::Mat{}, {}, 10.0f, 1.0f);
        });
    }
}

