// ---------------------------------------------------------------------------
// batch/tests/test_bundle_adjust.cpp
//
// Native C++ unit tests for base::bundle_adjust functions.
//
// Tests (no Python references):
//   bundle_adjust_affine      : output count, frame-0 anchor, ty monotone
//   spanning_tree_inlier_filter : returns subset, clean chain survives
//   compute_adaptive_f_scale  : positive float, floor respected
//
// Tagged [not_impl]; REQUIRE_THROWS_AS until Phase 3.
// ---------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <stdexcept>
#include <vector>
#include <random>
#include <cmath>

// ---------------------------------------------------------------------------
// POD types matching the internal C++ representation.
// (In the real implementation these come from affine_types.hpp)
// ---------------------------------------------------------------------------
#include "affine_types.hpp"
using AffineResult = base::AffineParams;
using Edge = base::Edge;

// ---------------------------------------------------------------------------
// Forward declarations of internal impl functions (bundle_adjust.cpp)
// ---------------------------------------------------------------------------
std::vector<AffineResult> bundle_adjust_affine_impl(
    std::vector<Edge> edges,
    int   N,
    float f_scale,
    bool  use_gnc,
    bool  adaptive_f_scale);

std::vector<Edge> spanning_tree_inlier_filter_impl(
    const std::vector<Edge>& edges,
    int   N,
    float inlier_threshold);

float compute_adaptive_f_scale_impl(
    const std::vector<Edge>& edges,
    const std::vector<AffineResult>& affines,
    float floor_scale);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<Edge> make_chain(int N, float step_px = 100.0f, float noise_sigma = 1.0f, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.0f, noise_sigma);
    std::vector<Edge> edges;
    for (int i = 0; i < N - 1; ++i)
        edges.push_back({i, i + 1, noise(rng), step_px + noise(rng), 0.95f});
    return edges;
}

// ---------------------------------------------------------------------------
// bundle_adjust_affine tests
// ---------------------------------------------------------------------------

TEST_CASE("bundle_adjust_affine: output count equals N", "[bundle_adjust][not_impl]") {
    const int N = 8;
    auto edges = make_chain(N);
    auto affines = bundle_adjust_affine_impl(edges, N, 10.0f, true, true);
    REQUIRE(static_cast<int>(affines.size()) == N);
}

TEST_CASE("bundle_adjust_affine: frame 0 is anchored at (tx=0, ty=0)", "[bundle_adjust][not_impl]") {
    const int N = 6;
    auto edges = make_chain(N, 80.0f);
    auto affines = bundle_adjust_affine_impl(edges, N, 10.0f, false, false);
    CHECK(std::abs(affines[0].tx) < 1e-3f);
    CHECK(std::abs(affines[0].ty) < 1e-3f);
}

TEST_CASE("bundle_adjust_affine: ty sequence is monotone for clean vertical chain", "[bundle_adjust][not_impl]") {
    const int N = 8;
    auto edges = make_chain(N, 100.0f, 0.5f, 0);  // tight noise
    auto affines = bundle_adjust_affine_impl(edges, N, 10.0f, false, false);
    for (int i = 0; i < N - 1; ++i)
        CHECK(affines[i+1].ty < affines[i].ty);
}

TEST_CASE("bundle_adjust_affine: GNC flag does not break output count", "[bundle_adjust][not_impl]") {
    const int N = 6;
    auto edges = make_chain(N);
    auto affines = bundle_adjust_affine_impl(edges, N, 10.0f, true, false);
    REQUIRE(static_cast<int>(affines.size()) == N);
}

// ---------------------------------------------------------------------------
// spanning_tree_inlier_filter tests
// ---------------------------------------------------------------------------

TEST_CASE("spanning_tree_inlier_filter: output is a subset of input edges", "[bundle_adjust][not_impl]") {
    const int N = 8;
    auto edges = make_chain(N);
    auto filtered = spanning_tree_inlier_filter_impl(edges, N, 50.0f);
    CHECK(filtered.size() <= edges.size());
}

TEST_CASE("spanning_tree_inlier_filter: clean chain with tight noise survives", "[bundle_adjust][not_impl]") {
    const int N = 8;
    // noise ~1px << threshold 50px → all edges should survive
    auto edges = make_chain(N, 100.0f, 1.0f, 0);
    auto filtered = spanning_tree_inlier_filter_impl(edges, N, 50.0f);
    CHECK(filtered.size() == edges.size());
}

TEST_CASE("spanning_tree_inlier_filter: large-residual outlier is removed", "[bundle_adjust][not_impl]") {
    const int N = 6;
    auto edges = make_chain(N, 80.0f, 1.0f, 7);
    // Inject a skip edge with wrong displacement
    edges.push_back({0, 5, 0.0f, 9999.0f, 0.9f});
    auto filtered = spanning_tree_inlier_filter_impl(edges, N, 50.0f);
    bool found_outlier = false;
    for (auto& e : filtered) if (e.src == 0 && e.dst == 5) found_outlier = true;
    CHECK(!found_outlier);
}

// ---------------------------------------------------------------------------
// compute_adaptive_f_scale tests
// ---------------------------------------------------------------------------

TEST_CASE("compute_adaptive_f_scale: returns positive value", "[bundle_adjust][not_impl]") {
    const int N = 6;
    auto edges = make_chain(N);
    std::vector<AffineResult> affines;
    for (int i = 0; i < N; ++i)
        affines.push_back({0.0f, static_cast<float>(i * 100), 1.0f, 0.0f, i});
    float scale = compute_adaptive_f_scale_impl(edges, affines, 5.0f);
    CHECK(scale > 0.0f);
}

TEST_CASE("compute_adaptive_f_scale: returned value is at least floor_scale", "[bundle_adjust][not_impl]") {
    const int N = 4;
    // Tiny noise → small residuals → scale should be clamped to floor
    auto edges = make_chain(N, 1.0f, 0.01f, 0);
    std::vector<AffineResult> affines;
    for (int i = 0; i < N; ++i)
        affines.push_back({0.0f, static_cast<float>(i), 1.0f, 0.0f, i});
    const float floor = 7.5f;
    float scale = compute_adaptive_f_scale_impl(edges, affines, floor);
    CHECK(scale >= floor - 1e-6f);
}
