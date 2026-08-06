// ---------------------------------------------------------------------------
// batch/tests/test_validation.cpp
//
// Native C++ unit tests for base::validation functions.
//
// Tests:
//   validate_affines_impl       : ok/fail for ratio, min_gap, rotation,
//                                 scale, and Kendall-τ monotonicity checks
//   compute_adaptive_min_gap_impl : formula max(20.0, span/(N×3))
//   compute_adaptive_rot_scale_impl : tight vs loose thresholds
// ---------------------------------------------------------------------------

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <string>
#include <vector>
#include <cmath>

// ---------------------------------------------------------------------------
// Forward declarations of impl functions (validation.cpp)
// ---------------------------------------------------------------------------
struct ValidationResult {
    bool        valid;
    float       ratio, min_gap, max_rotation, max_scale_dev;
    std::string reason;
};

ValidationResult validate_affines_impl(
    const std::vector<float>& txs,
    const std::vector<float>& tys,
    const std::vector<float>& rots,
    const std::vector<float>& scales,
    float min_step    = 25.0f,
    float max_ratio   = 3.0f,
    float max_rotation   = 0.10f,
    float max_scale_dev  = 0.10f);

float compute_adaptive_min_gap_impl(
    const std::vector<float>& txs,
    const std::vector<float>& tys);

std::pair<float, float> compute_adaptive_rot_scale_impl(
    const std::vector<float>& rots,
    const std::vector<float>& scales);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a clean vertical-scroll sequence: N frames, uniform step `step`.
// tx=0 for all (pure vertical), zero rotation and scale deviation.
static auto make_vertical_sequence(int N, float step, float tx_all = 0.0f) {
    std::vector<float> txs(N, tx_all);
    std::vector<float> tys(N);
    std::vector<float> rots(N, 0.0f);
    std::vector<float> scs(N, 0.0f);
    for (int i = 0; i < N; ++i) tys[i] = i * step;
    return std::make_tuple(txs, tys, rots, scs);
}

// ---------------------------------------------------------------------------
// validate_affines_impl — basic pass
// ---------------------------------------------------------------------------

TEST_CASE("validate_affines: clean vertical sequence passes", "[validation]") {
    auto [txs, tys, rots, scs] = make_vertical_sequence(8, 100.0f);
    auto res = validate_affines_impl(txs, tys, rots, scs, 25.0f, 3.0f);
    CHECK(res.valid);
    CHECK(res.reason == "ok");
}

TEST_CASE("validate_affines: single frame always passes", "[validation]") {
    std::vector<float> txs{0.0f}, tys{0.0f}, rots{0.0f}, scs{0.0f};
    auto res = validate_affines_impl(txs, tys, rots, scs);
    CHECK(res.valid);
}

// ---------------------------------------------------------------------------
// validate_affines_impl — ratio failure
// ---------------------------------------------------------------------------

TEST_CASE("validate_affines: high gap ratio fails", "[validation]") {
    // Frame 2 is far ahead → creates a gap 10× larger than the rest
    std::vector<float> txs = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> tys = {0.0f, 100.0f, 1100.0f, 1200.0f};  // gap[1]=1000 vs gap[0]=100
    std::vector<float> rots(4, 0.0f), scs(4, 0.0f);
    auto res = validate_affines_impl(txs, tys, rots, scs, 25.0f, 3.0f);
    CHECK_FALSE(res.valid);
    // reason should mention "ratio="
    CHECK(res.reason.find("ratio=") != std::string::npos);
}

// ---------------------------------------------------------------------------
// validate_affines_impl — min_gap failure
// ---------------------------------------------------------------------------

TEST_CASE("validate_affines: min_gap below floor fails", "[validation]") {
    // Frames very close together
    std::vector<float> txs = {0.0f, 0.0f, 0.0f};
    std::vector<float> tys = {0.0f, 5.0f, 10.0f};  // steps of 5px < min_step=25
    std::vector<float> rots(3, 0.0f), scs(3, 0.0f);
    auto res = validate_affines_impl(txs, tys, rots, scs, 25.0f, 3.0f);
    CHECK_FALSE(res.valid);
    CHECK(res.reason.find("min_gap=") != std::string::npos);
}

// ---------------------------------------------------------------------------
// validate_affines_impl — rotation failure
// ---------------------------------------------------------------------------

TEST_CASE("validate_affines: high rotation fails", "[validation]") {
    auto [txs, tys, rots0, scs] = make_vertical_sequence(4, 100.0f);
    std::vector<float> rots = {0.0f, 0.0f, 0.25f, 0.0f};  // frame 2 has high rotation
    auto res = validate_affines_impl(txs, tys, rots, scs, 25.0f, 3.0f, 0.10f, 0.10f);
    CHECK_FALSE(res.valid);
    CHECK(res.reason.find("rotation=") != std::string::npos);
}

// ---------------------------------------------------------------------------
// validate_affines_impl — scale failure
// ---------------------------------------------------------------------------

TEST_CASE("validate_affines: high scale deviation fails", "[validation]") {
    auto [txs, tys, rots, scs0] = make_vertical_sequence(4, 100.0f);
    std::vector<float> scs = {0.0f, 0.0f, 0.25f, 0.0f};  // frame 2 has scale drift
    auto res = validate_affines_impl(txs, tys, rots, scs, 25.0f, 3.0f, 0.10f, 0.10f);
    CHECK_FALSE(res.valid);
    CHECK(res.reason.find("scale_dev=") != std::string::npos);
}

// ---------------------------------------------------------------------------
// validate_affines_impl — Kendall τ monotonicity failure
// ---------------------------------------------------------------------------

TEST_CASE("validate_affines: scrambled frame order fails monotonicity", "[validation]") {
    // 8 frames with alternating ty values (interleaved high/low) — tau_abs ≈ 0.14
    // A purely reversed sequence has tau_abs=1.0 and IS considered monotone.
    // The scrambled pattern is what actually fails Kendall-τ (tau_abs < MONO_TAU_MIN=0.40).
    int N = 8;
    std::vector<float> txs(N, 0.0f);
    std::vector<float> tys = {0.0f, 700.0f, 100.0f, 600.0f,
                              200.0f, 500.0f, 300.0f, 400.0f};
    std::vector<float> rots(N, 0.0f), scs(N, 0.0f);
    // Use a very large ratio/gap limit so only monotonicity can trigger failure
    auto res = validate_affines_impl(txs, tys, rots, scs, 1.0f, 200.0f, 0.5f, 0.5f);
    // Scrambled sequence (tau_abs ≈ 0.14 < 0.40) should fail monotonicity
    CHECK_FALSE(res.valid);
    CHECK(res.reason.find("monotonicity=") != std::string::npos);
}

// ---------------------------------------------------------------------------
// compute_adaptive_min_gap_impl
// ---------------------------------------------------------------------------

TEST_CASE("compute_adaptive_min_gap: floor is 20px", "[validation]") {
    // Very small span → should return floor 20.0
    std::vector<float> txs = {0.0f, 0.0f};
    std::vector<float> tys = {0.0f, 10.0f};  // span=10, N=2, adaptive=10/6≈1.7 < 20
    float gap = compute_adaptive_min_gap_impl(txs, tys);
    CHECK(gap == Catch::Approx(20.0f));
}

TEST_CASE("compute_adaptive_min_gap: single frame returns 20.0", "[validation]") {
    std::vector<float> txs = {0.0f};
    std::vector<float> tys = {0.0f};
    CHECK(compute_adaptive_min_gap_impl(txs, tys) == Catch::Approx(20.0f));
}

TEST_CASE("compute_adaptive_min_gap: large span exceeds floor", "[validation]") {
    // canvas_span = 3000px, N=10 → 3000/(10*3) = 100 > 20
    std::vector<float> txs(10, 0.0f);
    std::vector<float> tys(10);
    for (int i = 0; i < 10; ++i) tys[i] = i * 333.3f;  // span ≈ 3000
    float gap = compute_adaptive_min_gap_impl(txs, tys);
    CHECK(gap > 20.0f);
    CHECK(gap == Catch::Approx(tys[9] / 30.0f).epsilon(0.02f));
}

// ---------------------------------------------------------------------------
// compute_adaptive_rot_scale_impl
// ---------------------------------------------------------------------------

TEST_CASE("compute_adaptive_rot_scale: consistent rotations use loose threshold", "[validation]") {
    // All frames have the same tiny rotation → std < 0.02 → loose
    std::vector<float> rots(6, 0.11f);  // consistent, max_rot=0.11
    std::vector<float> scs(6, 0.0f);
    auto [max_rot, max_sc] = compute_adaptive_rot_scale_impl(rots, scs);
    CHECK(max_rot == Catch::Approx(0.15f));  // ROT_LOOSE
}

TEST_CASE("compute_adaptive_rot_scale: variable rotations use tight threshold", "[validation]") {
    // High variance in rotations → std ≥ 0.02 → tight
    std::vector<float> rots = {0.0f, 0.5f, 0.0f, 0.5f, 0.0f, 0.5f};
    std::vector<float> scs(6, 0.0f);
    auto [max_rot, max_sc] = compute_adaptive_rot_scale_impl(rots, scs);
    CHECK(max_rot == Catch::Approx(0.10f));  // ROT_TIGHT
}

TEST_CASE("compute_adaptive_rot_scale: <2 frames returns tight", "[validation]") {
    std::vector<float> rots = {0.0f};
    std::vector<float> scs  = {0.0f};
    auto [max_rot, max_sc] = compute_adaptive_rot_scale_impl(rots, scs);
    CHECK(max_rot == Catch::Approx(0.10f));
    CHECK(max_sc  == Catch::Approx(0.10f));
}
