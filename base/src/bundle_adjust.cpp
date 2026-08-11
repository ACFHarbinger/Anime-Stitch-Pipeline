// ---------------------------------------------------------------------------
// batch/src/bundle_adjust.cpp
//
// Affine bundle adjustment: LM solver, GNC-TLS outer loop, spanning-tree
// inlier filter, adaptive f_scale, wave correct.
//
// Replaces:
//   alignment/bundle_adjust.py  :: _bundle_adjust_affine,
//                                   _spanning_tree_inlier_filter,
//                                   _compute_adaptive_f_scale
//
// Implementation roadmap: Phase 3.
// See docs/moon/roadmaps/asp_cpp_migration.md §base::bundle_adjust
// ---------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common.hpp"
#include "affine_types.hpp"
#include "math/graph.hpp"

#include <Eigen/Dense>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>

using namespace base;

// ---------------------------------------------------------------------------
// spanning_tree_inlier_filter_impl
// ---------------------------------------------------------------------------
std::vector<Edge> spanning_tree_inlier_filter_impl(
    const std::vector<Edge>& edges,
    int N,
    float inlier_threshold)
{
    if (edges.size() < 2 || N < 2) return edges;

    std::vector<Edge> sorted_edges = edges;
    std::sort(sorted_edges.begin(), sorted_edges.end(), [](const Edge& a, const Edge& b) {
        return a.weight > b.weight;
    });

    base::math::UnionFind uf(N);

    struct TreeEdge { int to; float dtx, dty; };
    std::vector<std::vector<TreeEdge>> tree_adj(N);
    int n_tree_edges = 0;

    for (const auto& e : sorted_edges) {
        int i = e.src, j = e.dst;
        if (i < 0 || i >= N || j < 0 || j >= N) continue;
        if (uf.unite(i, j)) {
            float dtx = -e.dx;
            float dty = -e.dy;
            tree_adj[i].push_back({j, dtx, dty});
            tree_adj[j].push_back({i, -dtx, -dty});
            n_tree_edges++;
        }
        if (n_tree_edges == N - 1) break;
    }

    std::vector<bool> visited(N, false);
    std::vector<float> tx_ref(N, 0.0f);
    std::vector<float> ty_ref(N, 0.0f);
    std::queue<int> q;
    q.push(0);
    visited[0] = true;

    while (!q.empty()) {
        int curr = q.front();
        q.pop();
        for (const auto& nbr : tree_adj[curr]) {
            if (!visited[nbr.to]) {
                tx_ref[nbr.to] = tx_ref[curr] + nbr.dtx;
                ty_ref[nbr.to] = ty_ref[curr] + nbr.dty;
                visited[nbr.to] = true;
                q.push(nbr.to);
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        if (!visited[i]) return edges; // disconnected graph
    }

    std::vector<Edge> inlier_edges;
    for (const auto& e : edges) {
        int i = e.src, j = e.dst;
        float pred_dx = tx_ref[j] - tx_ref[i];
        float pred_dy = ty_ref[j] - ty_ref[i];
        float obs_dx = -e.dx;
        float obs_dy = -e.dy;
        float residual = std::sqrt(std::pow(pred_dx - obs_dx, 2) + std::pow(pred_dy - obs_dy, 2));
        if (residual <= inlier_threshold) {
            inlier_edges.push_back(e);
        }
    }

    if (inlier_edges.size() < std::max(2, N - 1)) {
        return edges;
    }

    return inlier_edges;
}

// ---------------------------------------------------------------------------
// compute_adaptive_f_scale_impl
// ---------------------------------------------------------------------------
float compute_adaptive_f_scale_impl(
    const std::vector<Edge>& edges,
    const std::vector<AffineParams>& affines,
    float floor_scale)
{
    if (edges.empty() || affines.empty()) return floor_scale;
    std::vector<float> res_mags;
    res_mags.reserve(edges.size());
    for (const auto& e : edges) {
        int i = e.src, j = e.dst;
        if (i >= (int)affines.size() || j >= (int)affines.size()) continue;
        float pred_dx = affines[j].tx - affines[i].tx;
        float pred_dy = affines[j].ty - affines[i].ty;
        float obs_dx = -e.dx;
        float obs_dy = -e.dy;
        res_mags.push_back(std::sqrt(std::pow(pred_dx - obs_dx, 2) + std::pow(pred_dy - obs_dy, 2)));
    }
    if (res_mags.empty()) return floor_scale;
    std::sort(res_mags.begin(), res_mags.end());
    float median = res_mags[res_mags.size() / 2];
    if (res_mags.size() % 2 == 0) {
        median = (median + res_mags[res_mags.size() / 2 - 1]) / 2.0f;
    }
    return std::max(floor_scale, 2.0f * median);
}

// ---------------------------------------------------------------------------
// bundle_adjust_affine_impl
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// bundle_adjust_affine_impl
// ---------------------------------------------------------------------------
enum MotionModel {
    BUNDLE_ADJUST_2D_TRANSLATION = 0,
    BUNDLE_ADJUST_2D_TRANSLATION_SCALE = 1,
    BUNDLE_ADJUST_AFFINE = 2
};

std::vector<AffineParams> bundle_adjust_affine_impl(
    std::vector<Edge> edges,
    int N,
    float f_scale,
    bool use_gnc,
    bool adaptive_f_scale,
    int motion_model)
{
    edges = spanning_tree_inlier_filter_impl(edges, N, 50.0f);

    int dof = 2; // default: 2D translation [tx, ty]
    if (motion_model == BUNDLE_ADJUST_2D_TRANSLATION_SCALE) {
        dof = 3; // [tx, ty, scale]
    } else if (motion_model == BUNDLE_ADJUST_AFFINE) {
        dof = 4; // [a, b, tx, ty]
    }

    int num_vars = N * dof;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(num_vars);

    for (int f = 0; f < N; ++f) {
        if (motion_model == BUNDLE_ADJUST_2D_TRANSLATION_SCALE) {
            x(f * dof + 2) = 1.0;
        } else if (motion_model == BUNDLE_ADJUST_AFFINE) {
            x(f * dof + 0) = 1.0;
        }
    }

    for (int f = 1; f < N; ++f) {
        for (const auto& e : edges) {
            if (e.src == f - 1 && e.dst == f) {
                if (motion_model == BUNDLE_ADJUST_2D_TRANSLATION_SCALE) {
                    x(f * dof + 0) = x((f - 1) * dof + 0) - e.dx;
                    x(f * dof + 1) = x((f - 1) * dof + 1) - e.dy;
                    x(f * dof + 2) = 1.0;
                } else if (motion_model == BUNDLE_ADJUST_AFFINE) {
                    x(f * dof + 2) = x((f - 1) * dof + 2) - e.dx;
                    x(f * dof + 3) = x((f - 1) * dof + 3) - e.dy;
                } else {
                    x(f * dof + 0) = x((f - 1) * dof + 0) - e.dx;
                    x(f * dof + 1) = x((f - 1) * dof + 1) - e.dy;
                }
                break;
            }
        }
    }

    auto solve_irls = [&](float c_sq, int max_iters, std::vector<float>& gnc_ws) {
        Eigen::VectorXd x_cur = x;
        for (int iter = 0; iter < max_iters; ++iter) {
            Eigen::MatrixXd JTWJ = Eigen::MatrixXd::Zero(num_vars, num_vars);
            Eigen::VectorXd JTWr = Eigen::VectorXd::Zero(num_vars);

            // Anchor frame 0
            if (motion_model == BUNDLE_ADJUST_2D_TRANSLATION_SCALE) {
                JTWJ(0, 0) += 2000.0 * 2000.0;
                JTWJ(1, 1) += 2000.0 * 2000.0;
                JTWJ(2, 2) += 2000.0 * 2000.0;
                JTWr(0) -= 2000.0 * 2000.0 * x_cur(0);
                JTWr(1) -= 2000.0 * 2000.0 * x_cur(1);
                JTWr(2) -= 2000.0 * 2000.0 * (x_cur(2) - 1.0);
            } else if (motion_model == BUNDLE_ADJUST_AFFINE) {
                JTWJ(0, 0) += 2000.0 * 2000.0;
                JTWJ(1, 1) += 2000.0 * 2000.0;
                JTWJ(2, 2) += 2000.0 * 2000.0;
                JTWJ(3, 3) += 2000.0 * 2000.0;
                JTWr(0) -= 2000.0 * 2000.0 * (x_cur(0) - 1.0);
                JTWr(1) -= 2000.0 * 2000.0 * x_cur(1);
                JTWr(2) -= 2000.0 * 2000.0 * x_cur(2);
                JTWr(3) -= 2000.0 * 2000.0 * x_cur(3);
            } else {
                JTWJ(0, 0) += 2000.0 * 2000.0;
                JTWJ(1, 1) += 2000.0 * 2000.0;
                JTWr(0) -= 2000.0 * 2000.0 * x_cur(0);
                JTWr(1) -= 2000.0 * 2000.0 * x_cur(1);
            }

            // Identity / scale prior for non-anchor frames
            if (motion_model == BUNDLE_ADJUST_2D_TRANSLATION_SCALE) {
                double reg_scale = 100.0 * 100.0;
                for (int f = 1; f < N; ++f) {
                    int s_idx = f * 3 + 2;
                    JTWJ(s_idx, s_idx) += reg_scale;
                    JTWr(s_idx) -= reg_scale * (x_cur(s_idx) - 1.0);
                }
            } else if (motion_model == BUNDLE_ADJUST_AFFINE) {
                double reg_id = 1000.0 * 1000.0;
                for (int f = 1; f < N; ++f) {
                    int a_idx = f * 4 + 0;
                    int b_idx = f * 4 + 1;
                    JTWJ(a_idx, a_idx) += reg_id;
                    JTWr(a_idx) -= reg_id * (x_cur(a_idx) - 1.0);
                    JTWJ(b_idx, b_idx) += reg_id;
                    JTWr(b_idx) -= reg_id * x_cur(b_idx);
                }
            }

            // Trajectory smoothness regularizer
            double reg_traj = 0.10;
            double w_reg = reg_traj * reg_traj;
            for (int f = 1; f < N - 1; ++f) {
                int tx_off = (motion_model == BUNDLE_ADJUST_AFFINE) ? 2 : 0;
                int ty_off = (motion_model == BUNDLE_ADJUST_AFFINE) ? 3 : 1;

                int idx0_x = (f - 1) * dof + tx_off;
                int idx1_x = f * dof + tx_off;
                int idx2_x = (f + 1) * dof + tx_off;

                JTWJ(idx0_x, idx0_x) += w_reg; JTWJ(idx0_x, idx1_x) -= 2*w_reg; JTWJ(idx0_x, idx2_x) += w_reg;
                JTWJ(idx1_x, idx0_x) -= 2*w_reg; JTWJ(idx1_x, idx1_x) += 4*w_reg; JTWJ(idx1_x, idx2_x) -= 2*w_reg;
                JTWJ(idx2_x, idx0_x) += w_reg; JTWJ(idx2_x, idx1_x) -= 2*w_reg; JTWJ(idx2_x, idx2_x) += w_reg;

                double tx_acc = x_cur(idx2_x) - 2 * x_cur(idx1_x) + x_cur(idx0_x);
                JTWr(idx0_x) -= w_reg * tx_acc;
                JTWr(idx1_x) -= w_reg * tx_acc * (-2);
                JTWr(idx2_x) -= w_reg * tx_acc;

                int idx0_y = (f - 1) * dof + ty_off;
                int idx1_y = f * dof + ty_off;
                int idx2_y = (f + 1) * dof + ty_off;

                JTWJ(idx0_y, idx0_y) += w_reg; JTWJ(idx0_y, idx1_y) -= 2*w_reg; JTWJ(idx0_y, idx2_y) += w_reg;
                JTWJ(idx1_y, idx0_y) -= 2*w_reg; JTWJ(idx1_y, idx1_y) += 4*w_reg; JTWJ(idx1_y, idx2_y) -= 2*w_reg;
                JTWJ(idx2_y, idx0_y) += w_reg; JTWJ(idx2_y, idx1_y) -= 2*w_reg; JTWJ(idx2_y, idx2_y) += w_reg;

                double ty_acc = x_cur(idx2_y) - 2 * x_cur(idx1_y) + x_cur(idx0_y);
                JTWr(idx0_y) -= w_reg * ty_acc;
                JTWr(idx1_y) -= w_reg * ty_acc * (-2);
                JTWr(idx2_y) -= w_reg * ty_acc;

                if (motion_model == BUNDLE_ADJUST_2D_TRANSLATION_SCALE) {
                    int idx0_s = (f - 1) * dof + 2;
                    int idx1_s = f * dof + 2;
                    int idx2_s = (f + 1) * dof + 2;

                    JTWJ(idx0_s, idx0_s) += w_reg; JTWJ(idx0_s, idx1_s) -= 2*w_reg; JTWJ(idx0_s, idx2_s) += w_reg;
                    JTWJ(idx1_s, idx0_s) -= 2*w_reg; JTWJ(idx1_s, idx1_s) += 4*w_reg; JTWJ(idx1_s, idx2_s) -= 2*w_reg;
                    JTWJ(idx2_s, idx0_s) += w_reg; JTWJ(idx2_s, idx1_s) -= 2*w_reg; JTWJ(idx2_s, idx2_s) += w_reg;

                    double s_acc = x_cur(idx2_s) - 2 * x_cur(idx1_s) + x_cur(idx0_s);
                    JTWr(idx0_s) -= w_reg * s_acc;
                    JTWr(idx1_s) -= w_reg * s_acc * (-2);
                    JTWr(idx2_s) -= w_reg * s_acc;
                }
            }

            for (size_t idx = 0; idx < edges.size(); ++idx) {
                const auto& e = edges[idx];
                int i = e.src, j = e.dst;

                int tx_off = (motion_model == BUNDLE_ADJUST_AFFINE) ? 2 : 0;
                int ty_off = (motion_model == BUNDLE_ADJUST_AFFINE) ? 3 : 1;

                double pred_dx = x_cur(j * dof + tx_off) - x_cur(i * dof + tx_off);
                double pred_dy = x_cur(j * dof + ty_off) - x_cur(i * dof + ty_off);
                double obs_dx = -e.dx;
                double obs_dy = -e.dy;

                double res_x = pred_dx - obs_dx;
                double res_y = pred_dy - obs_dy;
                double res_sq = res_x * res_x + res_y * res_y;

                double w = e.weight * e.weight * gnc_ws[idx] * gnc_ws[idx];

                // Cauchy loss IRLS weight
                double cauchy_w = 1.0 / (1.0 + res_sq / c_sq);
                w *= cauchy_w;

                int ix = i * dof + tx_off;
                int jx = j * dof + tx_off;
                JTWJ(ix, ix) += w; JTWJ(jx, jx) += w;
                JTWJ(ix, jx) -= w; JTWJ(jx, ix) -= w;
                JTWr(ix) -= w * (-res_x);
                JTWr(jx) -= w * (res_x);

                int iy = i * dof + ty_off;
                int jy = j * dof + ty_off;
                JTWJ(iy, iy) += w; JTWJ(jy, jy) += w;
                JTWJ(iy, jy) -= w; JTWJ(jy, iy) -= w;
                JTWr(iy) -= w * (-res_y);
                JTWr(jy) -= w * (res_y);

                if (motion_model == BUNDLE_ADJUST_2D_TRANSLATION_SCALE) {
                    int is = i * dof + 2;
                    int js = j * dof + 2;
                    double res_s = x_cur(js) - x_cur(is);
                    double w_s = w * 100.0;
                    JTWJ(is, is) += w_s; JTWJ(js, js) += w_s;
                    JTWJ(is, js) -= w_s; JTWJ(js, is) -= w_s;
                    JTWr(is) -= w_s * (-res_s);
                    JTWr(js) -= w_s * (res_s);
                }
            }

            Eigen::VectorXd dx = JTWJ.ldlt().solve(JTWr);
            x_cur += dx;

            if (dx.norm() < 1e-4) break;
        }
        return x_cur;
    };

    if (use_gnc) {
        std::vector<float> gnc_ws(edges.size(), 1.0f);
        float mu = -1.0f;
        float c_sq = f_scale * f_scale;

        for (int outer = 0; outer < 8; ++outer) {
            std::vector<double> edge_res_sq(edges.size(), 0.0);
            double max_sq = 0.0;
            int tx_off = (motion_model == BUNDLE_ADJUST_AFFINE) ? 2 : 0;
            int ty_off = (motion_model == BUNDLE_ADJUST_AFFINE) ? 3 : 1;

            for (size_t idx = 0; idx < edges.size(); ++idx) {
                const auto& e = edges[idx];
                int i = e.src, j = e.dst;
                double pred_dx = x(j * dof + tx_off) - x(i * dof + tx_off);
                double pred_dy = x(j * dof + ty_off) - x(i * dof + ty_off);
                double obs_dx = -e.dx;
                double obs_dy = -e.dy;
                double sq = (pred_dx - obs_dx) * (pred_dx - obs_dx) + (pred_dy - obs_dy) * (pred_dy - obs_dy);
                edge_res_sq[idx] = sq;
                max_sq = std::max(max_sq, sq);
            }
            if (mu < 0.0f) {
                mu = std::max(1.0, max_sq / (2.0 * c_sq));
            }

            for (size_t idx = 0; idx < edges.size(); ++idx) {
                double denom = mu * c_sq + edge_res_sq[idx];
                double w = (mu * c_sq) / std::max(denom, 1e-12);
                gnc_ws[idx] = static_cast<float>(w);
            }

            x = solve_irls(1e9f, 200, gnc_ws);
            mu /= 1.4f; // anneal
        }
    } else {
        std::vector<float> gnc_ws(edges.size(), 1.0f);
        x = solve_irls(f_scale * f_scale, 200, gnc_ws);

        if (adaptive_f_scale) {
            std::vector<AffineParams> cur_affines;
            int tx_off = (motion_model == BUNDLE_ADJUST_AFFINE) ? 2 : 0;
            int ty_off = (motion_model == BUNDLE_ADJUST_AFFINE) ? 3 : 1;
            for (int f = 0; f < N; ++f) {
                float s = (motion_model == BUNDLE_ADJUST_2D_TRANSLATION_SCALE) ? (float)x(f * dof + 2) : 1.0f;
                cur_affines.push_back({(float)x(f * dof + tx_off), (float)x(f * dof + ty_off), s, 0.0f, f});
            }
            float adapt = compute_adaptive_f_scale_impl(edges, cur_affines, f_scale);
            if (adapt > f_scale * 1.5f) {
                x = solve_irls(adapt * adapt, 200, gnc_ws);
            }
        }
    }

    std::vector<AffineParams> out;
    int tx_off = (motion_model == BUNDLE_ADJUST_AFFINE) ? 2 : 0;
    int ty_off = (motion_model == BUNDLE_ADJUST_AFFINE) ? 3 : 1;
    for (int f = 0; f < N; ++f) {
        float s = 1.0f;
        float r = 0.0f;
        if (motion_model == BUNDLE_ADJUST_2D_TRANSLATION_SCALE) {
            s = (float)x(f * dof + 2);
        } else if (motion_model == BUNDLE_ADJUST_AFFINE) {
            float a = (float)x(f * dof + 0);
            float b = (float)x(f * dof + 1);
            s = std::sqrt(a * a + b * b);
            r = std::atan2(b, a);
        }
        out.push_back({(float)x(f * dof + tx_off), (float)x(f * dof + ty_off), s, r, f});
    }
    return out;
}

#ifndef BATCH_TESTS
// ---------------------------------------------------------------------------
// bundle_adjust_affine wrapper
// ---------------------------------------------------------------------------
static py::list bundle_adjust_affine(
    py::list edges_py,
    int      N,
    float    f_scale          = 10.0f,
    bool     use_gnc          = true,
    bool     adaptive_f_scale = true,
    int      motion_model     = 0)
{
    std::vector<Edge> edges;
    for (auto item : edges_py) edges.push_back(edge_from_dict(item.cast<py::dict>()));
    auto result = bundle_adjust_affine_impl(edges, N, f_scale, use_gnc, adaptive_f_scale, motion_model);
    return affines_to_list(result);
}

// ---------------------------------------------------------------------------
// spanning_tree_inlier_filter wrapper
// ---------------------------------------------------------------------------
static py::list spanning_tree_inlier_filter(
    py::list edges_py,
    int      N,
    float    inlier_threshold = 50.0f)
{
    std::vector<Edge> edges;
    for (auto item : edges_py) edges.push_back(edge_from_dict(item.cast<py::dict>()));
    auto result = spanning_tree_inlier_filter_impl(edges, N, inlier_threshold);
    return edges_to_list(result);
}

// ---------------------------------------------------------------------------
// compute_adaptive_f_scale wrapper
// ---------------------------------------------------------------------------
static float compute_adaptive_f_scale(
    py::list edges_py,
    py::list affines_py,
    float    floor_scale = 5.0f)
{
    std::vector<Edge> edges;
    for (auto item : edges_py) edges.push_back(edge_from_dict(item.cast<py::dict>()));
    std::vector<AffineParams> affines;
    for (auto item : affines_py) affines.push_back(affine_from_dict(item.cast<py::dict>()));
    return compute_adaptive_f_scale_impl(edges, affines, floor_scale);
}

// ---------------------------------------------------------------------------
// register_bundle_adjust — called from bindings.cpp
// ---------------------------------------------------------------------------
void register_bundle_adjust(py::module_& m) {
    m.doc() = R"doc(
        batch.bundle_adjust — Affine bundle adjustment via Eigen LDLT.

        Functions
        ---------
        bundle_adjust_affine(edges, N, f_scale, use_gnc, adaptive_f_scale, motion_model) -> list[dict]
        spanning_tree_inlier_filter(edges, N, inlier_threshold) -> list[dict]
        compute_adaptive_f_scale(edges, affines, floor_scale) -> float
    )doc";

    m.attr("BUNDLE_ADJUST_2D_TRANSLATION")       = 0;
    m.attr("BUNDLE_ADJUST_2D_TRANSLATION_SCALE") = 1;
    m.attr("BUNDLE_ADJUST_AFFINE")               = 2;

    m.def("bundle_adjust_affine", &bundle_adjust_affine,
        py::arg("edges"),
        py::arg("N"),
        py::arg("f_scale")          = 10.0f,
        py::arg("use_gnc")          = true,
        py::arg("adaptive_f_scale") = true,
        py::arg("motion_model")     = 0,
        R"doc(
            Full affine bundle adjustment.

            Args
            ----
            edges   : list[dict]  — each has "i","j","dx","dy","weight"
            N       : int  — number of frames
            f_scale : float  — Cauchy robust loss scale
            use_gnc : bool  — enable GNC-TLS outer loop (8 iters)
            adaptive_f_scale : bool  — re-solve with median-residual f
            motion_model : int — 0: 2D Translation, 1: 2D Translation+Scale, 2: Affine

            Returns
            -------
            list[dict] with keys "tx","ty","scale","rotation","frame_idx"
        )doc");

    m.def("spanning_tree_inlier_filter", &spanning_tree_inlier_filter,
        py::arg("edges"),
        py::arg("N"),
        py::arg("inlier_threshold") = 50.0f,
        R"doc(
            Kruskal maximum spanning tree inlier filter.

            Drops edges with predicted–observed displacement > inlier_threshold.
            Falls back to original edges if graph becomes disconnected.

            Returns list[dict] — filtered edges.
        )doc");

    m.def("compute_adaptive_f_scale", &compute_adaptive_f_scale,
        py::arg("edges"),
        py::arg("affines"),
        py::arg("floor_scale") = 5.0f,
        R"doc(
            Compute adaptive_scale = max(floor_scale, 2 × median_residual_px).

            Returns float.
        )doc");
}
#endif // BATCH_TESTS
