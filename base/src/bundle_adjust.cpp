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
std::vector<AffineParams> bundle_adjust_affine_impl(
    std::vector<Edge> edges,
    int N,
    float f_scale,
    bool use_gnc,
    bool adaptive_f_scale)
{
    edges = spanning_tree_inlier_filter_impl(edges, N, 50.0f);

    int dof = 2; // tx, ty
    int num_vars = N * dof;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(num_vars);

    for (int f = 1; f < N; ++f) {
        for (const auto& e : edges) {
            if (e.src == f - 1 && e.dst == f) {
                x(f * dof + 0) = x((f - 1) * dof + 0) - e.dx;
                x(f * dof + 1) = x((f - 1) * dof + 1) - e.dy;
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
            JTWJ(0, 0) += 2000.0 * 2000.0;
            JTWJ(1, 1) += 2000.0 * 2000.0;
            JTWr(0) -= 2000.0 * 2000.0 * x_cur(0);
            JTWr(1) -= 2000.0 * 2000.0 * x_cur(1);

            // Trajectory regularizer
            double reg_traj = 0.10;
            for (int f = 1; f < N - 1; ++f) {
                // tx_acc = x[f+1] - 2*x[f] + x[f-1]
                // We add J_reg^T J_reg x = -J_reg^T r_reg
                // Here J_reg is [1, -2, 1]
                int idx0 = (f - 1) * dof;
                int idx1 = f * dof;
                int idx2 = (f + 1) * dof;
                
                double w_reg = reg_traj * reg_traj;
                
                // Add to JTWJ and JTWr for tx
                JTWJ(idx0, idx0) += w_reg; JTWJ(idx0, idx1) -= 2*w_reg; JTWJ(idx0, idx2) += w_reg;
                JTWJ(idx1, idx0) -= 2*w_reg; JTWJ(idx1, idx1) += 4*w_reg; JTWJ(idx1, idx2) -= 2*w_reg;
                JTWJ(idx2, idx0) += w_reg; JTWJ(idx2, idx1) -= 2*w_reg; JTWJ(idx2, idx2) += w_reg;
                
                double tx_acc = x_cur(idx2) - 2 * x_cur(idx1) + x_cur(idx0);
                JTWr(idx0) -= w_reg * tx_acc;
                JTWr(idx1) -= w_reg * tx_acc * (-2);
                JTWr(idx2) -= w_reg * tx_acc;

                // Add to JTWJ and JTWr for ty
                JTWJ(idx0+1, idx0+1) += w_reg; JTWJ(idx0+1, idx1+1) -= 2*w_reg; JTWJ(idx0+1, idx2+1) += w_reg;
                JTWJ(idx1+1, idx0+1) -= 2*w_reg; JTWJ(idx1+1, idx1+1) += 4*w_reg; JTWJ(idx1+1, idx2+1) -= 2*w_reg;
                JTWJ(idx2+1, idx0+1) += w_reg; JTWJ(idx2+1, idx1+1) -= 2*w_reg; JTWJ(idx2+1, idx2+1) += w_reg;
                
                double ty_acc = x_cur(idx2+1) - 2 * x_cur(idx1+1) + x_cur(idx0+1);
                JTWr(idx0+1) -= w_reg * ty_acc;
                JTWr(idx1+1) -= w_reg * ty_acc * (-2);
                JTWr(idx2+1) -= w_reg * ty_acc;
            }

            for (size_t idx = 0; idx < edges.size(); ++idx) {
                const auto& e = edges[idx];
                int i = e.src, j = e.dst;
                double pred_dx = x_cur(j * dof) - x_cur(i * dof);
                double pred_dy = x_cur(j * dof + 1) - x_cur(i * dof + 1);
                double obs_dx = -e.dx;
                double obs_dy = -e.dy;
                
                double res_x = pred_dx - obs_dx;
                double res_y = pred_dy - obs_dy;
                double res_sq = res_x * res_x + res_y * res_y;
                
                double w = e.weight * e.weight * gnc_ws[idx] * gnc_ws[idx];
                
                // Cauchy loss IRLS weight
                double cauchy_w = 1.0 / (1.0 + res_sq / c_sq);
                w *= cauchy_w;
                
                // J for res_x is [ ..., -1 (at i), ..., 1 (at j), ... ]
                // J^T W J adds w to (i,i), (j,j), -w to (i,j) and (j,i)
                JTWJ(i * dof, i * dof) += w;
                JTWJ(j * dof, j * dof) += w;
                JTWJ(i * dof, j * dof) -= w;
                JTWJ(j * dof, i * dof) -= w;
                
                JTWr(i * dof) -= w * (-res_x);
                JTWr(j * dof) -= w * (res_x);
                
                JTWJ(i * dof + 1, i * dof + 1) += w;
                JTWJ(j * dof + 1, j * dof + 1) += w;
                JTWJ(i * dof + 1, j * dof + 1) -= w;
                JTWJ(j * dof + 1, i * dof + 1) -= w;
                
                JTWr(i * dof + 1) -= w * (-res_y);
                JTWr(j * dof + 1) -= w * (res_y);
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
        float c_sq = 100.0f; // 10^2
        
        for (int outer = 0; outer < 8; ++outer) {
            std::vector<double> edge_res_sq(edges.size(), 0.0);
            double max_sq = 0.0;
            for (size_t idx = 0; idx < edges.size(); ++idx) {
                const auto& e = edges[idx];
                int i = e.src, j = e.dst;
                double pred_dx = x(j * dof) - x(i * dof);
                double pred_dy = x(j * dof + 1) - x(i * dof + 1);
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
                gnc_ws[idx] = w; // We don't square because w is applied as w^2 in solve_irls
            }
            
            x = solve_irls(1e9f, 200, gnc_ws); // Use 1e9 to practically disable Cauchy during GNC
            mu /= 1.4f; // anneal
        }
    } else {
        std::vector<float> gnc_ws(edges.size(), 1.0f);
        x = solve_irls(f_scale * f_scale, 200, gnc_ws);
        
        if (adaptive_f_scale) {
            std::vector<AffineParams> cur_affines;
            for (int f = 0; f < N; ++f) {
                cur_affines.push_back({(float)x(f * dof), (float)x(f * dof + 1), 1.0f, 0.0f, f});
            }
            float adapt = compute_adaptive_f_scale_impl(edges, cur_affines, f_scale);
            if (adapt > f_scale * 1.5f) {
                x = solve_irls(adapt * adapt, 200, gnc_ws);
            }
        }
    }

    std::vector<AffineParams> out;
    for (int f = 0; f < N; ++f) {
        out.push_back({(float)x(f * dof), (float)x(f * dof + 1), 1.0f, 0.0f, f});
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
    bool     adaptive_f_scale = true)
{
    std::vector<Edge> edges;
    for (auto item : edges_py) edges.push_back(edge_from_dict(item.cast<py::dict>()));
    auto result = bundle_adjust_affine_impl(edges, N, f_scale, use_gnc, adaptive_f_scale);
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
// bundle_adjust_affine
//
// Full affine bundle adjustment:
//   - Optional GNC-TLS outer loop (8 iterations, Geman-McClure weights)
//   - Eigen LDLT inner solve: (J^T W J) Δx = J^T W r
//   - Cauchy robust loss on inner iterations
//   - Optional adaptive f_scale re-solve
//
// Args
// ----
// edges            : list of Edge dicts {"i","j","dx","dy","weight"}
// N                : int, number of frames
// f_scale          : float, Cauchy loss scale (default 10.0)
// use_gnc          : bool, enable GNC-TLS outer loop
// adaptive_f_scale : bool, re-solve with median_residual-scaled f
//
// Returns
// -------
// list of AffineParams dicts {"tx","ty","scale","rotation","frame_idx"}
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// spanning_tree_inlier_filter
//
// Kruskal maximum spanning tree (highest-weight-first) with Union-Find.
// BFS from frame 0 propagates reference translations.
// Drops edges where predicted − observed displacement > inlier_threshold.
// Falls back to original edges if graph disconnects or < max(2,N-1) inliers.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// compute_adaptive_f_scale
//
// After an initial solve, compute adaptive_scale = max(floor, 2 × median_residual).
// Returns float.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// register_bundle_adjust — called from bindings.cpp
// ---------------------------------------------------------------------------
void register_bundle_adjust(py::module_& m) {
    m.doc() = R"doc(
        batch.bundle_adjust — Affine bundle adjustment via Eigen LDLT.

        Functions
        ---------
        bundle_adjust_affine(edges, N, f_scale, use_gnc, adaptive_f_scale) -> list[dict]
        spanning_tree_inlier_filter(edges, N, inlier_threshold) -> list[dict]
        compute_adaptive_f_scale(edges, affines, floor_scale) -> float
    )doc";

    m.def("bundle_adjust_affine", &bundle_adjust_affine,
        py::arg("edges"),
        py::arg("N"),
        py::arg("f_scale")          = 10.0f,
        py::arg("use_gnc")          = true,
        py::arg("adaptive_f_scale") = true,
        R"doc(
            Full affine bundle adjustment.

            Args
            ----
            edges   : list[dict]  — each has "i","j","dx","dy","weight"
            N       : int  — number of frames
            f_scale : float  — Cauchy robust loss scale
            use_gnc : bool  — enable GNC-TLS outer loop (8 iters)
            adaptive_f_scale : bool  — re-solve with median-residual f

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
