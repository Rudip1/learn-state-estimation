// Chapter 10 tests, pose graphs: the edge error is the compounding (-)z (+) ((-)x_i (+) x_j) and its Jacobians match
// central differences; a graph with exact measurements is solved exactly from a perturbed start; an odometry chain
// reproduces dead reckoning, mean and covariance; a loop closure removes most of the drift; node 0 stays fixed; H
// has non-zero blocks only where edges are.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "state_estimation/pose_graph.hpp"
#include "state_estimation/scan_matching.hpp"
#include "state_estimation/simulation.hpp"

using namespace state_estimation;
using Catch::Approx;

namespace {
constexpr double kPi = 3.14159265358979323846;

/// Poses around a square of side 4 m, n per side, heading along the side.
std::vector<Pose2> square_loop(int per_side) {
    std::vector<Pose2> poses;
    Pose2 x(0.0, 0.0, 0.0);
    for (int side = 0; side < 4; ++side) {
        for (int k = 0; k < per_side; ++k) {
            poses.push_back(x);
            x = compose(x, Pose2(4.0 / per_side, 0.0, 0.0));
        }
        x = compose(x, Pose2(0.0, 0.0, kPi / 2));
    }
    poses.push_back(x);  // back at the start
    return poses;
}
}  // namespace

TEST_CASE("Edge error is the compounding of the measurement and the relative pose", "[ch10]") {
    const Pose2 xi(1.0, 2.0, 0.4), xj(2.5, 1.0, -0.8), z(1.2, -1.0, -1.1);
    const Vector3d e = pose_graph_error(xi, xj, z);
    Pose2 ref = compose(inverse(z), between(xi, xj));
    ref.z() = wrap_angle(ref.z());
    CHECK((e - ref).norm() < 1e-12);
    CHECK(pose_graph_error(xi, xj, between(xi, xj)).norm() < 1e-12);
}

TEST_CASE("Edge Jacobians match central differences", "[ch10]") {
    const Pose2 xi(1.0, 2.0, 0.4), xj(2.5, 1.0, -0.8), z(1.2, -1.0, -1.1);
    Matrix3d A, B;
    pose_graph_jacobians(xi, xj, z, A, B);
    const MatrixXd An = numerical_jacobian([&](const VectorXd& v) -> VectorXd { return pose_graph_error(v, xj, z); }, xi);
    const MatrixXd Bn = numerical_jacobian([&](const VectorXd& v) -> VectorXd { return pose_graph_error(xi, v, z); }, xj);
    CHECK((A - An).norm() < 1e-8);
    CHECK((B - Bn).norm() < 1e-8);
}

TEST_CASE("A graph with exact measurements is solved exactly", "[ch10]") {
    const std::vector<Pose2> truth = square_loop(4);
    PoseGraph g;
    Rng rng(1);
    for (size_t k = 0; k < truth.size(); ++k) {
        Pose2 guess = truth[k];
        if (k > 0) guess += Pose2(rng.normal(0, 0.3), rng.normal(0, 0.3), rng.normal(0, 0.15));
        g.add_node(guess);
    }
    for (size_t k = 0; k + 1 < truth.size(); ++k) g.add_edge(k, k + 1, between(truth[k], truth[k + 1]), Matrix3d::Identity());
    g.add_edge(0, static_cast<int>(truth.size()) - 1, between(truth.front(), truth.back()), Matrix3d::Identity());
    g.add_edge(2, 10, between(truth[2], truth[10]), Matrix3d::Identity());
    const OptimizationSummary s = g.optimize(30, 1e-12);
    CHECK(s.converged);
    CHECK(s.final_chi2 < 1e-18);
    CHECK(s.initial_chi2 > 1.0);
    for (size_t k = 0; k < truth.size(); ++k) {
        CHECK((g.node(k).head<2>() - truth[k].head<2>()).norm() < 1e-9);
        CHECK(std::abs(wrap_angle(g.node(k).z() - truth[k].z())) < 1e-9);
    }
}

TEST_CASE("An odometry chain reproduces dead reckoning, mean and covariance", "[ch10]") {
    // Without loop closures the optimum composes the measurements, and the marginal covariance of the last node is
    // the dead-reckoning covariance P_k = F P F^T + W Q W^T, with F = J1(+), W = J2(+) — provided the edge
    // information is expressed in the coordinates of the error, eq. (10.17). With Omega = Q^{-1} it is not.
    const Matrix3d Q = Vector3d(0.01, 0.004, 0.002).asDiagonal();
    PoseGraph g;
    g.add_node(Pose2::Zero());
    Pose2 x = Pose2::Zero();
    Matrix3d P = Matrix3d::Zero();
    const Pose2 u(0.5, 0.05, 0.12);
    for (int k = 0; k < 15; ++k) {
        const Matrix3d F = compose_jacobian_1(x, u), W = compose_jacobian_2(x, u);
        P = F * P * F.transpose() + W * Q * W.transpose();
        x = compose(x, u);
        g.add_node(Pose2::Zero());  // poor initial guess on purpose
        g.add_edge(k, k + 1, u, pose_graph_information(u, Q));
    }
    g.optimize(50, 1e-12);
    const Pose2 last = g.node(g.n_nodes() - 1);
    CHECK((last.head<2>() - x.head<2>()).norm() < 1e-9);
    CHECK(std::abs(wrap_angle(last.z() - x.z())) < 1e-9);
    CHECK((g.marginal_covariance(g.n_nodes() - 1) - P).norm() < 1e-8 * P.norm());
    // The shortcut Omega = Q^{-1} gives a visibly different covariance (Q is not symmetric in x-y here).
    PoseGraph g2;
    g2.add_node(Pose2::Zero());
    for (int k = 0; k < 15; ++k) {
        g2.add_node(Pose2::Zero());
        g2.add_edge(k, k + 1, u, Q.inverse());
    }
    g2.optimize(50, 1e-12);
    CHECK((g2.marginal_covariance(g2.n_nodes() - 1) - P).norm() > 1e-3 * P.norm());
}

TEST_CASE("Edge information from the measurement covariance matches Monte Carlo of the error", "[ch10]") {
    const Pose2 xi(1.0, 0.5, 0.3), z(1.5, 0.4, 0.9);
    const Pose2 xj = compose(xi, z);
    const Matrix3d Sz = (Matrix3d() << 0.02, 0.005, 0.001, 0.005, 0.01, 0.0, 0.001, 0.0, 0.004).finished();
    Rng rng(11);
    const MatrixXd zs = sample_gaussian(z, Sz, 100000, rng);
    MatrixXd es(zs.rows(), 3);
    for (int k = 0; k < zs.rows(); ++k) es.row(k) = pose_graph_error(xi, xj, zs.row(k).transpose()).transpose();
    const Matrix3d C = sample_mean_cov(es).cov;
    CHECK((pose_graph_information(z, Sz).inverse() - C).norm() < 0.03 * C.norm());
}

TEST_CASE("A loop closure removes most of the odometry drift", "[ch10]") {
    const std::vector<Pose2> truth = square_loop(10);
    const int n = static_cast<int>(truth.size());
    const Matrix3d Q = Vector3d(0.02 * 0.02, 0.02 * 0.02, 0.01 * 0.01).asDiagonal();
    Rng rng(9);
    double dr_err = 0.0, opt_err = 0.0;
    const int runs = 20;
    for (int r = 0; r < runs; ++r) {
        PoseGraph g;
        Pose2 x = truth[0];
        g.add_node(x);
        for (int k = 0; k + 1 < n; ++k) {
            Pose2 z = between(truth[k], truth[k + 1]) + sample_gaussian(Vector3d::Zero(), Q, 1, rng).row(0).transpose();
            x = compose(x, z);
            g.add_node(x);
            g.add_edge(k, k + 1, z, pose_graph_information(z, Q));
        }
        dr_err += (g.node(n - 1).head<2>() - truth[n - 1].head<2>()).norm();
        // The robot recognises the start: an accurate relative-pose measurement between the last and the first node.
        const Matrix3d R = Vector3d(1e-4, 1e-4, 1e-5).asDiagonal();
        g.add_edge(n - 1, 0, between(truth[n - 1], truth[0]), R.inverse());
        g.optimize();
        opt_err += (g.node(n - 1).head<2>() - truth[n - 1].head<2>()).norm();
        CHECK(g.node(0) == truth[0]);  // the gauge does not move
    }
    INFO("mean final error: dead reckoning " << dr_err / runs << " m, pose graph " << opt_err / runs << " m");
    CHECK(opt_err < 0.1 * dr_err);
}

TEST_CASE("The information matrix is sparse: blocks only where edges are", "[ch10]") {
    const std::vector<Pose2> truth = square_loop(3);
    PoseGraph g;
    for (const auto& x : truth) g.add_node(x);
    for (size_t k = 0; k + 1 < truth.size(); ++k) g.add_edge(k, k + 1, between(truth[k], truth[k + 1]), Matrix3d::Identity());
    g.add_edge(1, 7, between(truth[1], truth[7]), Matrix3d::Identity());
    const MatrixXd H = g.information_matrix();
    for (int i = 0; i < g.n_nodes(); ++i)
        for (int j = 0; j < g.n_nodes(); ++j) {
            const bool linked = std::abs(i - j) <= 1 || (std::min(i, j) == 1 && std::max(i, j) == 7);
            const double block = H.block<3, 3>(3 * i, 3 * j).norm();
            if (linked)
                CHECK(block > 0.0);
            else
                CHECK(block == 0.0);
        }
}

TEST_CASE("A wrong loop closure stands out in its edge chi2 and can be removed", "[ch10]") {
    const std::vector<Pose2> truth = square_loop(10);
    const int n = static_cast<int>(truth.size());
    const Matrix3d Q = Vector3d(0.02 * 0.02, 0.02 * 0.02, 0.01 * 0.01).asDiagonal();
    Rng rng(12);
    PoseGraph g;
    g.add_node(truth[0]);
    for (int k = 0; k + 1 < n; ++k) {
        const Pose2 z = between(truth[k], truth[k + 1]) + sample_gaussian(Vector3d::Zero(), Q, 1, rng).row(0).transpose();
        g.add_node(compose(g.node(k), z));
        g.add_edge(k, k + 1, z, pose_graph_information(z, Q));
    }
    const Matrix3d info = Vector3d(2500.0, 2500.0, 1e4).asDiagonal();  // an ICP match: sigma 2 cm, 0.6 deg
    g.add_edge(n - 1, 0, between(truth[n - 1], truth[0]), info);       // correct closure
    g.add_edge(25, 5, between(truth[25], truth[5]) + Pose2(0.8, 0.0, 0.0), info);  // false match, 0.8 m off
    g.optimize();
    // One false match in a loop is spread over the loop's edges, so no single edge needs to exceed its own gate;
    // the total chi2 against its degrees of freedom (3 per edge minus 3 per free node) is the reliable test.
    const int dof = 3 * g.n_edges() - 3 * (g.n_nodes() - 1);
    CHECK(g.chi2() > chi2_quantile(0.999, dof));
    const std::vector<double> c = g.edge_chi2();
    int worst = 0;
    for (int k = 1; k < g.n_edges(); ++k)
        if (c[k] > c[worst]) worst = k;
    CHECK(worst == g.n_edges() - 1);  // the false match carries the largest share
    g.remove_edge(worst);
    g.optimize();
    const int dof2 = 3 * g.n_edges() - 3 * (g.n_nodes() - 1);
    CHECK(g.chi2() < chi2_quantile(0.999, dof2));
    CHECK((g.node(25).head<2>() - truth[25].head<2>()).norm() < 0.15);
}

TEST_CASE("Incremental pose-graph SLAM maps the office to centimetres", "[ch10]") {
    // Two laps; every keyframe edge is checked against ground truth, and the final graph against its chi2 test.
    const DifferentialDrive dd{0.1, 0.5, 0};
    const double k = 2e-4;
    for (int seed : {3, 5, 7}) {
        Rng rng(seed);
        const Pose2 x0(-5.5, -4.0, 0.0);
        const LidarRun run = simulate_lidar_run(x0, office_lap_controls(2, 0.1), 0.1, dd, k, office_world(), 10, 180,
                                                8.0, 0.01, rng);
        PoseGraphSlam slam(x0, run.scans[0]);
        for (size_t f = 1; f < run.scans.size(); ++f) slam.add_keyframe(run.odometry[f], run.odometry_cov[f], run.scans[f]);
        const PoseGraph& g = slam.graph();
        INFO("seed " << seed << ": " << slam.loop_closures().size() << " closures");
        CHECK(slam.loop_closures().size() > 50);
        for (const auto& [a, b] : slam.loop_closures()) {
            const Pose2 truth = between(run.truth.row(run.keyframes[a]).transpose(), run.truth.row(run.keyframes[b]).transpose());
            bool found = false;
            for (const auto& e : g.edges())
                if (e.i == a && e.j == b) found = (e.z.head<2>() - truth.head<2>()).norm() < 0.05;
            CHECK(found);  // no false loop closure
        }
        const int dof = 3 * g.n_edges() - 3 * (g.n_nodes() - 1);
        CHECK(g.chi2() < chi2_quantile(0.999, dof));
        double se2 = 0.0;
        for (int f = 0; f < g.n_nodes(); ++f)
            se2 += (g.node(f).head<2>() - run.truth.row(run.keyframes[f]).head<2>().transpose()).squaredNorm();
        CHECK(std::sqrt(se2 / g.n_nodes()) < 0.1);  // absolute trajectory error below 10 cm
    }
}
