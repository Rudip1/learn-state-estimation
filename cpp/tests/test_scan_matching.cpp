// Chapter 10 tests, scan matching: the closed-form alignment is exact without noise and optimal with noise; the
// scan simulator measures the right distances; both ICP variants recover a known relative pose, point-to-line in
// fewer iterations; the match covariance shows the corridor degeneracy and agrees with Monte Carlo in a room.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "state_estimation/scan_matching.hpp"

using namespace state_estimation;
using Catch::Approx;

namespace {
constexpr double kPi = 3.14159265358979323846;

/// A 10 m x 6 m room with a pillar and an inner wall: enough structure in every direction.
std::vector<Segment> room() {
    auto s = [](double ax, double ay, double bx, double by) { return Segment{Vector2d(ax, ay), Vector2d(bx, by)}; };
    return {s(-5, -3, 5, -3), s(5, -3, 5, 3),   s(5, 3, -5, 3),     s(-5, 3, -5, -3),
            s(1, -3, 1, -0.5), s(-2, 1, -1, 1), s(-1, 1, -1, 2),    s(-1, 2, -2, 2),
            s(-2, 2, -2, 1),   s(3, 1.5, 4, 2.5)};
}

double cost(const Pose2& x, const MatrixXd& P, const MatrixXd& Q) {
    return (transform_points(x, P) - Q).squaredNorm();
}
}  // namespace

TEST_CASE("Closed-form alignment is exact without noise", "[ch10]") {
    Rng rng(3);
    MatrixXd P(20, 2);
    for (int i = 0; i < P.rows(); ++i) P.row(i) << rng.uniform(-3, 3), rng.uniform(-3, 3);
    const Pose2 x(0.7, -1.2, 2.5);
    const MatrixXd Q = transform_points(x, P);
    const Pose2 x_hat = align_point_sets(P, Q);
    CHECK((x_hat.head<2>() - x.head<2>()).norm() < 1e-12);
    CHECK(std::abs(wrap_angle(x_hat.z() - x.z())) < 1e-12);
}

TEST_CASE("Closed-form alignment minimises the squared error with noise", "[ch10]") {
    Rng rng(4);
    MatrixXd P(30, 2);
    for (int i = 0; i < P.rows(); ++i) P.row(i) << rng.uniform(-3, 3), rng.uniform(-3, 3);
    MatrixXd Q = transform_points(Pose2(0.3, 0.1, -0.4), P);
    for (int i = 0; i < Q.rows(); ++i) Q.row(i) += 0.05 * rng.normal_vector(2).transpose();
    const Pose2 x_hat = align_point_sets(P, Q);
    const double c0 = cost(x_hat, P, Q);
    for (int k = 0; k < 3; ++k)
        for (double h : {-1e-4, 1e-4}) {
            Pose2 x = x_hat;
            x(k) += h;
            CHECK(cost(x, P, Q) > c0);
        }
}

TEST_CASE("The scan simulator measures distances to the walls", "[ch10]") {
    Rng rng(0);
    const MatrixXd scan = simulate_scan(Pose2(0.0, 0.0, 0.0), room(), 3, kPi, 20.0, 0.0, rng);
    REQUIRE(scan.rows() == 3);
    // Bearings -90, 0 and +90 deg from the origin: the walls y = -3, x = 5 and y = 3 (the inner walls are missed).
    CHECK((scan.row(0).transpose() - Vector2d(0.0, -3.0)).norm() < 1e-12);
    CHECK((scan.row(1).transpose() - Vector2d(5.0, 0.0)).norm() < 1e-12);
    CHECK((scan.row(2).transpose() - Vector2d(0.0, 3.0)).norm() < 1e-12);
    // Seen from a rotated pose the points stay in the robot frame: facing +y, the wall y = 3 is straight ahead.
    const MatrixXd s2 = simulate_scan(Pose2(0.0, 0.0, kPi / 2), room(), 1, 0.0, 20.0, 0.0, rng);
    REQUIRE(s2.rows() == 1);
    CHECK((s2.row(0).transpose() - Vector2d(3.0, 0.0)).norm() < 1e-12);
    // Beams that hit nothing within range are dropped.
    CHECK(simulate_scan(Pose2(0.0, 0.0, 0.0), room(), 3, kPi, 2.5, 0.0, rng).rows() == 0);
}

TEST_CASE("Normals of points on a line are perpendicular to it and face the sensor", "[ch10]") {
    MatrixXd pts(11, 2);
    for (int i = 0; i < 11; ++i) pts.row(i) << -1.0 + 0.2 * i, 1.5;
    const MatrixXd n = estimate_normals(pts, 4);
    for (int i = 0; i < n.rows(); ++i) {
        CHECK(n(i, 0) == Approx(0.0).margin(1e-12));
        CHECK(n(i, 1) == Approx(-1.0));
    }
}

TEST_CASE("Both ICP variants recover the relative pose of two scans", "[ch10]") {
    Rng rng(5);
    const Pose2 xa(-3.0, -1.0, 0.2), xb(-2.4, -0.7, 0.45);
    const MatrixXd ref = simulate_scan(xa, room(), 360, 2 * kPi - 0.01, 8.0, 0.01, rng);
    const MatrixXd cur = simulate_scan(xb, room(), 360, 2 * kPi - 0.01, 8.0, 0.01, rng);
    const Pose2 truth = between(xa, xb);
    const Pose2 guess = truth + Pose2(0.2, -0.15, 0.1);
    const IcpResult p2p = icp_point_to_point(ref, cur, guess);
    const IcpResult p2l = icp_point_to_line(ref, cur, guess);
    for (const IcpResult* r : {&p2p, &p2l}) {
        CHECK(r->converged);
        CHECK((r->pose.head<2>() - truth.head<2>()).norm() < 0.02);
        CHECK(std::abs(wrap_angle(r->pose.z() - truth.z())) < 0.005);
    }
    CHECK(p2l.iterations < p2p.iterations);
    CHECK(p2l.rmse < 0.03);
}

TEST_CASE("In a corridor the match is unconstrained along the corridor", "[ch10]") {
    auto s = [](double ax, double ay, double bx, double by) { return Segment{Vector2d(ax, ay), Vector2d(bx, by)}; };
    const std::vector<Segment> corridor = {s(-50, -1, 50, -1), s(-50, 1, 50, 1)};
    const Pose2 x(0.3, 0.1, 0.05);
    auto smallest_eigen = [](const Matrix3d& H) {
        Eigen::SelfAdjointEigenSolver<Matrix3d> eig(H);
        return std::make_pair(eig.eigenvalues()(0) / eig.eigenvalues()(2), eig.eigenvectors().col(0));
    };
    {   // Noise-free walls: J^T J has an exactly zero eigenvalue, and its eigenvector is the corridor axis.
        Rng rng(6);
        const MatrixXd ref = simulate_scan(Pose2::Zero(), corridor, 360, 2 * kPi - 0.01, 6.0, 0.0, rng);
        const MatrixXd cur = simulate_scan(x, corridor, 360, 2 * kPi - 0.01, 6.0, 0.0, rng);
        const auto [ratio, v] = smallest_eigen(icp_point_to_line(ref, cur, x).hessian);
        CHECK(ratio < 1e-12);
        CHECK(std::abs(v.x()) == Approx(1.0).margin(1e-6));
        // Without a radius limit, normals at the sparse far ends mix points of both walls and invent information
        // along the corridor: the zero eigenvalue disappears.
        IcpOptions no_radius;
        no_radius.normal_neighbours = 10;
        no_radius.normal_radius = 1e9;
        CHECK(smallest_eigen(icp_point_to_line(ref, cur, x, no_radius).hessian).first > 1e-4);
    }
    {   // Noisy ranges: across the corridor and in heading the match is sharp, along it the variance is far larger.
        Rng rng(6);
        const MatrixXd ref = simulate_scan(Pose2::Zero(), corridor, 360, 2 * kPi - 0.01, 6.0, 0.01, rng);
        const MatrixXd cur = simulate_scan(x, corridor, 360, 2 * kPi - 0.01, 6.0, 0.01, rng);
        const IcpResult r = icp_point_to_line(ref, cur, x);
        CHECK(std::abs(r.pose.y() - 0.1) < 0.01);
        CHECK(std::abs(r.pose.z() - 0.05) < 0.005);
        CHECK(r.covariance(0, 0) > 30.0 * r.covariance(1, 1));
    }
}

TEST_CASE("Point-to-line covariance agrees with Monte Carlo in a structured room", "[ch10]") {
    const Pose2 xa(2.0, -1.5, 1.0), xb(2.3, -1.2, 1.2);
    const Pose2 truth = between(xa, xb);
    Rng rng(7);
    const int runs = 200;
    double nees = 0.0;
    for (int k = 0; k < runs; ++k) {
        const MatrixXd ref = simulate_scan(xa, room(), 360, 2 * kPi - 0.01, 8.0, 0.01, rng);
        const MatrixXd cur = simulate_scan(xb, room(), 360, 2 * kPi - 0.01, 8.0, 0.01, rng);
        const IcpResult r = icp_point_to_line(ref, cur, truth + Pose2(0.05, -0.05, 0.03));
        Vector3d e = r.pose - truth;
        e.z() = wrap_angle(e.z());
        nees += e.dot(r.covariance.ldlt().solve(e));
    }
    nees /= runs;
    INFO("average NEES " << nees << " (consistent: 3)");
    CHECK(nees > 1.0);
    CHECK(nees < 8.0);
}
