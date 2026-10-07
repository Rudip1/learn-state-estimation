// Chapter 3 tests: hand-worked measurements, inverse models as exact inverses, Jacobians against numerical
// differentiation, noise statistics of the simulated sensor and the observability (rank) statements of
// section 3.7.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "state_estimation/measurement_models.hpp"

using namespace state_estimation;
using Catch::Approx;

namespace {
constexpr double kPi = 3.14159265358979323846;
}

TEST_CASE("Range-bearing: worked example of section 3.2", "[ch03]") {
    const Pose2 x(1.0, 1.0, kPi / 2.0);
    const Vector2d ahead = range_bearing(x, Vector2d(1.0, 3.0));
    CHECK(ahead(0) == Approx(2.0));
    CHECK(ahead(1) == Approx(0.0).margin(1e-15));
    const Vector2d right = range_bearing(x, Vector2d(3.0, 1.0));
    CHECK(right(0) == Approx(2.0));
    CHECK(right(1) == Approx(-kPi / 2.0));
    // Behind the robot the bearing is wrapped, not 3pi/2.
    const Vector2d behind = range_bearing(Pose2(0.0, 0.0, kPi - 0.1), Vector2d(1.0, -0.2));
    CHECK(std::abs(behind(1)) <= kPi);
}

TEST_CASE("Inverse models invert the direct models", "[ch03]") {
    const Pose2 x(-0.5, 2.0, 2.4);
    const Vector2d m(3.0, -1.0);
    CHECK((range_bearing_inverse(x, range_bearing(x, m)) - m).norm() < 1e-12);
    CHECK((transform_point(x, cartesian_feature(x, m)) - m).norm() < 1e-12);
    // The Cartesian feature equals the translation part of (-)x (+) [m, 0].
    const Pose2 rel = compose(inverse(x), Pose2(m.x(), m.y(), 0.0));
    CHECK((cartesian_feature(x, m) - rel.head<2>()).norm() < 1e-12);
    // Polar and Cartesian observations describe the same point.
    CHECK((polar_to_cartesian(range_bearing(x, m)) - cartesian_feature(x, m)).norm() < 1e-12);
}

TEST_CASE("Measurement Jacobians match numerical differentiation", "[ch03]") {
    const Pose2 x(0.3, -1.2, 0.8);
    const Vector2d m(2.5, 1.5), z(3.0, -0.4), lever(0.4, 0.1);
    auto num = [](auto f, const VectorXd& v) { return numerical_jacobian(f, v, 1e-6); };
    auto close = [](const MatrixXd& a, const MatrixXd& b) { return (a - b).norm() < 1e-7; };
    CHECK(close(num([&](const VectorXd& p) -> VectorXd { return range_bearing(Pose2(p), m); }, x),
                range_bearing_jacobian_pose(x, m)));
    CHECK(close(num([&](const VectorXd& l) -> VectorXd { return range_bearing(x, Vector2d(l)); }, m),
                range_bearing_jacobian_landmark(x, m)));
    CHECK(close(num([&](const VectorXd& p) -> VectorXd { return range_bearing_inverse(Pose2(p), z); }, x),
                range_bearing_inverse_jacobian_pose(x, z)));
    CHECK(close(num([&](const VectorXd& q) -> VectorXd { return range_bearing_inverse(x, Vector2d(q)); }, z),
                range_bearing_inverse_jacobian_measurement(x, z)));
    CHECK(close(num([&](const VectorXd& p) -> VectorXd { return cartesian_feature(Pose2(p), m); }, x),
                cartesian_feature_jacobian_pose(x, m)));
    CHECK(close(num([&](const VectorXd& l) -> VectorXd { return cartesian_feature(x, Vector2d(l)); }, m),
                cartesian_feature_jacobian_landmark(x)));
    CHECK(close(num([&](const VectorXd& p) -> VectorXd { return position_fix(Pose2(p), lever); }, x),
                position_fix_jacobian(x, lever)));
}

TEST_CASE("Residuals wrap angles; the likelihood is Gaussian", "[ch03]") {
    VectorXd z(2), zh(2);
    z << 2.0, kPi - 0.01;
    zh << 2.1, -kPi + 0.01;
    const VectorXd r = measurement_residual(z, zh, {1});
    CHECK(r(0) == Approx(-0.1));
    CHECK(r(1) == Approx(-0.02));
    MatrixXd R = MatrixXd::Identity(2, 2) * 0.01;
    const double expected = -std::log(2.0 * kPi * 0.01) - 0.5 * (0.01 + 0.0004) / 0.01;
    CHECK(measurement_log_likelihood(z, zh, R, {1}) == Approx(expected));
}

TEST_CASE("Simulated sensor: field of view, range limit and noise statistics", "[ch03]") {
    RangeBearingSensor sensor;
    sensor.max_range = 5.0;
    sensor.fov = kPi / 2.0;  // +-45 degrees
    const Pose2 x(0.0, 0.0, 0.0);
    MatrixXd map(4, 2);
    map << 3.0, 0.0,  // ahead: visible
        3.0, 2.9,     // 44 degrees: visible
        0.0, 3.0,     // 90 degrees: outside the field of view
        6.0, 0.0;     // too far
    Rng rng(1);
    const auto obs = sensor.observe(x, map, rng);
    REQUIRE(obs.size() == 2);
    CHECK(obs[0].landmark == 0);
    CHECK(obs[1].landmark == 1);

    sensor.fov = 2.0 * kPi;
    sensor.max_range = 100.0;
    MatrixXd one(1, 2);
    one << 4.0, 3.0;
    const Vector2d truth = range_bearing(x, Vector2d(4.0, 3.0));
    const int n = 50000;
    MatrixXd samples(n, 2);
    for (int i = 0; i < n; ++i) samples.row(i) = sensor.observe(x, one, rng)[0].z.transpose();
    const Gaussian est = sample_mean_cov(samples);
    CHECK((est.mean - truth).norm() < 2e-3);
    CHECK((est.cov - MatrixXd(sensor.R())).norm() < 0.03 * sensor.R().norm());
}

TEST_CASE("Polar noise becomes range-dependent Cartesian noise", "[ch03]") {
    const Matrix2d R = Vector2d(0.01, 0.0004).asDiagonal();  // 10 cm, 20 mrad
    // At bearing 0 the Cartesian covariance is diag(sigma_r^2, (r sigma_phi)^2).
    for (double r : {1.0, 10.0}) {
        const Matrix2d C = polar_to_cartesian_covariance(Vector2d(r, 0.0), R);
        CHECK(C(0, 0) == Approx(0.01));
        CHECK(C(1, 1) == Approx(r * r * 0.0004));
    }
}

TEST_CASE("Observability: rank of the Fisher information (section 3.7)", "[ch03]") {
    const Pose2 x(1.0, 2.0, 0.3);
    const Matrix2d R = Vector2d(0.01, 0.0004).asDiagonal();
    auto rank = [](const MatrixXd& I) {
        Eigen::SelfAdjointEigenSolver<MatrixXd> es(I);
        return static_cast<int>((es.eigenvalues().array() > 1e-9 * es.eigenvalues().maxCoeff()).count());
    };
    // One landmark: two equations, rank 2. Two landmarks: the full pose.
    const Vector2d m1(4.0, 3.0), m2(0.0, 6.0);
    MatrixXd H1 = range_bearing_jacobian_pose(x, m1);
    CHECK(rank(fisher_information(H1, R)) == 2);
    MatrixXd H2(4, 3);
    H2 << range_bearing_jacobian_pose(x, m1), range_bearing_jacobian_pose(x, m2);
    MatrixXd R2 = MatrixXd::Zero(4, 4);
    R2.topLeftCorner<2, 2>() = R;
    R2.bottomRightCorner<2, 2>() = R;
    CHECK(rank(fisher_information(H2, R2)) == 3);
    // A position fix at the robot centre does not observe the heading; range-only to two landmarks
    // gives rank 2 and leaves the heading free as well.
    CHECK(rank(fisher_information(position_fix_jacobian(x, Vector2d::Zero()), Matrix2d::Identity())) == 2);
    MatrixXd Hr(2, 3);
    Hr << range_bearing_jacobian_pose(x, m1).row(0), range_bearing_jacobian_pose(x, m2).row(0);
    CHECK(rank(fisher_information(Hr, Matrix2d::Identity() * 0.01)) == 2);
    // A compass alone observes only the heading.
    MatrixXd Hc(1, 3);
    Hc << 0.0, 0.0, 1.0;
    CHECK(rank(fisher_information(Hc, MatrixXd::Identity(1, 1))) == 1);
    // Two position fixes, before and after driving 1 m straight, observe the heading too: the second fix
    // sees x (+) u, whose Jacobian with respect to x is J1(x, u).
    const Pose2 u(1.0, 0.0, 0.0);
    MatrixXd Hf(4, 3);
    Hf << position_fix_jacobian(x, Vector2d::Zero()),
        position_fix_jacobian(compose(x, u), Vector2d::Zero()) * compose_jacobian_1(x, u);
    CHECK(rank(fisher_information(Hf, MatrixXd::Identity(4, 4))) == 3);
}
