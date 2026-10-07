// Chapter 2 tests: pose algebra against homogeneous matrices, analytic Jacobians against numerical
// differentiation, motion models against closed forms and fine numerical integration, and the
// dead-reckoning covariance against the closed-form drift law of section 2.6.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "state_estimation/motion_models.hpp"
#include "state_estimation/simulation.hpp"

using namespace state_estimation;
using Catch::Approx;

namespace {
constexpr double kPi = 3.14159265358979323846;

bool pose_near(const Pose2& a, const Pose2& b, double tol = 1e-12) {
    return (a.head<2>() - b.head<2>()).norm() < tol && std::abs(wrap_angle(a.z() - b.z())) < tol;
}
}  // namespace

TEST_CASE("wrap_angle maps angles into the half-open interval from -pi to pi", "[ch02]") {
    CHECK(wrap_angle(kPi) == Approx(kPi));
    CHECK(wrap_angle(-kPi) == Approx(kPi));
    CHECK(wrap_angle(3.0 * kPi / 2.0) == Approx(-kPi / 2.0));
    CHECK(wrap_angle(0.1 + 20.0 * kPi) == Approx(0.1));
}

TEST_CASE("Compounding and inversion agree with homogeneous matrices", "[ch02]") {
    const Pose2 a(1.0, 2.0, 0.7), b(-0.5, 0.3, -2.0);
    CHECK(pose_near(compose(a, b), from_matrix(to_matrix(a) * to_matrix(b))));
    CHECK(pose_near(inverse(a), from_matrix(to_matrix(a).inverse())));
    CHECK(pose_near(compose(a, inverse(a)), Pose2::Zero()));
    CHECK(pose_near(compose(inverse(a), a), Pose2::Zero()));
    CHECK(pose_near(between(a, compose(a, b)), b));
    const Pose2 c(0.2, -1.0, 2.5);
    CHECK(pose_near(compose(compose(a, b), c), compose(a, compose(b, c))));  // associativity
    const Vector2d p(0.4, -0.9);
    CHECK((transform_point(a, p) - (to_matrix(a) * p.homogeneous()).head<2>()).norm() < 1e-12);
    CHECK((inverse_transform_point(a, transform_point(a, p)) - p).norm() < 1e-12);
}

TEST_CASE("Worked compounding example of section 2.1", "[ch02]") {
    // Robot at (2, 1) facing 90 degrees; a second pose 1 m ahead and 0.5 m to its left, turned 90 deg.
    const Pose2 a(2.0, 1.0, kPi / 2.0), b(1.0, 0.5, kPi / 2.0);
    CHECK(pose_near(compose(a, b), Pose2(1.5, 2.0, kPi), 1e-12));
    CHECK(pose_near(inverse(a), Pose2(-1.0, 2.0, -kPi / 2.0), 1e-12));
}

TEST_CASE("Pose Jacobians match numerical differentiation", "[ch02]") {
    const Pose2 a(1.0, -2.0, 0.4), b(0.7, 0.2, 1.1);
    const Vector2d p(1.5, -0.3);
    auto num = [](auto f, const VectorXd& x) { return numerical_jacobian(f, x, 1e-6); };
    const MatrixXd J1 = num([&](const VectorXd& v) -> VectorXd { return compose(Pose2(v), b); }, a);
    const MatrixXd J2 = num([&](const VectorXd& v) -> VectorXd { return compose(a, Pose2(v)); }, b);
    const MatrixXd Ji = num([&](const VectorXd& v) -> VectorXd { return inverse(Pose2(v)); }, a);
    const MatrixXd Jp = num([&](const VectorXd& v) -> VectorXd { return transform_point(Pose2(v), p); }, a);
    const MatrixXd Jip =
        num([&](const VectorXd& v) -> VectorXd { return inverse_transform_point(Pose2(v), p); }, a);
    CHECK((J1 - MatrixXd(compose_jacobian_1(a, b))).norm() < 1e-8);
    CHECK((J2 - MatrixXd(compose_jacobian_2(a, b))).norm() < 1e-8);
    CHECK((Ji - MatrixXd(inverse_jacobian(a))).norm() < 1e-8);
    CHECK((Jp - MatrixXd(transform_point_jacobian_pose(a, p))).norm() < 1e-8);
    CHECK((Jip - MatrixXd(inverse_transform_point_jacobian_pose(a, p))).norm() < 1e-8);
}

TEST_CASE("Differential-drive kinematics", "[ch02]") {
    DifferentialDrive dd{0.1, 0.5, 0};
    // Both wheels at 10 rad/s: straight at 1 m/s. Opposite wheels: turn on the spot.
    CHECK(dd.wheel_to_body(10.0, 10.0).isApprox(Vector2d(1.0, 0.0)));
    CHECK(dd.wheel_to_body(-5.0, 5.0).isApprox(Vector2d(0.0, 2.0)));
    const Vector2d w = dd.body_to_wheel(0.8, -0.6);
    CHECK(dd.wheel_to_body(w(0), w(1)).isApprox(Vector2d(0.8, -0.6)));
    dd.ticks_per_revolution = 1024;
    CHECK(dd.ticks_to_distance(1024) == Approx(2.0 * kPi * 0.1));
}

TEST_CASE("Arc displacement: closed form, continuity at phi = 0 and Jacobian", "[ch02]") {
    // A quarter circle of radius 2: s = pi, phi = pi/2 ends at (2, 2) in the start frame.
    const Pose2 d = arc_displacement(kPi, kPi / 2.0);
    CHECK(d.x() == Approx(2.0));
    CHECK(d.y() == Approx(2.0));
    // The series branch agrees with the closed form just below the switch point.
    const double phi0 = 0.99e-4;
    const Pose2 series = arc_displacement(1.3, phi0);
    CHECK(series.x() == Approx(1.3 * std::sin(phi0) / phi0).epsilon(1e-14));
    CHECK(series.y() == Approx(1.3 * (1.0 - std::cos(phi0)) / phi0).epsilon(1e-7));
    for (double phi : {0.0, 5e-5, 0.3, -1.2, 2.9}) {
        const MatrixXd Jn = numerical_jacobian(
            [](const VectorXd& u) -> VectorXd { return arc_displacement(u(0), u(1)); }, Vector2d(0.8, phi),
            1e-6);
        CHECK((Jn - MatrixXd(arc_displacement_jacobian(0.8, phi))).norm() < 1e-7);
    }
}

TEST_CASE("Velocity motion model: closed form and fine Euler integration", "[ch02]") {
    const Pose2 x(1.0, 2.0, 0.3);
    // Full circle returns to the start.
    const double v = 0.5, omega = 0.25, T = 2.0 * kPi / omega;
    CHECK(pose_near(velocity_motion(x, v, omega, T), x, 1e-12));
    // Compare with Euler integration using a tiny step.
    const double dt = 1.7;
    Pose2 e = x;
    const int n = 200000;
    for (int i = 0; i < n; ++i) {
        e.x() += v * std::cos(e.z()) * dt / n;
        e.y() += v * std::sin(e.z()) * dt / n;
        e.z() += omega * dt / n;
    }
    CHECK(pose_near(velocity_motion(x, v, omega, dt), e, 1e-5));
    // Jacobians.
    const MatrixXd Jx = numerical_jacobian(
        [&](const VectorXd& p) -> VectorXd { return velocity_motion(Pose2(p), v, omega, dt); }, x);
    const MatrixXd Ju = numerical_jacobian(
        [&](const VectorXd& u) -> VectorXd { return velocity_motion(x, u(0), u(1), dt); }, Vector2d(v, omega));
    CHECK((Jx - MatrixXd(velocity_motion_jacobian_pose(x, v, omega, dt))).norm() < 1e-7);
    CHECK((Ju - MatrixXd(velocity_motion_jacobian_control(x, v, omega, dt))).norm() < 1e-6);
}

TEST_CASE("Odometry model: decomposition, sampling statistics and density", "[ch02]") {
    const Pose2 o0(0.0, 0.0, 0.2), o1(1.0, 0.5, 0.9);
    const Vector3d rtr = odometry_to_rtr(o0, o1);
    CHECK(pose_near(apply_rtr(o0, rtr), o1, 1e-12));
    CHECK(rtr(1) == Approx(std::hypot(1.0, 0.5)));
    // Pure rotation: no spurious rot1/rot2 split.
    const Vector3d spin = odometry_to_rtr(o0, Pose2(0.0, 0.0, 1.0));
    CHECK(spin(0) == Approx(0.8));
    CHECK(spin(1) == 0.0);
    CHECK(spin(2) == Approx(0.0).margin(1e-15));

    // Sampling from the same start pose: the spread of the travelled distance has the model variance.
    const std::array<double, 4> alpha{0.01, 0.002, 0.02, 0.005};
    Rng rng(11);
    const int n = 100000;
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < n; ++i) {
        const Pose2 s = sample_odometry_motion(o0, o0, o1, alpha, rng);
        const double t = std::hypot(s.x() - o0.x(), s.y() - o0.y());
        sum += t;
        sum2 += t * t;
    }
    const double var_t = sum2 / n - (sum / n) * (sum / n);
    const double model = alpha[2] * rtr(1) * rtr(1) + alpha[3] * (rtr(0) * rtr(0) + rtr(2) * rtr(2));
    CHECK(var_t == Approx(model).epsilon(0.03));

    // The density is maximal at the noise-free pose and decreases away from it.
    const double p0 = odometry_motion_probability(o1, o0, o0, o1, alpha);
    CHECK(p0 > odometry_motion_probability(Pose2(1.05, 0.5, 0.9), o0, o0, o1, alpha));
    CHECK(p0 > odometry_motion_probability(Pose2(1.0, 0.5, 0.95), o0, o0, o1, alpha));
}

TEST_CASE("Dead-reckoning covariance follows the closed-form drift law", "[ch02]") {
    // Straight line, step s, heading noise only: Var[y_n] = s^2 sigma^2 (n-1) n (2n-1) / 6 (section 2.6).
    const double s = 0.1, sigma = 0.01;
    Matrix3d Q = Matrix3d::Zero();
    Q(2, 2) = sigma * sigma;
    DeadReckoning dr(Pose2::Zero(), Matrix3d::Zero());
    const int n = 50;
    for (int k = 0; k < n; ++k) dr.predict_displacement(Pose2(s, 0.0, 0.0), Q);
    const double expected = s * s * sigma * sigma * (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
    CHECK(dr.covariance()(1, 1) == Approx(expected).epsilon(1e-12));
    CHECK(dr.covariance()(2, 2) == Approx(n * sigma * sigma).epsilon(1e-12));
    CHECK(dr.covariance()(0, 0) == Approx(0.0).margin(1e-20));
}

TEST_CASE("Dead-reckoning covariance agrees with Monte Carlo spread", "[ch02]") {
    // Small noise, so that the first-order propagation is accurate (with larger noise the second-order
    // terms make the true spread larger than predicted; see the notebook).
    DifferentialDrive dd{0.1, 0.5, 0};
    const double dt = 0.1, var_per_meter = 1e-5;
    const MatrixXd u = circle_controls(0.5, 2.0, 120);
    const MatrixXd truth = integrate_controls(Pose2::Zero(), u, dt);
    Rng rng(3);
    const int runs = 10000;
    MatrixXd finals(runs, 3);
    DeadReckoningRun last;
    for (int r = 0; r < runs; ++r) {
        const MatrixXd w = simulate_wheel_travel(u, dt, dd, var_per_meter, rng);
        last = run_dead_reckoning(Pose2::Zero(), Matrix3d::Zero(), w, dd, var_per_meter);
        finals.row(r) = last.poses.bottomRows<1>();
    }
    const Gaussian mc = sample_mean_cov(finals.leftCols<2>());
    const Matrix2d predicted = last.covariances.back().topLeftCorner<2, 2>();
    CHECK((mc.mean - truth.bottomRows<1>().leftCols<2>().transpose()).norm() < 0.01);
    CHECK((mc.cov - MatrixXd(predicted)).norm() < 0.05 * predicted.norm());
}
