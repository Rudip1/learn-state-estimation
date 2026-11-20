// Chapter 6 tests: the EKF reduces to the Kalman filter on linear models, the iterated EKF reaches the MAP
// estimate found by brute force, and map-based localisation is consistent in Monte Carlo simulation.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "state_estimation/ekf.hpp"
#include "state_estimation/simulation.hpp"

using namespace state_estimation;
using Catch::Approx;

namespace {
constexpr double kPi = 3.14159265358979323846;
}

TEST_CASE("EKF equals the Kalman filter on a linear model", "[ch06]") {
    const LinearModel m = constant_velocity_model(2, 0.1, 0.3);
    MatrixXd H = MatrixXd::Zero(2, 4);
    H(0, 0) = H(1, 1) = 1.0;
    const MatrixXd R = MatrixXd::Identity(2, 2) * 0.1;
    VectorXd x0(4);
    x0 << 0.0, 0.0, 1.0, -0.5;
    const MatrixXd P0 = MatrixXd::Identity(4, 4);
    Rng rng(2);
    const LinearRun run = simulate_linear_system(m.F, m.Q, H, R, x0, 40, rng);
    KalmanFilter kf(x0, P0);
    ExtendedKalmanFilter ekf(x0, P0);
    MeasurementFunction lin{[&](const VectorXd& x) -> VectorXd { return H * x; },
                            [&](const VectorXd&) -> MatrixXd { return H; },
                            {}};
    for (int k = 0; k < 40; ++k) {
        kf.predict(m.F, m.Q);
        ekf.predict([&](const VectorXd& x) -> VectorXd { return m.F * x; }, m.F, MatrixXd::Identity(4, 4), m.Q);
        const VectorXd z = run.measurements.row(k).transpose();
        const double nis_kf = kf.update(z, H, R).nis;
        const double nis_ekf = ekf.update(z, lin, R).nis;
        CHECK(nis_kf == Approx(nis_ekf).epsilon(1e-10));
    }
    CHECK((kf.x() - ekf.x()).norm() < 1e-10);
    CHECK((kf.P() - ekf.P()).norm() < 1e-10);
}

TEST_CASE("Iterated EKF converges to the MAP estimate of a nonlinear measurement", "[ch06]") {
    // Prior x ~ N(1, 1); measurement z = x^2 + v, v ~ N(0, 0.01), z = 4. The MAP estimate minimises
    // (x - 1)^2 / 1 + (4 - x^2)^2 / 0.01; it is found here by a fine grid search followed by Newton steps.
    const double R = 0.01, z = 4.0;
    auto cost_grad = [&](double x) { return 2.0 * (x - 1.0) - 2.0 * (z - x * x) * 2.0 * x / R; };
    auto cost_hess = [&](double x) { return 2.0 + (12.0 * x * x - 4.0 * z) / R; };
    double best = 0.0, best_cost = 1e300;
    for (double x = 0.0; x <= 4.0; x += 1e-4) {
        const double c = (x - 1.0) * (x - 1.0) + (z - x * x) * (z - x * x) / R;
        if (c < best_cost) {
            best_cost = c;
            best = x;
        }
    }
    for (int i = 0; i < 20; ++i) best -= cost_grad(best) / cost_hess(best);

    MeasurementFunction sq{[](const VectorXd& x) -> VectorXd { return VectorXd::Constant(1, x(0) * x(0)); },
                           [](const VectorXd& x) -> MatrixXd { return MatrixXd::Constant(1, 1, 2.0 * x(0)); },
                           {}};
    ExtendedKalmanFilter ekf(VectorXd::Constant(1, 1.0), MatrixXd::Constant(1, 1, 1.0));
    ExtendedKalmanFilter iekf = ekf;
    ekf.update(VectorXd::Constant(1, z), sq, MatrixXd::Constant(1, 1, R), 1);
    iekf.update(VectorXd::Constant(1, z), sq, MatrixXd::Constant(1, 1, R), 30);
    CHECK(iekf.x()(0) == Approx(best).epsilon(1e-9));
    CHECK(std::abs(ekf.x()(0) - best) > 0.1);  // one linearisation at x = 1 overshoots badly
}

TEST_CASE("Compass update on an uncorrelated heading is a scalar Kalman update", "[ch06]") {
    Matrix3d P = Vector3d(0.5, 0.4, 0.09).asDiagonal();
    EkfLocalization loc(Pose2(1.0, 2.0, 0.3), P);
    loc.update_compass(0.4, 0.01);
    const double k = 0.09 / (0.09 + 0.01);
    CHECK(loc.pose().z() == Approx(0.3 + k * 0.1));
    CHECK(loc.covariance()(2, 2) == Approx(0.09 * 0.01 / 0.1));
    CHECK(loc.covariance()(0, 0) == Approx(0.5));
    // Wrapping: a reading at -pi + 0.01 for a heading at pi - 0.01 is a 0.02 rad innovation.
    EkfLocalization w(Pose2(0.0, 0.0, kPi - 0.01), P);
    const Innovation inn = w.update_compass(-kPi + 0.01, 0.01);
    CHECK(inn.nu(0) == Approx(0.02));
}

TEST_CASE("EKF localisation is consistent and bounded", "[ch06]") {
    MatrixXd landmarks(6, 2);
    landmarks << 0.0, 4.0, 4.0, 4.0, 4.0, 0.0, -4.0, 0.0, -4.0, -4.0, 0.0, -4.0;
    const DifferentialDrive dd{0.1, 0.5, 0};
    RangeBearingSensor sensor{6.0, 2.0 * kPi, 0.05, 0.01};
    const double dt = 0.1, k = 1e-4;
    const MatrixXd u = figure_eight_controls(2.0, 200, dt);
    const Pose2 x0(0.0, 0.0, 0.0);
    const Matrix3d P0 = Vector3d(0.01, 0.01, 0.001).asDiagonal();
    Rng rng(21);
    const int runs = 30, steps = static_cast<int>(u.rows());
    VectorXd sum_nees = VectorXd::Zero(steps);
    double max_err = 0.0;
    for (int r = 0; r < runs; ++r) {
        const Pose2 start = sample_gaussian(x0, P0, 1, rng).row(0).transpose();
        const LandmarkRun run = simulate_landmark_run(start, u, dt, dd, k, sensor, landmarks, 5, rng);
        EkfLocalization ekf(x0, P0);
        for (int s = 0; s < steps; ++s) {
            const double dl = run.wheel_travel(s, 0), dr = run.wheel_travel(s, 1);
            ekf.predict_wheels(dd, dl, dr, k * std::abs(dl), k * std::abs(dr));
            ekf.update_landmarks(run.scans[s], landmarks, sensor.R());
            Vector3d e = run.truth.row(s + 1).transpose() - ekf.pose();
            e.z() = wrap_angle(e.z());
            sum_nees(s) += e.dot(ekf.covariance().ldlt().solve(e));
            max_err = std::max(max_err, e.head<2>().norm());
        }
    }
    const double mean_nees = sum_nees.sum() / (runs * steps);
    CHECK(mean_nees == Approx(3.0).epsilon(0.2));
    const Eigen::Vector2d b = average_chi2_bounds(3, runs);
    int inside = 0;
    for (int s = 0; s < steps; ++s) inside += (sum_nees(s) / runs >= b(0) && sum_nees(s) / runs <= b(1));
    CHECK(inside >= 0.8 * steps);
    CHECK(max_err < 0.5);  // the landmarks keep the error bounded
}
