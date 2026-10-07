// Chapter 9 tests: EKF SLAM with a perfectly known map reproduces EKF localisation; state augmentation matches
// Monte Carlo; the covariance properties of section 9.6 (monotone landmark uncertainty, lower bound set by the
// initial robot uncertainty, correlation of landmarks); JCBB association reproduces the known correspondences.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "state_estimation/ekf.hpp"
#include "state_estimation/ekf_slam.hpp"
#include "state_estimation/simulation.hpp"

using namespace state_estimation;
using Catch::Approx;

namespace {
constexpr double kPi = 3.14159265358979323846;

MatrixXd square_map() {
    MatrixXd m(6, 2);
    m << 0.0, 4.5, 3.5, 3.0, 3.0, -3.5, -3.0, -3.0, -3.5, 2.5, 0.5, -5.0;
    return m;
}
}  // namespace

TEST_CASE("SLAM with a perfectly known map equals EKF localisation", "[ch09]") {
    const MatrixXd map = square_map();
    const DifferentialDrive dd{0.1, 0.5, 0};
    RangeBearingSensor sensor{4.0, kPi, 0.05, 0.02};
    const double k = 2e-4;
    const Matrix3d P0 = Vector3d(1e-3, 1e-3, 1e-4).asDiagonal();
    Rng rng(1);
    const LandmarkRun run =
        simulate_landmark_run(Pose2::Zero(), figure_eight_controls(2.0, 200, 0.1), 0.1, dd, k, sensor, map, 5, rng);
    EkfSlam slam(Pose2::Zero(), P0);
    for (int j = 0; j < map.rows(); ++j) slam.add_known_landmark(map.row(j).transpose(), Matrix2d::Zero());
    EkfLocalization loc(Pose2::Zero(), P0);
    for (int s = 0; s < run.wheel_travel.rows(); ++s) {
        const double dl = run.wheel_travel(s, 0), dr = run.wheel_travel(s, 1);
        slam.predict_wheels(dd, dl, dr, k * std::abs(dl), k * std::abs(dr));
        loc.predict_wheels(dd, dl, dr, k * std::abs(dl), k * std::abs(dr));
        slam.update(run.scans[s], sensor.R());
        loc.update_landmarks(run.scans[s], map, sensor.R());
    }
    CHECK((slam.pose() - loc.pose()).norm() < 1e-9);
    CHECK((slam.pose_covariance() - loc.covariance()).norm() < 1e-9);
    CHECK(slam.covariance().bottomRightCorner(12, 12).norm() < 1e-12);  // the map stays exact
}

TEST_CASE("State augmentation matches Monte Carlo", "[ch09]") {
    const Pose2 x(1.0, -0.5, 0.6);
    const Matrix3d P = (Matrix3d() << 0.04, 0.01, 0.0, 0.01, 0.03, 0.002, 0.0, 0.002, 0.01).finished();
    const Vector2d z(3.0, 0.4);
    const Matrix2d R = Vector2d(0.05 * 0.05, 0.02 * 0.02).asDiagonal();
    EkfSlam slam(x, P);
    slam.add_landmark(z, R);
    // Monte Carlo of [x; g(x, z + v)].
    Rng rng(2);
    const int n = 200000;
    MatrixXd s(n, 5);
    for (int i = 0; i < n; ++i) {
        const Pose2 xi = sample_gaussian(x, P, 1, rng).row(0).transpose();
        const Vector2d zi = sample_gaussian(z, R, 1, rng).row(0).transpose();
        s.row(i) << xi.transpose(), range_bearing_inverse(xi, zi).transpose();
    }
    const Gaussian mc = sample_mean_cov(s);
    CHECK((mc.cov - slam.covariance()).cwiseAbs().maxCoeff() < 0.004);
    // The augmented mean is g(x_hat, z); the true mean is pulled towards the robot by the heading spread:
    // E[m] = t + rho exp(-s^2 / 2) [cos a, sin a] with a = theta + phi, s^2 = Var[theta] + Var[phi].
    const double a = x.z() + z(1), s2 = P(2, 2) + R(1, 1);
    const Vector2d expected = x.head<2>() + z(0) * std::exp(-s2 / 2.0) * Vector2d(std::cos(a), std::sin(a));
    CHECK((mc.mean.tail<2>() - expected).norm() < 3e-3);
    CHECK((slam.landmark(0) - range_bearing_inverse(x, z)).norm() < 1e-12);
}

TEST_CASE("Landmark uncertainty never grows and is bounded below by the initial robot uncertainty", "[ch09]") {
    // Section 9.6: the determinant of any landmark block is non-increasing, and in the limit the absolute landmark
    // uncertainty cannot fall below what the initial robot uncertainty allows; the relative uncertainty of two
    // landmarks becomes much smaller than their absolute uncertainty.
    const MatrixXd map = square_map();
    const DifferentialDrive dd{0.1, 0.5, 0};
    RangeBearingSensor sensor{4.0, kPi, 0.05, 0.02};
    const double k = 2e-4;
    const Matrix3d P0 = Vector3d(0.01, 0.01, 0.0).asDiagonal();  // 10 cm initial position uncertainty
    Rng rng(4);
    const MatrixXd u(MatrixXd(figure_eight_controls(2.0, 200, 0.1)).replicate(3, 1));
    const LandmarkRun run = simulate_landmark_run(Pose2::Zero(), u, 0.1, dd, k, sensor, map, 2, rng);
    EkfSlam slam(Pose2::Zero(), P0);
    std::map<int, double> last_det;
    bool monotone = true;
    for (int s = 0; s < run.wheel_travel.rows(); ++s) {
        const double dl = run.wheel_travel(s, 0), dr = run.wheel_travel(s, 1);
        slam.predict_wheels(dd, dl, dr, k * std::abs(dl), k * std::abs(dr));
        slam.process_known(run.scans[s], sensor.R());
        for (int i = 0; i < slam.n_landmarks(); ++i) {
            const double d = slam.landmark_covariance(i).determinant();
            if (last_det.count(i) && d > last_det[i] * (1.0 + 1e-9)) monotone = false;
            last_det[i] = d;
        }
    }
    CHECK(monotone);
    REQUIRE(slam.n_landmarks() == 6);
    for (int i = 0; i < 6; ++i) {
        // Absolute uncertainty of each landmark: at least the initial robot position uncertainty (0.01 per axis).
        CHECK(slam.landmark_covariance(i).trace() >= 0.95 * 0.02);
    }
    // Relative position of two landmarks is known much better than either absolutely.
    const auto& P = slam.covariance();
    const MatrixXd D = (MatrixXd(2, P.cols()) << MatrixXd::Zero(2, 3), MatrixXd::Identity(2, 2), -MatrixXd::Identity(2, 2),
                        MatrixXd::Zero(2, P.cols() - 7))
                           .finished();
    const double rel = (D * P * D.transpose()).trace();
    CHECK(rel < 0.4 * slam.landmark_covariance(0).trace());
}

TEST_CASE("JCBB association reproduces the known correspondences on a clean run", "[ch09]") {
    const MatrixXd map = square_map();
    const DifferentialDrive dd{0.1, 0.5, 0};
    RangeBearingSensor sensor{4.0, kPi, 0.05, 0.02};
    const double k = 1e-4;
    Rng rng(6);
    const LandmarkRun run =
        simulate_landmark_run(Pose2::Zero(), figure_eight_controls(2.0, 200, 0.1), 0.1, dd, k, sensor, map, 5, rng);
    EkfSlam a(Pose2::Zero(), Matrix3d::Zero()), b(Pose2::Zero(), Matrix3d::Zero());
    std::map<int, int> jcbb_of_true;  // true landmark id -> JCBB state index
    bool consistent = true;
    int dropped = 0, total = 0;
    for (int s = 0; s < run.wheel_travel.rows(); ++s) {
        const double dl = run.wheel_travel(s, 0), dr = run.wheel_travel(s, 1);
        a.predict_wheels(dd, dl, dr, k * std::abs(dl), k * std::abs(dr));
        b.predict_wheels(dd, dl, dr, k * std::abs(dl), k * std::abs(dr));
        a.process_known(run.scans[s], sensor.R());
        const ScanResult r = b.process_jcbb(run.scans[s], sensor.R());
        for (size_t i = 0; i < run.scans[s].size(); ++i) {
            const int truth = run.scans[s][i].landmark;
            ++total;
            dropped += (r.landmark[i] < 0);
            if (r.is_new[i]) {
                if (jcbb_of_true.count(truth)) consistent = false;  // a landmark was duplicated
                jcbb_of_true[truth] = r.landmark[i];
            } else if (r.landmark[i] >= 0 && jcbb_of_true[truth] != r.landmark[i]) {
                consistent = false;
            }
        }
    }
    CHECK(consistent);
    CHECK(b.n_landmarks() == 6);
    // Readings rejected by the association gate but not new enough to be landmarks are dropped (about alpha of
    // them), so the two filters differ slightly.
    CHECK(dropped <= 0.1 * total);
    CHECK((a.pose() - b.pose()).norm() < 0.1);
}
