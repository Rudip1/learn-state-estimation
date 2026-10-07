// Chapter 5 tests: the Kalman filter against the scalar closed forms of the theory, batch least squares, the
// steady-state Riccati solution, Monte Carlo error statistics (Joseph form, NEES/NIS) and tabulated chi-square
// quantiles.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "state_estimation/kalman_filter.hpp"

using namespace state_estimation;
using Catch::Approx;

TEST_CASE("Scalar update reproduces the fusion example of chapter 1", "[ch05]") {
    KalmanFilter kf(VectorXd::Constant(1, 10.0), MatrixXd::Constant(1, 1, 4.0));
    const Innovation inn = kf.update(VectorXd::Constant(1, 12.0), MatrixXd::Identity(1, 1), MatrixXd::Constant(1, 1, 1.0));
    CHECK(kf.x()(0) == Approx(11.6));
    CHECK(kf.P()(0, 0) == Approx(0.8));
    CHECK(inn.K(0, 0) == Approx(0.8));
    CHECK(inn.S(0, 0) == Approx(5.0));
    CHECK(inn.nis == Approx(4.0 / 5.0));
}

TEST_CASE("Sequential updates of a static state equal batch least squares", "[ch05]") {
    Rng rng(5);
    const int n = 3;
    VectorXd x0 = VectorXd::Zero(n);
    MatrixXd P0 = MatrixXd::Identity(n, n) * 10.0;
    KalmanFilter kf(x0, P0);
    std::vector<VectorXd> zs;
    std::vector<MatrixXd> Hs, Rs;
    for (int i = 0; i < 8; ++i) {
        MatrixXd H = MatrixXd::Zero(2, n);
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < n; ++c) H(r, c) = rng.normal();
        MatrixXd R(2, 2);
        R << 0.5 + i * 0.1, 0.1, 0.1, 0.3;
        VectorXd z(2);
        z << rng.normal(), rng.normal();
        kf.predict(MatrixXd::Identity(n, n), MatrixXd::Zero(n, n));
        kf.update(z, H, R);
        zs.push_back(z);
        Hs.push_back(H);
        Rs.push_back(R);
    }
    const Gaussian batch = batch_least_squares(x0, P0, zs, Hs, Rs);
    CHECK((kf.x() - batch.mean).norm() < 1e-10);
    CHECK((kf.P() - batch.cov).norm() < 1e-10);
}

TEST_CASE("Steady state of the scalar random walk (section 5.6)", "[ch05]") {
    // x_k = x_{k-1} + w, z = x + v with q = 1, r = 4: P_bar = (q + sqrt(q^2 + 4 q r)) / 2 = 2.5616,
    // P = 1.5616, K = 0.3904.
    const double q = 1.0, r = 4.0;
    const double pbar = (q + std::sqrt(q * q + 4.0 * q * r)) / 2.0;
    const MatrixXd I = MatrixXd::Identity(1, 1);
    const MatrixXd Pss = steady_state_covariance(I, I, I * q, I * r);
    CHECK(Pss(0, 0) == Approx(pbar).epsilon(1e-10));
    CHECK(pbar == Approx(2.5616).margin(5e-5));
    CHECK(pbar * r / (pbar + r) == Approx(1.5616).margin(5e-5));
    CHECK(pbar / (pbar + r) == Approx(0.3904).margin(5e-5));
    // The filter converges to it from any start.
    KalmanFilter kf(VectorXd::Zero(1), MatrixXd::Constant(1, 1, 100.0));
    for (int k = 0; k < 50; ++k) {
        kf.predict(I, I * q);
        kf.update(VectorXd::Zero(1), I, I * r);
    }
    CHECK(kf.P()(0, 0) == Approx(pbar * r / (pbar + r)).epsilon(1e-10));
}

TEST_CASE("Constant-velocity process noise matches integrated white acceleration", "[ch05]") {
    // Integrate a(t) = white noise with spectral density q over dt with a fine Euler scheme and compare the
    // covariance of (position, velocity) with Q of eq. (5.18).
    const double q = 0.7, dt = 0.5;
    const LinearModel m = constant_velocity_model(1, dt, q);
    Rng rng(9);
    const int runs = 20000, sub = 200;
    const double h = dt / sub;
    MatrixXd s(runs, 2);
    for (int r = 0; r < runs; ++r) {
        double p = 0.0, v = 0.0;
        for (int i = 0; i < sub; ++i) {
            const double a = std::sqrt(q / h) * rng.normal();
            p += v * h + 0.5 * a * h * h;
            v += a * h;
        }
        s.row(r) << p, v;
    }
    const Gaussian g = sample_mean_cov(s);
    CHECK(g.cov(0, 0) == Approx(m.Q(0, 0)).epsilon(0.04));
    CHECK(g.cov(0, 1) == Approx(m.Q(0, 1)).epsilon(0.04));
    CHECK(g.cov(1, 1) == Approx(m.Q(1, 1)).epsilon(0.04));
}

TEST_CASE("Joseph form is exact for any gain, the short form only for the optimal one", "[ch05]") {
    VectorXd x0(2);
    x0 << 1.0, -1.0;
    MatrixXd P0(2, 2);
    P0 << 2.0, 0.5, 0.5, 1.0;
    MatrixXd H(1, 2);
    H << 1.0, 0.0;
    const MatrixXd R = MatrixXd::Constant(1, 1, 0.5);
    MatrixXd K_bad(2, 1);
    K_bad << 0.3, 0.05;  // optimal would be (0.8, 0.2)
    // Optimal gain: both forms agree.
    KalmanFilter a(x0, P0), b(x0, P0);
    const Innovation opt = a.update(VectorXd::Constant(1, 1.5), H, R);
    b.update_with_gain(VectorXd::Constant(1, 1.5), H, R, opt.K, false);
    CHECK((a.P() - b.P()).norm() < 1e-12);
    // Suboptimal gain: the actual error covariance (Monte Carlo) equals the Joseph form.
    KalmanFilter jo(x0, P0), sh(x0, P0);
    jo.update_with_gain(VectorXd::Constant(1, 0.0), H, R, K_bad, true);
    sh.update_with_gain(VectorXd::Constant(1, 0.0), H, R, K_bad, false);
    Rng rng(3);
    const int runs = 100000;
    MatrixXd err(runs, 2);
    for (int i = 0; i < runs; ++i) {
        const VectorXd x = sample_gaussian(x0, P0, 1, rng).row(0).transpose();
        const VectorXd z = H * x + VectorXd::Constant(1, std::sqrt(0.5) * rng.normal());
        const VectorXd est = x0 + K_bad * (z - H * x0);
        err.row(i) = (est - x).transpose();
    }
    const MatrixXd emp = sample_mean_cov(err).cov;
    CHECK((emp - jo.P()).norm() < 0.02 * jo.P().norm());
    CHECK((emp - sh.P()).norm() > 0.1 * jo.P().norm());
}

TEST_CASE("Chi-square bounds for averaged NEES match tables", "[ch05]") {
    // 50 runs of a 2-D state: chi2 with 100 degrees of freedom, quantiles 74.2219 and 129.5612.
    const Eigen::Vector2d b = average_chi2_bounds(2, 50, 0.05);
    CHECK(b(0) == Approx(74.2219 / 50.0).epsilon(1e-5));
    CHECK(b(1) == Approx(129.5612 / 50.0).epsilon(1e-5));
}

TEST_CASE("A correctly tuned filter is consistent: NEES and NIS inside the bounds", "[ch05]") {
    const LinearModel m = constant_velocity_model(1, 0.1, 0.5);
    MatrixXd H(1, 2);
    H << 1.0, 0.0;
    const MatrixXd R = MatrixXd::Constant(1, 1, 0.04);
    VectorXd x0(2);
    x0 << 0.0, 1.0;
    const MatrixXd P0 = Eigen::Vector2d(0.1, 0.1).asDiagonal();
    Rng rng(17);
    const int runs = 100, steps = 100;
    VectorXd sum_nees = VectorXd::Zero(steps), sum_nis = VectorXd::Zero(steps);
    for (int r = 0; r < runs; ++r) {
        const VectorXd start = sample_gaussian(x0, P0, 1, rng).row(0).transpose();
        const LinearRun run = simulate_linear_system(m.F, m.Q, H, R, start, steps, rng);
        KalmanFilter kf(x0, P0);
        for (int k = 0; k < steps; ++k) {
            kf.predict(m.F, m.Q);
            sum_nis(k) += kf.update(run.measurements.row(k).transpose(), H, R).nis;
            sum_nees(k) += nees(run.states.row(k + 1).transpose(), kf.x(), kf.P());
        }
    }
    // At each time step the average over the runs is chi2(runs * dof) / runs: about 95 % of the steps must fall
    // inside the 95 % interval (consecutive steps are correlated, so the count is only approximately binomial).
    const Eigen::Vector2d bn = average_chi2_bounds(2, runs), bi = average_chi2_bounds(1, runs);
    int nees_in = 0, nis_in = 0;
    for (int k = 0; k < steps; ++k) {
        const double e = sum_nees(k) / runs, v = sum_nis(k) / runs;
        nees_in += (e >= bn(0) && e <= bn(1));
        nis_in += (v >= bi(0) && v <= bi(1));
    }
    CHECK(nees_in >= 85);
    CHECK(nis_in >= 85);
    CHECK(sum_nees.sum() / (runs * steps) == Approx(2.0).epsilon(0.1));
    CHECK(sum_nis.sum() / (runs * steps) == Approx(1.0).epsilon(0.1));
}
