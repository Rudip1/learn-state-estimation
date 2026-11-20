// Chapter 7 tests: effective sample size, unbiasedness and variance of the resampling schemes, the deterministic
// guarantees of systematic and residual resampling, the particle filter against the exact Kalman posterior in a
// linear-Gaussian case, and global localisation.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "state_estimation/kalman_filter.hpp"
#include "state_estimation/particle_filter.hpp"

using namespace state_estimation;
using Catch::Approx;

namespace {
constexpr double kPi = 3.14159265358979323846;

VectorXd counts(const std::vector<int>& idx, int M) {
    VectorXd c = VectorXd::Zero(M);
    for (int i : idx) c(i) += 1.0;
    return c;
}
}  // namespace

TEST_CASE("Effective sample size", "[ch07]") {
    CHECK(effective_sample_size(VectorXd::Ones(50)) == Approx(50.0));
    VectorXd one_hot = VectorXd::Zero(50);
    one_hot(7) = 3.0;
    CHECK(effective_sample_size(one_hot) == Approx(1.0));
    VectorXd half(4);
    half << 1.0, 1.0, 0.0, 0.0;
    CHECK(effective_sample_size(half) == Approx(2.0));
}

TEST_CASE("All resampling schemes are unbiased; systematic has the smallest variance", "[ch07]") {
    VectorXd w(5);
    w << 0.05, 0.4, 0.1, 0.3, 0.15;
    const int M = 5, trials = 40000;
    Rng rng(1);
    for (Resampling m : {Resampling::Multinomial, Resampling::Stratified, Resampling::Systematic,
                         Resampling::Residual}) {
        VectorXd sum = VectorXd::Zero(M), sum2 = VectorXd::Zero(M);
        for (int t = 0; t < trials; ++t) {
            const VectorXd c = counts(resample(w, m, rng), M);
            REQUIRE(c.sum() == M);
            sum += c;
            sum2 += c.cwiseProduct(c);
        }
        const VectorXd mean = sum / trials;
        CHECK((mean - M * w).cwiseAbs().maxCoeff() < 0.02);
        const VectorXd var = sum2 / trials - mean.cwiseProduct(mean);
        // Multinomial: Var[N_i] = M w_i (1 - w_i). The other schemes must not exceed it.
        const VectorXd var_multi = M * w.array() * (1.0 - w.array());
        if (m == Resampling::Multinomial) {
            CHECK((var - var_multi).cwiseAbs().maxCoeff() < 0.03);
        } else {
            CHECK((var.array() <= var_multi.array() + 0.02).all());
        }
    }
}

TEST_CASE("Systematic copies floor or ceil of M w; residual at least floor of M w", "[ch07]") {
    Rng rng(4);
    const int M = 100;
    for (int t = 0; t < 200; ++t) {
        VectorXd w(M);
        for (int i = 0; i < M; ++i) w(i) = std::pow(rng.uniform(), 3.0);
        w /= w.sum();
        const VectorXd cs = counts(resample_systematic(w, rng), M);
        const VectorXd cr = counts(resample_residual(w, rng), M);
        for (int i = 0; i < M; ++i) {
            CHECK(cs(i) >= std::floor(M * w(i)) - 1e-9);
            CHECK(cs(i) <= std::ceil(M * w(i)) + 1e-9);
            CHECK(cr(i) >= std::floor(M * w(i)) - 1e-9);
        }
    }
}

TEST_CASE("Particle filter matches the Kalman posterior in a linear-Gaussian case", "[ch07]") {
    // Prior on the pose N(mu, P); a position fix is linear in (x, y), so the exact posterior is the Kalman update.
    const Pose2 mu(1.0, -0.5, 0.3);
    const Matrix3d P = (Matrix3d() << 0.5, 0.1, 0.02, 0.1, 0.3, -0.01, 0.02, -0.01, 0.05).finished();
    const Vector2d z(1.4, -0.2);
    const Matrix2d R = Vector2d(0.2, 0.1).asDiagonal();
    KalmanFilter kf(mu, P);
    MatrixXd H = MatrixXd::Zero(2, 3);
    H(0, 0) = H(1, 1) = 1.0;
    kf.update(z, H, R);

    ParticleFilterLocalization pf(200000, 7);
    pf.init_gaussian(mu, P);
    pf.update_position_fix(z, R);
    CHECK((pf.mean() - kf.x()).norm() < 0.01);
    CHECK((pf.covariance() - kf.P()).norm() < 0.01);
    // Resampling does not change the represented distribution (beyond Monte Carlo noise).
    pf.resample_if_needed(1.1);
    CHECK((pf.mean() - kf.x()).norm() < 0.01);
    CHECK(pf.effective_sample_size() == Approx(200000.0));
}

TEST_CASE("Global localisation with known landmarks", "[ch07]") {
    MatrixXd landmarks(4, 2);
    landmarks << 0.0, 0.0, 6.0, 0.0, 6.0, 6.0, 0.0, 6.0;
    RangeBearingSensor sensor{20.0, 2.0 * kPi, 0.1, 0.03};
    const DifferentialDrive dd{0.1, 0.5, 0};
    ParticleFilterLocalization pf(5000, 3);
    pf.init_uniform(0.0, 6.0, 0.0, 6.0);
    Rng rng(5);
    Pose2 truth(2.0, 3.0, 0.4);
    for (int k = 0; k < 20; ++k) {
        const double dl = 0.05, dr = 0.055;
        const Vector2d arc = dd.wheel_travel_to_arc(dl, dr);
        truth = compose(truth, arc_displacement(arc(0), arc(1)));
        pf.predict_wheels(dd, dl, dr, 1e-3);
        pf.update_landmarks(sensor.observe(truth, landmarks, rng), landmarks, sensor.R());
        pf.resample_if_needed(0.5);
    }
    CHECK((pf.mean().head<2>() - truth.head<2>()).norm() < 0.1);
    CHECK(std::abs(wrap_angle(pf.mean().z() - truth.z())) < 0.05);
}
