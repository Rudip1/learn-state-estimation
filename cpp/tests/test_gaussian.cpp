// Chapter 1 tests: every routine is checked against a closed form, a tabulated value or a worked
// example of 1_theory/01_probability.md.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "state_estimation/gaussian.hpp"

using namespace state_estimation;
using Catch::Approx;

TEST_CASE("Gaussian density matches the scalar closed form", "[ch01]") {
    // eq. (1.3) evaluated by hand: N(1; 0, 4) = exp(-1/8) / sqrt(8 pi).
    VectorXd x(1), m(1);
    x << 1.0;
    m << 0.0;
    MatrixXd P(1, 1);
    P << 4.0;
    const double expected = std::exp(-1.0 / 8.0) / std::sqrt(8.0 * M_PI);
    CHECK(gaussian_pdf(x, m, P) == Approx(expected).epsilon(1e-14));
}

TEST_CASE("Multivariate density with diagonal covariance factorises", "[ch01]") {
    VectorXd x(3), m(3);
    x << 0.3, -1.0, 2.0;
    m << 0.0, 0.5, 1.0;
    VectorXd var(3);
    var << 0.5, 2.0, 1.5;
    double log_product = 0.0;
    for (int i = 0; i < 3; ++i) {
        log_product += -0.5 * std::log(2.0 * M_PI * var(i)) - 0.5 * std::pow(x(i) - m(i), 2) / var(i);
    }
    CHECK(gaussian_log_pdf(x, m, var.asDiagonal().toDenseMatrix()) == Approx(log_product).epsilon(1e-13));
}

TEST_CASE("Mahalanobis distance equals the explicit quadratic form", "[ch01]") {
    MatrixXd P(2, 2);
    P << 4.0, 1.2, 1.2, 1.0;
    VectorXd x(2), m(2);
    x << 1.0, 2.0;
    m << -0.5, 0.5;
    const VectorXd d = x - m;
    const double expected = d.dot(P.inverse() * d);
    CHECK(mahalanobis_squared(x, m, P) == Approx(expected).epsilon(1e-13));
    // Non positive-definite covariance is rejected.
    MatrixXd bad(2, 2);
    bad << 1.0, 2.0, 2.0, 1.0;
    CHECK_THROWS(mahalanobis_squared(x, m, bad));
}

TEST_CASE("Chi-square CDF and quantiles match tabulated values", "[ch01]") {
    // Tabulated quantiles (Abramowitz & Stegun, table 26.8).
    CHECK(chi2_quantile(0.95, 1) == Approx(3.841458820694124).epsilon(1e-10));
    CHECK(chi2_quantile(0.95, 2) == Approx(5.991464547107979).epsilon(1e-10));
    CHECK(chi2_quantile(0.99, 3) == Approx(11.344866730144373).epsilon(1e-10));
    CHECK(chi2_quantile(0.05, 10) == Approx(3.940299136119240).epsilon(1e-10));
    // Closed forms: k = 2 is exponential, k = 1 is related to erf.
    CHECK(chi2_cdf(1.0, 2) == Approx(1.0 - std::exp(-0.5)).epsilon(1e-14));  // 0.3935, 1-sigma 2-D
    CHECK(chi2_cdf(4.0, 2) == Approx(1.0 - std::exp(-2.0)).epsilon(1e-14));  // 0.8647, 2-sigma 2-D
    CHECK(chi2_cdf(1.0, 1) == Approx(std::erf(1.0 / std::sqrt(2.0))).epsilon(1e-13));  // 0.6827
    CHECK(chi2_cdf(1.0, 1) == Approx(0.6827).margin(5e-5));
    CHECK(chi2_cdf(1.0, 2) == Approx(0.3935).margin(5e-5));
    CHECK(chi2_cdf(4.0, 2) == Approx(0.8647).margin(5e-5));
    // Large x goes through the continued-fraction branch.
    CHECK(chi2_cdf(chi2_quantile(0.999, 4), 4) == Approx(0.999).epsilon(1e-12));
}

TEST_CASE("Linear transformation is exact (eq. 1.7)", "[ch01]") {
    Gaussian g{VectorXd::Zero(2), MatrixXd::Identity(2, 2)};
    g.mean << 1.0, 2.0;
    g.cov << 2.0, 0.5, 0.5, 1.0;
    MatrixXd A(3, 2);
    A << 1, 0, 0, 1, 1, 1;
    VectorXd b(3);
    b << 0.0, 1.0, -1.0;
    const Gaussian y = linear_transform(g, A, b);
    CHECK(y.mean(2) == Approx(2.0));
    // Var[x1 + x2] = 2 + 1 + 2 * 0.5 = 4
    CHECK(y.cov(2, 2) == Approx(4.0));
    // Linearised propagation of a linear map reproduces the exact result.
    const Gaussian lin = propagate_linearized([&](const VectorXd& x) -> VectorXd { return A * x + b; }, g);
    CHECK((lin.mean - y.mean).norm() < 1e-9);
    CHECK((lin.cov - y.cov).norm() < 1e-8);
}

TEST_CASE("Polar-to-Cartesian Jacobian matches numerical differentiation", "[ch01]") {
    Eigen::Vector2d p(2.5, 0.7);
    const MatrixXd Jn = numerical_jacobian(
        [](const VectorXd& v) -> VectorXd { return polar_to_cartesian(Eigen::Vector2d(v)); }, p);
    CHECK((Jn - MatrixXd(polar_to_cartesian_jacobian(p))).norm() < 1e-8);
}

TEST_CASE("Monte Carlo mean of rho cos(phi) shows the linearisation bias", "[ch01]") {
    // Closed form: E[rho cos phi] = rho exp(-sigma^2/2) for phi ~ N(0, sigma^2). For sigma = 0.5 the
    // factor is 0.8825 (quoted in the theory), while linearisation predicts 1.
    const double sigma = 0.5;
    CHECK(std::exp(-sigma * sigma / 2) == Approx(0.8825).margin(5e-5));
    Rng rng(7);
    VectorXd m(2);
    m << 1.0, 0.0;
    MatrixXd P = MatrixXd::Zero(2, 2);
    P(1, 1) = sigma * sigma;  // exact range, uncertain bearing: semi-definite covariance
    const MatrixXd s = sample_gaussian(m, P, 200000, rng);
    double mean_x = 0.0;
    for (int i = 0; i < s.rows(); ++i) mean_x += polar_to_cartesian(Eigen::Vector2d(s.row(i))).x();
    mean_x /= s.rows();
    CHECK(mean_x == Approx(std::exp(-sigma * sigma / 2)).margin(3e-3));
}

TEST_CASE("Sampling reproduces mean and covariance", "[ch01]") {
    Rng rng(42);
    VectorXd m(3);
    m << 1.0, -2.0, 0.5;
    MatrixXd P(3, 3);
    P << 2.0, 0.6, -0.3, 0.6, 1.0, 0.2, -0.3, 0.2, 0.5;
    const MatrixXd s = sample_gaussian(m, P, 200000, rng);
    const Gaussian est = sample_mean_cov(s);
    CHECK((est.mean - m).cwiseAbs().maxCoeff() < 0.02);
    CHECK((est.cov - P).cwiseAbs().maxCoeff() < 0.03);
    // Fraction inside the 95 % Mahalanobis gate.
    const double gate = chi2_quantile(0.95, 3);
    int inside = 0;
    for (int i = 0; i < s.rows(); ++i) inside += mahalanobis_squared(s.row(i).transpose(), m, P) <= gate;
    CHECK(static_cast<double>(inside) / s.rows() == Approx(0.95).margin(0.003));
}

TEST_CASE("Bayes rule: the door example of section 1.5", "[ch01]") {
    VectorXd prior(2), likelihood(2);
    prior << 0.5, 0.5;       // open, closed
    likelihood << 0.6, 0.3;  // p(z = "open" | open), p(z = "open" | closed)
    const VectorXd post = bayes_rule(prior, likelihood);
    CHECK(post(0) == Approx(2.0 / 3.0));
    CHECK(post(0) == Approx(0.6667).margin(5e-5));
    const VectorXd post2 = bayes_rule(post, likelihood);
    CHECK(post2(0) == Approx(0.8));
    CHECK_THROWS(bayes_rule(prior, VectorXd::Zero(2)));
}

TEST_CASE("Fusion: the worked example of section 1.8 and information addition", "[ch01]") {
    Gaussian a{VectorXd::Constant(1, 10.0), MatrixXd::Constant(1, 1, 4.0)};
    Gaussian b{VectorXd::Constant(1, 12.0), MatrixXd::Constant(1, 1, 1.0)};
    const Gaussian c = fuse(a, b);
    CHECK(c.mean(0) == Approx(11.6));
    CHECK(c.cov(0, 0) == Approx(0.8));
    // Matrix case: compare against the information form eq. (1.18)-(1.19).
    Gaussian p{VectorXd(2), MatrixXd(2, 2)}, q{VectorXd(2), MatrixXd(2, 2)};
    p.mean << 1.0, 0.0;
    p.cov << 2.0, 0.3, 0.3, 1.0;
    q.mean << 0.0, 2.0;
    q.cov << 0.5, -0.1, -0.1, 3.0;
    const Gaussian r = fuse(p, q);
    const MatrixXd info = p.cov.inverse() + q.cov.inverse();
    CHECK((r.cov - info.inverse()).norm() < 1e-12);
    CHECK((r.mean - info.inverse() * (p.cov.inverse() * p.mean + q.cov.inverse() * q.mean)).norm() < 1e-12);
}

TEST_CASE("Confidence ellipse of a diagonal covariance", "[ch01]") {
    Eigen::Matrix2d P;
    P << 4.0, 0.0, 0.0, 1.0;
    const double p = 1.0 - std::exp(-0.5);  // 1-sigma: q = 1
    const CovarianceEllipse e = covariance_ellipse(P, p);
    CHECK(e.semi_major == Approx(2.0));
    CHECK(e.semi_minor == Approx(1.0));
    CHECK(std::abs(std::sin(e.angle)) < 1e-12);
    const MatrixXd pts = ellipse_points(Eigen::Vector2d(1.0, 1.0), P, p, 33);
    for (int i = 0; i < pts.rows(); ++i) {
        VectorXd v = pts.row(i).transpose();
        CHECK(mahalanobis_squared(v, Eigen::Vector2d(1.0, 1.0), P) == Approx(1.0).epsilon(1e-9));
    }
}
