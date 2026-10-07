#pragma once
/// \file gaussian.hpp
/// Chapter 1 — probability for robotics: Gaussians, covariance propagation, Bayes' rule, Mahalanobis
/// distance and the chi-square distribution. Equation numbers refer to 1_theory/01_probability.md.

#include <Eigen/Dense>
#include <cstdint>
#include <functional>
#include <random>

namespace state_estimation {

using Eigen::MatrixXd;
using Eigen::VectorXd;

/// A function R^n -> R^m, used for nonlinear propagation and numerical Jacobians.
using VectorFunction = std::function<VectorXd(const VectorXd&)>;

/// A multivariate Gaussian N(mean, cov).
struct Gaussian {
    VectorXd mean;
    MatrixXd cov;
};

/// Seeded random number generator shared by every sampling routine in the library, so that each
/// experiment is reproducible from its seed.
class Rng {
public:
    explicit Rng(std::uint64_t seed = 0) : engine_(seed) {}
    void seed(std::uint64_t s) { engine_.seed(s); }
    /// Standard normal sample.
    double normal() { return normal_(engine_); }
    /// Normal sample with given mean and standard deviation.
    double normal(double mean, double stddev) { return mean + stddev * normal_(engine_); }
    /// Uniform sample in [lo, hi).
    double uniform(double lo = 0.0, double hi = 1.0) { return lo + (hi - lo) * uniform_(engine_); }
    /// Vector of i.i.d. standard normal samples.
    VectorXd normal_vector(int n);
    std::mt19937_64& engine() { return engine_; }

private:
    std::mt19937_64 engine_;
    std::normal_distribution<double> normal_{0.0, 1.0};
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};
};

/// Squared Mahalanobis distance d^2 = (x - mean)^T cov^{-1} (x - mean), eq. (1.14).
/// Computed with a Cholesky factorisation, never with an explicit inverse.
double mahalanobis_squared(const VectorXd& x, const VectorXd& mean, const MatrixXd& cov);

/// Natural logarithm of the Gaussian density N(x; mean, cov), eq. (1.4).
double gaussian_log_pdf(const VectorXd& x, const VectorXd& mean, const MatrixXd& cov);

/// Gaussian density N(x; mean, cov), eq. (1.4).
double gaussian_pdf(const VectorXd& x, const VectorXd& mean, const MatrixXd& cov);

/// Exact image of a Gaussian under y = A x + b: N(A mean + b, A cov A^T), eq. (1.6)-(1.7).
Gaussian linear_transform(const Gaussian& g, const MatrixXd& A, const VectorXd& b);

/// Jacobian of f at x by central differences (step h). Used to check analytic Jacobians.
MatrixXd numerical_jacobian(const VectorFunction& f, const VectorXd& x, double h = 1e-6);

/// First-order (linearised) propagation through y = f(x): N(f(mean), J cov J^T) with J = df/dx at
/// the mean, eq. (1.9). If \p jacobian is empty the Jacobian is computed numerically.
Gaussian propagate_linearized(const VectorFunction& f, const Gaussian& g,
                              const std::function<MatrixXd(const VectorXd&)>& jacobian = {});

/// Product of two Gaussian densities over the same variable, renormalised: the information-form
/// fusion of eq. (1.18)-(1.20).
Gaussian fuse(const Gaussian& a, const Gaussian& b);

/// Draw n samples from N(mean, cov) as the rows of an n x d matrix, using x = mean + L z with
/// cov = L L^T (Cholesky), eq. (1.16). A singular positive semi-definite cov falls back to the
/// eigendecomposition cov = V diag(lambda) V^T, L = V diag(sqrt(lambda)).
MatrixXd sample_gaussian(const VectorXd& mean, const MatrixXd& cov, int n, Rng& rng);

/// Sample mean and unbiased sample covariance of the rows of \p samples.
Gaussian sample_mean_cov(const MatrixXd& samples);

/// Bayes' rule over a discrete hypothesis set: posterior_i = likelihood_i prior_i / sum_j (...),
/// eq. (1.12). Throws if the evidence is zero.
VectorXd bayes_rule(const VectorXd& prior, const VectorXd& likelihood);

/// Regularised lower incomplete gamma function P(a, x).
double regularized_gamma_p(double a, double x);

/// Cumulative distribution of the chi-square distribution with k degrees of freedom, eq. (1.15).
double chi2_cdf(double x, int k);

/// Quantile (inverse CDF) of the chi-square distribution with k degrees of freedom: the value q
/// with chi2_cdf(q, k) = p. Used as the gate of Mahalanobis tests.
double chi2_quantile(double p, int k);

/// Geometry of the confidence ellipse of a 2x2 covariance at probability p.
struct CovarianceEllipse {
    double semi_major;  ///< length of the major semi-axis
    double semi_minor;  ///< length of the minor semi-axis
    double angle;       ///< orientation of the major axis [rad]
};

/// Confidence ellipse {x : (x-m)^T cov^{-1} (x-m) <= chi2_quantile(p, 2)}, eq. (1.17).
CovarianceEllipse covariance_ellipse(const Eigen::Matrix2d& cov, double p);

/// n points (rows of an n x 2 matrix) on the boundary of the confidence ellipse, for plotting.
MatrixXd ellipse_points(const Eigen::Vector2d& mean, const Eigen::Matrix2d& cov, double p, int n = 64);

/// Polar to Cartesian conversion (r, theta) -> (r cos theta, r sin theta); the standard example of a
/// nonlinear transformation, eq. (1.10).
Eigen::Vector2d polar_to_cartesian(const Eigen::Vector2d& polar);

/// Jacobian of polar_to_cartesian, eq. (1.11).
Eigen::Matrix2d polar_to_cartesian_jacobian(const Eigen::Vector2d& polar);

}  // namespace state_estimation
