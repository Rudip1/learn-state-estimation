#include "state_estimation/gaussian.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace state_estimation {

namespace {
constexpr double kPi = 3.14159265358979323846;

Eigen::LLT<MatrixXd> checked_llt(const MatrixXd& cov) {
    Eigen::LLT<MatrixXd> llt(cov);
    if (llt.info() != Eigen::Success) {
        throw std::invalid_argument("covariance is not positive definite");
    }
    return llt;
}

/// A matrix S with S S^T = cov for a symmetric positive semi-definite cov.
MatrixXd square_root(const MatrixXd& cov) {
    Eigen::LLT<MatrixXd> llt(cov);
    if (llt.info() == Eigen::Success) return llt.matrixL();
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(cov);
    if (es.eigenvalues().minCoeff() < -1e-9 * std::max(1.0, es.eigenvalues().cwiseAbs().maxCoeff())) {
        throw std::invalid_argument("covariance is not positive semi-definite");
    }
    return es.eigenvectors() * es.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal();
}
}  // namespace

VectorXd Rng::normal_vector(int n) {
    VectorXd z(n);
    for (int i = 0; i < n; ++i) z(i) = normal();
    return z;
}

double mahalanobis_squared(const VectorXd& x, const VectorXd& mean, const MatrixXd& cov) {
    // eq. (1.14): solve L y = (x - mean); then d^2 = y^T y.
    const auto llt = checked_llt(cov);
    const VectorXd y = llt.matrixL().solve(x - mean);
    return y.squaredNorm();
}

double gaussian_log_pdf(const VectorXd& x, const VectorXd& mean, const MatrixXd& cov) {
    // eq. (1.4) in logarithmic form; log|cov| = 2 sum log L_ii.
    const auto llt = checked_llt(cov);
    const VectorXd y = llt.matrixL().solve(x - mean);
    const MatrixXd L = llt.matrixL();
    const double log_det = 2.0 * L.diagonal().array().log().sum();
    const double n = static_cast<double>(x.size());
    return -0.5 * (n * std::log(2.0 * kPi) + log_det + y.squaredNorm());
}

double gaussian_pdf(const VectorXd& x, const VectorXd& mean, const MatrixXd& cov) {
    return std::exp(gaussian_log_pdf(x, mean, cov));
}

Gaussian linear_transform(const Gaussian& g, const MatrixXd& A, const VectorXd& b) {
    // eq. (1.6)-(1.7)
    return {A * g.mean + b, A * g.cov * A.transpose()};
}

MatrixXd numerical_jacobian(const VectorFunction& f, const VectorXd& x, double h) {
    const VectorXd f0 = f(x);
    MatrixXd J(f0.size(), x.size());
    VectorXd xp = x, xm = x;
    for (int j = 0; j < x.size(); ++j) {
        xp(j) = x(j) + h;
        xm(j) = x(j) - h;
        J.col(j) = (f(xp) - f(xm)) / (2.0 * h);
        xp(j) = xm(j) = x(j);
    }
    return J;
}

Gaussian propagate_linearized(const VectorFunction& f, const Gaussian& g,
                              const std::function<MatrixXd(const VectorXd&)>& jacobian) {
    // eq. (1.9): mean through f, covariance through the Jacobian at the mean.
    const MatrixXd J = jacobian ? jacobian(g.mean) : numerical_jacobian(f, g.mean);
    return {f(g.mean), J * g.cov * J.transpose()};
}

Gaussian fuse(const Gaussian& a, const Gaussian& b) {
    // eq. (1.18)-(1.20), written in the gain form that avoids inverting either covariance:
    // K = Pa (Pa + Pb)^{-1}, mean = ma + K (mb - ma), cov = (I - K) Pa.
    const MatrixXd S = a.cov + b.cov;
    const MatrixXd K = S.transpose().ldlt().solve(a.cov.transpose()).transpose();
    Gaussian out;
    out.mean = a.mean + K * (b.mean - a.mean);
    out.cov = a.cov - K * a.cov;
    out.cov = 0.5 * (out.cov + out.cov.transpose());
    return out;
}

MatrixXd sample_gaussian(const VectorXd& mean, const MatrixXd& cov, int n, Rng& rng) {
    // eq. (1.16): x = mean + S z, S S^T = cov, z ~ N(0, I).
    const MatrixXd S = square_root(cov);
    const int d = static_cast<int>(mean.size());
    MatrixXd out(n, d);
    for (int i = 0; i < n; ++i) out.row(i) = (mean + S * rng.normal_vector(d)).transpose();
    return out;
}

Gaussian sample_mean_cov(const MatrixXd& samples) {
    const double n = static_cast<double>(samples.rows());
    if (samples.rows() < 2) throw std::invalid_argument("need at least two samples");
    const VectorXd mean = samples.colwise().mean().transpose();
    const MatrixXd centered = samples.rowwise() - mean.transpose();
    return {mean, centered.transpose() * centered / (n - 1.0)};
}

VectorXd bayes_rule(const VectorXd& prior, const VectorXd& likelihood) {
    // eq. (1.12)
    if (prior.size() != likelihood.size()) throw std::invalid_argument("size mismatch");
    const VectorXd unnormalized = prior.cwiseProduct(likelihood);
    const double evidence = unnormalized.sum();
    if (!(evidence > 0.0)) throw std::domain_error("evidence p(z) is zero");
    return unnormalized / evidence;
}

double regularized_gamma_p(double a, double x) {
    if (a <= 0.0) throw std::invalid_argument("a must be positive");
    if (x <= 0.0) return 0.0;
    const double log_prefactor = a * std::log(x) - x - std::lgamma(a);
    if (x < a + 1.0) {
        // Series: P(a,x) = e^{-x} x^a / Gamma(a) * sum_n x^n / (a (a+1) ... (a+n)).
        double term = 1.0 / a, sum = term, ap = a;
        for (int n = 0; n < 1000; ++n) {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if (std::abs(term) < std::abs(sum) * 1e-16) break;
        }
        return sum * std::exp(log_prefactor);
    }
    // Continued fraction for Q(a,x) = 1 - P(a,x), evaluated with the modified Lentz method.
    const double tiny = 1e-300;
    double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
    for (int i = 1; i < 1000; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < tiny) d = tiny;
        c = b + an / c;
        if (std::abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::abs(delta - 1.0) < 1e-16) break;
    }
    return 1.0 - std::exp(log_prefactor) * h;
}

double chi2_cdf(double x, int k) {
    if (k < 1) throw std::invalid_argument("degrees of freedom must be >= 1");
    return regularized_gamma_p(0.5 * k, 0.5 * x);  // eq. (1.15)
}

double chi2_quantile(double p, int k) {
    if (!(p > 0.0 && p < 1.0)) throw std::invalid_argument("p must be in (0, 1)");
    double lo = 0.0, hi = std::max(1.0, static_cast<double>(k));
    while (chi2_cdf(hi, k) < p) hi *= 2.0;
    for (int i = 0; i < 200 && hi - lo > 1e-13 * hi; ++i) {
        const double mid = 0.5 * (lo + hi);
        (chi2_cdf(mid, k) < p ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

CovarianceEllipse covariance_ellipse(const Eigen::Matrix2d& cov, double p) {
    // eq. (1.17): semi-axes sqrt(q lambda_i) along the eigenvectors, q = chi2_quantile(p, 2).
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
    const double q = chi2_quantile(p, 2);
    const Eigen::Vector2d lambda = es.eigenvalues().cwiseMax(0.0);  // ascending
    const Eigen::Vector2d major = es.eigenvectors().col(1);
    return {std::sqrt(q * lambda(1)), std::sqrt(q * lambda(0)), std::atan2(major.y(), major.x())};
}

MatrixXd ellipse_points(const Eigen::Vector2d& mean, const Eigen::Matrix2d& cov, double p, int n) {
    const CovarianceEllipse e = covariance_ellipse(cov, p);
    const double c = std::cos(e.angle), s = std::sin(e.angle);
    MatrixXd pts(n, 2);
    for (int i = 0; i < n; ++i) {
        const double t = 2.0 * kPi * i / (n - 1);
        const double u = e.semi_major * std::cos(t), v = e.semi_minor * std::sin(t);
        pts(i, 0) = mean.x() + c * u - s * v;
        pts(i, 1) = mean.y() + s * u + c * v;
    }
    return pts;
}

Eigen::Vector2d polar_to_cartesian(const Eigen::Vector2d& polar) {
    return {polar(0) * std::cos(polar(1)), polar(0) * std::sin(polar(1))};  // eq. (1.10)
}

Eigen::Matrix2d polar_to_cartesian_jacobian(const Eigen::Vector2d& polar) {
    const double r = polar(0), c = std::cos(polar(1)), s = std::sin(polar(1));
    Eigen::Matrix2d J;  // eq. (1.11)
    J << c, -r * s, s, r * c;
    return J;
}

}  // namespace state_estimation
