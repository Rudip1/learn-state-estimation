#include "state_estimation/particle_filter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace state_estimation {

namespace {
constexpr double kPi = 3.14159265358979323846;

/// Cumulative sum of the weights normalised to end exactly at 1.
std::vector<double> normalized_cdf(const VectorXd& w) {
    if ((w.array() < 0.0).any() || !(w.sum() > 0.0)) throw std::invalid_argument("invalid weights");
    std::vector<double> c(w.size());
    double s = 0.0;
    const double total = w.sum();
    for (int i = 0; i < w.size(); ++i) c[i] = (s += w(i) / total);
    c.back() = 1.0;
    return c;
}

/// Ancestors for sorted positions u_0 <= ... <= u_{M-1} in [0, 1): one pass over the CDF.
std::vector<int> sweep(const std::vector<double>& cdf, const std::vector<double>& u) {
    std::vector<int> idx(u.size());
    size_t j = 0;
    for (size_t i = 0; i < u.size(); ++i) {
        while (j + 1 < cdf.size() && u[i] >= cdf[j]) ++j;
        idx[i] = static_cast<int>(j);
    }
    return idx;
}

double log_sum_exp(const VectorXd& v) {
    const double m = v.maxCoeff();
    if (!std::isfinite(m)) return m;
    return m + std::log((v.array() - m).exp().sum());
}
}  // namespace

double effective_sample_size(const VectorXd& weights) {
    const VectorXd w = weights / weights.sum();
    return 1.0 / w.squaredNorm();  // eq. (7.7)
}

std::vector<int> resample_multinomial(const VectorXd& weights, Rng& rng) {
    // eq. (7.8): M independent spins of the roulette wheel, each located by binary search: O(M log M).
    const auto cdf = normalized_cdf(weights);
    std::vector<int> idx(weights.size());
    for (auto& i : idx) {
        i = static_cast<int>(std::upper_bound(cdf.begin(), cdf.end(), rng.uniform()) - cdf.begin());
        i = std::min(i, static_cast<int>(cdf.size()) - 1);
    }
    return idx;
}

std::vector<int> resample_stratified(const VectorXd& weights, Rng& rng) {
    // eq. (7.9): one uniform draw inside each of the M strata [i/M, (i+1)/M).
    const auto cdf = normalized_cdf(weights);
    const int M = static_cast<int>(weights.size());
    std::vector<double> u(M);
    for (int i = 0; i < M; ++i) u[i] = (i + rng.uniform()) / M;
    return sweep(cdf, u);
}

std::vector<int> resample_systematic(const VectorXd& weights, Rng& rng) {
    // eq. (7.10): a single draw, M equally spaced pointers (stochastic universal sampling): O(M).
    const auto cdf = normalized_cdf(weights);
    const int M = static_cast<int>(weights.size());
    const double u0 = rng.uniform();
    std::vector<double> u(M);
    for (int i = 0; i < M; ++i) u[i] = (i + u0) / M;
    return sweep(cdf, u);
}

std::vector<int> resample_residual(const VectorXd& weights, Rng& rng) {
    // eq. (7.11): floor(M w_i) deterministic copies, the remaining R draws multinomially from the residuals.
    const int M = static_cast<int>(weights.size());
    const VectorXd w = weights / weights.sum();
    std::vector<int> idx;
    idx.reserve(M);
    VectorXd residual(M);
    for (int i = 0; i < M; ++i) {
        const int copies = static_cast<int>(std::floor(M * w(i)));
        for (int c = 0; c < copies; ++c) idx.push_back(i);
        residual(i) = M * w(i) - copies;
    }
    const int rest = M - static_cast<int>(idx.size());
    if (rest > 0) {
        const auto cdf = normalized_cdf(residual);
        for (int r = 0; r < rest; ++r) {
            int i = static_cast<int>(std::upper_bound(cdf.begin(), cdf.end(), rng.uniform()) - cdf.begin());
            idx.push_back(std::min(i, M - 1));
        }
    }
    return idx;
}

std::vector<int> resample(const VectorXd& weights, Resampling method, Rng& rng) {
    switch (method) {
        case Resampling::Multinomial:
            return resample_multinomial(weights, rng);
        case Resampling::Stratified:
            return resample_stratified(weights, rng);
        case Resampling::Systematic:
            return resample_systematic(weights, rng);
        case Resampling::Residual:
            return resample_residual(weights, rng);
    }
    throw std::invalid_argument("unknown resampling method");
}

// ---------------------------------------------------------------- MCL

ParticleFilterLocalization::ParticleFilterLocalization(int n, std::uint64_t seed)
    : particles_(MatrixXd::Zero(n, 3)), log_w_(VectorXd::Zero(n)), rng_(seed) {
    if (n < 1) throw std::invalid_argument("need at least one particle");
}

void ParticleFilterLocalization::init_gaussian(const Pose2& mean, const Matrix3d& cov) {
    particles_ = sample_gaussian(mean, cov, size(), rng_);
    for (int i = 0; i < size(); ++i) particles_(i, 2) = wrap_angle(particles_(i, 2));
    log_w_.setZero();
}

void ParticleFilterLocalization::init_uniform(double x_min, double x_max, double y_min, double y_max) {
    for (int i = 0; i < size(); ++i) {
        particles_.row(i) << rng_.uniform(x_min, x_max), rng_.uniform(y_min, y_max), rng_.uniform(-kPi, kPi);
    }
    log_w_.setZero();
}

void ParticleFilterLocalization::set_particles(const MatrixXd& p) {
    if (p.cols() != 3) throw std::invalid_argument("particles must be M x 3");
    particles_ = p;
    log_w_ = VectorXd::Zero(p.rows());
}

void ParticleFilterLocalization::predict_wheels(const DifferentialDrive& dd, double d_left, double d_right,
                                                double var_per_meter) {
    // eq. (7.12): sample the proposal = the motion model, one noise draw per particle.
    const double sl = std::sqrt(var_per_meter * std::abs(d_left)), sr = std::sqrt(var_per_meter * std::abs(d_right));
    for (int i = 0; i < size(); ++i) {
        const Vector2d arc = dd.wheel_travel_to_arc(d_left + sl * rng_.normal(), d_right + sr * rng_.normal());
        particles_.row(i) = compose(particles_.row(i).transpose(), arc_displacement(arc(0), arc(1))).transpose();
    }
}

void ParticleFilterLocalization::predict_odometry(const Pose2& odom_prev, const Pose2& odom_curr,
                                                  const std::array<double, 4>& alpha) {
    for (int i = 0; i < size(); ++i) {
        particles_.row(i) =
            sample_odometry_motion(particles_.row(i).transpose(), odom_prev, odom_curr, alpha, rng_).transpose();
    }
}

double ParticleFilterLocalization::add_log_likelihoods(const VectorXd& loglik) {
    // Weights are kept as logarithms and shifted so that the largest is 0 (no underflow, eq. (7.13)).
    const VectorXd prev = log_w_.array() - log_sum_exp(log_w_);  // log of the normalised previous weights
    const double log_evidence = log_sum_exp(prev + loglik);
    log_w_ += loglik;
    const double m = log_w_.maxCoeff();
    if (std::isfinite(m)) log_w_.array() -= m;
    return log_evidence;
}

double ParticleFilterLocalization::update_landmarks(const std::vector<Observation>& obs, const MatrixXd& landmarks,
                                                    const Matrix2d& R) {
    if (obs.empty()) return 0.0;
    VectorXd loglik = VectorXd::Zero(size());
    for (int i = 0; i < size(); ++i) {
        const Pose2 x = particles_.row(i).transpose();
        for (const auto& o : obs) {
            loglik(i) += measurement_log_likelihood(o.z, range_bearing(x, landmarks.row(o.landmark).transpose()), R,
                                                    {1});
        }
    }
    return add_log_likelihoods(loglik);
}

double ParticleFilterLocalization::update_landmarks_anonymous(const std::vector<Observation>& obs,
                                                              const MatrixXd& landmarks, const Matrix2d& R) {
    if (obs.empty()) return 0.0;
    const int n = static_cast<int>(landmarks.rows());
    VectorXd loglik = VectorXd::Zero(size()), terms(n);
    for (int i = 0; i < size(); ++i) {
        const Pose2 x = particles_.row(i).transpose();
        for (const auto& o : obs) {
            for (int j = 0; j < n; ++j)
                terms(j) = measurement_log_likelihood(o.z, range_bearing(x, landmarks.row(j).transpose()), R, {1});
            loglik(i) += log_sum_exp(terms) - std::log(static_cast<double>(n));
        }
    }
    return add_log_likelihoods(loglik);
}

double ParticleFilterLocalization::update_position_fix(const Vector2d& z, const Matrix2d& R) {
    VectorXd loglik(size());
    for (int i = 0; i < size(); ++i) {
        loglik(i) = measurement_log_likelihood(z, particles_.row(i).head<2>().transpose(), R);
    }
    return add_log_likelihoods(loglik);
}

VectorXd ParticleFilterLocalization::weights() const {
    VectorXd w = (log_w_.array() - log_w_.maxCoeff()).exp();
    return w / w.sum();
}

double ParticleFilterLocalization::effective_sample_size() const {
    return state_estimation::effective_sample_size(weights());
}

void ParticleFilterLocalization::apply_resampling(const std::vector<int>& idx) {
    MatrixXd next(idx.size(), 3);
    for (size_t i = 0; i < idx.size(); ++i) next.row(i) = particles_.row(idx[i]);
    particles_ = next;
    log_w_.setZero();
}

bool ParticleFilterLocalization::resample_if_needed(double threshold, Resampling method) {
    if (effective_sample_size() >= threshold * size()) return false;
    apply_resampling(resample(weights(), method, rng_));
    return true;
}

void ParticleFilterLocalization::augmented_resample(double log_evidence, double alpha_slow, double alpha_fast,
                                                    double x_min, double x_max, double y_min, double y_max,
                                                    double max_fraction, Resampling method) {
    // Thrun et al., table 8.3: short- and long-term averages of the measurement likelihood.
    const double w_avg = std::exp(log_evidence);
    if (w_slow_ == 0.0) w_slow_ = w_fast_ = w_avg;
    w_slow_ += alpha_slow * (w_avg - w_slow_);
    w_fast_ += alpha_fast * (w_avg - w_fast_);
    injected_ = w_slow_ > 0.0 ? std::min(max_fraction, std::max(0.0, 1.0 - w_fast_ / w_slow_)) : 0.0;
    apply_resampling(resample(weights(), method, rng_));
    for (int i = 0; i < size(); ++i) {
        if (rng_.uniform() < injected_) {
            particles_.row(i) << rng_.uniform(x_min, x_max), rng_.uniform(y_min, y_max), rng_.uniform(-kPi, kPi);
        }
    }
}

Pose2 ParticleFilterLocalization::mean() const {
    const VectorXd w = weights();
    const double c = w.dot(particles_.col(2).array().cos().matrix());
    const double s = w.dot(particles_.col(2).array().sin().matrix());
    return {w.dot(particles_.col(0)), w.dot(particles_.col(1)), std::atan2(s, c)};
}

Matrix3d ParticleFilterLocalization::covariance() const {
    const VectorXd w = weights();
    const Pose2 mu = mean();
    Matrix3d P = Matrix3d::Zero();
    for (int i = 0; i < size(); ++i) {
        Vector3d d = particles_.row(i).transpose() - mu;
        d.z() = wrap_angle(d.z());
        P += w(i) * d * d.transpose();
    }
    return P;
}

}  // namespace state_estimation
