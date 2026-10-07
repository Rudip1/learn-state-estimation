#pragma once
/// \file particle_filter.hpp
/// Chapter 7 — the particle filter: resampling schemes, effective sample size and Monte Carlo localisation
/// (MCL) of a planar pose with random-particle injection. Equations: 1_theory/07_particle_filter.md.

#include <Eigen/Dense>
#include <vector>

#include "state_estimation/measurement_models.hpp"
#include "state_estimation/motion_models.hpp"

namespace state_estimation {

/// Effective sample size 1 / sum w_i^2 of normalised weights, eq. (7.7).
double effective_sample_size(const VectorXd& weights);

/// Resampling schemes of section 7.4. Each returns M ancestor indices drawn so that particle i is copied
/// N_i times with E[N_i] = M w_i (weights need not be normalised).
enum class Resampling { Multinomial, Stratified, Systematic, Residual };

std::vector<int> resample_multinomial(const VectorXd& weights, Rng& rng);  ///< eq. (7.8), roulette wheel
std::vector<int> resample_stratified(const VectorXd& weights, Rng& rng);   ///< eq. (7.9)
std::vector<int> resample_systematic(const VectorXd& weights, Rng& rng);   ///< eq. (7.10), stochastic universal
std::vector<int> resample_residual(const VectorXd& weights, Rng& rng);     ///< eq. (7.11)
std::vector<int> resample(const VectorXd& weights, Resampling method, Rng& rng);

/// Monte Carlo localisation of a planar pose with known point landmarks, section 7.5.
class ParticleFilterLocalization {
public:
    explicit ParticleFilterLocalization(int n_particles, std::uint64_t seed = 0);

    /// Particles drawn from N(mean, cov), equal weights.
    void init_gaussian(const Pose2& mean, const Matrix3d& cov);
    /// Particles uniform over [x_min, x_max] x [y_min, y_max] x (-pi, pi]: global localisation.
    void init_uniform(double x_min, double x_max, double y_min, double y_max);

    /// Prediction: each particle drives the measured wheel travel corrupted by its own sample of the encoder
    /// noise (variance var_per_meter |d| per wheel), eq. (7.12).
    void predict_wheels(const DifferentialDrive& dd, double d_left, double d_right, double var_per_meter);
    /// Prediction with the odometry motion model of chapter 2, eq. (2.17).
    void predict_odometry(const Pose2& odom_prev, const Pose2& odom_curr, const std::array<double, 4>& alpha);

    /// Weighting with range-bearing readings of known landmarks, eq. (7.13). Returns the log of the average
    /// likelihood (an estimate of the evidence p(z | z_{1:k-1})).
    double update_landmarks(const std::vector<Observation>& obs, const MatrixXd& landmarks, const Matrix2d& R);
    /// Weighting with indistinguishable landmarks (mixture likelihood, eq. (4.11)).
    double update_landmarks_anonymous(const std::vector<Observation>& obs, const MatrixXd& landmarks,
                                      const Matrix2d& R);
    /// Weighting with a position fix (antenna at the robot centre).
    double update_position_fix(const Vector2d& z, const Matrix2d& R);

    /// Resample if N_eff < threshold * M; returns true if it resampled.
    bool resample_if_needed(double threshold = 0.5, Resampling method = Resampling::Systematic);

    /// Augmented MCL (section 7.6): track short- and long-term averages of the evidence, resample, then replace
    /// a fraction min(max_fraction, max(0, 1 - w_fast / w_slow)) of the particles by uniform samples over the box,
    /// eq. (7.14).
    void augmented_resample(double log_evidence, double alpha_slow, double alpha_fast, double x_min, double x_max,
                            double y_min, double y_max, double max_fraction = 1.0,
                            Resampling method = Resampling::Systematic);

    /// Weighted mean (circular mean for the heading) and covariance.
    Pose2 mean() const;
    Matrix3d covariance() const;

    int size() const { return static_cast<int>(particles_.rows()); }
    const MatrixXd& particles() const { return particles_; }
    VectorXd weights() const;  ///< normalised
    double effective_sample_size() const;
    /// Fraction of particles that were replaced by random ones in the last augmented_resample call.
    double last_injection_fraction() const { return injected_; }
    void set_particles(const MatrixXd& particles);

private:
    void apply_resampling(const std::vector<int>& idx);
    double add_log_likelihoods(const VectorXd& loglik);

    MatrixXd particles_;  ///< M x 3
    VectorXd log_w_;      ///< unnormalised log weights
    Rng rng_;
    double w_slow_ = 0.0, w_fast_ = 0.0, injected_ = 0.0;
};

}  // namespace state_estimation
