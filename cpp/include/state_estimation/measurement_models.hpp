#pragma once
/// \file measurement_models.hpp
/// Chapter 3 — measurement models: range-bearing and Cartesian observations of point landmarks, their
/// inverse models, GNSS-like position fixes, a compass, Gaussian likelihoods and a simulated landmark
/// sensor. Equations: 1_theory/03_measurement_models.md.

#include <Eigen/Dense>
#include <vector>

#include "state_estimation/gaussian.hpp"
#include "state_estimation/pose2.hpp"

namespace state_estimation {

// ---- range-bearing (section 3.2) ----

/// Expected range and bearing of landmark m from pose x, eq. (3.4).
Vector2d range_bearing(const Pose2& x, const Vector2d& m);
/// Jacobian of range_bearing with respect to the pose, eq. (3.5).
Matrix23d range_bearing_jacobian_pose(const Pose2& x, const Vector2d& m);
/// Jacobian of range_bearing with respect to the landmark, eq. (3.5).
Matrix2d range_bearing_jacobian_landmark(const Pose2& x, const Vector2d& m);

/// Inverse model: landmark position from pose x and a range-bearing measurement z, eq. (3.6).
Vector2d range_bearing_inverse(const Pose2& x, const Vector2d& z);
/// Jacobian of range_bearing_inverse with respect to the pose, eq. (3.7).
Matrix23d range_bearing_inverse_jacobian_pose(const Pose2& x, const Vector2d& z);
/// Jacobian of range_bearing_inverse with respect to the measurement, eq. (3.7).
Matrix2d range_bearing_inverse_jacobian_measurement(const Pose2& x, const Vector2d& z);

// ---- Cartesian feature (section 3.4) ----

/// Expected position of landmark m in the robot frame, (-)x [+] m = R^T (m - t), eq. (3.9).
Vector2d cartesian_feature(const Pose2& x, const Vector2d& m);
/// Jacobian of cartesian_feature with respect to the pose, eq. (3.10).
Matrix23d cartesian_feature_jacobian_pose(const Pose2& x, const Vector2d& m);
/// Jacobian of cartesian_feature with respect to the landmark (= R^T), eq. (3.10).
Matrix2d cartesian_feature_jacobian_landmark(const Pose2& x);

/// Covariance of a Cartesian point obtained from a polar measurement z with covariance R_polar,
/// J R_polar J^T with J from eq. (1.11), eq. (3.11).
Matrix2d polar_to_cartesian_covariance(const Vector2d& z, const Matrix2d& R_polar);

// ---- position fix and compass (section 3.5) ----

/// GNSS-like fix of an antenna mounted at lever arm l (robot frame): x [+] l, eq. (3.12).
Vector2d position_fix(const Pose2& x, const Vector2d& lever_arm);
/// Jacobian of position_fix with respect to the pose.
Matrix23d position_fix_jacobian(const Pose2& x, const Vector2d& lever_arm);

/// Compass: h(x) = theta, H = [0 0 1], eq. (3.13).
double compass(const Pose2& x);

// ---- likelihoods (section 3.1) ----

/// Innovation z - z_hat with the components listed in angle_indices wrapped to (-pi, pi].
VectorXd measurement_residual(const VectorXd& z, const VectorXd& z_hat, const std::vector<int>& angle_indices);

/// log N(z; z_hat, R) with angle wrapping, eq. (3.2).
double measurement_log_likelihood(const VectorXd& z, const VectorXd& z_hat, const MatrixXd& R,
                                  const std::vector<int>& angle_indices = {});

/// Fisher information H^T R^{-1} H of a stack of linearised measurements, eq. (3.14).
MatrixXd fisher_information(const MatrixXd& H, const MatrixXd& R);

// ---- simulated sensor (section 3.6) ----

/// One observation of landmark `landmark` (index into the map); z is (range, bearing) or (x, y).
struct Observation {
    int landmark = -1;
    Vector2d z = Vector2d::Zero();
};

/// A range-bearing landmark sensor with limited range and field of view.
struct RangeBearingSensor {
    double max_range = 10.0;     ///< [m]
    double fov = 2.0 * 3.14159265358979323846;  ///< full field of view [rad], centred on the heading
    double sigma_range = 0.1;    ///< [m]
    double sigma_bearing = 0.02;  ///< [rad]

    /// Measurement noise covariance diag(sigma_range^2, sigma_bearing^2).
    Matrix2d R() const;
    /// True if landmark m is within range and field of view of pose x.
    bool visible(const Pose2& x, const Vector2d& m) const;
    /// Noisy observations of every visible landmark of the map (rows of an N x 2 matrix).
    std::vector<Observation> observe(const Pose2& x, const MatrixXd& landmarks, Rng& rng) const;
};

/// Convert range-bearing observations to Cartesian (robot frame) observations.
std::vector<Observation> to_cartesian(const std::vector<Observation>& polar);

}  // namespace state_estimation
