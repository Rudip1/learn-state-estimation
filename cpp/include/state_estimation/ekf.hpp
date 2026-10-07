#pragma once
/// \file ekf.hpp
/// Chapter 6 — the extended Kalman filter and map-based localisation of a planar robot with known point
/// landmarks. Equations: 1_theory/06_ekf_localization.md.

#include <Eigen/Dense>
#include <functional>
#include <vector>

#include "state_estimation/kalman_filter.hpp"
#include "state_estimation/measurement_models.hpp"
#include "state_estimation/motion_models.hpp"
#include "state_estimation/simulation.hpp"

namespace state_estimation {

/// A measurement model for the generic EKF: prediction h(x) and Jacobian H(x).
struct MeasurementFunction {
    std::function<VectorXd(const VectorXd&)> h;
    std::function<MatrixXd(const VectorXd&)> H;
    std::vector<int> angle_indices;  ///< components of z that are angles (residuals are wrapped)
};

/// Generic extended Kalman filter on a vector state.
class ExtendedKalmanFilter {
public:
    ExtendedKalmanFilter(const VectorXd& x0, const MatrixXd& P0) : x_(x0), P_(P0) {}

    /// Prediction, eq. (6.3)-(6.4): x <- f(x), P <- F P F^T + W Q W^T, with F and W evaluated at the old mean.
    void predict(const std::function<VectorXd(const VectorXd&)>& f, const MatrixXd& F, const MatrixXd& W,
                 const MatrixXd& Q);

    /// Update, eq. (6.5)-(6.9) (Joseph form). With iterations > 1 the iterated EKF of section 6.5 relinearises
    /// h about the current iterate, eq. (6.14).
    Innovation update(const VectorXd& z, const MeasurementFunction& model, const MatrixXd& R, int iterations = 1);

    const VectorXd& x() const { return x_; }
    const MatrixXd& P() const { return P_; }
    void set_state(const VectorXd& x, const MatrixXd& P) {
        x_ = x;
        P_ = P;
    }
    /// Indices of x that are angles; they are wrapped after every step.
    void set_angle_states(const std::vector<int>& idx) { angle_states_ = idx; }

private:
    void wrap_state();
    VectorXd x_;
    MatrixXd P_;
    std::vector<int> angle_states_;
};

/// EKF localisation of a planar pose [x, y, theta] in a map of known point landmarks, section 6.3.
class EkfLocalization {
public:
    EkfLocalization(const Pose2& x0, const Matrix3d& P0);

    /// Prediction with a robot-frame displacement u ~ N(u, Q_u): x <- x (+) u, eq. (6.10).
    void predict_displacement(const Pose2& u, const Matrix3d& Q_u);
    /// Prediction from wheel travel (d_l, d_r) with variances (var_l, var_r), eq. (6.10) and (2.21).
    void predict_wheels(const DifferentialDrive& dd, double d_left, double d_right, double var_left,
                        double var_right);

    /// Update with range-bearing observations of known landmarks (correspondences in Observation::landmark),
    /// stacked into one update, eq. (6.11)-(6.12). Returns the innovation statistics of the stacked update; an
    /// empty scan returns an Innovation with nis = 0 and empty matrices.
    Innovation update_landmarks(const std::vector<Observation>& observations, const MatrixXd& landmarks,
                                const Matrix2d& R, int iterations = 1);
    /// Update with Cartesian (robot-frame) observations of known landmarks, eq. (6.13).
    Innovation update_cartesian(const std::vector<Observation>& observations, const MatrixXd& landmarks,
                                const Matrix2d& R);
    /// Update with a compass reading of the heading, eq. (3.13).
    Innovation update_compass(double theta, double variance);
    /// Update with a position fix of an antenna at the given lever arm, eq. (3.12).
    Innovation update_position_fix(const Vector2d& z, const Vector2d& lever_arm, const Matrix2d& R);

    Pose2 pose() const { return ekf_.x(); }
    Matrix3d covariance() const { return ekf_.P(); }
    void set_state(const Pose2& x, const Matrix3d& P) { ekf_.set_state(x, P); }

private:
    ExtendedKalmanFilter ekf_;
};

/// One simulated run of a robot observing landmarks: (N+1) x 3 true poses, N x 2 measured wheel travel, and
/// the scan taken after each step (empty when no scan was taken).
struct LandmarkRun {
    MatrixXd truth;
    MatrixXd wheel_travel;
    std::vector<std::vector<Observation>> scans;
};

/// Simulate controls, encoders and a range-bearing sensor that scans every `scan_every` steps.
LandmarkRun simulate_landmark_run(const Pose2& x0, const MatrixXd& controls, double dt, const DifferentialDrive& dd,
                                  double var_per_meter, const RangeBearingSensor& sensor, const MatrixXd& landmarks,
                                  int scan_every, Rng& rng);

}  // namespace state_estimation
