#pragma once
/// \file ekf_slam.hpp
/// Chapter 9 — feature-based EKF SLAM: the state holds the robot pose and the positions of point landmarks;
/// new landmarks are added by state augmentation. Equations: 1_theory/09_ekf_slam.md.

#include <Eigen/Dense>
#include <map>
#include <vector>

#include "state_estimation/data_association.hpp"
#include "state_estimation/kalman_filter.hpp"
#include "state_estimation/measurement_models.hpp"
#include "state_estimation/motion_models.hpp"

namespace state_estimation {

/// What happened to each reading of a scan processed with data association.
struct ScanResult {
    std::vector<int> landmark;  ///< state index of the landmark paired with or added as; -1 if dropped
    std::vector<bool> is_new;   ///< true if the reading created a new landmark
    double joint_nis = 0.0;     ///< joint NIS of the paired readings
};

/// EKF SLAM with range-bearing observations of point landmarks, state y = [x, y, theta, m_1, ..., m_n].
class EkfSlam {
public:
    EkfSlam(const Pose2& x0, const Matrix3d& P0);

    /// Prediction with a robot-frame displacement u ~ N(u, Q_u). Only the robot block and the robot-landmark
    /// cross-covariances change, eq. (9.3)-(9.4): O(n) instead of O(n^2).
    void predict_displacement(const Pose2& u, const Matrix3d& Q_u);
    void predict_wheels(const DifferentialDrive& dd, double d_left, double d_right, double var_left,
                        double var_right);

    /// Add a landmark from a range-bearing reading by state augmentation, eq. (9.5)-(9.7). Returns its index.
    int add_landmark(const Vector2d& z, const Matrix2d& R);
    /// Add a landmark with a given position and covariance, uncorrelated with the rest (a prior map).
    int add_known_landmark(const Vector2d& m, const Matrix2d& P_m);

    /// Stacked update with readings of landmarks already in the state; observation.landmark is the state index of
    /// the landmark, eq. (9.8)-(9.9).
    Innovation update(const std::vector<Observation>& observations, const Matrix2d& R);

    /// Readings labelled with external identifiers: identifiers seen before update the filter, new identifiers
    /// are added as landmarks afterwards (section 9.4).
    Innovation process_known(const std::vector<Observation>& observations, const Matrix2d& R);

    /// Unlabelled readings: associate with JCBB (confidence 1 - alpha) against all landmarks in the state and
    /// update with the pairings. An unpaired reading becomes a new landmark only if it is incompatible with every
    /// landmark even at the much wider gate chi2_2(1 - new_alpha); otherwise it is ambiguous and dropped
    /// (section 9.5).
    ScanResult process_jcbb(const std::vector<Observation>& observations, const Matrix2d& R, double alpha = 0.05,
                            double new_alpha = 1e-4);

    /// The association problem of a scan against the whole state (H with robot and landmark blocks).
    AssociationProblem association_problem(const std::vector<Observation>& observations, const Matrix2d& R) const;

    int n_landmarks() const { return static_cast<int>((y_.size() - 3) / 2); }
    Pose2 pose() const { return y_.head<3>(); }
    Matrix3d pose_covariance() const { return P_.topLeftCorner<3, 3>(); }
    Vector2d landmark(int i) const { return y_.segment<2>(3 + 2 * i); }
    Matrix2d landmark_covariance(int i) const { return P_.block<2, 2>(3 + 2 * i, 3 + 2 * i); }
    /// Landmark positions as an n x 2 matrix.
    MatrixXd landmarks() const;
    const VectorXd& state() const { return y_; }
    const MatrixXd& covariance() const { return P_; }
    /// External identifier -> state index, for landmarks added by process_known.
    const std::map<int, int>& id_map() const { return ids_; }

private:
    /// h for landmark i and its Jacobian with respect to the full state (2 x dim).
    Vector2d predict_reading(int i) const;
    MatrixXd reading_jacobian(int i) const;

    VectorXd y_;
    MatrixXd P_;
    std::map<int, int> ids_;
};

}  // namespace state_estimation
