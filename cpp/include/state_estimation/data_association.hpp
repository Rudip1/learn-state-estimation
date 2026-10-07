#pragma once
/// \file data_association.hpp
/// Chapter 8 — data association: which landmark produced which reading? Nearest neighbour, individual
/// compatibility (ICNN) and joint compatibility branch and bound (JCBB). Equations: 1_theory/08_data_association.md.

#include <Eigen/Dense>
#include <vector>

#include "state_estimation/gaussian.hpp"
#include "state_estimation/measurement_models.hpp"

namespace state_estimation {

/// An association problem in the linearised form of section 8.1: observations z_i, the predicted reading of
/// every candidate feature zhat_j with its Jacobian H_j with respect to the estimated state (covariance P), and
/// the noise R of one reading. Components listed in angle_indices are wrapped in residuals.
struct AssociationProblem {
    std::vector<VectorXd> observations;
    std::vector<VectorXd> predictions;
    std::vector<MatrixXd> jacobians;
    MatrixXd P;
    MatrixXd R;
    std::vector<int> angle_indices;

    int n_observations() const { return static_cast<int>(observations.size()); }
    int n_features() const { return static_cast<int>(predictions.size()); }
};

/// pairing[i] = feature associated with observation i, or -1 (spurious or new).
struct Association {
    std::vector<int> pairing;
    int n_paired = 0;
    double joint_nis = 0.0;  ///< joint NIS of the paired observations, eq. (8.6)
};

/// Squared Mahalanobis distance of observation i to feature j, eq. (8.2)-(8.3).
double individual_nis(const AssociationProblem& p, int i, int j);

/// Joint NIS and degrees of freedom of a pairing, eq. (8.5)-(8.6). Unpaired observations are ignored.
std::pair<double, int> joint_nis(const AssociationProblem& p, const std::vector<int>& pairing);

/// Nearest neighbour in Euclidean distance of the residual, with a fixed distance gate (no uncertainty used).
Association associate_nearest_neighbor(const AssociationProblem& p, double max_distance);

/// Individual compatibility nearest neighbour, eq. (8.4): for each observation the feature with the smallest
/// individual NIS among those inside the chi-square gate of confidence 1 - alpha.
Association associate_icnn(const AssociationProblem& p, double alpha = 0.05);

/// Joint compatibility branch and bound, section 8.4: the hypothesis with the largest number of pairings whose
/// joint NIS passes the gate chi2_{m k}(1 - alpha); ties are broken by the smaller joint NIS. Each feature is
/// used at most once.
Association associate_jcbb(const AssociationProblem& p, double alpha = 0.05);

/// Exhaustive search over all one-to-one hypotheses with the same criterion as JCBB (for testing; exponential).
Association associate_brute_force(const AssociationProblem& p, double alpha = 0.05);

/// Build the problem for range-bearing readings and a map of known landmarks, with the state being the robot pose
/// (x_bar, P). The landmark indices stored in the observations are ignored.
AssociationProblem landmark_association_problem(const Pose2& x_bar, const Matrix3d& P,
                                                const std::vector<Observation>& observations,
                                                const MatrixXd& landmarks, const Matrix2d& R);

/// Append n spurious range-bearing readings (landmark = -1) uniform within the sensor's range and field of view.
std::vector<Observation> add_clutter(const std::vector<Observation>& observations, int n,
                                     const RangeBearingSensor& sensor, Rng& rng);

}  // namespace state_estimation
