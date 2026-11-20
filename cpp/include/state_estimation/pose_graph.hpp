#pragma once
/// \file pose_graph.hpp
/// Chapter 10 — pose-graph SLAM: nodes are robot poses, edges are relative-pose measurements (odometry and
/// scan matching); the poses are found by Gauss-Newton on the sum of squared Mahalanobis edge errors.
/// Equations: 1_theory/10_pose_graph_slam.md, sections 10.5-10.7.

#include <Eigen/Dense>
#include <vector>

#include "state_estimation/gaussian.hpp"
#include "state_estimation/pose2.hpp"
#include "state_estimation/scan_matching.hpp"

namespace state_estimation {

/// A relative-pose measurement z_ij of pose j seen from pose i, with information matrix Omega_ij.
struct PoseGraphEdge {
    int i = 0;
    int j = 0;
    Pose2 z = Pose2::Zero();
    Matrix3d information = Matrix3d::Identity();
};

struct OptimizationSummary {
    int iterations = 0;
    double initial_chi2 = 0.0;
    double final_chi2 = 0.0;
    bool converged = false;
};

/// Error of an edge, eq. (10.11): e_ij = (-)z_ij (+) ((-)x_i (+) x_j), heading wrapped.
Vector3d pose_graph_error(const Pose2& xi, const Pose2& xj, const Pose2& z);

/// Jacobians A = de/dx_i and B = de/dx_j of the edge error, eq. (10.12).
void pose_graph_jacobians(const Pose2& xi, const Pose2& xj, const Pose2& z, Matrix3d& A, Matrix3d& B);

/// Information matrix of an edge from the covariance Sigma_z of its measurement, eq. (10.17):
/// Omega = (J_z Sigma_z J_z^T)^{-1} with J_z = de/dz = -diag(R(theta_z)^T, 1), evaluated at e = 0.
/// Passing Sigma_z^{-1} directly is only right when Sigma_z is rotationally symmetric in x-y.
Matrix3d pose_graph_information(const Pose2& z, const Matrix3d& cov_z);

/// A 2-D pose graph. Node 0 is held fixed: it defines the map frame (the gauge, section 10.6).
class PoseGraph {
public:
    /// Add a node with an initial guess; returns its index.
    int add_node(const Pose2& x);
    void add_edge(int i, int j, const Pose2& z, const Matrix3d& information);
    void set_node(int i, const Pose2& x);

    /// Total cost chi^2 = sum_ij e_ij^T Omega_ij e_ij, eq. (10.13).
    double chi2() const;
    /// The term e_ij^T Omega_ij e_ij of every edge, in insertion order: large values flag wrong loop closures
    /// (section 10.8).
    std::vector<double> edge_chi2() const;
    /// Remove edge k (insertion order); the remaining edges keep their order.
    void remove_edge(int k);

    /// Gauss-Newton, eq. (10.14)-(10.16): build the sparse system H dx = -b over all nodes but node 0, solve it
    /// with a sparse Cholesky factorisation and add dx to the poses; stop when |dx|_inf < tolerance.
    OptimizationSummary optimize(int max_iterations = 20, double tolerance = 1e-9);

    /// Marginal covariance of node i at the current estimate: the 3x3 block of H^{-1} (section 10.7).
    /// Zero for node 0.
    Matrix3d marginal_covariance(int i) const;

    /// The information matrix H (3n x 3n, node 0 included) at the current estimate, for inspecting its sparsity.
    MatrixXd information_matrix() const;

    int n_nodes() const { return static_cast<int>(nodes_.size()); }
    int n_edges() const { return static_cast<int>(edges_.size()); }
    Pose2 node(int i) const { return nodes_.at(i); }
    /// Poses as an n x 3 matrix.
    MatrixXd nodes() const;
    const std::vector<PoseGraphEdge>& edges() const { return edges_; }

private:
    std::vector<Pose2> nodes_;
    std::vector<PoseGraphEdge> edges_;
};

struct PoseGraphSlamOptions {
    IcpOptions icp;                  ///< options of every scan match
    double sequential_max_rmse = 0.03;  ///< accept a match of consecutive scans only below this rmse ...
    double odometry_gate = 0.999;    ///< ... and only if it agrees with odometry at this confidence (chi2, 3 dof)
    int loop_min_separation = 15;    ///< loop closures only with keyframes at least this many keyframes back
    double loop_search_radius = 1.5; ///< search radius [m], widened by 3 sigma of the current position
    double loop_max_rmse = 0.02;
    int loop_min_pairs = 80;
    bool optimize_on_closure = true; ///< re-optimise after every accepted loop closure
};

/// What happened when a keyframe was added.
struct KeyframeResult {
    bool scan_matched = false;  ///< the odometry edge came from scan matching (false: wheel odometry)
    int loop_closure = -1;      ///< index of the old keyframe a loop closure was added to, or -1
};

/// Incremental pose-graph SLAM with a 2-D lidar (section 10.8): one node per keyframe; the edge to the previous
/// keyframe comes from scan matching gated against wheel odometry; at most one loop closure per keyframe, searched
/// around the current estimate within a radius that grows with its uncertainty and accepted only if the match is
/// good; the graph is re-optimised after each closure.
class PoseGraphSlam {
public:
    PoseGraphSlam(const Pose2& x0, const MatrixXd& scan0, const PoseGraphSlamOptions& options = {});

    /// Add a keyframe: \p odometry and \p odometry_cov are the wheel-odometry displacement since the previous
    /// keyframe and its covariance; \p scan is the new scan in the robot frame.
    KeyframeResult add_keyframe(const Pose2& odometry, const Matrix3d& odometry_cov, const MatrixXd& scan);

    const PoseGraph& graph() const { return graph_; }
    PoseGraph& graph() { return graph_; }
    const std::vector<MatrixXd>& scans() const { return scans_; }
    /// (old, new) keyframe pairs of the accepted loop closures.
    const std::vector<std::pair<int, int>>& loop_closures() const { return loops_; }
    int n_scan_matched() const { return n_matched_; }
    /// All scans in the map frame at the current estimate, stacked (for plotting the map).
    MatrixXd map_points() const;

private:
    PoseGraphSlamOptions options_;
    PoseGraph graph_;
    std::vector<MatrixXd> scans_;
    std::vector<std::pair<int, int>> loops_;
    int n_matched_ = 0;
};

}  // namespace state_estimation
