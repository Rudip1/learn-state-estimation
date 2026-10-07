#pragma once
/// \file histogram_filter.hpp
/// Chapter 4 — the Bayes filter on a discrete state space: a generic discrete (histogram) filter and grid
/// localisation of a planar pose over an (x, y, theta) grid. Equations: 1_theory/04_bayes_filter.md.

#include <Eigen/Dense>
#include <array>
#include <vector>

#include "state_estimation/measurement_models.hpp"
#include "state_estimation/motion_models.hpp"

namespace state_estimation {

/// Bayes filter over a finite set of states, eq. (4.6)-(4.7).
class DiscreteBayesFilter {
public:
    explicit DiscreteBayesFilter(const VectorXd& prior);

    /// Prediction with a column-stochastic transition matrix T(i, j) = p(x_k = i | x_{k-1} = j), eq. (4.6).
    void predict(const MatrixXd& transition);

    /// Prediction on a cyclic 1-D world: the robot moves by offset + i cells with probability kernel(i).
    void predict_cyclic(const VectorXd& kernel, int offset);

    /// Correction with likelihood(i) = p(z | x = i), eq. (4.7). Returns the evidence p(z).
    double update(const VectorXd& likelihood);

    const VectorXd& belief() const { return belief_; }

private:
    VectorXd belief_;
};

/// Extent and resolution of a pose grid.
struct GridSpec {
    double x_min = 0.0, x_max = 10.0;
    double y_min = 0.0, y_max = 10.0;
    double resolution = 0.2;  ///< cell size in x and y [m]
    int n_theta = 36;         ///< number of heading cells over 2 pi
};

/// Grid (Markov) localisation of a planar pose, section 4.3.
class GridLocalization {
public:
    explicit GridLocalization(const GridSpec& spec);

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    int n_theta() const { return nt_; }
    int size() const { return nx_ * ny_ * nt_; }
    const GridSpec& spec() const { return spec_; }

    /// Pose at the centre of cell (ix, iy, it).
    Pose2 cell_center(int ix, int iy, int it) const;

    /// Uniform belief: global localisation.
    void set_uniform();
    /// Belief proportional to a Gaussian N(mean, cov) evaluated at the cell centres.
    void set_gaussian(const Pose2& mean, const Matrix3d& cov);

    /// Motion update with the odometry model, eq. (4.8): each cell's mass is moved deterministically to
    /// apply_rtr(cell centre, u) and split trilinearly between neighbouring cells, then blurred with the
    /// odometry noise of eq. (2.17). Mass that leaves the grid is dropped.
    void predict_odometry(const Pose2& odom_prev, const Pose2& odom_curr, const std::array<double, 4>& alpha);

    /// Measurement update with range-bearing observations of known landmarks, eq. (4.9). If
    /// account_for_cell_size is true, the variance of a pose uniformly distributed within a cell is added to
    /// R, eq. (4.10). Returns the log evidence of the observations; if it is -infinity (every cell is ruled
    /// out) the belief is left unchanged.
    double update_range_bearing(const std::vector<Observation>& observations, const MatrixXd& landmarks,
                                const Matrix2d& R, bool account_for_cell_size = true);

    /// Measurement update with indistinguishable landmarks: each reading may come from any landmark, so its
    /// likelihood is the mixture (1/n) sum_j p(z | x, m_j), eq. (4.11). The `landmark` field of the
    /// observations is ignored. Returns the log evidence.
    double update_range_bearing_anonymous(const std::vector<Observation>& observations, const MatrixXd& landmarks,
                                          const Matrix2d& R, bool account_for_cell_size = true);

    /// Mix the belief with a uniform distribution: bel <- (1 - eps) bel + eps / N, eq. (4.12).
    void mix_uniform(double eps);

    /// Weighted mean; the heading is a circular mean.
    Pose2 mean() const;
    /// Covariance about the mean (heading differences wrapped).
    Matrix3d covariance() const;
    /// Centre of the most probable cell.
    Pose2 map_estimate() const;
    /// Marginal over (x, y), as an ny x nx matrix (row = y index).
    MatrixXd marginal_xy() const;
    /// Flattened belief, index = (it * ny + iy) * nx + ix.
    const VectorXd& belief() const { return bel_; }
    void set_belief(const VectorXd& b);

private:
    int index(int ix, int iy, int it) const { return (it * ny_ + iy) * nx_ + ix; }
    void normalize();
    void blur(double sigma_xy_cells, double sigma_theta_cells);
    double update_impl(const std::vector<Observation>& observations, const MatrixXd& landmarks, const Matrix2d& R,
                       bool account_for_cell_size, bool known_correspondences);

    GridSpec spec_;
    int nx_, ny_, nt_;
    double dtheta_;
    VectorXd bel_;
};

}  // namespace state_estimation
