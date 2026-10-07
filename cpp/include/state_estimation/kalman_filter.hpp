#pragma once
/// \file kalman_filter.hpp
/// Chapter 5 — the linear Kalman filter, its consistency tests (NEES, NIS) and its steady state.
/// Equations: 1_theory/05_kalman_filter.md.

#include <Eigen/Dense>
#include <vector>

#include "state_estimation/gaussian.hpp"

namespace state_estimation {

/// Innovation statistics returned by a measurement update.
struct Innovation {
    VectorXd nu;     ///< innovation z - H x_bar, eq. (5.7)
    MatrixXd S;      ///< innovation covariance H P_bar H^T + R, eq. (5.8)
    MatrixXd K;      ///< gain used
    double nis = 0;  ///< normalised innovation squared nu^T S^{-1} nu, eq. (5.13)
};

/// Linear Kalman filter for x_k = F x_{k-1} + B u_k + w_k, z_k = H x_k + v_k.
class KalmanFilter {
public:
    KalmanFilter(const VectorXd& x0, const MatrixXd& P0) : x_(x0), P_(P0) {}

    /// Prediction, eq. (5.4)-(5.5): x <- F x + B u, P <- F P F^T + Q.
    void predict(const MatrixXd& F, const MatrixXd& Q);
    void predict(const MatrixXd& F, const MatrixXd& Q, const MatrixXd& B, const VectorXd& u);

    /// Optimal update, eq. (5.7)-(5.11), with the covariance in Joseph form (5.12).
    Innovation update(const VectorXd& z, const MatrixXd& H, const MatrixXd& R);

    /// Update with an arbitrary gain K. With joseph = true the covariance is exact for any K, eq. (5.12);
    /// with joseph = false the short form (I - K H) P_bar is used, which is only correct for the optimal gain.
    Innovation update_with_gain(const VectorXd& z, const MatrixXd& H, const MatrixXd& R, const MatrixXd& K,
                                bool joseph);

    const VectorXd& x() const { return x_; }
    const MatrixXd& P() const { return P_; }
    void set_state(const VectorXd& x, const MatrixXd& P) {
        x_ = x;
        P_ = P;
    }

private:
    VectorXd x_;
    MatrixXd P_;
};

/// Normalised estimation error squared (x - x_hat)^T P^{-1} (x - x_hat), eq. (5.14).
double nees(const VectorXd& x_true, const VectorXd& x_hat, const MatrixXd& P);

/// Two-sided (1 - alpha) acceptance interval for the average of `runs` independent chi-square(dof)
/// statistics, eq. (5.15): [chi2_{runs dof}(alpha/2), chi2_{runs dof}(1 - alpha/2)] / runs.
Eigen::Vector2d average_chi2_bounds(int dof, int runs, double alpha = 0.05);

/// Steady-state predicted covariance P_bar of a time-invariant filter, by iterating the Riccati recursion
/// eq. (5.16) until it converges (tolerance on the relative change).
MatrixXd steady_state_covariance(const MatrixXd& F, const MatrixXd& H, const MatrixXd& Q, const MatrixXd& R,
                                 double tol = 1e-12, int max_iter = 100000);

/// Discrete white-noise-acceleration (constant velocity) model with state [p_1..p_d, v_1..v_d]:
/// F and Q of eq. (5.18) for spectral density q (per axis).
struct LinearModel {
    MatrixXd F, Q;
};
LinearModel constant_velocity_model(int dims, double dt, double q);

/// Simulated run of a linear-Gaussian system: states (steps+1 rows, x_0 first) and measurements (steps rows,
/// z_1 .. z_steps).
struct LinearRun {
    MatrixXd states;
    MatrixXd measurements;
};
LinearRun simulate_linear_system(const MatrixXd& F, const MatrixXd& Q, const MatrixXd& H, const MatrixXd& R,
                                 const VectorXd& x0, int steps, Rng& rng);

/// Batch weighted least squares estimate of a constant x from z_i = H_i x + v_i, v_i ~ N(0, R_i), with prior
/// N(x0, P0): returns the posterior Gaussian. Used as an independent check of the Kalman update.
Gaussian batch_least_squares(const VectorXd& x0, const MatrixXd& P0, const std::vector<VectorXd>& z,
                             const std::vector<MatrixXd>& H, const std::vector<MatrixXd>& R);

}  // namespace state_estimation
