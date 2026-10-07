#include "state_estimation/kalman_filter.hpp"

#include <cmath>
#include <stdexcept>

namespace state_estimation {

namespace {
MatrixXd symmetrize(const MatrixXd& P) { return 0.5 * (P + P.transpose()); }
}  // namespace

void KalmanFilter::predict(const MatrixXd& F, const MatrixXd& Q) {
    // eq. (5.4)-(5.5)
    x_ = F * x_;
    P_ = symmetrize(F * P_ * F.transpose() + Q);
}

void KalmanFilter::predict(const MatrixXd& F, const MatrixXd& Q, const MatrixXd& B, const VectorXd& u) {
    x_ = F * x_ + B * u;
    P_ = symmetrize(F * P_ * F.transpose() + Q);
}

Innovation KalmanFilter::update(const VectorXd& z, const MatrixXd& H, const MatrixXd& R) {
    // eq. (5.8)-(5.9): K = P_bar H^T S^{-1}, computed as the solution of S K^T = H P_bar.
    const MatrixXd S = symmetrize(H * P_ * H.transpose() + R);
    const MatrixXd K = S.ldlt().solve(H * P_).transpose();
    return update_with_gain(z, H, R, K, true);
}

Innovation KalmanFilter::update_with_gain(const VectorXd& z, const MatrixXd& H, const MatrixXd& R,
                                          const MatrixXd& K, bool joseph) {
    Innovation inn;
    inn.nu = z - H * x_;                                     // eq. (5.7)
    inn.S = symmetrize(H * P_ * H.transpose() + R);          // eq. (5.8)
    inn.K = K;
    inn.nis = inn.nu.dot(inn.S.ldlt().solve(inn.nu));        // eq. (5.13)
    x_ = x_ + K * inn.nu;                                    // eq. (5.10)
    const MatrixXd IKH = MatrixXd::Identity(x_.size(), x_.size()) - K * H;
    if (joseph) {
        P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();  // eq. (5.12)
    } else {
        P_ = IKH * P_;  // eq. (5.11)
    }
    P_ = symmetrize(P_);
    return inn;
}

double nees(const VectorXd& x_true, const VectorXd& x_hat, const MatrixXd& P) {
    return mahalanobis_squared(x_true, x_hat, P);  // eq. (5.14)
}

Eigen::Vector2d average_chi2_bounds(int dof, int runs, double alpha) {
    // eq. (5.15): the sum of `runs` independent chi2(dof) variables is chi2(runs * dof).
    const int k = dof * runs;
    return {chi2_quantile(alpha / 2.0, k) / runs, chi2_quantile(1.0 - alpha / 2.0, k) / runs};
}

MatrixXd steady_state_covariance(const MatrixXd& F, const MatrixXd& H, const MatrixXd& Q, const MatrixXd& R,
                                 double tol, int max_iter) {
    // eq. (5.16): P_bar <- F (P_bar - P_bar H^T (H P_bar H^T + R)^{-1} H P_bar) F^T + Q
    MatrixXd P = Q + MatrixXd::Identity(Q.rows(), Q.cols());
    for (int i = 0; i < max_iter; ++i) {
        const MatrixXd S = H * P * H.transpose() + R;
        const MatrixXd post = P - P * H.transpose() * S.ldlt().solve(H * P);
        const MatrixXd next = symmetrize(F * post * F.transpose() + Q);
        const double change = (next - P).norm() / std::max(1.0, next.norm());
        P = next;
        if (change < tol) return P;
    }
    throw std::runtime_error("Riccati recursion did not converge (is the system detectable?)");
}

LinearModel constant_velocity_model(int dims, double dt, double q) {
    // eq. (5.18)
    LinearModel m;
    const int n = 2 * dims;
    m.F = MatrixXd::Identity(n, n);
    m.Q = MatrixXd::Zero(n, n);
    for (int i = 0; i < dims; ++i) {
        m.F(i, dims + i) = dt;
        m.Q(i, i) = q * dt * dt * dt / 3.0;
        m.Q(i, dims + i) = m.Q(dims + i, i) = q * dt * dt / 2.0;
        m.Q(dims + i, dims + i) = q * dt;
    }
    return m;
}

LinearRun simulate_linear_system(const MatrixXd& F, const MatrixXd& Q, const MatrixXd& H, const MatrixXd& R,
                                 const VectorXd& x0, int steps, Rng& rng) {
    LinearRun run;
    run.states.resize(steps + 1, x0.size());
    run.measurements.resize(steps, H.rows());
    const VectorXd zero_x = VectorXd::Zero(x0.size()), zero_z = VectorXd::Zero(H.rows());
    VectorXd x = x0;
    run.states.row(0) = x.transpose();
    for (int k = 0; k < steps; ++k) {
        x = F * x + sample_gaussian(zero_x, Q, 1, rng).row(0).transpose();
        run.states.row(k + 1) = x.transpose();
        run.measurements.row(k) = (H * x + sample_gaussian(zero_z, R, 1, rng).row(0).transpose()).transpose();
    }
    return run;
}

Gaussian batch_least_squares(const VectorXd& x0, const MatrixXd& P0, const std::vector<VectorXd>& z,
                             const std::vector<MatrixXd>& H, const std::vector<MatrixXd>& R) {
    // Normal equations of the stacked problem: (P0^-1 + sum H^T R^-1 H) x = P0^-1 x0 + sum H^T R^-1 z.
    MatrixXd info = P0.inverse();
    VectorXd rhs = info * x0;
    for (size_t i = 0; i < z.size(); ++i) {
        const MatrixXd Ri = R[i].inverse();
        info += H[i].transpose() * Ri * H[i];
        rhs += H[i].transpose() * Ri * z[i];
    }
    const MatrixXd P = info.inverse();
    return {P * rhs, P};
}

}  // namespace state_estimation
