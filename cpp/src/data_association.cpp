#include "state_estimation/data_association.hpp"

#include <functional>
#include <limits>

namespace state_estimation {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();

VectorXd residual(const AssociationProblem& p, int i, int j) {
    return measurement_residual(p.observations[i], p.predictions[j], p.angle_indices);
}

double gate(const AssociationProblem& p, int pairs, double alpha) {
    return chi2_quantile(1.0 - alpha, pairs * static_cast<int>(p.R.rows()));
}

Association finish(const AssociationProblem& p, const std::vector<int>& pairing) {
    Association a;
    a.pairing = pairing;
    for (int j : pairing) a.n_paired += (j >= 0);
    a.joint_nis = a.n_paired > 0 ? joint_nis(p, pairing).first : 0.0;
    return a;
}

/// Search shared by JCBB (with pruning) and the brute-force reference (without).
Association search(const AssociationProblem& p, double alpha, bool prune) {
    const int n = p.n_observations(), nf = p.n_features();
    const double ic_gate = chi2_quantile(1.0 - alpha, static_cast<int>(p.R.rows()));
    std::vector<int> current(n, -1), best(n, -1);
    std::vector<bool> used(nf, false);
    int best_pairs = 0;
    double best_nis = kInf;
    std::function<void(int, int)> rec = [&](int i, int pairs) {
        if (prune && pairs + (n - i) < best_pairs) return;  // bound: cannot reach the best any more
        if (i == n) {
            const double nis = pairs > 0 ? joint_nis(p, current).first : 0.0;
            if (pairs > best_pairs || (pairs == best_pairs && nis < best_nis)) {
                best = current;
                best_pairs = pairs;
                best_nis = nis;
            }
            return;
        }
        for (int j = 0; j < nf; ++j) {
            if (used[j] || individual_nis(p, i, j) > ic_gate) continue;
            current[i] = j;
            if (joint_nis(p, current).first <= gate(p, pairs + 1, alpha)) {  // eq. (8.7)
                used[j] = true;
                rec(i + 1, pairs + 1);
                used[j] = false;
            }
            current[i] = -1;
        }
        rec(i + 1, pairs);  // observation i unpaired (the "star" branch)
    };
    rec(0, 0);
    return finish(p, best);
}
}  // namespace

double individual_nis(const AssociationProblem& p, int i, int j) {
    // eq. (8.2)-(8.3): nu_ij = z_i - zhat_j, S_ij = H_j P H_j^T + R
    const VectorXd nu = residual(p, i, j);
    const MatrixXd S = p.jacobians[j] * p.P * p.jacobians[j].transpose() + p.R;
    return nu.dot(S.ldlt().solve(nu));
}

std::pair<double, int> joint_nis(const AssociationProblem& p, const std::vector<int>& pairing) {
    // eq. (8.5)-(8.6): stacked innovation and its full covariance, including the cross-covariances
    // H_a P H_b^T that make the innovations of different readings correlated.
    std::vector<int> obs, feat;
    for (size_t i = 0; i < pairing.size(); ++i) {
        if (pairing[i] >= 0) {
            obs.push_back(static_cast<int>(i));
            feat.push_back(pairing[i]);
        }
    }
    const int m = static_cast<int>(p.R.rows()), k = static_cast<int>(obs.size()), n = static_cast<int>(p.P.rows());
    if (k == 0) return {0.0, 0};
    VectorXd nu(m * k);
    MatrixXd H(m * k, n), R = MatrixXd::Zero(m * k, m * k);
    for (int a = 0; a < k; ++a) {
        nu.segment(m * a, m) = residual(p, obs[a], feat[a]);
        H.block(m * a, 0, m, n) = p.jacobians[feat[a]];
        R.block(m * a, m * a, m, m) = p.R;
    }
    const MatrixXd S = H * p.P * H.transpose() + R;
    return {nu.dot(S.ldlt().solve(nu)), m * k};
}

Association associate_nearest_neighbor(const AssociationProblem& p, double max_distance) {
    std::vector<int> pairing(p.n_observations(), -1);
    for (int i = 0; i < p.n_observations(); ++i) {
        double best = max_distance;
        for (int j = 0; j < p.n_features(); ++j) {
            const double d = residual(p, i, j).norm();
            if (d < best) {
                best = d;
                pairing[i] = j;
            }
        }
    }
    return finish(p, pairing);
}

Association associate_icnn(const AssociationProblem& p, double alpha) {
    // eq. (8.4)
    const double ic_gate = chi2_quantile(1.0 - alpha, static_cast<int>(p.R.rows()));
    std::vector<int> pairing(p.n_observations(), -1);
    for (int i = 0; i < p.n_observations(); ++i) {
        double best = ic_gate;
        for (int j = 0; j < p.n_features(); ++j) {
            const double d2 = individual_nis(p, i, j);
            if (d2 < best) {
                best = d2;
                pairing[i] = j;
            }
        }
    }
    return finish(p, pairing);
}

Association associate_jcbb(const AssociationProblem& p, double alpha) { return search(p, alpha, true); }

Association associate_brute_force(const AssociationProblem& p, double alpha) { return search(p, alpha, false); }

AssociationProblem landmark_association_problem(const Pose2& x_bar, const Matrix3d& P,
                                                const std::vector<Observation>& observations,
                                                const MatrixXd& landmarks, const Matrix2d& R) {
    AssociationProblem p;
    for (const auto& o : observations) p.observations.push_back(o.z);
    for (int j = 0; j < landmarks.rows(); ++j) {
        const Vector2d m = landmarks.row(j).transpose();
        p.predictions.push_back(range_bearing(x_bar, m));
        p.jacobians.push_back(range_bearing_jacobian_pose(x_bar, m));
    }
    p.P = P;
    p.R = R;
    p.angle_indices = {1};
    return p;
}

std::vector<Observation> add_clutter(const std::vector<Observation>& observations, int n,
                                     const RangeBearingSensor& sensor, Rng& rng) {
    std::vector<Observation> out = observations;
    for (int c = 0; c < n; ++c) {
        out.push_back({-1, Vector2d(rng.uniform(0.0, sensor.max_range), rng.uniform(-sensor.fov / 2, sensor.fov / 2))});
    }
    return out;
}

}  // namespace state_estimation
