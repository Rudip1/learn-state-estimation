#include "state_estimation/ekf_slam.hpp"

#include <stdexcept>

namespace state_estimation {

namespace {
MatrixXd symmetrize(const MatrixXd& P) { return 0.5 * (P + P.transpose()); }
}  // namespace

EkfSlam::EkfSlam(const Pose2& x0, const Matrix3d& P0) : y_(x0), P_(P0) {}

void EkfSlam::predict_displacement(const Pose2& u, const Matrix3d& Q_u) {
    // eq. (9.3)-(9.4): the landmarks do not move; only the robot rows and columns of P change.
    const Pose2 x = pose();
    const Matrix3d F = compose_jacobian_1(x, u), W = compose_jacobian_2(x, u);
    const int n = static_cast<int>(y_.size());
    y_.head<3>() = compose(x, u);
    P_.topLeftCorner<3, 3>() = F * P_.topLeftCorner<3, 3>() * F.transpose() + W * Q_u * W.transpose();
    if (n > 3) {
        P_.topRightCorner(3, n - 3) = F * P_.topRightCorner(3, n - 3);
        P_.bottomLeftCorner(n - 3, 3) = P_.topRightCorner(3, n - 3).transpose();
    }
}

void EkfSlam::predict_wheels(const DifferentialDrive& dd, double d_left, double d_right, double var_left,
                             double var_right) {
    const Vector2d arc = dd.wheel_travel_to_arc(d_left, d_right);
    predict_displacement(arc_displacement(arc(0), arc(1)),
                         wheel_displacement_covariance(dd, d_left, d_right, var_left, var_right));
}

int EkfSlam::add_landmark(const Vector2d& z, const Matrix2d& R) {
    // eq. (9.5)-(9.7): m = g(x, z); P_mm = Gx Pxx Gx^T + Gz R Gz^T; P_my = Gx P_xy.
    const Pose2 x = pose();
    const Matrix23d Gx = range_bearing_inverse_jacobian_pose(x, z);
    const Matrix2d Gz = range_bearing_inverse_jacobian_measurement(x, z);
    const int n = static_cast<int>(y_.size());
    VectorXd y(n + 2);
    y << y_, range_bearing_inverse(x, z);
    MatrixXd P(n + 2, n + 2);
    P.topLeftCorner(n, n) = P_;
    const MatrixXd cross = Gx * P_.topRows(3);  // 2 x n
    P.bottomLeftCorner(2, n) = cross;
    P.topRightCorner(n, 2) = cross.transpose();
    P.bottomRightCorner<2, 2>() = Gx * P_.topLeftCorner<3, 3>() * Gx.transpose() + Gz * R * Gz.transpose();
    y_ = y;
    P_ = P;
    return n_landmarks() - 1;
}

int EkfSlam::add_known_landmark(const Vector2d& m, const Matrix2d& P_m) {
    const int n = static_cast<int>(y_.size());
    VectorXd y(n + 2);
    y << y_, m;
    MatrixXd P = MatrixXd::Zero(n + 2, n + 2);
    P.topLeftCorner(n, n) = P_;
    P.bottomRightCorner<2, 2>() = P_m;
    y_ = y;
    P_ = P;
    return n_landmarks() - 1;
}

Vector2d EkfSlam::predict_reading(int i) const { return range_bearing(pose(), landmark(i)); }

MatrixXd EkfSlam::reading_jacobian(int i) const {
    // eq. (9.8): non-zero only in the robot columns and in the two columns of landmark i.
    MatrixXd H = MatrixXd::Zero(2, y_.size());
    H.leftCols<3>() = range_bearing_jacobian_pose(pose(), landmark(i));
    H.block<2, 2>(0, 3 + 2 * i) = range_bearing_jacobian_landmark(pose(), landmark(i));
    return H;
}

Innovation EkfSlam::update(const std::vector<Observation>& obs, const Matrix2d& R) {
    if (obs.empty()) return {};
    const int k = static_cast<int>(obs.size()), n = static_cast<int>(y_.size());
    VectorXd nu(2 * k);
    MatrixXd H(2 * k, n), Rs = MatrixXd::Zero(2 * k, 2 * k);
    for (int a = 0; a < k; ++a) {
        if (obs[a].landmark < 0 || obs[a].landmark >= n_landmarks()) throw std::out_of_range("no such landmark");
        nu.segment<2>(2 * a) = measurement_residual(obs[a].z, predict_reading(obs[a].landmark), {1});
        H.middleRows<2>(2 * a) = reading_jacobian(obs[a].landmark);
        Rs.block<2, 2>(2 * a, 2 * a) = R;
    }
    // eq. (9.9): the usual (E)KF update on the full state, Joseph form.
    Innovation inn;
    inn.nu = nu;
    inn.S = symmetrize(H * P_ * H.transpose() + Rs);
    inn.K = inn.S.ldlt().solve(H * P_).transpose();
    inn.nis = nu.dot(inn.S.ldlt().solve(nu));
    y_ += inn.K * nu;
    y_(2) = wrap_angle(y_(2));
    const MatrixXd IKH = MatrixXd::Identity(n, n) - inn.K * H;
    P_ = symmetrize(IKH * P_ * IKH.transpose() + inn.K * Rs * inn.K.transpose());
    return inn;
}

Innovation EkfSlam::process_known(const std::vector<Observation>& obs, const Matrix2d& R) {
    std::vector<Observation> known, fresh;
    for (const auto& o : obs) {
        const auto it = ids_.find(o.landmark);
        if (it != ids_.end()) {
            known.push_back({it->second, o.z});
        } else {
            fresh.push_back(o);
        }
    }
    const Innovation inn = update(known, R);
    for (const auto& o : fresh) ids_[o.landmark] = add_landmark(o.z, R);
    return inn;
}

AssociationProblem EkfSlam::association_problem(const std::vector<Observation>& obs, const Matrix2d& R) const {
    AssociationProblem p;
    for (const auto& o : obs) p.observations.push_back(o.z);
    for (int i = 0; i < n_landmarks(); ++i) {
        p.predictions.push_back(predict_reading(i));
        p.jacobians.push_back(reading_jacobian(i));
    }
    p.P = P_;
    p.R = R;
    p.angle_indices = {1};
    return p;
}

ScanResult EkfSlam::process_jcbb(const std::vector<Observation>& obs, const Matrix2d& R, double alpha,
                                 double new_alpha) {
    std::vector<int> pairing(obs.size(), -1);
    std::vector<bool> candidate(obs.size(), true);  // unpaired and far from every landmark
    ScanResult result;
    if (n_landmarks() > 0) {
        const AssociationProblem p = association_problem(obs, R);
        const Association a = associate_jcbb(p, alpha);
        pairing = a.pairing;
        result.joint_nis = a.joint_nis;
        const double new_gate = chi2_quantile(1.0 - new_alpha, 2);
        for (size_t i = 0; i < obs.size(); ++i) {
            for (int j = 0; j < n_landmarks() && pairing[i] < 0; ++j) {
                if (individual_nis(p, static_cast<int>(i), j) < new_gate) candidate[i] = false;
            }
        }
    }
    std::vector<Observation> paired;
    for (size_t i = 0; i < obs.size(); ++i) {
        if (pairing[i] >= 0) paired.push_back({pairing[i], obs[i].z});
    }
    update(paired, R);
    for (size_t i = 0; i < obs.size(); ++i) {
        const bool fresh = pairing[i] < 0 && candidate[i];
        result.landmark.push_back(pairing[i] >= 0 ? pairing[i] : (fresh ? add_landmark(obs[i].z, R) : -1));
        result.is_new.push_back(fresh);
    }
    return result;
}

MatrixXd EkfSlam::landmarks() const {
    MatrixXd m(n_landmarks(), 2);
    for (int i = 0; i < n_landmarks(); ++i) m.row(i) = landmark(i).transpose();
    return m;
}

}  // namespace state_estimation
