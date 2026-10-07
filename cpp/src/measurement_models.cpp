#include "state_estimation/measurement_models.hpp"

#include <cmath>
#include <stdexcept>

namespace state_estimation {

Vector2d range_bearing(const Pose2& x, const Vector2d& m) {
    // eq. (3.4)
    const double dx = m.x() - x.x(), dy = m.y() - x.y();
    return {std::hypot(dx, dy), wrap_angle(std::atan2(dy, dx) - x.z())};
}

Matrix23d range_bearing_jacobian_pose(const Pose2& x, const Vector2d& m) {
    // eq. (3.5)
    const double dx = m.x() - x.x(), dy = m.y() - x.y();
    const double q = dx * dx + dy * dy, r = std::sqrt(q);
    if (r < 1e-12) throw std::domain_error("landmark at the sensor: bearing undefined");
    Matrix23d H;
    H << -dx / r, -dy / r, 0.0,  //
        dy / q, -dx / q, -1.0;
    return H;
}

Matrix2d range_bearing_jacobian_landmark(const Pose2& x, const Vector2d& m) {
    // eq. (3.5): the negative of the position block of the pose Jacobian
    return -range_bearing_jacobian_pose(x, m).leftCols<2>();
}

Vector2d range_bearing_inverse(const Pose2& x, const Vector2d& z) {
    // eq. (3.6)
    const double a = x.z() + z(1);
    return {x.x() + z(0) * std::cos(a), x.y() + z(0) * std::sin(a)};
}

Matrix23d range_bearing_inverse_jacobian_pose(const Pose2& x, const Vector2d& z) {
    // eq. (3.7)
    const double a = x.z() + z(1);
    Matrix23d G;
    G << 1.0, 0.0, -z(0) * std::sin(a),  //
        0.0, 1.0, z(0) * std::cos(a);
    return G;
}

Matrix2d range_bearing_inverse_jacobian_measurement(const Pose2& x, const Vector2d& z) {
    // eq. (3.7)
    const double a = x.z() + z(1), c = std::cos(a), s = std::sin(a);
    Matrix2d G;
    G << c, -z(0) * s,  //
        s, z(0) * c;
    return G;
}

Vector2d cartesian_feature(const Pose2& x, const Vector2d& m) { return inverse_transform_point(x, m); }

Matrix23d cartesian_feature_jacobian_pose(const Pose2& x, const Vector2d& m) {
    return inverse_transform_point_jacobian_pose(x, m);  // eq. (3.10)
}

Matrix2d cartesian_feature_jacobian_landmark(const Pose2& x) { return rotation(x.z()).transpose(); }

Matrix2d polar_to_cartesian_covariance(const Vector2d& z, const Matrix2d& R_polar) {
    const Matrix2d J = polar_to_cartesian_jacobian(z);  // eq. (3.11)
    return J * R_polar * J.transpose();
}

Vector2d position_fix(const Pose2& x, const Vector2d& lever_arm) { return transform_point(x, lever_arm); }

Matrix23d position_fix_jacobian(const Pose2& x, const Vector2d& lever_arm) {
    return transform_point_jacobian_pose(x, lever_arm);
}

double compass(const Pose2& x) { return wrap_angle(x.z()); }

VectorXd measurement_residual(const VectorXd& z, const VectorXd& z_hat, const std::vector<int>& angle_indices) {
    VectorXd r = z - z_hat;
    for (int i : angle_indices) r(i) = wrap_angle(r(i));
    return r;
}

double measurement_log_likelihood(const VectorXd& z, const VectorXd& z_hat, const MatrixXd& R,
                                  const std::vector<int>& angle_indices) {
    // eq. (3.2)
    const VectorXd r = measurement_residual(z, z_hat, angle_indices);
    return gaussian_log_pdf(r, VectorXd::Zero(r.size()), R);
}

MatrixXd fisher_information(const MatrixXd& H, const MatrixXd& R) {
    // eq. (3.14)
    return H.transpose() * R.ldlt().solve(H);
}

Matrix2d RangeBearingSensor::R() const {
    return Vector2d(sigma_range * sigma_range, sigma_bearing * sigma_bearing).asDiagonal();
}

bool RangeBearingSensor::visible(const Pose2& x, const Vector2d& m) const {
    const Vector2d z = range_bearing(x, m);
    return z(0) <= max_range && std::abs(z(1)) <= fov / 2.0;
}

std::vector<Observation> RangeBearingSensor::observe(const Pose2& x, const MatrixXd& landmarks, Rng& rng) const {
    std::vector<Observation> out;
    for (int j = 0; j < landmarks.rows(); ++j) {
        const Vector2d m = landmarks.row(j).transpose();
        if (!visible(x, m)) continue;
        Vector2d z = range_bearing(x, m);
        z(0) += sigma_range * rng.normal();
        z(1) = wrap_angle(z(1) + sigma_bearing * rng.normal());
        out.push_back({j, z});
    }
    return out;
}

std::vector<Observation> to_cartesian(const std::vector<Observation>& polar) {
    std::vector<Observation> out;
    out.reserve(polar.size());
    for (const auto& o : polar) out.push_back({o.landmark, polar_to_cartesian(o.z)});
    return out;
}

}  // namespace state_estimation
