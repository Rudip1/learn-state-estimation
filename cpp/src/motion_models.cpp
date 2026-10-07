#include "state_estimation/motion_models.hpp"

#include <cmath>
#include <stdexcept>

namespace state_estimation {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kSmallAngle = 1e-4;

double normal_density(double x, double var) {
    if (var <= 0.0) var = 1e-300;
    return std::exp(-0.5 * x * x / var) / std::sqrt(2.0 * kPi * var);
}
}  // namespace

Vector2d DifferentialDrive::wheel_to_body(double omega_left, double omega_right) const {
    // eq. (2.8)
    return {wheel_radius * (omega_right + omega_left) / 2.0,
            wheel_radius * (omega_right - omega_left) / wheel_base};
}

Vector2d DifferentialDrive::body_to_wheel(double v, double omega) const {
    return {(v - omega * wheel_base / 2.0) / wheel_radius, (v + omega * wheel_base / 2.0) / wheel_radius};
}

Vector2d DifferentialDrive::wheel_travel_to_arc(double d_left, double d_right) const {
    return {(d_right + d_left) / 2.0, (d_right - d_left) / wheel_base};  // eq. (2.9)
}

Matrix2d DifferentialDrive::wheel_travel_to_arc_jacobian() const {
    Matrix2d J;
    J << 0.5, 0.5, -1.0 / wheel_base, 1.0 / wheel_base;
    return J;
}

double DifferentialDrive::ticks_to_distance(int ticks) const {
    if (ticks_per_revolution <= 0) throw std::logic_error("encoder resolution not set");
    return 2.0 * kPi * wheel_radius * ticks / ticks_per_revolution;
}

Pose2 arc_displacement(double s, double phi) {
    // eq. (2.11): [s sin(phi)/phi, s (1 - cos(phi))/phi, phi]
    double a, b;  // sin(phi)/phi, (1 - cos(phi))/phi
    if (std::abs(phi) < kSmallAngle) {
        const double p2 = phi * phi;
        a = 1.0 - p2 / 6.0 + p2 * p2 / 120.0;
        b = phi / 2.0 - phi * p2 / 24.0;
    } else {
        a = std::sin(phi) / phi;
        b = (1.0 - std::cos(phi)) / phi;
    }
    return {s * a, s * b, phi};
}

Matrix32d arc_displacement_jacobian(double s, double phi) {
    // eq. (2.12)
    double a, b, da, db;
    if (std::abs(phi) < kSmallAngle) {
        const double p2 = phi * phi;
        a = 1.0 - p2 / 6.0 + p2 * p2 / 120.0;
        b = phi / 2.0 - phi * p2 / 24.0;
        da = -phi / 3.0 + phi * p2 / 30.0;
        db = 0.5 - p2 / 8.0 + p2 * p2 / 144.0;
    } else {
        const double c = std::cos(phi), sn = std::sin(phi);
        a = sn / phi;
        b = (1.0 - c) / phi;
        da = (phi * c - sn) / (phi * phi);
        db = (phi * sn - (1.0 - c)) / (phi * phi);
    }
    Matrix32d J;
    J << a, s * da,  //
        b, s * db,   //
        0.0, 1.0;
    return J;
}

Pose2 velocity_motion(const Pose2& x, double v, double omega, double dt) {
    // eq. (2.13): the arc model is compounding with the arc displacement.
    return compose(x, arc_displacement(v * dt, omega * dt));
}

Matrix3d velocity_motion_jacobian_pose(const Pose2& x, double v, double omega, double dt) {
    return compose_jacobian_1(x, arc_displacement(v * dt, omega * dt));
}

Matrix32d velocity_motion_jacobian_control(const Pose2& x, double v, double omega, double dt) {
    const Pose2 d = arc_displacement(v * dt, omega * dt);
    return compose_jacobian_2(x, d) * arc_displacement_jacobian(v * dt, omega * dt) * dt;
}

Pose2 sample_velocity_motion(const Pose2& x, double v, double omega, double dt,
                             const std::array<double, 6>& alpha, Rng& rng) {
    // eq. (2.14), Thrun et al. table 5.3
    const double v2 = v * v, w2 = omega * omega;
    const double v_hat = v + std::sqrt(alpha[0] * v2 + alpha[1] * w2) * rng.normal();
    const double w_hat = omega + std::sqrt(alpha[2] * v2 + alpha[3] * w2) * rng.normal();
    const double g_hat = std::sqrt(alpha[4] * v2 + alpha[5] * w2) * rng.normal();
    Pose2 out = velocity_motion(x, v_hat, w_hat, dt);
    out.z() = wrap_angle(out.z() + g_hat * dt);
    return out;
}

Vector3d odometry_to_rtr(const Pose2& odom_prev, const Pose2& odom_curr) {
    // eq. (2.15)
    const double dx = odom_curr.x() - odom_prev.x(), dy = odom_curr.y() - odom_prev.y();
    const double trans = std::hypot(dx, dy);
    // For a pure rotation the direction of travel is undefined: put the whole turn into rot1 and
    // leave rot2 = 0, otherwise atan2(0, 0) would invent a rotation.
    const double rot1 = trans < 1e-9 ? wrap_angle(odom_curr.z() - odom_prev.z())
                                     : wrap_angle(std::atan2(dy, dx) - odom_prev.z());
    const double rot2 = wrap_angle(odom_curr.z() - odom_prev.z() - rot1);
    return {rot1, trans, rot2};
}

Pose2 apply_rtr(const Pose2& x, const Vector3d& rtr) {
    // eq. (2.16)
    const double heading = x.z() + rtr(0);
    return {x.x() + rtr(1) * std::cos(heading), x.y() + rtr(1) * std::sin(heading),
            wrap_angle(x.z() + rtr(0) + rtr(2))};
}

Pose2 sample_odometry_motion(const Pose2& x, const Pose2& odom_prev, const Pose2& odom_curr,
                             const std::array<double, 4>& alpha, Rng& rng) {
    // eq. (2.17), Thrun et al. table 5.6
    const Vector3d d = odometry_to_rtr(odom_prev, odom_curr);
    const double r1 = d(0), t = d(1), r2 = d(2);
    Vector3d noisy;
    noisy(0) = r1 - std::sqrt(alpha[0] * r1 * r1 + alpha[1] * t * t) * rng.normal();
    noisy(1) = t - std::sqrt(alpha[2] * t * t + alpha[3] * (r1 * r1 + r2 * r2)) * rng.normal();
    noisy(2) = r2 - std::sqrt(alpha[0] * r2 * r2 + alpha[1] * t * t) * rng.normal();
    return apply_rtr(x, noisy);
}

double odometry_motion_probability(const Pose2& x_curr, const Pose2& x_prev, const Pose2& odom_prev,
                                   const Pose2& odom_curr, const std::array<double, 4>& alpha) {
    // eq. (2.18), Thrun et al. table 5.5
    const Vector3d u = odometry_to_rtr(odom_prev, odom_curr);
    const Vector3d h = odometry_to_rtr(x_prev, x_curr);
    const double r1 = h(0), t = h(1), r2 = h(2);
    const double p1 = normal_density(wrap_angle(u(0) - r1), alpha[0] * r1 * r1 + alpha[1] * t * t);
    const double p2 = normal_density(u(1) - t, alpha[2] * t * t + alpha[3] * (r1 * r1 + r2 * r2));
    const double p3 = normal_density(wrap_angle(u(2) - r2), alpha[0] * r2 * r2 + alpha[1] * t * t);
    return p1 * p2 * p3;
}

void DeadReckoning::predict_displacement(const Pose2& u, const Matrix3d& Q) {
    // eq. (2.19)-(2.20): Jacobians at the current estimate, before the mean is overwritten.
    const Matrix3d J1 = compose_jacobian_1(x_, u);
    const Matrix3d J2 = compose_jacobian_2(x_, u);
    x_ = compose(x_, u);
    P_ = J1 * P_ * J1.transpose() + J2 * Q * J2.transpose();
    P_ = 0.5 * (P_ + P_.transpose());
}

Matrix3d wheel_displacement_covariance(const DifferentialDrive& dd, double d_left, double d_right,
                                       double var_left, double var_right) {
    // eq. (2.21): Q_u = J diag(var_l, var_r) J^T with J = d(arc displacement)/d(s, phi) d(s, phi)/d(d_l, d_r)
    const Vector2d arc = dd.wheel_travel_to_arc(d_left, d_right);
    const Matrix32d J = arc_displacement_jacobian(arc(0), arc(1)) * dd.wheel_travel_to_arc_jacobian();
    return J * Vector2d(var_left, var_right).asDiagonal() * J.transpose();
}

void DeadReckoning::predict_wheels(const DifferentialDrive& dd, double d_left, double d_right,
                                   double var_left, double var_right) {
    const Vector2d arc = dd.wheel_travel_to_arc(d_left, d_right);
    predict_displacement(arc_displacement(arc(0), arc(1)),
                         wheel_displacement_covariance(dd, d_left, d_right, var_left, var_right));
}

}  // namespace state_estimation
