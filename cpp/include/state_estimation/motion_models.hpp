#pragma once
/// \file motion_models.hpp
/// Chapter 2 — motion models for a differential-drive robot: wheel kinematics, the arc (velocity)
/// model, the odometry (rotation-translation-rotation) model, the displacement-compounding model and dead
/// reckoning with covariance propagation. Equations: 1_theory/02_motion_models.md.

#include <Eigen/Dense>
#include <array>

#include "state_estimation/gaussian.hpp"
#include "state_estimation/pose2.hpp"

namespace state_estimation {

/// Geometry of a differential-drive robot.
struct DifferentialDrive {
    double wheel_radius = 0.1;   ///< r [m]
    double wheel_base = 0.5;     ///< b, distance between the wheels [m]
    int ticks_per_revolution = 0;  ///< encoder resolution; 0 means an ideal (continuous) encoder

    /// (v, omega) from wheel angular velocities (omega_l, omega_r), eq. (2.8).
    Vector2d wheel_to_body(double omega_left, double omega_right) const;
    /// (omega_l, omega_r) from (v, omega): the inverse of eq. (2.8).
    Vector2d body_to_wheel(double v, double omega) const;
    /// Arc length s and heading change phi from wheel travel distances (d_l, d_r), eq. (2.9).
    Vector2d wheel_travel_to_arc(double d_left, double d_right) const;
    /// Jacobian of wheel_travel_to_arc with respect to (d_l, d_r).
    Matrix2d wheel_travel_to_arc_jacobian() const;
    /// Distance travelled by a wheel for an encoder increment.
    double ticks_to_distance(int ticks) const;
};

/// Displacement [dx, dy, dtheta] in the robot frame produced by driving an arc of length s while turning
/// by phi, eq. (2.11). Uses a series expansion for |phi| -> 0.
Pose2 arc_displacement(double s, double phi);

/// Jacobian of arc_displacement with respect to (s, phi), eq. (2.12).
Matrix32d arc_displacement_jacobian(double s, double phi);

/// Velocity (arc) motion model: pose after driving with constant (v, omega) for dt, eq. (2.13).
Pose2 velocity_motion(const Pose2& x, double v, double omega, double dt);

/// Jacobian of velocity_motion with respect to the pose.
Matrix3d velocity_motion_jacobian_pose(const Pose2& x, double v, double omega, double dt);

/// Jacobian of velocity_motion with respect to the control (v, omega).
Matrix32d velocity_motion_jacobian_control(const Pose2& x, double v, double omega, double dt);

/// Sample from the velocity motion model of Thrun et al. (table 5.3). alpha = (a1..a6): the variances of
/// v, omega and the final rotation are a1 v^2 + a2 w^2, a3 v^2 + a4 w^2 and a5 v^2 + a6 w^2, eq. (2.14).
Pose2 sample_velocity_motion(const Pose2& x, double v, double omega, double dt,
                             const std::array<double, 6>& alpha, Rng& rng);

/// Odometry motion model: (rot1, trans, rot2) between two odometry poses, eq. (2.15).
Vector3d odometry_to_rtr(const Pose2& odom_prev, const Pose2& odom_curr);

/// Apply a (rot1, trans, rot2) increment to a pose, eq. (2.16).
Pose2 apply_rtr(const Pose2& x, const Vector3d& rtr);

/// Sample from the odometry motion model (Thrun et al., table 5.6) with variances
/// a1 rot1^2 + a2 trans^2, a3 trans^2 + a4 (rot1^2 + rot2^2), a1 rot2^2 + a2 trans^2, eq. (2.17).
Pose2 sample_odometry_motion(const Pose2& x, const Pose2& odom_prev, const Pose2& odom_curr,
                             const std::array<double, 4>& alpha, Rng& rng);

/// Density p(x_curr | x_prev, u) of the odometry motion model (Thrun et al., table 5.5), eq. (2.18).
double odometry_motion_probability(const Pose2& x_curr, const Pose2& x_prev, const Pose2& odom_prev,
                                   const Pose2& odom_curr, const std::array<double, 4>& alpha);

/// Dead reckoning: integrate displacement measurements and propagate their uncertainty with
/// P <- J1 P J1^T + J2 Q J2^T, eq. (2.19)-(2.20).
class DeadReckoning {
public:
    DeadReckoning(const Pose2& x0, const Matrix3d& P0) : x_(x0), P_(P0) {}

    /// One step with a robot-frame displacement u ~ N(u, Q).
    void predict_displacement(const Pose2& u, const Matrix3d& Q);

    /// One step from wheel travel distances with independent variances var_l, var_r, eq. (2.21).
    void predict_wheels(const DifferentialDrive& dd, double d_left, double d_right, double var_left,
                        double var_right);

    const Pose2& pose() const { return x_; }
    const Matrix3d& covariance() const { return P_; }

private:
    Pose2 x_;
    Matrix3d P_;
};

/// Covariance of the robot-frame displacement produced by wheel travel (d_l, d_r) with variances
/// (var_l, var_r): first-order propagation through eq. (2.9) and (2.11).
Matrix3d wheel_displacement_covariance(const DifferentialDrive& dd, double d_left, double d_right,
                                       double var_left, double var_right);

}  // namespace state_estimation
