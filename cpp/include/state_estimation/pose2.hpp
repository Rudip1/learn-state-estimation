#pragma once
/// \file pose2.hpp
/// Planar poses x = [x, y, theta]^T and their algebra: compounding (oplus), inversion (ominus), point
/// transformation (boxplus) and the Jacobians of each. Equations: 1_theory/02_motion_models.md, section 2.1.

#include <Eigen/Dense>

namespace state_estimation {

using Pose2 = Eigen::Vector3d;
using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Matrix23d = Eigen::Matrix<double, 2, 3>;
using Matrix32d = Eigen::Matrix<double, 3, 2>;

/// Wrap an angle to (-pi, pi].
double wrap_angle(double a);

/// Rotation matrix R(theta).
Matrix2d rotation(double theta);

/// Compounding a (+) b: the pose b, expressed in the frame of a, seen from the frame a is expressed in,
/// eq. (2.1).
Pose2 compose(const Pose2& a, const Pose2& b);

/// Inversion (-)a: the pose of the reference frame seen from a, eq. (2.4).
Pose2 inverse(const Pose2& a);

/// Relative pose (-)a (+) b of b seen from a.
Pose2 between(const Pose2& a, const Pose2& b);

/// Jacobian of a (+) b with respect to a, eq. (2.2).
Matrix3d compose_jacobian_1(const Pose2& a, const Pose2& b);

/// Jacobian of a (+) b with respect to b, eq. (2.3).
Matrix3d compose_jacobian_2(const Pose2& a, const Pose2& b);

/// Jacobian of (-)a with respect to a, eq. (2.5).
Matrix3d inverse_jacobian(const Pose2& a);

/// Point compounding x [+] p = t + R(theta) p: a point given in the robot frame, expressed in the world
/// frame, eq. (2.6).
Vector2d transform_point(const Pose2& x, const Vector2d& p);

/// Jacobian of transform_point with respect to the pose, eq. (2.7).
Matrix23d transform_point_jacobian_pose(const Pose2& x, const Vector2d& p);

/// Jacobian of transform_point with respect to the point (= R(theta)), eq. (2.7).
Matrix2d transform_point_jacobian_point(const Pose2& x);

/// Inverse point transformation R(theta)^T (m - t): a world point expressed in the robot frame.
Vector2d inverse_transform_point(const Pose2& x, const Vector2d& m);

/// Jacobian of inverse_transform_point with respect to the pose.
Matrix23d inverse_transform_point_jacobian_pose(const Pose2& x, const Vector2d& m);

/// Homogeneous 3x3 matrix of a pose (used to check the algebra against matrix products).
Matrix3d to_matrix(const Pose2& x);

/// Pose from a homogeneous 3x3 matrix.
Pose2 from_matrix(const Matrix3d& T);

}  // namespace state_estimation
