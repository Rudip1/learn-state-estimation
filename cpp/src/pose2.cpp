#include "state_estimation/pose2.hpp"

#include <cmath>

namespace state_estimation {

double wrap_angle(double a) {
    constexpr double kPi = 3.14159265358979323846;
    a = std::fmod(a + kPi, 2.0 * kPi);
    if (a <= 0.0) a += 2.0 * kPi;
    return a - kPi;
}

Matrix2d rotation(double theta) {
    const double c = std::cos(theta), s = std::sin(theta);
    Matrix2d R;
    R << c, -s, s, c;
    return R;
}

Pose2 compose(const Pose2& a, const Pose2& b) {
    // eq. (2.1)
    const double c = std::cos(a.z()), s = std::sin(a.z());
    return {a.x() + c * b.x() - s * b.y(), a.y() + s * b.x() + c * b.y(), wrap_angle(a.z() + b.z())};
}

Pose2 inverse(const Pose2& a) {
    // eq. (2.4)
    const double c = std::cos(a.z()), s = std::sin(a.z());
    return {-c * a.x() - s * a.y(), s * a.x() - c * a.y(), wrap_angle(-a.z())};
}

Pose2 between(const Pose2& a, const Pose2& b) { return compose(inverse(a), b); }

Matrix3d compose_jacobian_1(const Pose2& a, const Pose2& b) {
    // eq. (2.2)
    const double c = std::cos(a.z()), s = std::sin(a.z());
    Matrix3d J;
    J << 1, 0, -s * b.x() - c * b.y(),  //
        0, 1, c * b.x() - s * b.y(),    //
        0, 0, 1;
    return J;
}

Matrix3d compose_jacobian_2(const Pose2& a, const Pose2& /*b*/) {
    // eq. (2.3)
    const double c = std::cos(a.z()), s = std::sin(a.z());
    Matrix3d J;
    J << c, -s, 0,  //
        s, c, 0,    //
        0, 0, 1;
    return J;
}

Matrix3d inverse_jacobian(const Pose2& a) {
    // eq. (2.5)
    const double c = std::cos(a.z()), s = std::sin(a.z());
    Matrix3d J;
    J << -c, -s, s * a.x() - c * a.y(),  //
        s, -c, c * a.x() + s * a.y(),    //
        0, 0, -1;
    return J;
}

Vector2d transform_point(const Pose2& x, const Vector2d& p) {
    return x.head<2>() + rotation(x.z()) * p;  // eq. (2.6)
}

Matrix23d transform_point_jacobian_pose(const Pose2& x, const Vector2d& p) {
    // eq. (2.7)
    const double c = std::cos(x.z()), s = std::sin(x.z());
    Matrix23d J;
    J << 1, 0, -s * p.x() - c * p.y(),  //
        0, 1, c * p.x() - s * p.y();
    return J;
}

Matrix2d transform_point_jacobian_point(const Pose2& x) { return rotation(x.z()); }

Vector2d inverse_transform_point(const Pose2& x, const Vector2d& m) {
    return rotation(x.z()).transpose() * (m - x.head<2>());
}

Matrix23d inverse_transform_point_jacobian_pose(const Pose2& x, const Vector2d& m) {
    const double c = std::cos(x.z()), s = std::sin(x.z());
    const double dx = m.x() - x.x(), dy = m.y() - x.y();
    Matrix23d J;
    J << -c, -s, -s * dx + c * dy,  //
        s, -c, -c * dx - s * dy;
    return J;
}

Matrix3d to_matrix(const Pose2& x) {
    Matrix3d T = Matrix3d::Identity();
    T.topLeftCorner<2, 2>() = rotation(x.z());
    T.topRightCorner<2, 1>() = x.head<2>();
    return T;
}

Pose2 from_matrix(const Matrix3d& T) { return {T(0, 2), T(1, 2), std::atan2(T(1, 0), T(0, 0))}; }

}  // namespace state_estimation
