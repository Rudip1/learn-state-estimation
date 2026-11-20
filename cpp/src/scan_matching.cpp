#include "state_estimation/scan_matching.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "state_estimation/simulation.hpp"

namespace state_estimation {

namespace {

/// d(R(theta) p)/d theta = R'(theta) p.
Vector2d rotation_derivative_times(double theta, const Vector2d& p) {
    const double c = std::cos(theta), s = std::sin(theta);
    return {-s * p.x() - c * p.y(), c * p.x() - s * p.y()};
}

/// Distance along the ray o + s d (|d| = 1) to the segment [a, b]; +inf if they do not meet ahead of o.
double ray_segment_distance(const Vector2d& o, const Vector2d& d, const Segment& seg) {
    const Vector2d e = seg.b - seg.a;
    const double den = d.x() * e.y() - d.y() * e.x();
    if (std::abs(den) < 1e-12) return std::numeric_limits<double>::infinity();
    const Vector2d w = seg.a - o;
    const double s = (w.x() * e.y() - w.y() * e.x()) / den;   // along the ray
    const double u = (w.x() * d.y() - w.y() * d.x()) / den;   // along the segment
    if (s <= 0.0 || u < 0.0 || u > 1.0) return std::numeric_limits<double>::infinity();
    return s;
}

struct Pairs {
    std::vector<int> current;
    std::vector<int> reference;
};

/// Pair each current point, moved by x, with its nearest reference point; reject pairs farther than d_max.
Pairs associate(const MatrixXd& reference, const MatrixXd& current, const Pose2& x, double d_max) {
    Pairs pairs;
    const double d2 = d_max * d_max;
    for (int i = 0; i < current.rows(); ++i) {
        const Vector2d p = transform_point(x, current.row(i).transpose());
        const auto [j, dist2] = nearest_neighbour(reference, p);
        if (j >= 0 && dist2 <= d2) {
            pairs.current.push_back(i);
            pairs.reference.push_back(j);
        }
    }
    return pairs;
}

/// The pairing has reached a fixed point: the same pairs as in the previous iteration and an already small step.
/// Waiting for the step to fall below a tight tolerance is pointless once the pairs stop changing, and with
/// nearest-neighbour pairing the step may never get there (pairs flip back and forth by one point).
bool same_pairs(const Pairs& a, const Pairs& b, double step) {
    return step < 1e-3 && a.current == b.current && a.reference == b.reference;
}

void check_inputs(const MatrixXd& reference, const MatrixXd& current) {
    if (reference.cols() != 2 || current.cols() != 2) throw std::invalid_argument("scans must be n x 2");
    if (reference.rows() < 3 || current.rows() < 3) throw std::invalid_argument("scans need at least 3 points");
}

}  // namespace

MatrixXd simulate_scan(const Pose2& x, const std::vector<Segment>& walls, int n_beams, double fov,
                       double max_range, double sigma_range, Rng& rng) {
    std::vector<Vector2d> hits;
    const Vector2d o = x.head<2>();
    for (int k = 0; k < n_beams; ++k) {
        const double phi = n_beams > 1 ? -fov / 2.0 + fov * k / (n_beams - 1) : 0.0;
        const Vector2d d(std::cos(x.z() + phi), std::sin(x.z() + phi));
        double r = std::numeric_limits<double>::infinity();
        for (const auto& w : walls) r = std::min(r, ray_segment_distance(o, d, w));
        if (!std::isfinite(r) || r > max_range) continue;
        r += rng.normal(0.0, sigma_range);
        hits.emplace_back(r * std::cos(phi), r * std::sin(phi));
    }
    MatrixXd out(hits.size(), 2);
    for (size_t i = 0; i < hits.size(); ++i) out.row(i) = hits[i].transpose();
    return out;
}

MatrixXd transform_points(const Pose2& x, const MatrixXd& points) {
    MatrixXd out(points.rows(), 2);
    for (int i = 0; i < points.rows(); ++i) out.row(i) = transform_point(x, points.row(i).transpose()).transpose();
    return out;
}

Pose2 align_point_sets(const MatrixXd& P, const MatrixXd& Q) {
    if (P.rows() != Q.rows() || P.cols() != 2 || Q.cols() != 2 || P.rows() < 2)
        throw std::invalid_argument("align_point_sets: need two n x 2 matrices with n >= 2");
    const Vector2d p_bar = P.colwise().mean().transpose();  // eq. (10.3)
    const Vector2d q_bar = Q.colwise().mean().transpose();
    Matrix2d S = Matrix2d::Zero();  // cross-covariance sum (p - p_bar)(q - q_bar)^T, eq. (10.4)
    for (int i = 0; i < P.rows(); ++i)
        S += (P.row(i).transpose() - p_bar) * (Q.row(i).transpose() - q_bar).transpose();
    const double theta = std::atan2(S(0, 1) - S(1, 0), S(0, 0) + S(1, 1));  // eq. (10.5)
    const Vector2d t = q_bar - rotation(theta) * p_bar;
    return {t.x(), t.y(), theta};
}

std::pair<int, double> nearest_neighbour(const MatrixXd& cloud, const Vector2d& p) {
    int best = -1;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (int j = 0; j < cloud.rows(); ++j) {
        const double d2 = (cloud.row(j).transpose() - p).squaredNorm();
        if (d2 < best_d2) {
            best_d2 = d2;
            best = j;
        }
    }
    return {best, best_d2};
}

MatrixXd estimate_normals(const MatrixXd& points, int k, double max_radius) {
    const int n = static_cast<int>(points.rows());
    k = std::max(2, std::min(k, n));
    MatrixXd normals(n, 2);
    std::vector<int> idx(n);
    std::vector<double> d2(n);
    for (int i = 0; i < n; ++i) {
        const Vector2d p = points.row(i).transpose();
        for (int j = 0; j < n; ++j) d2[j] = (points.row(j).transpose() - p).squaredNorm();
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return d2[a] < d2[b]; });
        int used = 0;  // neighbours within the radius: points of other surfaces must not tilt the normal
        while (used < k && d2[idx[used]] <= max_radius * max_radius) ++used;
        if (used < 3) {
            normals.row(i).setConstant(std::numeric_limits<double>::quiet_NaN());
            continue;
        }
        Vector2d mean = Vector2d::Zero();
        for (int m = 0; m < used; ++m) mean += points.row(idx[m]).transpose();
        mean /= used;
        Matrix2d C = Matrix2d::Zero();
        for (int m = 0; m < used; ++m) {
            const Vector2d d = points.row(idx[m]).transpose() - mean;
            C += d * d.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Matrix2d> eig(C);
        Vector2d nrm = eig.eigenvectors().col(0);  // smallest eigenvalue: across the line
        if (nrm.dot(-p) < 0.0) nrm = -nrm;         // towards the sensor
        normals.row(i) = nrm.transpose();
    }
    return normals;
}

IcpResult icp_point_to_point(const MatrixXd& reference, const MatrixXd& current, const Pose2& guess,
                             const IcpOptions& options) {
    check_inputs(reference, current);
    IcpResult result;
    Pose2 x = guess;
    Pairs pairs, previous;
    for (int it = 0; it < options.max_iterations; ++it) {
        previous = pairs;
        pairs = associate(reference, current, x, options.max_correspondence_distance);
        result.iterations = it + 1;
        if (pairs.current.size() < 3) break;
        MatrixXd P(pairs.current.size(), 2), Q(pairs.current.size(), 2);
        for (size_t m = 0; m < pairs.current.size(); ++m) {
            P.row(m) = current.row(pairs.current[m]);
            Q.row(m) = reference.row(pairs.reference[m]);
        }
        const Pose2 x_new = align_point_sets(P, Q);
        const double dt = (x_new.head<2>() - x.head<2>()).norm(), dth = std::abs(wrap_angle(x_new.z() - x.z()));
        x = x_new;
        if (same_pairs(pairs, previous, std::max(dt, dth)) || (dt < options.tolerance && dth < options.tolerance)) {
            result.converged = true;
            break;
        }
    }
    // Residuals, Jacobian and covariance at the final estimate, eq. (10.10).
    pairs = associate(reference, current, x, options.max_correspondence_distance);
    const int n = static_cast<int>(pairs.current.size());
    result.pose = x;
    result.n_pairs = n;
    if (n >= 2) {
        Matrix3d JtJ = Matrix3d::Zero();
        double sum_r2 = 0.0;
        for (int m = 0; m < n; ++m) {
            const Vector2d p = current.row(pairs.current[m]).transpose();
            const Vector2d r = transform_point(x, p) - reference.row(pairs.reference[m]).transpose();
            Matrix23d J;
            J << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0;
            J.col(2) = rotation_derivative_times(x.z(), p);
            JtJ += J.transpose() * J;
            sum_r2 += r.squaredNorm();
        }
        result.rmse = std::sqrt(sum_r2 / n);
        result.hessian = JtJ;
        const double s2 = sum_r2 / std::max(1, 2 * n - 3);
        result.covariance = s2 * JtJ.inverse();
    } else {
        result.converged = false;
    }
    return result;
}

IcpResult icp_point_to_line(const MatrixXd& reference, const MatrixXd& current, const Pose2& guess,
                            const IcpOptions& options) {
    check_inputs(reference, current);
    const MatrixXd normals = estimate_normals(reference, options.normal_neighbours, options.normal_radius);
    IcpResult result;
    Pose2 x = guess;

    // Gauss-Newton normal equations for the point-to-line residuals at pose x, eq. (10.7)-(10.8).
    Pairs pairs, previous;
    auto linearise = [&](const Pose2& pose, Matrix3d& JtJ, Vector3d& Jtr, double& sum_r2) {
        pairs = associate(reference, current, pose, options.max_correspondence_distance);
        JtJ.setZero();
        Jtr.setZero();
        sum_r2 = 0.0;
        int used = 0;
        for (size_t m = 0; m < pairs.current.size(); ++m) {
            const Vector2d p = current.row(pairs.current[m]).transpose();
            const Vector2d q = reference.row(pairs.reference[m]).transpose();
            const Vector2d nrm = normals.row(pairs.reference[m]).transpose();
            if (!nrm.allFinite()) continue;  // no reliable line through this reference point
            const double r = nrm.dot(transform_point(pose, p) - q);
            const Eigen::RowVector3d J(nrm.x(), nrm.y(), nrm.dot(rotation_derivative_times(pose.z(), p)));
            JtJ += J.transpose() * J;
            Jtr += J.transpose() * r;
            sum_r2 += r * r;
            ++used;
        }
        return used;
    };

    Matrix3d JtJ;
    Vector3d Jtr;
    double sum_r2 = 0.0;
    for (int it = 0; it < options.max_iterations; ++it) {
        previous = pairs;
        const int n = linearise(x, JtJ, Jtr, sum_r2);
        result.iterations = it + 1;
        if (n < 3) break;
        const Vector3d dx = JtJ.ldlt().solve(-Jtr);  // eq. (10.9)
        x += dx;
        x.z() = wrap_angle(x.z());
        const double step = std::max(dx.head<2>().norm(), std::abs(dx.z()));
        if (same_pairs(pairs, previous, step) || step < options.tolerance) {
            result.converged = true;
            break;
        }
    }
    const int n = linearise(x, JtJ, Jtr, sum_r2);
    result.pose = x;
    result.n_pairs = n;
    if (n >= 4) {
        result.rmse = std::sqrt(sum_r2 / n);
        result.hessian = JtJ;
        result.covariance = (sum_r2 / (n - 3)) * JtJ.inverse();  // eq. (10.10)
    } else {
        result.converged = false;
    }
    return result;
}

std::vector<Segment> office_world() {
    auto s = [](double ax, double ay, double bx, double by) { return Segment{Vector2d(ax, ay), Vector2d(bx, by)}; };
    return {s(-8, -6, 8, -6),     s(8, -6, 8, 6),      s(8, 6, -8, 6),        s(-8, 6, -8, -6),    // outer walls
            s(-3, -6, -3, -4.9),  s(-3, -3.1, -3, 3.7), s(-3, 5.3, -3, 6),                          // inner wall, 2 doors
            s(3, -6, 3, -4.9),    s(3, -3.3, 3, 3.4),   s(3, 5.0, 3, 6),                            // inner wall, 2 doors
            s(-1, -2.5, 1, -2.5), s(0, 2.5, 1, 3.3),    s(-4.6, 0.5, -3.8, 1.3), s(3.8, -2, 4.6, -1.2),  // clutter
            s(-7.5, -1, -7.5, 1), s(7.4, 2, 7.4, 3.5)};
}

MatrixXd office_lap_controls(int laps, double dt) {
    std::vector<Vector2d> u;
    auto straight = [&](double length, double v) {
        for (int k = 0; k < static_cast<int>(length / (v * dt)); ++k) u.emplace_back(v, 0.0);
    };
    auto turn = [&](double angle, double v, double radius) {
        const int n = static_cast<int>(std::abs(angle) * radius / (v * dt));
        for (int k = 0; k < n; ++k) u.emplace_back(v, (angle > 0 ? 1.0 : -1.0) * v / radius);
    };
    constexpr double kHalfPi = 1.57079632679489661923;
    for (int lap = 0; lap < laps; ++lap)
        for (double length : {11.0, 7.0, 11.0, 7.0}) {
            straight(length, 0.5);
            turn(kHalfPi, 0.3, 0.6);
        }
    straight(3.0, 0.5);
    MatrixXd out(u.size(), 2);
    for (size_t k = 0; k < u.size(); ++k) out.row(k) = u[k].transpose();
    return out;
}

LidarRun simulate_lidar_run(const Pose2& x0, const MatrixXd& controls, double dt, const DifferentialDrive& dd,
                            double var_per_meter, const std::vector<Segment>& walls, int every, int n_beams,
                            double max_range, double sigma_range, Rng& rng) {
    constexpr double kTwoPi = 6.28318530717958647692;
    LidarRun run;
    run.truth = integrate_controls(x0, controls, dt);
    run.wheel_travel = simulate_wheel_travel(controls, dt, dd, var_per_meter, rng);
    for (int k = 0; k <= controls.rows(); k += every) {
        run.keyframes.push_back(k);
        run.scans.push_back(simulate_scan(run.truth.row(k).transpose(), walls, n_beams, kTwoPi - 1e-3, max_range,
                                          sigma_range, rng));
        if (k == 0) {
            run.odometry.push_back(Pose2::Zero());
            run.odometry_cov.push_back(Matrix3d::Zero());
        } else {
            const DeadReckoningRun rel = run_dead_reckoning(Pose2::Zero(), Matrix3d::Zero(),
                                                            run.wheel_travel.middleRows(k - every, every), dd,
                                                            var_per_meter);
            run.odometry.push_back(rel.poses.bottomRows<1>().transpose());
            run.odometry_cov.push_back(rel.covariances.back());
        }
    }
    return run;
}

}  // namespace state_estimation
