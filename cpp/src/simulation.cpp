#include "state_estimation/simulation.hpp"

#include <cmath>

namespace state_estimation {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

MatrixXd circle_controls(double v, double radius, int steps) {
    MatrixXd u(steps, 2);
    u.col(0).setConstant(v);
    u.col(1).setConstant(v / radius);
    return u;
}

MatrixXd figure_eight_controls(double radius, int steps_per_loop, double dt) {
    const double omega = 2.0 * kPi / (steps_per_loop * dt);
    const double v = omega * radius;
    MatrixXd u(2 * steps_per_loop, 2);
    u.topRows(steps_per_loop) = circle_controls(v, radius, steps_per_loop);
    u.bottomRows(steps_per_loop) = circle_controls(v, -radius, steps_per_loop);
    return u;
}

MatrixXd integrate_controls(const Pose2& x0, const MatrixXd& controls, double dt) {
    MatrixXd poses(controls.rows() + 1, 3);
    Pose2 x = x0;
    poses.row(0) = x.transpose();
    for (int k = 0; k < controls.rows(); ++k) {
        x = velocity_motion(x, controls(k, 0), controls(k, 1), dt);
        poses.row(k + 1) = x.transpose();
    }
    return poses;
}

MatrixXd simulate_wheel_travel(const MatrixXd& controls, double dt, const DifferentialDrive& dd,
                               double var_per_meter, Rng& rng) {
    MatrixXd out(controls.rows(), 2);
    const bool quantised = dd.ticks_per_revolution > 0;
    const double tick = quantised ? dd.ticks_to_distance(1) : 0.0;
    double travelled[2] = {0.0, 0.0};  // accumulated (noisy) wheel travel
    for (int k = 0; k < controls.rows(); ++k) {
        const double s = controls(k, 0) * dt, phi = controls(k, 1) * dt;
        const double d[2] = {s - phi * dd.wheel_base / 2.0, s + phi * dd.wheel_base / 2.0};
        for (int w = 0; w < 2; ++w) {
            const double noisy = d[w] + std::sqrt(var_per_meter * std::abs(d[w])) * rng.normal();
            if (quantised) {
                const double before = std::floor(travelled[w] / tick);
                travelled[w] += noisy;
                out(k, w) = (std::floor(travelled[w] / tick) - before) * tick;
            } else {
                travelled[w] += noisy;
                out(k, w) = noisy;
            }
        }
    }
    return out;
}

DeadReckoningRun run_dead_reckoning(const Pose2& x0, const Matrix3d& P0, const MatrixXd& wheel_travel,
                                    const DifferentialDrive& dd, double var_per_meter) {
    const double tick = dd.ticks_per_revolution > 0 ? dd.ticks_to_distance(1) : 0.0;
    DeadReckoning dr(x0, P0);
    DeadReckoningRun run;
    run.poses.resize(wheel_travel.rows() + 1, 3);
    run.poses.row(0) = x0.transpose();
    run.covariances.push_back(P0);
    for (int k = 0; k < wheel_travel.rows(); ++k) {
        const double dl = wheel_travel(k, 0), dr_ = wheel_travel(k, 1);
        dr.predict_wheels(dd, dl, dr_, var_per_meter * std::abs(dl) + tick * tick / 12.0,
                          var_per_meter * std::abs(dr_) + tick * tick / 12.0);
        run.poses.row(k + 1) = dr.pose().transpose();
        run.covariances.push_back(dr.covariance());
    }
    return run;
}

}  // namespace state_estimation
