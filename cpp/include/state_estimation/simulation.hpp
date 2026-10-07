#pragma once
/// \file simulation.hpp
/// Small simulator used by the examples, tests and notebooks: trajectories of a differential-drive robot
/// and the noisy wheel encoders that observe them. Sensors that observe landmarks are added in chapter 3
/// (measurement_models.hpp).

#include <Eigen/Dense>
#include <vector>

#include "state_estimation/gaussian.hpp"
#include "state_estimation/motion_models.hpp"

namespace state_estimation {

/// Constant controls (v, omega = v / radius) that drive a circle; steps x 2 matrix.
MatrixXd circle_controls(double v, double radius, int steps);

/// Controls of a figure eight: a counter-clockwise circle followed by a clockwise one, each in
/// steps_per_loop steps of length dt. Returns (2 steps_per_loop) x 2.
MatrixXd figure_eight_controls(double radius, int steps_per_loop, double dt);

/// Integrate controls (rows (v, omega)) exactly with the arc model; returns the (N+1) x 3 poses.
MatrixXd integrate_controls(const Pose2& x0, const MatrixXd& controls, double dt);

/// Wheel travel (d_l, d_r) measured by encoders while executing the controls. Each wheel's travel gets
/// zero-mean Gaussian noise with variance var_per_meter * |d|; with a finite encoder resolution the
/// readings are quantised to whole ticks of the accumulated wheel rotation. Returns N x 2.
MatrixXd simulate_wheel_travel(const MatrixXd& controls, double dt, const DifferentialDrive& dd,
                               double var_per_meter, Rng& rng);

/// Result of a dead-reckoning run: (N+1) x 3 poses and their covariances.
struct DeadReckoningRun {
    MatrixXd poses;
    std::vector<Matrix3d> covariances;
};

/// Dead reckoning from wheel travel, with the encoder noise model var = var_per_meter * |d| (plus the
/// quantisation variance tick^2/12 when the encoder resolution is finite).
DeadReckoningRun run_dead_reckoning(const Pose2& x0, const Matrix3d& P0, const MatrixXd& wheel_travel,
                                    const DifferentialDrive& dd, double var_per_meter);

}  // namespace state_estimation
