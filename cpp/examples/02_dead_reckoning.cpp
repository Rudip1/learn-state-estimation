// Chapter 2 example: dead reckoning of a differential-drive robot on a figure eight from noisy, quantised
// wheel encoders. Prints the error and the predicted standard deviations every quarter of the run.
// Usage: example_02_dead_reckoning [encoder_variance_per_meter] [seed]
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "state_estimation/simulation.hpp"

using namespace state_estimation;

int main(int argc, char** argv) {
    const double k = argc > 1 ? std::atof(argv[1]) : 1e-4;
    const unsigned seed = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 1;
    const DifferentialDrive dd{0.1, 0.5, 1024};
    const double dt = 0.1;

    const MatrixXd u = figure_eight_controls(2.0, 250, dt);
    const MatrixXd truth = integrate_controls(Pose2::Zero(), u, dt);
    Rng rng(seed);
    const MatrixXd wheels = simulate_wheel_travel(u, dt, dd, k, rng);
    const DeadReckoningRun run = run_dead_reckoning(Pose2::Zero(), Matrix3d::Zero(), wheels, dd, k);

    std::cout << std::fixed << std::setprecision(3) << "step   error_xy[m]  sigma_x  sigma_y  sigma_theta\n";
    for (int i = 0; i <= u.rows(); i += u.rows() / 4) {
        const double err = (run.poses.row(i).head<2>() - truth.row(i).head<2>()).norm();
        const Matrix3d& P = run.covariances[i];
        std::cout << std::setw(4) << i << "   " << std::setw(10) << err << "  " << std::sqrt(P(0, 0)) << "    "
                  << std::sqrt(P(1, 1)) << "    " << std::sqrt(P(2, 2)) << "\n";
    }
    return 0;
}
