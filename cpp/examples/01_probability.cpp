// Chapter 1 example: linearised vs. Monte Carlo propagation of a range-bearing measurement through the
// polar-to-Cartesian conversion. Usage: example_01_probability [bearing_sigma_rad] [samples]
#include <cstdlib>
#include <iostream>

#include "state_estimation/gaussian.hpp"

using namespace state_estimation;

int main(int argc, char** argv) {
    const double sigma_phi = argc > 1 ? std::atof(argv[1]) : 0.5;
    const int n = argc > 2 ? std::atoi(argv[2]) : 100000;

    Gaussian polar{Eigen::Vector2d(1.0, 0.0), Eigen::Matrix2d::Zero()};
    polar.cov(0, 0) = 0.01 * 0.01;
    polar.cov(1, 1) = sigma_phi * sigma_phi;

    const Gaussian lin = propagate_linearized(
        [](const VectorXd& p) -> VectorXd { return polar_to_cartesian(Eigen::Vector2d(p)); }, polar,
        [](const VectorXd& p) -> MatrixXd { return polar_to_cartesian_jacobian(Eigen::Vector2d(p)); });

    Rng rng(1);
    const MatrixXd s = sample_gaussian(polar.mean, polar.cov, n, rng);
    MatrixXd xy(n, 2);
    for (int i = 0; i < n; ++i) xy.row(i) = polar_to_cartesian(Eigen::Vector2d(s.row(i))).transpose();
    const Gaussian mc = sample_mean_cov(xy);

    std::cout << "bearing sigma = " << sigma_phi << " rad, " << n << " samples\n"
              << "linearised mean : " << lin.mean.transpose() << "\n"
              << "Monte Carlo mean: " << mc.mean.transpose() << "\n"
              << "linearised cov  :\n"
              << lin.cov << "\nMonte Carlo cov :\n"
              << mc.cov << "\n";
    return 0;
}
