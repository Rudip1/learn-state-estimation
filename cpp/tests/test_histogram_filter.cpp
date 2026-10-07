// Chapter 4 tests: the discrete Bayes filter against the hand-worked corridor example and against brute-force
// enumeration of all state sequences; grid localisation against exact shifts, a known pose and a symmetric
// map that must stay bimodal.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <algorithm>
#include <functional>

#include "state_estimation/histogram_filter.hpp"

using namespace state_estimation;
using Catch::Approx;

namespace {
constexpr double kPi = 3.14159265358979323846;

/// The corridor of section 4.2: 10 cyclic cells, doors at cells 1, 4 and 8.
VectorXd door_likelihood(bool saw_door) {
    VectorXd l(10);
    for (int i = 0; i < 10; ++i) {
        const bool door = (i == 1 || i == 4 || i == 8);
        l(i) = saw_door ? (door ? 0.8 : 0.1) : (door ? 0.2 : 0.9);
    }
    return l;
}
}  // namespace

TEST_CASE("Corridor: the worked example of section 4.2", "[ch04]") {
    DiscreteBayesFilter f(VectorXd::Ones(10));
    // Reading 1: "door".
    const double evidence = f.update(door_likelihood(true));
    CHECK(evidence == Approx(0.31));
    CHECK(f.belief()(1) == Approx(0.8 / 3.1));
    CHECK(f.belief()(0) == Approx(0.1 / 3.1));
    CHECK(f.belief()(1) == Approx(0.2581).margin(5e-5));
    CHECK(f.belief()(0) == Approx(0.0323).margin(5e-5));
    // Move 3 cells (2, 3 or 4 with probabilities 0.1, 0.8, 0.1); reading 2: "door". Only the doors at 1 and 8
    // have a door 3 cells further on (4 and 1), so the belief is bimodal at cells 4 and 1.
    VectorXd kernel(3);
    kernel << 0.1, 0.8, 0.1;
    f.predict_cyclic(kernel, 2);
    CHECK(f.belief().sum() == Approx(1.0));
    f.update(door_likelihood(true));
    std::vector<int> order(10);
    for (int i = 0; i < 10; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return f.belief()(a) > f.belief()(b); });
    CHECK(((order[0] == 4 && order[1] == 1) || (order[0] == 1 && order[1] == 4)));
    CHECK(f.belief()(4) == Approx(f.belief()(1)).epsilon(1e-12));
    CHECK(f.belief()(4) == Approx(0.3902).margin(5e-5));
    // Move 3 again; reading 3: "wall". From 4 the robot reaches the wall at 7, from 1 the door at 4: cell 7 wins.
    f.predict_cyclic(kernel, 2);
    f.update(door_likelihood(false));
    Eigen::Index best;
    f.belief().maxCoeff(&best);
    CHECK(best == 7);
    CHECK(f.belief()(7) == Approx(0.4848).margin(5e-5));
}

TEST_CASE("Cyclic prediction equals multiplication by the circulant transition matrix", "[ch04]") {
    VectorXd prior(6);
    prior << 0.1, 0.3, 0.05, 0.25, 0.2, 0.1;
    VectorXd kernel(3);
    kernel << 0.2, 0.7, 0.1;
    DiscreteBayesFilter a(prior), b(prior);
    a.predict_cyclic(kernel, 1);
    MatrixXd T = MatrixXd::Zero(6, 6);
    for (int j = 0; j < 6; ++j)
        for (int i = 0; i < 3; ++i) T((j + 1 + i) % 6, j) += kernel(i);
    b.predict(T);
    CHECK((a.belief() - b.belief()).norm() < 1e-15);
}

TEST_CASE("Discrete Bayes filter equals brute-force enumeration of state sequences", "[ch04]") {
    // A 3-state hidden Markov model observed three times. The posterior over the last state is computed by
    // summing p(x0) prod p(x_k | x_{k-1}) p(z_k | x_k) over all 3^4 sequences.
    VectorXd prior(3);
    prior << 0.5, 0.3, 0.2;
    MatrixXd T(3, 3);  // column-stochastic
    T << 0.7, 0.2, 0.1, 0.2, 0.6, 0.3, 0.1, 0.2, 0.6;
    std::vector<VectorXd> L(3, VectorXd(3));
    L[0] << 0.9, 0.2, 0.1;
    L[1] << 0.3, 0.5, 0.4;
    L[2] << 0.1, 0.3, 0.8;

    DiscreteBayesFilter f(prior);
    for (int k = 0; k < 3; ++k) {
        f.predict(T);
        f.update(L[k]);
    }

    VectorXd brute = VectorXd::Zero(3);
    std::function<void(int, int, double)> rec = [&](int k, int prev, double p) {
        if (k == 3) {
            brute(prev) += p;
            return;
        }
        for (int s = 0; s < 3; ++s) rec(k + 1, s, p * T(s, prev) * L[k](s));
    };
    for (int s0 = 0; s0 < 3; ++s0) rec(0, s0, prior(s0));
    brute /= brute.sum();
    CHECK((f.belief() - brute).norm() < 1e-14);
}

TEST_CASE("Grid prediction: exact rotations and the identity motion", "[ch04]") {
    GridSpec spec{0.0, 4.0, 0.0, 4.0, 0.5, 8};
    GridLocalization g(spec);
    VectorXd b = VectorXd::Zero(g.size());
    b((4 * g.ny() + 3) * g.nx() + 2) = 1.0;  // all mass in cell (ix, iy, it) = (2, 3, 4)
    g.set_belief(b);
    const Pose2 c = g.cell_center(2, 3, 4);
    // Turning on the spot by exactly one heading cell (2 pi / 8) moves the mass to the next heading cell.
    g.predict_odometry(Pose2(0, 0, 0), Pose2(0, 0, kPi / 4.0), {0.0, 0.0, 0.0, 0.0});
    const Pose2 m = g.map_estimate();
    CHECK(m.x() == Approx(c.x()));
    CHECK(m.y() == Approx(c.y()));
    CHECK(wrap_angle(m.z() - c.z() - kPi / 4.0) == Approx(0.0).margin(1e-12));
    CHECK(g.belief().maxCoeff() == Approx(1.0));
    // Zero motion without noise leaves any belief unchanged.
    g.set_gaussian(Pose2(2.0, 2.0, 0.3), Vector3d(0.3, 0.2, 0.5).asDiagonal());
    const VectorXd before = g.belief();
    g.predict_odometry(Pose2(1, 1, 1), Pose2(1, 1, 1), {0.0, 0.0, 0.0, 0.0});
    CHECK((g.belief() - before).norm() < 1e-9);
}

TEST_CASE("Grid localisation converges on a known map", "[ch04]") {
    GridSpec spec{0.0, 10.0, 0.0, 10.0, 0.25, 36};
    GridLocalization g(spec);
    MatrixXd landmarks(4, 2);
    landmarks << 2.0, 2.0, 8.0, 3.0, 7.0, 8.0, 1.0, 7.0;
    RangeBearingSensor sensor{20.0, 2.0 * kPi, 0.1, 0.03};
    const Pose2 truth(4.1, 5.3, 0.7);
    Rng rng(4);
    g.update_range_bearing(sensor.observe(truth, landmarks, rng), landmarks, sensor.R());
    const Pose2 est = g.mean();
    CHECK((est.head<2>() - truth.head<2>()).norm() < 0.25);
    CHECK(std::abs(wrap_angle(est.z() - truth.z())) < 0.1);
    // The posterior covariance is small compared with the map.
    CHECK(g.covariance().topLeftCorner<2, 2>().trace() < 0.05);
}

TEST_CASE("Global localisation in a symmetric map stays bimodal", "[ch04]") {
    // Two indistinguishable landmarks symmetric about the map centre: a range-bearing scan cannot distinguish a
    // pose from its 180-degree rotation about the centre. (With known correspondences it could.)
    GridSpec spec{0.0, 10.0, 0.0, 10.0, 0.25, 36};
    GridLocalization g(spec);
    MatrixXd landmarks(2, 2);
    landmarks << 3.0, 5.0, 7.0, 5.0;
    RangeBearingSensor sensor{20.0, 2.0 * kPi, 0.1, 0.03};
    const Pose2 truth(5.0, 2.0, 0.0);
    const Pose2 mirror(5.0, 8.0, kPi);
    Rng rng(1);
    const auto scan = sensor.observe(truth, landmarks, rng);
    GridLocalization known(spec);
    known.update_range_bearing(scan, landmarks, sensor.R());
    CHECK((known.mean().head<2>() - truth.head<2>()).norm() < 0.3);
    g.update_range_bearing_anonymous(scan, landmarks, sensor.R());
    const MatrixXd m = g.marginal_xy();
    auto mass_near = [&](const Pose2& p) {
        double s = 0.0;
        for (int iy = 0; iy < g.ny(); ++iy)
            for (int ix = 0; ix < g.nx(); ++ix) {
                const Pose2 c = g.cell_center(ix, iy, 0);
                if ((c.head<2>() - p.head<2>()).norm() < 1.0) s += m(iy, ix);
            }
        return s;
    };
    CHECK(mass_near(truth) > 0.3);
    CHECK(mass_near(mirror) > 0.3);
    CHECK(mass_near(truth) + mass_near(mirror) > 0.95);
}

TEST_CASE("Mixing with a uniform distribution", "[ch04]") {
    GridSpec spec{0.0, 2.0, 0.0, 2.0, 0.5, 4};
    GridLocalization g(spec);
    g.set_gaussian(Pose2(1.0, 1.0, 0.0), Vector3d(0.01, 0.01, 0.01).asDiagonal());
    g.mix_uniform(0.1);
    CHECK(g.belief().sum() == Approx(1.0));
    CHECK(g.belief().minCoeff() >= 0.1 / g.size() - 1e-15);
}
