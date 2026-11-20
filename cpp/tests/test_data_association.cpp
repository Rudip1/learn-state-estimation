// Chapter 8 tests: the worked example of section 8.3 where individual compatibility fails and joint
// compatibility succeeds, JCBB against exhaustive search, joint NIS against its definition, clutter rejection.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "state_estimation/data_association.hpp"

using namespace state_estimation;
using Catch::Approx;

namespace {
constexpr double kPi = 3.14159265358979323846;

/// Section 8.3: four landmarks 1 m apart on a line; the robot position estimate is uncertain along the line
/// (sigma 1 m) and in fact 0.9 m off. Readings are landmark positions relative to the robot (H = -I).
AssociationProblem corridor_problem() {
    AssociationProblem p;
    p.P = Eigen::Vector2d(1.0, 0.01).asDiagonal();
    p.R = Eigen::Matrix2d::Identity() * 0.01;
    const double offset = 0.9;
    for (int j = 0; j < 4; ++j) {
        const Vector2d m(j, 0.0);
        p.predictions.push_back(m);  // estimated robot at the origin
        p.jacobians.push_back(-MatrixXd::Identity(2, 2));
        p.observations.push_back(m - Vector2d(offset, 0.0) + Vector2d(0.01 * (j % 2 ? 1 : -1), 0.005 * j));
    }
    return p;
}
}  // namespace

TEST_CASE("Individual compatibility is fooled by a common offset; JCBB is not", "[ch08]") {
    const AssociationProblem p = corridor_problem();
    const Association ic = associate_icnn(p);
    // Every reading but the first is individually closest to the landmark 1 m before its own.
    CHECK(ic.pairing[1] == 0);
    CHECK(ic.pairing[2] == 1);
    CHECK(ic.pairing[3] == 2);
    const Association jc = associate_jcbb(p);
    CHECK(jc.pairing == std::vector<int>({0, 1, 2, 3}));
    CHECK(jc.n_paired == 4);
    CHECK(jc.joint_nis < chi2_quantile(0.95, 8));
    // The ICNN hypothesis is not jointly compatible.
    CHECK(joint_nis(p, ic.pairing).first > chi2_quantile(0.95, 2 * ic.n_paired));
}

TEST_CASE("Joint NIS equals its definition and the sum of individual NIS for independent predictions", "[ch08]") {
    AssociationProblem p = corridor_problem();
    // With P = 0 the innovations are independent: the joint NIS is the sum of the individual ones.
    p.P.setZero();
    const std::vector<int> pairing{0, 1, -1, 3};
    double sum = 0.0;
    for (int i : {0, 1, 3}) sum += individual_nis(p, i, pairing[i]);
    const auto [nis, dof] = joint_nis(p, pairing);
    CHECK(nis == Approx(sum));
    CHECK(dof == 6);
}

TEST_CASE("JCBB finds the same hypothesis as exhaustive search", "[ch08]") {
    Rng rng(12);
    for (int trial = 0; trial < 40; ++trial) {
        const Pose2 x(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-kPi, kPi));
        const Matrix3d P = Vector3d(0.3, 0.3, 0.05).asDiagonal();
        MatrixXd landmarks(5, 2);
        for (int j = 0; j < 5; ++j) landmarks.row(j) << rng.uniform(-4, 4), rng.uniform(-4, 4);
        RangeBearingSensor sensor{20.0, 2.0 * kPi, 0.1, 0.03};
        const Pose2 truth = sample_gaussian(x, P, 1, rng).row(0).transpose();
        auto obs = sensor.observe(truth, landmarks, rng);
        obs.resize(3);
        obs = add_clutter(obs, 1, sensor, rng);
        const AssociationProblem p = landmark_association_problem(x, P, obs, landmarks, sensor.R());
        const Association a = associate_jcbb(p), b = associate_brute_force(p);
        CHECK(a.n_paired == b.n_paired);
        CHECK(a.joint_nis == Approx(b.joint_nis).margin(1e-9));
        CHECK(a.pairing == b.pairing);
    }
}

TEST_CASE("Clutter: JCBB leaves spurious readings unpaired, Euclidean NN pairs them", "[ch08]") {
    MatrixXd landmarks(3, 2);
    landmarks << 3.0, 0.0, 0.0, 3.0, -3.0, -1.0;
    const Pose2 x(0.0, 0.0, 0.0);
    const Matrix3d P = Vector3d(0.01, 0.01, 0.001).asDiagonal();
    RangeBearingSensor sensor{6.0, 2.0 * kPi, 0.05, 0.01};
    Rng rng(3);
    std::vector<Observation> obs = sensor.observe(x, landmarks, rng);
    obs.push_back({-1, Vector2d(2.5, 0.3)});  // a spurious reading near landmark 0
    const AssociationProblem p = landmark_association_problem(x, P, obs, landmarks, sensor.R());
    const Association jc = associate_jcbb(p);
    CHECK(jc.pairing == std::vector<int>({0, 1, 2, -1}));
    const Association nn = associate_nearest_neighbor(p, 1.0);
    CHECK(nn.pairing[3] == 0);
}
