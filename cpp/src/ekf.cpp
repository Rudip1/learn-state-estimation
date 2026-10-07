#include "state_estimation/ekf.hpp"

#include <cmath>

namespace state_estimation {

namespace {
MatrixXd symmetrize(const MatrixXd& P) { return 0.5 * (P + P.transpose()); }
}  // namespace

void ExtendedKalmanFilter::wrap_state() {
    for (int i : angle_states_) x_(i) = wrap_angle(x_(i));
}

void ExtendedKalmanFilter::predict(const std::function<VectorXd(const VectorXd&)>& f, const MatrixXd& F,
                                   const MatrixXd& W, const MatrixXd& Q) {
    // eq. (6.3)-(6.4); the Jacobians were evaluated by the caller at the old mean.
    x_ = f(x_);
    P_ = symmetrize(F * P_ * F.transpose() + W * Q * W.transpose());
    wrap_state();
}

Innovation ExtendedKalmanFilter::update(const VectorXd& z, const MeasurementFunction& model, const MatrixXd& R,
                                        int iterations) {
    const VectorXd x_bar = x_;
    const MatrixXd P_bar = P_;
    const int n = static_cast<int>(x_.size());
    Innovation inn;
    VectorXd xi = x_bar;
    MatrixXd H, K;
    for (int it = 0; it < std::max(1, iterations); ++it) {
        H = model.H(xi);
        const MatrixXd S = symmetrize(H * P_bar * H.transpose() + R);  // eq. (6.6)
        K = S.ldlt().solve(H * P_bar).transpose();                      // eq. (6.7)
        // eq. (6.14): residual linearised about the iterate xi; for it = 0 this is the EKF innovation (6.5).
        const VectorXd r = measurement_residual(z, model.h(xi), model.angle_indices) - H * (x_bar - xi);
        if (it == 0) {
            inn.nu = r;
            inn.S = S;
            inn.nis = r.dot(S.ldlt().solve(r));
        }
        xi = x_bar + K * r;
        for (int i : angle_states_) xi(i) = wrap_angle(xi(i));
    }
    inn.K = K;
    x_ = xi;
    const MatrixXd IKH = MatrixXd::Identity(n, n) - K * H;
    P_ = symmetrize(IKH * P_bar * IKH.transpose() + K * R * K.transpose());  // eq. (6.9), Joseph form
    return inn;
}

// ---------------------------------------------------------------- map-based localisation

EkfLocalization::EkfLocalization(const Pose2& x0, const Matrix3d& P0) : ekf_(x0, P0) {
    ekf_.set_angle_states({2});
}

void EkfLocalization::predict_displacement(const Pose2& u, const Matrix3d& Q_u) {
    // eq. (6.10): f(x) = x (+) u, F = J1(x, u), W = J2(x, u)
    const Pose2 x = ekf_.x();
    ekf_.predict([&](const VectorXd& s) -> VectorXd { return compose(Pose2(s), u); }, compose_jacobian_1(x, u),
                 compose_jacobian_2(x, u), Q_u);
}

void EkfLocalization::predict_wheels(const DifferentialDrive& dd, double d_left, double d_right, double var_left,
                                     double var_right) {
    const Vector2d arc = dd.wheel_travel_to_arc(d_left, d_right);
    predict_displacement(arc_displacement(arc(0), arc(1)),
                         wheel_displacement_covariance(dd, d_left, d_right, var_left, var_right));
}

Innovation EkfLocalization::update_landmarks(const std::vector<Observation>& obs, const MatrixXd& landmarks,
                                             const Matrix2d& R, int iterations) {
    if (obs.empty()) return {};
    const int m = static_cast<int>(obs.size());
    // eq. (6.11)-(6.12): stack all readings; the noise of different readings is independent.
    VectorXd z(2 * m);
    MatrixXd Rs = MatrixXd::Zero(2 * m, 2 * m);
    MeasurementFunction model;
    for (int i = 0; i < m; ++i) {
        z.segment<2>(2 * i) = obs[i].z;
        Rs.block<2, 2>(2 * i, 2 * i) = R;
        model.angle_indices.push_back(2 * i + 1);
    }
    model.h = [&](const VectorXd& s) -> VectorXd {
        VectorXd out(2 * m);
        for (int i = 0; i < m; ++i)
            out.segment<2>(2 * i) = range_bearing(Pose2(s), landmarks.row(obs[i].landmark).transpose());
        return out;
    };
    model.H = [&](const VectorXd& s) -> MatrixXd {
        MatrixXd H(2 * m, 3);
        for (int i = 0; i < m; ++i)
            H.block<2, 3>(2 * i, 0) =
                range_bearing_jacobian_pose(Pose2(s), landmarks.row(obs[i].landmark).transpose());
        return H;
    };
    return ekf_.update(z, model, Rs, iterations);
}

Innovation EkfLocalization::update_cartesian(const std::vector<Observation>& obs, const MatrixXd& landmarks,
                                             const Matrix2d& R) {
    if (obs.empty()) return {};
    const int m = static_cast<int>(obs.size());
    // eq. (6.13)
    VectorXd z(2 * m);
    MatrixXd Rs = MatrixXd::Zero(2 * m, 2 * m);
    for (int i = 0; i < m; ++i) {
        z.segment<2>(2 * i) = obs[i].z;
        Rs.block<2, 2>(2 * i, 2 * i) = R;
    }
    MeasurementFunction model;
    model.h = [&](const VectorXd& s) -> VectorXd {
        VectorXd out(2 * m);
        for (int i = 0; i < m; ++i)
            out.segment<2>(2 * i) = cartesian_feature(Pose2(s), landmarks.row(obs[i].landmark).transpose());
        return out;
    };
    model.H = [&](const VectorXd& s) -> MatrixXd {
        MatrixXd H(2 * m, 3);
        for (int i = 0; i < m; ++i)
            H.block<2, 3>(2 * i, 0) =
                cartesian_feature_jacobian_pose(Pose2(s), landmarks.row(obs[i].landmark).transpose());
        return H;
    };
    return ekf_.update(z, model, Rs);
}

Innovation EkfLocalization::update_compass(double theta, double variance) {
    MeasurementFunction model;
    model.h = [](const VectorXd& s) -> VectorXd { return VectorXd::Constant(1, s(2)); };
    model.H = [](const VectorXd&) -> MatrixXd {
        MatrixXd H = MatrixXd::Zero(1, 3);
        H(0, 2) = 1.0;
        return H;
    };
    model.angle_indices = {0};
    return ekf_.update(VectorXd::Constant(1, theta), model, MatrixXd::Constant(1, 1, variance));
}

Innovation EkfLocalization::update_position_fix(const Vector2d& z, const Vector2d& lever_arm, const Matrix2d& R) {
    MeasurementFunction model;
    model.h = [&](const VectorXd& s) -> VectorXd { return position_fix(Pose2(s), lever_arm); };
    model.H = [&](const VectorXd& s) -> MatrixXd { return position_fix_jacobian(Pose2(s), lever_arm); };
    return ekf_.update(z, model, R);
}

LandmarkRun simulate_landmark_run(const Pose2& x0, const MatrixXd& controls, double dt, const DifferentialDrive& dd,
                                  double var_per_meter, const RangeBearingSensor& sensor, const MatrixXd& landmarks,
                                  int scan_every, Rng& rng) {
    LandmarkRun run;
    run.truth = integrate_controls(x0, controls, dt);
    run.wheel_travel = simulate_wheel_travel(controls, dt, dd, var_per_meter, rng);
    run.scans.resize(controls.rows());
    for (int k = 0; k < controls.rows(); ++k) {
        if ((k + 1) % scan_every == 0) run.scans[k] = sensor.observe(run.truth.row(k + 1).transpose(), landmarks, rng);
    }
    return run;
}

}  // namespace state_estimation
