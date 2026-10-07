#include "state_estimation/histogram_filter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace state_estimation {

namespace {
constexpr double kPi = 3.14159265358979323846;

/// Normalised Gaussian kernel with standard deviation sigma (in cells), truncated at 3 sigma.
std::vector<double> gaussian_kernel(double sigma) {
    const int half = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    std::vector<double> k(2 * half + 1);
    double sum = 0.0;
    for (int i = -half; i <= half; ++i) sum += k[i + half] = std::exp(-0.5 * i * i / (sigma * sigma));
    for (double& v : k) v /= sum;
    return k;
}
}  // namespace

// ---------------------------------------------------------------- discrete Bayes filter

DiscreteBayesFilter::DiscreteBayesFilter(const VectorXd& prior) : belief_(prior / prior.sum()) {
    if ((prior.array() < 0.0).any() || !(prior.sum() > 0.0)) throw std::invalid_argument("invalid prior");
}

void DiscreteBayesFilter::predict(const MatrixXd& transition) {
    // eq. (4.6): bel_bar(i) = sum_j p(i | j) bel(j)
    if (transition.cols() != belief_.size() || transition.rows() != belief_.size()) {
        throw std::invalid_argument("transition matrix has the wrong size");
    }
    belief_ = transition * belief_;
}

void DiscreteBayesFilter::predict_cyclic(const VectorXd& kernel, int offset) {
    const int n = static_cast<int>(belief_.size());
    VectorXd out = VectorXd::Zero(n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < kernel.size(); ++i) {
            const int target = (((j + offset + i) % n) + n) % n;
            out(target) += kernel(i) * belief_(j);
        }
    }
    belief_ = out;
}

double DiscreteBayesFilter::update(const VectorXd& likelihood) {
    // eq. (4.7): bel(i) = eta p(z | i) bel_bar(i)
    const VectorXd unnormalized = belief_.cwiseProduct(likelihood);
    const double evidence = unnormalized.sum();
    if (!(evidence > 0.0)) throw std::domain_error("measurement has zero probability under the belief");
    belief_ = unnormalized / evidence;
    return evidence;
}

// ---------------------------------------------------------------- grid localisation

GridLocalization::GridLocalization(const GridSpec& spec) : spec_(spec) {
    nx_ = static_cast<int>(std::lround((spec.x_max - spec.x_min) / spec.resolution));
    ny_ = static_cast<int>(std::lround((spec.y_max - spec.y_min) / spec.resolution));
    nt_ = spec.n_theta;
    if (nx_ < 1 || ny_ < 1 || nt_ < 1) throw std::invalid_argument("empty grid");
    dtheta_ = 2.0 * kPi / nt_;
    set_uniform();
}

Pose2 GridLocalization::cell_center(int ix, int iy, int it) const {
    return {spec_.x_min + (ix + 0.5) * spec_.resolution, spec_.y_min + (iy + 0.5) * spec_.resolution,
            -kPi + (it + 0.5) * dtheta_};
}

void GridLocalization::set_uniform() { bel_ = VectorXd::Constant(size(), 1.0 / size()); }

void GridLocalization::set_gaussian(const Pose2& mean, const Matrix3d& cov) {
    const Matrix3d info = cov.inverse();
    bel_.resize(size());
    for (int it = 0; it < nt_; ++it)
        for (int iy = 0; iy < ny_; ++iy)
            for (int ix = 0; ix < nx_; ++ix) {
                Vector3d d = cell_center(ix, iy, it) - mean;
                d.z() = wrap_angle(d.z());
                bel_(index(ix, iy, it)) = std::exp(-0.5 * d.dot(info * d));
            }
    normalize();
}

void GridLocalization::set_belief(const VectorXd& b) {
    if (b.size() != size()) throw std::invalid_argument("belief has the wrong size");
    bel_ = b;
    normalize();
}

void GridLocalization::normalize() {
    const double s = bel_.sum();
    if (!(s > 0.0)) {
        set_uniform();  // all mass left the grid or was ruled out: fall back to global localisation
        return;
    }
    bel_ /= s;
}

void GridLocalization::predict_odometry(const Pose2& odom_prev, const Pose2& odom_curr,
                                        const std::array<double, 4>& alpha) {
    // eq. (4.8), with the shift-and-blur approximation of section 4.3.
    const Vector3d u = odometry_to_rtr(odom_prev, odom_curr);
    const double threshold = 1e-12 * bel_.maxCoeff();
    VectorXd out = VectorXd::Zero(size());
    for (int it = 0; it < nt_; ++it)
        for (int iy = 0; iy < ny_; ++iy)
            for (int ix = 0; ix < nx_; ++ix) {
                const double mass = bel_(index(ix, iy, it));
                if (mass <= threshold) continue;
                const Pose2 p = apply_rtr(cell_center(ix, iy, it), u);
                const double fx = (p.x() - spec_.x_min) / spec_.resolution - 0.5;
                const double fy = (p.y() - spec_.y_min) / spec_.resolution - 0.5;
                const double ft = (wrap_angle(p.z()) + kPi) / dtheta_ - 0.5;
                const int x0 = static_cast<int>(std::floor(fx)), y0 = static_cast<int>(std::floor(fy));
                const int t0 = static_cast<int>(std::floor(ft));
                const double wx = fx - x0, wy = fy - y0, wt = ft - t0;
                for (int a = 0; a < 2; ++a)
                    for (int b = 0; b < 2; ++b)
                        for (int c = 0; c < 2; ++c) {
                            const int jx = x0 + a, jy = y0 + b, jt = (((t0 + c) % nt_) + nt_) % nt_;
                            if (jx < 0 || jx >= nx_ || jy < 0 || jy >= ny_) continue;
                            const double w = (a ? wx : 1.0 - wx) * (b ? wy : 1.0 - wy) * (c ? wt : 1.0 - wt);
                            out(index(jx, jy, jt)) += w * mass;
                        }
            }
    bel_ = out;
    // Odometry noise, eq. (2.17): lateral spread from rot1, heading spread from rot1 and rot2.
    const double r1 = u(0), t = u(1), r2 = u(2);
    const double var_trans = alpha[2] * t * t + alpha[3] * (r1 * r1 + r2 * r2);
    const double var_rot1 = alpha[0] * r1 * r1 + alpha[1] * t * t;
    const double var_rot2 = alpha[0] * r2 * r2 + alpha[1] * t * t;
    blur(std::sqrt(var_trans + t * t * var_rot1) / spec_.resolution, std::sqrt(var_rot1 + var_rot2) / dtheta_);
    normalize();
}

void GridLocalization::blur(double sx, double st) {
    // Separable convolution: x and y with zero padding (mass leaves the map), theta cyclic.
    if (sx > 0.2) {
        const auto k = gaussian_kernel(sx);
        const int h = static_cast<int>(k.size() / 2);
        VectorXd tmp = VectorXd::Zero(size());
        for (int it = 0; it < nt_; ++it)
            for (int iy = 0; iy < ny_; ++iy)
                for (int ix = 0; ix < nx_; ++ix) {
                    const double v = bel_(index(ix, iy, it));
                    if (v == 0.0) continue;
                    for (int d = -h; d <= h; ++d) {
                        const int jx = ix + d;
                        if (jx >= 0 && jx < nx_) tmp(index(jx, iy, it)) += k[d + h] * v;
                    }
                }
        bel_.setZero();
        for (int it = 0; it < nt_; ++it)
            for (int iy = 0; iy < ny_; ++iy)
                for (int ix = 0; ix < nx_; ++ix) {
                    const double v = tmp(index(ix, iy, it));
                    if (v == 0.0) continue;
                    for (int d = -h; d <= h; ++d) {
                        const int jy = iy + d;
                        if (jy >= 0 && jy < ny_) bel_(index(ix, jy, it)) += k[d + h] * v;
                    }
                }
    }
    if (st > 0.2) {
        const auto k = gaussian_kernel(st);
        const int h = static_cast<int>(k.size() / 2);
        VectorXd tmp = VectorXd::Zero(size());
        for (int it = 0; it < nt_; ++it)
            for (int iy = 0; iy < ny_; ++iy)
                for (int ix = 0; ix < nx_; ++ix) {
                    const double v = bel_(index(ix, iy, it));
                    if (v == 0.0) continue;
                    for (int d = -h; d <= h; ++d) tmp(index(ix, iy, (((it + d) % nt_) + nt_) % nt_)) += k[d + h] * v;
                }
        bel_ = tmp;
    }
}

double GridLocalization::update_range_bearing(const std::vector<Observation>& observations,
                                              const MatrixXd& landmarks, const Matrix2d& R,
                                              bool account_for_cell_size) {
    return update_impl(observations, landmarks, R, account_for_cell_size, true);
}

double GridLocalization::update_range_bearing_anonymous(const std::vector<Observation>& observations,
                                                        const MatrixXd& landmarks, const Matrix2d& R,
                                                        bool account_for_cell_size) {
    return update_impl(observations, landmarks, R, account_for_cell_size, false);
}

double GridLocalization::update_impl(const std::vector<Observation>& observations, const MatrixXd& landmarks,
                                     const Matrix2d& R, bool account_for_cell_size, bool known_correspondences) {
    // eq. (4.9)-(4.10) and (4.11), evaluated in the log domain.
    if (observations.empty()) return 0.0;
    const double cell_var = spec_.resolution * spec_.resolution / 12.0;
    const double heading_var = dtheta_ * dtheta_ / 12.0;
    const int n_landmarks = static_cast<int>(landmarks.rows());
    // log N(z; h(c, m), R_eff) for one reading and one landmark.
    auto log_density = [&](const Pose2& c, const Vector2d& z, const Vector2d& m) {
        const Vector2d zhat = range_bearing(c, m);
        double vr = R(0, 0), vb = R(1, 1);
        if (account_for_cell_size) {
            vr += cell_var;
            vb += cell_var / std::max(zhat(0) * zhat(0), cell_var) + heading_var;
        }
        const double er = z(0) - zhat(0), eb = wrap_angle(z(1) - zhat(1));
        return -0.5 * (er * er / vr + eb * eb / vb) - 0.5 * std::log(4.0 * kPi * kPi * vr * vb);
    };
    VectorXd loglik(size());
    std::vector<double> terms(n_landmarks);
    for (int it = 0; it < nt_; ++it)
        for (int iy = 0; iy < ny_; ++iy)
            for (int ix = 0; ix < nx_; ++ix) {
                const Pose2 c = cell_center(ix, iy, it);
                double l = 0.0;
                for (const auto& o : observations) {
                    if (known_correspondences) {
                        l += log_density(c, o.z, landmarks.row(o.landmark).transpose());
                        continue;
                    }
                    // log of the mixture (1/n) sum_j exp(t_j), computed stably.
                    double tmax = -std::numeric_limits<double>::infinity();
                    for (int j = 0; j < n_landmarks; ++j) {
                        terms[j] = log_density(c, o.z, landmarks.row(j).transpose());
                        tmax = std::max(tmax, terms[j]);
                    }
                    double sum = 0.0;
                    for (int j = 0; j < n_landmarks; ++j) sum += std::exp(terms[j] - tmax);
                    l += tmax + std::log(sum / n_landmarks);
                }
                loglik(index(ix, iy, it)) = l;
            }
    const double lmax = loglik.maxCoeff();
    const VectorXd w = bel_.cwiseProduct((loglik.array() - lmax).exp().matrix());
    const double s = w.sum();
    if (!(s > 0.0)) {
        // The readings are impossible under the current belief (e.g. after a kidnapping): leave the belief
        // unchanged and report it through the evidence; recovering is the caller's decision (section 4.4).
        return -std::numeric_limits<double>::infinity();
    }
    bel_ = w / s;
    return lmax + std::log(s);
}

void GridLocalization::mix_uniform(double eps) {
    bel_ = (1.0 - eps) * bel_ + VectorXd::Constant(size(), eps / size());  // eq. (4.12)
}

Pose2 GridLocalization::mean() const {
    double sx = 0, sy = 0, sc = 0, ss = 0;
    for (int it = 0; it < nt_; ++it)
        for (int iy = 0; iy < ny_; ++iy)
            for (int ix = 0; ix < nx_; ++ix) {
                const double w = bel_(index(ix, iy, it));
                const Pose2 c = cell_center(ix, iy, it);
                sx += w * c.x();
                sy += w * c.y();
                sc += w * std::cos(c.z());
                ss += w * std::sin(c.z());
            }
    return {sx, sy, std::atan2(ss, sc)};
}

Matrix3d GridLocalization::covariance() const {
    const Pose2 mu = mean();
    Matrix3d P = Matrix3d::Zero();
    for (int it = 0; it < nt_; ++it)
        for (int iy = 0; iy < ny_; ++iy)
            for (int ix = 0; ix < nx_; ++ix) {
                const double w = bel_(index(ix, iy, it));
                if (w == 0.0) continue;
                Vector3d d = cell_center(ix, iy, it) - mu;
                d.z() = wrap_angle(d.z());
                P += w * d * d.transpose();
            }
    return P;
}

Pose2 GridLocalization::map_estimate() const {
    Eigen::Index best;
    bel_.maxCoeff(&best);
    const int i = static_cast<int>(best);
    return cell_center(i % nx_, (i / nx_) % ny_, i / (nx_ * ny_));
}

MatrixXd GridLocalization::marginal_xy() const {
    MatrixXd m = MatrixXd::Zero(ny_, nx_);
    for (int it = 0; it < nt_; ++it)
        for (int iy = 0; iy < ny_; ++iy)
            for (int ix = 0; ix < nx_; ++ix) m(iy, ix) += bel_(index(ix, iy, it));
    return m;
}

}  // namespace state_estimation
