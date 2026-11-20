#include "state_estimation/pose_graph.hpp"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <cmath>
#include <stdexcept>

#include "state_estimation/gaussian.hpp"

namespace state_estimation {

namespace {

using SparseMatrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

/// d R(theta)^T / d theta.
Matrix2d rotation_transpose_derivative(double theta) {
    const double c = std::cos(theta), s = std::sin(theta);
    Matrix2d D;
    D << -s, c, -c, -s;
    return D;
}

/// Gauss-Newton system over the free nodes 1..n-1 (node 0 fixed), eq. (10.14)-(10.15).
void build_system(const std::vector<Pose2>& nodes, const std::vector<PoseGraphEdge>& edges, SparseMatrix& H,
                  VectorXd& b) {
    const int n_free = static_cast<int>(nodes.size()) - 1;
    std::vector<Triplet> triplets;
    triplets.reserve(edges.size() * 36);
    b = VectorXd::Zero(3 * n_free);
    auto add_block = [&](int r, int c, const Matrix3d& M) {
        if (r == 0 || c == 0) return;  // node 0 is the gauge
        for (int a = 0; a < 3; ++a)
            for (int d = 0; d < 3; ++d) triplets.emplace_back(3 * (r - 1) + a, 3 * (c - 1) + d, M(a, d));
    };
    for (const auto& e : edges) {
        Matrix3d A, B;
        pose_graph_jacobians(nodes[e.i], nodes[e.j], e.z, A, B);
        const Vector3d err = pose_graph_error(nodes[e.i], nodes[e.j], e.z);
        const Matrix3d& W = e.information;
        add_block(e.i, e.i, A.transpose() * W * A);
        add_block(e.i, e.j, A.transpose() * W * B);
        add_block(e.j, e.i, B.transpose() * W * A);
        add_block(e.j, e.j, B.transpose() * W * B);
        if (e.i != 0) b.segment<3>(3 * (e.i - 1)) += A.transpose() * W * err;
        if (e.j != 0) b.segment<3>(3 * (e.j - 1)) += B.transpose() * W * err;
    }
    H.resize(3 * n_free, 3 * n_free);
    H.setFromTriplets(triplets.begin(), triplets.end());  // duplicates are summed
}

}  // namespace

Vector3d pose_graph_error(const Pose2& xi, const Pose2& xj, const Pose2& z) {
    // e = (-)z (+) ((-)x_i (+) x_j), written out: eq. (10.11).
    const Matrix2d Ri = rotation(xi.z()), Rz = rotation(z.z());
    Vector3d e;
    e.head<2>() = Rz.transpose() * (Ri.transpose() * (xj.head<2>() - xi.head<2>()) - z.head<2>());
    e.z() = wrap_angle(xj.z() - xi.z() - z.z());
    return e;
}

void pose_graph_jacobians(const Pose2& xi, const Pose2& xj, const Pose2& z, Matrix3d& A, Matrix3d& B) {
    // eq. (10.12)
    const Matrix2d Ri = rotation(xi.z()), Rz = rotation(z.z());
    const Matrix2d RzRi = Rz.transpose() * Ri.transpose();
    A.setZero();
    B.setZero();
    A.topLeftCorner<2, 2>() = -RzRi;
    A.block<2, 1>(0, 2) = Rz.transpose() * rotation_transpose_derivative(xi.z()) * (xj.head<2>() - xi.head<2>());
    A(2, 2) = -1.0;
    B.topLeftCorner<2, 2>() = RzRi;
    B(2, 2) = 1.0;
}

Matrix3d pose_graph_information(const Pose2& z, const Matrix3d& cov_z) {
    Matrix3d Jz = Matrix3d::Zero();
    Jz.topLeftCorner<2, 2>() = -rotation(z.z()).transpose();
    Jz(2, 2) = -1.0;
    return (Jz * cov_z * Jz.transpose()).inverse();
}

int PoseGraph::add_node(const Pose2& x) {
    nodes_.push_back(x);
    return static_cast<int>(nodes_.size()) - 1;
}

void PoseGraph::add_edge(int i, int j, const Pose2& z, const Matrix3d& information) {
    if (i < 0 || j < 0 || i >= n_nodes() || j >= n_nodes() || i == j)
        throw std::invalid_argument("PoseGraph::add_edge: bad node index");
    edges_.push_back({i, j, z, information});
}

void PoseGraph::set_node(int i, const Pose2& x) { nodes_.at(i) = x; }

double PoseGraph::chi2() const {
    double total = 0.0;
    for (const auto& e : edges_) {
        const Vector3d err = pose_graph_error(nodes_[e.i], nodes_[e.j], e.z);
        total += err.dot(e.information * err);
    }
    return total;
}

std::vector<double> PoseGraph::edge_chi2() const {
    std::vector<double> out;
    out.reserve(edges_.size());
    for (const auto& e : edges_) {
        const Vector3d err = pose_graph_error(nodes_[e.i], nodes_[e.j], e.z);
        out.push_back(err.dot(e.information * err));
    }
    return out;
}

void PoseGraph::remove_edge(int k) {
    if (k < 0 || k >= n_edges()) throw std::out_of_range("PoseGraph::remove_edge");
    edges_.erase(edges_.begin() + k);
}

OptimizationSummary PoseGraph::optimize(int max_iterations, double tolerance) {
    OptimizationSummary summary;
    summary.initial_chi2 = chi2();
    summary.final_chi2 = summary.initial_chi2;
    if (n_nodes() < 2) {
        summary.converged = true;
        return summary;
    }
    SparseMatrix H;
    VectorXd b;
    Eigen::SimplicialLDLT<SparseMatrix> solver;
    for (int it = 0; it < max_iterations; ++it) {
        build_system(nodes_, edges_, H, b);
        solver.compute(H);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("PoseGraph::optimize: H is singular (is every node connected to node 0?)");
        const VectorXd dx = solver.solve(-b);  // eq. (10.16)
        for (int k = 1; k < n_nodes(); ++k) {
            nodes_[k] += dx.segment<3>(3 * (k - 1));
            nodes_[k].z() = wrap_angle(nodes_[k].z());
        }
        summary.iterations = it + 1;
        if (dx.lpNorm<Eigen::Infinity>() < tolerance) {
            summary.converged = true;
            break;
        }
    }
    summary.final_chi2 = chi2();
    return summary;
}

Matrix3d PoseGraph::marginal_covariance(int i) const {
    if (i < 0 || i >= n_nodes()) throw std::out_of_range("PoseGraph::marginal_covariance");
    if (i == 0) return Matrix3d::Zero();
    SparseMatrix H;
    VectorXd b;
    build_system(nodes_, edges_, H, b);
    Eigen::SimplicialLDLT<SparseMatrix> solver(H);
    if (solver.info() != Eigen::Success) throw std::runtime_error("PoseGraph::marginal_covariance: H is singular");
    MatrixXd E = MatrixXd::Zero(H.rows(), 3);
    E.block<3, 3>(3 * (i - 1), 0) = Matrix3d::Identity();
    const MatrixXd X = solver.solve(E);
    return X.block<3, 3>(3 * (i - 1), 0);
}

MatrixXd PoseGraph::information_matrix() const {
    MatrixXd H = MatrixXd::Zero(3 * n_nodes(), 3 * n_nodes());
    for (const auto& e : edges_) {
        Matrix3d A, B;
        pose_graph_jacobians(nodes_[e.i], nodes_[e.j], e.z, A, B);
        const Matrix3d& W = e.information;
        H.block<3, 3>(3 * e.i, 3 * e.i) += A.transpose() * W * A;
        H.block<3, 3>(3 * e.i, 3 * e.j) += A.transpose() * W * B;
        H.block<3, 3>(3 * e.j, 3 * e.i) += B.transpose() * W * A;
        H.block<3, 3>(3 * e.j, 3 * e.j) += B.transpose() * W * B;
    }
    return H;
}

MatrixXd PoseGraph::nodes() const {
    MatrixXd out(n_nodes(), 3);
    for (int i = 0; i < n_nodes(); ++i) out.row(i) = nodes_[i].transpose();
    return out;
}

PoseGraphSlam::PoseGraphSlam(const Pose2& x0, const MatrixXd& scan0, const PoseGraphSlamOptions& options)
    : options_(options) {
    graph_.add_node(x0);
    scans_.push_back(scan0);
}

KeyframeResult PoseGraphSlam::add_keyframe(const Pose2& odometry, const Matrix3d& odometry_cov, const MatrixXd& scan) {
    KeyframeResult out;
    const int prev = graph_.n_nodes() - 1;

    // 1. Edge to the previous keyframe: scan matching from the odometry guess, kept only if it is good and agrees
    //    with odometry (a wrong local minimum of ICP is reported with a small covariance and must not get in).
    Pose2 z = odometry;
    Matrix3d Sz = odometry_cov;
    if (scan.rows() >= 3 && scans_.back().rows() >= 3) {
        const IcpResult r = icp_point_to_line(scans_.back(), scan, odometry, options_.icp);
        if (r.converged && r.rmse < options_.sequential_max_rmse) {
            Vector3d d = r.pose - odometry;
            d.z() = wrap_angle(d.z());
            if (d.dot((odometry_cov + r.covariance).ldlt().solve(d)) < chi2_quantile(options_.odometry_gate, 3)) {
                z = r.pose;
                Sz = r.covariance;
                out.scan_matched = true;
                ++n_matched_;
            }
        }
    }
    const int cur = graph_.add_node(compose(graph_.node(prev), z));
    graph_.add_edge(prev, cur, z, pose_graph_information(z, Sz));
    scans_.push_back(scan);

    // 2. Loop closure: the nearest old keyframe within the search radius, widened by 3 sigma of the new position.
    if (cur <= options_.loop_min_separation || scan.rows() < 3) return out;
    const Matrix3d P = graph_.marginal_covariance(cur);
    const double sigma = std::sqrt(Eigen::SelfAdjointEigenSolver<Matrix2d>(P.topLeftCorner<2, 2>()).eigenvalues()(1));
    const Pose2 x_cur = graph_.node(cur);
    int best = -1;
    double best_d = options_.loop_search_radius + 3.0 * sigma;
    for (int b = 0; b < cur - options_.loop_min_separation; ++b) {
        const double d = (graph_.node(b).head<2>() - x_cur.head<2>()).norm();
        if (d < best_d && scans_[b].rows() >= 3) {
            best_d = d;
            best = b;
        }
    }
    if (best < 0) return out;
    const Pose2 pred = between(graph_.node(best), x_cur);
    IcpOptions coarse = options_.icp;  // first with a pairing distance that covers the uncertainty ...
    coarse.max_correspondence_distance = std::max(options_.icp.max_correspondence_distance, 3.0 * sigma);
    IcpResult r = icp_point_to_line(scans_[best], scan, pred, coarse);
    if (!r.converged || r.n_pairs < options_.loop_min_pairs) return out;
    r = icp_point_to_line(scans_[best], scan, r.pose, options_.icp);  // ... then refined with the normal one
    const double jump = (r.pose.head<2>() - pred.head<2>()).norm();
    if (!r.converged || r.rmse > options_.loop_max_rmse || r.n_pairs < options_.loop_min_pairs ||
        jump > options_.loop_search_radius + 3.0 * sigma)
        return out;
    graph_.add_edge(best, cur, r.pose, pose_graph_information(r.pose, r.covariance));
    loops_.emplace_back(best, cur);
    out.loop_closure = best;
    if (options_.optimize_on_closure) graph_.optimize();
    return out;
}

MatrixXd PoseGraphSlam::map_points() const {
    Eigen::Index total = 0;
    for (const auto& s : scans_) total += s.rows();
    MatrixXd out(total, 2);
    Eigen::Index row = 0;
    for (int k = 0; k < graph_.n_nodes(); ++k) {
        out.middleRows(row, scans_[k].rows()) = transform_points(graph_.node(k), scans_[k]);
        row += scans_[k].rows();
    }
    return out;
}

}  // namespace state_estimation
