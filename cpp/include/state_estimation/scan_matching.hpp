#pragma once
/// \file scan_matching.hpp
/// Chapter 10 — scan matching: the closed-form alignment of two point sets, ICP with point-to-point and
/// point-to-line errors, the covariance of a match, and a small 2-D range-scan simulator.
/// Equations: 1_theory/10_pose_graph_slam.md, sections 10.1-10.4.

#include <Eigen/Dense>
#include <vector>

#include "state_estimation/gaussian.hpp"
#include "state_estimation/motion_models.hpp"
#include "state_estimation/pose2.hpp"

namespace state_estimation {

/// A wall of the simulated world: the segment from a to b.
struct Segment {
    Vector2d a;
    Vector2d b;
};

/// Simulated 2-D range scan from pose x: n_beams bearings evenly spread over the field of view fov (centred on
/// the heading), each returning the distance to the nearest wall plus N(0, sigma_range^2). Beams that hit nothing
/// within max_range are dropped. Returns the hit points in the robot frame as the rows of an m x 2 matrix.
MatrixXd simulate_scan(const Pose2& x, const std::vector<Segment>& walls, int n_beams, double fov,
                       double max_range, double sigma_range, Rng& rng);

/// Express points given in the frame of pose x in the world frame: each row p becomes x [+] p, eq. (2.6).
MatrixXd transform_points(const Pose2& x, const MatrixXd& points);

/// Closed-form least-squares alignment of paired points, eq. (10.3)-(10.5): the pose x that minimises
/// sum_i |x [+] p_i - q_i|^2, where p_i (rows of P) and q_i (rows of Q) are corresponding points.
Pose2 align_point_sets(const MatrixXd& P, const MatrixXd& Q);

/// Unit normals of a scan by principal component analysis of the k nearest neighbours of each point that lie within
/// max_radius of it (section 10.3). The sign of a normal is chosen to point towards the sensor origin. A point with
/// fewer than three points (itself included) within the radius has no reliable normal: its row is NaN.
MatrixXd estimate_normals(const MatrixXd& points, int k = 5, double max_radius = 0.5);

/// Index of the nearest row of \p cloud to \p p and the squared distance (brute force).
std::pair<int, double> nearest_neighbour(const MatrixXd& cloud, const Vector2d& p);

struct IcpOptions {
    int max_iterations = 50;
    /// Pairs farther apart than this (after transforming the current scan) are rejected.
    double max_correspondence_distance = 0.5;
    /// Stop when an iteration changes the translation by less than this and the heading by less than it in rad.
    double tolerance = 1e-6;
    /// Neighbours and neighbourhood radius used for the normals of the reference scan (point-to-line only).
    int normal_neighbours = 5;
    double normal_radius = 0.5;
};

struct IcpResult {
    Pose2 pose = Pose2::Zero();          ///< pose of the current scan's frame in the reference scan's frame
    Matrix3d covariance = Matrix3d::Zero();  ///< s^2 (J^T J)^{-1}, eq. (10.10)
    Matrix3d hessian = Matrix3d::Zero();     ///< J^T J at the final pairs; a zero eigenvalue = an unconstrained direction
    int iterations = 0;
    bool converged = false;
    double rmse = 0.0;                   ///< root mean square of the residuals of the final pairs
    int n_pairs = 0;                     ///< number of accepted pairs in the final iteration
};

/// Point-to-point ICP (section 10.2): alternate nearest-neighbour pairing and the closed-form alignment of
/// eq. (10.5), starting from \p guess. Reference points q satisfy q ~ pose [+] p for current points p.
IcpResult icp_point_to_point(const MatrixXd& reference, const MatrixXd& current, const Pose2& guess,
                             const IcpOptions& options = {});

/// Point-to-line ICP (section 10.3): pair each current point with its nearest reference point and minimise the
/// distance to the reference line through it, r_i = n_i^T (x [+] p_i - q_i), by Gauss-Newton, eq. (10.7)-(10.9).
IcpResult icp_point_to_line(const MatrixXd& reference, const MatrixXd& current, const Pose2& guess,
                            const IcpOptions& options = {});

/// The world of the chapter-10 examples: a 16 m x 12 m office split in three by two inner walls, each with two
/// doorways on the path of office_lap_controls, and some clutter; every wall is at least 0.45 m from that path.
std::vector<Segment> office_world();

/// Controls (v, omega) of laps around the office: four straights joined by quarter turns, then 3 m more so that the
/// last lap overlaps the start. Returns steps x 2.
MatrixXd office_lap_controls(int laps, double dt);

/// A simulated lidar run: ground truth, wheel encoders, and one scan every \p every steps (a keyframe).
struct LidarRun {
    MatrixXd truth;                       ///< (N+1) x 3 true poses
    MatrixXd wheel_travel;                ///< N x 2 encoder readings
    std::vector<int> keyframes;           ///< step index of each keyframe (0, every, 2 every, ...)
    std::vector<MatrixXd> scans;          ///< scan of each keyframe, robot frame
    std::vector<Pose2> odometry;          ///< wheel-odometry displacement from the previous keyframe (first: 0)
    std::vector<Matrix3d> odometry_cov;   ///< its covariance (first: 0)
};

LidarRun simulate_lidar_run(const Pose2& x0, const MatrixXd& controls, double dt, const DifferentialDrive& dd,
                            double var_per_meter, const std::vector<Segment>& walls, int every, int n_beams,
                            double max_range, double sigma_range, Rng& rng);

}  // namespace state_estimation
