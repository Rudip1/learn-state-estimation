#include "bindings.hpp"
#include "state_estimation/pose_graph.hpp"
#include "state_estimation/scan_matching.hpp"

using namespace state_estimation;

void bind_pose_graph(py::module_& m) {
    // ---- scan matching, sections 10.1-10.4
    py::class_<Segment>(m, "Segment", "A wall of the simulated world, from a to b")
        .def(py::init([](const Vector2d& a, const Vector2d& b) { return Segment{a, b}; }), py::arg("a"), py::arg("b"))
        .def_readwrite("a", &Segment::a)
        .def_readwrite("b", &Segment::b);
    m.def("simulate_scan", &simulate_scan, py::arg("x"), py::arg("walls"), py::arg("n_beams"), py::arg("fov"),
          py::arg("max_range"), py::arg("sigma_range"), py::arg("rng"),
          "Simulated 2-D range scan from pose x; hit points in the robot frame (m x 2)");
    m.def("transform_points", &transform_points, py::arg("x"), py::arg("points"));
    m.def("align_point_sets", &align_point_sets, py::arg("P"), py::arg("Q"),
          "Closed-form alignment, eq. (10.3)-(10.5): the pose x minimising sum |x [+] p_i - q_i|^2");
    m.def("estimate_normals", &estimate_normals, py::arg("points"), py::arg("k") = 5, py::arg("max_radius") = 0.5);
    m.def("nearest_neighbour", &nearest_neighbour, py::arg("cloud"), py::arg("p"));

    py::class_<IcpOptions>(m, "IcpOptions")
        .def(py::init<>())
        .def_readwrite("max_iterations", &IcpOptions::max_iterations)
        .def_readwrite("max_correspondence_distance", &IcpOptions::max_correspondence_distance)
        .def_readwrite("tolerance", &IcpOptions::tolerance)
        .def_readwrite("normal_neighbours", &IcpOptions::normal_neighbours)
        .def_readwrite("normal_radius", &IcpOptions::normal_radius);
    py::class_<IcpResult>(m, "IcpResult")
        .def_readonly("pose", &IcpResult::pose)
        .def_readonly("covariance", &IcpResult::covariance)
        .def_readonly("hessian", &IcpResult::hessian)
        .def_readonly("iterations", &IcpResult::iterations)
        .def_readonly("converged", &IcpResult::converged)
        .def_readonly("rmse", &IcpResult::rmse)
        .def_readonly("n_pairs", &IcpResult::n_pairs);
    m.def("icp_point_to_point", &icp_point_to_point, py::arg("reference"), py::arg("current"), py::arg("guess"),
          py::arg("options") = IcpOptions{});
    m.def("icp_point_to_line", &icp_point_to_line, py::arg("reference"), py::arg("current"), py::arg("guess"),
          py::arg("options") = IcpOptions{});

    // ---- pose graphs, sections 10.5-10.7
    m.def("pose_graph_error", &pose_graph_error, py::arg("xi"), py::arg("xj"), py::arg("z"));
    m.def(
        "pose_graph_jacobians",
        [](const Pose2& xi, const Pose2& xj, const Pose2& z) {
            Matrix3d A, B;
            pose_graph_jacobians(xi, xj, z, A, B);
            return py::make_tuple(A, B);
        },
        py::arg("xi"), py::arg("xj"), py::arg("z"), "(A, B) = (de/dx_i, de/dx_j), eq. (10.12)");
    m.def("pose_graph_information", &pose_graph_information, py::arg("z"), py::arg("cov_z"));

    py::class_<PoseGraphEdge>(m, "PoseGraphEdge")
        .def_readonly("i", &PoseGraphEdge::i)
        .def_readonly("j", &PoseGraphEdge::j)
        .def_readonly("z", &PoseGraphEdge::z)
        .def_readonly("information", &PoseGraphEdge::information);
    py::class_<OptimizationSummary>(m, "OptimizationSummary")
        .def_readonly("iterations", &OptimizationSummary::iterations)
        .def_readonly("initial_chi2", &OptimizationSummary::initial_chi2)
        .def_readonly("final_chi2", &OptimizationSummary::final_chi2)
        .def_readonly("converged", &OptimizationSummary::converged);
    py::class_<PoseGraph>(m, "PoseGraph", "2-D pose graph optimised by Gauss-Newton, chapter 10")
        .def(py::init<>())
        .def("add_node", &PoseGraph::add_node, py::arg("x"))
        .def("add_edge", &PoseGraph::add_edge, py::arg("i"), py::arg("j"), py::arg("z"), py::arg("information"))
        .def("set_node", &PoseGraph::set_node, py::arg("i"), py::arg("x"))
        .def("chi2", &PoseGraph::chi2)
        .def("edge_chi2", &PoseGraph::edge_chi2)
        .def("remove_edge", &PoseGraph::remove_edge, py::arg("k"))
        .def("optimize", &PoseGraph::optimize, py::arg("max_iterations") = 20, py::arg("tolerance") = 1e-9)
        .def("marginal_covariance", &PoseGraph::marginal_covariance, py::arg("i"))
        .def("information_matrix", &PoseGraph::information_matrix)
        .def("node", &PoseGraph::node, py::arg("i"))
        .def_property_readonly("n_nodes", &PoseGraph::n_nodes)
        .def_property_readonly("n_edges", &PoseGraph::n_edges)
        .def_property_readonly("nodes", &PoseGraph::nodes)
        .def_property_readonly("edges", &PoseGraph::edges);

    // ---- the office simulator and incremental SLAM, section 10.8
    m.def("office_world", &office_world);
    m.def("office_lap_controls", &office_lap_controls, py::arg("laps"), py::arg("dt"));
    py::class_<LidarRun>(m, "LidarRun")
        .def_readonly("truth", &LidarRun::truth)
        .def_readonly("wheel_travel", &LidarRun::wheel_travel)
        .def_readonly("keyframes", &LidarRun::keyframes)
        .def_readonly("scans", &LidarRun::scans)
        .def_readonly("odometry", &LidarRun::odometry)
        .def_readonly("odometry_cov", &LidarRun::odometry_cov);
    m.def("simulate_lidar_run", &simulate_lidar_run, py::arg("x0"), py::arg("controls"), py::arg("dt"), py::arg("dd"),
          py::arg("var_per_meter"), py::arg("walls"), py::arg("every"), py::arg("n_beams"), py::arg("max_range"),
          py::arg("sigma_range"), py::arg("rng"));

    py::class_<PoseGraphSlamOptions>(m, "PoseGraphSlamOptions")
        .def(py::init<>())
        .def_readwrite("icp", &PoseGraphSlamOptions::icp)
        .def_readwrite("sequential_max_rmse", &PoseGraphSlamOptions::sequential_max_rmse)
        .def_readwrite("odometry_gate", &PoseGraphSlamOptions::odometry_gate)
        .def_readwrite("loop_min_separation", &PoseGraphSlamOptions::loop_min_separation)
        .def_readwrite("loop_search_radius", &PoseGraphSlamOptions::loop_search_radius)
        .def_readwrite("loop_max_rmse", &PoseGraphSlamOptions::loop_max_rmse)
        .def_readwrite("loop_min_pairs", &PoseGraphSlamOptions::loop_min_pairs)
        .def_readwrite("optimize_on_closure", &PoseGraphSlamOptions::optimize_on_closure);
    py::class_<KeyframeResult>(m, "KeyframeResult")
        .def_readonly("scan_matched", &KeyframeResult::scan_matched)
        .def_readonly("loop_closure", &KeyframeResult::loop_closure);
    py::class_<PoseGraphSlam>(m, "PoseGraphSlam", "Incremental pose-graph SLAM with a 2-D lidar, section 10.8")
        .def(py::init<const Pose2&, const MatrixXd&, const PoseGraphSlamOptions&>(), py::arg("x0"), py::arg("scan0"),
             py::arg("options") = PoseGraphSlamOptions{})
        .def("add_keyframe", &PoseGraphSlam::add_keyframe, py::arg("odometry"), py::arg("odometry_cov"), py::arg("scan"))
        .def_property_readonly("graph", [](const PoseGraphSlam& s) { return s.graph(); }, "a copy of the pose graph")
        .def_property_readonly("loop_closures", &PoseGraphSlam::loop_closures)
        .def_property_readonly("n_scan_matched", &PoseGraphSlam::n_scan_matched)
        .def("map_points", &PoseGraphSlam::map_points);
}
