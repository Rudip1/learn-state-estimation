#include "bindings.hpp"
#include "state_estimation/motion_models.hpp"
#include "state_estimation/pose2.hpp"
#include "state_estimation/simulation.hpp"

using namespace state_estimation;

void bind_motion(py::module_& m) {
    // ---- pose algebra (section 2.1) ----
    m.def("wrap_angle", &wrap_angle, py::arg("a"), "Wrap an angle to (-pi, pi]");
    m.def("rotation", &rotation, py::arg("theta"));
    m.def("compose", &compose, py::arg("a"), py::arg("b"), "Pose compounding a (+) b, eq. (2.1)");
    m.def("inverse", &inverse, py::arg("a"), "Pose inversion (-)a, eq. (2.4)");
    m.def("between", &between, py::arg("a"), py::arg("b"), "Relative pose (-)a (+) b");
    m.def("compose_jacobian_1", &compose_jacobian_1, py::arg("a"), py::arg("b"));
    m.def("compose_jacobian_2", &compose_jacobian_2, py::arg("a"), py::arg("b"));
    m.def("inverse_jacobian", &inverse_jacobian, py::arg("a"));
    m.def("transform_point", &transform_point, py::arg("x"), py::arg("p"));
    m.def("transform_point_jacobian_pose", &transform_point_jacobian_pose, py::arg("x"), py::arg("p"));
    m.def("transform_point_jacobian_point", &transform_point_jacobian_point, py::arg("x"));
    m.def("inverse_transform_point", &inverse_transform_point, py::arg("x"), py::arg("m"));
    m.def("inverse_transform_point_jacobian_pose", &inverse_transform_point_jacobian_pose, py::arg("x"),
          py::arg("m"));
    m.def("to_matrix", &to_matrix, py::arg("x"));
    m.def("from_matrix", &from_matrix, py::arg("T"));

    // ---- motion models (sections 2.2-2.6) ----
    py::class_<DifferentialDrive>(m, "DifferentialDrive")
        .def(py::init<>())
        .def(py::init([](double r, double b, int ticks) { return DifferentialDrive{r, b, ticks}; }),
             py::arg("wheel_radius"), py::arg("wheel_base"), py::arg("ticks_per_revolution") = 0)
        .def_readwrite("wheel_radius", &DifferentialDrive::wheel_radius)
        .def_readwrite("wheel_base", &DifferentialDrive::wheel_base)
        .def_readwrite("ticks_per_revolution", &DifferentialDrive::ticks_per_revolution)
        .def("wheel_to_body", &DifferentialDrive::wheel_to_body, py::arg("omega_left"), py::arg("omega_right"))
        .def("body_to_wheel", &DifferentialDrive::body_to_wheel, py::arg("v"), py::arg("omega"))
        .def("wheel_travel_to_arc", &DifferentialDrive::wheel_travel_to_arc, py::arg("d_left"),
             py::arg("d_right"))
        .def("ticks_to_distance", &DifferentialDrive::ticks_to_distance, py::arg("ticks"));

    m.def("arc_displacement", &arc_displacement, py::arg("s"), py::arg("phi"));
    m.def("arc_displacement_jacobian", &arc_displacement_jacobian, py::arg("s"), py::arg("phi"));
    m.def("velocity_motion", &velocity_motion, py::arg("x"), py::arg("v"), py::arg("omega"), py::arg("dt"));
    m.def("velocity_motion_jacobian_pose", &velocity_motion_jacobian_pose, py::arg("x"), py::arg("v"),
          py::arg("omega"), py::arg("dt"));
    m.def("velocity_motion_jacobian_control", &velocity_motion_jacobian_control, py::arg("x"), py::arg("v"),
          py::arg("omega"), py::arg("dt"));
    m.def("sample_velocity_motion", &sample_velocity_motion, py::arg("x"), py::arg("v"), py::arg("omega"),
          py::arg("dt"), py::arg("alpha"), py::arg("rng"));
    m.def("odometry_to_rtr", &odometry_to_rtr, py::arg("odom_prev"), py::arg("odom_curr"));
    m.def("apply_rtr", &apply_rtr, py::arg("x"), py::arg("rtr"));
    m.def("sample_odometry_motion", &sample_odometry_motion, py::arg("x"), py::arg("odom_prev"),
          py::arg("odom_curr"), py::arg("alpha"), py::arg("rng"));
    m.def("odometry_motion_probability", &odometry_motion_probability, py::arg("x_curr"), py::arg("x_prev"),
          py::arg("odom_prev"), py::arg("odom_curr"), py::arg("alpha"));
    m.def("wheel_displacement_covariance", &wheel_displacement_covariance, py::arg("dd"), py::arg("d_left"),
          py::arg("d_right"), py::arg("var_left"), py::arg("var_right"));

    py::class_<DeadReckoning>(m, "DeadReckoning")
        .def(py::init<const Pose2&, const Matrix3d&>(), py::arg("x0"), py::arg("P0"))
        .def("predict_displacement", &DeadReckoning::predict_displacement, py::arg("u"), py::arg("Q"))
        .def("predict_wheels", &DeadReckoning::predict_wheels, py::arg("dd"), py::arg("d_left"),
             py::arg("d_right"), py::arg("var_left"), py::arg("var_right"))
        .def_property_readonly("pose", &DeadReckoning::pose)
        .def_property_readonly("covariance", &DeadReckoning::covariance);

    // ---- simulation ----
    m.def("circle_controls", &circle_controls, py::arg("v"), py::arg("radius"), py::arg("steps"));
    m.def("figure_eight_controls", &figure_eight_controls, py::arg("radius"), py::arg("steps_per_loop"),
          py::arg("dt"));
    m.def("integrate_controls", &integrate_controls, py::arg("x0"), py::arg("controls"), py::arg("dt"));
    m.def("simulate_wheel_travel", &simulate_wheel_travel, py::arg("controls"), py::arg("dt"), py::arg("dd"),
          py::arg("var_per_meter"), py::arg("rng"));
    py::class_<DeadReckoningRun>(m, "DeadReckoningRun")
        .def_readonly("poses", &DeadReckoningRun::poses)
        .def_readonly("covariances", &DeadReckoningRun::covariances);
    m.def("run_dead_reckoning", &run_dead_reckoning, py::arg("x0"), py::arg("P0"), py::arg("wheel_travel"),
          py::arg("dd"), py::arg("var_per_meter"));
}
