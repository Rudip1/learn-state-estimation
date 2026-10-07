#include "bindings.hpp"
#include "state_estimation/measurement_models.hpp"

using namespace state_estimation;

void bind_measurement(py::module_& m) {
    m.def("range_bearing", &range_bearing, py::arg("x"), py::arg("m"), "Expected (range, bearing), eq. (3.4)");
    m.def("range_bearing_jacobian_pose", &range_bearing_jacobian_pose, py::arg("x"), py::arg("m"));
    m.def("range_bearing_jacobian_landmark", &range_bearing_jacobian_landmark, py::arg("x"), py::arg("m"));
    m.def("range_bearing_inverse", &range_bearing_inverse, py::arg("x"), py::arg("z"),
          "Landmark from pose and (range, bearing), eq. (3.6)");
    m.def("range_bearing_inverse_jacobian_pose", &range_bearing_inverse_jacobian_pose, py::arg("x"), py::arg("z"));
    m.def("range_bearing_inverse_jacobian_measurement", &range_bearing_inverse_jacobian_measurement, py::arg("x"),
          py::arg("z"));
    m.def("cartesian_feature", &cartesian_feature, py::arg("x"), py::arg("m"),
          "Landmark in the robot frame, eq. (3.9)");
    m.def("cartesian_feature_jacobian_pose", &cartesian_feature_jacobian_pose, py::arg("x"), py::arg("m"));
    m.def("cartesian_feature_jacobian_landmark", &cartesian_feature_jacobian_landmark, py::arg("x"));
    m.def("polar_to_cartesian_covariance", &polar_to_cartesian_covariance, py::arg("z"), py::arg("R_polar"));
    m.def("position_fix", &position_fix, py::arg("x"), py::arg("lever_arm"));
    m.def("position_fix_jacobian", &position_fix_jacobian, py::arg("x"), py::arg("lever_arm"));
    m.def("compass", &compass, py::arg("x"));
    m.def("measurement_residual", &measurement_residual, py::arg("z"), py::arg("z_hat"),
          py::arg("angle_indices") = std::vector<int>{});
    m.def("measurement_log_likelihood", &measurement_log_likelihood, py::arg("z"), py::arg("z_hat"), py::arg("R"),
          py::arg("angle_indices") = std::vector<int>{});
    m.def("fisher_information", &fisher_information, py::arg("H"), py::arg("R"));

    py::class_<Observation>(m, "Observation")
        .def(py::init<>())
        .def(py::init([](int j, const Vector2d& z) { return Observation{j, z}; }), py::arg("landmark"), py::arg("z"))
        .def_readwrite("landmark", &Observation::landmark)
        .def_readwrite("z", &Observation::z)
        .def("__repr__", [](const Observation& o) {
            return "Observation(landmark=" + std::to_string(o.landmark) + ", z=[" + std::to_string(o.z(0)) + ", " +
                   std::to_string(o.z(1)) + "])";
        });

    py::class_<RangeBearingSensor>(m, "RangeBearingSensor")
        .def(py::init<>())
        .def(py::init([](double max_range, double fov, double sr, double sb) {
                 return RangeBearingSensor{max_range, fov, sr, sb};
             }),
             py::arg("max_range") = 10.0, py::arg("fov") = 2.0 * 3.14159265358979323846, py::arg("sigma_range") = 0.1,
             py::arg("sigma_bearing") = 0.02)
        .def_readwrite("max_range", &RangeBearingSensor::max_range)
        .def_readwrite("fov", &RangeBearingSensor::fov)
        .def_readwrite("sigma_range", &RangeBearingSensor::sigma_range)
        .def_readwrite("sigma_bearing", &RangeBearingSensor::sigma_bearing)
        .def("R", &RangeBearingSensor::R)
        .def("visible", &RangeBearingSensor::visible, py::arg("x"), py::arg("m"))
        .def("observe", &RangeBearingSensor::observe, py::arg("x"), py::arg("landmarks"), py::arg("rng"));
    m.def("to_cartesian", &to_cartesian, py::arg("observations"));
}
