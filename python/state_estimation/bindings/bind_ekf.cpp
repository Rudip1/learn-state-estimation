#include "bindings.hpp"
#include "state_estimation/ekf.hpp"

using namespace state_estimation;

void bind_ekf(py::module_& m) {
    py::class_<MeasurementFunction>(m, "MeasurementFunction", "h(x), its Jacobian H(x) and the angular components")
        .def(py::init([](std::function<VectorXd(const VectorXd&)> h, std::function<MatrixXd(const VectorXd&)> H,
                         std::vector<int> angles) { return MeasurementFunction{h, H, angles}; }),
             py::arg("h"), py::arg("H"), py::arg("angle_indices") = std::vector<int>{});

    py::class_<ExtendedKalmanFilter>(m, "ExtendedKalmanFilter", "Generic EKF, eq. (6.3)-(6.9)")
        .def(py::init<const VectorXd&, const MatrixXd&>(), py::arg("x0"), py::arg("P0"))
        .def("predict", &ExtendedKalmanFilter::predict, py::arg("f"), py::arg("F"), py::arg("W"), py::arg("Q"))
        .def("update", &ExtendedKalmanFilter::update, py::arg("z"), py::arg("model"), py::arg("R"),
             py::arg("iterations") = 1)
        .def("set_state", &ExtendedKalmanFilter::set_state, py::arg("x"), py::arg("P"))
        .def("set_angle_states", &ExtendedKalmanFilter::set_angle_states, py::arg("indices"))
        .def_property_readonly("x", [](const ExtendedKalmanFilter& f) -> VectorXd { return f.x(); })
        .def_property_readonly("P", [](const ExtendedKalmanFilter& f) -> MatrixXd { return f.P(); });

    py::class_<EkfLocalization>(m, "EkfLocalization", "EKF localisation with known landmarks, section 6.3")
        .def(py::init<const Pose2&, const Matrix3d&>(), py::arg("x0"), py::arg("P0"))
        .def("predict_displacement", &EkfLocalization::predict_displacement, py::arg("u"), py::arg("Q_u"))
        .def("predict_wheels", &EkfLocalization::predict_wheels, py::arg("dd"), py::arg("d_left"), py::arg("d_right"),
             py::arg("var_left"), py::arg("var_right"))
        .def("update_landmarks", &EkfLocalization::update_landmarks, py::arg("observations"), py::arg("landmarks"),
             py::arg("R"), py::arg("iterations") = 1)
        .def("update_cartesian", &EkfLocalization::update_cartesian, py::arg("observations"), py::arg("landmarks"),
             py::arg("R"))
        .def("update_compass", &EkfLocalization::update_compass, py::arg("theta"), py::arg("variance"))
        .def("update_position_fix", &EkfLocalization::update_position_fix, py::arg("z"), py::arg("lever_arm"),
             py::arg("R"))
        .def("set_state", &EkfLocalization::set_state, py::arg("x"), py::arg("P"))
        .def_property_readonly("pose", &EkfLocalization::pose)
        .def_property_readonly("covariance", &EkfLocalization::covariance);

    py::class_<LandmarkRun>(m, "LandmarkRun")
        .def_readonly("truth", &LandmarkRun::truth)
        .def_readonly("wheel_travel", &LandmarkRun::wheel_travel)
        .def_readonly("scans", &LandmarkRun::scans);
    m.def("simulate_landmark_run", &simulate_landmark_run, py::arg("x0"), py::arg("controls"), py::arg("dt"),
          py::arg("dd"), py::arg("var_per_meter"), py::arg("sensor"), py::arg("landmarks"), py::arg("scan_every"),
          py::arg("rng"));
}
