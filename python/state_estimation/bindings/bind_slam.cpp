#include "bindings.hpp"
#include "state_estimation/ekf_slam.hpp"

using namespace state_estimation;

void bind_slam(py::module_& m) {
    py::class_<ScanResult>(m, "ScanResult")
        .def_readonly("landmark", &ScanResult::landmark)
        .def_readonly("is_new", &ScanResult::is_new)
        .def_readonly("joint_nis", &ScanResult::joint_nis);

    py::class_<EkfSlam>(m, "EkfSlam", "Feature-based EKF SLAM, chapter 9")
        .def(py::init<const Pose2&, const Matrix3d&>(), py::arg("x0"), py::arg("P0"))
        .def("predict_displacement", &EkfSlam::predict_displacement, py::arg("u"), py::arg("Q_u"))
        .def("predict_wheels", &EkfSlam::predict_wheels, py::arg("dd"), py::arg("d_left"), py::arg("d_right"),
             py::arg("var_left"), py::arg("var_right"))
        .def("add_landmark", &EkfSlam::add_landmark, py::arg("z"), py::arg("R"))
        .def("add_known_landmark", &EkfSlam::add_known_landmark, py::arg("m"), py::arg("P_m"))
        .def("update", &EkfSlam::update, py::arg("observations"), py::arg("R"))
        .def("process_known", &EkfSlam::process_known, py::arg("observations"), py::arg("R"))
        .def("process_jcbb", &EkfSlam::process_jcbb, py::arg("observations"), py::arg("R"), py::arg("alpha") = 0.05,
             py::arg("new_alpha") = 1e-4)
        .def("association_problem", &EkfSlam::association_problem, py::arg("observations"), py::arg("R"))
        .def_property_readonly("n_landmarks", &EkfSlam::n_landmarks)
        .def_property_readonly("pose", &EkfSlam::pose)
        .def_property_readonly("pose_covariance", &EkfSlam::pose_covariance)
        .def("landmark", &EkfSlam::landmark, py::arg("i"))
        .def("landmark_covariance", &EkfSlam::landmark_covariance, py::arg("i"))
        .def_property_readonly("landmarks", &EkfSlam::landmarks)
        .def_property_readonly("state", [](const EkfSlam& s) -> VectorXd { return s.state(); })
        .def_property_readonly("covariance", [](const EkfSlam& s) -> MatrixXd { return s.covariance(); })
        .def_property_readonly("id_map", &EkfSlam::id_map);
}
