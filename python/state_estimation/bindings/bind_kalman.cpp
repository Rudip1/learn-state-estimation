#include "bindings.hpp"
#include "state_estimation/kalman_filter.hpp"

using namespace state_estimation;

void bind_kalman(py::module_& m) {
    py::class_<Innovation>(m, "Innovation")
        .def_readonly("nu", &Innovation::nu)
        .def_readonly("S", &Innovation::S)
        .def_readonly("K", &Innovation::K)
        .def_readonly("nis", &Innovation::nis);

    py::class_<KalmanFilter>(m, "KalmanFilter", "Linear Kalman filter, eq. (5.4)-(5.12)")
        .def(py::init<const VectorXd&, const MatrixXd&>(), py::arg("x0"), py::arg("P0"))
        .def("predict", py::overload_cast<const MatrixXd&, const MatrixXd&>(&KalmanFilter::predict), py::arg("F"),
             py::arg("Q"))
        .def("predict",
             py::overload_cast<const MatrixXd&, const MatrixXd&, const MatrixXd&, const VectorXd&>(
                 &KalmanFilter::predict),
             py::arg("F"), py::arg("Q"), py::arg("B"), py::arg("u"))
        .def("update", &KalmanFilter::update, py::arg("z"), py::arg("H"), py::arg("R"))
        .def("update_with_gain", &KalmanFilter::update_with_gain, py::arg("z"), py::arg("H"), py::arg("R"),
             py::arg("K"), py::arg("joseph"))
        .def("set_state", &KalmanFilter::set_state, py::arg("x"), py::arg("P"))
        .def_property_readonly("x", [](const KalmanFilter& f) -> VectorXd { return f.x(); })
        .def_property_readonly("P", [](const KalmanFilter& f) -> MatrixXd { return f.P(); });

    m.def("nees", &nees, py::arg("x_true"), py::arg("x_hat"), py::arg("P"), "eq. (5.14)");
    m.def("average_chi2_bounds", &average_chi2_bounds, py::arg("dof"), py::arg("runs"), py::arg("alpha") = 0.05,
          "Acceptance interval of an average of chi-square statistics, eq. (5.15)");
    m.def("steady_state_covariance", &steady_state_covariance, py::arg("F"), py::arg("H"), py::arg("Q"), py::arg("R"),
          py::arg("tol") = 1e-12, py::arg("max_iter") = 100000);

    py::class_<LinearModel>(m, "LinearModel").def_readonly("F", &LinearModel::F).def_readonly("Q", &LinearModel::Q);
    m.def("constant_velocity_model", &constant_velocity_model, py::arg("dims"), py::arg("dt"), py::arg("q"));
    py::class_<LinearRun>(m, "LinearRun")
        .def_readonly("states", &LinearRun::states)
        .def_readonly("measurements", &LinearRun::measurements);
    m.def("simulate_linear_system", &simulate_linear_system, py::arg("F"), py::arg("Q"), py::arg("H"), py::arg("R"),
          py::arg("x0"), py::arg("steps"), py::arg("rng"));
    m.def("batch_least_squares", &batch_least_squares, py::arg("x0"), py::arg("P0"), py::arg("z"), py::arg("H"),
          py::arg("R"));
}
