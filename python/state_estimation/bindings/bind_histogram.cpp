#include "bindings.hpp"
#include "state_estimation/histogram_filter.hpp"

using namespace state_estimation;

void bind_histogram(py::module_& m) {
    py::class_<DiscreteBayesFilter>(m, "DiscreteBayesFilter", "Bayes filter over a finite state set, eq. (4.6)-(4.7)")
        .def(py::init<const VectorXd&>(), py::arg("prior"))
        .def("predict", &DiscreteBayesFilter::predict, py::arg("transition"))
        .def("predict_cyclic", &DiscreteBayesFilter::predict_cyclic, py::arg("kernel"), py::arg("offset"))
        .def("update", &DiscreteBayesFilter::update, py::arg("likelihood"))
        .def_property_readonly("belief", &DiscreteBayesFilter::belief);

    py::class_<GridSpec>(m, "GridSpec")
        .def(py::init([](double x_min, double x_max, double y_min, double y_max, double resolution, int n_theta) {
                 return GridSpec{x_min, x_max, y_min, y_max, resolution, n_theta};
             }),
             py::arg("x_min"), py::arg("x_max"), py::arg("y_min"), py::arg("y_max"), py::arg("resolution"),
             py::arg("n_theta"))
        .def_readwrite("x_min", &GridSpec::x_min)
        .def_readwrite("x_max", &GridSpec::x_max)
        .def_readwrite("y_min", &GridSpec::y_min)
        .def_readwrite("y_max", &GridSpec::y_max)
        .def_readwrite("resolution", &GridSpec::resolution)
        .def_readwrite("n_theta", &GridSpec::n_theta);

    py::class_<GridLocalization>(m, "GridLocalization", "Grid localisation over (x, y, theta), section 4.3")
        .def(py::init<const GridSpec&>(), py::arg("spec"))
        .def_property_readonly("nx", &GridLocalization::nx)
        .def_property_readonly("ny", &GridLocalization::ny)
        .def_property_readonly("n_theta", &GridLocalization::n_theta)
        .def_property_readonly("size", &GridLocalization::size)
        .def_property_readonly("spec", &GridLocalization::spec)
        .def("cell_center", &GridLocalization::cell_center, py::arg("ix"), py::arg("iy"), py::arg("it"))
        .def("set_uniform", &GridLocalization::set_uniform)
        .def("set_gaussian", &GridLocalization::set_gaussian, py::arg("mean"), py::arg("cov"))
        .def("predict_odometry", &GridLocalization::predict_odometry, py::arg("odom_prev"), py::arg("odom_curr"),
             py::arg("alpha"))
        .def("update_range_bearing", &GridLocalization::update_range_bearing, py::arg("observations"),
             py::arg("landmarks"), py::arg("R"), py::arg("account_for_cell_size") = true)
        .def("update_range_bearing_anonymous", &GridLocalization::update_range_bearing_anonymous,
             py::arg("observations"), py::arg("landmarks"), py::arg("R"), py::arg("account_for_cell_size") = true)
        .def("mix_uniform", &GridLocalization::mix_uniform, py::arg("eps"))
        .def("mean", &GridLocalization::mean)
        .def("covariance", &GridLocalization::covariance)
        .def("map_estimate", &GridLocalization::map_estimate)
        .def("marginal_xy", &GridLocalization::marginal_xy)
        .def_property("belief", &GridLocalization::belief, &GridLocalization::set_belief);
}
