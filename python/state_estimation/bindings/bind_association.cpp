#include "bindings.hpp"
#include "state_estimation/data_association.hpp"

using namespace state_estimation;

void bind_association(py::module_& m) {
    py::class_<AssociationProblem>(m, "AssociationProblem", "Linearised association problem, section 8.1")
        .def(py::init<>())
        .def(py::init([](std::vector<VectorXd> z, std::vector<VectorXd> zhat, std::vector<MatrixXd> H, MatrixXd P,
                         MatrixXd R, std::vector<int> angles) {
                 return AssociationProblem{z, zhat, H, P, R, angles};
             }),
             py::arg("observations"), py::arg("predictions"), py::arg("jacobians"), py::arg("P"), py::arg("R"),
             py::arg("angle_indices") = std::vector<int>{})
        .def_readwrite("observations", &AssociationProblem::observations)
        .def_readwrite("predictions", &AssociationProblem::predictions)
        .def_readwrite("jacobians", &AssociationProblem::jacobians)
        .def_readwrite("P", &AssociationProblem::P)
        .def_readwrite("R", &AssociationProblem::R)
        .def_readwrite("angle_indices", &AssociationProblem::angle_indices)
        .def_property_readonly("n_observations", &AssociationProblem::n_observations)
        .def_property_readonly("n_features", &AssociationProblem::n_features);

    py::class_<Association>(m, "Association")
        .def_readonly("pairing", &Association::pairing)
        .def_readonly("n_paired", &Association::n_paired)
        .def_readonly("joint_nis", &Association::joint_nis)
        .def("__repr__", [](const Association& a) {
            std::string s = "Association([";
            for (size_t i = 0; i < a.pairing.size(); ++i) s += (i ? ", " : "") + std::to_string(a.pairing[i]);
            return s + "], n_paired=" + std::to_string(a.n_paired) + ")";
        });

    m.def("individual_nis", &individual_nis, py::arg("problem"), py::arg("i"), py::arg("j"), "eq. (8.2)-(8.3)");
    m.def("joint_nis", &joint_nis, py::arg("problem"), py::arg("pairing"), "eq. (8.5)-(8.6); returns (nis, dof)");
    m.def("associate_nearest_neighbor", &associate_nearest_neighbor, py::arg("problem"), py::arg("max_distance"));
    m.def("associate_icnn", &associate_icnn, py::arg("problem"), py::arg("alpha") = 0.05);
    m.def("associate_jcbb", &associate_jcbb, py::arg("problem"), py::arg("alpha") = 0.05);
    m.def("associate_brute_force", &associate_brute_force, py::arg("problem"), py::arg("alpha") = 0.05);
    m.def("landmark_association_problem", &landmark_association_problem, py::arg("x_bar"), py::arg("P"),
          py::arg("observations"), py::arg("landmarks"), py::arg("R"));
    m.def("add_clutter", &add_clutter, py::arg("observations"), py::arg("n"), py::arg("sensor"), py::arg("rng"));
}
