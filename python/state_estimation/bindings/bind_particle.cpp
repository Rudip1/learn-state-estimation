#include "bindings.hpp"
#include "state_estimation/particle_filter.hpp"

using namespace state_estimation;

void bind_particle(py::module_& m) {
    py::enum_<Resampling>(m, "Resampling")
        .value("Multinomial", Resampling::Multinomial)
        .value("Stratified", Resampling::Stratified)
        .value("Systematic", Resampling::Systematic)
        .value("Residual", Resampling::Residual);

    m.def("effective_sample_size", py::overload_cast<const VectorXd&>(&effective_sample_size), py::arg("weights"),
          "1 / sum w_i^2, eq. (7.7)");
    m.def("resample_multinomial", &resample_multinomial, py::arg("weights"), py::arg("rng"));
    m.def("resample_stratified", &resample_stratified, py::arg("weights"), py::arg("rng"));
    m.def("resample_systematic", &resample_systematic, py::arg("weights"), py::arg("rng"));
    m.def("resample_residual", &resample_residual, py::arg("weights"), py::arg("rng"));
    m.def("resample", &resample, py::arg("weights"), py::arg("method"), py::arg("rng"));

    py::class_<ParticleFilterLocalization>(m, "ParticleFilterLocalization", "Monte Carlo localisation, section 7.5")
        .def(py::init<int, std::uint64_t>(), py::arg("n_particles"), py::arg("seed") = 0)
        .def("init_gaussian", &ParticleFilterLocalization::init_gaussian, py::arg("mean"), py::arg("cov"))
        .def("init_uniform", &ParticleFilterLocalization::init_uniform, py::arg("x_min"), py::arg("x_max"),
             py::arg("y_min"), py::arg("y_max"))
        .def("predict_wheels", &ParticleFilterLocalization::predict_wheels, py::arg("dd"), py::arg("d_left"),
             py::arg("d_right"), py::arg("var_per_meter"))
        .def("predict_odometry", &ParticleFilterLocalization::predict_odometry, py::arg("odom_prev"),
             py::arg("odom_curr"), py::arg("alpha"))
        .def("update_landmarks", &ParticleFilterLocalization::update_landmarks, py::arg("observations"),
             py::arg("landmarks"), py::arg("R"))
        .def("update_landmarks_anonymous", &ParticleFilterLocalization::update_landmarks_anonymous,
             py::arg("observations"), py::arg("landmarks"), py::arg("R"))
        .def("update_position_fix", &ParticleFilterLocalization::update_position_fix, py::arg("z"), py::arg("R"))
        .def("resample_if_needed", &ParticleFilterLocalization::resample_if_needed, py::arg("threshold") = 0.5,
             py::arg("method") = Resampling::Systematic)
        .def("augmented_resample", &ParticleFilterLocalization::augmented_resample, py::arg("log_evidence"),
             py::arg("alpha_slow"), py::arg("alpha_fast"), py::arg("x_min"), py::arg("x_max"), py::arg("y_min"),
             py::arg("y_max"), py::arg("max_fraction") = 1.0, py::arg("method") = Resampling::Systematic)
        .def("mean", &ParticleFilterLocalization::mean)
        .def("covariance", &ParticleFilterLocalization::covariance)
        .def("set_particles", &ParticleFilterLocalization::set_particles, py::arg("particles"))
        .def_property_readonly("size", &ParticleFilterLocalization::size)
        .def_property_readonly("particles",
                               [](const ParticleFilterLocalization& p) -> MatrixXd { return p.particles(); })
        .def_property_readonly("weights", &ParticleFilterLocalization::weights)
        .def_property_readonly("effective_sample_size",
                               py::overload_cast<>(&ParticleFilterLocalization::effective_sample_size, py::const_))
        .def_property_readonly("last_injection_fraction", &ParticleFilterLocalization::last_injection_fraction);
}
