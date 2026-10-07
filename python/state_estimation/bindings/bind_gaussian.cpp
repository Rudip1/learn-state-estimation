#include "bindings.hpp"
#include "state_estimation/gaussian.hpp"

using namespace state_estimation;

void bind_gaussian(py::module_& m) {
    py::class_<Rng>(m, "Rng", "Seeded random number generator (Mersenne Twister 64)")
        .def(py::init<std::uint64_t>(), py::arg("seed") = 0)
        .def("seed", &Rng::seed, py::arg("seed"))
        .def("normal", py::overload_cast<>(&Rng::normal), "Standard normal sample")
        .def("uniform", &Rng::uniform, py::arg("lo") = 0.0, py::arg("hi") = 1.0);

    py::class_<Gaussian>(m, "Gaussian", "Multivariate Gaussian N(mean, cov)")
        .def(py::init<>())
        .def(py::init([](const VectorXd& mean, const MatrixXd& cov) { return Gaussian{mean, cov}; }),
             py::arg("mean"), py::arg("cov"))
        .def_readwrite("mean", &Gaussian::mean)
        .def_readwrite("cov", &Gaussian::cov)
        .def("__repr__", [](const Gaussian& g) {
            return "Gaussian(dim=" + std::to_string(g.mean.size()) + ")";
        });

    py::class_<CovarianceEllipse>(m, "CovarianceEllipse")
        .def_readonly("semi_major", &CovarianceEllipse::semi_major)
        .def_readonly("semi_minor", &CovarianceEllipse::semi_minor)
        .def_readonly("angle", &CovarianceEllipse::angle);

    m.def("mahalanobis_squared", &mahalanobis_squared, py::arg("x"), py::arg("mean"), py::arg("cov"),
          "Squared Mahalanobis distance (x-mean)^T cov^-1 (x-mean), eq. (1.14)");
    m.def("gaussian_log_pdf", &gaussian_log_pdf, py::arg("x"), py::arg("mean"), py::arg("cov"));
    m.def("gaussian_pdf", &gaussian_pdf, py::arg("x"), py::arg("mean"), py::arg("cov"));
    m.def("linear_transform", &linear_transform, py::arg("g"), py::arg("A"), py::arg("b"),
          "Exact image of a Gaussian under y = A x + b, eq. (1.7)");
    m.def("numerical_jacobian", &numerical_jacobian, py::arg("f"), py::arg("x"), py::arg("h") = 1e-6);
    m.def("propagate_linearized", &propagate_linearized, py::arg("f"), py::arg("g"),
          py::arg("jacobian") = std::function<MatrixXd(const VectorXd&)>{},
          "First-order propagation N(f(mean), J cov J^T), eq. (1.9)");
    m.def("fuse", &fuse, py::arg("a"), py::arg("b"), "Product of two Gaussians, eq. (1.18)-(1.20)");
    m.def("sample_gaussian", &sample_gaussian, py::arg("mean"), py::arg("cov"), py::arg("n"), py::arg("rng"),
          "n samples (rows) from N(mean, cov), eq. (1.16)");
    m.def("sample_mean_cov", &sample_mean_cov, py::arg("samples"));
    m.def("bayes_rule", &bayes_rule, py::arg("prior"), py::arg("likelihood"), "Discrete Bayes rule, eq. (1.12)");
    m.def("regularized_gamma_p", &regularized_gamma_p, py::arg("a"), py::arg("x"));
    m.def("chi2_cdf", &chi2_cdf, py::arg("x"), py::arg("k"));
    m.def("chi2_quantile", &chi2_quantile, py::arg("p"), py::arg("k"));
    m.def("covariance_ellipse", &covariance_ellipse, py::arg("cov"), py::arg("p"));
    m.def("ellipse_points", &ellipse_points, py::arg("mean"), py::arg("cov"), py::arg("p"), py::arg("n") = 64);
    m.def("polar_to_cartesian", &polar_to_cartesian, py::arg("polar"));
    m.def("polar_to_cartesian_jacobian", &polar_to_cartesian_jacobian, py::arg("polar"));
}
