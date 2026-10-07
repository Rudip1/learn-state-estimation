#include "bindings.hpp"

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core of the state_estimation learning module";
    bind_gaussian(m);
    bind_motion(m);
    bind_measurement(m);
    bind_histogram(m);
    bind_kalman(m);
    bind_ekf(m);
}
