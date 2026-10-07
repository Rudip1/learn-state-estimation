#pragma once
// One bind_* function per chapter; module.cpp calls them in order.
// Properties return copies, never views into C++ objects that keep changing.
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_gaussian(py::module_& m);
void bind_motion(py::module_& m);
void bind_measurement(py::module_& m);
void bind_histogram(py::module_& m);
void bind_kalman(py::module_& m);
void bind_ekf(py::module_& m);
