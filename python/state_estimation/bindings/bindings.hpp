#pragma once
// One bind_* function per chapter; module.cpp calls them in order.
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_gaussian(py::module_& m);
