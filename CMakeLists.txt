cmake_minimum_required(VERSION 3.20)
project(state_estimation VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# When built by scikit-build-core (pip install .) only the Python module is needed.
if(SKBUILD)
    set(_se_default_tests OFF)
    set(_se_default_python ON)
else()
    set(_se_default_tests ON)
    set(_se_default_python OFF)
endif()
option(SE_BUILD_TESTS "Build the Catch2 test suite" ${_se_default_tests})
option(SE_BUILD_EXAMPLES "Build the command-line examples" ${_se_default_tests})
option(SE_BUILD_PYTHON "Build the pybind11 module" ${_se_default_python})

include(FetchContent)

# ---- Eigen 3.4: system package if present, otherwise a header-only download ----
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 not found on the system: fetching 3.4.0")
    FetchContent_Declare(
        eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR _header_only_do_not_configure)
    FetchContent_MakeAvailable(eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}")
endif()

# ---- The library ----
add_library(state_estimation STATIC
    cpp/src/gaussian.cpp
    cpp/src/motion_models.cpp
    cpp/src/pose2.cpp
    cpp/src/simulation.cpp
)
target_include_directories(state_estimation PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>)
target_link_libraries(state_estimation PUBLIC Eigen3::Eigen)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(state_estimation PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(SE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(cpp/tests)
endif()

if(SE_BUILD_EXAMPLES)
    add_subdirectory(cpp/examples)
endif()

if(SE_BUILD_PYTHON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_core MODULE
        python/state_estimation/bindings/module.cpp
        python/state_estimation/bindings/bind_gaussian.cpp
        python/state_estimation/bindings/bind_motion.cpp
    )
    target_link_libraries(_core PRIVATE state_estimation)
    install(TARGETS _core DESTINATION state_estimation)
endif()
