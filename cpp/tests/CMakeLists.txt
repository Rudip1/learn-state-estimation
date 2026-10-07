FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(Catch2)

set(SE_TESTS
    test_gaussian.cpp
    test_motion_models.cpp
    test_measurement_models.cpp
    test_histogram_filter.cpp
    test_kalman_filter.cpp
    test_ekf.cpp
)

add_executable(state_estimation_tests ${SE_TESTS})
target_link_libraries(state_estimation_tests PRIVATE state_estimation Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(state_estimation_tests)
