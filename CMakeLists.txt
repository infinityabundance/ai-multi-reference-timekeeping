cmake_minimum_required(VERSION 3.20)
project(ai_multi_reference_timekeeping LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(AIMRT_BUILD_TESTS "Build unit tests" ON)

add_library(aimrt
    cpp/src/api.cpp
    cpp/src/characterization.cpp
    cpp/src/chrony.cpp
    cpp/src/config.cpp
    cpp/src/daemon.cpp
    cpp/src/fusion.cpp
    cpp/src/kalman.cpp
    cpp/src/logging.cpp
    cpp/src/metrics.cpp
    cpp/src/ml_model.cpp
    cpp/src/observability.cpp
    cpp/src/partitioning.cpp
    cpp/src/references.cpp
    cpp/src/safety.cpp
    cpp/src/time_server.cpp
)

target_include_directories(aimrt PUBLIC cpp/include)

if (AIMRT_BUILD_TESTS)
    enable_testing()
    add_executable(aimrt_tests
        cpp/tests/test_main.cpp
    )
    target_link_libraries(aimrt_tests PRIVATE aimrt)
    add_test(NAME aimrt_tests COMMAND aimrt_tests)
endif()
