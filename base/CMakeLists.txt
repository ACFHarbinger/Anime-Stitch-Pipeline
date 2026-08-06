cmake_minimum_required(VERSION 3.20)
project(anime_stitch_pipeline_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTING "Build tests" OFF)

# ---------------------------------------------------------------------------
# NOTE: this C++ core does not build a standalone Python extension. It is
# compiled directly into Image-Toolkit's single `base` pybind11 module
# (see ../../../base/CMakeLists.txt, which lists these same src/
# sources by relative path) — that is the only way these sources are
# actually loaded at runtime. This CMakeLists.txt exists so `base/tests/`
# can be built and run in isolation (`just test::cpp`, this submodule's own
# CI), and therefore assumes this repo is checked out at
# Image-Toolkit/submodules/Anime-Stitch-Pipeline/ — it depends on shared
# headers (affine_types.hpp, common.hpp, image_utils.hpp, math/graph.hpp)
# from Image-Toolkit's base/include/ that are not vendored here.
# ---------------------------------------------------------------------------

set(IMAGE_TOOLKIT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

find_package(OpenCV 4.6 REQUIRED COMPONENTS core imgproc imgcodecs stitching photo)

include(FetchContent)
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING_SAVED ${BUILD_TESTING})
    set(BUILD_TESTING OFF)
    FetchContent_MakeAvailable(eigen)
    set(BUILD_TESTING ${BUILD_TESTING_SAVED})
endif()

find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

find_package(OpenMP REQUIRED)

# Static library — the same source list base/CMakeLists.txt's `base` pybind11
# module compiles, built here without the pybind11 module entry point (that
# lives in ../../../base/src/bindings.cpp) so Catch2 tests can link it
# directly, matching ../../../base/tests/CMakeLists.txt's own approach.
add_library(animation_impl STATIC
    src/matching.cpp
    src/bundle_adjust.cpp
    src/validation.cpp
    src/canvas.cpp
    src/seam.cpp
    src/compositing.cpp
    src/exposure.cpp
    src/frame_selection.cpp
    src/fg_register.cpp
)

target_include_directories(animation_impl PUBLIC
    ${IMAGE_TOOLKIT_ROOT}/base/include
    ${OpenCV_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

if(Eigen3_FOUND)
    target_link_libraries(animation_impl PUBLIC Eigen3::Eigen)
else()
    target_include_directories(animation_impl PUBLIC ${eigen_SOURCE_DIR})
endif()

target_link_libraries(animation_impl PUBLIC
    ${OpenCV_LIBS}
    OpenMP::OpenMP_CXX
)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
