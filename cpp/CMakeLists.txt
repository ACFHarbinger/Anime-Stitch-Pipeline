cmake_minimum_required(VERSION 3.20)
project(dev_repo_template_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTING "Build tests" OFF)
option(BUILD_BENCHMARK "Build benchmarks" OFF)

add_library(dev_repo_template_cpp STATIC
    src/greet.cpp
)
target_include_directories(dev_repo_template_cpp PUBLIC include)

add_executable(dev_repo_template_cli src/main.cpp)
target_link_libraries(dev_repo_template_cli PRIVATE dev_repo_template_cpp)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()

if(BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()
