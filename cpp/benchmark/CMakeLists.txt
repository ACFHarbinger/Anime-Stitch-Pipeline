find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.9.0.zip
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
endif()

add_executable(dev_repo_template_benchmark greet_benchmark.cpp)
target_link_libraries(dev_repo_template_benchmark PRIVATE dev_repo_template_cpp benchmark::benchmark_main)
