#include <benchmark/benchmark.h>

#include "dev_repo_template/greet.hpp"

static void BM_Greet(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(dev_repo_template::greet("world"));
    }
}
BENCHMARK(BM_Greet);
