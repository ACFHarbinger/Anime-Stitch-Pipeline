# cpp/

C++17 module template, built with CMake.

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build cpp/build --parallel
ctest --test-dir cpp/build --output-on-failure
```

| Directory | Purpose |
| --- | --- |
| `include/` | Public headers |
| `src/` | Implementation |
| `test/` | GoogleTest tests (registered with CTest) |
| `benchmark/` | Google Benchmark micro-benchmarks |
| `config/` | Runtime configuration |
