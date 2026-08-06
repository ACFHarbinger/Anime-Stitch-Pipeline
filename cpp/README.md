# cpp/

ASP's C++ compute core (animation stage sources), built with CMake and
Catch2. These sources are compiled into Image-Toolkit's single `base`
pybind11 extension (`../../../base/CMakeLists.txt`), which is the only way
they're actually loaded at runtime — this CMakeLists.txt exists so
`test/` can be built and run in isolation for this submodule's own CI.
Assumes checkout at `Image-Toolkit/submodules/Anime-Stitch-Pipeline/`,
since it depends on shared headers from `../../../base/include/`.

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build cpp/build --parallel
ctest --test-dir cpp/build --output-on-failure
```

| Directory | Purpose |
| --- | --- |
| `src/animation/` | Pipeline stage implementations (matching, bundle adjust, canvas, seam, compositing, exposure, frame selection, foreground registration) |
| `test/animation/` | Catch2 unit tests (registered with CTest) |
| `config/` | Runtime configuration |

Requires: OpenCV >= 4.6, Eigen3, pybind11 headers (for `py::array_t`
interop types only — no Python module entry point is built here), OpenMP.
