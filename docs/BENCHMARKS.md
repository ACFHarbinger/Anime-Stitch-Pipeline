# Benchmarks

There are two distinct kinds of benchmark in this repo — don't confuse them:

| Kind | Tool | Location | Measures |
| --- | --- | --- | --- |
| Micro-benchmarks | pytest-benchmark / Google Benchmark | `backend/benchmark/bench_*.py`, `base/` (via CTest) | Function/stage-level speed. |
| **ASP quality benchmark** | Custom harness | `backend/benchmark/bench_anime_stitch.py`, run via `just asp-benchmark` (full 97 tests, ~2.5h) / `just asp-benchmark-verify` (5-test subset) | **The real quality signal for the ASP pipeline** — ASP vs. OpenCV SCANS vs. Overmix vs. Hugin vs. ground truth, across 97 test sequences (55 with ground truth). This is what the roadmap's "one change → one benchmark" rule refers to. See `docs/moon/ROADMAP.md` and `.agent/cache/asp_state_of_the_pipeline.md` §4–5 for current results and metric definitions. **No automated metric here measures structural coherence** — side-by-side visual review (`just asp-benchmark-assess`) is mandatory before trusting any verdict.

Run micro-benchmarks with `just bench`. CI runs them on pushes to `main` that
touch a `benchmark/`/`benches/` directory — see
[`.github/workflows/benchmark.yml`](../.github/workflows/benchmark.yml). The
quality benchmark is run manually/on-demand, not in CI (it's multi-hour and
resource-intensive — see the ROADMAP's host-freeze note, Phase 2.6).
