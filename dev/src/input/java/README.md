# input/java/

Not yet implemented — tracked as part of moon/roadmaps/developer_tools.md D9
(added after the roadmap's original language list; java/ was added to the
template after developer_tools.md was written).

Once built, this parser must emit a `CodeGraph` matching
`dev/src/input/protobuf/codegraph.proto` (see `dev/src/input/python/parser.py`
for the reference shape). `dev/src/core/aggregate.py` will pick it up
automatically — no core code changes needed to add a language, only wiring
in `dev/src/cli.py`'s `build_graph()`.

If this repo has no java module, delete this directory entirely; nothing
else in `dev/` depends on it.
