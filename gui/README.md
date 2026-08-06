# gui/

**Anime Stitch Pipeline (ASP) — desktop Qt/PySide6 UI.**

The working, shipped desktop surface for ASP: the automated-pipeline Stitch
tab and the manual/interactive `HybridStitchPanel` stitching tool. See the
top-level [README](../README.md) for the project overview and
[`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for how this module fits
into the rest of the app.

## Layout

```
gui/
  src/tabs/      Tab controllers (stitch_tab_backend.py, dialog.py)
  src/elements/  Stitch tab panels (canvas, graph editor, stats, HITL review, ...)
  src/tabs/stencil/hybrid_stitch_panel.py   The manual/interactive HybridStitch tool
  test/          pytest suite (dialogs/, core/, tabs/)
```

## Development

```bash
cd gui
uv sync --group dev
uv run pytest test -v          # QT_QPA_PLATFORM=offscreen for headless runs
uv run ruff check .
uv run mypy src
```
