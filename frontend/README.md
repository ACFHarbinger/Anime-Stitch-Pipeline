# frontend/

**Scaffold — not yet implemented, and frozen.** A Tauri-based cross-platform
desktop UI for ASP, as an alternative to the PySide6 desktop UI in `../gui/`.
Intended to talk to the same `backend/` pipeline (`AnimeStitchPipeline`) via a
local HTTP/IPC bridge, the same way Image-Toolkit's own `frontend/` talks to
its Django backend — not yet wired up.

**Frozen as of 2026-08-06** (see `docs/moon/ROADMAP.md` §0 "Product Scope"):
the PySide6 GUI in `../gui/` is the one working desktop surface and the one
that gets ongoing investment. Do not start implementing this scaffold until
the ASP roadmap's Phase 4 exit gate is met — standing up a second UI
framework before the core product (a stitcher whose output an artist would
keep) is validated is exactly the kind of scope creep the project doesn't
need more of. The skeleton stays in git because it's cheap to keep and
expensive to rebuild later, not because it's an active workstream.

```
frontend/
  src/            TypeScript/React UI (placeholder: src/main.ts)
  src-tauri/      Rust Tauri shell (placeholder: a no-op window)
```

## Status

This is a directory skeleton, not a working app. Building it out means:

1. Deciding the bridge protocol to `backend/` (REST via a small FastAPI/
   Flask wrapper around `AnimeStitchPipeline`, or PyO3 bindings called
   directly from the Tauri Rust shell).
2. Porting the stitch tab's panels (frame review, canvas preview, HITL
   edge/seam correction dialogs — see `../gui/src/elements/`) to React
   components.
3. Wiring `npm run tauri dev` / `tauri build` for desktop packaging.

## Local dev (once implemented)

```bash
cd frontend
npm install
npm run tauri dev
```
