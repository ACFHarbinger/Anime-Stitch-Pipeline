# frontend/

**Scaffold — not yet implemented.** A Tauri-based cross-platform desktop UI
for ASP, as an alternative to the PySide6 desktop UI in `../gui/`. Intended
to talk to the same `backend/` pipeline (`AnimeStitchPipeline`) via a local
HTTP/IPC bridge, the same way Image-Toolkit's own `frontend/` talks to its
Django backend — not yet wired up.

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
