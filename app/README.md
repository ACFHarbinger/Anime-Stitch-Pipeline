# app/

**Scaffold — not yet implemented.** Native mobile UIs for ASP.

```
app/
  ios/       Swift/SwiftUI app skeleton
  android/   Kotlin/Jetpack Compose app skeleton
```

Both are expected to talk to `backend/`'s pipeline through a remote API
(there is no on-device path for the full stitching pipeline — it needs
the C++ `base` extension and GPU-capable ML matchers) rather than running
`AnimeStitchPipeline` locally. See each subdirectory's README for status.
