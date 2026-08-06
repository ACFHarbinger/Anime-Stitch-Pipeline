"""
HITL session persistence and replay (§S88).

Saves the override dicts from every HITL checkpoint into a single JSON file
so that a completed interactive run can be replayed non-interactively.

Serialisation rules
-------------------
- Plain Python scalars / lists / dicts: stored as-is.
- numpy arrays: ``{"__ndarray__": true, "dtype": <str>, "shape": <list>,
  "data_b64": <base64-encoded raw bytes>}``.
- Lists whose first element is an ndarray are serialised element-by-element.
- Arrays larger than MAX_ARRAY_BYTES are replaced with a sentinel
  ``{"__ndarray__": true, "skipped": true}`` to keep session files small.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import numpy as np
    _NP_OK = True
except ImportError:  # pragma: no cover
    _NP_OK = False

HITL_SESSION_DIR = Path.home() / ".config" / "image-toolkit" / "hitl_sessions"

MAX_ARRAY_BYTES = 8 * 1024 * 1024  # 8 MB threshold — skip serialising larger arrays

# ---------------------------------------------------------------------------
# ndarray ↔ JSON helpers
# ---------------------------------------------------------------------------


def _encode_array(arr: Any) -> dict:
    """Encode a numpy array to a JSON-safe dict."""
    if arr.nbytes > MAX_ARRAY_BYTES:
        return {"__ndarray__": True, "skipped": True}
    raw = arr.tobytes()
    return {
        "__ndarray__": True,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data_b64": base64.b64encode(raw).decode("ascii"),
    }


def _decode_array(d: dict) -> Optional[Any]:
    """Decode a JSON dict back to a numpy array, or None if it was skipped."""
    if d.get("skipped"):
        return None
    if not _NP_OK:
        return None
    raw = base64.b64decode(d["data_b64"])
    return np.frombuffer(raw, dtype=d["dtype"]).reshape(d["shape"]).copy()


def _to_json(obj: Any) -> Any:
    """Recursively convert an object to a JSON-serialisable form."""
    if _NP_OK and isinstance(obj, np.ndarray):
        return _encode_array(obj)
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if _NP_OK and isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _from_json(obj: Any) -> Any:
    """Recursively decode a JSON structure, restoring numpy arrays."""
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            return _decode_array(obj)
        return {k: _from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_json(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_session(overrides: Dict[str, dict], path: str) -> None:
    """Write accumulated checkpoint overrides to *path* as JSON."""
    payload = {
        "version": 1,
        "timestamp": time.time(),
        "checkpoints": _to_json(overrides),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_session(path: str) -> Dict[str, dict]:
    """Read a session JSON file and return the checkpoints dict."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    checkpoints = raw.get("checkpoints", {})
    return _from_json(checkpoints)


def autosave_path() -> str:
    """Return a timestamped path in the default session directory."""
    HITL_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return str(HITL_SESSION_DIR / f"session_{ts}.json")


__all__ = ["save_session", "load_session", "autosave_path", "HITL_SESSION_DIR"]
