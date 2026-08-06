"""
Tests for HITL session persistence and replay (§S88).

Validates serialisation round-trips, array encoding, large-array truncation,
and the save/load API without requiring GPU or a running Qt application.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from backend.src.animation.hitl.hitl_session import (
    _decode_array,
    _encode_array,
    _from_json,
    _to_json,
    load_session,
    save_session,
)

# ---------------------------------------------------------------------------
# ndarray codec
# ---------------------------------------------------------------------------


class TestNdarrayCodec:
    def test_encode_decode_uint8_roundtrip(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        enc = _encode_array(arr)
        assert enc["__ndarray__"] is True
        assert enc["dtype"] == "uint8"
        assert enc["shape"] == [2, 2]
        restored = _decode_array(enc)
        np.testing.assert_array_equal(restored, arr)

    @pytest.mark.gc_heavy
    def test_large_array_is_skipped(self):
        big = np.zeros((2048, 2048), dtype=np.float32)  # 16 MB > 8 MB threshold
        enc = _encode_array(big)
        assert enc.get("skipped") is True
        assert _decode_array(enc) is None

    def test_to_json_nested_dict(self):
        arr = np.array([10, 20], dtype=np.int32)
        obj = {"frame_override": ["a.png"], "my_arr": arr}
        serial = _to_json(obj)
        assert serial["frame_override"] == ["a.png"]
        assert serial["my_arr"]["__ndarray__"] is True

    def test_from_json_restores_arrays(self):
        arr = np.array([7, 8, 9], dtype=np.float64)
        serial = _to_json({"data": arr})
        restored = _from_json(serial)
        np.testing.assert_array_almost_equal(restored["data"], arr)


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadSession:
    def test_plain_overrides_roundtrip(self, tmp_path):
        overrides = {
            "frames": {"frame_override": ["/tmp/a.png", "/tmp/b.png"]},
            "boundaries": {"boundaries": [120.5, 340.0]},
        }
        path = str(tmp_path / "session.json")
        save_session(overrides, path)
        loaded = load_session(path)
        assert loaded["frames"]["frame_override"] == ["/tmp/a.png", "/tmp/b.png"]
        assert loaded["boundaries"]["boundaries"] == pytest.approx([120.5, 340.0])

    def test_ndarray_in_overrides_roundtrip(self, tmp_path):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        overrides = {"composite": {"paint_mask": mask}}
        path = str(tmp_path / "session.json")
        save_session(overrides, path)
        loaded = load_session(path)
        restored_mask = loaded["composite"]["paint_mask"]
        assert restored_mask is not None
        np.testing.assert_array_equal(restored_mask, mask)

    def test_empty_overrides_produces_valid_file(self, tmp_path):
        path = str(tmp_path / "empty.json")
        save_session({}, path)
        raw = json.loads(__import__("pathlib").Path(path).read_text())
        assert raw["version"] == 1
        assert raw["checkpoints"] == {}
        loaded = load_session(path)
        assert loaded == {}

    def test_missing_checkpoint_returns_empty_dict(self, tmp_path):
        overrides = {"frames": {"frame_override": []}}
        path = str(tmp_path / "partial.json")
        save_session(overrides, path)
        loaded = load_session(path)
        assert loaded.get("edges", {}) == {}

    def test_save_creates_parent_directory(self, tmp_path):
        nested = str(tmp_path / "a" / "b" / "c" / "session.json")
        save_session({"render": {"cancel": False}}, nested)
        loaded = load_session(nested)
        assert loaded["render"]["cancel"] is False
