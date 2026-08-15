"""M1b: canonical benchmark adapter keeps Raw ASP and applies Safe ASP policy."""

from __future__ import annotations

import os
from types import SimpleNamespace

import cv2
import numpy as np
from asp_backend.core.pipeline.bench_adapter import run_canonical_asp
from asp_backend.core.pipeline.safety_policy import SafeAspPolicy
from asp_backend.core.pipeline.session import PipelineSession, ResultIdentity
from PIL import Image


def _write_png(path, lum: int = 128, h: int = 32, w: int = 32) -> str:
    Image.new("RGB", (w, h), (lum, lum, lum)).save(path)
    return str(path)


def _banded(h: int = 80, w: int = 48) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2] = 20
    img[h // 2 :] = 220
    return img


class _FakePipeline:
    def __init__(self, raw: np.ndarray, identity: str = ResultIdentity.RAW_ASP):
        self.raw = raw
        self.forced_identity = identity
        self.last_session = None
        self.use_basic = False
        self.use_birefnet = True
        self.use_loftr = True
        self.use_efficient_loftr = False
        self.use_aliked = False
        self.use_roma = False
        self.use_sea_raft = False
        self.use_jamma = False
        self.use_ecc = True
        self.renderer = "median"
        self.composite_fg = True
        self.bands = 5
        self.edge_crop = 30
        self.motion_model = "translation"
        self.kwargs = {}
        self.calls: list[tuple] = []

    def run(self, image_paths, output_path, session=None, **_kwargs):
        self.calls.append((list(image_paths), output_path))
        cv2.imwrite(output_path, self.raw)
        sess = session or PipelineSession.create(image_paths, output_path, host=self)
        sess.finish(success=True, identity=self.forced_identity)
        self.last_session = sess
        return SimpleNamespace()


class TestRunCanonicalAsp:
    def test_accept_copies_raw_to_safe(self, tmp_path):
        raw_path = str(tmp_path / "raw.png")
        safe_path = str(tmp_path / "safe.png")
        scans_path = _write_png(tmp_path / "scans.png", 128)
        frames = [
            _write_png(tmp_path / "a.png", 40),
            _write_png(tmp_path / "b.png", 80),
        ]
        pipe = _FakePipeline(np.full((32, 32, 3), 128, dtype=np.uint8))
        policy = SafeAspPolicy(
            composite_sc_floor=999,
            composite_sb_floor=999,
            ghost_ratio=99,
            seam_vis_ratio=99,
        )
        result = run_canonical_asp(
            frames,
            raw_asp_path=raw_path,
            safe_asp_path=safe_path,
            scans_path=scans_path,
            policy=policy,
            pipeline=pipe,
        )
        assert pipe.calls, "canonical adapter must call pipeline.run()"
        assert result.identity == ResultIdentity.RAW_ASP
        assert not result.used_fallback
        assert cv2.imread(raw_path) is not None
        assert cv2.imread(safe_path) is not None

    def test_policy_reject_keeps_raw_and_publishes_scans(self, tmp_path):
        raw_path = str(tmp_path / "raw.png")
        safe_path = str(tmp_path / "safe.png")
        scans = np.full((80, 48, 3), 128, dtype=np.uint8)
        scans_path = str(tmp_path / "scans.png")
        cv2.imwrite(scans_path, scans)
        frames = [
            _write_png(tmp_path / "a.png", 40),
            _write_png(tmp_path / "b.png", 80),
        ]
        pipe = _FakePipeline(_banded())
        policy = SafeAspPolicy(
            composite_sc_floor=5.0,
            composite_sb_floor=5.0,
            ghost_ratio=99,
            seam_vis_ratio=99,
        )
        result = run_canonical_asp(
            frames,
            raw_asp_path=raw_path,
            safe_asp_path=safe_path,
            scans_path=scans_path,
            policy=policy,
            pipeline=pipe,
        )
        assert result.used_fallback
        assert result.identity == ResultIdentity.SAFE_ASP
        assert result.fallback_reason.startswith("composite_gate_")
        raw = cv2.imread(raw_path)
        safe = cv2.imread(safe_path)
        assert raw is not None and safe is not None
        # Raw stays the banded ASP composite; published safe is SCANS.
        assert int(raw.mean()) != int(safe.mean())
        assert abs(int(safe.mean()) - 128) < 2

    def test_internal_scans_identity_skips_policy(self, tmp_path):
        raw_path = str(tmp_path / "raw.png")
        safe_path = str(tmp_path / "safe.png")
        scans_path = _write_png(tmp_path / "scans.png", 200)
        frames = [
            _write_png(tmp_path / "a.png", 40),
            _write_png(tmp_path / "b.png", 80),
        ]
        internal = np.full((32, 32, 3), 10, dtype=np.uint8)
        pipe = _FakePipeline(internal, identity=ResultIdentity.SCANS)
        policy = SafeAspPolicy(composite_sc_floor=0.0, composite_sb_floor=0.0)
        result = run_canonical_asp(
            frames,
            raw_asp_path=raw_path,
            safe_asp_path=safe_path,
            scans_path=scans_path,
            policy=policy,
            pipeline=pipe,
        )
        assert result.identity == ResultIdentity.SCANS
        assert result.used_fallback
        assert result.raw_asp_available is False
        assert result.raw_asp_path == ""
        assert not os.path.isfile(raw_path)
        published = cv2.imread(safe_path)
        assert published is not None
        assert int(published.mean()) < 20

    def test_ungated_keeps_raw_even_when_policy_would_reject(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASP_BENCH_UNGATED", "1")
        raw_path = str(tmp_path / "raw.png")
        safe_path = str(tmp_path / "safe.png")
        scans = np.full((80, 48, 3), 128, dtype=np.uint8)
        scans_path = str(tmp_path / "scans.png")
        cv2.imwrite(scans_path, scans)
        frames = [
            _write_png(tmp_path / "a.png", 40),
            _write_png(tmp_path / "b.png", 80),
        ]
        pipe = _FakePipeline(_banded())
        policy = SafeAspPolicy(
            composite_sc_floor=5.0,
            composite_sb_floor=5.0,
            ghost_ratio=99,
            seam_vis_ratio=99,
        )
        result = run_canonical_asp(
            frames,
            raw_asp_path=raw_path,
            safe_asp_path=safe_path,
            scans_path=scans_path,
            policy=policy,
            pipeline=pipe,
        )
        assert result.raw_asp_available
        assert result.identity == ResultIdentity.RAW_ASP
        assert result.extra.get("policy_would_reject", "").startswith("composite_gate_")
        raw = cv2.imread(raw_path)
        published = cv2.imread(safe_path)
        assert raw is not None and published is not None
        assert int(raw.mean()) == int(published.mean())
