"""M1c: headless parity across CLI, bench adapter, and GUI adapter."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from asp_backend.core.pipeline.bench_adapter import run_canonical_asp
from asp_backend.core.pipeline.manager import AnimeStitchPipeline
from asp_backend.core.pipeline.safety_policy import SafeAspPolicy
from asp_gui.helpers._progress_pipeline import _ProgressPipeline
from PIL import Image

_SAMPLES = (
    Path(__file__).resolve().parents[3] / "data" / "samples" / "test_scroll_gradient"
)


def _kwargs() -> dict:
    return dict(
        use_basic=False,
        use_birefnet=False,
        use_loftr=False,
        use_efficient_loftr=False,
        use_aliked=False,
        use_roma=False,
        use_sea_raft=False,
        use_ecc=False,
        composite_fg=False,
        renderer="first",
        motion_model="translation",
        edge_crop=0,
    )


def _sample_frames() -> list[str]:
    paths = sorted(str(p) for p in _SAMPLES.glob("frame_*.png"))
    assert len(paths) >= 3, f"missing sample frames in {_SAMPLES}"
    return paths[:4]


def _png_bytes(path: str) -> bytes:
    arr = cv2.imread(path)
    assert arr is not None
    return arr.tobytes()


class TestHeadlessEntryParity:
    def test_cli_gui_bench_same_raw_bytes(self, tmp_path):
        frames = _sample_frames()
        cli_out = str(tmp_path / "cli.png")
        gui_out = str(tmp_path / "gui.png")
        bench_raw = str(tmp_path / "bench_raw.png")
        bench_safe = str(tmp_path / "bench_safe.png")

        cli = AnimeStitchPipeline(**_kwargs())
        cli.run(frames, cli_out)

        gui = _ProgressPipeline(
            progress_cb=lambda *_a: None,
            log_cb=lambda *_a: None,
            **_kwargs(),
        )
        gui.run(frames, gui_out)

        policy = SafeAspPolicy(
            composite_sc_floor=999,
            composite_sb_floor=999,
            ghost_ratio=99,
            seam_vis_ratio=99,
        )
        run_canonical_asp(
            frames,
            raw_asp_path=bench_raw,
            safe_asp_path=bench_safe,
            scans_path=None,
            policy=policy,
            pipeline=AnimeStitchPipeline(**_kwargs()),
        )

        assert os.path.isfile(cli_out)
        assert os.path.isfile(gui_out)
        assert os.path.isfile(bench_raw)
        assert _png_bytes(cli_out) == _png_bytes(gui_out)
        assert _png_bytes(cli_out) == _png_bytes(bench_raw)
        assert cli.last_session is not None
        assert gui.last_session is not None
        assert cli.last_session.digest() == gui.last_session.digest()

    def test_gui_default_is_not_legacy_override(self):
        assert _ProgressPipeline.run is not _ProgressPipeline._run_legacy

    def test_canonical_pause_applies_exclusion_masks(self, tmp_path):
        frames = _sample_frames()
        probe = cv2.imread(frames[0])
        assert probe is not None
        h, w = probe.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[10:20, 10:20] = 255

        def hook(event, _data):
            if event == "masks":
                return {"exclusion_masks": [mask] * len(frames)}
            return {}

        pipe = AnimeStitchPipeline(**_kwargs())
        pipe.pause_hook = hook
        pipe.run(frames, str(tmp_path / "out.png"), pause_hook=hook)
        assert pipe.exclusion_masks is not None
        assert len(pipe.exclusion_masks) == len(frames)
