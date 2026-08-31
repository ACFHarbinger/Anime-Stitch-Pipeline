"""``StitchWorker`` -- QThread driving ``_ProgressPipeline`` with HITL checkpoints.

Extracted from ``stitch_worker.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from collections.abc import Callable

import cv2
import numpy as np
from asp_backend.hitl.hitl_session import load_session
from asp_backend.hitl.hitl_session import save_session as _save_session_impl
from asp_backend.ingestion.video_ingestion import ingest_video
from gui.src.helpers.gc_safe import gc_disabled_run
from PySide6.QtCore import QMutex, QThread, QWaitCondition, Signal

from ._progress_pipeline import _TOTAL_STAGES, _build_pipeline_kwargs, _ProgressPipeline


class StitchWorker(QThread):
    sig_stage = Signal(int, int, str)  # (current_stage, total_stages, label)
    sig_log = Signal(str)
    sig_finished = Signal(str)  # output_path
    sig_error = Signal(str)

    # HITL checkpoint signals — emitted when the pipeline pauses for review.
    # Each carries a plain dict of intermediate state (numpy arrays included).
    sig_review_video = Signal(object)  # checkpoint 0: video frame review (Issue 9 S84)
    sig_review_frames = Signal(object)  # after Stage 4: frame selection review
    sig_review_masks = Signal(
        object
    )  # after Stage 4.5: mask / segmentation review (Issue 10A2)
    sig_review_edges = Signal(object)  # after Stage 5: edge graph review
    sig_review_canvas = Signal(object)  # after Stage 8: canvas layout review
    sig_review_boundaries = Signal(object)  # checkpoint 3.5: seam boundary editor (S85)
    sig_review_seams = Signal(object)  # checkpoint 4.6: seam diagnostic inspector (S95)
    sig_review_composite = Signal(
        object
    )  # checkpoint 4.5: post-composite seam painter (S86)
    sig_review_render = Signal(object)  # after Stage 9: render / coverage review
    sig_review_output = Signal(object)  # checkpoint 5: final output RLHF feedback (S87)

    TOTAL_STAGES = _TOTAL_STAGES

    def __init__(
        self,
        image_paths: list[str],
        output_path: str,
        pipeline_config: dict,
        manual_affines: dict | None = None,
        hitl_mode: bool = False,
        video_path: str | None = None,
        video_n_frames: int = 20,
        video_mode: str = "uniform",
        session_path: str | None = None,
    ):
        super().__init__()
        self._image_paths = image_paths
        self._output_path = output_path
        self._pipeline_config = pipeline_config
        self._manual_affines = manual_affines or {}
        self._cancel_flag: list = [False]
        self._hitl_mode = hitl_mode

        # Issue 9 S84: optional video source for frame extraction
        self._video_path: str | None = video_path
        self._video_n_frames: int = video_n_frames
        self._video_mode: str = video_mode

        # Derive intermediate output directory from output path if requested.
        self._save_intermediate = pipeline_config.get("save_intermediate", False)
        if self._save_intermediate:
            stem = os.path.splitext(os.path.abspath(output_path))[0]
            self._intermediate_dir = stem + "_stages"
        else:
            self._intermediate_dir = ""

        # HITL pause/resume synchronization
        self._hitl_mutex = QMutex()
        self._hitl_wait = QWaitCondition()
        self._hitl_paused: bool = False
        self._hitl_override: dict = {}

        # Issue 10A3: NL seam-routing exclusion masks (set via set_exclusion_masks())
        self._exclusion_masks: list | None = None

        # S88: session persistence — accumulated overrides + optional replay source
        self._hitl_session_overrides: dict[str, dict] = {}
        self._current_session_path: str | None = None
        self._replay_session: dict[str, dict] = {}
        if session_path:
            with contextlib.suppress(Exception):
                self._replay_session = load_session(session_path)

    def cancel(self):
        self._cancel_flag[0] = True
        # Wake up any paused HITL checkpoint so cancel propagates immediately
        if self._hitl_paused:
            self.resume()

    def resume(self):
        """Call from the main thread to resume a paused pipeline checkpoint."""
        self._hitl_mutex.lock()
        self._hitl_paused = False
        self._hitl_wait.wakeAll()
        self._hitl_mutex.unlock()

    def set_frame_override(self, paths: list[str]) -> None:
        """Set frame list override (call before resume() at the frame checkpoint)."""
        self._hitl_override["frame_override"] = paths

    def set_mask_override(self, masks: list) -> None:
        """Set bg_mask list override (call before resume() at the mask checkpoint)."""
        self._hitl_override["bg_masks"] = masks

    def set_exclusion_masks(self, exclusion_masks: list) -> None:
        """Set NL seam-routing exclusion masks (Issue 10A3). Call before resume()."""
        self._hitl_override["exclusion_masks"] = exclusion_masks

    def set_edge_override(self, edges: list[dict]) -> None:
        """Set edge list override (call before resume() at the edge checkpoint)."""
        self._hitl_override["edges"] = edges

    def set_affine_override(self, affines: list) -> None:
        """Set affine matrix override (call before resume() at the canvas checkpoint)."""
        self._hitl_override["affines"] = affines

    def set_boundary_override(self, boundaries: list) -> None:
        """Set seam-boundary y-coordinate override (call before resume() at checkpoint 3.5)."""
        self._hitl_override["boundaries"] = boundaries

    def set_seam_override(self, overrides: dict) -> None:
        """Set per-seam override dict for HITL checkpoint 4.6.

        *overrides* maps seam index k (int or str) → option dict with optional
        keys ``"force_single_pose"`` (bool) and ``"force_blend"`` (bool).
        Call before :meth:`resume` at the seam-diagnostic checkpoint.
        """
        self._hitl_override["seam_overrides"] = {
            int(k): v for k, v in overrides.items()
        }

    def set_paint_mask(self, mask: np.ndarray) -> None:
        """Set canvas-space paint mask for re-composite (call before resume() at checkpoint 4.5)."""
        self._hitl_override["paint_mask"] = mask

    def set_render_cancel(self) -> None:
        """Signal the pipeline to abort at the render review checkpoint."""
        self._hitl_override["cancel"] = True

    def set_output_feedback(self, overall_rating: float, annotations: list) -> None:
        """Store RLHF feedback to persist after checkpoint 5 (call before resume())."""
        self._hitl_override["output_feedback"] = {
            "overall_rating": overall_rating,
            "annotations": annotations,
        }

    # S88 ----------------------------------------------------------------- #

    @property
    def current_session_path(self) -> str | None:
        """Path of the autosaved session JSON (set after a successful run)."""
        return self._current_session_path

    def save_session(self, path: str) -> None:
        """Write accumulated checkpoint overrides to *path*."""
        try:
            _save_session_impl(self._hitl_session_overrides, path)
            self._current_session_path = path
        except Exception:
            pass

    def _make_hitl_pause_cb(self) -> Callable:
        """Return a callable that emits the right signal then blocks the worker thread."""
        _signal_map = {
            "frames": self.sig_review_frames,
            "masks": self.sig_review_masks,
            "edges": self.sig_review_edges,
            "canvas": self.sig_review_canvas,
            "boundaries": self.sig_review_boundaries,
            "seams": self.sig_review_seams,
            "composite": self.sig_review_composite,
            "render": self.sig_review_render,
            "output": self.sig_review_output,
        }

        def _pause_cb(event: str, data: dict) -> dict:
            # Replay mode: return stored override without blocking the worker
            if not self._hitl_mode:
                override = dict(self._replay_session.get(event, {}))
                if override:
                    self._hitl_session_overrides[event] = override
                return override
            sig = _signal_map.get(event)
            if sig is not None:
                sig.emit(data)
            self._hitl_mutex.lock()
            self._hitl_paused = True
            self._hitl_override = {}
            while self._hitl_paused:
                self._hitl_wait.wait(self._hitl_mutex)
            override = dict(self._hitl_override)
            self._hitl_mutex.unlock()
            # Accumulate override for session autosave (skip empty/cancel)
            if override and not override.get("cancel"):
                self._hitl_session_overrides[event] = override
            return override

        return _pause_cb

    def _hitl_video_pause(self, data: dict) -> dict:
        """Pause StitchWorker.run() for video frame review (HITL checkpoint 0 — S84)."""
        # Replay mode: return stored video override without blocking
        if not self._hitl_mode:
            override = dict(self._replay_session.get("video", {}))
            if override:
                self._hitl_session_overrides["video"] = override
            return override
        self.sig_review_video.emit(data)
        self._hitl_mutex.lock()
        self._hitl_paused = True
        self._hitl_override = {}
        while self._hitl_paused:
            self._hitl_wait.wait(self._hitl_mutex)
        override = dict(self._hitl_override)
        self._hitl_mutex.unlock()
        if override and not override.get("cancel"):
            self._hitl_session_overrides["video"] = override
        return override

    @gc_disabled_run
    def run(self):
        cfg = self._pipeline_config

        def _progress_cb(idx: int, label: str):
            self.sig_stage.emit(idx, _TOTAL_STAGES, label)

        def _log_cb(msg: str):
            self.sig_log.emit(msg)

        # ── Video ingestion pre-run (Issue 9 / S84) ──────────────────────
        image_paths = list(self._image_paths)
        _video_tmp_dir: str | None = None

        if self._video_path:
            self.sig_log.emit(f"[Video] Extracting frames from '{self._video_path}'…")
            try:
                _video_tmp_dir = tempfile.mkdtemp(prefix="asp_video_")
                _vframes, image_paths = ingest_video(
                    self._video_path,
                    _video_tmp_dir,
                    n_frames=self._video_n_frames,
                    mode=self._video_mode,
                )
                self.sig_log.emit(f"[Video] Extracted {len(image_paths)} frames.")
            except Exception as _ve:
                self.sig_error.emit(f"[Video] Ingestion failed: {_ve}")
                return

            # HITL checkpoint 0: let user review and deselect video frames
            if self._hitl_mode and image_paths:
                _thumbs = []
                for _f in _vframes:
                    _fh, _fw = _f.shape[:2]
                    _sc = min(1.0, 256 / max(_fh, _fw, 1))
                    _thumbs.append(
                        # pyrefly: ignore [no-matching-overload]
                        cv2.resize(
                            _f,
                            (max(1, int(_fw * _sc)), max(1, int(_fh * _sc))),
                            cv2.INTER_AREA,
                        )
                    )
                _diffs = [0.0]
                for _i in range(1, len(_vframes)):
                    _a = (
                        # pyrefly: ignore [no-matching-overload]
                        cv2.resize(_vframes[_i - 1], (64, 64), cv2.INTER_AREA).astype(
                            np.float32
                        )
                        / 255.0
                    )
                    _b = (
                        # pyrefly: ignore [no-matching-overload]
                        cv2.resize(_vframes[_i], (64, 64), cv2.INTER_AREA).astype(
                            np.float32
                        )
                        / 255.0
                    )
                    _diffs.append(float(np.mean(np.abs(_a - _b))))

                _ov0 = self._hitl_video_pause(
                    {
                        "paths": list(image_paths),
                        "thumbnails": _thumbs,
                        "frame_diffs": _diffs,
                        "video_path": self._video_path,
                    }
                )
                if "frame_override" in _ov0:
                    _new_paths = _ov0["frame_override"]
                    if len(_new_paths) >= 2:
                        image_paths = _new_paths
                        self.sig_log.emit(
                            f"[HITL] Video frame selection: {len(image_paths)} frames accepted."
                        )

        try:
            pipeline = _ProgressPipeline(
                progress_cb=_progress_cb,
                log_cb=_log_cb,
                manual_affines=self._manual_affines,
                cancel_flag=self._cancel_flag,
                save_intermediate=self._save_intermediate,
                intermediate_dir=self._intermediate_dir,
                pause_cb=self._make_hitl_pause_cb(),
                pipeline_config=self._pipeline_config,
                hitl_session_overrides=self._hitl_session_overrides,
                **_build_pipeline_kwargs(cfg),
            )
            # Apply any exclusion masks set via set_exclusion_masks() before run
            if self._exclusion_masks:
                pipeline.exclusion_masks = self._exclusion_masks
            pipeline.run(image_paths, self._output_path)
            self.sig_finished.emit(self._output_path)
        except InterruptedError as e:
            self.sig_error.emit(f"Cancelled: {e}")
        except Exception as e:
            self.sig_error.emit(str(e))
        finally:
            if _video_tmp_dir:
                shutil.rmtree(_video_tmp_dir, ignore_errors=True)


__all__ = ["StitchWorker"]
