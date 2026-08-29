"""``AnimeStitchPipeline.run()`` -- the 13-stage pipeline orchestrator.

This is the one file in the §5.17 file-size epic left over 500 code lines
as a deliberate, documented exception (matching the precedent already set
in architecture.md §5.17 for stitch_tab.py and settings_window.py.__init__):
run() has no end-to-end regression test (the only automated coverage is the
ASP benchmark corpus, out of scope for verifying this issue per its own
acceptance criteria, and with a documented host-freeze history). Four
self-contained, no-early-return-coupling blocks were extracted to their own
files (_photometric_stage.py, _content_trim.py, _dedup_stage.py's single
early-return uses a sentinel-return to preserve exact control flow,
_matcher_selection.py as a mixin method since it mutates self state) --
pure code motion, no logic change.

M1a adds a ``PipelineSession`` bookkeeping object (see ``session.py``)
alongside the existing control flow. Stage marks and fallback labels are
appended next to the original log/return sites; they do not change which
function writes pixels. Full decomposition of ``run()`` into session-driven
stage functions is M1b/M1c.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from asp_backend.alignment.bundle_adjust import _bundle_adjust_affine
from asp_backend.alignment.canvas import (
    _compute_canvas,
    _crop_to_valid,
    _detect_scroll_axis,
    _load_frames,
    _normalise_widths,
    _panorama_stitch_fallback,
    _scan_stitch_fallback,
    _telea_fill_gaps,
)
from asp_backend.core.validation import (
    _compute_adaptive_min_gap,
    _compute_adaptive_rot_scale,
    _validate_affines,
)
from asp_backend.ingestion.frame_selection import detect_animation_phases
from asp_backend.ingestion.mask_uncertainty import (
    compute_temporal_mask_uncertainty,
    mask_uncertainty_enabled,
)
from asp_backend.ingestion.masking import _compute_fg_masks
from asp_backend.rendering.compositing import _composite_foreground
from asp_backend.rendering.photometric import _apply_basic, _correct_vignetting
from asp_backend.rendering.rendering import _render
from backend.src.constants import SPATIAL_DEDUP_PX
from backend.src.errors import CanvasError, PipelineError

from ._affine_recovery import _recover_affine_health
from ._content_trim import _trim_content_crop
from ._dedup_stage import _dedup_near_static_frames
from ._edge_filters import _check_edge_graph_connectivity
from ._frame_utils import (
    _apply_hires_keyframes,
    _compute_adaptive_dy_cv_max,
    _compute_dy_cv,
    _compute_row_coverage,
    _reload_scans_frames,
    _sort_frames_by_index,
    _spatial_dedup_frames,
)
from ._photometric_stage import _apply_background_photometric_normalization
from ._pipeline_protocol import _PipelineHost
from ._probes import _DY_CV_MAX, BaSiCWrapper, Image, torch
from .session import (
    PipelineSession,
    PipelineStage,
    ResultIdentity,
    snapshot_pipeline_config,
)

logger = logging.getLogger(__name__)


def _panorama_fallback_allowed(n_frames: int) -> bool:
    """Keep OpenCV PANORAMA fallback within a bounded workload."""
    try:
        max_frames = int(os.environ.get("ASP_PANORAMA_MAX_FRAMES", "12"))
    except ValueError:
        max_frames = 12
    return max_frames > 0 and n_frames <= max_frames


def _stage4_memory_snapshot() -> dict[str, float | None]:
    """Return lightweight host/CUDA evidence around foreground masking."""
    snapshot: dict[str, float | None] = {
        "rss_mb": None,
        "cuda_allocated_mb": None,
        "cuda_reserved_mb": None,
    }
    try:
        import psutil

        snapshot["rss_mb"] = round(psutil.Process().memory_info().rss / 1024**2, 1)
    except Exception:
        pass
    if torch.cuda.is_available():
        snapshot["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
        snapshot["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 1)
    return snapshot


def _refine_masks_for_plate(
    frames: list[np.ndarray],
    bg_masks: list[np.ndarray | None],
    affines: list[np.ndarray],
) -> tuple[list[np.ndarray | None], int]:
    """Apply P4 only for the P1/P2 consumer after alignment is final."""
    if not mask_uncertainty_enabled():
        return bg_masks, 0

    refined = compute_temporal_mask_uncertainty(frames, bg_masks, affines)
    if len(refined) != len(bg_masks):
        logger.warning(
            "P4 mask refinement returned %d masks for %d frames; retaining originals.",
            len(refined),
            len(bg_masks),
        )
        return bg_masks, 0

    uncertain_px = sum(
        int(np.count_nonzero(mask == 128)) for mask in refined if mask is not None
    )
    return refined, uncertain_px


if TYPE_CHECKING:
    from asp_backend.models.wrappers.aliked_lg_wrapper import ALIKEDLightGlueWrapper
    from asp_backend.models.wrappers.efficient_loftr_wrapper import EfficientLoFTRWrapper
    from asp_backend.models.wrappers.roma_wrapper import RoMaWrapper
    from backend.src.models.wrappers.basic_wrapper import BaSiCWrapper as _BaSiCWrapperT
    from backend.src.models.wrappers.birefnet_wrapper import BiRefNetWrapper
    from backend.src.models.wrappers.loftr_wrapper import LoFTRWrapper

    # Type-checking-only base: gives mypy visibility into attributes set by
    # AnimeStitchPipeline.__init__ / sibling mixins (see _pipeline_protocol.py).
    # Zero runtime effect -- at runtime this mixin still only inherits object.
    _Base = _PipelineHost
else:
    _Base = object


class _RunStageMixin(_Base):
    """Provides ``run()``, the full pipeline entry point, for ``AnimeStitchPipeline``."""

    if TYPE_CHECKING:
        # Explicit re-declarations (mirroring _PipelineHost) -- mypy cannot
        # otherwise determine these attributes' types here: each is both
        # read (in a condition) and reassigned to differing types (wrapper
        # instance vs. None) across branches below, and without a local
        # declaration mypy tries (and fails) to infer a type from those
        # in-method assignments alone rather than deferring to the
        # inherited Protocol attribute.
        _basic: _BaSiCWrapperT | None
        _birefnet: BiRefNetWrapper | None
        _loftr: LoFTRWrapper | None
        _eloftr: EfficientLoFTRWrapper | None
        _aliked: ALIKEDLightGlueWrapper | None
        _roma: RoMaWrapper | None

    def run(  # noqa: C901
        self,
        image_paths: list[str],
        output_path: str,
        hires_keyframes: dict[int, str] | None = None,
        *,
        session: PipelineSession | None = None,
        pause_hook=None,
    ) -> Image.Image:
        """
        Execute the full stitching pipeline.

        Parameters
        ----------
        image_paths : ordered list of source frame paths (first = leftmost/topmost).
        output_path : destination PNG/WEBP path.
        hires_keyframes : optional mapping of {frame_idx: hires_path} (§9C Sprint 8).
            When provided, all heavy computation runs at proxy (1080p) resolution;
            after Stage 8 (ECC/SEA-RAFT refinement), the selected frames are
            replaced by their hires counterparts and affines are scaled accordingly.
            Frame indices not listed are bicubic-upscaled from the proxy.
            The final panorama is rendered at the hires resolution.
        session : optional M1a ``PipelineSession``. When omitted, one is created.
            Bookkeeping only — does not change image operations.
        pause_hook : optional HITL callback ``(event, data) -> overrides``.
            Stored on the session; canonical ``run()`` does not insert new
            checkpoints (M1c).

        Returns
        -------
        PIL.Image of the final stitched panorama.
        """
        # Exclude the output file if it was accidentally included in the input list.
        out_abs = os.path.abspath(output_path)
        image_paths = [p for p in image_paths if os.path.abspath(p) != out_abs]

        # §1.63: Sort frame paths by numeric suffix so glob-discovered frames are
        # always in temporal order, regardless of OS directory-entry order.
        image_paths = _sort_frames_by_index(image_paths)

        if session is None:
            session = PipelineSession.create(
                image_paths,
                output_path,
                hires_keyframes,
                config=snapshot_pipeline_config(self),
                pause_hook=pause_hook or getattr(self, "pause_hook", None),
            )
        elif pause_hook is not None and session.pause_hook is None:
            session.pause_hook = pause_hook
        self.last_session = session
        session.record_artifact("output_path", output_path)
        session.init_frame_provenance(image_paths)

        logger.info(
            f"[Stitch] Starting AnimeStitchPipeline on {len(image_paths)} frames."
        )
        # OpenCV defaults to NVIDIA OpenCL on this machine; BaSiC/BiRefNet
        # then take the same GPU via CUDA. That pairing livelocks Stage 4
        # (CPU busy, GPU idle) — see ASP #49.
        with contextlib.suppress(Exception):
            cv2.ocl.setUseOpenCL(False)
        self._baselines = None

        # ── §3.16B: Per-test HITL preset ─────────────────────────────────────
        _test_name = Path(image_paths[0]).parent.name if image_paths else ""
        _hitl_pipeline_state: dict = {}

        # ── Stage 1: Load and trim ─────────────────────────────────────────────
        frames = _load_frames(image_paths)
        N = len(frames)
        if N < 2:
            session.finish(
                success=False,
                error="Need at least 2 valid frames to stitch.",
            )
            raise PipelineError("Need at least 2 valid frames to stitch.")
        session.mark(PipelineStage.LOAD, n=N)
        _h0, _w0 = frames[0].shape[:2]
        session.note_geometry(PipelineStage.LOAD, width=_w0, height=_h0, n_frames=N)
        logger.info(f"[Stitch] Stage 1 complete: {N} frames loaded.")

        phase_ids: list[int] | None = None

        # ── Stage 2: Width normalisation ─────────────────────────────────────
        frames = _normalise_widths(frames)
        H, W = frames[0].shape[:2]
        scans_frames = list(frames)
        session.mark(PipelineStage.NORMALISE, width=W, height=H)
        session.note_geometry(PipelineStage.NORMALISE, width=W, height=H, n_frames=N)
        logger.info(f"[Stitch] Stage 2 complete: all frames at {W}×{H}.")

        # ── Stage 3: BaSiC photometric correction ────────────────────────────
        if self.use_basic:
            if self._basic is None:
                self._basic = BaSiCWrapper()
            frames, baselines = _apply_basic(frames, self._basic)
            self._baselines = baselines
            frames = _correct_vignetting(frames)
            session.mark(PipelineStage.PHOTOMETRIC_BASIC)
            logger.info(
                "[Stitch] Stage 3 complete: BaSiC + Vignette correction applied."
            )
        else:
            session.mark(PipelineStage.PHOTOMETRIC_BASIC, skipped=True)
            logger.info("[Stitch] Stage 3 skipped (use_basic=False).")

        # ── Stage 4: Foreground masking ──────────────────────────────────────
        _mask_memory_before = _stage4_memory_snapshot()
        if self.use_birefnet and self._birefnet is None:
            from backend.src.models.wrappers.birefnet_wrapper import (
                BiRefNetWrapper,
            )  # §3.14 lazy

            self._birefnet = BiRefNetWrapper()
        bg_masks = _compute_fg_masks(
            frames,
            self._birefnet,
            use_birefnet=self.use_birefnet,
        )
        if self._birefnet is not None:
            with contextlib.suppress(Exception):
                self._birefnet.unload()
            self._birefnet = None
        _mask_memory_after = _stage4_memory_snapshot()
        session.record_artifact(
            "stage4_mask_memory",
            {
                "before": _mask_memory_before,
                "after": _mask_memory_after,
                "rss_delta_mb": (
                    round(_mask_memory_after["rss_mb"] - _mask_memory_before["rss_mb"], 1)
                    if _mask_memory_before["rss_mb"] is not None
                    and _mask_memory_after["rss_mb"] is not None
                    else None
                ),
            },
        )
        session.mark(
            PipelineStage.MASK,
            use_birefnet=bool(self.use_birefnet),
        )
        logger.debug(
            f"[Stitch] Stage 4 complete: foreground masks ready "
            f"({'BiRefNet' if self.use_birefnet else 'None'})."
        )
        _mask_ov = session.pause(
            "masks",
            {"image_paths": list(image_paths), "n_frames": N},
        )
        if _mask_ov.get("bg_masks") is not None and len(_mask_ov["bg_masks"]) == N:
            bg_masks = _mask_ov["bg_masks"]
        if _mask_ov.get("exclusion_masks"):
            self.exclusion_masks = _mask_ov["exclusion_masks"]

        # ── Stage 4.5/4.5b: Photometric normalisation ─────────────────────────
        _gain_telem: dict = {}
        frames = _apply_background_photometric_normalization(
            frames, bg_masks, N, telemetry=_gain_telem
        )
        session.note_gain_telemetry(_gain_telem)
        session.mark(PipelineStage.PHOTOMETRIC_BG)

        # ── Pre-stage 5: Deduplicate near-static consecutive frames ─────────
        _early, frames, scans_frames, bg_masks, image_paths, N = (
            _dedup_near_static_frames(
                frames, scans_frames, bg_masks, image_paths, N, output_path
            )
        )
        if _early is not None:
            session.mark_dropped_paths(image_paths, "near_static")
            session.record_fallback(ResultIdentity.SCANS, "dedup_too_few_frames")
            session.record_artifact("scans_path", output_path)
            session.finish(success=True, identity=ResultIdentity.SCANS)
            return _early
        session.mark_dropped_paths(image_paths, "near_static")
        session.mark(PipelineStage.DEDUP, n=N)

        # P1's one-plate model must not cross a pose change. Keep this
        # conservative source-sequence signal even if later spatial dedup
        # changes the aligned per-frame phase vector used by seam logic.
        try:
            _plate_source_phase_ids = detect_animation_phases(image_paths)
            plate_source_has_multiple_phases = len(set(_plate_source_phase_ids)) > 1
        except Exception:
            plate_source_has_multiple_phases = False

        # ── Stage 5-6: Pairwise matching (+ skip-pair edges) ────────────────
        # ── Matcher selection (P1.4 EfficientLoFTR / P3.2 JamMa) ───────────────
        _active_loftr = self._select_matcher(H, W)
        _pair_proposal_telemetry: dict = {}
        edges = self._pairwise_match_with(
            frames, bg_masks, _active_loftr, proposal_telemetry=_pair_proposal_telemetry
        )
        print(
            f"[Stitch]   Pairwise matching complete ({len(edges)} edges); "
            "starting spatial dedup...",
            flush=True,
        )

        # ── Post-match: Spatial dedup of near-static consecutive frames ──────
        # Frames whose measured adj displacement is < SPATIAL_DEDUP_PX add no
        # meaningful new content and confuse BA (effective gap ≈ 0).  Run in a
        # loop so chains (A≈B≈C) are resolved in successive passes after
        # re-indexing turns a former skip-edge into an adj-edge.

        _total_spa_dropped = 0
        _spa_changed = True
        while _spa_changed:
            frames, scans_frames, bg_masks, image_paths, edges, _n_dropped = (
                _spatial_dedup_frames(
                    frames,
                    scans_frames,
                    bg_masks,
                    image_paths,
                    edges,
                    SPATIAL_DEDUP_PX,
                )
            )
            _spa_changed = _n_dropped > 0
            if _n_dropped:
                _total_spa_dropped += _n_dropped
                logger.debug(
                    f"[Stitch]   Spatial dedup pass: {_n_dropped} frame(s) dropped, "
                    f"{len(frames)} remain."
                )
                N = len(frames)
                if N < 2:
                    session.record_fallback(ResultIdentity.SCANS, "spatial_dedup_too_few_frames")
                    session.record_artifact("scans_path", output_path)
                    session.finish(success=True, identity=ResultIdentity.SCANS)
                    _sf = scans_frames or _reload_scans_frames(image_paths)
                    return _scan_stitch_fallback(_sf, output_path)
        session.mark(PipelineStage.SPATIAL_DEDUP, dropped=_total_spa_dropped)
        session.record_artifact("frame_count", N)
        session.mark_dropped_paths(image_paths, "spatial_dedup")
        if _total_spa_dropped:
            logger.debug(
                f"[Stitch]   Spatial dedup complete: {_total_spa_dropped} frames "
                f"removed, {N} remain."
            )
        print(
            f"[Stitch]   Spatial dedup complete ({N} frames); detecting phases...",
            flush=True,
        )

        # ── §2.2/2.3 animation-phase clustering ──────────────────────────────
        # Measurement-only unless ASP_PHASE_COMPOSITE=1 (compositing.py reads
        # that flag itself). Computed here, after both dedup passes above, so
        # phase_ids indices stay aligned with the final image_paths/frames/
        # affines Stage 11 actually uses — either dedup pass can drop frames
        # by index, which would desync a phase_ids list computed earlier.
        try:
            phase_ids = detect_animation_phases(image_paths)
            session.mark(PipelineStage.PHASE, n_phases=len(set(phase_ids)))
            logger.info(
                f"[Stitch] {len(set(phase_ids))} animation phase(s) "
                f"detected across {N} frames."
            )
        except Exception as _phase_exc:
            logger.warning(
                f"[Stitch] Phase detection failed ({_phase_exc}); "
                "phase-consistent compositing disabled for this run."
            )
            phase_ids = None
            session.mark(PipelineStage.PHASE, skipped=True, error=str(_phase_exc))

        print("[Stitch]   Phase detection complete; filtering edges...", flush=True)
        raw_registration_edges = list(edges)
        edges = self._filter_edges(edges, image_paths, H, W, frames, bg_masks)
        from asp_backend.alignment.registration_telemetry import (
            collect_registration_telemetry,
            edge_graph_components,
        )

        _proposed_adjacent = [
            candidate
            for candidate in _pair_proposal_telemetry.get("candidates", [])
            if candidate.get("span") == 1
        ]
        _raw_adjacent = [edge for edge in raw_registration_edges if edge["j"] == edge["i"] + 1]
        _filtered_adjacent = [edge for edge in edges if edge["j"] == edge["i"] + 1]
        _pair_proposal_telemetry.update(
            {
                "adjacent_pair_survival": {
                    "proposed": len(_proposed_adjacent),
                    "matched_pre_filter": len(_raw_adjacent),
                    "survived_filter": len(_filtered_adjacent),
                },
                "components": {
                    "pre_filter": edge_graph_components(raw_registration_edges, N),
                    "post_filter": edge_graph_components(edges, N),
                },
            }
        )
        if os.environ.get("ASP_CLEANCP_RESOLVE", "0") == "1" and (
            not edges or len(edge_graph_components(edges, N)) != 1
        ):
            from ._cleancp_recovery import recover_clean_correspondence_edges

            edges, cleancp_telemetry = recover_clean_correspondence_edges(
                raw_registration_edges, edges, N
            )
            _pair_proposal_telemetry["cleancp_recovery"] = cleancp_telemetry
            if cleancp_telemetry["accepted"]:
                logger.info(
                    "[Stitch] CleanCP local re-solve accepted %d robust edges.", len(edges)
                )
            else:
                logger.info(
                    "[Stitch] CleanCP local re-solve stopped: %s.",
                    cleancp_telemetry.get("stopped_reason"),
                )
        session.mark(PipelineStage.FILTER_EDGES, n_edges=len(edges))
        session.record_artifact("n_edges", len(edges))
        session.record_artifact("pair_proposal_telemetry", _pair_proposal_telemetry)

        # Persist pair evidence even if connectivity rejects the graph before
        # bundle adjustment; the M2 probe needs to distinguish that failure
        # from a clean graph with a bad rendered result.
        session.record_artifact(
            "registration_telemetry",
            collect_registration_telemetry(
                raw_registration_edges, edges, [], pair_proposal=_pair_proposal_telemetry
            ),
        )

        # §3.16B: apply HITL drop_edges after filter
        if _hitl_pipeline_state.get("boundaries"):
            logger.info(
                f"[Stitch] §3.16B: HITL preset '{_test_name}' — "
                f"forced_boundaries={_hitl_pipeline_state['boundaries']}."
            )

        print(
            f"[Stitch]   Edge filtering complete ({len(edges)} edges); unloading matchers...",
            flush=True,
        )
        for _mdl in [self._loftr, self._eloftr, self._aliked, self._roma]:
            if _mdl is not None:
                try:
                    _mdl.unload()
                except Exception:
                    with contextlib.suppress(Exception):
                        _mdl.offload()
        self._loftr = None
        self._eloftr = None
        self._aliked = None
        self._roma = None
        if (
            os.environ.get("ITK_MODEL_FLUSH_CUDA_ON_UNLOAD", "0") == "1"
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()
        if os.environ.get("ITK_MODEL_FORCE_GC_ON_UNLOAD", "0") == "1":
            gc.collect()
        print("[Stitch]   Matcher unload complete.", flush=True)
        session.mark(PipelineStage.MATCH, n_edges=len(edges))
        logger.info(f"[Stitch] Stages 5-6 complete: {len(edges)} valid edges found.")
        if not edges:
            warnings.warn("[Stitch] No valid edges — falling back to scan stitch.", stacklevel=2)
            session.record_fallback(ResultIdentity.SCANS, "no_valid_edges")
            session.record_artifact("scans_path", output_path)
            session.finish(success=True, identity=ResultIdentity.SCANS)
            _sf = scans_frames or _reload_scans_frames(image_paths)
            return _scan_stitch_fallback(_sf, output_path)

        # ── §1.15: Edge graph connectivity gate ───────────────────────────────
        # A disconnected edge graph would make bundle adjustment assign wrong
        # translations to isolated frames. This used to hard-bail to SCANS
        # right here -- but Stage 7b's ``_recover_affine_health`` (adjacent-
        # only bundle, then sequential gap-fill that reconstructs isolated
        # frames from the pan step) exists precisely for that case, and
        # ``_validate_affines`` is the backstop if recovery can't produce a
        # trustworthy solve. Rejecting a whole pan because one pair went
        # unmatched is what dropped the full-corpus RAW_ASP rate from ~51/97
        # to ~8/97 once M1b routed the benchmark through this path (the
        # legacy inline harness never had this gate). So: record the
        # disconnection and fall through to recovery, unless
        # ``ASP_STRICT_EDGE_GRAPH_GATE=1`` restores the hard bail.
        print("[Stitch]   Checking edge-graph connectivity...", flush=True)
        _edge_graph_connected = _check_edge_graph_connectivity(edges, N)
        session.record_artifact("edge_graph_connected", bool(_edge_graph_connected))
        if not _edge_graph_connected:
            _n_components = len(edge_graph_components(edges, N))
            logger.info(
                "[Stitch] §1.15: Edge graph is disconnected (%d edges, %d frames, "
                "%d components).",
                len(edges),
                N,
                _n_components,
            )
            if os.environ.get("ASP_STRICT_EDGE_GRAPH_GATE", "0") == "1":
                session.record_fallback(ResultIdentity.SCANS, "disconnected_edge_graph")
                session.record_artifact("scans_path", output_path)
                session.finish(success=True, identity=ResultIdentity.SCANS)
                _sf = scans_frames or _reload_scans_frames(image_paths)
                return _scan_stitch_fallback(_sf, output_path)
            logger.info(
                "[Stitch] §1.15: proceeding to bundle adjust + affine recovery "
                "(gap-fill reconstructs isolated frames; set "
                "ASP_STRICT_EDGE_GRAPH_GATE=1 to hard-bail instead)."
            )

        # ── Stage 7: Global bundle adjustment ────────────────────────────────
        print("[Stitch]   Starting bundle adjustment...", flush=True)
        _motion_model = getattr(self, "motion_model", "affine")
        use_affine_ba = _motion_model == "affine"
        affines = _bundle_adjust_affine(
            edges,
            N,
            use_affine=use_affine_ba,
            motion_model=_motion_model,
        )
        session.record_artifact(
            "registration_telemetry",
            collect_registration_telemetry(
                raw_registration_edges, edges, affines, pair_proposal=_pair_proposal_telemetry
            ),
        )
        print("[Stitch]   Bundle adjustment complete.", flush=True)
        session.mark(PipelineStage.BUNDLE_ADJUST, motion_model=_motion_model)
        logger.debug(
            f"[Stitch] Stage 7 complete: bundle adjustment done "
            f"(mode={_motion_model})."
        )

        # ── Stage 7.2: Wave correction (chain-drift straightening, default off) ──
        # Experimental candidate from the 2026-08-23 roadmap critique round:
        # the 2D translation-domain analog of OpenCV detail::waveCorrect.
        # Only rewrites translation slots of the BA output -- never touches the
        # 2x2 part and never runs the validity gate, so it cannot collide with
        # the affine_invalid/min_gap rejection path.
        if os.environ.get("ASP_WAVE_CORRECT", "0") == "1":
            from ._wave_correction import wave_correct_affines

            _wc_kind = os.environ.get("ASP_WAVE_CORRECT_KIND", "auto")
            _wc_before = np.array(
                [[float(a[0, 2]), float(a[1, 2])] for a in affines], dtype=np.float64
            )
            affines = wave_correct_affines(affines, kind=_wc_kind)
            _wc_after = np.array(
                [[float(a[0, 2]), float(a[1, 2])] for a in affines], dtype=np.float64
            )
            _wc_delta = float(np.max(np.abs(_wc_before - _wc_after))) if len(_wc_before) else 0.0
            print(
                f"[Stitch]   Wave correction applied (kind={_wc_kind}, "
                f"max_|delta|={_wc_delta:.2f}px)",
                flush=True,
            )
            session.mark(PipelineStage.BUNDLE_ADJUST, wave_corrected=True)

        # ── Stage 7b: Affine validation gate ─────────────────────────────────
        # §0.5C: adaptive min_gap — scales with canvas span so fast-scroll
        # (4K, >400 px/frame) applies a proportionally higher floor than the
        # fixed 25 px default, while slow-scroll sequences use 20 px.
        _adaptive_min_gap = _compute_adaptive_min_gap(affines)
        _adaptive_rot, _adaptive_sc = _compute_adaptive_rot_scale(affines)
        health = _validate_affines(
            affines,
            min_step=_adaptive_min_gap,
            max_rotation=_adaptive_rot,
            max_scale_dev=_adaptive_sc,
        )
        logger.debug(
            f"[Stitch]   Affine health: valid={health.valid}, "
            f"ratio={health.ratio:.1f}×, min_gap={health.min_gap:.0f}px "
            f"(adaptive_floor={_adaptive_min_gap:.1f}px), "
            f"max_rot={health.max_rotation:.4f} (thresh={_adaptive_rot:.2f}), "
            f"scale_dev={health.max_scale_dev:.4f} (thresh={_adaptive_sc:.2f})"
        )
        _pose_source = "bundle_adjust"
        _deferred_min_gap = False
        if not health.valid:
            affines, health = _recover_affine_health(
                edges,
                N,
                affines,
                health,
                use_affine_ba,
                _adaptive_min_gap,
                _adaptive_rot,
                _adaptive_sc,
                logger,
                _motion_model,
            )
            if health.valid:
                _pose_source = "affine_recovery"
            if not health.valid:
                _affine_health = {
                    "valid": bool(health.valid),
                    "reason": health.reason,
                    "ratio": float(health.ratio),
                    "min_gap": float(health.min_gap),
                    "max_rotation": float(health.max_rotation),
                    "max_scale_dev": float(health.max_scale_dev),
                }
                session.record_artifact("affine_health", _affine_health)
                # M2 experiment: defer only the adaptive min-gap heuristic to
                # the frozen registration-risk rule.  The flag is default-off;
                # ratio, rotation, scale, and monotonicity failures still take
                # the existing safe fallback path unchanged.
                if (
                    os.environ.get("ASP_DEFER_MIN_GAP_TO_REGISTRATION_GATE", "0") == "1"
                    and health.reason.startswith("min_gap=")
                ):
                    from .registration_gate import RegistrationRiskGate

                    _deferred_decision = RegistrationRiskGate().evaluate(
                        session.artifacts.get("registration_telemetry"),
                        affine_health=_affine_health,
                    )
                    session.record_artifact(
                        "deferred_min_gap_registration_decision",
                        {
                            "accept": _deferred_decision.accept,
                            "reason": _deferred_decision.reason,
                            "status": _deferred_decision.status,
                            "scores": _deferred_decision.scores,
                        },
                    )
                    if _deferred_decision.accept:
                        _deferred_min_gap = True
                        logger.warning(
                            "[Stitch] Deferring affine min-gap failure to registration "
                            "review (%s).",
                            _deferred_decision.reason,
                        )
                        health = health._replace(
                            valid=True,
                            reason="deferred_min_gap_to_registration_review",
                        )
                if not health.valid:
                    # §1.3B: PANORAMA stitcher handles scale/rotation that
                    # translation-only validation rejects; try before SCANS.
                    _disable_panorama = os.environ.get(
                        "ASP_DISABLE_PANORAMA_FALLBACK", "0"
                    ) == "1"
                    if _panorama_fallback_allowed(N) and not _disable_panorama:
                        try:
                            print(
                                f"[Stitch]   Affine recovery failed; trying PANORAMA "
                                f"fallback for {N} frames...",
                                flush=True,
                            )
                            _sf = scans_frames or _reload_scans_frames(image_paths)
                            _pano = _panorama_stitch_fallback(_sf, output_path)
                            # PANORAMA is a safe-policy fallback algorithm, not a
                            # fourth output identity. Raw/Safe/SCANS stay disjoint for
                            # the M0 result schema and future parity comparisons.
                            session.record_fallback(
                                ResultIdentity.SAFE_ASP,
                                str(health.reason),
                                algorithm="panorama",
                            )
                            session.record_artifact("safe_asp_path", output_path)
                            session.record_artifact("fallback_algorithm", "panorama")
                            session.finish(success=True, identity=ResultIdentity.SAFE_ASP)
                            return _pano
                        except Exception as _pano_e:
                            logger.info(
                                f"[Stitch]   PANORAMA fallback failed ({_pano_e}); using SCANS."
                            )
                    else:
                        logger.info(
                            "[Stitch]   Skipping PANORAMA fallback for %d frames "
                            "(disabled=%s, ASP_PANORAMA_MAX_FRAMES=%s); using SCANS.",
                            N,
                            _disable_panorama,
                            os.environ.get("ASP_PANORAMA_MAX_FRAMES", "12"),
                        )
                        print(
                            f"[Stitch]   Affine recovery failed; {N} frames exceed the "
                            "PANORAMA safety limit, using SCANS.",
                            flush=True,
                        )
                    warnings.warn(
                        f"[Stitch] Affine validation FAILED ({health.reason}) after retries. "
                        f"Falling back to SCANS stitch.", stacklevel=2
                    )
                    session.record_fallback(ResultIdentity.SCANS, f"affine_invalid:{health.reason}")
                    session.record_artifact("scans_path", output_path)
                    session.finish(success=True, identity=ResultIdentity.SCANS)
                    _sf = scans_frames or _reload_scans_frames(image_paths)
                    return _scan_stitch_fallback(_sf, output_path)
        if not session.artifacts.get("affine_health"):
            session.record_artifact(
                "affine_health",
                {
                    "valid": bool(health.valid),
                    "reason": health.reason,
                    "ratio": float(health.ratio),
                    "min_gap": float(health.min_gap),
                    "max_rotation": float(health.max_rotation),
                    "max_scale_dev": float(health.max_scale_dev),
                    "deferred_min_gap": _deferred_min_gap,
                },
            )
        session.mark(
            PipelineStage.AFFINE_VALIDATE,
            valid=bool(health.valid),
            reason=getattr(health, "reason", None),
        )

        # ── Stage 8: Sub-pixel refinement ────────────────────────────────────
        affines, _refine_src = self._refine_subpixel(frames, affines, bg_masks)
        session.mark(PipelineStage.REFINE, method=_refine_src)

        # ── Stage 8.8: Hires keyframe substitution (§9C — Sprint 8) ────────
        # All heavy computation above ran on proxy (1080p) frames. If the caller
        # provided hires_keyframes, swap in the full-resolution images now and
        # scale the locked affines so Stage 9 (canvas) operates at hires resolution.
        if hires_keyframes:
            _n_hires, frames, affines, bg_masks = _apply_hires_keyframes(
                frames, affines, bg_masks, hires_keyframes
            )
            session.mark(PipelineStage.HIRES, n_hires=_n_hires)
            if _n_hires > 0:
                logger.info(
                    f"[Stitch] Stage 8.8: substituted {_n_hires} hires frame(s); "
                    f"canvas will render at {frames[0].shape[1]}×{frames[0].shape[0]} px."
                )
            else:
                logger.warning(
                    "[Stitch] Stage 8.8: hires_keyframes provided but no valid paths "
                    "could be loaded — continuing at proxy resolution."
                )
        else:
            session.mark(PipelineStage.HIRES, skipped=True)

        # ── Stage 9: Canvas construction ────────────────────────────────────
        canvas_h, canvas_w, T_global = _compute_canvas(frames, affines)
        logger.info(f"[Stitch] Stage 9: canvas size {canvas_w}×{canvas_h}.")
        if canvas_h <= 0 or canvas_w <= 0:
            raise CanvasError("Computed canvas has zero size.")

        for i in range(N):
            affines[i][0, 2] += T_global[0]
            affines[i][1, 2] += T_global[1]

        # P1.9 — Bidirectional midplane projection (StabStitch++).
        # Centres the affine coordinate system on the temporal midplane rather
        # than anchoring everything to frame 0.  For long pans (e.g. 14 frames,
        # 150px/step) this halves the maximum per-frame distortion distance,
        # reducing warp artefacts symmetrically across the sequence.
        T_mid_x = float(np.mean([a[0, 2] for a in affines]))
        T_mid_y = float(np.mean([a[1, 2] for a in affines]))
        for i in range(N):
            affines[i][0, 2] -= T_mid_x
            affines[i][1, 2] -= T_mid_y
        # Recompute canvas after midplane shift so T_global absorbs the offset.
        canvas_h, canvas_w, T_global2 = _compute_canvas(frames, affines)
        for i in range(N):
            affines[i][0, 2] += T_global2[0]
            affines[i][1, 2] += T_global2[1]
        session.mark(PipelineStage.CANVAS, width=canvas_w, height=canvas_h)
        session.note_geometry(
            PipelineStage.CANVAS, width=canvas_w, height=canvas_h, n_frames=N
        )
        session.record_artifact("canvas_size", [canvas_w, canvas_h])
        session.record_artifact("affines", [a.tolist() for a in affines])
        session.note_pose_provenance(
            [
                {
                    "frame": i,
                    "tx": round(float(affines[i][0, 2]), 3),
                    "ty": round(float(affines[i][1, 2]), 3),
                    "motion_model": str(_motion_model),
                    "source": _pose_source,
                    "refined_by": _refine_src,
                    "valid": bool(health.valid),
                }
                for i in range(N)
            ]
        )
        logger.debug(
            f"[Stitch] Stage 9 complete: midplane shift ({T_mid_x:.1f}, {T_mid_y:.1f}), "
            f"canvas {canvas_w}×{canvas_h}."
        )

        # §3.14 — Scroll axis classification (logged; horizontal → SCANS fallback).
        # Compositing assumes vertical strips; horizontal scroll produces garbled output
        # without a full horizontal-strip compositing mode (not yet implemented).
        scroll_axis = _detect_scroll_axis(affines)
        logger.info(f"[Stitch] Stage 9.5: scroll axis = '{scroll_axis}'.")
        if scroll_axis == "horizontal":
            logger.info(
                "[Stitch] Horizontal scroll (tx_range >> ty_range) — vertical-strip "
                "compositing not applicable; falling back to SCANS."
            )
            session.record_fallback(ResultIdentity.SCANS, "horizontal_scroll")
            session.record_artifact("scans_path", output_path)
            session.finish(success=True, identity=ResultIdentity.SCANS)
            return _scan_stitch_fallback(scans_frames, output_path)

        # ── §4.7: dy_cv pre-detection gate ───────────────────────────────────
        # When step-size CV is high the scroll is too irregular for ARAP/seam
        # compositing — SCANS trivially handles these sequences.
        if _DY_CV_MAX > 0.0:
            _dy_cv_gate = _compute_dy_cv(affines)
            _dy_cv_adaptive_max = _compute_adaptive_dy_cv_max(N, _DY_CV_MAX)
            if _dy_cv_gate >= _dy_cv_adaptive_max:
                logger.info(
                    "[Stitch] §4.7/§5.8: dy_cv=%.3f ≥ %.2f (irregular scroll, N=%d) "
                    "→ SCANS fallback (ASP seam routing degrades severely at high dy_cv).",
                    _dy_cv_gate,
                    _dy_cv_adaptive_max,
                    N,
                )
                session.record_fallback(ResultIdentity.SCANS, "dy_cv_gate")
                session.record_artifact("scans_path", output_path)
                session.finish(success=True, identity=ResultIdentity.SCANS)
                _sf = scans_frames or _reload_scans_frames(image_paths)
                return _scan_stitch_fallback(_sf, output_path)

        # P1.3 — Compute per-frame matching confidence for weighted median (W3).
        # Each frame's confidence = the maximum edge weight of its adjacent edges.
        # LoFTR edges have weight ~0.9; TM/PC fallbacks have 0.15–0.55.
        # Frame 0 is always the anchor (confidence 1.0 by convention).
        _frame_confs = np.ones(N, dtype=np.float32)
        for _e in edges:
            _fi, _fj, _w = _e["i"], _e["j"], float(_e.get("weight", 1.0))
            if _e["j"] == _e["i"] + 1:  # only adjacent edges for per-frame confidence
                _frame_confs[_fi] = max(_frame_confs[_fi], _w)
                _frame_confs[_fj] = max(_frame_confs[_fj], _w)
        _frame_confs = np.clip(_frame_confs, 0.0, 1.0)

        # ── Stage 9.5: Alignment stability gate ─────────────────────────────
        # Log severe 2D motion but only abort at a very high threshold — the
        # render gate (in the calling benchmark) uses a SCANS-relative comparison
        # and catches genuinely degraded composites regardless of motion pattern.
        # Hard-abort threshold raised to 200px (was 50px); scenes with horizontal
        # drift up to ~2 frame-widths can still produce acceptable composites.
        # Override: ASP_ALIGN_GATE_DX env var (default 200; set to 50 to restore
        # the old strict behaviour; set to 9999 to disable entirely).
        try:
            _align_dx_limit = float(os.environ.get("ASP_ALIGN_GATE_DX", "200"))
        except ValueError:
            _align_dx_limit = 200.0
        _txs_gate = [float(affines[i][0, 2]) for i in range(N)]
        _dx_gate = [abs(_txs_gate[i + 1] - _txs_gate[i]) for i in range(N - 1)]
        if _dx_gate:
            _dx_p75 = float(np.percentile(_dx_gate, 75))
            if _dx_p75 > _align_dx_limit:
                logger.info(
                    f"[Stitch] Alignment stability gate: 75th-pct |dx|={_dx_p75:.1f}px "
                    f"> {_align_dx_limit:.0f}px limit — extreme 2D motion, "
                    f"falling back to SCANS."
                )
                session.record_fallback(ResultIdentity.SCANS, "align_dx_gate")
                session.record_artifact("scans_path", output_path)
                session.finish(success=True, identity=ResultIdentity.SCANS)
                return _scan_stitch_fallback(scans_frames, output_path)
        session.mark(PipelineStage.ALIGN_GATES, scroll_axis=scroll_axis)

        # ── Stage 10: Temporal renderer ─────────────────────────────────────
        # P1.2 — Variable-step renderer switch (W2 fix for test16).
        # When step-size variance is high (dy_cv > 0.20), the temporal median
        # blurs in proportion to overlap inconsistency across frames.  Switching
        # to 'first' (first-frame-wins per canvas pixel) avoids cross-frame
        # averaging at boundary zones and matches what SCANS naturally produces.
        effective_renderer = self.renderer
        if self.renderer == "median" and N >= 3:
            _dy_steps = [
                abs(float(affines[k][1, 2]) - float(affines[k - 1][1, 2]))
                for k in range(1, N)
            ]
            _mean_dy = float(np.mean(_dy_steps)) if _dy_steps else 1.0
            _dy_cv = float(np.std(_dy_steps)) / max(_mean_dy, 1.0) if _dy_steps else 0.0
            if _dy_cv > 0.20:
                effective_renderer = "first"
                logger.debug(
                    f"[Stitch]   High step variance (dy_cv={_dy_cv:.3f} > 0.20) — "
                    f"switching renderer to 'first'."
                )

        canvas, valid_mask, warped_corr, warped_fgs = _render(
            frames,
            affines,
            bg_masks,
            canvas_h,
            canvas_w,
            renderer=effective_renderer,
            baselines=self._baselines,
            confidence_weights=_frame_confs,
        )
        session.mark(PipelineStage.RENDER, renderer=effective_renderer)
        logger.info("[Stitch] Stage 10 complete: temporal render done.")

        # P4 is deliberately late: temporal rendering retains its established
        # binary-mask behavior, while the P1/P2 plate receives uncertain pixels.
        bg_masks, _p4_uncertain_px = _refine_masks_for_plate(frames, bg_masks, affines)
        if mask_uncertainty_enabled():
            session.record_artifact(
                "mask_uncertainty",
                {"enabled": True, "uncertain_pixels": _p4_uncertain_px},
            )
            logger.info(
                "[Stitch] P4 mask uncertainty refinement complete (%d uncertain pixels).",
                _p4_uncertain_px,
            )

        # ── Stage 10.5: Multi-frame canvas coverage gate (§0 item 2) ─────────
        # For each canvas row count how many frames contribute content.
        # If < ASP_COV_MIN_MULTI_PCT (default 30%) of content rows have ≥2-frame
        # coverage, the temporal median is effectively "first-frame-wins" across
        # the entire canvas — it cannot suppress animation ghosting.  Composite
        # on such a canvas would amplify ghosting rather than remove it.
        # Conservative default (30%) avoids false positives while catching truly
        # degenerate selections (e.g., 2 widely-spaced frames in a tall canvas).
        _row_cov, _pct_cov_multi, _cov_median = _compute_row_coverage(
            affines, frames, canvas_h
        )
        _n_cov_total = int((_row_cov > 0).sum())
        _n_cov_multi = (
            int((_row_cov[_row_cov > 0] >= 2).sum()) if _n_cov_total > 0 else 0
        )
        logger.info(
            f"[Stitch] Stage 10.5: coverage — "
            f"{_n_cov_multi}/{_n_cov_total} rows ({_pct_cov_multi:.0%}) "
            f"have ≥2-frame coverage; median={_cov_median:.1f}"
        )
        if _n_cov_total > 0:
            try:
                _cov_min_pct = float(os.environ.get("ASP_COV_MIN_MULTI_PCT", "0.30"))
            except ValueError:
                _cov_min_pct = 0.30
            if _pct_cov_multi < _cov_min_pct:
                logger.info(
                    f"[Stitch] Stage 10.5: coverage gate — {_pct_cov_multi:.0%} < "
                    f"{_cov_min_pct:.0%} threshold, temporal median insufficient "
                    f"for deghosting → SCANS fallback."
                )
                session.record_fallback(ResultIdentity.SCANS, "coverage_gate")
                session.record_artifact("scans_path", output_path)
                session.finish(success=True, identity=ResultIdentity.SCANS)
                return _scan_stitch_fallback(scans_frames, output_path)
        session.mark(
            PipelineStage.COVERAGE_GATE,
            pct_multi=_pct_cov_multi,
            median=_cov_median,
        )

        # ── Stage 11: Foreground composite ──────────────────────────────────
        if self.composite_fg and self.use_birefnet:
            _seam_meta: dict = {}
            canvas = _composite_foreground(
                [],
                [],
                canvas,
                canvas_h,
                canvas_w,
                frames,
                affines,
                bg_masks,
                frame_keys=tuple(image_paths),
                seam_path_cache=self._seam_path_cache,
                exclusion_masks=self.exclusion_masks or None,
                phase_ids=phase_ids,
                source_has_multiple_phases=plate_source_has_multiple_phases,
                seam_meta_out=_seam_meta,
            )
            _single = _seam_meta.get("seam_single_pose") or {}
            _bounds = _seam_meta.get("boundaries") or []
            _plate_sp = bool(_seam_meta.get("plate_single_pose"))
            _coherence = bool(_seam_meta.get("coherence_v2"))
            session.note_seam_feasibility(
                {
                    "attempted": True,
                    "feasible": bool(_bounds or _plate_sp or _coherence),
                    "n_boundaries": len(_bounds),
                    "n_single_pose": sum(1 for v in _single.values() if v),
                    "max_seam_lum_step": _seam_meta.get("max_seam_lum_step"),
                    "exclusion_masks": (
                        0 if not self.exclusion_masks else len(self.exclusion_masks)
                    ),
                    # M3 (#34): coherence_v2 ownership decisions, present only
                    # when ASP_COHERENCE_V2=1 took the single-pose path.
                    "coherence_ownership": _seam_meta.get("coherence_ownership"),
                    # P1/P2/P3: plate_single_pose ownership decisions and claimed pixels
                    "plate_single_pose": _plate_sp,
                    "plate_ownership": _seam_meta.get("plate_ownership"),
                }
            )
            session.mark(PipelineStage.COMPOSITE)
            logger.info("[Stitch] Stage 11 complete: foreground composited.")
        else:
            session.note_seam_feasibility({"attempted": False, "feasible": None})
            session.mark(PipelineStage.COMPOSITE, skipped=True)

        # ── Stage 12: Remaining seam blend (handled inside _render). ────────

        # ── Stage 12.5: Scroll-axis-aware content crop (§2.6) ───────────────
        canvas, valid_mask = _trim_content_crop(
            canvas, valid_mask, affines, bg_masks, N, canvas_h, canvas_w
        )
        session.mark(PipelineStage.CONTENT_TRIM)

        # ── Stage 13: Morphological boundary crop ───────────────────────────
        canvas = _crop_to_valid(canvas, valid_mask)
        if getattr(self, "edge_crop", 0) > 0:
            ec = self.edge_crop
            if ec * 2 < canvas.shape[0] and ec * 2 < canvas.shape[1]:
                canvas = canvas[ec:-ec, ec:-ec]
        session.mark(PipelineStage.CROP)
        session.note_geometry(
            PipelineStage.CROP,
            width=int(canvas.shape[1]),
            height=int(canvas.shape[0]),
        )
        logger.info("[Stitch] Stage 13 complete: boundary crop done.")

        # P1.8 — Auto-trigger diffusion inpainting for coverage gaps (W4 fix).
        # test7 (diagonal motion) leaves black corners at 81.5% coverage.
        # After the crop, recalculate the valid-pixel ratio and call the existing
        # inpaint_gaps module when coverage drops below 95%.
        _gap_mask = (canvas.max(axis=2) == 0).astype(np.uint8) * 255
        _coverage = 1.0 - float(_gap_mask.mean()) / 255.0
        if _coverage < 0.95 and _gap_mask.any():
            logger.debug(
                f"[Stitch]   Coverage {_coverage * 100:.1f}% < 95%; "
                f"auto-activating border fill for black corners."
            )
            try:
                canvas = _telea_fill_gaps(canvas, _gap_mask)
                session.mark(PipelineStage.INPAINT)
                logger.info("[Stitch]   TELEA border fill complete.")
            except Exception as _telea_e:
                session.mark(PipelineStage.INPAINT, skipped=True, error=str(_telea_e))
                logger.info(
                    f"[Stitch]   TELEA border fill failed ({_telea_e}); keeping canvas as-is."
                )
        else:
            session.mark(PipelineStage.INPAINT, skipped=True)

        # ── Save ─────────────────────────────────────────────────────────────
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        out = Image.fromarray(rgb)
        out.save(output_path)
        gc.collect()
        session.mark(PipelineStage.SAVE)
        session.note_geometry(
            PipelineStage.SAVE,
            width=int(canvas.shape[1]),
            height=int(canvas.shape[0]),
        )
        session.record_artifact("raw_asp_path", output_path)
        session.finish(success=True, identity=ResultIdentity.RAW_ASP)
        logger.info(f"[Stitch] Done. Saved to '{output_path}'.")

        return out

    def _pairwise_match_with(self, frames, bg_masks, active_loftr, proposal_telemetry=None):
        """Stage 5-6's actual pairwise-match call, using the matcher chosen by
        ``_select_matcher``. Split out of ``run()`` only to keep that one call
        readable; not a meaningful behavioural boundary on its own."""
        from asp_backend.alignment.matching import _pairwise_match

        # P2 connectivity (Hugin CalculateOverlap/ImageGraph analog, default-off):
        # provisional phase-correlation anchors -> overlap/component bridge pairs.
        extra_proposals = None
        if os.environ.get("ASP_OVERLAP_PROPOSAL", "0") == "1":
            from asp_backend.alignment.matching._overlap_proposal import (
                propose_overlap_bridge_pairs,
            )

            extra_proposals, ov_telemetry = propose_overlap_bridge_pairs(
                frames, bg_masks
            )
            if proposal_telemetry is not None:
                proposal_telemetry["overlap_proposal"] = ov_telemetry
            if ov_telemetry.get("stopped_reason"):
                logger.warning(
                    "[Stitch]   Overlap proposal stopped: %s "
                    "(anchors=%d, bridges=%d)",
                    ov_telemetry["stopped_reason"],
                    ov_telemetry["provisional_anchors"],
                    ov_telemetry["overlap_bridge_added"]
                    + ov_telemetry["component_bridge_added"],
                )

        return _pairwise_match(
            frames,
            bg_masks,
            loftr_wrapper=active_loftr,
            use_loftr=active_loftr is not None,
            motion_model=self.motion_model,
            aliked_wrapper=self._aliked if self.use_aliked else None,
            roma_wrapper=self._roma if self.use_roma else None,
            bg_masked_matching=getattr(self, "bg_masked_matching", False),
            proposal_telemetry=proposal_telemetry,
            extra_proposals=extra_proposals,
        )

    def _refine_subpixel(self, frames, affines, bg_masks):
        """Stage 8: SEA-RAFT flow refinement (preferred) or ECC fallback.

        ECC fails on flat anime cells (near-zero gradients → singular
        Hessian); SEA-RAFT uses learned cost volumes that remain informative
        over uniform colour regions.

        Returns ``(affines, source)`` where source is ``sea_raft``,
        ``ecc``, ``sea_raft_ecc_fallback``, or ``none``.
        """
        from asp_backend.alignment.ecc import _ecc_refine

        from ._probes import _flow_refine, _load_sea_raft

        refine_src = "none"
        if self.use_sea_raft:
            try:
                if self._sea_raft is None:
                    _dev = "cuda" if torch.cuda.is_available() else "cpu"
                    self._sea_raft = _load_sea_raft(device=_dev)
                    logger.info("[Stitch]   SEA-RAFT model loaded.")
                affines = _flow_refine(
                    frames,
                    affines,
                    bg_masks,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    raft_model=self._sea_raft,
                )
                refine_src = "sea_raft"
                logger.info("[Stitch] Stage 8 complete: SEA-RAFT flow refinement done.")
                # Offload SEA-RAFT after use
                if torch.cuda.is_available():
                    with contextlib.suppress(Exception):
                        self._sea_raft.cpu()
                    torch.cuda.empty_cache()
                    self._sea_raft = None
            except Exception as _ecc_e:
                logger.info(
                    f"[Stitch]   SEA-RAFT failed ({_ecc_e}); falling back to ECC."
                )
                if self.use_ecc:
                    affines = _ecc_refine(frames, affines, bg_masks)
                    refine_src = "sea_raft_ecc_fallback"
                    logger.info(
                        "[Stitch] Stage 8 complete: ECC refinement done (fallback)."
                    )
        elif self.use_ecc:
            affines = _ecc_refine(frames, affines, bg_masks)
            refine_src = "ecc"
            logger.info("[Stitch] Stage 8 complete: ECC refinement done.")
        else:
            logger.info("[Stitch] Stage 8 skipped (use_ecc=False, use_sea_raft=False).")
        return affines, refine_src


__all__ = ["_RunStageMixin"]
