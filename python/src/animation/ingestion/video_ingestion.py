"""
backend/src/animation/video_ingestion.py
=====================================
Issue 9 — PyAV-based proxy-first video frame decoder for AnimeStitchPipeline.

``VideoIngestionStream`` decodes a video file and extracts a sparse set of
frames suitable for stitching:

1. **Proxy pass** — decode a low-resolution I-frame index at ¼ resolution
   using ``pyav``'s fast seek.  Used for quick temporal browsing and telecine
   drop detection.

2. **Frame selection** — choose a sparse subset via one of:
   - ``"uniform"``   : evenly-spaced N frames from the proxy index
   - ``"keyframe"``  : only use I-frames (GOP boundaries)
   - ``"smart"``     : defer to :func:`smart_select_frames` in frame_selection.py

3. **Full-resolution decode** — for the selected proxy frame indices only,
   decode the full-resolution frame (avoids buffering the entire video).

4. **Telecine decimation** — when ``telecine=True``, consecutive frames whose
   proxy MAD is below ``telecine_mad_thresh`` (default 2.0) are deduplicated
   (keeps first of each run).  Handles 3:2 pull-down artefacts common in
   anime releases.

Requirements
------------
``pip install av``  (pyav — wraps libavcodec/libavformat).  Graceful ImportError
fallback: :func:`VideoIngestionStream.ingest` raises ``RuntimeError`` with a
helpful install message so the ASP degrades cleanly when pyav is absent.

Environment variables
---------------------
``ASP_VIDEO_PROXY_SCALE``   : proxy scale factor (default 0.25).
``ASP_VIDEO_MAX_FRAMES``    : hard cap on returned frame count (default 200).
``ASP_VIDEO_TELECINE_MAD``  : telecine-drop MAD threshold (default 2.0).
``ASP_VIDEO_KEYFRAMES_ONLY``: if "1", use I-frame-only decode (default 0).
"""

from __future__ import annotations

# --------------------------------
import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

# --- Relocated Nested Imports ---
from backend.src.animation.ingestion.frame_selection import smart_select_frames

logger = logging.getLogger(__name__)

# ── env-var flags ─────────────────────────────────────────────────────────────

_PROXY_SCALE = float(os.environ.get("ASP_VIDEO_PROXY_SCALE", "0.25"))
_MAX_FRAMES = int(os.environ.get("ASP_VIDEO_MAX_FRAMES", "200"))
_TELECINE_MAD = float(os.environ.get("ASP_VIDEO_TELECINE_MAD", "2.0"))
_KEYFRAMES_ONLY = os.environ.get("ASP_VIDEO_KEYFRAMES_ONLY", "0") != "0"

# ── optional pyav ─────────────────────────────────────────────────────────────

try:
    import av as _av

    _AV_OK = True
except ImportError:
    _AV_OK = False
    _av = None  # type: ignore[assignment]


__all__ = [
    "VideoIngestionStream",
    "ingest_video",
    "_decode_proxy_frames",
    "_telecine_dedup",
    "_uniform_select",
]


# ── helpers ───────────────────────────────────────────────────────────────────


def _decode_proxy_frames(
    video_path: str,
    scale: float = _PROXY_SCALE,
    keyframes_only: bool = _KEYFRAMES_ONLY,
) -> List[Tuple[int, np.ndarray]]:
    """
    Decode a sparse proxy (¼-resolution by default) of the video.

    Returns
    -------
    list of (frame_index, uint8 BGR ndarray at proxy resolution)
    """
    if not _AV_OK:
        raise RuntimeError(
            "pyav is required for video ingestion. Install with: pip install av"
        )
    results: List[Tuple[int, np.ndarray]] = []
    container = _av.open(video_path)
    stream = container.streams.video[0]
    if keyframes_only:
        stream.codec_context.skip_frame = "NONKEY"

    fi = 0
    for packet in container.demux(stream):
        for frame in packet.decode():
            img = frame.to_ndarray(format="bgr24")
            if scale != 1.0 and scale > 0:
                h, w = img.shape[:2]
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            results.append((fi, img))
            fi += 1

    container.close()
    return results


def _telecine_dedup(
    proxy_frames: List[Tuple[int, np.ndarray]],
    mad_thresh: float = _TELECINE_MAD,
) -> List[int]:
    """
    Drop duplicate frames caused by 3:2 pull-down telecine.

    Returns kept frame *indices into proxy_frames* (not PTS indices).
    """
    if not proxy_frames:
        return []
    kept = [0]
    prev_gray = cv2.cvtColor(proxy_frames[0][1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    for i in range(1, len(proxy_frames)):
        gray = cv2.cvtColor(proxy_frames[i][1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        mad = float(np.mean(np.abs(gray - prev_gray)))
        if mad >= mad_thresh:
            kept.append(i)
            prev_gray = gray
    return kept


def _uniform_select(n_total: int, n_select: int) -> List[int]:
    """Return ``n_select`` evenly-spaced indices in [0, n_total)."""
    if n_select <= 0 or n_total == 0:
        return []
    if n_select >= n_total:
        return list(range(n_total))
    step = n_total / n_select
    return [int(i * step) for i in range(n_select)]


def _decode_full_frame(video_path: str, frame_pts_index: int) -> Optional[np.ndarray]:
    """
    Decode a single full-resolution frame at the given presentation frame index.

    Uses libavformat fast seek to avoid buffering the entire video.
    Returns None on error.
    """
    if not _AV_OK:
        return None
    try:
        container = _av.open(video_path)
        stream = container.streams.video[0]
        # Seek to approximately the right PTS
        fps = float(stream.average_rate or 24)
        target_ts = int(frame_pts_index / fps / stream.time_base)
        container.seek(target_ts, stream=stream, backward=True, any_frame=False)

        fi = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                if fi >= frame_pts_index:
                    img = frame.to_ndarray(format="bgr24")
                    container.close()
                    return img
                fi += 1
        container.close()
        return None
    except Exception as e:
        logger.warning(
            f"[VideoIngestion] Full-res decode failed at index {frame_pts_index}: {e}"
        )
        return None


# ── main class ────────────────────────────────────────────────────────────────


class VideoIngestionStream:
    """
    Proxy-first video frame decoder for AnimeStitchPipeline.

    Usage
    -----
    >>> stream = VideoIngestionStream("episode01.mp4", n_frames=20)
    >>> frames, paths = stream.ingest(tmp_dir="/tmp/asp_frames/")
    >>> # paths = list of saved PNG paths; frames = list of BGR ndarrays
    >>> pipeline.run(paths, "output.png")

    Parameters
    ----------
    video_path  : path to the video file (MP4, MKV, AVI, etc.)
    n_frames    : number of frames to extract (after telecine dedup + selection)
    mode        : ``"uniform"`` | ``"keyframe"`` | ``"smart"``
                  "smart" defers to smart_select_frames() from frame_selection.py
    telecine    : if True, apply telecine-drop deduplication before selection
    proxy_scale : proxy decode resolution factor (default: ASP_VIDEO_PROXY_SCALE env)
    max_frames  : hard cap on extracted frames (default: ASP_VIDEO_MAX_FRAMES env)
    """

    def __init__(
        self,
        video_path: str,
        n_frames: int = 20,
        mode: str = "uniform",
        telecine: bool = True,
        proxy_scale: float = _PROXY_SCALE,
        max_frames: int = _MAX_FRAMES,
    ):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        self.video_path = video_path
        self.n_frames = min(n_frames, max_frames)
        self.mode = mode
        self.telecine = telecine
        self.proxy_scale = proxy_scale
        self.max_frames = max_frames

        # Populated by ingest()
        self._proxy_frames: List[Tuple[int, np.ndarray]] = []
        self._selected_pts_indices: List[int] = []

    # ── public API ────────────────────────────────────────────────────────────

    def ingest(self, tmp_dir: str) -> Tuple[List[np.ndarray], List[str]]:
        """
        Decode + select frames and save them as PNGs in ``tmp_dir``.

        Returns
        -------
        (frames, paths) where frames is a list of BGR ndarrays and paths is a
        list of the saved PNG paths (usable directly in AnimeStitchPipeline.run()).
        """
        if not _AV_OK:
            raise RuntimeError(
                "pyav is required for video ingestion. Install with: pip install av"
            )

        os.makedirs(tmp_dir, exist_ok=True)

        # Step 1 — proxy decode
        logger.info(
            f"[VideoIngestion] Proxy decode: {self.video_path} (scale={self.proxy_scale})"
        )
        self._proxy_frames = _decode_proxy_frames(
            self.video_path,
            scale=self.proxy_scale,
            keyframes_only=(self.mode == "keyframe"),
        )
        logger.info(
            f"[VideoIngestion] Proxy: {len(self._proxy_frames)} frames decoded."
        )

        # Step 2 — telecine deduplication
        proxy_subset = list(range(len(self._proxy_frames)))
        if self.telecine and len(self._proxy_frames) > 1:
            proxy_subset = _telecine_dedup(self._proxy_frames, mad_thresh=_TELECINE_MAD)
            logger.info(
                f"[VideoIngestion] After telecine dedup: {len(proxy_subset)} frames."
            )

        # Step 3 — frame selection within deduped subset
        selected_proxy_idx = self._select(proxy_subset)
        self._selected_pts_indices = [
            self._proxy_frames[i][0] for i in selected_proxy_idx
        ]
        logger.info(
            f"[VideoIngestion] Selected {len(self._selected_pts_indices)} frames "
            f"via mode='{self.mode}'."
        )

        # Step 4 — full-resolution decode for selected frames only
        frames: List[np.ndarray] = []
        paths: List[str] = []
        for seq_idx, pts_idx in enumerate(self._selected_pts_indices):
            img = _decode_full_frame(self.video_path, pts_idx)
            if img is None:
                # Fall back to the proxy frame (already decoded)
                proxy_img = self._proxy_frames[selected_proxy_idx[seq_idx]][1]
                img = (
                    cv2.resize(
                        proxy_img,
                        (
                            int(proxy_img.shape[1] / self.proxy_scale),
                            int(proxy_img.shape[0] / self.proxy_scale),
                        ),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                    if self.proxy_scale < 1.0
                    else proxy_img
                )
                logger.warning(
                    f"[VideoIngestion] Full-res decode failed for PTS {pts_idx}; "
                    "using proxy upscale."
                )
            out_path = os.path.join(
                tmp_dir, f"frame_{seq_idx:04d}_pts{pts_idx:06d}.png"
            )
            cv2.imwrite(out_path, img)
            frames.append(img)
            paths.append(out_path)
            logger.debug(f"[VideoIngestion]   Saved {out_path}")

        logger.info(
            f"[VideoIngestion] Done. {len(frames)} full-res frames saved to {tmp_dir}"
        )
        return frames, paths

    def proxy_frames(self) -> List[Tuple[int, np.ndarray]]:
        """Return the decoded proxy frames (populated after ingest())."""
        return self._proxy_frames

    # ── private ───────────────────────────────────────────────────────────────

    def _select(self, proxy_subset: List[int]) -> List[int]:
        """Select frame indices within ``proxy_subset`` according to self.mode."""
        n = len(proxy_subset)
        want = min(self.n_frames, n)

        if self.mode in ("keyframe", "uniform") or n <= want:
            chosen_positions = _uniform_select(n, want)
            return [proxy_subset[p] for p in chosen_positions]

        if self.mode == "smart":
            try:
                # relocated: from backend.src.animation.ingestion.frame_selection import smart_select_frames
                proxy_imgs = [self._proxy_frames[i][1] for i in proxy_subset]
                # smart_select_frames returns list of BGR ndarrays; we need indices
                selected_imgs = smart_select_frames(proxy_imgs, target_n=want)
                # Map back to indices by identity
                img_id_to_idx = {
                    id(self._proxy_frames[i][1]): pos
                    for pos, i in enumerate(proxy_subset)
                }
                selected_proxy_pos = [
                    img_id_to_idx.get(id(img), -1) for img in selected_imgs
                ]
                valid = [proxy_subset[p] for p in selected_proxy_pos if p >= 0]
                if valid:
                    return valid
            except Exception as e:
                logger.warning(
                    f"[VideoIngestion] smart_select_frames failed ({e}); falling back to uniform."
                )
            # Fall back to uniform
            chosen_positions = _uniform_select(n, want)
            return [proxy_subset[p] for p in chosen_positions]

        # Unknown mode — uniform fallback
        chosen_positions = _uniform_select(n, want)
        return [proxy_subset[p] for p in chosen_positions]


# ── convenience function ──────────────────────────────────────────────────────


def ingest_video(
    video_path: str,
    tmp_dir: str,
    n_frames: int = 20,
    mode: str = "uniform",
    telecine: bool = True,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    One-call convenience wrapper around :class:`VideoIngestionStream`.

    Returns ``(frames, paths)`` ready for ``AnimeStitchPipeline.run(paths, output)``.
    """
    stream = VideoIngestionStream(
        video_path, n_frames=n_frames, mode=mode, telecine=telecine
    )
    return stream.ingest(tmp_dir)
