import os
from typing import List

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal
from sklearn.cluster import KMeans


class _AnimClusterSignals(QObject):
    progress = Signal(int, int)  # (completed, total) — §5.9 Option C
    finished = Signal(list)
    error = Signal(str)


class AnimClusterWorker(QRunnable):
    """
    Groups a list of image paths into animation phases using per-pixel temporal
    FFT analysis (replicating AnimeStitchPipeline._cluster_animation_phases).

    Each result dict:
        path         : str   — absolute image path
        cluster      : int   — 0-based phase index  (-1 = unassigned)
        cluster_name : str   — human-readable label
        is_animated  : bool  — True if temporal animation was detected
        ac_ratio     : float — mean AC/(DC+AC) ratio across the frame set
    """

    def __init__(
        self,
        paths: List[str],
        ac_threshold: float = 0.25,
        min_anim_pixels: int = 500,
    ):
        super().__init__()
        self.setAutoDelete(True)
        self._paths = list(paths)
        self._ac_threshold = ac_threshold
        self._min_anim_pixels = min_anim_pixels
        self.signals = _AnimClusterSignals()
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self._compute()
        except Exception as exc:
            self.signals.error.emit(str(exc))

    def _compute(self):
        paths = self._paths
        N = len(paths)
        if N == 0:
            self.signals.finished.emit([])
            return

        total_steps = 2 * N

        frames: List[np.ndarray] = []
        H = W = 0
        for i, p in enumerate(paths):
            if self._cancelled:
                return
            img = cv2.imread(p)
            if img is None:
                img = np.zeros((100, 100, 3), np.uint8)
            if i == 0:
                H, W = img.shape[:2]
            elif img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            frames.append(img)
            self.signals.progress.emit(i + 1, total_steps)

        if N < 4:
            rows = [
                {
                    "path": p,
                    "cluster": 0,
                    "cluster_name": "Static (need ≥ 4 frames)",
                    "is_animated": False,
                    "ac_ratio": 0.0,
                }
                for p in paths
            ]
            self.signals.finished.emit(rows)
            return

        target_w = 320
        scale = target_w / max(W, 1)
        th = max(1, int(H * scale))
        tw = target_w

        small_stack: List[np.ndarray] = []
        for i, frame in enumerate(frames):
            if self._cancelled:
                return
            M_small = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0]], np.float32)
            warped = cv2.warpAffine(frame, M_small, (tw, th), flags=cv2.INTER_AREA)
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            small_stack.append(gray)
            self.signals.progress.emit(N + i + 1, total_steps)

        stack_arr = np.stack(small_stack, axis=0)  # (N, th, tw)

        F = np.fft.rfft(stack_arr, axis=0)
        power = np.abs(F) ** 2
        dc_power = power[0]
        ac_power = power[1:].sum(axis=0)
        ratio = ac_power / (dc_power + ac_power + 1e-8)
        mean_ratio = float(ratio.mean())

        anim_mask = (ratio > self._ac_threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        anim_mask = cv2.morphologyEx(anim_mask, cv2.MORPH_OPEN, kernel)
        anim_mask = cv2.morphologyEx(anim_mask, cv2.MORPH_CLOSE, kernel)

        n_anim_px = int(anim_mask.sum()) // 255
        if n_anim_px < self._min_anim_pixels:
            rows = [
                {
                    "path": p,
                    "cluster": 0,
                    "cluster_name": "Static (no animation detected)",
                    "is_animated": False,
                    "ac_ratio": mean_ratio,
                }
                for p in paths
            ]
            self.signals.finished.emit(rows)
            return

        anim_ys, anim_xs = np.where(anim_mask > 0)
        sigs: List[np.ndarray] = []
        for gray in small_stack:
            edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
            sigs.append(edges[anim_ys, anim_xs].astype(np.float32))
        sig_matrix = np.stack(sigs, axis=0)  # (N, K)

        n_clusters = max(2, min(8, N // 2))
        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
        labels = km.fit_predict(sig_matrix)

        self.signals.progress.emit(total_steps, total_steps)

        rows = []
        for i, p in enumerate(paths):
            c = int(labels[i])
            rows.append(
                {
                    "path": p,
                    "cluster": c,
                    "cluster_name": f"Phase {c + 1}",
                    "is_animated": True,
                    "ac_ratio": mean_ratio,
                }
            )
        rows.sort(key=lambda r: (r["cluster"], os.path.basename(r["path"])))

        self.signals.finished.emit(rows)
