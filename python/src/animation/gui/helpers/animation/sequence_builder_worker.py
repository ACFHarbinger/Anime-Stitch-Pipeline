import os
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal


class _SeqBuilderSignals(QObject):
    progress = Signal(int, int)  # (percent, 100) — §5.9 Option C; weighted 3-phase estimate
    result = Signal(list)  # List[dict]: ordered chain items
    error = Signal(str)


class SequenceBuilderWorker(QRunnable):
    """
    Given an anchor image and a pool of candidates, builds the longest
    sequential stitching chain greedily.

    Scoring — stitchability, not similarity
    ----------------------------------------
    Two frames are good for stitching when they share overlapping content AND
    the camera has panned enough to reveal new content.  The old approach
    (SSIM + hist_corr + ORB inliers) measured raw similarity, so near-identical
    consecutive frames scored highest — the opposite of what is needed.

    This version scores each candidate by:
      1. ORB feature matching + RANSAC homography against the current tail.
      2. Extracting the translation (dx, dy) from the homography.
      3. Rejecting near-duplicates  : |translation| < min_pan  (same view)
      4. Rejecting non-overlapping  : |translation| > max_pan  (no shared content)
      5. Fitness = inlier_ratio × displacement_quality(ratio)
         where displacement_quality peaks at ~30% of frame diagonal and falls
         off toward 0 at the min/max boundaries.

    Sharpness filter
    ----------------
    Each candidate is compared against the anchor's Laplacian variance.
    Candidates whose sharpness is below `blur_threshold × anchor_sharpness`
    are excluded before the chain search begins.
    """

    def __init__(
        self,
        anchor: str,
        candidates: List[str],
        min_score: float = 0.25,
        blur_threshold: float = 0.5,
        min_pan_ratio: float = 0.03,
        max_pan_ratio: float = 0.85,
    ):
        super().__init__()
        self.setAutoDelete(True)
        self._anchor = anchor
        self._candidates = [p for p in candidates if p != anchor]
        self._min_score = min_score
        self._blur_threshold = blur_threshold
        self._min_pan = min_pan_ratio
        self._max_pan = max_pan_ratio
        self.signals = _SeqBuilderSignals()
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self._build()
        except Exception as exc:
            self.signals.error.emit(str(exc))

    def _build(self):  # noqa: C901
        all_paths = [self._anchor] + self._candidates
        n = len(all_paths)
        if n < 2:
            self.signals.result.emit(
                [
                    {
                        "path": self._anchor,
                        "name": os.path.basename(self._anchor),
                        "score_to_prev": None,
                    }
                ]
            )
            return

        orb = cv2.ORB_create(nfeatures=800)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        cache: Dict[str, Optional[np.ndarray]] = {}
        feats: Dict[str, tuple] = {}  # (kp, des)
        sharpness: Dict[str, float] = {}

        for idx, p in enumerate(all_paths):
            if self._cancelled:
                return
            bgr = cv2.imread(p)
            if bgr is not None:
                h, w = bgr.shape[:2]
                scale = min(1.0, 512 / max(h, w, 1))
                if scale < 1.0:
                    bgr = cv2.resize(
                        bgr,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
                sharpness[p] = float(lap.var())
                kp, des = orb.detectAndCompute(gray, None)
                feats[p] = (kp, des)
            else:
                sharpness[p] = 0.0
                feats[p] = ([], None)
            cache[p] = bgr
            self.signals.progress.emit(int((idx + 1) / n * 45), 100)

        anchor_sharp = max(sharpness.get(self._anchor, 1.0), 1.0)
        sharp_thresh = anchor_sharp * self._blur_threshold

        valid_candidates = [
            p for p in self._candidates if sharpness.get(p, 0.0) >= sharp_thresh
        ]
        n_rejected = len(self._candidates) - len(valid_candidates)
        if n_rejected:
            print(
                f"[SeqBuilder] Rejected {n_rejected} blurry candidates "
                f"(sharpness < {sharp_thresh:.1f})."
            )

        fitness_cache: Dict[tuple, tuple] = {}  # key → (score, dx, dy)

        def stitch_fitness(ref_p: str, cand_p: str) -> tuple:
            key = (min(ref_p, cand_p), max(ref_p, cand_p))
            if key in fitness_cache:
                return fitness_cache[key]

            kp_r, des_r = feats.get(ref_p, ([], None))
            kp_c, des_c = feats.get(cand_p, ([], None))
            zero = (0.0, 0.0, 0.0)
            if des_r is None or des_c is None:
                fitness_cache[key] = zero
                return zero
            if len(kp_r) < 6 or len(kp_c) < 6:
                fitness_cache[key] = zero
                return zero

            try:
                matches = bf.knnMatch(des_r, des_c, k=2)
            except Exception:
                fitness_cache[key] = zero
                return zero

            good = [
                m
                for m, n2 in matches
                if len((m, n2)) == 2 and m.distance < 0.75 * n2.distance
            ]
            if len(good) < 8:
                fitness_cache[key] = zero
                return zero

            src_pts = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_c[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None or mask is None:
                fitness_cache[key] = zero
                return zero

            inliers = int(mask.sum())
            if inliers < 8:
                fitness_cache[key] = zero
                return zero

            dx, dy = float(M[0, 2]), float(M[1, 2])

            ref_img = cache.get(ref_p)
            if ref_img is None:
                fitness_cache[key] = zero
                return zero
            fh, fw = ref_img.shape[:2]
            diag = float(np.sqrt(fw**2 + fh**2))
            dist = float(np.sqrt(dx**2 + dy**2))
            ratio = dist / diag

            if ratio < self._min_pan or ratio > self._max_pan:
                fitness_cache[key] = zero
                return zero

            peak = 0.30
            disp_q = ratio / peak if ratio <= peak else (self._max_pan - ratio) / (self._max_pan - peak)
            disp_q = max(0.0, disp_q)

            inlier_ratio = inliers / max(len(good), 1)
            score = round(inlier_ratio * disp_q, 4)

            result = (score, dx, dy)
            fitness_cache[key] = result
            return result

        chain: List[str] = [self._anchor]
        used: set = {self._anchor}

        def best_next(ref: str) -> tuple:
            best_p, best_s, best_dx, best_dy = None, -1.0, 0.0, 0.0
            for p in valid_candidates:
                if p in used:
                    continue
                s, dx, dy = stitch_fitness(ref, p)
                if s > best_s:
                    best_s, best_p, best_dx, best_dy = s, p, dx, dy
            return best_p, best_s, best_dx, best_dy

        total = len(valid_candidates)
        done = 0

        while True:
            if self._cancelled:
                return
            nxt, s, _dx, _dy = best_next(chain[-1])
            if nxt is None or s < self._min_score:
                break
            chain.append(nxt)
            used.add(nxt)
            done += 1
            self.signals.progress.emit(45 + int(done / max(total, 1) * 27), 100)

        while True:
            if self._cancelled:
                return
            prv, s, _dx, _dy = best_next(chain[0])
            if prv is None or s < self._min_score:
                break
            chain.insert(0, prv)
            used.add(prv)
            done += 1
            self.signals.progress.emit(72 + int(done / max(total, 1) * 27), 100)

        result: List[dict] = []
        for idx, p in enumerate(chain):
            s_prev = None if idx == 0 else stitch_fitness(chain[idx - 1], p)[0]
            result.append(
                {"path": p, "name": os.path.basename(p), "score_to_prev": s_prev}
            )

        self.signals.progress.emit(100, 100)
        self.signals.result.emit(result)
