import os
from math import gcd

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal


class _StatsSignals(QObject):
    individual_done = Signal(list)  # List[dict] — one dict per image
    pairwise_done = Signal(list)  # List[dict] — one dict per pair
    progress = Signal(int, int)  # (completed, total) — §5.9 Option C
    error = Signal(str)


class StatsWorker(QRunnable):
    """
    Computes per-image and pairwise statistics for a list of image paths.

    Per-image metrics
    -----------------
    resolution, aspect_ratio, brightness, contrast, sharpness, saturation,
    dominant_hue, noise_estimate, file_size_kb

    Pairwise metrics (consecutive pairs + all pairs if ≤ 12 images)
    ---------------------------------------------------------------
    hist_correlation, ssim, orb_inliers, mean_diff
    """

    def __init__(self, paths: list[str], knn_window: int = 20):
        super().__init__()
        self.setAutoDelete(True)
        self._paths = list(paths)
        self._knn_window = max(1, knn_window)
        self.signals = _StatsSignals()
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
        n = len(paths)
        if n == 0:
            self.signals.individual_done.emit([])
            self.signals.pairwise_done.emit([])
            return

        individual: list[dict] = []
        knn = self._knn_window
        bgr_cache: dict[str, np.ndarray] = {}

        if n <= 12:
            pairs = [(i, j, True) for i in range(n) for j in range(i + 1, n)]
        else:
            seen: set = set()
            pairs = []
            for i in range(n - 1):
                if (i, i + 1) not in seen:
                    pairs.append((i, i + 1, True))
                    seen.add((i, i + 1))
            for i in range(n):
                for step in range(2, knn + 1):
                    j = i + step
                    if j < n and (i, j) not in seen:
                        pairs.append((i, j, False))
                        seen.add((i, j))

        total_steps = n + len(pairs)

        for done, path in enumerate(paths, start=1):
            if self._cancelled:
                return
            row = self._image_stats(path)
            individual.append(row)
            bgr = cv2.imread(path)
            if bgr is not None:
                h, w = bgr.shape[:2]
                scale = min(1.0, 512 / max(h, w, 1))
                if scale < 1.0:
                    bgr = cv2.resize(
                        bgr,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
            bgr_cache[path] = bgr # pyrefly: ignore [unsupported-operation]
            self.signals.progress.emit(done, total_steps)

        self.signals.individual_done.emit(individual)

        pairwise: list[dict] = []

        orb = cv2.ORB_create(nfeatures=500) # pyrefly: ignore [missing-attribute]
        for done_pw, (i, j, is_consec) in enumerate(pairs, start=1):
            if self._cancelled:
                return
            pa, pb = paths[i], paths[j]
            a = bgr_cache.get(pa)
            b = bgr_cache.get(pb)
            row = self._pair_stats(pa, pb, a, b, i, j, orb)
            row["consecutive"] = is_consec
            pairwise.append(row)
            self.signals.progress.emit(n + done_pw, total_steps)

        self.signals.pairwise_done.emit(pairwise)
        self.signals.progress.emit(total_steps, total_steps)

    @staticmethod
    def _image_stats(path: str) -> dict:
        row: dict = {"path": path, "name": os.path.basename(path)}
        try:
            file_size_kb = round(os.path.getsize(path) / 1024, 1)
        except OSError:
            file_size_kb = 0.0
        row["file_size_kb"] = file_size_kb

        bgr = cv2.imread(path)
        if bgr is None:
            row.update(
                {
                    "width": 0,
                    "height": 0,
                    "aspect_ratio": "—",
                    "brightness": 0.0,
                    "contrast": 0.0,
                    "sharpness": 0.0,
                    "saturation": 0.0,
                    "dominant_hue": 0,
                    "noise": 0.0,
                }
            )
            return row

        h, w = bgr.shape[:2]
        row["width"] = w
        row["height"] = h

        g = gcd(w, h)
        row["aspect_ratio"] = f"{w // g}:{h // g}"

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        row["brightness"] = round(float(gray.mean()), 2)
        row["contrast"] = round(float(gray.std()), 2)

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        row["sharpness"] = round(float(lap.var()), 2)

        lap_abs = np.abs(lap - np.median(lap))
        row["noise"] = round(float(np.median(lap_abs)), 2)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        row["saturation"] = round(float(hsv[:, :, 1].mean()), 2)

        sat_mask = (hsv[:, :, 1] > 30).astype(np.uint8)
        if sat_mask.sum() > 100:
            hue_hist = cv2.calcHist([hsv], [0], sat_mask, [180], [0, 180])
            row["dominant_hue"] = int(np.argmax(hue_hist))
        else:
            row["dominant_hue"] = -1

        return row

    @staticmethod
    def _pair_stats(pa: str, pb: str, a, b, i: int, j: int, orb) -> dict:
        row = {
            "idx_a": i,
            "idx_b": j,
            "path_a": pa,
            "path_b": pb,
            "name_a": os.path.basename(pa),
            "name_b": os.path.basename(pb),
            "hist_corr": 0.0,
            "ssim": 0.0,
            "orb_inliers": 0,
            "mean_diff": 0.0,
        }

        if a is None or b is None:
            return row

        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        ar = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        br = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)

        corrs = []
        for c in range(3):
            ha = cv2.calcHist([ar], [c], None, [64], [0, 256])
            hb = cv2.calcHist([br], [c], None, [64], [0, 256])
            corrs.append(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
        row["hist_corr"] = round(float(np.mean(corrs)), 4)

        ga = cv2.cvtColor(ar, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gb = cv2.cvtColor(br, cv2.COLOR_BGR2GRAY).astype(np.float32)
        C1, C2 = 6.5025, 58.5225
        mu_a = cv2.GaussianBlur(ga, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(gb, (11, 11), 1.5)
        mu_a2, mu_b2, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b
        sig_a2 = cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - mu_a2
        sig_b2 = cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - mu_b2
        sig_ab = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - mu_ab
        ssim_map = ((2 * mu_ab + C1) * (2 * sig_ab + C2)) / (
            (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)
        )
        row["ssim"] = round(float(ssim_map.mean()), 4)

        row["mean_diff"] = round(
            float(np.abs(ar.astype(np.float32) - br.astype(np.float32)).mean()), 2
        )

        try:
            kp_a, des_a = orb.detectAndCompute(
                cv2.cvtColor(ar, cv2.COLOR_BGR2GRAY), None
            )
            kp_b, des_b = orb.detectAndCompute(
                cv2.cvtColor(br, cv2.COLOR_BGR2GRAY), None
            )
            if (
                des_a is not None
                and des_b is not None
                and len(kp_a) >= 4
                and len(kp_b) >= 4
            ):
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = bf.knnMatch(des_a, des_b, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                if len(good) >= 4:
                    # pyrefly: ignore [bad-argument-type]
                    src_pts = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(
                        -1, 1, 2
                    )
                    # pyrefly: ignore [bad-argument-type]
                    dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(
                        -1, 1, 2
                    )
                    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    row["orb_inliers"] = (
                        int(mask.sum()) if mask is not None else len(good)
                    )
                else:
                    row["orb_inliers"] = len(good)
        except Exception:
            pass

        return row
