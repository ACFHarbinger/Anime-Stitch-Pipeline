from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


class MatchWorker(QThread):
    sig_finished = Signal(object, object, object)  # pts1 (K,2), pts2 (K,2), conf (K,)
    sig_error = Signal(str)

    def __init__(
        self,
        img_path_a: str,
        img_path_b: str,
        conf_thresh: float = 0.4,
        use_birefnet: bool = True,
    ):
        super().__init__()
        self._path_a = img_path_a
        self._path_b = img_path_b
        self._conf_thresh = conf_thresh
        self._use_birefnet = use_birefnet

    def run(self):
        try:
            from backend.src.models.wrappers.birefnet_wrapper import BiRefNetWrapper
            from backend.src.models.wrappers.loftr_wrapper import LoFTRWrapper

            img_a = cv2.imread(self._path_a)
            img_b = cv2.imread(self._path_b)
            if img_a is None or img_b is None:
                self.sig_error.emit("Could not read one or both images.")
                return

            mask_a: Optional[np.ndarray] = None
            mask_b: Optional[np.ndarray] = None
            if self._use_birefnet:
                br = BiRefNetWrapper()
                if hasattr(br, "get_background_mask"):
                    mask_a = br.get_background_mask(img_a)
                    mask_b = br.get_background_mask(img_b)

            wrapper = LoFTRWrapper()
            pts1, pts2, conf = wrapper.match_masked(
                img_a, img_b, mask_a, mask_b, conf_thresh=self._conf_thresh
            )
            self.sig_finished.emit(pts1, pts2, conf)
        except Exception as e:
            self.sig_error.emit(str(e))
