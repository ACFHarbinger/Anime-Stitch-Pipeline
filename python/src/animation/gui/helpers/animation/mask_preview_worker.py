from __future__ import annotations

import cv2
from PySide6.QtCore import QThread, Signal


class MaskPreviewWorker(QThread):
    sig_finished = Signal(object)  # np.ndarray (H,W) uint8
    sig_error = Signal(str)

    def __init__(self, img_path: str):
        super().__init__()
        self._path = img_path

    def run(self):
        try:
            from backend.src.models.wrappers.birefnet_wrapper import BiRefNetWrapper

            img = cv2.imread(self._path)
            if img is None:
                self.sig_error.emit("Could not read image.")
                return
            br = BiRefNetWrapper()
            if hasattr(br, "get_background_mask"):
                mask = br.get_background_mask(img)
            else:
                fg = br.get_mask(img)
                mask = cv2.bitwise_not(fg)
            self.sig_finished.emit(mask)
        except Exception as e:
            self.sig_error.emit(str(e))
