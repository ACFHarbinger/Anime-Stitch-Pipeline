import cv2
from gui.src.constants import STITCH_THUMB_H, STITCH_THUMB_W
from PySide6.QtCore import QObject, QRunnable, Signal
from PySide6.QtGui import QColor, QImage, QPixmap


def _load_thumb(path: str, w: int = STITCH_THUMB_W, h: int = STITCH_THUMB_H) -> QPixmap:
    bgr = cv2.imread(path)
    if bgr is None:
        pm = QPixmap(w, h)
        pm.fill(QColor("#333"))
        return pm
    ih, iw = bgr.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qi = QImage(rgb.data, nw, nh, nw * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qi)


class _ThumbSignals(QObject):
    done = Signal(str, QPixmap)  # path, pixmap


class ThumbLoader(QRunnable):
    def __init__(self, path: str):
        super().__init__()
        self.setAutoDelete(True)
        self._path = path
        self.signals = _ThumbSignals()

    def run(self):
        pm = _load_thumb(self._path)
        self.signals.done.emit(self._path, pm)
