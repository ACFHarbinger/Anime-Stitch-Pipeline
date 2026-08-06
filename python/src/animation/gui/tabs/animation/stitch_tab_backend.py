"""
gui/src/tabs/animation/stitch_tab_backend.py
=============================================
QML-facing QObject backend for StitchTab.qml.

Exposes the AnimeStitchPipeline through a clean set of Qt Properties,
Signals, and Slots so that StitchTab.qml can bind to it via
``mainBackend.stitchTab``.

Property summary
----------------
source_dir       string   directory of input frames
output_path      string   output file path (empty = auto-name)
is_stitching     bool     True while the pipeline is running
progress         int      0-100 percentage progress
status_text      string   one-line human-readable status
log_output       string   accumulated pipeline log text

Toggle properties (all bool, default True unless noted)
-----------------------------------------------
use_loftr        bool     LoFTR dense feature matching
use_birefnet     bool     BiRefNet foreground masking
use_apap         bool     As-Projective-As-Possible mesh warping
use_ecc          bool     ECC sub-pixel alignment refinement
use_basic        bool     BaSiC photometric correction
composite_fg     bool     foreground compositing pass
use_poisson      bool     Poisson seam blending (default False)

Slot summary
------------
browse_input_directory_qml(current_path: str)
browse_output_qml()
start_stitch_qml()
cancel_stitch_qml()
"""

from __future__ import annotations

import glob
import os
from typing import Optional

from gui.src.helpers.animation.stitch_worker import StitchWorker
from PySide6.QtCore import Property, QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QFileDialog
from gui.src.constants.tabs import _IMAGE_EXTS


def _collect_images(directory: str) -> list[str]:
    paths: list[str] = []
    for ext in _IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(paths)


class StitchTabBackend(QObject):
    """QML-facing backend for the Stitch tab."""

    # ── Signals ──────────────────────────────────────────────────────────────
    sourceDirChanged = Signal()
    outputPathChanged = Signal()
    isStitchingChanged = Signal()
    progressChanged = Signal()
    statusTextChanged = Signal()
    logOutputChanged = Signal()

    useLoftrChanged = Signal()
    useBirefnetChanged = Signal()
    useApapChanged = Signal()
    useEccChanged = Signal()
    useBasicChanged = Signal()
    compositeFgChanged = Signal()
    usePoissonChanged = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)

        self._source_dir: str = ""
        self._output_path: str = ""
        self._is_stitching: bool = False
        self._progress: int = 0
        self._status_text: str = "Ready."
        self._log_output: str = ""

        self._use_loftr: bool = True
        self._use_birefnet: bool = True
        self._use_apap: bool = True
        self._use_ecc: bool = True
        self._use_basic: bool = True
        self._composite_fg: bool = True
        self._use_poisson: bool = False

        self._worker: Optional[StitchWorker] = None
        self._thread: Optional[QThread] = None

    # ── source_dir ────────────────────────────────────────────────────────────
    def _get_source_dir(self) -> str:
        return self._source_dir

    def _set_source_dir(self, value: str) -> None:
        if self._source_dir != value:
            self._source_dir = value
            self.sourceDirChanged.emit()

    source_dir = Property(str, _get_source_dir, _set_source_dir, notify=sourceDirChanged)

    # ── output_path ───────────────────────────────────────────────────────────
    def _get_output_path(self) -> str:
        return self._output_path

    def _set_output_path(self, value: str) -> None:
        if self._output_path != value:
            self._output_path = value
            self.outputPathChanged.emit()

    output_path = Property(str, _get_output_path, _set_output_path, notify=outputPathChanged)

    # ── is_stitching ──────────────────────────────────────────────────────────
    def _get_is_stitching(self) -> bool:
        return self._is_stitching

    is_stitching = Property(bool, _get_is_stitching, notify=isStitchingChanged)

    # ── progress ──────────────────────────────────────────────────────────────
    def _get_progress(self) -> int:
        return self._progress

    progress = Property(int, _get_progress, notify=progressChanged)

    # ── status_text ───────────────────────────────────────────────────────────
    def _get_status_text(self) -> str:
        return self._status_text

    status_text = Property(str, _get_status_text, notify=statusTextChanged)

    # ── log_output ────────────────────────────────────────────────────────────
    def _get_log_output(self) -> str:
        return self._log_output

    log_output = Property(str, _get_log_output, notify=logOutputChanged)

    # ── Toggle: use_loftr ─────────────────────────────────────────────────────
    def _get_use_loftr(self) -> bool:
        return self._use_loftr

    def _set_use_loftr(self, v: bool) -> None:
        if self._use_loftr != v:
            self._use_loftr = v
            self.useLoftrChanged.emit()

    use_loftr = Property(bool, _get_use_loftr, _set_use_loftr, notify=useLoftrChanged)

    # ── Toggle: use_birefnet ──────────────────────────────────────────────────
    def _get_use_birefnet(self) -> bool:
        return self._use_birefnet

    def _set_use_birefnet(self, v: bool) -> None:
        if self._use_birefnet != v:
            self._use_birefnet = v
            self.useBirefnetChanged.emit()

    use_birefnet = Property(bool, _get_use_birefnet, _set_use_birefnet, notify=useBirefnetChanged)

    # ── Toggle: use_apap (controls motion_model: affine vs translation) ───────
    def _get_use_apap(self) -> bool:
        return self._use_apap

    def _set_use_apap(self, v: bool) -> None:
        if self._use_apap != v:
            self._use_apap = v
            self.useApapChanged.emit()

    use_apap = Property(bool, _get_use_apap, _set_use_apap, notify=useApapChanged)

    # ── Toggle: use_ecc ───────────────────────────────────────────────────────
    def _get_use_ecc(self) -> bool:
        return self._use_ecc

    def _set_use_ecc(self, v: bool) -> None:
        if self._use_ecc != v:
            self._use_ecc = v
            self.useEccChanged.emit()

    use_ecc = Property(bool, _get_use_ecc, _set_use_ecc, notify=useEccChanged)

    # ── Toggle: use_basic ─────────────────────────────────────────────────────
    def _get_use_basic(self) -> bool:
        return self._use_basic

    def _set_use_basic(self, v: bool) -> None:
        if self._use_basic != v:
            self._use_basic = v
            self.useBasicChanged.emit()

    use_basic = Property(bool, _get_use_basic, _set_use_basic, notify=useBasicChanged)

    # ── Toggle: composite_fg ──────────────────────────────────────────────────
    def _get_composite_fg(self) -> bool:
        return self._composite_fg

    def _set_composite_fg(self, v: bool) -> None:
        if self._composite_fg != v:
            self._composite_fg = v
            self.compositeFgChanged.emit()

    composite_fg = Property(bool, _get_composite_fg, _set_composite_fg, notify=compositeFgChanged)

    # ── Toggle: use_poisson ───────────────────────────────────────────────────
    def _get_use_poisson(self) -> bool:
        return self._use_poisson

    def _set_use_poisson(self, v: bool) -> None:
        if self._use_poisson != v:
            self._use_poisson = v
            self.usePoissonChanged.emit()

    use_poisson = Property(bool, _get_use_poisson, _set_use_poisson, notify=usePoissonChanged)

    # ── Slots ─────────────────────────────────────────────────────────────────

    @Slot(str)
    def browse_input_directory_qml(self, current_path: str = "") -> None:
        directory = QFileDialog.getExistingDirectory(
            None,
            "Select Frame Directory",
            current_path or os.path.expanduser("~"),
            QFileDialog.Option.DontUseNativeDialog,
        )
        if directory:
            self._set_source_dir(directory)

    @Slot()
    def browse_output_qml(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Stitched Output",
            os.path.expanduser("~"),
            "Images (*.jpg *.jpeg *.png *.webp);;All files (*.*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self._set_output_path(path)

    @Slot()
    def start_stitch_qml(self) -> None:
        if self._is_stitching:
            return
        if not self._source_dir or not os.path.isdir(self._source_dir):
            self._set_status("No valid source directory selected.")
            return

        image_paths = _collect_images(self._source_dir)
        if not image_paths:
            self._set_status("No images found in selected directory.")
            return

        out = self._output_path or os.path.join(
            self._source_dir, "stitch_output.jpg"
        )

        pipeline_config = dict(
            use_loftr=self._use_loftr,
            use_birefnet=self._use_birefnet,
            use_ecc=self._use_ecc,
            use_basic=self._use_basic,
            composite_fg=self._composite_fg,
            motion_model="affine" if self._use_apap else "translation",
        )
        if self._use_poisson:
            os.environ["ASP_POISSON_SEAM"] = "1"
        else:
            os.environ.pop("ASP_POISSON_SEAM", None)

        self._worker = StitchWorker(
            image_paths=image_paths,
            output_path=out,
            pipeline_config=pipeline_config,
        )
        self._thread = self._worker

        self._worker.sig_stage.connect(self._on_stage)
        self._worker.sig_log.connect(self._on_log)
        self._worker.sig_finished.connect(self._on_finished)
        self._worker.sig_error.connect(self._on_error)

        self._is_stitching = True
        self.isStitchingChanged.emit()
        self._set_status("Starting…")
        self._log_output = ""
        self.logOutputChanged.emit()

        self._worker.start()

    @Slot()
    def cancel_stitch_qml(self) -> None:
        if self._worker:
            self._worker.cancel()
        self._set_status("Cancelling…")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _set_status(self, text: str) -> None:
        self._status_text = text
        self.statusTextChanged.emit()

    def _on_stage(self, current: int, total: int, label: str) -> None:
        pct = int(current / max(total, 1) * 100)
        if self._progress != pct:
            self._progress = pct
            self.progressChanged.emit()
        self._set_status(f"[{current}/{total}] {label}")

    def _on_log(self, msg: str) -> None:
        self._log_output += msg + "\n"
        self.logOutputChanged.emit()

    def _on_finished(self, output_path: str) -> None:
        self._set_output_path(output_path)
        self._is_stitching = False
        self.isStitchingChanged.emit()
        self._progress = 100
        self.progressChanged.emit()
        self._set_status("Done. Output: " + output_path)
        self._cleanup_thread()

    def _on_error(self, error: str) -> None:
        self._is_stitching = False
        self.isStitchingChanged.emit()
        self._set_status("Error: " + error)
        self._on_log("ERROR: " + error)
        self._cleanup_thread()

    def _cleanup_thread(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
