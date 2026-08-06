from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ...helpers.animation.batch_stitch_worker import BatchStitchWorker
from gui.src.constants.components import _STATUS_ICON


class BatchStitchDialog(QDialog):
    """
    Roadmap §4.1 Option A -- directory-level batch stitching with a
    per-item progress list. Scans a root directory's immediate
    subdirectories (each one a distinct frame group) and runs
    `AnimeStitchPipeline` on each via `BatchStitchWorker`, reusing the same
    `_collect_image_paths`/`_run_single_stitch` helpers the CLI's
    `stitch --batch-dir` mode already validated (§4.1 Option C, done) --
    including the same `.stitch_progress.json` resume convention (Option E,
    done), so a batch interrupted from one entry point can be resumed from
    the other.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Stitch")
        self.resize(520, 480)
        self._worker: Optional[BatchStitchWorker] = None
        self._items: dict = {}

        layout = QVBoxLayout(self)

        dir_row = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText(
            "Root directory containing one subdirectory per frame group…"
        )
        dir_row.addWidget(self.dir_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse_btn)
        layout.addLayout(dir_row)

        form = QFormLayout()

        self.renderer_combo = QComboBox()
        self.renderer_combo.addItems(["median", "average", "blend"])
        form.addRow("Renderer:", self.renderer_combo)

        self.suffix_edit = QLineEdit("_stitched")
        form.addRow("Output suffix:", self.suffix_edit)

        self.resume_check = QCheckBox(
            "Resume: skip sequences already stitched (uses .stitch_progress.json)"
        )
        form.addRow(self.resume_check)

        layout.addLayout(form)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.progress_list = QListWidget()
        layout.addWidget(self.progress_list, 1)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._start)
        btn_row.addWidget(self.start_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(
            self.reject
        )
        layout.addWidget(buttons)

    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select root directory",
            self.dir_edit.text(),
            QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self.dir_edit.setText(path)

    def _start(self):
        batch_dir = self.dir_edit.text().strip()
        if not batch_dir:
            self.status_label.setText("Pick a root directory first.")
            return

        self.progress_list.clear()
        self._items.clear()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_label.setText("Running…")

        self._worker = BatchStitchWorker(
            batch_dir=batch_dir,
            renderer=self.renderer_combo.currentText(),
            output_suffix=self.suffix_edit.text().strip() or "_stitched",
            resume=self.resume_check.isChecked(),
        )
        self._worker.sig_item_started.connect(self._on_item_started)
        self._worker.sig_item_finished.connect(self._on_item_finished)
        self._worker.sig_batch_finished.connect(self._on_batch_finished)
        self._worker.sig_log.connect(self.status_label.setText)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _cancel(self):
        if self._worker is not None:
            self.cancel_btn.setEnabled(False)
            self.status_label.setText("Cancelling…")
            self._worker.cancel()

    def _on_item_started(self, seq_name: str, index: int, total: int):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(index - 1)
        item = QListWidgetItem(f"{_STATUS_ICON['running']} {seq_name}")
        item.setData(Qt.ItemDataRole.UserRole, seq_name)
        self.progress_list.addItem(item)
        self._items[seq_name] = item
        self.status_label.setText(f"[{index}/{total}] {seq_name}")

    def _on_item_finished(self, seq_name: str, status: str):
        item = self._items.get(seq_name)
        if item is not None:
            item.setText(f"{_STATUS_ICON.get(status, '')} {seq_name} ({status})")
        self.progress_bar.setValue(self.progress_bar.value() + 1)

    def _on_batch_finished(self, done: int, skipped: int, failed: int):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText(
            f"Batch complete: {done} done, {skipped} skipped, {failed} failed."
        )

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            # Deliberately unbounded: cancel() only takes effect between
            # items (a single AnimeStitchPipeline.run() call, ~90s typical,
            # isn't interruptible mid-stage), and the worker's signals are
            # connected to this dialog's own widgets. A fixed timeout here
            # would let the worker outlive the dialog and later emit into
            # already-destroyed widgets -- the same class of bug fixed in
            # the gallery base classes this same session (see
            # .agent/cache/gallery_crash_deleteorphaned_2026-07-27.md):
            # a bounded wait that expires before the worker actually stops
            # doesn't prevent the race, it just makes it rarer.
            self._worker.wait()
        super().closeEvent(event)
