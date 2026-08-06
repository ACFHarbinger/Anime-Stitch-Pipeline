"""SettingsDialog: the gear-icon menu — default save directory + theme
toggle (issue #123 followup item 9).

The directory browser passes ``DontUseNativeDialog`` per this repo's
project-wide convention (see ``project_tab_widget_constraints`` in the
Image-Toolkit memory): a native GTK file dialog alongside a live JPype JVM
has repeatedly SIGSEGV'd elsewhere in this app, and there is no reason to
reintroduce that risk here for a plain directory picker.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ..constants.user_interface import THEMES
from ..other.settings import AppSettings


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(440)
        self._result = AppSettings(out_dir=settings.out_dir, theme=settings.theme)

        form = QFormLayout()
        form.setSpacing(8)

        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit(settings.out_dir or "")
        self._dir_edit.setPlaceholderText("data/benchmarks (repo default)")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        dir_row.addWidget(self._dir_edit, stretch=1)
        dir_row.addWidget(browse_btn)
        form.addRow("Save feedback to:", dir_row)

        self._theme_combo = QComboBox()
        for key, label in THEMES:
            self._theme_combo.addItem(label, key)
        index = self._theme_combo.findData(settings.theme)
        self._theme_combo.setCurrentIndex(max(0, index))
        form.addRow("Theme:", self._theme_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Choose a default save directory",
            self._dir_edit.text() or "",
            QFileDialog.Option.DontUseNativeDialog,
        )
        if chosen:
            self._dir_edit.setText(chosen)

    def settings(self) -> AppSettings:
        return AppSettings(out_dir=self._dir_edit.text().strip() or None, theme=self._theme_combo.currentData())
