from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
)
from gui.src.constants.elements import _CHECKPOINT_LABELS, _DEFAULT_SESSION_DIR


def _list_sessions(session_dir: Path) -> List[Path]:
    """Return .json session files in session_dir sorted newest-first."""
    if not session_dir.is_dir():
        return []
    paths = list(session_dir.glob("*.json"))
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths


def _load_session_meta(path: Path) -> Optional[dict]:
    """Load session JSON without decoding numpy arrays. Returns None on error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _format_session_info(data: dict, path: Path) -> str:
    """Build a human-readable summary of a session override dict."""
    ts = data.get("timestamp", 0.0)
    dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "—"
    version = data.get("version", "?")
    checkpoints: dict = data.get("checkpoints", {})
    size_kb = path.stat().st_size / 1024 if path.exists() else 0.0

    lines = [
        f"File:         {path.name}",
        f"Timestamp:    {dt_str}",
        f"Version:      {version}",
        f"Size:         {size_kb:.1f} KB",
        f"Checkpoints:  {len(checkpoints)}",
        "",
        "Checkpoint overrides:",
    ]
    if not checkpoints:
        lines.append("  (none)")
    for event, override in checkpoints.items():
        label = _CHECKPOINT_LABELS.get(event, event)
        if isinstance(override, dict):
            keys = [k for k in override if not str(k).startswith("_")]
            key_summary = ", ".join(keys[:6])
            if len(keys) > 6:
                key_summary += f" … (+{len(keys) - 6})"
            lines.append(f"  [{label}]  {key_summary or '(empty)'}")
        else:
            lines.append(f"  [{label}]  (non-dict value)")
    return "\n".join(lines)


class HITLSessionViewerDialog(QDialog):
    """Browse, inspect, delete, and load HITL session files for replay."""

    def __init__(self, session_dir: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HITL Session Browser")
        self.resize(780, 480)

        self._session_dir: Path = session_dir or _DEFAULT_SESSION_DIR
        self._paths: List[Path] = []
        self._selected_path: Optional[str] = None

        self._build_ui()
        self._refresh()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        header = QLabel(
            f"Session directory: <tt>{self._session_dir}</tt>"
        )
        header.setTextFormat(Qt.TextFormat.RichText)
        header.setWordWrap(True)
        root.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: session list
        left = QVBoxLayout()
        left.setSpacing(4)
        self._list = QListWidget()
        self._list.setMinimumWidth(260)
        self._list.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._list.currentRowChanged.connect(self._on_selection_changed)
        left.addWidget(self._list)

        btn_row = QHBoxLayout()
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self._refresh)
        self._btn_delete = QPushButton("Delete")
        self._btn_delete.setEnabled(False)
        self._btn_delete.clicked.connect(self._on_delete)
        self._btn_export = QPushButton("Export…")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._on_export)
        btn_row.addWidget(self._btn_refresh)
        btn_row.addWidget(self._btn_delete)
        btn_row.addWidget(self._btn_export)
        left.addLayout(btn_row)

        left_widget = _layout_to_widget(left)
        splitter.addWidget(left_widget)

        # Right: detail view
        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setPlaceholderText("Select a session to view details.")
        self._detail.setFontFamily("Monospace")
        splitter.addWidget(self._detail)

        splitter.setSizes([270, 480])
        root.addWidget(splitter, stretch=1)

        # Bottom buttons
        bottom = QHBoxLayout()
        self._btn_load = QPushButton("Load for Replay")
        self._btn_load.setDefault(True)
        self._btn_load.setEnabled(False)
        self._btn_load.clicked.connect(self._on_load)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        bottom.addStretch()
        bottom.addWidget(self._btn_load)
        bottom.addWidget(btn_close)
        root.addLayout(bottom)

    # ------------------------------------------------------------------
    # List management
    # ------------------------------------------------------------------

    def _refresh(self):
        self._paths = _list_sessions(self._session_dir)
        self._list.blockSignals(True)
        self._list.clear()
        for p in self._paths:
            meta = _load_session_meta(p)
            if meta is not None:
                ts = meta.get("timestamp", 0.0)
                dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "—"
                n_ckpt = len(meta.get("checkpoints", {}))
                size_kb = p.stat().st_size / 1024
                label = f"{p.name}\n{dt_str} · {n_ckpt} checkpoint(s) · {size_kb:.1f} KB"
            else:
                label = f"{p.name}\n(could not parse)"
            item = QListWidgetItem(label)
            self._list.addItem(item)
        self._list.blockSignals(False)

        if not self._paths:
            self._detail.setPlainText("No session files found.")
        self._set_buttons_enabled(False)

    def _on_selection_changed(self, row: int):
        if not (0 <= row < len(self._paths)):
            self._set_buttons_enabled(False)
            self._detail.clear()
            return
        path = self._paths[row]
        meta = _load_session_meta(path)
        if meta is not None:
            self._detail.setPlainText(_format_session_info(meta, path))
        else:
            self._detail.setPlainText(f"Could not parse {path.name}.")
        self._set_buttons_enabled(True)

    def _set_buttons_enabled(self, enabled: bool):
        self._btn_delete.setEnabled(enabled)
        self._btn_export.setEnabled(enabled)
        self._btn_load.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _current_path(self) -> Optional[Path]:
        row = self._list.currentRow()
        if 0 <= row < len(self._paths):
            return self._paths[row]
        return None

    def _on_load(self):
        path = self._current_path()
        if path is None:
            return
        self._selected_path = str(path)
        self.accept()

    def _on_delete(self):
        path = self._current_path()
        if path is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete Session",
            f"Delete session file?\n\n{path.name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            try:
                path.unlink()
            except Exception as exc:
                QMessageBox.warning(self, "Delete Failed", str(exc))
            self._refresh()

    def _on_export(self):
        path = self._current_path()
        if path is None:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Export Session",
            str(Path.home() / path.name),
            "JSON files (*.json);;All files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if dest:
            try:
                shutil.copy2(str(path), dest)
            except Exception as exc:
                QMessageBox.warning(self, "Export Failed", str(exc))

    # ------------------------------------------------------------------
    # Result accessor
    # ------------------------------------------------------------------

    def selected_path(self) -> Optional[str]:
        """Returns the path chosen via 'Load for Replay', or None."""
        return self._selected_path


def _layout_to_widget(layout):
    """Wrap a QLayout in a plain QWidget (helper to add layouts to splitter)."""
    from PySide6.QtWidgets import QWidget
    w = QWidget()
    w.setLayout(layout)
    return w
