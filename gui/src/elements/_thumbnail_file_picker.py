"""Thumbnail-grid file picker dialog used to add source frames.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

import os
from typing import Dict, List

from PySide6.QtCore import QSize, Qt, QThreadPool, Slot
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
)

from gui.src.windows.settings.app_settings import AppSettings
from gui.src.windows.settings.splitter_persistence import persist_splitter
from ._thumb_workers import _ThumbHub, _ThumbTask


class _ThumbnailFilePicker(QDialog):
    _EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

    def __init__(self, parent=None, start_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Add Source Frames")
        self.resize(960, 640)
        self._current_dir = start_dir or os.path.expanduser("~")
        self._selected_paths: List[str] = []
        self._item_map: Dict[str, QListWidgetItem] = {}
        self._generation = 0
        self._pool = QThreadPool.globalInstance()
        self._hub = _ThumbHub()
        self._hub.loaded.connect(self._on_thumb_loaded)
        self._thumb_size = 128
        self._folder_icon = _ThumbnailFilePicker._make_folder_icon(64)
        self._build_ui()
        self._navigate(self._current_dir)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Navigation bar
        nav = QHBoxLayout()
        btn_up = QPushButton("↑ Up")
        btn_up.setFixedWidth(80)
        btn_up.clicked.connect(self._go_up)
        self._addr_bar = QLineEdit()
        self._addr_bar.returnPressed.connect(
            lambda: self._navigate(self._addr_bar.text())
        )
        nav.addWidget(btn_up)
        nav.addWidget(self._addr_bar)
        layout.addLayout(nav)

        # Sidebar + thumbnail grid
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._sidebar = QListWidget()
        self._sidebar.setMaximumWidth(150)
        self._sidebar.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._sidebar.setFrameShape(QListWidget.Shape.NoFrame)
        self._sidebar.setStyleSheet(
            "QListWidget { background:#1e1e1e; color:#ccc; }"
            "QListWidget::item { padding:5px 8px; color:#ccc; }"
            "QListWidget::item:selected { background:#1e5080; color:#fff; }"
            "QListWidget::item:hover { background:#2a3a4a; }"
        )
        self._populate_sidebar()
        self._sidebar.itemClicked.connect(
            lambda item: self._navigate(item.data(Qt.ItemDataRole.UserRole))
        )
        self._sidebar.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._sidebar.customContextMenuRequested.connect(self._on_sidebar_context_menu)
        splitter.addWidget(self._sidebar)

        self._grid = QListWidget()
        self._grid.setViewMode(QListWidget.ViewMode.IconMode)
        self._grid.setIconSize(QSize(self._thumb_size, self._thumb_size))
        self._grid.setGridSize(QSize(self._thumb_size + 30, self._thumb_size + 40))
        self._grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._grid.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._grid.setMovement(QListWidget.Movement.Static)
        self._grid.setWrapping(True)
        self._grid.setWordWrap(True)
        self._grid.setSpacing(8)
        self._grid.setFrameShape(QListWidget.Shape.NoFrame)
        self._grid.setStyleSheet(
            "QListWidget { background:#1e1e1e; color:#ccc; }"
            "QListWidget::item { color:#ccc; border-radius:4px; padding:2px; }"
            "QListWidget::item:selected { background:#1e5080; color:#fff; }"
            "QListWidget::item:hover { background:#2a3a4a; }"
        )
        self._grid.itemDoubleClicked.connect(self._on_double_click)
        self._grid.itemSelectionChanged.connect(self._update_status)
        self._grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._grid.customContextMenuRequested.connect(self._on_grid_context_menu)
        splitter.addWidget(self._grid)
        splitter.setSizes([150, 800])
        splitter.setStretchFactor(1, 1)

        persist_splitter(splitter, "ThumbnailFilePicker/sidebar")
        layout.addWidget(splitter)

        # Status + icon-size slider + buttons
        bottom = QHBoxLayout()
        self._status_label = QLabel("No files selected")

        size_lbl = QLabel("Icon size:")
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setRange(64, 256)
        self._size_slider.setValue(self._thumb_size)
        self._size_slider.setFixedWidth(120)
        self._size_slider.valueChanged.connect(self._on_size_changed)

        btn_open = QPushButton("Open")
        btn_open.setDefault(True)
        btn_open.clicked.connect(self._accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        bottom.addWidget(self._status_label)
        bottom.addStretch()
        bottom.addWidget(size_lbl)
        bottom.addWidget(self._size_slider)
        bottom.addWidget(btn_open)
        bottom.addWidget(btn_cancel)
        layout.addLayout(bottom)

    def _populate_sidebar(self):
        self._sidebar.clear()
        home = os.path.expanduser("~")
        bookmarks = [
            ("Home", home),
            ("Desktop", os.path.join(home, "Desktop")),
            ("Pictures", os.path.join(home, "Pictures")),
            ("Downloads", os.path.join(home, "Downloads")),
            ("Documents", os.path.join(home, "Documents")),
        ]
        favs = AppSettings.favourite_directories()
        norm_std = {os.path.normpath(p) for _, p in bookmarks if p}
        for fav_path in favs:
            if fav_path and os.path.isdir(fav_path):
                norm_fav = os.path.normpath(fav_path)
                if norm_fav not in norm_std:
                    label = f"⭐ {os.path.basename(norm_fav) or norm_fav}"
                    bookmarks.append((label, norm_fav))

        for label, path in bookmarks:
            if os.path.isdir(path):
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, path)
                self._sidebar.addItem(item)

    def _apply_menu_style(self, menu: QMenu):
        is_dark = AppSettings.get("preferences/theme", "dark") == "dark"
        if is_dark:
            menu.setStyleSheet("""
                QMenu {
                    background-color: #2d2d30;
                    color: white;
                    border: 1px solid #3e3e42;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 12px;
                }
                QMenu::item {
                    padding: 6px 20px;
                }
                QMenu::item:selected {
                    background-color: #00bcd4;
                    color: black;
                }
                QMenu::separator {
                    height: 1px;
                    background-color: #3e3e42;
                    margin: 4px 0px;
                }
            """)
        else:
            menu.setStyleSheet("""
                QMenu {
                    background-color: #ffffff;
                    color: #333;
                    border: 1px solid #ccc;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 12px;
                }
                QMenu::item {
                    padding: 6px 20px;
                }
                QMenu::item:selected {
                    background-color: #007AFF;
                    color: white;
                }
                QMenu::separator {
                    height: 1px;
                    background-color: #ccc;
                    margin: 4px 0px;
                }
            """)

    def _on_sidebar_context_menu(self, pos):
        item = self._sidebar.itemAt(pos)
        if not item:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if not path or not os.path.isdir(path):
            return

        favs = AppSettings.favourite_directories()
        norm_path = os.path.normpath(path)
        norm_favs = [os.path.normpath(f) for f in favs]
        is_fav = norm_path in norm_favs

        menu = QMenu(self._sidebar)
        self._apply_menu_style(menu)
        fav_act = QAction("❌ Remove from Favourites", menu) if is_fav else QAction("⭐ Add to Favourites", menu)
        menu.addAction(fav_act)

        act = menu.exec(self._sidebar.mapToGlobal(pos))
        if act == fav_act:
            if is_fav:
                new_favs = [f for f in favs if os.path.normpath(f) != norm_path]
                AppSettings.set_favourite_directories(new_favs)
                QMessageBox.information(self, "Favourite Removed", f"Removed from favourites:\n{path}")
            else:
                favs.append(path)
                AppSettings.set_favourite_directories(favs)
                QMessageBox.information(self, "Favourite Added", f"Added to favourites:\n{path}")
            self._populate_sidebar()

    def _on_grid_context_menu(self, pos):
        item = self._grid.itemAt(pos)
        if item and item.data(Qt.ItemDataRole.UserRole + 1) == "dir":
            path = item.data(Qt.ItemDataRole.UserRole)
        else:
            path = self._current_dir

        if not path or not os.path.isdir(path):
            return

        favs = AppSettings.favourite_directories()
        norm_path = os.path.normpath(path)
        norm_favs = [os.path.normpath(f) for f in favs]
        is_fav = norm_path in norm_favs

        menu = QMenu(self._grid)
        self._apply_menu_style(menu)
        fav_act = QAction("❌ Remove from Favourites", menu) if is_fav else QAction("⭐ Add to Favourites", menu)
        menu.addAction(fav_act)

        act = menu.exec(self._grid.mapToGlobal(pos))
        if act == fav_act:
            if is_fav:
                new_favs = [f for f in favs if os.path.normpath(f) != norm_path]
                AppSettings.set_favourite_directories(new_favs)
                QMessageBox.information(self, "Favourite Removed", f"Removed from favourites:\n{path}")
            else:
                favs.append(path)
                AppSettings.set_favourite_directories(favs)
                QMessageBox.information(self, "Favourite Added", f"Added to favourites:\n{path}")
            self._populate_sidebar()

    def _navigate(self, path: str):
        path = os.path.normpath(path)
        if not os.path.isdir(path):
            return
        self._generation += 1
        gen = self._generation
        self._grid.clear()
        self._item_map.clear()
        self._current_dir = path
        self._addr_bar.setText(path)

        try:
            entries = sorted(
                os.scandir(path),
                key=lambda e: (not e.is_dir(), e.name.lower()),
            )
        except PermissionError:
            return

        for entry in entries:
            if entry.is_dir() and not entry.name.startswith("."):
                item = QListWidgetItem(self._folder_icon, entry.name)
                item.setData(Qt.ItemDataRole.UserRole, entry.path)
                item.setData(Qt.ItemDataRole.UserRole + 1, "dir")
                self._grid.addItem(item)
            elif os.path.splitext(entry.name)[1].lower() in self._EXTS:
                item = QListWidgetItem(entry.name)
                item.setData(Qt.ItemDataRole.UserRole, entry.path)
                item.setData(Qt.ItemDataRole.UserRole + 1, "file")
                self._grid.addItem(item)
                self._item_map[entry.path] = item
                self._pool.start(
                    _ThumbTask(entry.path, self._thumb_size, gen, self._hub)
                )

    @Slot(str, int, object)
    def _on_thumb_loaded(self, path: str, generation: int, img: QImage):
        if generation != self._generation:
            return
        item = self._item_map.get(path)
        if item and not img.isNull():
            item.setIcon(QIcon(QPixmap.fromImage(img)))

    def _on_double_click(self, item: QListWidgetItem):
        if item.data(Qt.ItemDataRole.UserRole + 1) == "dir":
            self._navigate(item.data(Qt.ItemDataRole.UserRole))

    def _go_up(self):
        parent = os.path.dirname(self._current_dir)
        if parent != self._current_dir:
            self._navigate(parent)

    def _update_status(self):
        n = sum(
            1
            for it in self._grid.selectedItems()
            if it.data(Qt.ItemDataRole.UserRole + 1) == "file"
        )
        self._status_label.setText(
            f"{n} file{'s' if n != 1 else ''} selected" if n else "No files selected"
        )

    def _on_size_changed(self, value: int):
        self._thumb_size = value
        self._grid.setIconSize(QSize(value, value))
        self._grid.setGridSize(QSize(value + 28, value + 36))
        self._navigate(self._current_dir)

    def _accept(self):
        self._selected_paths = [
            it.data(Qt.ItemDataRole.UserRole)
            for it in self._grid.selectedItems()
            if it.data(Qt.ItemDataRole.UserRole + 1) == "file"
        ]
        if self._selected_paths:
            self.accept()

    def selected_paths(self) -> List[str]:
        return self._selected_paths

    @staticmethod
    def _make_folder_icon(size: int = 64) -> QIcon:
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        amber = QColor(240, 185, 60)
        dark_amber = QColor(200, 145, 20)
        p.setPen(QPen(dark_amber, 1))
        p.setBrush(QBrush(amber))
        tab_h = size // 8
        body_top = size // 4
        p.drawRoundedRect(2, body_top, size // 3, tab_h, 3, 3)
        p.drawRoundedRect(
            2, body_top + tab_h - 2, size - 4, size - body_top - tab_h - 2, 5, 5
        )
        p.end()
        return QIcon(pm)


__all__ = ["_ThumbnailFilePicker"]
