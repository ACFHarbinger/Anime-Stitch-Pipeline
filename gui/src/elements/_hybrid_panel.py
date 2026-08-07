"""Hybrid Stitch sub-tab: thin wrapper around HybridStitchPanel.

Extracted from ``stitch_tab.py`` -- pure code motion, no logic change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QMessageBox, QWidget

if TYPE_CHECKING:
    from ._stitch_tab_protocol import _StitchTabHost

    # Type-checking-only base: gives mypy visibility into attributes set by
    # StitchTab.__init__ / sibling mixins (see _stitch_tab_protocol.py).
    # Zero runtime effect -- at runtime this mixin still only inherits object.
    class _Base(_StitchTabHost, QWidget): ...
else:
    _Base = object


class _HybridPanelMixin(_Base):
    def _build_hybrid_panel(self) -> QWidget:
        from asp_gui.tabs.stencil import HybridStitchPanel

        self._hybrid_panel = HybridStitchPanel(self)
        self._hybrid_panel.sequence_accepted.connect(self._on_hybrid_sequence_accepted)
        return self._hybrid_panel

    @Slot(list)
    def _on_hybrid_sequence_accepted(self, paths: list[str]):
        """Load the sequence from the Hybrid Stitch panel into the Stitch tab."""
        if not paths:
            return
        self._frame_paths = list(paths)
        self._frame_item_map.clear()
        self._frame_list.clear()
        for p in self._frame_paths:
            self._frame_list.addItem(self._make_frame_item(p))
        self._refresh_pair_combo()
        self._tab_widget.setCurrentIndex(0)
        QMessageBox.information(
            self,
            "Hybrid Stitch",
            f"Loaded {len(paths)} frame(s) into the Stitch tab.\n"
            "Switch to the Stitch tab to run the pipeline.",
        )


__all__ = ["_HybridPanelMixin"]
