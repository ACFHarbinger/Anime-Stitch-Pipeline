"""The Settings dialog flow, as a mixin.

Split out of ``main_window.py`` to keep that file within the repo's 500-LoC
budget (§5.17 / issues #121-#122), following the same mixin-composition
pattern ``annotation_flow.py`` uses. Reads/writes ``self._settings``,
``self.grid``, ``self.scoring_panel``, ``self.session`` and
``self.status_label`` — all owned by ``InspectorWindow``.
"""

from __future__ import annotations

import os

from ..other.settings import save_settings
from .settings_dialog import SettingsDialog
from .theme import apply_theme


class SettingsFlowMixin:
    def _open_settings(self) -> None:
        dialog = SettingsDialog(self._settings, self)
        if dialog.exec() != SettingsDialog.DialogCode.Accepted:
            return
        new_settings = dialog.settings()
        theme_changed = new_settings.theme != self._settings.theme
        dir_changed = new_settings.out_dir != self._settings.out_dir
        self._settings = new_settings
        save_settings(self._settings)
        if theme_changed:
            apply_theme(self, self._settings.theme)
            # The stylesheet cascade repaints everything except the handful of
            # widgets that build their own inline style per instance (score
            # chips, the focused-panel title chip) — those need an explicit
            # nudge, see each widget's own refresh_theme().
            self.grid.refresh_theme()
            self.scoring_panel.refresh_theme()
        if dir_changed and self._settings.out_dir:
            # Only the directory going forward, not the current session's
            # already-loaded evaluations — reopening in the new directory
            # starts (or resumes) a file there rather than migrating this
            # one, so nothing already on disk is silently moved.
            basename = os.path.basename(self.session.out_path)
            self.session.out_path = os.path.join(self._settings.out_dir, basename)
            os.makedirs(self._settings.out_dir, exist_ok=True)
            self.status_label.setText(f"Future saves will go to {self.session.out_path}")
