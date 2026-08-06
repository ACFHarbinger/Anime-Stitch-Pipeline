"""Tests for the inspector's persisted preferences (issue #123 followup item 9):
default save directory and dark/light theme, plus the Settings dialog that
edits them.

``AppSettings``/``load_settings``/``save_settings`` have no Qt import, so the
round-trip and fallback behaviour is asserted directly; the dialog tests run
under the same offscreen QPA convention as the rest of this package.
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _repo_root)

from backend.benchmark.evaluation.constants.user_interface import (  # noqa: E402
    THEME_DARK,
    THEME_LIGHT,
)
from backend.benchmark.evaluation.other.settings import (  # noqa: E402
    AppSettings,
    load_settings,
    save_settings,
)

# ---------------------------------------------------------------------------
# AppSettings / load_settings / save_settings — no Qt needed
# ---------------------------------------------------------------------------


def test_defaults_are_dark_theme_and_no_out_dir():
    settings = AppSettings()
    assert settings.theme == THEME_DARK
    assert settings.out_dir is None


def test_missing_file_returns_defaults(tmp_path):
    missing = str(tmp_path / "does_not_exist.json")
    settings = load_settings(missing)
    assert settings == AppSettings()


def test_save_then_load_round_trips(tmp_path):
    path = str(tmp_path / "settings.json")
    save_settings(AppSettings(out_dir="/some/dir", theme=THEME_LIGHT), path)
    loaded = load_settings(path)
    assert loaded.out_dir == "/some/dir"
    assert loaded.theme == THEME_LIGHT


def test_save_creates_missing_parent_directories(tmp_path):
    path = str(tmp_path / "nested" / "config" / "settings.json")
    save_settings(AppSettings(theme=THEME_LIGHT), path)
    assert os.path.exists(path)
    assert load_settings(path).theme == THEME_LIGHT


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not valid json")
    assert load_settings(str(path)) == AppSettings()


def test_unknown_theme_in_file_falls_back_to_dark(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"out_dir": null, "theme": "neon"}')
    assert load_settings(str(path)).theme == THEME_DARK


def test_blank_out_dir_normalizes_to_none():
    assert AppSettings.from_dict({"out_dir": "", "theme": THEME_DARK}).out_dir is None


# ---------------------------------------------------------------------------
# SettingsDialog
# ---------------------------------------------------------------------------

pytest.importorskip("PySide6", reason="the settings dialog needs PySide6")

from backend.benchmark.evaluation.ui.settings_dialog import SettingsDialog  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_dialog_starts_pre_filled_from_the_given_settings(qapp):
    dialog = SettingsDialog(AppSettings(out_dir="/home/user/out", theme=THEME_LIGHT))
    assert dialog._dir_edit.text() == "/home/user/out"
    assert dialog._theme_combo.currentData() == THEME_LIGHT
    dialog.deleteLater()


def test_dialog_settings_reflects_edits_without_needing_to_accept(qapp):
    dialog = SettingsDialog(AppSettings())
    dialog._dir_edit.setText("/tmp/somewhere")
    dialog._theme_combo.setCurrentIndex(dialog._theme_combo.findData(THEME_LIGHT))
    result = dialog.settings()
    assert result.out_dir == "/tmp/somewhere"
    assert result.theme == THEME_LIGHT
    dialog.deleteLater()


def test_dialog_blank_directory_normalizes_to_none(qapp):
    dialog = SettingsDialog(AppSettings(out_dir="/was/set"))
    dialog._dir_edit.setText("   ")
    assert dialog.settings().out_dir is None
    dialog.deleteLater()


def test_dialog_does_not_mutate_the_settings_object_it_was_given(qapp):
    original = AppSettings(out_dir="/keep/me", theme=THEME_DARK)
    dialog = SettingsDialog(original)
    dialog._dir_edit.setText("/changed")
    dialog._theme_combo.setCurrentIndex(dialog._theme_combo.findData(THEME_LIGHT))
    dialog.settings()
    assert original.out_dir == "/keep/me"
    assert original.theme == THEME_DARK
    dialog.deleteLater()
