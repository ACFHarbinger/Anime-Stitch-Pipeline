"""Persisted inspector preferences: default save directory and chrome theme.

Kept separate from ``EvaluationSession`` (which owns one evaluation *pass*)
because these are cross-session, cross-corpus preferences — the theme and the
"where do I save feedback" default should survive between separate ``just
asp-benchmark-assess`` invocations against different data directories, unlike
the queue/history state in ``session.py``.
"""

from __future__ import annotations

import dataclasses
import json
import os

from ..constants.user_interface import THEME_DARK, THEMES

_THEME_KEYS = {key for key, _ in THEMES}


@dataclasses.dataclass
class AppSettings:
    out_dir: str | None = None  # None => bench_eval_dispatch's built-in default
    theme: str = THEME_DARK

    def to_dict(self) -> dict:
        return {"out_dir": self.out_dir, "theme": self.theme}

    @staticmethod
    def from_dict(d: dict) -> AppSettings:
        theme = d.get("theme", THEME_DARK)
        return AppSettings(
            out_dir=d.get("out_dir") or None,
            theme=theme if theme in _THEME_KEYS else THEME_DARK,
        )


def default_settings_path() -> str:
    config_dir = os.path.join(os.path.expanduser("~"), ".config", "image-toolkit")
    return os.path.join(config_dir, "asp_eval_settings.json")


def load_settings(path: str | None = None) -> AppSettings:
    path = path or default_settings_path()
    if not os.path.exists(path):
        return AppSettings()
    try:
        with open(path, encoding="utf-8") as fh:
            return AppSettings.from_dict(json.load(fh))
    except (OSError, ValueError):
        # A corrupt or unreadable prefs file shouldn't block the tool from
        # opening — fall back to defaults, same as a first run.
        return AppSettings()


def save_settings(settings: AppSettings, path: str | None = None) -> None:
    path = path or default_settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(settings.to_dict(), fh, indent=2)
