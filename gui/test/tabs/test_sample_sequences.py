"""Tests for the bundled "Try a sample" frame sequences (roadmap Phase 6.3,
issue #17: "Bundled sample projects"):

- the generator script (``gui/scripts/generate_sample_sequences.py``)
  produces valid, overlapping, synthetic frame sequences;
- ``list_sample_sequences`` discovers the bundled output correctly;
- ``RealHybridStitchPanel``'s "Try a Sample" menu loads a sample sequence
  into the panel exactly like a user's own frames.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from asp_gui.tabs.stencil.hybrid_stitch_panel import RealHybridStitchPanel
from asp_gui.tabs.stencil.sample_sequences import list_sample_sequences
from PIL import Image
from PySide6.QtCore import Qt

pytestmark = pytest.mark.gui

_GUI_ROOT = Path(__file__).resolve().parents[2]
_GENERATOR_PATH = _GUI_ROOT / "scripts" / "generate_sample_sequences.py"


def _load_generator_module():
    spec = importlib.util.spec_from_file_location(
        "generate_sample_sequences", _GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_generator_module()


class TestGenerateSampleSequences:
    def test_generate_sequence_writes_expected_frame_count(self, generator, tmp_path):
        paths = generator.generate_sequence("scroll_a_panels", tmp_path / "seq")
        assert len(paths) == generator._N_FRAMES
        for p in paths:
            assert p.is_file()

    def test_frames_are_valid_decodable_images_of_expected_size(self, generator, tmp_path):
        paths = generator.generate_sequence("scroll_b_lineart", tmp_path / "seq")
        for p in paths:
            with Image.open(p) as img:
                img.verify()
            with Image.open(p) as img:
                assert img.size == (generator._FRAME_W, generator._FRAME_H)

    def test_consecutive_frames_share_real_pixel_overlap(self, generator, tmp_path):
        """Consecutive crops of the same page must actually overlap by the
        configured stride, not just be independently-generated noise."""
        paths = generator.generate_sequence("scroll_c_textgrid", tmp_path / "seq")
        overlap_h = generator._FRAME_H - generator._STRIDE
        assert overlap_h > 0
        for a, b in zip(paths, paths[1:]):
            with Image.open(a) as img_a, Image.open(b) as img_b:
                bottom_of_a = img_a.crop((0, generator._STRIDE, img_a.width, img_a.height))
                top_of_b = img_b.crop((0, 0, img_b.width, overlap_h))
                assert bottom_of_a.tobytes() == top_of_b.tobytes()

    def test_all_bundled_pages_are_registered_and_generate(self, generator, tmp_path):
        for name in generator._PAGE_BUILDERS:
            paths = generator.generate_sequence(name, tmp_path / name)
            assert len(paths) == generator._N_FRAMES

    def test_generate_all_total_size_stays_small(self, generator, tmp_path):
        written = generator.generate_all(tmp_path)
        total_bytes = sum(p.stat().st_size for paths in written.values() for p in paths)
        # Generous ceiling well above the ~60 KiB actually produced -- this
        # guards against someone accidentally bloating the bundled assets,
        # not a tight byte-for-byte assertion.
        assert total_bytes < 2 * 1024 * 1024


class TestListSampleSequences:
    def test_empty_directory_returns_empty_mapping(self, tmp_path):
        assert list_sample_sequences(tmp_path) == {}

    def test_missing_directory_returns_empty_mapping(self, tmp_path):
        assert list_sample_sequences(tmp_path / "does_not_exist") == {}

    def test_discovers_generated_sequences_in_scroll_order(self, generator, tmp_path):
        generator.generate_all(tmp_path)
        found = list_sample_sequences(tmp_path)
        assert len(found) == 3
        for _label, paths in found.items():
            assert paths == sorted(paths)
            assert len(paths) == generator._N_FRAMES

    def test_bundled_repo_samples_are_present(self):
        """The committed gui/resources/samples/ output itself must exist and
        be discoverable -- this is the actual shipped asset, not just
        something the generator can produce on demand."""
        found = list_sample_sequences()
        assert len(found) >= 2
        for _label, paths in found.items():
            assert len(paths) >= 4
            for p in paths:
                assert Path(p).is_file()


class TestHybridStitchPanelTrySample:
    def test_sample_menu_is_populated(self, q_app):
        panel = RealHybridStitchPanel()
        actions = panel._sample_menu.actions()
        assert len(actions) >= 2
        assert all(a.isEnabled() for a in actions)

    def test_tool_tabs_keep_full_labels_with_scroll_buttons(self, q_app):
        panel = RealHybridStitchPanel()
        tab_bar = panel._tools.tabBar()

        assert tab_bar.usesScrollButtons() is True
        assert tab_bar.elideMode() == Qt.TextElideMode.ElideNone

    def test_selecting_a_sample_loads_it_into_the_sequence_sidebar(self, q_app):
        panel = RealHybridStitchPanel()
        samples = list_sample_sequences()
        label, paths = next(iter(samples.items()))
        assert panel._sequence == []

        panel._load_sample_sequence(paths)

        assert panel._sequence == list(paths)
        assert panel._seq_list.count() == len(paths)

    def test_loading_a_sample_does_not_touch_onboarding_state(self, q_app):
        """The sample action must be independent of the onboarding wizard --
        loading a sample is not the same as dismissing the tour."""
        panel = RealHybridStitchPanel()
        samples = list_sample_sequences()
        _label, paths = next(iter(samples.items()))
        panel._load_sample_sequence(paths)
        assert panel._onboarding_wizard is None
