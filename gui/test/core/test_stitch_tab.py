from unittest.mock import patch

from asp_gui.elements import StitchTab
from PySide6.QtWidgets import QScrollArea


class TestStitchTabFrameCounter:
    def test_pipeline_stage_labels_fit_at_minimum_width(self, q_app):
        with patch("asp_gui.elements._stitch_execution.StitchWorker"):
            host = QScrollArea()
            host.setWidgetResizable(True)
            tab = StitchTab()
            host.setWidget(tab)
            host.resize(800, 700)
            host.show()
            q_app.processEvents()

            for checkbox in (
                tab._cb_basic,
                tab._cb_birefnet,
                tab._cb_loftr,
                tab._cb_ecc,
                tab._cb_composite_fg,
            ):
                assert checkbox.width() >= checkbox.sizeHint().width()

    def test_frame_counter_initialization(self, q_app):
        with patch("asp_gui.elements._stitch_execution.StitchWorker"):
            tab = StitchTab()
            assert hasattr(tab, "_lbl_frame_count")
            assert tab._lbl_frame_count.text() == "Frames: 0"

    def test_frame_counter_update_on_add_and_remove(self, q_app, tmp_path):
        with patch("asp_gui.elements._stitch_execution.StitchWorker"), \
             patch.object(StitchTab, "_on_pair_changed"):
            tab = StitchTab()
            f1 = str(tmp_path / "frame1.png")
            f2 = str(tmp_path / "frame2.png")

            tab._frame_paths = [f1, f2]
            tab._refresh_pair_combo()

            assert tab._lbl_frame_count.text() == "Frames: 2"

            tab._frame_paths.pop()
            tab._refresh_pair_combo()

            assert tab._lbl_frame_count.text() == "Frames: 1"
