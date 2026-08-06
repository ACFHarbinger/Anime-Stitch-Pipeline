from unittest.mock import patch

from asp_gui.elements.stitch_tab import StitchTab


class TestStitchTabFrameCounter:
    def test_frame_counter_initialization(self, q_app):
        with patch("asp_gui.elements.stitch_tab._stitch_execution.StitchWorker"):
            tab = StitchTab()
            assert hasattr(tab, "_lbl_frame_count")
            assert tab._lbl_frame_count.text() == "Frames: 0"

    def test_frame_counter_update_on_add_and_remove(self, q_app, tmp_path):
        with patch("asp_gui.elements.stitch_tab._stitch_execution.StitchWorker"), \
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
