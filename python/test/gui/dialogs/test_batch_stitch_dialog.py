"""Tests for BatchStitchWorker and BatchStitchDialog (roadmap §4.1 Option A)."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
from gui.src.components.dialogs.batch_stitch_dialog import BatchStitchDialog
from gui.src.helpers.animation.batch_stitch_worker import BatchStitchWorker

pytestmark = pytest.mark.gui


def _make_batch_dir(tmp_path, names, n_images=3):
    from PIL import Image

    for name in names:
        seq_dir = tmp_path / name
        seq_dir.mkdir()
        for i in range(n_images):
            Image.new("RGB", (16, 16), color=(i * 30, 0, 0)).save(
                seq_dir / f"frame_{i:02d}.png"
            )
    return str(tmp_path)


class TestBatchStitchWorker:
    def test_collects_and_stitches_each_subdirectory(self, tmp_path, q_app):
        batch_dir = _make_batch_dir(tmp_path, ["seq_a", "seq_b"])
        worker = BatchStitchWorker(batch_dir=batch_dir, renderer="median")

        started = []
        finished = []
        batch_done = []
        with patch(
            "gui.src.helpers.animation.batch_stitch_worker._run_single_stitch",
            return_value=True,
        ) as mock_stitch:
            worker.sig_item_started.connect(lambda n, i, t: started.append((n, i, t)))
            worker.sig_item_finished.connect(lambda n, s: finished.append((n, s)))
            worker.sig_batch_finished.connect(
                lambda d, s, f: batch_done.append((d, s, f))
            )
            worker.run()  # run synchronously in-test, not as a real QThread

        assert mock_stitch.call_count == 2
        assert [n for n, _, _ in started] == ["seq_a", "seq_b"]
        assert finished == [("seq_a", "done"), ("seq_b", "done")]
        assert batch_done == [(2, 0, 0)]

    def test_resume_skips_sequences_with_existing_output(self, tmp_path, q_app):
        batch_dir = _make_batch_dir(tmp_path, ["seq_a", "seq_b"])
        # Pre-create seq_a's output so resume should skip it.
        open(os.path.join(batch_dir, "seq_a", "seq_a_stitched.png"), "w").close()

        worker = BatchStitchWorker(batch_dir=batch_dir, renderer="median", resume=True)
        finished = []
        with patch(
            "gui.src.helpers.animation.batch_stitch_worker._run_single_stitch",
            return_value=True,
        ) as mock_stitch:
            worker.sig_item_finished.connect(lambda n, s: finished.append((n, s)))
            worker.run()

        assert mock_stitch.call_count == 1  # only seq_b actually stitched
        assert ("seq_a", "skipped") in finished
        assert ("seq_b", "done") in finished

    def test_persists_progress_json_matching_cli_format(self, tmp_path, q_app):
        batch_dir = _make_batch_dir(tmp_path, ["seq_a"])
        worker = BatchStitchWorker(batch_dir=batch_dir, renderer="median")
        with patch(
            "gui.src.helpers.animation.batch_stitch_worker._run_single_stitch",
            return_value=True,
        ):
            worker.run()

        progress_path = os.path.join(batch_dir, ".stitch_progress.json")
        assert os.path.isfile(progress_path)
        with open(progress_path) as f:
            data = json.load(f)
        assert data == {"seq_a": "done"}

    def test_cancel_stops_before_remaining_items(self, tmp_path, q_app):
        batch_dir = _make_batch_dir(tmp_path, ["seq_a", "seq_b", "seq_c"])
        worker = BatchStitchWorker(batch_dir=batch_dir, renderer="median")

        def _stitch_and_cancel(*_args, **_kwargs):
            worker.cancel()
            return True

        finished = []
        with patch(
            "gui.src.helpers.animation.batch_stitch_worker._run_single_stitch",
            side_effect=_stitch_and_cancel,
        ) as mock_stitch:
            worker.sig_item_finished.connect(lambda n, s: finished.append((n, s)))
            worker.run()

        # Cancelled after the first item's stitch call requests it -- only
        # one sequence should have actually been processed.
        assert mock_stitch.call_count == 1
        assert finished == [("seq_a", "done")]

    def test_empty_directory_reports_zero_items(self, tmp_path, q_app):
        worker = BatchStitchWorker(batch_dir=str(tmp_path), renderer="median")
        batch_done = []
        worker.sig_batch_finished.connect(lambda d, s, f: batch_done.append((d, s, f)))
        worker.run()
        assert batch_done == [(0, 0, 0)]

    def test_sequence_with_too_few_images_is_skipped(self, tmp_path, q_app):
        seq_dir = tmp_path / "seq_a"
        seq_dir.mkdir()
        from PIL import Image

        Image.new("RGB", (16, 16)).save(seq_dir / "only_one.png")

        worker = BatchStitchWorker(batch_dir=str(tmp_path), renderer="median")
        finished = []
        with patch(
            "gui.src.helpers.animation.batch_stitch_worker._run_single_stitch"
        ) as mock_stitch:
            worker.sig_item_finished.connect(lambda n, s: finished.append((n, s)))
            worker.run()

        mock_stitch.assert_not_called()
        assert finished == [("seq_a", "skipped")]


class TestBatchStitchDialog:
    def test_constructs_with_expected_defaults(self, q_app):
        dlg = BatchStitchDialog()
        assert dlg.windowTitle() == "Batch Stitch"
        assert dlg.suffix_edit.text() == "_stitched"
        assert dlg.resume_check.isChecked() is False
        assert dlg.cancel_btn.isEnabled() is False

    def test_start_without_directory_shows_status_message(self, q_app):
        dlg = BatchStitchDialog()
        dlg.dir_edit.setText("")
        dlg._start()
        assert "root directory" in dlg.status_label.text().lower()

    def test_item_started_updates_progress_bar_and_list(self, q_app):
        dlg = BatchStitchDialog()
        dlg._on_item_started("seq_a", 1, 3)
        assert dlg.progress_bar.maximum() == 3
        assert dlg.progress_list.count() == 1
        assert "seq_a" in dlg.progress_list.item(0).text()

    def test_item_finished_updates_existing_list_entry(self, q_app):
        dlg = BatchStitchDialog()
        dlg._on_item_started("seq_a", 1, 2)
        dlg._on_item_finished("seq_a", "done")
        assert "done" in dlg.progress_list.item(0).text()

    def test_batch_finished_reenables_start_and_sets_summary(self, q_app):
        dlg = BatchStitchDialog()
        dlg.start_btn.setEnabled(False)
        dlg.cancel_btn.setEnabled(True)
        dlg._on_batch_finished(2, 1, 0)
        assert dlg.start_btn.isEnabled() is True
        assert dlg.cancel_btn.isEnabled() is False
        assert "2 done" in dlg.status_label.text()
