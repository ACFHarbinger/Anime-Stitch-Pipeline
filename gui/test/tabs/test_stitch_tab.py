import json
import math
import os
from unittest.mock import patch

import pytest
from asp_gui.elements.stitch_tab import StitchTab
from asp_gui.elements.stitch_tab.dialog.canvas_layout_inspector_dialog import (
    CanvasLayoutInspectorDialog,
    _canvas_frame_corners,
    parse_canvas_json,
)
from asp_gui.elements.stitch_tab.dialog.edge_graph_inspector_dialog import (
    EdgeGraphInspectorDialog,
    _edge_graph_node_positions,
    parse_edge_json,
)

pytestmark = pytest.mark.gui


class TestStitchTabBrowseOutput:
    def test_browse_output_starts_at_last_selected_source_directory(self, q_app):
        # Create StitchTab instance
        tab = StitchTab()

        # Setup mock frame paths mimicking added source frames
        tab._frame_paths = [
            "/home/user/pictures/frame1.png",
            "/home/user/downloads/frame2.png",
        ]

        # Mock QFileDialog.getSaveFileName to avoid popping up UI
        with patch(
            "asp_gui.elements.stitch_tab._stitch_execution.QFileDialog.getSaveFileName"
        ) as mock_save_dialog:
            mock_save_dialog.return_value = (
                "/home/user/downloads/my_panorama.png",
                "Images (*.png *.webp *.jpg)",
            )

            tab._browse_output()

            # Assert that getSaveFileName was called
            mock_save_dialog.assert_called_once()

            # Check the third positional argument (default_file) passed to QFileDialog.getSaveFileName
            # It should start in the directory of the last frame path: "/home/user/downloads"
            args, kwargs = mock_save_dialog.call_args
            assert args[2] == os.path.normpath("/home/user/downloads/panorama.png")

            # Verify the output path was set to the mock returned path
            assert tab._output_path.text() == "/home/user/downloads/my_panorama.png"

    def test_browse_output_preserves_existing_filename(self, q_app):
        tab = StitchTab()
        tab._frame_paths = [
            "/home/user/pictures/frame1.png",
            "/home/user/downloads/frame2.png",
        ]
        tab._output_path.setText("my_custom_panorama.png")

        with patch(
            "asp_gui.elements.stitch_tab._stitch_execution.QFileDialog.getSaveFileName"
        ) as mock_save_dialog:
            mock_save_dialog.return_value = ("", "")

            tab._browse_output()

            args, kwargs = mock_save_dialog.call_args
            assert args[2] == os.path.normpath(
                "/home/user/downloads/my_custom_panorama.png"
            )


# ---------------------------------------------------------------------------
# §2.2 Edge Graph Inspector tests
# ---------------------------------------------------------------------------


class TestParseEdgeJson:
    def test_valid_fixture(self, tmp_path):
        data = [
            {"i": 0, "j": 1, "dx": -50.0, "dy": 2.5, "conf": 0.82, "method": "loftr"},
            {"i": 1, "j": 2, "dx": -48.0, "dy": 1.0, "conf": 0.75, "method": "loftr"},
        ]
        p = tmp_path / "edges.json"
        p.write_text(json.dumps(data))
        result = parse_edge_json(str(p))
        assert len(result) == 2
        assert result[0]["i"] == 0
        assert result[0]["j"] == 1
        assert abs(result[0]["conf"] - 0.82) < 1e-6

    def test_missing_optional_fields_filled_with_defaults(self, tmp_path):
        data = [{"i": 0, "j": 1}]
        p = tmp_path / "edges.json"
        p.write_text(json.dumps(data))
        result = parse_edge_json(str(p))
        assert result[0]["dx"] == 0.0
        assert result[0]["dy"] == 0.0
        assert result[0]["conf"] == 0.0
        assert result[0]["method"] == "?"

    def test_records_without_i_or_j_are_skipped(self, tmp_path):
        data = [
            {"i": 0, "j": 1, "conf": 0.9},
            {"x": 5, "y": 3},
            {"i": 2},
        ]
        p = tmp_path / "edges.json"
        p.write_text(json.dumps(data))
        result = parse_edge_json(str(p))
        assert len(result) == 1

    def test_empty_array_returns_empty_list(self, tmp_path):
        p = tmp_path / "edges.json"
        p.write_text("[]")
        assert parse_edge_json(str(p)) == []


class TestEdgeGraphNodePositions:
    def test_zero_nodes_returns_empty(self):
        assert _edge_graph_node_positions(0) == []

    def test_single_node_at_origin(self):
        pos = _edge_graph_node_positions(1)
        assert len(pos) == 1
        assert pos[0] == (0.0, 0.0)

    def test_all_nodes_equidistant_from_centre(self):
        pos = _edge_graph_node_positions(6, radius=100.0)
        assert len(pos) == 6
        for x, y in pos:
            dist = math.sqrt(x**2 + y**2)
            assert abs(dist - 100.0) < 1e-6

    def test_first_node_at_twelve_oclock(self):
        pos = _edge_graph_node_positions(4, radius=100.0)
        x0, y0 = pos[0]
        assert abs(x0) < 1e-6
        assert abs(y0 + 100.0) < 1e-6


class TestEdgeGraphInspectorDialog:
    def test_dialog_populates_table_and_stats(self, q_app):
        edges = [
            {"i": 0, "j": 1, "dx": -50.0, "dy": 0.0, "conf": 0.8, "method": "loftr"},
            {"i": 1, "j": 2, "dx": -45.0, "dy": 1.0, "conf": 0.3, "method": "loftr"},
        ]
        dlg = EdgeGraphInspectorDialog(
            edges=edges, frame_paths=["a.png", "b.png", "c.png"]
        )
        assert dlg._table.rowCount() == 2
        assert "3 frames" in dlg._stats_label.text()
        assert "2 edges" in dlg._stats_label.text()
        assert "1 low-conf" in dlg._stats_label.text()

    def test_table_sorted_worst_first(self, q_app):
        edges = [
            {"i": 0, "j": 1, "dx": 0.0, "dy": 0.0, "conf": 0.9, "method": "loftr"},
            {"i": 1, "j": 2, "dx": 0.0, "dy": 0.0, "conf": 0.2, "method": "loftr"},
        ]
        dlg = EdgeGraphInspectorDialog(edges=edges)
        assert dlg._table.item(0, 2).text() == "0.200" # pyrefly: ignore[missing-attribute]
        assert dlg._table.item(1, 2).text() == "0.900" # pyrefly: ignore[missing-attribute]

    def test_empty_edges_shows_no_data_message(self, q_app):
        dlg = EdgeGraphInspectorDialog(edges=[])
        assert dlg._table.rowCount() == 0
        assert "No edges" in dlg._stats_label.text()


class TestParseCanvasJson:
    def test_valid_fixture_parses_all_fields(self, tmp_path):
        data = {
            "canvas_h": 800,
            "canvas_w": 2400,
            "frame_h": 400,
            "frame_w": 600,
            "T_global": [12.0, 5.0],
            "affines_final": [
                [[1.0, 0.0, 12.0], [0.0, 1.0, 5.0]],
                [[1.0, 0.0, 612.0], [0.0, 1.0, 5.0]],
            ],
        }
        p = tmp_path / "canvas.json"
        p.write_text(json.dumps(data))
        result = parse_canvas_json(str(p))
        assert result["canvas_h"] == 800
        assert result["canvas_w"] == 2400
        assert result["frame_h"] == 400
        assert result["frame_w"] == 600
        assert result["T_global"] == [12.0, 5.0]
        assert len(result["affines_final"]) == 2
        assert result["affines_final"][1][0][2] == 612.0

    def test_missing_frame_dimensions_default_to_zero(self, tmp_path):
        data = {
            "canvas_h": 800,
            "canvas_w": 2400,
            "T_global": [0.0, 0.0],
            "affines_final": [],
        }
        p = tmp_path / "canvas_no_dims.json"
        p.write_text(json.dumps(data))
        result = parse_canvas_json(str(p))
        assert result["frame_h"] == 0
        assert result["frame_w"] == 0

    def test_affines_parsed_as_float_lists(self, tmp_path):
        data = {
            "canvas_h": 100,
            "canvas_w": 100,
            "frame_h": 50,
            "frame_w": 50,
            "T_global": [0.0, 0.0],
            "affines_final": [[[1, 0, 0], [0, 1, 0]]],
        }
        p = tmp_path / "canvas_int.json"
        p.write_text(json.dumps(data))
        result = parse_canvas_json(str(p))
        row0 = result["affines_final"][0][0]
        assert all(isinstance(v, float) for v in row0)


class TestCanvasFrameCorners:
    def test_identity_affine_returns_raw_frame_corners(self):
        aff = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        corners = _canvas_frame_corners(aff, frame_h=100, frame_w=200)
        assert corners == [(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)]

    def test_pure_translation_shifts_all_corners(self):
        aff = [[1.0, 0.0, 50.0], [0.0, 1.0, 30.0]]
        corners = _canvas_frame_corners(aff, frame_h=100, frame_w=200)
        assert corners == [(50.0, 30.0), (250.0, 30.0), (250.0, 130.0), (50.0, 130.0)]

    def test_90_degree_rotation_affine(self):
        # 90° CCW: [[0,-1,tx],[1,0,ty]] with tx=100, ty=0
        aff = [[0.0, -1.0, 100.0], [1.0, 0.0, 0.0]]
        corners = _canvas_frame_corners(aff, frame_h=100, frame_w=200)
        assert corners[0] == (100.0, 0.0)  # (0,0)   → tx=100, ty=0
        assert corners[1] == (100.0, 200.0)  # (W,0)   → tx, W
        assert corners[2] == (0.0, 200.0)  # (W,H)   → tx-H, W
        assert corners[3] == (0.0, 0.0)  # (0,H)   → tx-H, 0


class TestCanvasLayoutInspectorDialog:
    def test_populates_table_and_stats(self, q_app):
        data = {
            "canvas_h": 800,
            "canvas_w": 1800,
            "frame_h": 400,
            "frame_w": 600,
            "T_global": [0.0, 0.0],
            "affines_final": [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 600.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 1200.0], [0.0, 1.0, 0.0]],
            ],
        }
        dlg = CanvasLayoutInspectorDialog(canvas_data=data)
        assert dlg._table.rowCount() == 3
        assert "3 frames" in dlg._stats_label.text()
        assert "1800×800" in dlg._stats_label.text()

    def test_zero_frame_dimensions_skips_polygons_but_fills_stats(self, q_app):
        data = {
            "canvas_h": 400,
            "canvas_w": 800,
            "frame_h": 0,
            "frame_w": 0,
            "T_global": [0.0, 0.0],
            "affines_final": [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        }
        dlg = CanvasLayoutInspectorDialog(canvas_data=data)
        assert dlg._table.rowCount() == 0
        assert "1 frames" in dlg._stats_label.text()

    def test_no_data_shows_initial_label(self, q_app):
        dlg = CanvasLayoutInspectorDialog()
        assert dlg._table.rowCount() == 0
        assert dlg._stats_label.text() == "No data loaded."


# ── TestStitchWorkerVideoPath ─────────────────────────────────────────────────


class TestStitchWorkerVideoPath:
    """Tests for StitchWorker video ingestion path (Issue 9, S84)."""

    def test_video_path_stored(self, q_app):
        """video_path is stored when passed to __init__."""
        from asp_gui.helpers.stitch_worker import StitchWorker

        w = StitchWorker([], "out.png", {}, video_path="/tmp/test.mp4")
        assert w._video_path == "/tmp/test.mp4"

    def test_no_video_path_defaults_to_none(self, q_app):
        """video_path is None when not provided."""
        from asp_gui.helpers.stitch_worker import StitchWorker

        w = StitchWorker([], "out.png", {})
        assert w._video_path is None

    def test_sig_review_video_signal_exists(self, q_app):
        """StitchWorker exposes sig_review_video signal for HITL checkpoint 0."""
        from asp_gui.helpers.stitch_worker import StitchWorker

        w = StitchWorker([], "out.png", {})
        assert hasattr(w, "sig_review_video")

    def test_video_mode_allows_empty_image_paths(self, q_app):
        """When video_path is set, image_paths can be empty (frames extracted by worker)."""
        from asp_gui.helpers.stitch_worker import StitchWorker

        w = StitchWorker([], "out.png", {}, video_path="/tmp/video.mp4")
        assert len(w._image_paths) == 0
        assert w._video_path == "/tmp/video.mp4"

    def test_video_n_frames_stored(self, q_app):
        """video_n_frames is stored and accessible."""
        from asp_gui.helpers.stitch_worker import StitchWorker

        w = StitchWorker([], "out.png", {}, video_path="/tmp/v.mp4", video_n_frames=42)
        assert w._video_n_frames == 42
