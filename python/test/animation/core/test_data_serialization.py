"""
Tests for backend.src.animation.core.data_serialization — Issue 10B1+10B2.
"""

import json
import os

import numpy as np
from backend.src.animation.core.data_serialization import (
    COCOAnnotationBuilder,
    LabelStudioExporter,
    _bbox_from_mask,
    _mask_to_polygon,
    create_session_serializers,
)

# ── helper fixtures ───────────────────────────────────────────────────────────


def _rect_mask(h=64, w=64, x=10, y=10, bw=20, bh=20) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[y : y + bh, x : x + bw] = 255
    return m


# ── TestBboxFromMask ──────────────────────────────────────────────────────────


class TestBboxFromMask:
    def test_simple_rect(self):
        m = _rect_mask(x=10, y=10, bw=20, bh=15)
        x, y, w, h = _bbox_from_mask(m)
        assert x == 10 and y == 10 and w == 20 and h == 15

    def test_empty_mask_returns_zeros(self):
        m = np.zeros((64, 64), dtype=np.uint8)
        assert _bbox_from_mask(m) == (0, 0, 0, 0)

    def test_full_mask(self):
        m = np.full((32, 48), 255, dtype=np.uint8)
        x, y, w, h = _bbox_from_mask(m)
        assert x == 0 and y == 0 and w == 48 and h == 32

    def test_single_pixel(self):
        m = np.zeros((10, 10), dtype=np.uint8)
        m[5, 7] = 255
        x, y, w, h = _bbox_from_mask(m)
        assert x == 7 and y == 5 and w == 1 and h == 1

    def test_threshold_at_128(self):
        m = np.zeros((20, 20), dtype=np.uint8)
        m[2:5, 2:5] = 127  # below threshold
        m[8:12, 8:12] = 128  # at threshold — included
        x, y, w, h = _bbox_from_mask(m)
        assert x == 8 and y == 8


# ── TestMaskToPolygon ─────────────────────────────────────────────────────────


class TestMaskToPolygon:
    def test_rect_polygon_has_points(self):
        m = _rect_mask(x=5, y=5, bw=10, bh=10)
        polys = _mask_to_polygon(m)
        assert len(polys) >= 1
        assert all(len(p) >= 6 for p in polys)  # at least 3 x,y pairs

    def test_empty_mask_returns_no_polygons(self):
        m = np.zeros((32, 32), dtype=np.uint8)
        assert _mask_to_polygon(m) == []

    def test_polygon_coordinates_in_bounds(self):
        h, w = 64, 64
        m = _rect_mask(h=h, w=w, x=10, y=10, bw=20, bh=20)
        for poly in _mask_to_polygon(m):
            xs = poly[0::2]
            ys = poly[1::2]
            assert all(0 <= x <= w for x in xs)
            assert all(0 <= y <= h for y in ys)


# ── TestCOCOAnnotationBuilder ─────────────────────────────────────────────────


class TestCOCOAnnotationBuilder:
    def test_add_image_returns_sequential_ids(self):
        b = COCOAnnotationBuilder()
        id1 = b.add_image("f0.jpg", width=1920, height=1080, temporal_id=0)
        id2 = b.add_image("f1.jpg", width=1920, height=1080, temporal_id=1)
        assert id1 == 1 and id2 == 2

    def test_add_segmentation_mask_creates_annotation(self):
        b = COCOAnnotationBuilder()
        img_id = b.add_image("f0.jpg", width=64, height=64)
        m = _rect_mask()
        ann_id = b.add_segmentation_mask(img_id, m)
        assert ann_id == 1
        assert len(b) == 1
        d = b.to_dict()
        ann = d["annotations"][0]
        assert ann["image_id"] == img_id
        assert ann["category_id"] == 1  # foreground
        assert ann["area"] == float(np.count_nonzero(m > 127))

    def test_add_seam_exclusion_with_bbox(self):
        b = COCOAnnotationBuilder()
        img_id = b.add_image("f0.jpg", width=640, height=360)
        b.add_seam_exclusion(
            img_id, bbox=[100, 50, 80, 60], text_prompt="right arm"
        )
        d = b.to_dict()
        ann = d["annotations"][0]
        assert ann["attributes"]["text_prompt"] == "right arm"
        assert ann["attributes"]["source"] == "human"
        assert ann["category_id"] == 2  # seam_exclusion

    def test_save_writes_valid_json(self, tmp_path):
        b = COCOAnnotationBuilder()
        img_id = b.add_image("frame.jpg", width=100, height=100)
        b.add_segmentation_mask(img_id, _rect_mask())
        path = str(tmp_path / "coco.json")
        b.save(path)
        assert os.path.isfile(path)
        with open(path) as f:
            d = json.load(f)
        assert "images" in d and "annotations" in d and "categories" in d
        assert len(d["images"]) == 1
        assert len(d["annotations"]) == 1

    def test_save_is_atomic_does_not_leave_tmp_on_success(self, tmp_path):
        b = COCOAnnotationBuilder()
        b.add_image("f.jpg")
        path = str(tmp_path / "out.json")
        b.save(path)
        # No .tmp files should remain
        tmps = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
        assert len(tmps) == 0

    def test_to_dict_structure(self):
        b = COCOAnnotationBuilder()
        d = b.to_dict()
        for key in ("info", "licenses", "images", "annotations", "categories"):
            assert key in d
        assert isinstance(d["categories"], list)
        assert len(d["categories"]) >= 2  # foreground + seam_exclusion

    def test_frame_selection_override_annotation(self):
        b = COCOAnnotationBuilder()
        img_id = b.add_image("f0.jpg")
        b.add_frame_selection_override(
            img_id, accepted=False, reason="duplicate pose"
        )
        d = b.to_dict()
        ann = d["annotations"][0]
        assert ann["attributes"]["accepted"] is False
        assert ann["attributes"]["reason"] == "duplicate pose"
        assert ann["attributes"]["type"] == "frame_selection"


# ── TestLabelStudioExporter ───────────────────────────────────────────────────


class TestLabelStudioExporter:
    def test_add_task_returns_id_string(self):
        exp = LabelStudioExporter()
        tid = exp.add_task("frame.jpg", temporal_id=0)
        assert isinstance(tid, str) and len(tid) == 8

    def test_task_with_both_masks(self):
        exp = LabelStudioExporter()
        model_mask = _rect_mask(x=5, y=5, bw=15, bh=15)
        human_mask = _rect_mask(x=7, y=7, bw=12, bh=12)
        exp.add_task(
            "frame.jpg",
            temporal_id=0,
            model_mask=model_mask,
            human_mask=human_mask,
            category="foreground",
        )
        tasks = exp.to_list()
        assert len(tasks) == 1
        assert len(tasks[0]["predictions"]) == 1
        assert len(tasks[0]["annotations"]) == 1

    def test_task_with_clicks_adds_keypoints(self):
        exp = LabelStudioExporter()
        human_mask = _rect_mask()
        exp.add_task(
            "frame.jpg",
            human_mask=human_mask,
            pos_clicks=[(30, 30), (40, 40)],
            neg_clicks=[(5, 5)],
        )
        ann = exp.to_list()[0]["annotations"][0]
        types = [r["type"] for r in ann["result"]]
        assert "keypointlabels" in types

    def test_save_writes_list(self, tmp_path):
        exp = LabelStudioExporter()
        exp.add_task("f.jpg", temporal_id=1)
        path = str(tmp_path / "ls.json")
        exp.save(path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_no_masks_task(self):
        exp = LabelStudioExporter()
        exp.add_task("frame.jpg", text_prompt="girl with red hair")
        tasks = exp.to_list()
        assert tasks[0]["data"]["text_prompt"] == "girl with red hair"
        assert tasks[0]["annotations"] == []
        assert tasks[0]["predictions"] == []


# ── TestCreateSessionSerializers ──────────────────────────────────────────────


class TestCreateSessionSerializers:
    def test_returns_three_values(self, tmp_path):
        builder, exporter, session_dir = create_session_serializers(
            str(tmp_path / "session")
        )
        assert isinstance(builder, COCOAnnotationBuilder)
        assert isinstance(exporter, LabelStudioExporter)
        assert os.path.isdir(session_dir)

    def test_default_dir_is_created(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        builder, exporter, session_dir = create_session_serializers()
        assert os.path.isdir(session_dir)
        assert "hitl_annotations" in session_dir
