"""
HITL annotation serialization — Issue 10B1 (COCO JSON) + Issue 10B2 (Label Studio JSON).

Every human interaction during HITL execution is captured here so the annotations
accumulate into training data for fine-tuning SAM-2, DWPose, and the reward model.

Storage path: ~/.image-toolkit/hitl_annotations/session_{timestamp}.json (COCO)
              ~/.image-toolkit/hitl_annotations/session_{timestamp}_ls.json (Label Studio)
"""

from __future__ import annotations

# --- Relocated Nested Imports ---
try:
    from pycocotools import mask as coco_mask  # type: ignore
except ImportError:
    coco_mask = None
# --------------------------------
import contextlib
import json
import os
import tempfile
import uuid
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    """Convert a binary uint8 mask to a COCO polygon (list of [x,y,...] contours)."""
    contours, _ = cv2.findContours(
        (mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons: List[List[float]] = []
    for cnt in contours:
        if cnt.shape[0] < 3:
            continue
        flat = cnt.squeeze(1).astype(float).flatten().tolist()
        if len(flat) >= 6:
            polygons.append(flat)
    return polygons


def _mask_to_rle(mask: np.ndarray) -> Optional[Dict]:
    """Encode mask as COCO RLE if pycocotools is available; otherwise returns None."""
    if coco_mask is None:
        return None
    try:
        binary = np.asfortranarray((mask > 127).astype(np.uint8))
        rle = coco_mask.encode(binary)
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle
    except Exception:
        return None


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Return COCO bbox [x, y, w, h] from a binary mask."""
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def _atomic_write(path: str, data: Any) -> None:
    """Serialize *data* to JSON and atomically replace *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# COCO JSON builder (Issue 10B1)
# ---------------------------------------------------------------------------

_DEFAULT_CATEGORIES = [
    {"id": 1, "name": "foreground", "supercategory": "character"},
    {"id": 2, "name": "seam_exclusion", "supercategory": "routing"},
]


class COCOAnnotationBuilder:
    """
    Accumulates HITL annotations in COCO JSON format.

    Usage::

        builder = COCOAnnotationBuilder()
        img_id = builder.add_image("frame00.jpg", width=1920, height=1080, temporal_id=0)
        builder.add_segmentation_mask(img_id, mask_uint8, category="foreground")
        builder.add_seam_exclusion(img_id, bbox=[x, y, w, h], text_prompt="right arm")
        builder.save("~/.image-toolkit/hitl_annotations/session_20260613.json")

    The saved JSON is COCO-compatible: images + annotations + categories arrays.
    """

    def __init__(self, categories: Optional[List[Dict]] = None):
        self._categories = categories or list(_DEFAULT_CATEGORIES)
        self._cat_name_to_id: Dict[str, int] = {c["name"]: c["id"] for c in self._categories}
        self._images: List[Dict] = []
        self._annotations: List[Dict] = []
        self._next_img_id = 1
        self._next_ann_id = 1

    # ── image registry ────────────────────────────────────────────────────

    def add_image(
        self,
        file_name: str,
        *,
        width: int = 0,
        height: int = 0,
        temporal_id: int = 0,
    ) -> int:
        """Register a frame and return its image_id."""
        img_id = self._next_img_id
        self._next_img_id += 1
        self._images.append({
            "id": img_id,
            "file_name": file_name,
            "width": int(width),
            "height": int(height),
            "temporal_id": int(temporal_id),
        })
        return img_id

    # ── annotation factories ──────────────────────────────────────────────

    def add_segmentation_mask(
        self,
        image_id: int,
        mask: np.ndarray,
        *,
        category: str = "foreground",
        source: str = "human",
        pre_correction_mask: Optional[np.ndarray] = None,
    ) -> int:
        """
        Add a binary segmentation mask annotation.

        mask : uint8 (H, W) — 255 = included in segment, 0 = excluded.
        source : "human" (post-correction) or "model" (pre-correction).
        pre_correction_mask : the model's original mask before click refinement.
            When provided it is stored in the annotation's ``attributes.model_mask``
            field as a polygon, which is the Label Studio prediction layer.
        """
        cat_id = self._cat_name_to_id.get(category, 1)
        rle = _mask_to_rle(mask)
        polygons = _mask_to_polygon(mask) if rle is None else _mask_to_polygon(mask)
        bbox = _bbox_from_mask(mask)
        area = float(np.count_nonzero(mask > 127))

        ann: Dict[str, Any] = {
            "id": self._next_ann_id,
            "image_id": image_id,
            "category_id": cat_id,
            "segmentation": rle if rle is not None else polygons,
            "area": area,
            "bbox": list(bbox),
            "iscrowd": 0,
            "attributes": {"source": source},
        }
        if pre_correction_mask is not None:
            ann["attributes"]["model_segmentation"] = _mask_to_polygon(pre_correction_mask)

        self._next_ann_id += 1
        self._annotations.append(ann)
        return ann["id"]

    def add_seam_exclusion(
        self,
        image_id: int,
        *,
        bbox: Optional[List[int]] = None,
        mask: Optional[np.ndarray] = None,
        text_prompt: str = "",
    ) -> int:
        """
        Add a seam-routing exclusion region annotation.

        Either bbox ([x, y, w, h]) or mask must be provided. Both may be provided.
        text_prompt is stored in attributes for fine-tuning the DINO text encoder.
        """
        cat_id = self._cat_name_to_id.get("seam_exclusion", 2)
        polygons: List[List[float]] = []
        final_bbox: List[int] = bbox or [0, 0, 0, 0]
        area = 0.0

        if mask is not None:
            polygons = _mask_to_polygon(mask)
            final_bbox = list(_bbox_from_mask(mask))
            area = float(np.count_nonzero(mask > 127))
        elif bbox is not None:
            x, y, w, h = bbox
            polygons = [[float(x), float(y), float(x + w), float(y), float(x + w), float(y + h), float(x), float(y + h)]]
            area = float(w * h)

        ann: Dict[str, Any] = {
            "id": self._next_ann_id,
            "image_id": image_id,
            "category_id": cat_id,
            "segmentation": polygons,
            "area": area,
            "bbox": final_bbox,
            "iscrowd": 0,
            "attributes": {"text_prompt": text_prompt, "source": "human"},
        }
        self._next_ann_id += 1
        self._annotations.append(ann)
        return ann["id"]

    def add_frame_selection_override(
        self,
        image_id: int,
        *,
        accepted: bool,
        reason: str = "",
    ) -> int:
        """Record that the user accepted or rejected a frame during frame-selection review."""
        cat_id = self._cat_name_to_id.get("foreground", 1)
        ann: Dict[str, Any] = {
            "id": self._next_ann_id,
            "image_id": image_id,
            "category_id": cat_id,
            "segmentation": [],
            "area": 0.0,
            "bbox": [0, 0, 0, 0],
            "iscrowd": 0,
            "attributes": {"type": "frame_selection", "accepted": accepted, "reason": reason},
        }
        self._next_ann_id += 1
        self._annotations.append(ann)
        return ann["id"]

    # ── serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Return the full COCO-format dict."""
        return {
            "info": {
                "description": "ASP HITL Annotations",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Image-Toolkit HITL",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "images": list(self._images),
            "annotations": list(self._annotations),
            "categories": list(self._categories),
        }

    def save(self, path: str) -> None:
        """Atomically write COCO JSON to *path*. Silently logs on failure."""
        path = os.path.expanduser(path)
        try:
            _atomic_write(path, self.to_dict())
        except Exception as exc:
            warnings.warn(f"[ASP] COCOAnnotationBuilder.save failed: {exc}", stacklevel=2)

    def __len__(self) -> int:
        return len(self._annotations)


# ---------------------------------------------------------------------------
# Label Studio JSON exporter (Issue 10B2)
# ---------------------------------------------------------------------------

class LabelStudioExporter:
    """
    Exports HITL annotation sessions to Label Studio JSON format.

    Captures the model-vs-human delta:
      predictions array  = SAM-2's pre-correction mask (what the model produced)
      annotations array  = human's post-correction mask (what the human accepted)

    This is the ideal supervision signal for RLHF preference learning: the
    preferred output is always the ``annotations`` array.

    Usage::

        exp = LabelStudioExporter()
        exp.add_task(
            frame_path="Data/asp_test07/frame000.jpg",
            temporal_id=0,
            model_mask=initial_sam2_mask,
            human_mask=accepted_mask,
            category="foreground",
        )
        exp.save("~/.image-toolkit/hitl_annotations/session_20260613_ls.json")
    """

    def __init__(self):
        self._tasks: List[Dict] = []

    def add_task(
        self,
        frame_path: str,
        *,
        temporal_id: int = 0,
        model_mask: Optional[np.ndarray] = None,
        human_mask: Optional[np.ndarray] = None,
        category: str = "foreground",
        text_prompt: str = "",
        pos_clicks: Optional[List[Tuple[int, int]]] = None,
        neg_clicks: Optional[List[Tuple[int, int]]] = None,
    ) -> str:
        """
        Add one annotation task.

        Returns the task id (a timestamp-based UUID string).
        """
        # relocated: import uuid
        task_id = str(uuid.uuid4())[:8]

        def _mask_to_ls_result(mask: np.ndarray, result_id: str, source: str) -> List[Dict]:
            """Encode a mask as a Label Studio polygon result."""
            polygons = _mask_to_polygon(mask)
            results = []
            h, w = mask.shape[:2]
            for poly_idx, pts in enumerate(polygons):
                points = [[pts[i] / w * 100, pts[i + 1] / h * 100] for i in range(0, len(pts) - 1, 2)]
                results.append({
                    "id": f"{result_id}_{poly_idx}",
                    "type": "polygonlabels",
                    "value": {
                        "points": points,
                        "polygonlabels": [category],
                    },
                    "from_name": "label",
                    "to_name": "image",
                    "origin": source,
                })
            return results

        predictions: List[Dict] = []
        annotations: List[Dict] = []

        if model_mask is not None:
            predictions = [{
                "id": 1,
                "model_version": "sam2_hiera_base_plus",
                "score": 0.0,
                "result": _mask_to_ls_result(model_mask, f"{task_id}_pred", "model"),
            }]

        if human_mask is not None:
            ann_results = _mask_to_ls_result(human_mask, f"{task_id}_ann", "human")
            # Add click prompt annotations as point markers
            if pos_clicks:
                h, w = human_mask.shape[:2]
                for ci, (cx, cy) in enumerate(pos_clicks):
                    ann_results.append({
                        "id": f"{task_id}_pos_{ci}",
                        "type": "keypointlabels",
                        "value": {"x": cx / w * 100, "y": cy / h * 100, "width": 0.5, "keypointlabels": ["positive_click"]},
                        "from_name": "keypoint",
                        "to_name": "image",
                        "origin": "human",
                    })
            if neg_clicks:
                h, w = human_mask.shape[:2]
                for ci, (cx, cy) in enumerate(neg_clicks):
                    ann_results.append({
                        "id": f"{task_id}_neg_{ci}",
                        "type": "keypointlabels",
                        "value": {"x": cx / w * 100, "y": cy / h * 100, "width": 0.5, "keypointlabels": ["negative_click"]},
                        "from_name": "keypoint",
                        "to_name": "image",
                        "origin": "human",
                    })
            annotations = [{
                "id": 1,
                "completed_by": 1,
                "result": ann_results,
                "lead_time": 0.0,
            }]

        self._tasks.append({
            "id": task_id,
            "data": {
                "image": f"/Data/local-files/?d={frame_path}",
                "temporal_id": temporal_id,
                "text_prompt": text_prompt,
            },
            "annotations": annotations,
            "predictions": predictions,
        })
        return task_id

    def to_list(self) -> List[Dict]:
        """Return the Label Studio task list."""
        return list(self._tasks)

    def save(self, path: str) -> None:
        """Atomically write Label Studio JSON to *path*. Silently logs on failure."""
        path = os.path.expanduser(path)
        try:
            _atomic_write(path, self.to_list())
        except Exception as exc:
            warnings.warn(f"[ASP] LabelStudioExporter.save failed: {exc}", stacklevel=2)

    def __len__(self) -> int:
        return len(self._tasks)


# ---------------------------------------------------------------------------
# Session serializer — combines both formats with one call
# ---------------------------------------------------------------------------

def create_session_serializers(
    session_dir: Optional[str] = None,
) -> Tuple["COCOAnnotationBuilder", "LabelStudioExporter", str]:
    """
    Create a paired (COCOAnnotationBuilder, LabelStudioExporter) for one HITL session.

    Returns (builder, exporter, session_dir) where session_dir is the directory
    where both files will be saved when .save() is called on each.
    """
    if session_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.expanduser(
            f"~/.image-toolkit/hitl_annotations/session_{ts}"
        )
    os.makedirs(session_dir, exist_ok=True)
    return COCOAnnotationBuilder(), LabelStudioExporter(), session_dir


__all__ = [
    "COCOAnnotationBuilder",
    "LabelStudioExporter",
    "create_session_serializers",
    "_mask_to_polygon",
    "_mask_to_rle",
    "_bbox_from_mask",
]
