"""The inspector's region/link annotation flow, as a mixin.

Split out of ``main_window.py`` to keep that file within the repo's 500-LoC
budget (§5.17 / issues #121-#122), following the same mixin-composition pattern
that epic used. The mixin reads ``self.session``, ``self.grid``, ``self.overlay``,
``self.annotation_list``, ``self.scoring_panel``, ``self.status_label`` and
``self._edge_builder`` — all owned by ``InspectorWindow``.
"""

from __future__ import annotations

from PySide6.QtWidgets import QInputDialog

from ..constants.schema import COMPARATOR_TITLES
from ..other.schema import BoundingBox, RatingEntry
from .annotations import DefectDialog


class AnnotationFlowMixin:
    """Turns panel-drawn geometry into schema objects on the current entry."""

    def _current_entry(self) -> RatingEntry:
        return self.session.entry()

    def _on_bbox_drawn(self, key: str, data: dict) -> None:
        dialog = DefectDialog(COMPARATOR_TITLES.get(key, key), self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            self.grid.restore_bboxes(self._current_entry().bboxes)  # drop the transient rect
            return
        defect, severity, label = dialog.result_values()
        entry = self._current_entry()
        entry.bboxes.append(BoundingBox(
            image=key, x=data["x"], y=data["y"], w=data["w"], h=data["h"],
            label=label, defect=defect, severity=severity,
        ))
        # A region tagged with a defect class implies the test-level tag too —
        # otherwise the corpus-level defect counts would miss anything the user
        # only recorded spatially.
        if defect and defect not in entry.defects:
            entry.defects = sorted(set(entry.defects) | {defect})
            self.scoring_panel.load_entry(entry)
        self.grid.restore_bboxes(entry.bboxes)
        self.annotation_list.refresh(entry.bboxes, entry.edges)
        self._commit()

    def _on_point_picked(self, key: str, x: float, y: float) -> None:
        self._add_edge_anchor(key, x, y)

    def _on_region_picked(self, key: str, x: float, y: float, w: float, h: float) -> None:
        self._add_edge_anchor(key, x, y, w, h)

    def _add_edge_anchor(
        self, key: str, x: float, y: float, w: float = 0.0, h: float = 0.0
    ) -> None:
        """One endpoint (point or region) added to the in-progress link chain.

        Does not finish or prompt anything itself — a chain of 2 or more
        images stays open until the user explicitly finishes (Enter) or
        cancels (Esc) it, since there is no click count that means "done" once
        3+ endpoints are allowed.
        """
        self._edge_builder.add(key, x, y, w, h)
        self.overlay.set_pending(self._edge_builder.pending())
        count = self._edge_builder.count()
        kind = "region" if w > 0 else "point"
        self.status_label.setText(
            f"Link: {count} endpoint(s) added (last: {kind} on {COMPARATOR_TITLES.get(key, key)}) — "
            "Enter to finish, Esc to cancel, or keep clicking to add more."
        )

    def _finish_link(self) -> None:
        if not self._edge_builder.can_finish():
            self.status_label.setText(
                f"Link needs at least 2 endpoints ({self._edge_builder.count()} so far)."
            )
            return
        label, ok = QInputDialog.getText(
            self, "Finish link", "Describe the relationship between these endpoints:"
        )
        if not ok:
            return
        edge = self._edge_builder.finish(label.strip())
        if edge is None:
            return
        entry = self._current_entry()
        entry.edges.append(edge)
        self.overlay.set_pending([])
        self.overlay.set_edges(entry.edges)
        self.annotation_list.refresh(entry.bboxes, entry.edges)
        self._commit()
        self.status_label.setText(f"Link saved across {len(edge.points)} endpoint(s).")

    def _cancel_link(self) -> None:
        if self._edge_builder.count() == 0:
            return
        self._edge_builder.reset()
        self.overlay.set_pending([])
        self.status_label.setText("Link cancelled.")

    def _on_remove_annotation(self, kind: str, index: int) -> None:
        entry = self._current_entry()
        target = entry.bboxes if kind == "bbox" else entry.edges
        if 0 <= index < len(target):
            del target[index]
        self.grid.restore_bboxes(entry.bboxes)
        self.overlay.set_edges(entry.edges)
        self.annotation_list.refresh(entry.bboxes, entry.edges)
        self._commit()
