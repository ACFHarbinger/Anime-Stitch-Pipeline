"""Queue, history and persistence for an evaluation pass — the navigation
state machine, with no Qt import so its semantics are unit-testable.

This exists because the old dashboard's navigation was wrong in three
independent ways (issue #123):

- **Back-after-Skip was impossible** (defect 2): ``_go_skip()`` advanced
  without pushing the current test onto the history stack that ``_go_back()``
  pops from, so a skipped test was unreachable for the rest of the session.
  Here *every* move records where it came from.
- **``--redo`` stopped after one test** (defect 4): the "what's next" search
  filtered candidates by ``name not in self.evaluations``, but in redo mode
  every name is already in there. Ordering is now positional over the full
  dataset list and the "is it rated" question is asked of the entry's scores
  (``RatingEntry.is_rated()``), never of dict membership.
- **Visiting a test marked it rated forever** (defect 5): scores were
  persisted for any test merely opened, and the next session's queue excluded
  anything present in the file. Persistence is now explicit, and an entry that
  carries no scores is never written at all.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional

from .schema import RatingEntry, load_evaluations, save_evaluations


@dataclasses.dataclass
class Progress:
    position: int  # 1-based index of the current test within `order`
    total: int  # datasets in this session's order
    rated: int  # datasets carrying a real judgment
    corpus: int  # datasets discovered on disk
    skipped: int


class EvaluationSession:
    """Owns the dataset order, the current position, the undo history, and
    writes to the evaluations file.

    Navigation is free-form: ``order`` is the full discovered dataset list, so
    any test can be revisited at any time. ``redo=False`` only changes where
    the session *starts* and which test ``next_unrated()`` jumps to — it never
    hides a test from the user.
    """

    def __init__(
        self,
        names: List[str],
        out_path: str,
        redo: bool = False,
        autosave: bool = True,
        on_change: Optional[Callable[[], None]] = None,
    ):
        self.out_path = out_path
        self.redo = redo
        self.autosave = autosave
        self.on_change = on_change
        self.evaluations: Dict[str, RatingEntry] = load_evaluations(out_path)
        self.order: List[str] = list(names)
        self._history: List[str] = []
        self._current: Optional[str] = None
        self._dirty = False
        # Working entries for tests that have been opened but have no user
        # input yet. Kept out of `evaluations` so they can never be persisted.
        self._pending: Dict[str, RatingEntry] = {}

        start = self.order[0] if self.order else None
        if not redo:
            start = self.next_unrated(from_index=-1) or start
        self._current = start

    # -- state ---------------------------------------------------------------

    @property
    def current(self) -> Optional[str]:
        return self._current

    def index_of(self, name: str) -> int:
        return self.order.index(name) if name in self.order else -1

    def entry(self, name: Optional[str] = None) -> RatingEntry:
        """The entry for ``name``, created on demand but *not* inserted into
        the persisted mapping — a merely-visited test must not end up in the
        file. ``commit()`` is what inserts it."""
        name = name or self._current
        if name is None:
            return RatingEntry()
        existing = self.evaluations.get(name)
        if existing is not None:
            return existing
        return self._pending.setdefault(name, RatingEntry())

    def is_rated(self, name: str) -> bool:
        entry = self.evaluations.get(name)
        return entry is not None and entry.is_rated()

    def is_skipped(self, name: str) -> bool:
        entry = self.evaluations.get(name)
        return entry is not None and entry.skipped

    def rated_count(self) -> int:
        return sum(1 for entry in self.evaluations.values() if entry.is_rated())

    def skipped_count(self) -> int:
        return sum(1 for entry in self.evaluations.values() if entry.skipped and not entry.is_rated())

    def progress(self) -> Progress:
        position = self.index_of(self._current) + 1 if self._current else 0
        return Progress(
            position=position,
            total=len(self.order),
            rated=self.rated_count(),
            corpus=len(self.order),
            skipped=self.skipped_count(),
        )

    def can_go_back(self) -> bool:
        return bool(self._history)

    # -- persistence ---------------------------------------------------------

    def commit(self, name: Optional[str] = None) -> None:
        """Fold the working entry for ``name`` into the persisted mapping.

        A no-op for an entry with nothing in it, which is what keeps a visited
        test out of the file (defect 5). An entry that has *anything* — a
        score, a note, an annotation, a defect tag, or an explicit skip — is
        real user input and gets written.
        """
        name = name or self._current
        if name is None:
            return
        entry = self._pending.get(name) or self.evaluations.get(name)
        if entry is None or not self._has_content(entry):
            return
        entry.touch()
        self.evaluations[name] = entry
        self._pending.pop(name, None)
        self._dirty = True
        if self.autosave:
            self.save()
        if self.on_change is not None:
            self.on_change()

    @staticmethod
    def _has_content(entry: RatingEntry) -> bool:
        return bool(
            entry.asp is not None
            or entry.simple is not None
            or entry.notes.strip()
            or entry.bboxes
            or entry.edges
            or entry.defects
            or entry.preference is not None
            or entry.skipped
            or any(v is not None for dims in entry.dimensions.values() for v in dims.values())
        )

    def save(self) -> None:
        if not self.evaluations and not self._dirty:
            return
        save_evaluations(self.out_path, self.evaluations)
        self._dirty = False

    # -- navigation ----------------------------------------------------------

    def go_to(self, name: str, record_history: bool = True) -> Optional[str]:
        """Move to ``name``, committing the test being left behind.

        ``record_history`` is what makes Back work from *every* kind of move,
        including a skip.
        """
        if name not in self.order:
            return self._current
        self.commit()
        if record_history and self._current is not None and self._current != name:
            self._history.append(self._current)
        self._current = name
        return self._current

    def go_back(self) -> Optional[str]:
        if not self._history:
            return None
        self.commit()
        self._current = self._history.pop()
        return self._current

    def next_unrated(self, from_index: Optional[int] = None) -> Optional[str]:
        """The next test with no real judgment, searching forward from the
        current position and wrapping once.

        Positional and score-based, so ``--redo`` works: in redo mode every
        name is in the evaluations file, which is exactly the condition that
        made the old membership test return nothing.
        """
        if not self.order:
            return None
        start = from_index if from_index is not None else self.index_of(self._current)
        total = len(self.order)
        for offset in range(1, total + 1):
            candidate = self.order[(start + offset) % total]
            if not self.is_rated(candidate):
                return candidate
        return None

    def advance(self, skip: bool = False) -> Optional[str]:
        """Move forward one test.

        ``skip=True`` marks the current test skipped (so the UI can show it as
        deferred and ``next_unrated`` can still find it later) and moves to the
        immediately following test rather than hunting for an unrated one — a
        skip means "not now", not "never".
        """
        current = self._current
        if skip and current is not None:
            entry = self.entry(current)
            entry.skipped = True
            self.commit(current)
        if current is None:
            return self.next_unrated(from_index=-1)
        index = self.index_of(current)
        if skip or self.redo:
            # A skip means "not now", and redo mode means "walk everything
            # again" — both want the literal next test. Asking next_unrated()
            # here is what dead-ended the old tool in redo mode, since in that
            # mode there is by definition nothing unrated left to find.
            target = self.order[index + 1] if index + 1 < len(self.order) else None
        else:
            target = self.next_unrated(from_index=index)
        if target is None:
            return None
        return self.go_to(target)

    def accept(self) -> Optional[str]:
        """Mark the current test reviewed, then advance to the next unrated."""
        current = self._current
        if current is not None:
            entry = self.entry(current)
            entry.reviewed = True
            entry.skipped = False
            self.commit(current)
        return self.advance(skip=False)
