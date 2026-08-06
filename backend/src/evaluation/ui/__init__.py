"""PySide6 widgets/windows for the benchmark evaluation inspector.

Everything here is presentation: QWidget/QGraphicsView/QMainWindow subclasses.
Pixel-data computation and figure building live in ``evaluation.logic``; data
models, dataset discovery, metric flattening and queue/persistence semantics
live in ``evaluation.other``.

That split is load-bearing rather than cosmetic: the navigation state machine
(``other/session.py``) and every figure builder are unit-testable without a
display, and the FiftyOne triage surface reuses both without importing Qt.
"""
