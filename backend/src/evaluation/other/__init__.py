"""Shared data layer for the evaluation tooling.

The evaluation/annotation schema (``schema.py``), on-disk dataset/asset
discovery (``discovery.py``), and benchmark-metric flattening
(``metrics_view.py``) — the parts both the PySide6 inspector and the FiftyOne
triage surface read. Nothing here imports Qt or FiftyOne, so either surface can
be absent without breaking the other.
"""
