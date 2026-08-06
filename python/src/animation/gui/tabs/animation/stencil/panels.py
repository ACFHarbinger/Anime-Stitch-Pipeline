from PySide6.QtWidgets import QWidget

from .hybrid_stitch_panel import RealHybridStitchPanel


class EditTabPanel:
    """Mixin class to expose EditTab configuration methods to panel proxies."""

    def __init__(self, parent_tab=None):
        self.parent_tab = parent_tab

    def collect(self) -> dict:
        if self.parent_tab:
            return self.parent_tab.collect()
        return {}

    def set_config(self, cfg: dict):
        if self.parent_tab:
            self.parent_tab.set_config(cfg)

    def get_default_config(self) -> dict:
        if self.parent_tab:
            return self.parent_tab.get_default_config()
        return {}


class StitchPanel(QWidget, EditTabPanel):
    def __init__(self, parent_tab=None):
        QWidget.__init__(self)
        EditTabPanel.__init__(self, parent_tab)


class GraphPanel(QWidget, EditTabPanel):
    def __init__(self, parent_tab=None):
        QWidget.__init__(self)
        EditTabPanel.__init__(self, parent_tab)


class AdjustPanel(QWidget, EditTabPanel):
    def __init__(self, parent_tab=None):
        QWidget.__init__(self)
        EditTabPanel.__init__(self, parent_tab)


class CanvasPanel(QWidget, EditTabPanel):
    def __init__(self, parent_tab=None):
        QWidget.__init__(self)
        EditTabPanel.__init__(self, parent_tab)


class StatsPanel(QWidget, EditTabPanel):
    def __init__(self, parent_tab=None):
        QWidget.__init__(self)
        EditTabPanel.__init__(self, parent_tab)


class SeqBuilderPanel(QWidget, EditTabPanel):
    def __init__(self, parent_tab=None):
        QWidget.__init__(self)
        EditTabPanel.__init__(self, parent_tab)


class HybridStitchPanel(RealHybridStitchPanel, EditTabPanel):
    def __init__(self, parent_tab=None):
        RealHybridStitchPanel.__init__(self)
        EditTabPanel.__init__(self, parent_tab)


class AnimClustersPanel(QWidget, EditTabPanel):
    def __init__(self, parent_tab=None):
        QWidget.__init__(self)
        EditTabPanel.__init__(self, parent_tab)
