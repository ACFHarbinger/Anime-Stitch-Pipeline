"""
Image-Toolkit Backend Evaluation Constants

Contains constants used in the backend evaluation module.

Re-exported flat (the star-imports below) so callers can reach any constant as
``evaluation.constants.NAME`` without knowing which of the three modules defines
it — the split is by *what the constant configures* (schema vocabulary, UI
chrome, computation tuning), which is a maintenance concern, not something a
caller should have to track.
"""
from .logic import *  # noqa: F403
from .schema import *  # noqa: F403
from .user_interface import *  # noqa: F403
