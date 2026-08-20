"""`zemble home`: does this feature already exist, and which module should it live in.

Answers the question a new capability starts with, from what the workspace itself declares.
"""

from zemble.home.answers import build_answer, home_payload
from zemble.home.config import CONFIG_RELATIVE_PATH, ConfigError, ForbiddenRule, HomeConfig, TableSpec, WorkspaceRule
from zemble.home.decide import Candidate, Checklist, Confidence, HomeAnswer, Mechanism, Verdict, decide
from zemble.home.tables import DeclaredRow, RowMatch, load_rows, match_rows

__all__ = [
    "CONFIG_RELATIVE_PATH",
    "Candidate",
    "Checklist",
    "Confidence",
    "ConfigError",
    "DeclaredRow",
    "ForbiddenRule",
    "HomeAnswer",
    "HomeConfig",
    "Mechanism",
    "RowMatch",
    "TableSpec",
    "Verdict",
    "WorkspaceRule",
    "build_answer",
    "decide",
    "home_payload",
    "load_rows",
    "match_rows",
]
