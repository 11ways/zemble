"""`zemble home`: does this feature already exist, and which module should it live in.

Answers the question a new capability starts with, from what the workspace itself declares.
"""

from zemble.home.answers import build_answer, home_payload
from zemble.home.config import CONFIG_RELATIVE_PATH, ConfigError, ForbiddenRule, HomeConfig, TableSpec, WorkspaceRule
from zemble.home.decide import (
    Candidate,
    Checklist,
    Confidence,
    DeclaredEvidence,
    HomeAnswer,
    Mechanism,
    Verdict,
    decide,
)
from zemble.home.deps import DependencyEdge, DependencyGraph, DependencySource, EdgeOrigin, Reachability
from zemble.home.source_sets import SourceSet
from zemble.home.tables import DeclaredRow, RowMatch, RowMatchKind, load_rows, match_rows, row_match_kind

__all__ = [
    "CONFIG_RELATIVE_PATH",
    "Candidate",
    "Checklist",
    "Confidence",
    "ConfigError",
    "DeclaredEvidence",
    "DeclaredRow",
    "DependencyEdge",
    "DependencyGraph",
    "DependencySource",
    "EdgeOrigin",
    "ForbiddenRule",
    "HomeAnswer",
    "HomeConfig",
    "Mechanism",
    "Reachability",
    "RowMatch",
    "RowMatchKind",
    "SourceSet",
    "TableSpec",
    "Verdict",
    "WorkspaceRule",
    "build_answer",
    "decide",
    "home_payload",
    "load_rows",
    "match_rows",
    "row_match_kind",
]
