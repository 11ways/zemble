"""Which fold of a module a file belongs to, and which folds may use which.

A javaweb module compiles the same package three times - `common`, `server` and the
browser fold - and a mechanism in the server fold is unreachable from the browser one
however close the two modules are. A verdict that ignores that recommends reuse that
cannot compile, so the fold is a fact the answer carries beside the module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from fnmatch import fnmatch

from zemble.graph.model import is_test_path


class SourceSet(str, Enum):
    """The fold of a module a file is compiled into.

    UNKNOWN is the honest answer for a workspace that does not split its sources, and it
    is compatible with itself only: a fold nobody classified may not be claimed reusable
    from a classified one.
    """

    COMMON = "common"
    SERVER = "server"
    BROWSER = "browser"
    TEST = "test"
    UNKNOWN = "unknown"


#: Path globs per fold, used when a workspace declares none.
#:
#: Matched against the workspace-relative path AND against that path with its first
#: segment removed, so one pattern covers both a single-module repository (`src/common/**`)
#: and a workspace of sibling repositories (`zenit/src/common/**`).
DEFAULT_PATTERNS: Mapping[SourceSet, tuple[str, ...]] = {
    SourceSet.COMMON: ("src/common/**", "common/**"),
    SourceSet.SERVER: ("src/server/**", "server/**"),
    # AIDEV-NOTE: javaweb spells the browser fold `client` (its TeaVM source set), so both
    # names are defaults; a workspace that means something else by `client` overrides them.
    SourceSet.BROWSER: ("src/browser/**", "browser/**", "src/client/**", "client/**"),
    SourceSet.TEST: (),
}

#: Which folds a consumer fold may use. One table, exhaustive over the enum.
COMPATIBILITY: Mapping[SourceSet, frozenset[SourceSet]] = {
    SourceSet.COMMON: frozenset({SourceSet.COMMON}),
    SourceSet.SERVER: frozenset({SourceSet.COMMON, SourceSet.SERVER}),
    SourceSet.BROWSER: frozenset({SourceSet.COMMON, SourceSet.BROWSER}),
    SourceSet.TEST: frozenset(SourceSet),
    SourceSet.UNKNOWN: frozenset({SourceSet.UNKNOWN}),
}


def classify(file_path: str, patterns: Mapping[SourceSet, Sequence[str]] | None = None) -> SourceSet:
    """Return the fold a workspace-relative path is compiled into.

    Test wins first, and by the graph's own rule rather than by a glob: `src/browserTest`
    is a test source set, not the browser fold, and only one place in zemble decides that.
    """
    relative = file_path.replace("\\", "/").lstrip("./")
    rules = dict(DEFAULT_PATTERNS if patterns is None else patterns)
    if is_test_path(relative) or _matches(relative, rules.get(SourceSet.TEST, ())):
        return SourceSet.TEST
    for fold in (SourceSet.COMMON, SourceSet.SERVER, SourceSet.BROWSER):
        if _matches(relative, rules.get(fold, ())):
            return fold
    return SourceSet.UNKNOWN


def compatible(consumer: SourceSet, provider: SourceSet) -> bool:
    """Whether code in the consumer fold may use code in the provider fold."""
    allowed = COMPATIBILITY.get(consumer)
    if allowed is None:  # pragma: no cover - a new member without a row is a build error
        raise ValueError(f"unhandled source set: {consumer!r}")
    return provider in allowed


def _matches(relative: str, patterns: Sequence[str]) -> bool:
    """Whether a path, or the same path inside its module, matches any pattern."""
    _, separator, inner = relative.partition("/")
    candidates = (relative, inner) if separator else (relative,)
    return any(fnmatch(candidate, pattern) for candidate in candidates for pattern in patterns)


__all__ = ["COMPATIBILITY", "DEFAULT_PATTERNS", "SourceSet", "classify", "compatible"]
