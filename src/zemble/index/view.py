"""Which of an index's files a view of it may answer from.

One predicate for both narrowings an index can carry: the sub-directory a request was routed
to, and the paths and exclude patterns a caller asked for. Both are matched against the same
repo-relative path, so a filtered sub-tree view needs no second rebasing rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from pathspec import GitIgnoreSpec

from zemble.index.file_walker import compile_ignore


@lru_cache(maxsize=64)
def _spec(patterns: tuple[str, ...]) -> GitIgnoreSpec | None:
    """Compile one pattern tuple once; a view is rebuilt per query and must not recompile."""
    return compile_ignore(patterns)


def _normalize_path(path: str) -> str:
    """Return a path in the one shape every comparison here uses: posix, no leading or trailing slash."""
    return path.replace("\\", "/").strip("/").removeprefix("./")


@dataclass(frozen=True)
class IndexView:
    """A restriction on which indexed files may answer, evaluated per chunk file path."""

    #: The routing prefix, ending in "/", that a sub-path request was served from an ancestor with.
    prefix: str | None = None
    #: Root-relative sub-paths to keep, relative to the REQUESTED repo, not to the index root.
    paths: tuple[str, ...] = ()
    #: Gitignore-style patterns to drop, relative to the requested repo.
    exclude: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        prefix: str | None = None,
        paths: Sequence[str] = (),
        exclude: Sequence[str] = (),
    ) -> IndexView | None:
        """Normalise a restriction, or return None when it restricts nothing.

        :param prefix: A root-relative directory the whole view sits under.
        :param paths: Sub-paths of the requested repo to keep.
        :param exclude: Gitignore-style patterns to drop.
        :return: The view, or None when every indexed file passes.
        """
        normalized_prefix = _normalize_path(prefix) if prefix else ""
        kept = tuple(dict.fromkeys(filter(None, (_normalize_path(path) for path in paths))))
        dropped = tuple(dict.fromkeys(filter(None, (pattern.strip() for pattern in exclude))))
        if not normalized_prefix and not kept and not dropped:
            return None
        return cls(prefix=f"{normalized_prefix}/" if normalized_prefix else None, paths=kept, exclude=dropped)

    @property
    def key(self) -> str:
        """A stable identity for caching built views."""
        return "\x00".join((self.prefix or "", *self.paths, "\x01", *self.exclude))

    def with_filter(self, paths: Sequence[str] = (), exclude: Sequence[str] = ()) -> IndexView | None:
        """Return this view's routing prefix carrying a caller's filter instead of its own.

        A caller's paths and patterns are relative to the repo it named, which is exactly what
        the prefix strips off, so the two compose without rewriting either.
        """
        return IndexView.build(self.prefix, paths, exclude)

    def keeps(self, file_path: str) -> bool:
        """Return whether an indexed file may appear in answers from this view."""
        path = file_path.replace("\\", "/")
        if self.prefix is not None:
            if not path.startswith(self.prefix):
                return False
            path = path[len(self.prefix) :]
        if self.paths and not any(path == kept or path.startswith(f"{kept}/") for kept in self.paths):
            return False
        spec = _spec(self.exclude)
        return not (spec is not None and spec.match_file(path))


__all__ = ["IndexView"]
