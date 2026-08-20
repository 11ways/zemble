import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from pathspec import GitIgnoreSpec
from pathspec.pattern import Pattern


@dataclass(frozen=True)
class PreparedPattern:
    """One ignore pattern with the two decisions that depend only on the pattern itself."""

    pattern: Pattern
    include: bool
    #: Whether a negation of this pattern lets a file bypass the extension filter.
    bypasses_extensions: bool


@dataclass(frozen=True)
class IgnoreSpec:
    """A gitignore spec plus where it applies from, as a path relative to the walk root."""

    base: Path
    spec: GitIgnoreSpec
    patterns: tuple[PreparedPattern, ...]
    #: One regex matching what any of the patterns match, so a path that matches nothing is
    #: rejected in one search instead of one per pattern. None when it could not be built.
    prefilter: re.Pattern[str] | None
    #: Number of leading characters of a root-relative path that belong to this spec's base.
    base_offset: int


@dataclass(frozen=True)
class WalkedFile:
    """A file the walker yielded, with its root-relative path and the stat scandir already did."""

    path: Path
    relative_path: str
    stat: os.stat_result


_DEFAULT_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git/",
        ".hg/",
        ".svn/",
        "__pycache__/",
        "node_modules/",
        ".venv/",
        "venv/",
        ".tox/",
        ".mypy_cache/",
        ".pytest_cache/",
        ".ruff_cache/",
        ".cache/",
        ".zemble/",
        ".next/",
        "dist/",
        "build/",
        ".eggs/",
    }
)

#: Compiled ignore specs, keyed by directory and the ignore files' modification times, so the
#: two walks a single index build performs do not recompile every .gitignore in the tree.
_SPEC_CACHE: dict[
    tuple[str, int | None, int | None],
    tuple[GitIgnoreSpec, tuple[PreparedPattern, ...], re.Pattern[str] | None] | None,
] = {}


def _prepare(spec: GitIgnoreSpec) -> tuple[PreparedPattern, ...]:
    """Precompute the per-pattern decisions that do not depend on the path being matched."""
    prepared = []
    for pattern in spec.patterns:
        if pattern.include is None:
            continue
        # Bypass the extension filter only for negation patterns with a file extension suffix
        # (e.g. !special.kjs, !*.py). Patterns without a suffix (e.g. !vendor/, !.github/*)
        # target directories or broad globs and should not bypass extension filtering.
        raw = pattern.pattern
        has_suffix = isinstance(raw, str) and bool(Path(raw.rstrip("/")).suffix)
        prepared.append(PreparedPattern(pattern=pattern, include=pattern.include, bypasses_extensions=has_suffix))
    return tuple(prepared)


def _prefilter(patterns: tuple[PreparedPattern, ...]) -> re.Pattern[str] | None:
    """Compile the union of *patterns* into one regex, or None if that cannot be done.

    pathspec names a group inside every gitignore regex; the name is dropped here because an
    alternation cannot repeat it and the union only has to answer whether anything matched.

    :param patterns: The prepared patterns of one spec.
    :return: A regex matching exactly the paths at least one pattern matches.
    """
    sources = []
    for prepared in patterns:
        regex = prepared.pattern.regex
        if regex is None:
            return None
        sources.append(f"(?:{regex.pattern.replace('(?P<ps_d>', '(?:')})")
    if not sources:
        return None
    try:
        return re.compile("|".join(sources))
    except re.error:  # pragma: no cover - a pattern shape the union cannot express
        return None


def _mtime_or_none(path: str) -> int | None:
    """Return a file's modification time, or None when it is not a readable file."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_mtime_ns


def _read_lines(path: str) -> list[str]:
    """Read a text file's lines, ignoring undecodable bytes."""
    with open(path, encoding="utf-8", errors="ignore") as handle:
        return handle.read().splitlines()


def _load_ignore_for_dir(
    directory: str,
) -> tuple[GitIgnoreSpec, tuple[PreparedPattern, ...], re.Pattern[str] | None] | None:
    """Load and compile the gitignore and zembleignore of a dir, with their per-pattern decisions."""
    gitignore = f"{directory}/.gitignore"
    zembleignore = f"{directory}/.zembleignore"
    key = (directory, _mtime_or_none(gitignore), _mtime_or_none(zembleignore))
    if key in _SPEC_CACHE:
        return _SPEC_CACHE[key]

    lines = []
    if key[1] is not None:
        lines.extend(_read_lines(gitignore))
    if key[2] is not None:
        lines.extend(_read_lines(zembleignore))
    loaded = None
    if lines:
        spec = GitIgnoreSpec.from_lines(lines)
        patterns = _prepare(spec)
        loaded = (spec, patterns, _prefilter(patterns))
    _SPEC_CACHE[key] = loaded
    return loaded


def walk_files(root: Path, extensions: Sequence[str], ignore: Sequence[str] | None = None) -> Iterator[Path]:
    """Yield files under root matching extensions, skipping ignored paths.

    :param root: Root directory to walk.
    :param extensions: List of file extensions to match.
    :param ignore: Additional patterns to ignore.
    :yield: Path to each file under root matching the criteria.
    :ytype: Path
    """
    for walked in walk_entries(root, extensions, ignore):
        yield walked.path


def walk_entries(root: Path, extensions: Sequence[str], ignore: Sequence[str] | None = None) -> Iterator[WalkedFile]:
    """Yield files under root matching extensions, with their root-relative paths.

    Directories matching DEFAULT_IGNORED_DIRS plus any names in ignore are always
    skipped. If the root contains a .gitignore, its patterns are also honoured.

    :param root: Root directory to walk.
    :param extensions: List of file extensions to match.
    :param ignore: Additional patterns to ignore.
    :yield: Each matching file and its path relative to root.
    :ytype: WalkedFile
    """
    extensions_set = frozenset(extensions)
    dir_patterns = list(sorted(_DEFAULT_IGNORED_DIRS)) + list(ignore or [])
    base_spec = GitIgnoreSpec.from_lines(dir_patterns, backend="simple")
    base_patterns = _prepare(base_spec)
    spec = IgnoreSpec(
        base=root,
        spec=base_spec,
        patterns=base_patterns,
        prefilter=_prefilter(base_patterns),
        base_offset=0,
    )
    yield from _walk(str(root), "", [spec], extensions_set)


def _is_ignored(relative_path: str, is_dir: bool, specs: list[IgnoreSpec]) -> tuple[bool, bool]:
    """Check if a root-relative path is ignored by any of the provided ignore specs."""
    ignored = False
    found = False
    for ignore_spec in specs:
        relative_str = relative_path[ignore_spec.base_offset :]
        # We need to add a trailing slash. Gitignore matches dirs as trailing '/'.
        if is_dir:
            relative_str += "/"

        if ignore_spec.prefilter is not None and ignore_spec.prefilter.search(relative_str) is None:
            continue

        for prepared in ignore_spec.patterns:
            if prepared.pattern.match_file(relative_str) is not None:
                ignored = prepared.include
                found = not ignored and prepared.bypasses_extensions

    return ignored, found


def _walk(
    directory: str,
    relative_path: str,
    inherited_specs: list[IgnoreSpec],
    extensions: frozenset[str],
) -> Iterator[WalkedFile]:
    """Recursive function for walking files under a directory."""
    loaded = _load_ignore_for_dir(directory)
    if loaded is not None:
        inherited_specs = [
            *inherited_specs,
            IgnoreSpec(
                base=Path(directory),
                spec=loaded[0],
                patterns=loaded[1],
                prefilter=loaded[2],
                base_offset=len(relative_path) + 1 if relative_path else 0,
            ),
        ]

    with os.scandir(directory) as scan:
        entries = sorted(scan, key=lambda entry: entry.name)

    for entry in entries:
        # Don't follow symlinks
        if entry.is_symlink():
            continue
        is_dir = entry.is_dir()
        child_relative = f"{relative_path}/{entry.name}" if relative_path else entry.name
        is_ignored, found = _is_ignored(child_relative, is_dir, inherited_specs)
        if is_ignored:
            continue

        if is_dir:
            yield from _walk(entry.path, child_relative, inherited_specs, extensions)
        elif entry.is_file() and (found or os.path.splitext(entry.name)[1].lower() in extensions):
            yield WalkedFile(path=Path(entry.path), relative_path=child_relative, stat=entry.stat())
