"""The graph facts overlay: precise edges written by an external analyzer.

zemble's own resolver is a name resolver, so an overload it cannot separate lands on
`AMBIGUOUS` and a receiver it cannot type lands on `UNIQUE_NAME`. A tool that already
knows the answer - a javac plugin, a language server, any real front end - can write
those edges into a JSONL file, and this module folds them into the graph: for every
source file the facts declare and whose content still hashes to what they were derived
from, the fact edges REPLACE the extracted ones.

The format is documented in `docs/graph-facts.md` and is the contract emitters are
written against; nothing here may rename one of its fields.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

import orjson

from zemble.graph.generated import GeneratedMapping, GeneratedSourceMapper, template_is_newer
from zemble.graph.model import (
    CALLABLE_KINDS,
    TYPE_KINDS,
    Edge,
    EdgeKind,
    Resolution,
    Symbol,
    SymbolKind,
)
from zemble.index.file_walker import ignored_prefix

logger = logging.getLogger(__name__)

#: Value of the header's `zemble_facts` key this reader understands.
FACTS_FORMAT_VERSION = 1
#: The `source` an edge carries when zemble's own extractor produced it.
TREE_SITTER_SOURCE = "tree-sitter"
#: Where facts files live when `.zemble/graph.toml` says nothing.
DEFAULT_SOURCE_GLOBS: tuple[str, ...] = (".zemble/facts/*.jsonl", "**/build/zemble/*.jsonl")
#: The graph configuration file, relative to the workspace root.
GRAPH_CONFIG_PATH = ".zemble/graph.toml"

#: Fact kinds that become edges. `file` and `symbol` are handled separately.
_EDGE_KIND_BY_FACT = {
    "call": EdgeKind.CALLS,
    "override": EdgeKind.OVERRIDES,
    "extends": EdgeKind.EXTENDS,
    "implements": EdgeKind.IMPLEMENTS,
}
#: The edge kinds the overlay owns: for a covered file these come from the facts and
#: from nowhere else, so an emitter that declares a file must emit all four.
OVERLAY_KINDS: frozenset[EdgeKind] = frozenset(_EDGE_KIND_BY_FACT.values())
_KNOWN_FACT_KINDS = frozenset({"file", "symbol", "annotation", *_EDGE_KIND_BY_FACT})

#: Directories the walker always skips, minus the ones a facts file lives in by design.
_SKIPPED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".cache",
        ".next",
        "dist",
        ".eggs",
    }
)


class FactsFormatError(ValueError):
    """A facts file could not be read at all: no header, or a header this reader refuses."""


# ---- the file format ----------------------------------------------------


@dataclass(frozen=True)
class FactsHeader:
    """Line 1 of a facts file."""

    tool: str
    tool_version: str
    generated_at: str
    language: str
    root: str

    @property
    def age_seconds(self) -> float | None:
        """Seconds since `generated_at`, or None when it is not a readable timestamp."""
        stamp = self.generated_at.replace("Z", "+00:00")
        try:
            from datetime import datetime, timezone

            parsed = datetime.fromisoformat(stamp)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return time.time() - parsed.timestamp()


@dataclass
class SourceFacts:
    """Every fact one facts file carries about one source file."""

    path: str
    declared_sha256: str
    fresh: bool
    reason: str = ""
    facts: list[dict] = field(default_factory=list)


class SkipBucket(Enum):
    """Why a fact never became an edge.

    The buckets are the vocabulary the status command counts and lists by, and they are
    deliberately separate: a fact whose source file the index does not cover is not a
    defect of the mapper, while an unmapped ref is.
    """

    OUTSIDE_INDEX = ("source_outside_index", "source outside the index")
    SOURCE_IGNORED = ("source_ignored", "source ignored by the index")
    GENERATED_NO_TEMPLATE = ("generated_no_template", "generated (no template)")
    STALE = ("stale", "stale")
    UNMAPPED = ("unmapped", "unmapped")

    label: str

    def __new__(cls, key: str, label: str) -> SkipBucket:
        """Build a member whose value is its JSON key and whose label is its heading."""
        member = object.__new__(cls)
        member._value_ = key
        member.label = label
        return member


@dataclass
class SourceContribution:
    """What mapping one declared source file put into the graph.

    Kept per SOURCE rather than per facts file because a build maps only the sources it is
    re-resolving: a per-file total could not then be updated without mapping the whole file
    again, and would quietly drift instead.
    """

    edges: int = 0
    external_targets: int = 0
    generated_mapped: int = 0
    buckets: Counter = field(default_factory=Counter)

    def to_json(self) -> dict[str, object]:
        """Return the shape stored in the graph's `facts_status` row."""
        return {
            "edges": self.edges,
            "external_targets": self.external_targets,
            "generated_mapped": self.generated_mapped,
            "buckets": dict(self.buckets),
        }

    @classmethod
    def from_json(cls, payload: dict) -> SourceContribution:
        """Rebuild a contribution stored by an earlier build."""
        return cls(
            edges=int(payload.get("edges", 0)),
            external_targets=int(payload.get("external_targets", 0)),
            generated_mapped=int(payload.get("generated_mapped", 0)),
            buckets=Counter(payload.get("buckets", {})),
        )


@dataclass
class FactsFile:
    """One parsed facts file."""

    path: Path
    relative_path: str
    header: FactsHeader
    sources: dict[str, SourceFacts] = field(default_factory=dict)
    unknown_kinds: Counter = field(default_factory=Counter)
    orphan_facts: int = 0
    outside_root: int = 0
    mtime_ns: int = 0
    size: int = 0
    #: Facts this file's own parse already threw away, with the bucket that says why.
    skipped: list[SkippedFact] = field(default_factory=list)
    #: The `.hwk` templates this file's generated-source facts were mapped back onto. They
    #: are edges this facts file owns just as much as the ones it declares by path, so they
    #: are re-resolved with it when it moves.
    template_paths: set[str] = field(default_factory=set)
    #: Generated source path -> (template it maps onto, whether the template was newer than it).
    #: A later build re-decides that verdict with two `stat` calls instead of re-reading anything.
    generated_templates: dict[str, tuple[str, bool]] = field(default_factory=dict)
    #: Skips this file's own parse decided, which no mapping can change.
    parse_buckets: Counter = field(default_factory=Counter)
    #: Declared source path -> what mapping it contributed. Only the sources this build
    #: actually mapped are in here; the rest keep the accounting the graph already holds.
    contributions: dict[str, SourceContribution] = field(default_factory=dict)

    @property
    def fresh_files(self) -> list[str]:
        """The source files whose current content matches what the facts were derived from."""
        return [path for path, source in self.sources.items() if source.fresh]

    @property
    def stale_files(self) -> list[str]:
        """The source files whose content moved on, or that are gone."""
        return [path for path, source in self.sources.items() if not source.fresh]


@dataclass
class SkippedFact:
    """One fact that did not become an edge, in the bucket that says why."""

    bucket: SkipBucket
    #: What the report groups and lists by: the ref for `UNMAPPED`, because that is what an
    #: emitter fixes, and the source file for every bucket that is about where a fact came
    #: from, because there the ref is a consequence and the file is the lever.
    subject: str
    reason: str
    fact_kind: str
    facts_file: str
    source_path: str


# ---- discovery -----------------------------------------------------------


def _load_config_globs(root: Path) -> tuple[str, ...] | None:
    """Read `[facts] sources` from `.zemble/graph.toml`, or None when it says nothing."""
    config = root / GRAPH_CONFIG_PATH
    if not config.is_file():
        return None
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10 has no tomllib
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            logger.warning("%s ignored: no TOML reader on this interpreter", config)
            return None
    try:
        parsed = tomllib.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("%s could not be read; using the default facts globs", config, exc_info=True)
        return None
    sources = parsed.get("facts", {}).get("sources")
    if not isinstance(sources, list) or not all(isinstance(entry, str) for entry in sources):
        return None
    return tuple(sources)


#: Discovery globs per (root, graph.toml modification time), so the daemon's per-event
#: `matches_facts_glob` does not re-parse the configuration on every changed path.
_GLOB_CACHE: dict[tuple[str, int | None], tuple[str, ...]] = {}


def facts_source_globs(root: Path) -> tuple[str, ...]:
    """Return the globs a workspace's facts files are discovered by."""
    config = root / GRAPH_CONFIG_PATH
    try:
        stamp: int | None = config.stat().st_mtime_ns
    except OSError:
        stamp = None
    key = (str(root), stamp)
    cached = _GLOB_CACHE.get(key)
    if cached is not None:
        return cached
    configured = _load_config_globs(root)
    globs = configured if configured is not None else DEFAULT_SOURCE_GLOBS
    _GLOB_CACHE[key] = globs
    return globs


def _segments(pattern: str) -> list[str]:
    """Split a glob into path segments."""
    return [part for part in pattern.split("/") if part]


def _closure(positions: set[int], pattern: Sequence[str]) -> set[int]:
    """Expand positions across `**`, which is allowed to match nothing."""
    reached = set(positions)
    queue = list(positions)
    while queue:
        position = queue.pop()
        if position < len(pattern) and pattern[position] == "**" and position + 1 not in reached:
            reached.add(position + 1)
            queue.append(position + 1)
    return reached


def _advance(positions: set[int], segment: str, pattern: Sequence[str]) -> set[int]:
    """Consume one path segment, returning the pattern positions still reachable."""
    moved: set[int] = set()
    for position in _closure(positions, pattern):
        if position >= len(pattern):
            continue
        if pattern[position] == "**":
            moved.add(position)
        elif fnmatch.fnmatch(segment, pattern[position]):
            moved.add(position + 1)
    return moved


def _descend(states: Sequence[set[int]], patterns: Sequence[list[str]], name: str) -> list[set[int]] | None:
    """Advance every pattern into a directory, or None when none of them can still match."""
    moved = [_closure(_advance(state, name, pattern), pattern) for state, pattern in zip(states, patterns)]
    alive = any(position < len(pattern) for state, pattern in zip(moved, patterns) for position in state)
    return moved if alive else None


def _walk_for_globs(root: Path, patterns: Sequence[list[str]]) -> Iterator[Path]:
    """Walk the tree yielding files matching any glob, pruning directories that cannot match.

    The pruning is what makes `**/build/zemble/*.jsonl` affordable: a build directory is
    entered, but `build/classes` dies on the very next segment. The walker's usual
    ignored-directory list applies except for the two a facts file lives in by design,
    `build/` and `.zemble/`, and .gitignore is deliberately NOT consulted: generated facts
    are gitignored by construction, so honouring it would make the documented convention
    undiscoverable.
    """
    stack: list[tuple[Path, list[set[int]]]] = [(root, [_closure({0}, pattern) for pattern in patterns])]
    while stack:
        directory, states = stack.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                if entry.name in _SKIPPED_DIRS:
                    continue
                moved = _descend(states, patterns, entry.name)
                if moved is not None:
                    stack.append((Path(entry.path), moved))
            elif entry.is_file(follow_symlinks=False):
                for state, pattern in zip(states, patterns):
                    if len(pattern) in _closure(_advance(state, entry.name, pattern), pattern):
                        yield Path(entry.path)
                        break


def discover_facts_files(root: Path) -> list[Path]:
    """Find every facts file of a workspace, in a stable order."""
    patterns = [_segments(pattern) for pattern in facts_source_globs(root) if _segments(pattern)]
    return sorted({path for path in _walk_for_globs(root, patterns)})


def matches_facts_glob(root: Path, path: Path) -> bool:
    """Return whether a path is one of a workspace's facts files by name alone."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    segments = relative.as_posix().split("/")
    for pattern in (_segments(pattern) for pattern in facts_source_globs(root)):
        if not pattern:
            continue
        state = _closure({0}, pattern)
        for segment in segments:
            state = _closure(_advance(state, segment, pattern), pattern)
        if len(pattern) in state:
            return True
    return False


# ---- loading -------------------------------------------------------------


def _read_header(raw: bytes, path: Path) -> FactsHeader:
    """Parse and validate line 1 of a facts file."""
    try:
        payload = orjson.loads(raw)
    except orjson.JSONDecodeError as error:
        raise FactsFormatError(f"{path}: line 1 is not JSON") from error
    if not isinstance(payload, dict) or "zemble_facts" not in payload:
        raise FactsFormatError(f"{path}: line 1 is not a zemble facts header")
    version = payload.get("zemble_facts")
    if version != FACTS_FORMAT_VERSION:
        raise FactsFormatError(f"{path}: zemble_facts version {version!r}, this zemble reads {FACTS_FORMAT_VERSION}")
    return FactsHeader(
        tool=str(payload.get("tool", "unknown")),
        tool_version=str(payload.get("tool_version", "")),
        generated_at=str(payload.get("generated_at", "")),
        language=str(payload.get("language", "")),
        root=str(payload.get("root", ".")),
    )


def file_sha256(path: Path) -> str | None:
    """Return the hex sha256 of a file's content, or None when it cannot be read."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


#: Resolved (facts root, written path) -> workspace-relative path, or None when it is outside.
#: Every fact line of one source file writes the same path, and `Path.resolve` is a realpath
#: syscall per call, so without this a facts file's own body costs a million of them.
_RELATIVE_CACHE: dict[tuple[str, str, str], str | None] = {}


def _relative_to_workspace(root: Path, facts_root: Path, declared: str) -> str | None:
    """Turn a path written in a facts file into a workspace-relative path."""
    key = (str(root), str(facts_root), declared)
    if key in _RELATIVE_CACHE:
        return _RELATIVE_CACHE[key]
    candidate = (facts_root / declared).resolve()
    try:
        resolved: str | None = candidate.relative_to(root).as_posix()
    except ValueError:
        resolved = None
    _RELATIVE_CACHE[key] = resolved
    return resolved


def load_facts_file(path: Path, root: Path) -> FactsFile:
    """Read one facts file and decide which of its source files are still fresh.

    :param path: The facts file to read.
    :param root: The workspace root every source path is reported relative to.
    :return: The parsed file, with per-source freshness already decided.
    :raises FactsFormatError: If the header is missing, unreadable or of another version.
    """
    try:
        lines = path.read_bytes().split(b"\n")
    except OSError as error:
        raise FactsFormatError(f"{path}: cannot be read") from error
    if not lines or not lines[0].strip():
        raise FactsFormatError(f"{path}: is empty")
    header = _read_header(lines[0], path)
    facts_root = Path(header.root)
    if not facts_root.is_absolute():
        facts_root = (path.parent / facts_root).resolve()
    stat = path.stat()
    loaded = FactsFile(
        path=path,
        relative_path=_display_path(root, path),
        header=header,
        mtime_ns=stat.st_mtime_ns,
        size=stat.st_size,
    )
    _read_body(lines[1:], loaded, root, facts_root)
    return loaded


def _display_path(root: Path, path: Path) -> str:
    """Return a path relative to the workspace root when it is inside it."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_body(lines: Sequence[bytes], loaded: FactsFile, root: Path, facts_root: Path) -> None:
    """Group every fact under the source file it was declared for."""
    current: SourceFacts | None = None
    outside: str | None = None
    for raw in lines:
        if not raw.strip():
            continue
        try:
            payload = orjson.loads(raw)
        except orjson.JSONDecodeError:
            loaded.unknown_kinds["(not json)"] += 1
            continue
        kind = payload.get("t") if isinstance(payload, dict) else None
        if kind not in _KNOWN_FACT_KINDS:
            loaded.unknown_kinds[str(kind)] += 1
            continue
        if kind == "file":
            current = _declare_file(payload, loaded, root, facts_root)
            outside = str(payload.get("path", "")) if current is None else None
            continue
        target = _target_of(payload, loaded, current, root, facts_root)
        if target is None:
            _record_unattached(payload, kind, loaded, outside, root, facts_root)
            continue
        target.facts.append(payload)


def _record_unattached(
    payload: dict, kind: str, loaded: FactsFile, outside: str | None, root: Path, facts_root: Path
) -> None:
    """Bucket a fact that landed on no source file: outside the index, or a real orphan.

    A `file` line for a path outside the indexed workspace is skipped, and so is every fact
    that follows it. Those facts are not orphans - they were declared properly, for a file
    this workspace does not contain - so they are counted where a reader can act on them.
    """
    written = payload.get("path")
    if isinstance(written, str) and written:
        declared = written if _relative_to_workspace(root, facts_root, written) is None else None
    else:
        declared = outside
    if declared is None:
        loaded.orphan_facts += 1
        return
    loaded.parse_buckets[SkipBucket.OUTSIDE_INDEX.value] += 1
    loaded.skipped.append(
        SkippedFact(
            bucket=SkipBucket.OUTSIDE_INDEX,
            subject=declared,
            reason="the source file is outside the indexed workspace",
            fact_kind=str(_EDGE_KIND_BY_FACT[kind].value) if kind in _EDGE_KIND_BY_FACT else kind,
            facts_file=loaded.relative_path,
            source_path=declared,
        )
    )


def _declare_file(payload: dict, loaded: FactsFile, root: Path, facts_root: Path) -> SourceFacts | None:
    """Handle a `file` line: resolve its path and hash the current content."""
    declared = str(payload.get("path", ""))
    relative = _relative_to_workspace(root, facts_root, declared) if declared else None
    if relative is None:
        loaded.outside_root += 1
        return None
    expected = str(payload.get("sha256", ""))
    actual = file_sha256(root / relative)
    if actual is None:
        source = SourceFacts(path=relative, declared_sha256=expected, fresh=False, reason="file is gone")
    elif actual != expected:
        source = SourceFacts(path=relative, declared_sha256=expected, fresh=False, reason="content changed")
    else:
        source = SourceFacts(path=relative, declared_sha256=expected, fresh=True)
    existing = loaded.sources.get(relative)
    if existing is not None:
        # One facts file naming the same source twice: keep the facts together.
        existing.facts.extend(source.facts)
        return existing
    loaded.sources[relative] = source
    return source


def _target_of(
    payload: dict, loaded: FactsFile, current: SourceFacts | None, root: Path, facts_root: Path
) -> SourceFacts | None:
    """Find the source file a fact belongs to: its own `path`, else the last `file` line.

    `override`, `extends`, `implements` and `annotation` facts carry no path at all, which
    is why the last `file` line is a real part of the contract and not a convenience.
    """
    written = payload.get("path")
    if isinstance(written, str) and written:
        relative = _relative_to_workspace(root, facts_root, written)
        return loaded.sources.get(relative) if relative is not None else None
    return current


# ---- ref mapping ---------------------------------------------------------


@dataclass(frozen=True)
class ParsedRef:
    """A ref split into the pieces a mapper matches on."""

    type_name: str
    member: str | None = None
    params: tuple[str, ...] | None = None
    is_constructor: bool = False


@dataclass(frozen=True)
class MappedRef:
    """What a mapper made of one ref."""

    symbol_id: str | None
    dst_name: str
    #: True when nothing in the workspace declares the ref's type: a JDK or jar target.
    external: bool = False
    reason: str = ""

    @property
    def unmapped(self) -> bool:
        """True when the ref names something the workspace should have had but did not."""
        return self.symbol_id is None and not self.external


class DeclaredSymbols(Protocol):
    """Where a `symbol` fact placed a ref, asked one ref at a time.

    A plain dict satisfies it, and so does a table in the graph: a build that reads one facts
    file must still be able to reach the `symbol` facts of every other one, and parsing them
    all to answer a handful of refs is what this exists to avoid.
    """

    def get(self, ref: str) -> tuple[str, int] | None:
        """Return the file path and line a `symbol` fact gave a ref, or None."""
        ...


@runtime_checkable
class RefMapper(Protocol):
    """Maps a language's refs onto zemble symbol ids.

    This is the only language-aware piece of the overlay: everything else works on refs
    as opaque strings.
    """

    def map_ref(self, ref: str) -> MappedRef:
        """Map one ref onto a workspace symbol, an external target, or nothing."""
        ...


def _simple(name: str) -> str:
    """Reduce a possibly qualified type name to its last segment, keeping array brackets."""
    base, _, arrays = name.partition("[")
    suffix = f"[{arrays}" if arrays else ""
    return base.rsplit(".", 1)[-1].strip() + suffix


def parse_java_ref(ref: str) -> ParsedRef:
    """Split a Java-flavoured ref into its type, member and erased parameter types."""
    type_name, separator, member = ref.partition("#")
    if not separator:
        return ParsedRef(type_name=type_name.strip())
    member = member.strip()
    if "(" not in member:
        return ParsedRef(type_name=type_name.strip(), member=member)
    name, _, rest = member.partition("(")
    inside = rest.rstrip().rstrip(")")
    params = tuple(part.strip() for part in inside.split(",") if part.strip())
    return ParsedRef(type_name=type_name.strip(), member=name, params=params, is_constructor=name == "<init>")


#: Member names javac uses for the code that is not in any method: static and instance
#: initializers, including field initializers. zemble has no symbol for either, so both
#: are attributed to the enclosing TYPE.
_INITIALIZER_MEMBERS = frozenset({"<clinit>", "<instance-init>"})
_FIELD_KINDS = frozenset({SymbolKind.FIELD, SymbolKind.ENUM_CONSTANT})


@dataclass(frozen=True)
class _TypeMatch:
    """The type a ref's left-hand side named, or why it named nothing."""

    symbol: Symbol | None = None
    external: bool = False
    reason: str = ""


class JavaRefMapper:
    """Maps Java refs (`pkg.Type#name(erased.Params)`) onto zemble symbols.

    The type is resolved first and the member is then looked up among that type's own
    declarations, so the two flavours of type name a Java front end writes - dotted
    (`pkg.Outer.Inner`) and javac's flat form (`pkg.Outer$1`, `pkg.Outer$1Local`) - are
    the only place the difference has to be understood. Overloads are separated by arity
    and then by the SIMPLE names of the erased parameter types, because zemble records
    parameter types as the source wrote them. A ref whose type no workspace file declares
    is an external target; a ref whose type is known but whose member is not is reported
    unmapped, never guessed at.
    """

    language = "java"

    def __init__(self, symbols: Iterable[Symbol], declared: DeclaredSymbols | None = None) -> None:
        """Index the workspace symbols a ref can land on.

        :param symbols: Every symbol in the workspace.
        :param declared: Optional `symbol` facts, ref -> (file path, line), used as a second
            rung when name matching fails.
        """
        self.by_id: dict[str, Symbol] = {}
        self.by_qualified: dict[str, list[Symbol]] = {}
        self.by_container: dict[str, list[Symbol]] = {}
        self.by_position: dict[tuple[str, int], list[Symbol]] = {}
        for symbol in symbols:
            self.by_id[symbol.id] = symbol
            self.by_qualified.setdefault(symbol.qualified_name, []).append(symbol)
            self.by_position.setdefault((symbol.file_path, symbol.start_line), []).append(symbol)
            if symbol.container_id:
                self.by_container.setdefault(symbol.container_id, []).append(symbol)
        self.declared: DeclaredSymbols = declared if declared is not None else {}
        self._numbered: dict[str, list[Symbol]] = {}
        self._locals: dict[tuple[str, str], list[Symbol]] = {}
        self._index_flat_names()

    # ---- flat (javac) type names ----------------------------------------

    def _index_flat_names(self) -> None:
        """Group the declarations javac gives a `$N` name under their outermost type.

        javac numbers anonymous classes, enum-constant bodies and local classes per
        OUTERMOST class in source order, which is the only handle there is: zemble names
        an anonymous class after the line it starts on and keeps an enum constant's body
        on the constant itself. Order is therefore the mapping, and it is best effort -
        a ref that does not land is reported unmapped rather than guessed at.
        """
        for symbol in self.by_id.values():
            top = self._outermost(symbol)
            if top is None or top.id == symbol.id:
                continue
            if symbol.name.startswith("$anon@") or (symbol.kind is SymbolKind.ENUM_CONSTANT and self._has_body(symbol)):
                self._numbered.setdefault(top.qualified_name, []).append(symbol)
            elif symbol.kind in TYPE_KINDS and self._is_local(symbol):
                self._locals.setdefault((top.qualified_name, symbol.name), []).append(symbol)
        for group in (*self._numbered.values(), *self._locals.values()):
            group.sort(key=lambda symbol: (symbol.start_line, symbol.id))

    def _outermost(self, symbol: Symbol) -> Symbol | None:
        """Return the outermost type declaration enclosing a symbol."""
        found: Symbol | None = None
        current: Symbol | None = symbol
        while current is not None:
            if current.kind in TYPE_KINDS:
                found = current
            current = self.by_id.get(current.container_id) if current.container_id else None
        return found

    def _has_body(self, symbol: Symbol) -> bool:
        """Return whether an enum constant declares members of its own."""
        return bool(self.by_container.get(symbol.id))

    def _is_local(self, symbol: Symbol) -> bool:
        """Return whether a type is declared inside a method or constructor body."""
        container = self.by_id.get(symbol.container_id) if symbol.container_id else None
        return container is not None and container.kind in CALLABLE_KINDS

    def _flat_type(self, type_name: str) -> _TypeMatch:
        """Resolve a javac flat name (`pkg.Top$1`, `pkg.Top$1Local`) to a symbol."""
        top, _, tail = type_name.partition("$")
        digits = ""
        while tail[len(digits) :] and tail[len(digits)].isdigit():
            digits += tail[len(digits)]
        rest = tail[len(digits) :]
        if not digits or "$" in rest:
            return _TypeMatch(reason="anonymous flat name")
        group = self._numbered.get(top, []) if not rest else self._locals.get((top, rest), [])
        index = int(digits) - 1
        if len(group) == 1:
            return _TypeMatch(symbol=group[0])
        if 0 <= index < len(group):
            return _TypeMatch(symbol=group[index])
        return _TypeMatch(reason="anonymous flat name")

    def _type(self, type_name: str) -> _TypeMatch:
        """Resolve the left-hand side of a ref to the type it names."""
        if "$" in type_name:
            return self._flat_type(type_name)
        found = [symbol for symbol in self.by_qualified.get(type_name, []) if symbol.kind in TYPE_KINDS]
        if len(found) == 1:
            return _TypeMatch(symbol=found[0])
        if len(found) > 1:
            return _TypeMatch(reason=f"{len(found)} types answer to {type_name}")
        return _TypeMatch(external=True)

    # ---- the ladder ------------------------------------------------------

    def map_ref(self, ref: str) -> MappedRef:
        """Map one Java ref onto a workspace symbol, an external target, or nothing."""
        parsed = parse_java_ref(ref)
        found = self._type(parsed.type_name)
        if found.symbol is None:
            positioned = self._by_declared_position(ref, None)
            if positioned is not None:
                return MappedRef(symbol_id=positioned, dst_name=_ref_name(parsed))
            if found.external:
                return MappedRef(None, dst_name=ref, external=True)
            return MappedRef(None, dst_name=_ref_name(parsed), reason=found.reason)
        if parsed.member is None:
            return MappedRef(symbol_id=found.symbol.id, dst_name=found.symbol.name)
        return self._member(ref, parsed, found.symbol)

    def _member(self, ref: str, parsed: ParsedRef, owner: Symbol) -> MappedRef:
        """Find the member a ref names among one type's own declarations."""
        name = _ref_name(parsed)
        if parsed.member in _INITIALIZER_MEMBERS:
            # A field or static initializer has no symbol of its own; the type it runs for
            # is the honest source of the edge, and is where a reader would look anyway.
            return MappedRef(symbol_id=owner.id, dst_name=owner.name)
        members = self.by_container.get(owner.id, [])
        kinds = CALLABLE_KINDS if parsed.params is not None else _FIELD_KINDS
        candidates = [symbol for symbol in members if symbol.kind in kinds and symbol.name == name]
        if parsed.params is not None:
            candidates = _narrow_by_params(candidates, parsed.params)
        if len(candidates) == 1:
            return MappedRef(symbol_id=candidates[0].id, dst_name=name)
        positioned = self._by_declared_position(ref, candidates or None)
        if positioned is not None:
            return MappedRef(symbol_id=positioned, dst_name=name)
        if len(candidates) > 1:
            return MappedRef(None, dst_name=name, reason=f"{len(candidates)} members of {owner.qualified_name} match")
        accessor = self._record_accessor(parsed, owner, members)
        if accessor is not None:
            return MappedRef(symbol_id=accessor.id, dst_name=name)
        if parsed.is_constructor and not any(symbol.kind is SymbolKind.CONSTRUCTOR for symbol in members):
            # A type that declares no constructor keeps its implicit one on the type itself,
            # which is where zemble's own resolver puts `new Foo()` too.
            return MappedRef(symbol_id=owner.id, dst_name=owner.name)
        return MappedRef(None, dst_name=name, reason=f"no {name} in {owner.qualified_name}")

    @staticmethod
    def _record_accessor(parsed: ParsedRef, owner: Symbol, members: list[Symbol]) -> Symbol | None:
        """Match `rec.component()` onto the component itself.

        A record's accessors are implicit, so a compiler emits calls to methods no source
        file declares. zemble records a component as a FIELD on the record, which is both
        the only symbol there is and the place a reader would look.
        """
        if parsed.params != () or owner.kind is not SymbolKind.RECORD:
            return None
        found = [symbol for symbol in members if symbol.kind is SymbolKind.FIELD and symbol.name == parsed.member]
        return found[0] if len(found) == 1 else None

    def knows_type(self, type_name: str) -> bool:
        """Return whether any workspace file declares a type with this qualified name."""
        return self._type(type_name).symbol is not None

    def _by_declared_position(self, ref: str, candidates: list[Symbol] | None) -> str | None:
        """Use a `symbol` fact's file and line to pick a symbol name matching could not."""
        position = self.declared.get(ref)
        if position is None:
            return None
        found = self.by_position.get(position, [])
        if candidates:
            allowed = {symbol.id for symbol in candidates}
            found = [symbol for symbol in found if symbol.id in allowed]
        return found[0].id if len(found) == 1 else None


def _ref_name(parsed: ParsedRef) -> str:
    """Return the name a ref's target is written as in an edge."""
    if parsed.member is None:
        return _simple(parsed.type_name)
    if parsed.is_constructor:
        return _simple(parsed.type_name)
    return parsed.member


def _narrow_by_params(candidates: list[Symbol], params: tuple[str, ...]) -> list[Symbol]:
    """Keep the callables whose erased parameter types match, by arity first then by name."""
    same_arity = [symbol for symbol in candidates if len(symbol.param_types) == len(params)]
    if len(same_arity) <= 1:
        return same_arity
    wanted = [_simple(param) for param in params]
    return [symbol for symbol in same_arity if [_simple(param) for param in symbol.param_types] == wanted]


#: The ref flavour of each language a facts file may declare.
MAPPER_FACTORIES = {"java": JavaRefMapper}


# ---- the overlay ---------------------------------------------------------


#: Every symbol of the workspace, or a callable producing them. Mapping refs needs the whole
#: table, and a build that maps nothing must never pay for materialising it.
SymbolSource = Iterable[Symbol] | Callable[[], Iterable[Symbol]]


@dataclass
class FactsOverlay:
    """Every fact edge a workspace's facts files contribute, grouped by source file."""

    root: Path
    files: list[FactsFile] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    #: Every fact that did not become an edge, in the bucket that says why.
    skipped: list[SkippedFact] = field(default_factory=list)
    #: Edges kept with no `dst_id` because the target's type is a JDK or jar type.
    external_targets: int = 0
    #: Workspace-relative source path -> the edges replacing that file's extracted ones.
    edges: dict[str, list[Edge]] = field(default_factory=dict)
    #: Workspace-relative source path -> the tools whose fresh facts cover it.
    tools: dict[str, set[str]] = field(default_factory=dict)
    #: Maps generated Hawkeye Java back onto templates; None when nothing needed it.
    generated: GeneratedSourceMapper | None = None
    #: Covered file -> the edge kinds the overlay owns for it. Absent means all four.
    owned_kinds: dict[str, frozenset[EdgeKind]] = field(default_factory=dict)
    #: How many generated-source facts were mapped back onto a template.
    generated_mapped: int = 0
    #: Source path -> the prefix that keeps it out of the index, or "" when it is indexed.
    _ignored_prefixes: dict[str, str] = field(default_factory=dict, repr=False)
    #: Covered file -> the identities of the edges already collected for it.
    _edge_keys: dict[str, set[tuple]] = field(default_factory=dict, repr=False)
    #: Facts file -> the declared sources of it this overlay has already mapped. A build maps
    #: only the sources its targets need, so the rest keep the accounting the graph holds.
    mapped_sources: dict[str, set[str]] = field(default_factory=dict)
    #: Ref mappers built so far, per language, so a second round of mapping reuses them.
    mappers: dict[str, RefMapper] = field(default_factory=dict, repr=False)
    _symbols: list[Symbol] | None = field(default=None, repr=False)

    @property
    def materialised_symbols(self) -> list[Symbol] | None:
        """The workspace symbols, if mapping already had to read them; None otherwise."""
        return self._symbols

    def symbols(self, source: SymbolSource) -> list[Symbol]:
        """Materialise the workspace symbols once, however many rounds of mapping there are."""
        if self._symbols is None:
            self._symbols = list(source() if callable(source) else source)
            self.generated = GeneratedSourceMapper(self.root, self._symbols)
        return self._symbols

    @property
    def mapped_files(self) -> set[str]:
        """The facts files this overlay mapped anything of."""
        return set(self.mapped_sources)

    def kinds_owned(self, file_path: str) -> frozenset[EdgeKind]:
        """Return the edge kinds whose extracted edges this file's facts replace.

        A Java file's facts own all four kinds, because the emitter compiled the file itself.
        A template's do not: they arrive through a source map from the class the template was
        compiled INTO, which knows what the compiled code calls and nothing about what the
        template extends or renders - so only `CALLS` is replaced there, and the extractor
        keeps the rest. A file with no fresh facts owns nothing and keeps every edge it had.
        """
        if file_path not in self.edges:
            return frozenset()
        return self.owned_kinds.get(file_path, OVERLAY_KINDS)

    @property
    def template_targets(self) -> set[str]:
        """The templates that carry mapped edges, which are not declared by any `file` line."""
        return {path for path, kinds in self.owned_kinds.items() if kinds != OVERLAY_KINDS}

    @property
    def unmapped(self) -> list[SkippedFact]:
        """Only the refs the workspace should have answered and did not."""
        return [entry for entry in self.skipped if entry.bucket is SkipBucket.UNMAPPED]

    def bucket_counts(self) -> dict[str, int]:
        """How many facts each bucket holds, every bucket present even at zero."""
        counted = {bucket.value: 0 for bucket in SkipBucket}
        for entry in self.skipped:
            counted[entry.bucket.value] += 1
        return counted

    def source_bucket(self, source_path: str) -> tuple[SkipBucket, str] | None:
        """Bucket a fact by WHERE its source file lives, or None when the index covers it.

        Generated code is the case this exists for: a `build/generated-sources/...` file is
        compiled, so a javac emitter has facts about it, but the index never walked it, so
        every ref into it is unresolvable by construction rather than by mistake.
        """
        prefix = self._ignored_prefixes.get(source_path)
        if prefix is None:
            prefix = ignored_prefix(self.root, source_path) or ""
            self._ignored_prefixes[source_path] = prefix
        if not prefix:
            return None
        named = prefix if prefix == source_path else f"{prefix}/..."
        return SkipBucket.SOURCE_IGNORED, f"source is generated/ignored ({named})"

    def covers(self, file_path: str) -> bool:
        """Return whether a source file has fresh facts, so its extracted edges are dropped."""
        return file_path in self.edges

    @property
    def covered_files(self) -> set[str]:
        """Every source file the overlay owns the edges of."""
        return set(self.edges)

    @property
    def fresh_sources(self) -> set[str]:
        """Every DECLARED source file whose content still hashes to what the facts describe.

        Not the same set as `covered_files`: a generated Hawkeye class is a fresh declared
        source whose edges land on the template it was compiled from, never on itself.
        """
        return {path for loaded in self.files for path in loaded.fresh_files}

    @property
    def declared_files(self) -> set[str]:
        """Every source file any facts file mentions, fresh or stale."""
        return {path for loaded in self.files for path in loaded.sources}

    def stats(self) -> dict[str, object]:
        """Return a JSON-ready summary of what was loaded."""
        return {
            "facts_files": len(self.files),
            "errors": [{"path": path, "error": message} for path, message in self.errors],
            "files_declared": len(self.declared_files),
            "files_fresh": len(self.fresh_sources),
            "files_stale": len(self.declared_files) - len(self.fresh_sources),
            "edges": sum(len(edges) for edges in self.edges.values()),
            "external_targets": self.external_targets,
            "skipped": self.bucket_counts(),
            "unmapped": self.bucket_counts()[SkipBucket.UNMAPPED.value],
            "generated_mapped": self.generated_mapped,
            "generated_templates": len(self.template_targets),
        }


#: What one facts file must be mapped for: a set of declared source paths, or None for all
#: of them. A build asks for only the sources it is re-resolving, so the rest of a facts file
#: keeps the edges and the accounting the graph already holds for it.
MappingRequest = dict[str, set[str] | None]


def load_overlay(
    root: Path, symbols: SymbolSource, *, paths: Sequence[Path] | None = None, mapped: MappingRequest | None = None
) -> FactsOverlay:
    """Discover, read and map the facts files of a workspace.

    :param root: The workspace root.
    :param symbols: Every symbol zemble extracted, or a callable producing them.
    :param paths: The facts files to read; None discovers them.
    :param mapped: Per facts file, the sources to turn into edges; None maps every one whole.
    :return: The overlay, ready to be applied to a resolved edge list.
    """
    overlay = read_facts_files(root, paths)
    request = {loaded.relative_path: None for loaded in overlay.files} if mapped is None else mapped
    map_facts_files(overlay, symbols, request)
    return overlay


def read_facts_files(root: Path, paths: Sequence[Path] | None = None) -> FactsOverlay:
    """Parse facts files into an overlay, deciding per source file whether it is still fresh."""
    overlay = FactsOverlay(root=root)
    for path in discover_facts_files(root) if paths is None else paths:
        try:
            overlay.files.append(load_facts_file(path, root))
        except FactsFormatError as error:
            overlay.errors.append((_display_path(root, path), str(error)))
            logger.warning("Ignoring facts file: %s", error)
    return overlay


def map_facts_files(
    overlay: FactsOverlay,
    symbols: SymbolSource,
    request: MappingRequest,
    declared: DeclaredSymbols | None = None,
) -> None:
    """Turn the requested sources into edges, recording what could not be mapped.

    Callable more than once: a build learns from one round of mapping which further facts
    files cover a file it is now re-resolving, and asks for those too. Nothing here reads the
    symbol table until a source that is actually fresh has to be mapped, because a source
    whose content moved on contributes no edge and needs no lookup to say so.

    :param declared: Where a `symbol` fact placed a ref, for the whole workspace. None falls
        back to the `symbol` facts of the files this overlay itself parsed, which is only the
        whole workspace when it parsed all of them.
    """
    work: list[tuple[FactsFile, list[SourceFacts]]] = []
    for loaded in overlay.files:
        if loaded.relative_path not in request:
            continue
        only = request[loaded.relative_path]
        done = overlay.mapped_sources.get(loaded.relative_path, set())
        chosen = [
            source
            for source in loaded.sources.values()
            if (only is None or source.path in only) and source.path not in done
        ]
        if chosen:
            work.append((loaded, chosen))
    if not work:
        return
    if not any(source.fresh for _, chosen in work for source in chosen):
        # Nothing fresh to map: record the staleness, which the parse alone already decided.
        for loaded, chosen in work:
            _mark_mapped(overlay, loaded, chosen)
            _map_file(overlay, loaded, _NO_MAPPER, {}, chosen)
        return
    symbol_list = overlay.symbols(symbols)
    lines = {symbol.id: symbol.start_line for symbol in symbol_list}
    for loaded, chosen in work:
        factory = MAPPER_FACTORIES.get(loaded.header.language)
        if factory is None:
            overlay.errors.append((loaded.relative_path, f"no ref mapper for language {loaded.header.language!r}"))
            continue
        mapper = overlay.mappers.get(loaded.header.language)
        if mapper is None:
            found = declared if declared is not None else _declared_symbols(overlay.files, loaded.header.language)
            mapper = factory(symbol_list, found)
            overlay.mappers[loaded.header.language] = mapper
        _mark_mapped(overlay, loaded, chosen)
        _map_file(overlay, loaded, mapper, lines, chosen)


def _mark_mapped(overlay: FactsOverlay, loaded: FactsFile, chosen: list[SourceFacts]) -> None:
    """Record which sources of a facts file are now mapped, once per facts file."""
    done = overlay.mapped_sources.setdefault(loaded.relative_path, set())
    if not done:
        overlay.skipped.extend(loaded.skipped)
    done.update(source.path for source in chosen)


class _NoMapper:
    """Stands in where a source is known to be stale, so no ref is ever mapped through it."""

    def map_ref(self, ref: str) -> MappedRef:  # pragma: no cover - a stale source maps nothing
        """Refuse every ref: reaching this would mean a stale source was mapped after all."""
        raise AssertionError(f"a stale source cannot map {ref}")


_NO_MAPPER = _NoMapper()


def symbol_facts(loaded: FactsFile) -> Iterator[tuple[str, str, int]]:
    """Yield every `symbol` fact of one parsed facts file as (ref, source path, line).

    In the order it was written, because two facts of one file may name the same ref and the
    last one is the one a reader takes.
    """
    for source in loaded.sources.values():
        for payload in source.facts:
            ref, line = payload.get("ref"), payload.get("line")
            if payload.get("t") == "symbol" and isinstance(ref, str) and isinstance(line, int):
                yield ref, source.path, line


def _declared_symbols(files: list[FactsFile], language: str) -> dict[str, tuple[str, int]]:
    """Collect every `symbol` fact of one language as ref -> (file path, line).

    Sorted by facts file, because two files may declare the same ref and a build must not
    depend on the order it happened to read them in.
    """
    declared: dict[str, tuple[str, int]] = {}
    for loaded in sorted(files, key=lambda entry: entry.relative_path):
        if loaded.header.language != language:
            continue
        for ref, path, line in symbol_facts(loaded):
            declared[ref] = (path, line)
    return declared


def _map_file(
    overlay: FactsOverlay, loaded: FactsFile, mapper: RefMapper, lines: dict[str, int], sources: list[SourceFacts]
) -> None:
    """Map the chosen sources of one facts file into edges, merging with what is collected."""
    for source in sources:
        loaded.contributions[source.path] = SourceContribution()
        if not source.fresh:
            _record_stale(overlay, loaded, source)
            continue
        if overlay.generated is not None and overlay.generated.recognises(source.path):
            _map_generated_source(overlay, loaded, source, mapper)
            continue
        overlay.tools.setdefault(source.path, set()).add(loaded.header.tool)
        for payload in source.facts:
            kind = _EDGE_KIND_BY_FACT.get(str(payload.get("t")))
            if kind is None:
                continue
            edge = _edge_from_fact(payload, kind, source, loaded, mapper, overlay, lines)
            if edge is not None and _collect(overlay, source.path, edge):
                loaded.contributions[source.path].edges += 1


def _skip(overlay: FactsOverlay, loaded: FactsFile, entry: SkippedFact) -> None:
    """Record one fact that never became an edge, on the overlay and on its source's account."""
    overlay.skipped.append(entry)
    contribution = loaded.contributions.get(entry.source_path)
    if contribution is not None:
        contribution.buckets[entry.bucket.value] += 1


def _collect(overlay: FactsOverlay, file_path: str, edge: Edge) -> bool:
    """Add an edge to the file it belongs to, dropping one two facts files both wrote."""
    collected = overlay.edges.setdefault(file_path, [])
    seen = overlay._edge_keys.setdefault(file_path, set())  # noqa: SLF001 - same module
    key = _edge_key(edge)
    if key in seen:
        return False
    seen.add(key)
    collected.append(edge)
    return True


def _map_generated_source(overlay: FactsOverlay, loaded: FactsFile, source: SourceFacts, mapper: RefMapper) -> None:
    """Map one generated Hawkeye class's facts back onto the template it was compiled from.

    Only `call` facts make the trip. A generated class's supertypes and overrides are facts
    about `CompiledTemplate` machinery, not about the template: attributing them to the
    template symbol would put `extends CompiledTemplate` where the template's own
    `{% extends %}` belongs, so they are counted instead of invented.
    """
    assert overlay.generated is not None  # noqa: S101 - guarded by the caller
    for payload in source.facts:
        kind = _EDGE_KIND_BY_FACT.get(str(payload.get("t")))
        if kind is None:
            continue
        if kind is not EdgeKind.CALLS:
            _record_generated_skip(overlay, loaded, source, kind, "only calls map back through a template source map")
            continue
        line = payload.get("line")
        mapped = overlay.generated.resolve(source.path, line if isinstance(line, int) else 0)
        loaded.generated_templates[source.path] = (mapped.template_path or "", mapped.stale)
        if not mapped.mapped:
            bucket = SkipBucket.STALE if mapped.stale else SkipBucket.GENERATED_NO_TEMPLATE
            _record_generated_skip(overlay, loaded, source, kind, mapped.reason, bucket=bucket)
            continue
        edge = _template_edge(payload, mapped, loaded, mapper, overlay, source)
        if edge is None:
            continue
        template_path = mapped.template_path or ""
        overlay.owned_kinds[template_path] = frozenset({EdgeKind.CALLS})
        overlay.tools.setdefault(template_path, set()).add(loaded.header.tool)
        loaded.template_paths.add(template_path)
        # Counted per FACT, not per edge: several generated call sites collapse onto one
        # template line, and the count answers "how much did the source map recover".
        overlay.generated_mapped += 1
        loaded.contributions[source.path].generated_mapped += 1
        if _collect(overlay, template_path, edge):
            loaded.contributions[source.path].edges += 1


def _template_edge(
    payload: dict,
    mapped: GeneratedMapping,
    loaded: FactsFile,
    mapper: RefMapper,
    overlay: FactsOverlay,
    source: SourceFacts,
) -> Edge | None:
    """Build the template's call edge from a fact about the class it was compiled into."""
    to_ref = payload.get("to")
    from_ref = payload.get("from")
    if not isinstance(to_ref, str) or not isinstance(from_ref, str):
        loaded.orphan_facts += 1
        return None
    target = mapper.map_ref(to_ref)
    if target.unmapped:
        _skip(
            overlay,
            loaded,
            SkippedFact(
                bucket=SkipBucket.UNMAPPED,
                subject=to_ref,
                reason=target.reason,
                fact_kind=EdgeKind.CALLS.value,
                facts_file=loaded.relative_path,
                source_path=source.path,
            ),
        )
        return None
    if target.external:
        overlay.external_targets += 1
        loaded.contributions[source.path].external_targets += 1
    parsed = parse_java_ref(to_ref)
    return Edge(
        src_id=mapped.symbol_id or "",
        dst_name=target.dst_name,
        kind=EdgeKind.CALLS,
        line=mapped.line,
        dst_id=target.symbol_id,
        resolution=Resolution.EXACT if target.symbol_id else Resolution.UNRESOLVED,
        arity=len(parsed.params) if parsed.params is not None else -1,
        is_new=parsed.is_constructor,
        source=loaded.header.tool,
        origin_ref=from_ref,
    )


def _record_generated_skip(
    overlay: FactsOverlay,
    loaded: FactsFile,
    source: SourceFacts,
    kind: EdgeKind,
    reason: str,
    bucket: SkipBucket = SkipBucket.GENERATED_NO_TEMPLATE,
) -> None:
    """Count one generated-source fact that never reached a template."""
    _skip(
        overlay,
        loaded,
        SkippedFact(
            bucket=bucket,
            subject=source.path,
            reason=reason,
            fact_kind=kind.value,
            facts_file=loaded.relative_path,
            source_path=source.path,
        ),
    )


def _record_stale(overlay: FactsOverlay, loaded: FactsFile, source: SourceFacts) -> None:
    """Count the edge facts a source file lost because its content moved on."""
    classified = overlay.source_bucket(source.path)
    bucket, reason = classified if classified is not None else (SkipBucket.STALE, source.reason)
    for payload in source.facts:
        kind = _EDGE_KIND_BY_FACT.get(str(payload.get("t")))
        if kind is None:
            continue
        _skip(
            overlay,
            loaded,
            SkippedFact(
                bucket=bucket,
                subject=source.path,
                reason=reason,
                fact_kind=kind.value,
                facts_file=loaded.relative_path,
                source_path=source.path,
            ),
        )


def _edge_key(edge: Edge) -> tuple:
    """The identity two facts files must agree on for their edges to be one edge."""
    return (edge.src_id, edge.dst_id, edge.dst_name, edge.kind.value, edge.line, edge.arity, edge.is_new)


def _edge_from_fact(
    payload: dict,
    kind: EdgeKind,
    source: SourceFacts,
    loaded: FactsFile,
    mapper: RefMapper,
    overlay: FactsOverlay,
    lines: dict[str, int],
) -> Edge | None:
    """Build one edge from one fact, or record why it could not be built."""
    from_ref = payload.get("from")
    to_ref = payload.get("to")
    if not isinstance(from_ref, str) or not isinstance(to_ref, str):
        loaded.orphan_facts += 1
        return None
    origin = mapper.map_ref(from_ref)
    if origin.symbol_id is None:
        # Where the fact came FROM decides the bucket: a ref into a file the index never
        # walked is not an unmapped ref, and counting it as one reads as a defect.
        classified = overlay.source_bucket(source.path)
        bucket, reason = (
            classified
            if classified is not None
            else (SkipBucket.UNMAPPED, origin.reason or "the source of an edge is outside the workspace")
        )
        _skip(
            overlay,
            loaded,
            SkippedFact(
                bucket=bucket,
                subject=from_ref if bucket is SkipBucket.UNMAPPED else source.path,
                reason=reason,
                fact_kind=kind.value,
                facts_file=loaded.relative_path,
                source_path=source.path,
            ),
        )
        return None
    target = mapper.map_ref(to_ref)
    if target.unmapped:
        _skip(
            overlay,
            loaded,
            SkippedFact(
                bucket=SkipBucket.UNMAPPED,
                subject=to_ref,
                reason=target.reason,
                fact_kind=kind.value,
                facts_file=loaded.relative_path,
                source_path=source.path,
            ),
        )
        return None
    if target.external:
        overlay.external_targets += 1
        loaded.contributions[source.path].external_targets += 1
    # `override`, `extends` and `implements` facts carry no line: the declaration's own
    # line is the honest answer, and is what the derived tree-sitter edges use too.
    line = payload.get("line")
    if not isinstance(line, int):
        line = lines.get(origin.symbol_id, 0)
    parsed = parse_java_ref(to_ref)
    return Edge(
        src_id=origin.symbol_id,
        dst_name=target.dst_name,
        kind=kind,
        line=line,
        dst_id=target.symbol_id,
        # An external target is known precisely, but there is no symbol to point at, so it
        # is graded the way the resolver grades a JDK call: unresolved, with the ref kept.
        resolution=Resolution.EXACT if target.symbol_id else Resolution.UNRESOLVED,
        arity=len(parsed.params) if parsed.params is not None else -1,
        is_new=parsed.is_constructor,
        source=loaded.header.tool,
    )


# ---- incremental planning ------------------------------------------------


@dataclass(frozen=True)
class FactsFileState:
    """What a previous build recorded about one facts file, read back from the graph."""

    relative_path: str
    mtime_ns: int
    size: int
    #: The source files the facts file declares.
    sources: frozenset[str]
    #: The templates its generated-source facts were mapped onto.
    template_paths: frozenset[str]
    #: Generated source path -> (template it maps onto, whether the template was newer than it).
    generated: dict[str, tuple[str, bool]] = field(default_factory=dict)
    #: Declared source path -> what mapping it contributed, so a build that maps only part of
    #: a facts file can carry the rest of its accounting forward untouched.
    contributions: dict[str, SourceContribution] = field(default_factory=dict)

    @property
    def covers(self) -> frozenset[str]:
        """Every source file whose overlay edges this facts file owns."""
        return self.sources | self.template_paths


@dataclass(frozen=True)
class FactsPlan:
    """Which facts files a build must map again, and what that alone invalidates."""

    #: Relative path -> the facts file on disk, as discovered for this build.
    present: dict[str, Path]
    #: Relative path -> what the previous build recorded.
    states: dict[str, FactsFileState]
    #: Facts files that appeared, changed or vanished.
    moved: frozenset[str]
    #: Source files whose overlay edges this build must rebuild whatever else it touches.
    invalidated: frozenset[str]

    @property
    def paths(self) -> list[Path]:
        """Every facts file to read, in a stable order."""
        return [self.present[relative] for relative in sorted(self.present)]

    def moved_coverage(self, overlay: FactsOverlay) -> set[str]:
        """Return the source files a moved facts file covers, as the freshly read file says.

        Only mapping can tell which templates a generated-source fact reaches, so this is asked
        after the first round and decides which further files the build must re-resolve.
        """
        return {
            path
            for loaded in overlay.files
            if loaded.relative_path in self.moved
            for path in (set(loaded.sources) | loaded.template_paths)
        }

    @property
    def vanished(self) -> frozenset[str]:
        """Facts files the graph remembers that are no longer on disk."""
        return frozenset(self.states) - frozenset(self.present)

    def mapping_request(self, targets: set[str]) -> MappingRequest:
        """Return, per facts file, the declared sources this build must map from it.

        A file that moved is mapped whole, because its content decides everything about it. A
        file that did not is asked only for the sources the build is re-resolving: every other
        source it covers keeps the edges - and the accounting - already stored for it.
        """
        request: MappingRequest = {}
        for relative in self.present:
            state = self.states.get(relative)
            if relative in self.moved or state is None:
                request[relative] = None
                continue
            needed = set(state.sources & targets)
            needed |= {source for source, (template, _) in state.generated.items() if template in targets}
            if needed:
                request[relative] = needed
        return request


def plan_facts(root: Path, present: dict[str, Path], states: dict[str, FactsFileState]) -> FactsPlan:
    """Decide, from `stat` alone, which facts files moved and what that invalidates.

    Two things move a facts file: its own bytes and its disappearance. A third thing changes
    what it CONTRIBUTES without moving it - the modification time of a template it mapped facts
    onto crossing the generated class it was compiled into - and that invalidates the template
    alone. It is why `template newer than generated source` no longer costs a re-resolution of
    every mapped template on every build: the verdict is recorded per generated source and
    re-decided with two stats.
    """
    moved: set[str] = set()
    invalidated: set[str] = set()
    for relative, path in present.items():
        state = states.get(relative)
        try:
            stat = path.stat()
        except OSError:
            continue
        if state is None:
            moved.add(relative)
            continue
        if (state.mtime_ns, state.size) != (stat.st_mtime_ns, stat.st_size):
            moved.add(relative)
            invalidated |= state.covers
            continue
        # A template whose freshness flipped invalidates that template and nothing else: the
        # facts file itself did not move, so every other source it covers is untouched.
        invalidated |= {
            template
            for generated, (template, was_stale) in state.generated.items()
            if template and template_is_newer(root, template, generated) != was_stale
        }
    for relative, state in states.items():
        if relative not in present:
            moved.add(relative)
            invalidated |= state.covers
    return FactsPlan(present=present, states=states, moved=frozenset(moved), invalidated=frozenset(invalidated))
