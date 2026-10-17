"""Language-neutral symbol graph model: symbols, edges and their resolution quality."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath

#: Directory segments that are a test source set as a whole, matched case-insensitively.
TEST_PATH_SEGMENTS: frozenset[str] = frozenset({"test", "tests", "integrationtest"})
#: Camel-case words that make a longer source-set name a test one (`browserTestCommon`, `testFixtures`,
#: `artifactContractTest`); a lowercase run like `testdata` is one word and is not split.
_TEST_WORDS: frozenset[str] = frozenset({"test", "tests"})
_CAMEL_WORD = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")


def is_test_segment(name: str) -> bool:
    """Whether one directory or source-set name marks test code: a listed segment, or a camel word `test`."""
    return name.lower() in TEST_PATH_SEGMENTS or any(word.lower() in _TEST_WORDS for word in _CAMEL_WORD.findall(name))


#: How an anonymous class's name begins (`$anon@<line>`); nothing can name one, so nothing references it.
ANONYMOUS_MARKER = "$anon@"


class SymbolKind(str, Enum):
    """Kind of a declared symbol."""

    PACKAGE = "package"
    CLASS = "class"
    INTERFACE = "interface"
    ENUM = "enum"
    RECORD = "record"
    ANNOTATION = "annotation"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    FIELD = "field"
    ENUM_CONSTANT = "enum_constant"
    TEMPLATE = "template"
    BLOCK = "block"
    # Kinds the grammar-driven extractors add on top of the Java vocabulary.
    STRUCT = "struct"
    TYPE = "type"
    MODULE = "module"
    FUNCTION = "function"


#: Kinds that declare a type of their own: what a supertype clause or a type position names.
DECLARED_TYPE_KINDS: frozenset[SymbolKind] = frozenset(
    {
        SymbolKind.CLASS,
        SymbolKind.INTERFACE,
        SymbolKind.ENUM,
        SymbolKind.RECORD,
        SymbolKind.ANNOTATION,
        SymbolKind.STRUCT,
        SymbolKind.TYPE,
    }
)
#: Kinds that own members and anchor a resolution scope. A MODULE is the file, or a named
#: namespace inside it, in every language that has no Java-style package symbol: its members
#: are the top-level functions and constants, which is why it is a container like a type.
TYPE_KINDS: frozenset[SymbolKind] = DECLARED_TYPE_KINDS | {SymbolKind.MODULE}
CALLABLE_KINDS: frozenset[SymbolKind] = frozenset({SymbolKind.METHOD, SymbolKind.CONSTRUCTOR, SymbolKind.FUNCTION})
#: Kinds that name themselves: a template is displayed and looked up by its own name, never
#: as `Owner.member`, exactly like a type declaration.
NAMED_KINDS: frozenset[SymbolKind] = TYPE_KINDS | {SymbolKind.TEMPLATE}


class EdgeKind(str, Enum):
    """Kind of a relationship between two symbols."""

    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    OVERRIDES = "overrides"
    CALLS = "calls"
    REFERENCES_TYPE = "references_type"
    ANNOTATED_WITH = "annotated_with"
    IMPORTS = "imports"
    TESTS = "tests"
    EXERCISES = "exercises"


class Resolution(str, Enum):
    """How confidently an edge's destination was determined."""

    EXACT = "exact"
    UNIQUE_NAME = "unique_name"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"

    @property
    def strength(self) -> int:
        """Rank from most confident (highest) to least; a member without a rank raises."""
        return _RESOLUTION_STRENGTH[self]

    def at_least(self, floor: Resolution) -> bool:
        """Whether this resolution is as confident as `floor` or more."""
        return self.strength >= floor.strength


_RESOLUTION_STRENGTH: dict[Resolution, int] = {
    Resolution.EXACT: 3,
    Resolution.UNIQUE_NAME: 2,
    Resolution.AMBIGUOUS: 1,
    Resolution.UNRESOLVED: 0,
}


def is_test_path(relative_path: str) -> bool:
    """Return True if a workspace-relative path lives under a test source set."""
    parts = PurePosixPath(relative_path).parts
    return any(is_test_segment(part) for part in parts[:-1])


def make_symbol_id(relative_path: str, qualified_name: str, disambiguator: str | None = None) -> str:
    """Build a stable symbol id from its file, qualified name and optional signature disambiguator."""
    suffix = f"({disambiguator})" if disambiguator is not None else ""
    return f"{relative_path}#{qualified_name}{suffix}"


@dataclass
class Symbol:
    """A declared symbol in a source file."""

    id: str
    kind: SymbolKind
    name: str
    qualified_name: str
    file_path: str
    start_line: int
    end_line: int
    container_id: str | None = None
    modifiers: list[str] = field(default_factory=list)
    annotations: list[str] = field(default_factory=list)
    signature: str = ""
    is_test: bool = False
    # Erased parameter types, callables only. Kept out of `signature` so resolution
    # can match on arity/erasure without re-parsing the human-readable signature.
    param_types: list[str] = field(default_factory=list)
    # String-literal arguments of each annotation, keyed by the annotation's simple name and
    # then by element name (`value` for the single unnamed argument). Only literals are kept:
    # an argument written through a constant is not knowable from one file, and recording the
    # constant's name would read as a value it never has. This is what lets a registration
    # annotation - `@HawkeyeFunction(namespace = "String", name = "presence")` - be resolved
    # against the name a caller in another language writes.
    annotation_args: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def arity(self) -> int:
        """Number of declared parameters (0 for non-callables)."""
        return len(self.param_types)


@dataclass
class Edge:
    """A relationship from one symbol to another symbol or unresolved name."""

    src_id: str
    dst_name: str
    kind: EdgeKind
    line: int
    dst_id: str | None = None
    resolution: Resolution = Resolution.UNRESOLVED
    # The symbols an ambiguous edge could mean, while it is being resolved. The store keeps
    # only their number in `candidate_count`: an ambiguous edge has no destination, so no query
    # ever reached the list, and on javaweb it was 1.6 GB of a 2.6 GB graph.
    candidates: list[str] = field(default_factory=list)
    candidate_count: int = 0
    # Arity of a call site, used to pick between overloads. -1 when not applicable.
    arity: int = -1
    # Receiver text as written (`this`, `super`, a simple identifier or a dotted name).
    receiver: str | None = None
    # The receiver's declared type name when the extractor could see the declaration
    # in the same file (parameter, local, field) or the receiver is written as a type.
    receiver_type: str | None = None
    # True for `new Foo(...)`, `this(...)` and `super(...)`: the call targets a constructor.
    is_new: bool = False
    # Who produced this edge: "tree-sitter" for zemble's own extractor, else the `tool`
    # name of the facts file that replaced it (see `zemble.graph.facts`).
    source: str = "tree-sitter"
    # The ref the emitter actually wrote, when the edge did not start where it is now: a fact
    # about generated Hawkeye code mapped back onto its template keeps the generated member
    # here, so a reader can see the detour the source map made.
    origin_ref: str | None = None

    def ambiguity(self) -> int:
        """How many symbols this edge could mean: its list while resolving, its stored count once loaded."""
        return len(self.candidates) or self.candidate_count


@dataclass
class Hit:
    """One graph answer: the symbol at the other end of an edge plus why it is there."""

    symbol: Symbol
    edge_kind: EdgeKind
    line: int
    resolution: Resolution
    reason: str
    # Hops from the queried symbol, for the transitive hierarchy and neighbour walks.
    depth: int = 1
    # The edge's producer: "tree-sitter" or the name of the tool whose facts replaced it.
    source: str = "tree-sitter"


@dataclass(frozen=True)
class EdgeTally:
    """How many distinct symbols reach one symbol over one edge kind at one resolution."""

    symbol_id: str
    kind: EdgeKind
    resolution: Resolution
    #: Whether the reaching symbols live in a test source set.
    from_test: bool
    sources: int


@dataclass(frozen=True)
class FileDependency:
    """Resolved edges of one kind and resolution from one file into declarations of another."""

    source_path: str
    target_path: str
    kind: EdgeKind
    resolution: Resolution
    edges: int


@dataclass(frozen=True)
class RowCounts:
    """How many rows a graph holds, for sizing a whole-graph read before it starts."""

    symbols: int
    edges: int
    metrics: int
    files: int
