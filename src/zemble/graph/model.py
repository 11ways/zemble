"""Language-neutral symbol graph model: symbols, edges and their resolution quality."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath

# Directory segments that mark a file as belonging to a test source set. Matched
# case-insensitively against every directory segment of the file's relative path,
# so both `src/test/java/...` (Maven) and `src/browserTest/java/...` (Gradle
# source sets, as used across the javaweb workspace) are recognised.
TEST_PATH_SEGMENTS: frozenset[str] = frozenset({"test", "tests", "browsertest", "integrationtest", "testfixtures"})


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


TYPE_KINDS: frozenset[SymbolKind] = frozenset(
    {SymbolKind.CLASS, SymbolKind.INTERFACE, SymbolKind.ENUM, SymbolKind.RECORD, SymbolKind.ANNOTATION}
)
CALLABLE_KINDS: frozenset[SymbolKind] = frozenset({SymbolKind.METHOD, SymbolKind.CONSTRUCTOR})
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


def is_test_path(relative_path: str) -> bool:
    """Return True if a workspace-relative path lives under a test source set."""
    parts = PurePosixPath(relative_path).parts
    return any(part.lower() in TEST_PATH_SEGMENTS for part in parts[:-1])


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
    candidates: list[str] = field(default_factory=list)
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
