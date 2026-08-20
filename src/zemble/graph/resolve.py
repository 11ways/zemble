"""Workspace-wide resolution of extracted Java references.

Resolution is a ladder, and every rung is recorded on the edge so a consumer can
tell a fact from a guess: EXACT (the declaring type was pinned down by scope and
exactly one member matched), UNIQUE_NAME (only one symbol in the workspace carries
that name), AMBIGUOUS (several did) and UNRESOLVED (nothing did, so the target is
in the JDK or a third-party jar).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Container, Iterable
from dataclasses import dataclass, field

from zemble.graph.hwk import TAG_MODIFIER
from zemble.graph.java import FileImports
from zemble.graph.model import (
    CALLABLE_KINDS,
    TYPE_KINDS,
    Edge,
    EdgeKind,
    Resolution,
    Symbol,
    SymbolKind,
)
from zemble.hwk import (
    ELEMENT_ANNOTATION,
    ELEMENT_TAG_ARGUMENT,
    FUNCTION_ANNOTATION,
    FUNCTION_NAME_ARGUMENT,
    FUNCTION_NAMESPACE_ARGUMENT,
    template_id_path,
)

# Suffixes and prefixes that mark a test type as covering a subject type.
_TEST_SUFFIXES = ("Tests", "Test", "IT")
_TEST_PREFIXES = ("Test",)

_TYPE_EDGE_KINDS = frozenset({EdgeKind.EXTENDS, EdgeKind.IMPLEMENTS, EdgeKind.REFERENCES_TYPE, EdgeKind.ANNOTATED_WITH})
_TEMPLATE_SUFFIX = ".hwk"
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]")


@dataclass
class FileContext:
    """The package and imports a file's names are resolved against."""

    file_path: str
    package: str = ""
    imports: FileImports = field(default_factory=FileImports)


@dataclass
class _Match:
    """One resolution outcome."""

    symbol_id: str | None
    resolution: Resolution
    candidates: list[str] = field(default_factory=list)


_UNRESOLVED = _Match(None, Resolution.UNRESOLVED)


def _grade(matches: list[Symbol], best: Resolution) -> _Match:
    """Turn a candidate list into a match, never grading better than `best`."""
    if not matches:
        return _UNRESOLVED
    if len(matches) == 1:
        return _Match(matches[0].id, best)
    return _Match(None, Resolution.AMBIGUOUS, sorted(symbol.id for symbol in matches))


class Resolver:
    """Resolves extracted edges against the whole workspace symbol table."""

    def __init__(self, symbols: Iterable[Symbol], contexts: dict[str, FileContext]) -> None:
        """Index the workspace symbols so resolution is a dictionary lookup."""
        self.contexts = contexts
        self.by_id: dict[str, Symbol] = {}
        self.by_qualified: dict[str, list[Symbol]] = defaultdict(list)
        self.types_by_simple: dict[str, list[Symbol]] = defaultdict(list)
        self.callables_by_simple: dict[str, list[Symbol]] = defaultdict(list)
        self.members: dict[str, list[Symbol]] = defaultdict(list)
        self.types_in_file: dict[str, list[Symbol]] = defaultdict(list)
        self.templates_by_tag: dict[str, list[Symbol]] = defaultdict(list)
        self.templates_by_id: dict[str, list[Symbol]] = defaultdict(list)
        self.functions_by_key: dict[tuple[str, str], list[Symbol]] = defaultdict(list)
        self.functions_by_name: dict[str, list[Symbol]] = defaultdict(list)
        self.elements_by_tag: dict[str, list[Symbol]] = defaultdict(list)
        for symbol in symbols:
            self.by_id[symbol.id] = symbol
            self.by_qualified[symbol.qualified_name].append(symbol)
            if symbol.kind in TYPE_KINDS:
                self.types_by_simple[symbol.name].append(symbol)
                self.types_in_file[symbol.file_path].append(symbol)
            elif symbol.kind in CALLABLE_KINDS:
                self.callables_by_simple[symbol.name].append(symbol)
            if symbol.container_id is not None:
                self.members[symbol.container_id].append(symbol)
            self._index_hawkeye(symbol)
        self.supertypes: dict[str, list[str]] = defaultdict(list)
        self._chain_cache: dict[str, list[str]] = {}

    def _index_hawkeye(self, symbol: Symbol) -> None:
        """Index the declarations a Hawkeye template resolves against."""
        if symbol.kind is SymbolKind.TEMPLATE:
            if TAG_MODIFIER in symbol.modifiers:
                self.templates_by_tag[symbol.qualified_name].append(symbol)
            if symbol.container_id is None:
                self.templates_by_id[template_id_path(symbol.file_path)].append(symbol)
            return
        if symbol.kind is SymbolKind.METHOD and FUNCTION_ANNOTATION in symbol.annotations:
            arguments = symbol.annotation_args.get(FUNCTION_ANNOTATION, {})
            # `name` defaults to the Java method name, `namespace` to the global one.
            name = arguments.get(FUNCTION_NAME_ARGUMENT) or symbol.name
            self.functions_by_name[name].append(symbol)
            self.functions_by_key[(arguments.get(FUNCTION_NAMESPACE_ARGUMENT, ""), name)].append(symbol)
            return
        if symbol.kind in TYPE_KINDS:
            tag = symbol.annotation_args.get(ELEMENT_ANNOTATION, {}).get(ELEMENT_TAG_ARGUMENT)
            if tag:
                self.elements_by_tag[tag].append(symbol)

    # ---- lookup helpers -------------------------------------------------

    def enclosing_type(self, symbol_id: str) -> Symbol | None:
        """Walk containers upward until a type declaration is reached."""
        current = self.by_id.get(symbol_id)
        while current is not None:
            if current.kind in TYPE_KINDS:
                return current
            current = self.by_id.get(current.container_id) if current.container_id else None
        return None

    def top_level_type(self, symbol_id: str) -> Symbol | None:
        """Return the outermost type declaration enclosing a symbol."""
        found: Symbol | None = None
        current = self.by_id.get(symbol_id)
        while current is not None:
            if current.kind in TYPE_KINDS:
                found = current
            current = self.by_id.get(current.container_id) if current.container_id else None
        return found

    def chain(self, type_id: str) -> list[str]:
        """Return the type and its resolved supertypes, breadth first, without repeats."""
        cached = self._chain_cache.get(type_id)
        if cached is not None:
            return cached
        order: list[str] = []
        seen = {type_id}
        queue = [type_id]
        while queue:
            current = queue.pop(0)
            order.append(current)
            for parent in self.supertypes.get(current, ()):
                if parent not in seen:
                    seen.add(parent)
                    queue.append(parent)
        self._chain_cache[type_id] = order
        return order

    def _lookup_qualified(self, qualified: str, kinds: Container[SymbolKind]) -> list[Symbol]:
        """Return workspace symbols with an exact qualified name and an accepted kind."""
        return [symbol for symbol in self.by_qualified.get(qualified, ()) if symbol.kind in kinds]

    # ---- type resolution ------------------------------------------------

    def resolve_type_name(self, name: str, context: FileContext) -> _Match:
        """Resolve a written type name against a file's scope, then the workspace."""
        if "." in name:
            return self._resolve_dotted_type(name, context)
        return self._resolve_simple_type(name, context)

    def _resolve_dotted_type(self, name: str, context: FileContext) -> _Match:
        """Resolve `Outer.Inner` or a fully qualified name."""
        exact = self._lookup_qualified(name, TYPE_KINDS)
        if exact:
            return _grade(exact, Resolution.EXACT)
        head, _, tail = name.partition(".")
        head_match = self._resolve_simple_type(head, context)
        if head_match.symbol_id is not None:
            owner = self.by_id[head_match.symbol_id]
            nested = self._lookup_qualified(f"{owner.qualified_name}.{tail}", TYPE_KINDS)
            if nested:
                return _grade(nested, head_match.resolution)
            constant = self._enum_constant_type(f"{owner.qualified_name}.{tail}")
            if constant is not None:
                # `Palette.RED.tag()`: the receiver is written through a constant, whose
                # type is the enum that declares it.
                return _grade([constant], head_match.resolution)
        return self._resolve_simple_type(name.rsplit(".", 1)[-1], context)

    def _enum_constant_type(self, qualified: str) -> Symbol | None:
        """Return the enum declaring a constant written as `Enum.CONSTANT`, if that is what this is."""
        for symbol in self.by_qualified.get(qualified, ()):
            if symbol.kind is SymbolKind.ENUM_CONSTANT and symbol.container_id is not None:
                owner = self.by_id.get(symbol.container_id)
                if owner is not None and owner.kind in TYPE_KINDS:
                    return owner
        return None

    def _resolve_simple_type(self, name: str, context: FileContext) -> _Match:
        """Resolve a simple type name through the file scope ladder."""
        same_file = [symbol for symbol in self.types_in_file.get(context.file_path, ()) if symbol.name == name]
        if same_file:
            return _grade(sorted(same_file, key=lambda s: len(s.qualified_name))[:1], Resolution.EXACT)
        imported = context.imports.explicit.get(name)
        if imported is not None:
            # The import names exactly one type. If it is not in the workspace it is a
            # JDK or third-party type, and falling back to a same-named workspace type
            # would be wrong, not merely imprecise.
            return _grade(self._lookup_qualified(imported, TYPE_KINDS), Resolution.EXACT)
        if context.package:
            same_package = self._lookup_qualified(f"{context.package}.{name}", TYPE_KINDS)
            if same_package:
                return _grade(same_package, Resolution.EXACT)
        wildcard_hits = [
            symbol
            for package in context.imports.wildcards
            for symbol in self._lookup_qualified(f"{package}.{name}", TYPE_KINDS)
        ]
        if wildcard_hits:
            return _grade(wildcard_hits, Resolution.EXACT)
        return _grade(self.types_by_simple.get(name, []), Resolution.UNIQUE_NAME)

    # ---- call resolution ------------------------------------------------

    def _members_named(self, type_id: str, name: str, arity: int) -> list[Symbol]:
        """Find callables named `name` with a compatible arity anywhere in a type's chain."""
        found: list[Symbol] = []
        for owner_id in self.chain(type_id):
            for member in self.members.get(owner_id, ()):
                if member.kind in CALLABLE_KINDS and member.name == name and (arity < 0 or member.arity == arity):
                    found.append(member)
            if found:
                break  # the nearest declaring type in the chain wins
        return found

    def _constructor_of(self, type_id: str, arity: int, best: Resolution) -> _Match:
        """Resolve a constructor call, falling back to the type when no signature matches."""
        constructors = [m for m in self.members.get(type_id, ()) if m.kind is SymbolKind.CONSTRUCTOR]
        matching = [m for m in constructors if arity < 0 or m.arity == arity]
        if matching:
            return _grade(matching, best)
        if not constructors:
            # No declared constructor: the implicit one, whose home is the type itself.
            return _Match(type_id, best)
        return _Match(type_id, Resolution.UNIQUE_NAME)

    def _resolve_constructor_call(self, edge: Edge, context: FileContext) -> _Match:
        """Resolve `new Foo(...)`, `this(...)` and `super(...)`."""
        enclosing = self.enclosing_type(edge.src_id)
        if edge.dst_name == "this":
            return self._constructor_of(enclosing.id, edge.arity, Resolution.EXACT) if enclosing else _UNRESOLVED
        if edge.dst_name == "super":
            parents = self.supertypes.get(enclosing.id, []) if enclosing else []
            return self._constructor_of(parents[0], edge.arity, Resolution.EXACT) if parents else _UNRESOLVED
        type_match = self.resolve_type_name(edge.dst_name.removesuffix("[]"), context)
        if type_match.symbol_id is None:
            return type_match
        return self._constructor_of(type_match.symbol_id, edge.arity, type_match.resolution)

    def _resolve_method_call(self, edge: Edge, context: FileContext) -> _Match:
        """Resolve a method invocation: receiver chain, then static imports, then the workspace."""
        owner_match = self._call_owner(edge, context)
        if owner_match is not None:
            through_owner = self._members_of_owner(owner_match, edge)
            if through_owner is not None:
                return through_owner
        if edge.receiver is None:
            static_owner = self._static_import_owner(edge.dst_name, context)
            if static_owner is not None and static_owner.symbol_id is not None:
                found = self._members_named(static_owner.symbol_id, edge.dst_name, edge.arity)
                if found:
                    return _grade(found, static_owner.resolution)
        return self._resolve_call_by_name(edge)

    def _members_of_owner(self, owner_match: _Match, edge: Edge) -> _Match | None:
        """Search the receiver type, or every candidate when the receiver itself was ambiguous."""
        owner_ids = [owner_match.symbol_id] if owner_match.symbol_id else owner_match.candidates
        found: list[Symbol] = []
        for owner_id in owner_ids:
            found.extend(self._members_named(owner_id, edge.dst_name, edge.arity))
        if not found:
            return None
        # An ambiguous receiver can never yield an exact member, only a narrowed guess.
        best = owner_match.resolution if owner_match.symbol_id else Resolution.UNIQUE_NAME
        return _grade(found, best)

    def _call_owner(self, edge: Edge, context: FileContext) -> _Match | None:
        """Determine the type whose chain a call should be searched in, if it is knowable."""
        if edge.receiver in (None, "this"):
            enclosing = self.enclosing_type(edge.src_id)
            return _Match(enclosing.id, Resolution.EXACT) if enclosing else None
        if edge.receiver == "super":
            enclosing = self.enclosing_type(edge.src_id)
            parents = self.supertypes.get(enclosing.id, []) if enclosing else []
            return _Match(parents[0], Resolution.EXACT) if parents else None
        if edge.receiver_type:
            return self.resolve_type_name(edge.receiver_type.removesuffix("[]"), context)
        return None

    def _static_import_owner(self, name: str, context: FileContext) -> _Match | None:
        """Resolve the owning type of a statically imported member."""
        owner = context.imports.static_members.get(name)
        owners = [owner] if owner else list(context.imports.static_wildcards)
        for candidate in owners:
            match = self.resolve_type_name(candidate, context)
            if match.symbol_id is not None:
                return match
        return None

    def _resolve_call_by_name(self, edge: Edge) -> _Match:
        """Last rung: match a call against every same-named callable in the workspace."""
        candidates = [
            symbol
            for symbol in self.callables_by_simple.get(edge.dst_name, ())
            if edge.arity < 0 or symbol.arity == edge.arity
        ]
        return _grade(candidates, Resolution.UNIQUE_NAME)

    # ---- edge resolution ------------------------------------------------

    def resolve_edge(self, edge: Edge) -> None:
        """Resolve one edge in place."""
        context = self.contexts.get(_file_of(edge.src_id)) or FileContext(_file_of(edge.src_id))
        match = self._match_for(edge, context)
        edge.dst_id = match.symbol_id
        edge.resolution = match.resolution
        edge.candidates = match.candidates
        if edge.kind is EdgeKind.EXTENDS and match.symbol_id is not None:
            target = self.by_id.get(match.symbol_id)
            if target is not None and target.kind is SymbolKind.INTERFACE:
                edge.kind = EdgeKind.IMPLEMENTS

    def _match_for(self, edge: Edge, context: FileContext) -> _Match:
        """Pick the resolution strategy for an edge kind."""
        if _file_of(edge.src_id).endswith(_TEMPLATE_SUFFIX):
            # A template writes template ids, element tags and function namespaces - none of
            # which are Java names - so the Java ladder would answer every one of them wrong.
            return self._match_for_template(edge)
        if edge.kind in _TYPE_EDGE_KINDS:
            return self.resolve_type_name(edge.dst_name, context)
        if edge.kind is EdgeKind.IMPORTS:
            # A single-type import names a type; a static member import names a member.
            # A wildcard import names a package, which is no single symbol, so it stays unresolved.
            found = self._lookup_qualified(edge.dst_name, TYPE_KINDS) or self._lookup_qualified(
                edge.dst_name, CALLABLE_KINDS | {SymbolKind.FIELD, SymbolKind.ENUM_CONSTANT}
            )
            return _grade(found, Resolution.EXACT)
        if edge.kind is EdgeKind.CALLS:
            if edge.is_new:
                return self._resolve_constructor_call(edge, context)
            return self._resolve_method_call(edge, context)
        return _UNRESOLVED

    # ---- template resolution --------------------------------------------

    def _match_for_template(self, edge: Edge) -> _Match:
        """Resolve one edge written by a Hawkeye template."""
        if edge.kind in (EdgeKind.EXTENDS, EdgeKind.IMPORTS):
            return self._resolve_template_reference(edge.dst_name)
        if edge.kind is EdgeKind.REFERENCES_TYPE:
            return self._resolve_element_tag(edge.dst_name)
        if edge.kind is EdgeKind.CALLS:
            return self._resolve_template_call(edge)
        return _UNRESOLVED

    def _resolve_template_reference(self, written: str) -> _Match:
        """Resolve `namespace:path/below/templates` to the template file it names.

        The namespace is a build setting this extractor never reads, so it is used only to
        narrow: a path that is unique on its own is a by-name match, and a path the namespace
        also agrees with is exact.
        """
        namespace, separator, path = written.partition(":")
        if not separator:
            namespace, path = "", written
        candidates = self.templates_by_id.get(path, [])
        if not candidates:
            return _UNRESOLVED
        narrowed = [symbol for symbol in candidates if _namespace_matches(namespace, symbol.file_path)]
        if len(narrowed) == 1:
            return _Match(narrowed[0].id, Resolution.EXACT)
        return _grade(narrowed or candidates, Resolution.UNIQUE_NAME)

    def _resolve_element_tag(self, tag: str) -> _Match:
        """Resolve a custom element tag to the class or the template that declares it.

        A tag is a globally unique registration key - the Hawkeye compiler refuses a duplicate
        - so a single declaration of it IS the one meant, and the match is exact. A hand-written
        `@HawkeyeCustomElement` class wins over a template, because it is the implementation.
        """
        for declarations in (self.elements_by_tag.get(tag), self.templates_by_tag.get(tag)):
            if declarations:
                return _grade(declarations, Resolution.EXACT)
        return _UNRESOLVED

    def _resolve_template_call(self, edge: Edge) -> _Match:
        """Resolve a template function call against the `@HawkeyeFunction` methods.

        A call is only ever matched against a registered template function: nothing else in the
        workspace is callable from a template, so falling back to a same-named plain Java method
        would invent a relationship that cannot exist.
        """
        namespace = edge.receiver or ""
        exact = self.functions_by_key.get((namespace, edge.dst_name))
        if exact:
            return _grade(exact, Resolution.EXACT)
        return _grade(self.functions_by_name.get(edge.dst_name, []), Resolution.UNIQUE_NAME)

    def resolve_all(self, edges: Iterable[Edge]) -> None:
        """Resolve supertype edges first, then everything else, so call chains are usable."""
        edges = list(edges)
        hierarchy = [edge for edge in edges if edge.kind in (EdgeKind.EXTENDS, EdgeKind.IMPLEMENTS)]
        for edge in hierarchy:
            self.resolve_edge(edge)
        self.index_hierarchy(hierarchy)
        for edge in edges:
            if edge.kind not in (EdgeKind.EXTENDS, EdgeKind.IMPLEMENTS):
                self.resolve_edge(edge)

    def index_hierarchy(self, edges: Iterable[Edge]) -> None:
        """Record resolved EXTENDS/IMPLEMENTS edges as the supertype map."""
        for edge in edges:
            if edge.kind in (EdgeKind.EXTENDS, EdgeKind.IMPLEMENTS) and edge.dst_id is not None:
                if edge.dst_id not in self.supertypes[edge.src_id]:
                    self.supertypes[edge.src_id].append(edge.dst_id)
        self._chain_cache.clear()

    # ---- derived edges ---------------------------------------------------

    def derive_overrides(self, symbols: Iterable[Symbol]) -> list[Edge]:
        """Emit OVERRIDES edges for methods that redeclare a supertype method."""
        derived: list[Edge] = []
        for symbol in symbols:
            if symbol.kind is not SymbolKind.METHOD or symbol.container_id is None:
                continue
            owner = self.by_id.get(symbol.container_id)
            start = 1
            if owner is not None and owner.kind is SymbolKind.ENUM_CONSTANT:
                # A method in an enum constant's body overrides the enum's own method, so the
                # enum itself belongs in the walk rather than being skipped as the method's self.
                owner, start = self.enclosing_type(owner.id), 0
            if owner is None or owner.kind not in TYPE_KINDS:
                continue
            for parent_id in self.chain(owner.id)[start:]:
                matches = [
                    member
                    for member in self.members.get(parent_id, ())
                    if member.kind is SymbolKind.METHOD and member.name == symbol.name and member.arity == symbol.arity
                ]
                if matches:
                    derived.append(
                        Edge(
                            src_id=symbol.id,
                            dst_name=symbol.name,
                            kind=EdgeKind.OVERRIDES,
                            line=symbol.start_line,
                            dst_id=matches[0].id,
                            # Parameter types are never compared, only name and arity.
                            resolution=Resolution.UNIQUE_NAME,
                            candidates=sorted(m.id for m in matches[1:]),
                            arity=symbol.arity,
                        )
                    )
                    break
        return derived

    def derive_tests(self, symbols: Iterable[Symbol]) -> list[Edge]:
        """Emit TESTS edges from test types to the subject their name implies."""
        derived: list[Edge] = []
        for symbol in symbols:
            if not symbol.is_test or symbol.kind not in TYPE_KINDS:
                continue
            subject = _subject_name(symbol.name)
            if subject is None:
                continue
            context = self.contexts.get(symbol.file_path) or FileContext(symbol.file_path)
            match = self.resolve_type_name(subject, context)
            if match.symbol_id is None and match.resolution is not Resolution.AMBIGUOUS:
                continue
            derived.append(
                Edge(
                    src_id=symbol.id,
                    dst_name=subject,
                    kind=EdgeKind.TESTS,
                    line=symbol.start_line,
                    dst_id=match.symbol_id,
                    resolution=match.resolution,
                    candidates=match.candidates,
                )
            )
        return derived

    def derive_exercises(self, edges: Iterable[Edge]) -> list[Edge]:
        """Emit EXERCISES edges from a test type to every main-source symbol it touches."""
        seen: set[tuple[str, str]] = set()
        derived: list[Edge] = []
        for edge in edges:
            if edge.dst_id is None or edge.kind in (EdgeKind.TESTS, EdgeKind.EXERCISES):
                continue
            source = self.by_id.get(edge.src_id)
            target = self.by_id.get(edge.dst_id)
            if source is None or target is None or not source.is_test or target.is_test:
                continue
            holder = self.top_level_type(edge.src_id)
            if holder is None or (holder.id, target.id) in seen:
                continue
            seen.add((holder.id, target.id))
            derived.append(
                Edge(
                    src_id=holder.id,
                    dst_name=target.name,
                    kind=EdgeKind.EXERCISES,
                    line=edge.line,
                    dst_id=target.id,
                    resolution=edge.resolution,
                )
            )
        return derived


def _namespace_matches(written: str, file_path: str) -> bool:
    """Return True when a template id's namespace agrees with the repository a file lives in.

    A repository's Hawkeye namespace is its directory name with the separators dropped
    (`zenit-cms` -> `zenitcms`), and a source set may append to it (`plumage-browsertest`), so
    the repository's own form must be a prefix of what was written.
    """
    if not written:
        return False
    repository = _NON_ALPHANUMERIC.sub("", file_path.split("/", 1)[0].lower())
    return bool(repository) and _NON_ALPHANUMERIC.sub("", written.lower()).startswith(repository)


def _file_of(symbol_id: str) -> str:
    """Return the file path encoded in a symbol id."""
    return symbol_id.split("#", 1)[0]


def _subject_name(name: str) -> str | None:
    """Return the subject type name a test class name implies, if any."""
    for suffix in _TEST_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    for prefix in _TEST_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix) and name[len(prefix)].isupper():
            return name[len(prefix) :]
    return None
