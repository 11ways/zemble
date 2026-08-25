"""Grammar-driven symbol and reference extraction for every language with a spec.

One extractor serves every grammar in :mod:`zemble.languages.catalog`: the spec says which
nodes declare what and where their parts live, and this module turns that into the same
symbols and edges the hand-written Java extractor emits, so the resolver, the outline and
every graph tool downstream never learn that a second reader exists. Like the Java lane it
is purely local: cross-file questions are left to :mod:`zemble.graph.resolve`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cache
from logging import getLogger
from pathlib import PurePosixPath

from semble_grammars import LanguageNotFoundError, UnsupportedPlatformError, get_parser
from tree_sitter import Node, Parser

from zemble.graph.java import FileExtraction
from zemble.graph.model import DECLARED_TYPE_KINDS, Edge, EdgeKind, Symbol, SymbolKind, is_test_path, make_symbol_id
from zemble.languages.catalog import is_test_file, spec_for
from zemble.languages.paths import resolve, text
from zemble.languages.read import (
    NAME_KINDS,
    QUALIFIED_KINDS,
    Call,
    Declared,
    annotations_of,
    body_of,
    call_at,
    clean_name,
    match,
    modifiers_of,
    name_of,
    names_of,
    parameter_nodes,
    parameter_type,
    params_of,
    signature_of,
    split_qualified,
    supertypes_of,
    type_name,
)
from zemble.languages.spec import LanguageSpec, Role

logger = getLogger(__name__)

#: File stems that name their directory rather than themselves.
_DIRECTORY_STEMS = frozenset({"__init__"})
#: Separators a package header may be written with, all folded to the dot the graph uses.
_PACKAGE_SEPARATORS = ("::", "\\", "/")
_MAX_FIELD_SIGNATURE = 120


@dataclass(frozen=True)
class _Owner:
    """The symbol new declarations attach to, and what kind of scope it opens."""

    id: str
    qualified: str
    name: str
    is_type: bool = False
    is_callable: bool = False


@cache
def language_parser(language: str) -> Parser | None:
    """Return the bundled tree-sitter parser of a language, or None when it is unavailable."""
    try:
        return get_parser(language)
    except (LanguageNotFoundError, UnsupportedPlatformError):
        logger.warning("No bundled tree-sitter %s grammar on this platform", language)
    except Exception:
        logger.error("Uncaught exception while loading the %s grammar", language, exc_info=True)
    return None


def extract_generic_file(source: bytes, relative_path: str, language: str) -> FileExtraction:
    """Extract symbols and unresolved references from one file of a spec-driven language.

    :param source: Raw file bytes.
    :param relative_path: Path relative to the indexed root, posix style.
    :param language: The grammar name, as :func:`zemble.index.files.detect_language` reports it.
    :return: The file's symbols, edges and imports.
    :raises RuntimeError: If the language has no spec or no grammar on this platform.
    """
    spec = spec_for(language)
    if spec is None:
        raise RuntimeError(f"No language spec for {language}")
    parser = language_parser(language)
    if parser is None:
        raise RuntimeError(f"No tree-sitter {language} grammar available")
    tree = parser.parse(source)
    extractor = _GenericExtractor(spec, source, relative_path)
    extractor.run(tree.root_node)
    return extractor.result


class _GenericExtractor:
    """Walks one parsed file and records its symbols and references as the spec dictates."""

    def __init__(self, spec: LanguageSpec, source: bytes, relative_path: str) -> None:
        """Prepare an extractor for one file."""
        self.spec = spec
        self.source = source
        self.path = relative_path
        self.result = FileExtraction(file_path=relative_path, package="")
        self.is_test = is_test_path(relative_path) or is_test_file(relative_path)
        self._declared_types: dict[str, Symbol] = {}
        self._ids: set[str] = set()
        #: Bodies reached through a sibling step, which the enclosing walk must not visit again.
        self._consumed: set[int] = set()
        self.module: Symbol | None = None

    # ---- helpers -------------------------------------------------------

    def _text(self, node: Node | None) -> str:
        """Collapsed source text of a node."""
        return text(node, self.source)

    @staticmethod
    def _line(node: Node) -> int:
        """1-indexed start line of a node."""
        return node.start_point[0] + 1

    def _add_symbol(self, symbol: Symbol) -> Symbol:
        """Append a symbol, giving a second declaration of the same id a line disambiguator."""
        if symbol.id in self._ids:
            symbol.id = f"{symbol.id}@{symbol.start_line}"
        self._ids.add(symbol.id)
        self.result.symbols.append(symbol)
        return symbol

    def _add_edge(self, src_id: str, dst_name: str, kind: EdgeKind, line: int, **kwargs: object) -> None:
        """Append an unresolved edge."""
        if not dst_name:
            return
        self.result.edges.append(Edge(src_id=src_id, dst_name=dst_name, kind=kind, line=line, **kwargs))  # type: ignore[arg-type]

    def _qualify(self, owner: _Owner, name: str) -> str:
        """Join a name onto its owner's qualified name."""
        return f"{owner.qualified}.{name}" if owner.qualified else name

    @staticmethod
    def _dotted(written: str) -> str:
        """Fold a package or import path written with any separator onto dots."""
        value = clean_name(written)
        for separator in _PACKAGE_SEPARATORS:
            value = value.replace(separator, ".")
        return value.strip(".")

    # ---- top level -----------------------------------------------------

    def run(self, root: Node) -> None:
        """Extract the whole file."""
        package = self._read_package(root)
        self.result.package = package
        self.module = self._module_symbol(root, package)
        # With a package header the top-level names qualify under the package, as a reader
        # writes them (`app.core.Point`); without one they qualify under the file's path.
        owner = _Owner(id=self.module.id, qualified=package or self.module.qualified_name, name=self.module.name)
        self._walk(root, owner)
        self.result.edges = _dedupe_edges(self.result.edges)

    def _read_package(self, root: Node) -> str:
        """Find the file's package header, if the language writes one."""
        for child in _top_level(root):
            declared = match(self.spec, child, self.source)
            if declared is not None and declared.rule.role is Role.PACKAGE and declared.rule.name is not None:
                return self._dotted(self._text(resolve(declared.node, declared.rule.name)))
        return ""

    def _module_symbol(self, root: Node, package: str) -> Symbol:
        """Build the symbol standing for the file itself.

        Its qualified name is the package plus the file stem where the language declares a
        package, else the file's own dotted path, so `graph_definition pkg.module.func` finds a
        function the way a reader would write it.
        """
        pure = PurePosixPath(self.path)
        stem = pure.name.rsplit(".", 1)[0] if "." in pure.name else pure.name
        if stem in _DIRECTORY_STEMS:
            stem = pure.parent.name
            parts = list(pure.parent.parts)
        else:
            parts = [*pure.parent.parts, stem]
        parts = [part.rsplit(".", 1)[0] if "." in part else part for part in parts if part not in (".", "")]
        qualified = f"{package}.{stem}" if package else ".".join(parts) or stem
        end_line = max(1, root.end_point[0] + (0 if root.end_point[1] == 0 else 1))
        return self._add_symbol(
            Symbol(
                id=make_symbol_id(self.path, qualified),
                kind=SymbolKind.MODULE,
                name=stem,
                qualified_name=qualified,
                file_path=self.path,
                start_line=1,
                end_line=end_line,
                signature=f"module {qualified}",
                is_test=self.is_test,
            )
        )

    # ---- the walk -------------------------------------------------------

    def _walk(self, root: Node, owner: _Owner) -> None:
        """Visit every node under the root, iteratively, attributing references to their owner."""
        pending: deque[tuple[Node, _Owner]] = deque((child, owner) for child in root.named_children)
        while pending:
            node, current = pending.popleft()
            if node.id in self._consumed:
                continue
            if node.type == "ERROR":
                pending.extendleft(reversed([(child, current) for child in node.named_children]))
                continue
            if "comment" in node.type:
                continue
            declared = match(self.spec, node, self.source, in_type=current.is_type, in_callable=current.is_callable)
            if declared is not None:
                pending.extendleft(reversed(self._apply(declared, current)))
                continue
            if node.type in self.spec.type_ref_kinds:
                self._type_refs(node, current.id)
                continue
            call = call_at(self.spec, node, self.source)
            if call is not None:
                self._emit_call(call, current)
                # A callee that is itself a call node (Perl's bareword inside a bracketed call)
                # would emit the same call a second time.
                self._consumed.update(call.consumed_ids)
            pending.extendleft(reversed([(child, current) for child in node.named_children]))

    def _apply(self, declared: Declared, owner: _Owner) -> list[tuple[Node, _Owner]]:
        """Record one declaration and return the (node, owner) pairs to keep walking."""
        handler = _ROLE_HANDLERS.get(declared.rule.role)
        if handler is None:
            raise ValueError(f"unhandled rule role {declared.rule.role!r}")
        return handler(self, declared, owner)

    def _nothing(self, declared: Declared, owner: _Owner) -> list[tuple[Node, _Owner]]:
        """A skipped node or an already-read package header contributes nothing."""
        return []

    def _import_role(self, declared: Declared, owner: _Owner) -> list[tuple[Node, _Owner]]:
        """Record an import."""
        self._import(declared)
        return []

    def _supertype(self, declared: Declared, owner: _Owner) -> list[tuple[Node, _Owner]]:
        """A statement inside a type body naming a supertype (`include`, `use Trait;`)."""
        if owner.is_type:
            for node in names_of(self.spec, declared, self.source):
                name = type_name(self._text(node))
                if name:
                    self._add_edge(owner.id, name, EdgeKind.EXTENDS, self._line(node))
        return []

    def _constant_role(self, declared: Declared, owner: _Owner) -> list[tuple[Node, _Owner]]:
        """Record enum constants."""
        self._constants(declared, owner)
        return []

    # ---- declarations ---------------------------------------------------

    def _kind_of(self, declared: Declared) -> SymbolKind:
        """The symbol kind a TYPE or MODULE rule declares for this node."""
        rule = declared.rule
        if rule.keyword_kinds and declared.keyword in rule.keyword_kinds:
            return rule.keyword_kinds[declared.keyword]
        if rule.kind is not None:
            return rule.kind
        return SymbolKind.MODULE if rule.role is Role.MODULE else SymbolKind.CLASS

    def _type(self, declared: Declared, owner: _Owner) -> list[tuple[Node, _Owner]]:
        """Record a type or namespace declaration and hand back its members to walk."""
        node = declared.node
        owner_name, name = name_of(self.spec, declared, self.source)
        if not name:
            return [(child, owner) for child in node.named_children]
        kind = self._kind_of(declared)
        if owner_name:
            name = f"{owner_name}.{name}"
        # A namespace written at file level names itself in full; anything nested joins its owner.
        top_level_module = kind is SymbolKind.MODULE and owner.id == (self.module.id if self.module else "")
        qualified = self._dotted(name) if top_level_module else self._qualify(owner, name)
        body = body_of(declared)
        modifiers = modifiers_of(self.spec, declared, self.source)
        symbol = self._add_symbol(
            Symbol(
                id=make_symbol_id(self.path, qualified),
                kind=kind,
                name=name.rsplit(".", 1)[-1],
                qualified_name=qualified,
                file_path=self.path,
                start_line=self._line(declared.wrapper or node),
                end_line=node.end_point[0] + 1,
                container_id=owner.id,
                modifiers=modifiers,
                annotations=annotations_of(self.spec, declared, self.source),
                signature=self._type_signature(declared, body, kind),
                is_test=self.is_test,
            )
        )
        self._declared_types[qualified] = symbol
        for supertype, line in supertypes_of(declared, self.source):
            self._add_edge(symbol.id, supertype, EdgeKind.EXTENDS, line)
        params = params_of(declared)
        if params is not None:
            for parameter in parameter_nodes(self.spec, declared, params, self.source):
                symbol.param_types.append(parameter_type(parameter, self.source))
                self._parameter_refs(parameter, symbol.id)
        inner = _Owner(
            id=symbol.id,
            qualified=qualified,
            name=symbol.name,
            is_type=kind in DECLARED_TYPE_KINDS,
        )
        pending: list[tuple[Node, _Owner]] = []
        for path in declared.rule.members:
            holder = resolve(node, path)
            if holder is not None:
                pending.extend((child, inner) for child in holder.named_children)
        if body is not None:
            pending.extend((child, inner) for child in body.named_children)
        return pending

    def _type_signature(self, declared: Declared, body: Node | None, kind: SymbolKind) -> str:
        """The declaration line of a type, led by its kind when the source does not spell one."""
        signature = signature_of(declared, body, self.source, self.spec.annotation_kinds)
        if not signature:
            _, name = name_of(self.spec, declared, self.source)
            signature = name
        if kind.value not in signature.split(" ", 4)[:4]:
            signature = f"{kind.value} {signature}"
        return signature

    def _extension(self, declared: Declared, owner: _Owner) -> list[tuple[Node, _Owner]]:
        """Attach an `impl` / `extension` block's members to the type it names."""
        node = declared.node
        target = type_name(self._text(resolve(node, declared.rule.owner or "name")))
        if not target:
            return [(child, owner) for child in node.named_children]
        qualified = self._qualify(owner, target)
        declared_type = self._declared_types.get(qualified)
        if declared_type is not None and declared.rule.trait is not None:
            trait = type_name(self._text(resolve(node, declared.rule.trait)))
            if trait:
                self._add_edge(declared_type.id, trait, EdgeKind.IMPLEMENTS, self._line(node))
        inner = _Owner(
            id=declared_type.id if declared_type is not None else owner.id,
            qualified=qualified,
            name=target.rsplit(".", 1)[-1],
            is_type=True,
        )
        body = body_of(declared)
        return [(child, inner) for child in body.named_children] if body is not None else []

    def _callable(self, declared: Declared, owner: _Owner) -> list[tuple[Node, _Owner]]:
        """Record a function, method or constructor and hand back its body to walk."""
        node = declared.node
        owner_name, name = name_of(self.spec, declared, self.source)
        if not name:
            return [(child, owner) for child in node.named_children]
        if declared.rule.owner is not None:
            owner_name = type_name(self._text(resolve(node, declared.rule.owner))) or owner_name
        holder = owner
        if owner_name:
            # `func (p *Point) Area()`, `void Foo::bar()`, `function M.area()`: the member belongs
            # to the named type when this file declares it, and is qualified by it regardless.
            qualified_owner = self._qualify(owner, owner_name)
            declared_type = self._declared_types.get(qualified_owner)
            holder = _Owner(
                id=declared_type.id if declared_type is not None else owner.id,
                qualified=qualified_owner,
                name=owner_name.rsplit(".", 1)[-1],
                is_type=True,
            )
        params = params_of(declared)
        parameters = parameter_nodes(self.spec, declared, params, self.source)
        param_types = [parameter_type(parameter, self.source) for parameter in parameters]
        kind = self._callable_kind(declared, name, holder)
        qualified = self._qualify(holder, name)
        body = body_of(declared)
        symbol = self._add_symbol(
            Symbol(
                id=make_symbol_id(self.path, qualified, ",".join(param_types)),
                kind=kind,
                name=name,
                qualified_name=qualified,
                file_path=self.path,
                start_line=self._line(declared.wrapper or node),
                end_line=node.end_point[0] + 1,
                container_id=holder.id,
                modifiers=modifiers_of(self.spec, declared, self.source),
                annotations=annotations_of(self.spec, declared, self.source),
                signature=signature_of(declared, body, self.source, self.spec.annotation_kinds) or name,
                is_test=self.is_test,
                param_types=param_types,
            )
        )
        for parameter in parameters:
            self._parameter_refs(parameter, symbol.id)
        if declared.rule.return_type is not None:
            self._type_refs(resolve(node, declared.rule.return_type), symbol.id)
        inner = _Owner(id=symbol.id, qualified=qualified, name=name, is_callable=True)
        if body is None:
            return []
        if body.parent is not node and body is not node:
            self._consumed.add(body.id)
        if body is node:
            skip = {resolve(node, declared.rule.name), params}
            return [(child, inner) for child in node.named_children if child not in skip]
        return [(child, inner) for child in body.named_children]

    def _callable_kind(self, declared: Declared, name: str, holder: _Owner) -> SymbolKind:
        """Whether a callable is a constructor, a method or a free function."""
        spec = self.spec
        if declared.rule.is_constructor or name in spec.constructor_names:
            return SymbolKind.CONSTRUCTOR
        if spec.name_equal_to_type_is_constructor and holder.is_type and name == holder.name:
            return SymbolKind.CONSTRUCTOR
        return SymbolKind.METHOD if holder.is_type else SymbolKind.FUNCTION

    def _fields(self, declared: Declared, owner: _Owner) -> list[tuple[Node, _Owner]]:
        """Record every name a field declaration declares, then walk its initialisers."""
        node = declared.node
        name_nodes = names_of(self.spec, declared, self.source)
        modifiers = modifiers_of(self.spec, declared, self.source)
        annotations = annotations_of(self.spec, declared, self.source)
        declared_names: set[Node] = set()
        first: Symbol | None = None
        for name_node in name_nodes:
            name = clean_name(self._text(name_node))
            if name_node.type in QUALIFIED_KINDS or name_node.type not in NAME_KINDS:
                name = split_qualified(type_name(name))[1]
            if not name:
                continue
            declared_names.add(name_node)
            qualified = self._qualify(owner, name)
            symbol = self._add_symbol(
                Symbol(
                    id=make_symbol_id(self.path, qualified),
                    kind=SymbolKind.FIELD,
                    name=name,
                    qualified_name=qualified,
                    file_path=self.path,
                    start_line=self._line(declared.wrapper or node),
                    end_line=node.end_point[0] + 1,
                    container_id=owner.id,
                    modifiers=modifiers,
                    annotations=annotations,
                    signature=self._text(node).split("=", 1)[0].strip().rstrip(";:")[:_MAX_FIELD_SIGNATURE] or name,
                    is_test=self.is_test,
                )
            )
            first = first or symbol
        if first is None:
            return [(child, owner) for child in node.named_children]
        inner = _Owner(id=first.id, qualified=first.qualified_name, name=first.name, is_callable=True)
        return [(child, inner) for child in node.named_children if child not in declared_names]

    def _constants(self, declared: Declared, owner: _Owner) -> None:
        """Record enum constants, variants and cases."""
        node = declared.node
        for name_node in names_of(self.spec, declared, self.source):
            name = clean_name(self._text(name_node))
            if name_node.type not in NAME_KINDS:
                name = type_name(name)
            if not name:
                continue
            qualified = self._qualify(owner, name)
            self._add_symbol(
                Symbol(
                    id=make_symbol_id(self.path, qualified),
                    kind=SymbolKind.ENUM_CONSTANT,
                    name=name,
                    qualified_name=qualified,
                    file_path=self.path,
                    start_line=self._line(node),
                    end_line=node.end_point[0] + 1,
                    container_id=owner.id,
                    signature=self._text(node).split("{")[0].strip().rstrip(",;")[:_MAX_FIELD_SIGNATURE] or name,
                    is_test=self.is_test,
                )
            )

    def _import(self, declared: Declared) -> None:
        """Record an import: an IMPORTS edge from the module and the names it brings into scope."""
        if self.module is None:
            return
        node = declared.node
        module_node = resolve(node, declared.rule.name)
        module_text = self._text(module_node)
        # A bare `import x.y;` or `use a::b;` starts with the keyword the rule matched on.
        if module_node is node:
            module_text = module_text.split(" ", 1)[-1] if " " in module_text else module_text
        module = self._dotted(module_text.strip("<>"))
        if not module:
            return
        self._add_edge(self.module.id, module, EdgeKind.IMPORTS, self._line(node))
        imports = self.result.imports
        listed = [item for item in names_of(self.spec, declared, self.source) if item is not module_node]
        if declared.rule.names is None or not listed:
            imports.explicit.setdefault(module.rsplit(".", 1)[-1], module)
            return
        for item in listed:
            alias_node = item.child_by_field_name("alias")
            imported_node = item.child_by_field_name("name") or item
            imported = self._dotted(self._text(imported_node))
            if alias_node is not None:
                local = clean_name(self._text(alias_node))
            elif " as " in self._text(item):
                imported, _, local = self._text(item).partition(" as ")
                imported, local = self._dotted(imported), clean_name(local)
            else:
                local = imported.rsplit(".", 1)[-1]
            if not imported or not local:
                continue
            target = imported if module_node is None or module_node is node else f"{module}.{imported}"
            imports.explicit.setdefault(local, target)

    # ---- references ------------------------------------------------------

    def _parameter_refs(self, parameter: Node, owner_id: str) -> None:
        """Emit the type references of one parameter: its `type` field, else its typed descendants."""
        typed = parameter.child_by_field_name("type")
        if typed is not None:
            self._type_refs(typed, owner_id)
            return
        queue: deque[Node] = deque(parameter.named_children)
        while queue:
            current = queue.popleft()
            if current.type in self.spec.type_ref_kinds:
                self._type_refs(current, owner_id)
                continue
            queue.extend(current.named_children)

    def _type_refs(self, node: Node | None, owner_id: str) -> None:
        """Emit REFERENCES_TYPE edges for every type name inside a type position."""
        if node is None:
            return
        queue: deque[Node] = deque([node])
        while queue:
            current = queue.popleft()
            if current.type in NAME_KINDS:
                name = clean_name(self._text(current))
                if name and name not in self.spec.builtin_types and name not in self.spec.self_names:
                    self._add_edge(owner_id, name, EdgeKind.REFERENCES_TYPE, self._line(current))
                continue
            queue.extend(current.named_children)

    def _emit_call(self, call: Call, owner: _Owner) -> None:
        """Emit the CALLS edge of one recognised call, and the type reference a typed receiver is."""
        if call.receiver_type is not None and call.receiver == call.receiver_type:
            self._add_edge(owner.id, call.receiver_type, EdgeKind.REFERENCES_TYPE, call.line)
        self._add_edge(
            owner.id,
            call.name,
            EdgeKind.CALLS,
            call.line,
            arity=call.arity,
            receiver=call.receiver,
            receiver_type=call.receiver_type,
            is_new=call.is_new,
        )


_ROLE_HANDLERS = {
    Role.SKIP: _GenericExtractor._nothing,
    Role.PACKAGE: _GenericExtractor._nothing,
    Role.IMPORT: _GenericExtractor._import_role,
    Role.SUPERTYPE: _GenericExtractor._supertype,
    Role.TYPE: _GenericExtractor._type,
    Role.MODULE: _GenericExtractor._type,
    Role.EXTENSION: _GenericExtractor._extension,
    Role.CALLABLE: _GenericExtractor._callable,
    Role.FIELD: _GenericExtractor._fields,
    Role.CONSTANT: _GenericExtractor._constant_role,
}


def _top_level(root: Node) -> list[Node]:
    """The file's top-level nodes, error wrappers looked through."""
    found: list[Node] = []
    for child in root.named_children:
        if child.type == "ERROR":
            found.extend(child.named_children)
        else:
            found.append(child)
    return found


def _dedupe_edges(edges: list[Edge]) -> list[Edge]:
    """Drop edges that repeat the same relationship on the same line."""
    seen: set[tuple[str, str, str, int, int, bool]] = set()
    kept: list[Edge] = []
    for edge in edges:
        key = (edge.src_id, edge.dst_name, edge.kind.value, edge.line, edge.arity, edge.is_new)
        if key in seen:
            continue
        seen.add(key)
        kept.append(edge)
    return kept


__all__ = ["extract_generic_file", "language_parser"]
