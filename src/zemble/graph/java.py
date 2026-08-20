"""Tree-sitter based Java symbol and reference extraction for one file.

The extractor is purely local: it never looks outside the file it is given. Every
cross-file question (which type is `Foo`, which overload does this call hit) is
left to :mod:`zemble.graph.resolve`, which sees the whole workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from logging import getLogger

from semble_grammars import LanguageNotFoundError, UnsupportedPlatformError, get_parser
from tree_sitter import Node, Parser

from zemble.graph.model import Edge, EdgeKind, Symbol, SymbolKind, is_test_path, make_symbol_id

logger = getLogger(__name__)

_TYPE_DECLARATIONS: dict[str, SymbolKind] = {
    "class_declaration": SymbolKind.CLASS,
    "interface_declaration": SymbolKind.INTERFACE,
    "enum_declaration": SymbolKind.ENUM,
    "record_declaration": SymbolKind.RECORD,
    "annotation_type_declaration": SymbolKind.ANNOTATION,
}

_BODY_NODES = frozenset({"class_body", "interface_body", "enum_body", "annotation_type_body", "enum_body_declarations"})

# Nodes that name a type in a type position.
_TYPE_NAME_NODES = frozenset({"type_identifier", "scoped_type_identifier"})

# Nodes that hold a type in a binding position (for-each, catch, resource).
_TYPE_HOLDER_NODES = frozenset(
    {"type_identifier", "scoped_type_identifier", "generic_type", "array_type", "catch_type"}
)

# Primitive and pseudo types that are never workspace symbols.
_NON_TYPES = frozenset({"var", "void", "int", "long", "short", "byte", "char", "boolean", "float", "double"})


@dataclass
class FileImports:
    """The import declarations of a single compilation unit."""

    explicit: dict[str, str] = field(default_factory=dict)
    wildcards: list[str] = field(default_factory=list)
    static_members: dict[str, str] = field(default_factory=dict)
    static_wildcards: list[str] = field(default_factory=list)


@dataclass
class FileExtraction:
    """Everything one Java file contributes to the graph before resolution."""

    file_path: str
    package: str
    symbols: list[Symbol] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    imports: FileImports = field(default_factory=FileImports)


@cache
def java_parser() -> Parser | None:
    """Return the bundled tree-sitter Java parser, or None when it is unavailable."""
    try:
        return get_parser("java")
    except (LanguageNotFoundError, UnsupportedPlatformError):
        logger.warning("No bundled tree-sitter Java grammar on this platform; the graph cannot be built")
    except Exception:
        logger.error("Uncaught exception while loading the Java grammar", exc_info=True)
    return None


def extract_java_file(source: bytes, relative_path: str) -> FileExtraction:
    """Extract symbols and unresolved references from one Java source file.

    :param source: Raw file bytes.
    :param relative_path: Path relative to the indexed root, posix style.
    :return: The file's symbols, edges and imports.
    :raises RuntimeError: If no Java grammar is available on this platform.
    """
    parser = java_parser()
    if parser is None:
        raise RuntimeError("No tree-sitter Java grammar available")
    tree = parser.parse(source)
    extractor = _JavaExtractor(source, relative_path)
    extractor.run(tree.root_node)
    return extractor.result


class _Scope:
    """Declared variable types visible at a point in a method body."""

    def __init__(self, parent: _Scope | None = None) -> None:
        """Create a scope chained to an optional parent."""
        self.parent = parent
        self.names: dict[str, str] = {}

    def declare(self, name: str, type_name: str | None) -> None:
        """Record a variable's declared type, ignoring inferred (`var`) declarations."""
        if type_name and type_name not in _NON_TYPES:
            self.names[name] = type_name

    def lookup(self, name: str) -> str | None:
        """Find a variable's declared type in this scope or an enclosing one."""
        scope: _Scope | None = self
        while scope is not None:
            if name in scope.names:
                return scope.names[name]
            scope = scope.parent
        return None


class _JavaExtractor:
    """Walks one parsed Java file and records its symbols and references."""

    def __init__(self, source: bytes, relative_path: str) -> None:
        """Prepare an extractor for one file."""
        self.source = source
        self.result = FileExtraction(file_path=relative_path, package="")
        self.is_test = is_test_path(relative_path)
        self._anon_counter = 0
        # Type variables in scope (class/method type parameters). They name no workspace
        # symbol, so references to them are dropped rather than left permanently unresolved.
        self._type_vars: set[str] = set()

    # ---- helpers -------------------------------------------------------

    def text(self, node: Node) -> str:
        """Return the source text of a node with newlines and runs of spaces collapsed."""
        return " ".join(self.source[node.start_byte : node.end_byte].decode("utf-8", "replace").split())

    def _add_symbol(self, symbol: Symbol) -> Symbol:
        """Append a symbol to the extraction result."""
        self.result.symbols.append(symbol)
        return symbol

    def _add_edge(self, src_id: str, dst_name: str, kind: EdgeKind, line: int, **kwargs: object) -> None:
        """Append an unresolved edge to the extraction result."""
        if not dst_name:
            return
        self.result.edges.append(Edge(src_id=src_id, dst_name=dst_name, kind=kind, line=line, **kwargs))  # type: ignore[arg-type]

    @staticmethod
    def _line(node: Node) -> int:
        """Return the 1-indexed start line of a node."""
        return node.start_point[0] + 1

    def _modifiers_of(self, node: Node) -> tuple[list[str], list[str]]:
        """Return (keyword modifiers, annotation simple names) of a declaration."""
        modifiers_node = next((child for child in node.children if child.type == "modifiers"), None)
        if modifiers_node is None:
            return [], []
        keywords = [child.type for child in modifiers_node.children if not child.is_named]
        annotations = []
        for child in modifiers_node.children:
            if child.type in ("marker_annotation", "annotation"):
                name_node = child.child_by_field_name("name")
                if name_node is not None:
                    annotations.append(self.text(name_node).rsplit(".", 1)[-1])
        return keywords, annotations

    def _emit_annotations(self, node: Node, owner_id: str) -> None:
        """Emit ANNOTATED_WITH edges for every annotation on a declaration."""
        modifiers_node = next((child for child in node.children if child.type == "modifiers"), None)
        if modifiers_node is None:
            return
        for child in modifiers_node.children:
            if child.type in ("marker_annotation", "annotation"):
                name_node = child.child_by_field_name("name")
                if name_node is not None:
                    self._add_edge(owner_id, self.text(name_node), EdgeKind.ANNOTATED_WITH, self._line(child))

    def _erase(self, node: Node | None) -> str | None:
        """Reduce a type node to its erased name (generics dropped, arrays kept as `[]`)."""
        if node is None:
            return None
        if node.type == "generic_type":
            base = node.child_by_field_name("type") or next((c for c in node.children if c.is_named), None)
            return self._erase(base)
        if node.type == "array_type":
            element = self._erase(node.child_by_field_name("element"))
            return f"{element}[]" if element else None
        if node.type in _TYPE_NAME_NODES:
            return self.text(node)
        if node.type in ("integral_type", "floating_point_type", "boolean_type", "void_type"):
            return self.text(node)
        return None

    def _emit_type_refs(self, node: Node | None, owner_id: str) -> None:
        """Emit REFERENCES_TYPE edges for every named type inside a type node."""
        if node is None:
            return
        if node.type in _TYPE_NAME_NODES:
            name = self.text(node)
            if name not in _NON_TYPES and name not in self._type_vars:
                self._add_edge(owner_id, name, EdgeKind.REFERENCES_TYPE, self._line(node))
            return  # do not descend: `Map.Entry` is one reference, not two
        for child in node.children:
            self._emit_type_refs(child, owner_id)

    def _param_types(self, parameters: Node | None) -> list[str]:
        """Return the erased parameter types of a formal parameter list."""
        if parameters is None:
            return []
        types: list[str] = []
        for child in parameters.named_children:
            if child.type == "formal_parameter":
                types.append(self._erase(child.child_by_field_name("type")) or "?")
            elif child.type == "spread_parameter":
                erased = self._erase(next((c for c in child.named_children if c.type != "variable_declarator"), None))
                types.append(f"{erased}[]" if erased else "?[]")
            elif child.type == "receiver_parameter":
                continue
        return types

    def _declare_parameters(self, parameters: Node | None, scope: _Scope, owner_id: str) -> None:
        """Record parameter names and types in a scope and emit their type references."""
        if parameters is None:
            return
        for child in parameters.named_children:
            if child.type not in ("formal_parameter", "spread_parameter"):
                continue
            type_node = child.child_by_field_name("type") or next(
                (c for c in child.named_children if c.type != "variable_declarator"), None
            )
            self._emit_type_refs(type_node, owner_id)
            name_node = child.child_by_field_name("name")
            if name_node is None:
                declarator = next((c for c in child.named_children if c.type == "variable_declarator"), None)
                name_node = declarator.child_by_field_name("name") if declarator is not None else None
            if name_node is not None:
                scope.declare(self.text(name_node), self._erase(type_node))

    def _push_type_vars(self, node: Node) -> set[str]:
        """Add a declaration's type parameters to the in-scope type variables, returning the added names."""
        type_parameters = node.child_by_field_name("type_parameters")
        if type_parameters is None:
            return set()
        added = {
            self.text(name)
            for child in type_parameters.named_children
            if child.type == "type_parameter"
            for name in [next((c for c in child.named_children if c.type == "type_identifier"), None)]
            if name is not None
        } - self._type_vars
        self._type_vars |= added
        return added

    # ---- top level -----------------------------------------------------

    def run(self, root: Node) -> None:
        """Extract the whole compilation unit."""
        package_symbol = self._read_header(root)
        container_id = package_symbol.id if package_symbol is not None else None
        prefix = self.result.package
        for child in root.named_children:
            if child.type in _TYPE_DECLARATIONS:
                self._visit_type(child, container_id, prefix)
        self.result.edges = _dedupe_edges(self.result.edges)

    def _read_header(self, root: Node) -> Symbol | None:
        """Read the package declaration and imports, returning the file's package symbol."""
        package_node = next((c for c in root.named_children if c.type == "package_declaration"), None)
        package_symbol: Symbol | None = None
        if package_node is not None:
            name_node = next((c for c in package_node.named_children), None)
            self.result.package = self.text(name_node) if name_node is not None else ""
            package_symbol = self._add_symbol(
                Symbol(
                    id=make_symbol_id(self.result.file_path, self.result.package),
                    kind=SymbolKind.PACKAGE,
                    name=self.result.package.rsplit(".", 1)[-1],
                    qualified_name=self.result.package,
                    file_path=self.result.file_path,
                    start_line=self._line(package_node),
                    end_line=self._line(package_node),
                    signature=f"package {self.result.package}",
                    is_test=self.is_test,
                )
            )
        for child in root.named_children:
            if child.type == "import_declaration":
                self._read_import(child, package_symbol)
        return package_symbol

    def _read_import(self, node: Node, package_symbol: Symbol | None) -> None:
        """Record one import declaration."""
        is_static = any(child.type == "static" for child in node.children)
        is_wildcard = any(child.type == "asterisk" for child in node.children)
        name_node = next((c for c in node.named_children if c.type in ("scoped_identifier", "identifier")), None)
        if name_node is None:
            return
        qualified = self.text(name_node)
        imports = self.result.imports
        if is_static and is_wildcard:
            imports.static_wildcards.append(qualified)
        elif is_static:
            owner, _, member = qualified.rpartition(".")
            imports.static_members[member] = owner
        elif is_wildcard:
            imports.wildcards.append(qualified)
        else:
            imports.explicit[qualified.rsplit(".", 1)[-1]] = qualified
        if package_symbol is not None:
            self._add_edge(package_symbol.id, qualified, EdgeKind.IMPORTS, self._line(node))

    # ---- type declarations ---------------------------------------------

    def _visit_type(self, node: Node, container_id: str | None, prefix: str) -> Symbol:
        """Extract one type declaration and everything inside it."""
        kind = _TYPE_DECLARATIONS[node.type]
        added_vars = self._push_type_vars(node)
        name_node = node.child_by_field_name("name")
        name = self.text(name_node) if name_node is not None else "<anonymous>"
        qualified = f"{prefix}.{name}" if prefix else name
        modifiers, annotations = self._modifiers_of(node)
        symbol = self._add_symbol(
            Symbol(
                id=make_symbol_id(self.result.file_path, qualified),
                kind=kind,
                name=name,
                qualified_name=qualified,
                file_path=self.result.file_path,
                start_line=self._line(node),
                end_line=node.end_point[0] + 1,
                container_id=container_id,
                modifiers=modifiers,
                annotations=annotations,
                signature=self._type_signature(node, kind, name),
                is_test=self.is_test,
            )
        )
        self._emit_annotations(node, symbol.id)
        self._emit_supertypes(node, symbol)
        fields = self._record_components(node, symbol)
        body = node.child_by_field_name("body")
        if body is not None:
            self._visit_body(body, symbol, qualified, fields)
        self._type_vars -= added_vars
        return symbol

    def _type_signature(self, node: Node, kind: SymbolKind, name: str) -> str:
        """Build the one-line declaration signature of a type."""
        keyword = {SymbolKind.ANNOTATION: "@interface"}.get(kind, kind.value)
        parts = [f"{keyword} {name}"]
        for field_name in ("type_parameters", "parameters", "superclass", "interfaces"):
            child = node.child_by_field_name(field_name)
            if child is not None:
                parts.append(self.text(child))
        extends_interfaces = next((c for c in node.children if c.type == "extends_interfaces"), None)
        if extends_interfaces is not None:
            parts.append(self.text(extends_interfaces))
        return " ".join(parts).replace(" <", "<").replace(" (", "(")

    def _emit_supertypes(self, node: Node, symbol: Symbol) -> None:
        """Emit EXTENDS and IMPLEMENTS edges for a type declaration."""
        superclass = node.child_by_field_name("superclass")
        if superclass is not None:
            for child in superclass.named_children:
                self._add_edge(symbol.id, self.text(child), EdgeKind.EXTENDS, self._line(child))
        interfaces = node.child_by_field_name("interfaces")
        extends_interfaces = next((c for c in node.children if c.type == "extends_interfaces"), None)
        for holder, kind in ((interfaces, EdgeKind.IMPLEMENTS), (extends_interfaces, EdgeKind.EXTENDS)):
            if holder is None:
                continue
            type_list = next((c for c in holder.named_children if c.type == "type_list"), holder)
            for child in type_list.named_children:
                self._add_edge(
                    symbol.id,
                    self.text(child),
                    EdgeKind.EXTENDS if kind is EdgeKind.EXTENDS else kind,
                    self._line(child),
                )

    def _record_components(self, node: Node, owner: Symbol) -> dict[str, str]:
        """Turn record components into FIELD symbols and return the type's field table."""
        fields: dict[str, str] = {}
        parameters = node.child_by_field_name("parameters")
        if node.type != "record_declaration" or parameters is None:
            return fields
        for child in parameters.named_children:
            if child.type != "formal_parameter":
                continue
            name_node = child.child_by_field_name("name")
            type_node = child.child_by_field_name("type")
            if name_node is None:
                continue
            name = self.text(name_node)
            qualified = f"{owner.qualified_name}.{name}"
            self._add_symbol(
                Symbol(
                    id=make_symbol_id(self.result.file_path, qualified),
                    kind=SymbolKind.FIELD,
                    name=name,
                    qualified_name=qualified,
                    file_path=self.result.file_path,
                    start_line=self._line(child),
                    end_line=child.end_point[0] + 1,
                    container_id=owner.id,
                    signature=f"{self.text(type_node) if type_node else '?'} {name}",
                    is_test=self.is_test,
                )
            )
            self._emit_type_refs(type_node, owner.id)
            erased = self._erase(type_node)
            owner.param_types.append(erased or "?")
            if erased:
                fields[name] = erased
        return fields

    def _visit_body(self, body: Node, owner: Symbol, prefix: str, fields: dict[str, str]) -> None:
        """Extract every member of a type body."""
        fields = dict(fields)
        members = list(self._body_members(body))
        for member in members:
            if member.type == "field_declaration":
                fields.update(self._visit_field(member, owner))
        for member in members:
            self._visit_member(member, owner, prefix, fields)

    def _body_members(self, body: Node) -> list[Node]:
        """Flatten a type body, folding an enum's `enum_body_declarations` into the member list."""
        members: list[Node] = []
        for child in body.named_children:
            if child.type == "enum_body_declarations":
                members.extend(child.named_children)
            else:
                members.append(child)
        return members

    def _visit_member(self, node: Node, owner: Symbol, prefix: str, fields: dict[str, str]) -> None:
        """Dispatch one member of a type body."""
        if node.type in _TYPE_DECLARATIONS:
            self._visit_type(node, owner.id, prefix)
        elif node.type in ("method_declaration", "annotation_type_element_declaration"):
            self._visit_callable(node, owner, SymbolKind.METHOD, fields)
        elif node.type in ("constructor_declaration", "compact_constructor_declaration"):
            self._visit_callable(node, owner, SymbolKind.CONSTRUCTOR, fields)
        elif node.type == "enum_constant":
            self._visit_enum_constant(node, owner, prefix, fields)
        elif node.type in ("static_initializer", "block"):
            self._walk_statements(node, owner, _scope_with(fields), prefix)

    def _visit_field(self, node: Node, owner: Symbol) -> dict[str, str]:
        """Turn one field declaration into FIELD symbols, returning name -> erased type."""
        type_node = node.child_by_field_name("type")
        self._emit_type_refs(type_node, owner.id)
        erased = self._erase(type_node)
        modifiers, annotations = self._modifiers_of(node)
        declared: dict[str, str] = {}
        for declarator in node.named_children:
            if declarator.type != "variable_declarator":
                continue
            name_node = declarator.child_by_field_name("name")
            if name_node is None:
                continue
            name = self.text(name_node)
            qualified = f"{owner.qualified_name}.{name}"
            symbol = self._add_symbol(
                Symbol(
                    id=make_symbol_id(self.result.file_path, qualified),
                    kind=SymbolKind.FIELD,
                    name=name,
                    qualified_name=qualified,
                    file_path=self.result.file_path,
                    start_line=self._line(node),
                    end_line=node.end_point[0] + 1,
                    container_id=owner.id,
                    modifiers=modifiers,
                    annotations=annotations,
                    signature=f"{self.text(type_node) if type_node else '?'} {name}",
                    is_test=self.is_test,
                )
            )
            self._emit_annotations(node, symbol.id)
            if erased:
                declared[name] = erased
            value = declarator.child_by_field_name("value")
            if value is not None:
                self._walk_statements(value, symbol, _scope_with({}), owner.qualified_name)
        return declared

    def _visit_enum_constant(self, node: Node, owner: Symbol, prefix: str, fields: dict[str, str]) -> None:
        """Extract an enum constant and, when present, its constant class body."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = self.text(name_node)
        qualified = f"{owner.qualified_name}.{name}"
        symbol = self._add_symbol(
            Symbol(
                id=make_symbol_id(self.result.file_path, qualified),
                kind=SymbolKind.ENUM_CONSTANT,
                name=name,
                qualified_name=qualified,
                file_path=self.result.file_path,
                start_line=self._line(node),
                end_line=node.end_point[0] + 1,
                container_id=owner.id,
                signature=self.text(node).split("{")[0].strip(),
                is_test=self.is_test,
            )
        )
        self._emit_annotations(node, symbol.id)
        arguments = node.child_by_field_name("arguments")
        if arguments is not None:
            self._add_edge(
                symbol.id,
                owner.name,
                EdgeKind.CALLS,
                self._line(node),
                arity=len(arguments.named_children),
                is_new=True,
            )
            self._walk_statements(arguments, symbol, _scope_with(fields), prefix)
        body = next((c for c in node.named_children if c.type == "class_body"), None)
        if body is not None:
            self._visit_body(body, symbol, qualified, fields)

    def _visit_callable(self, node: Node, owner: Symbol, kind: SymbolKind, fields: dict[str, str]) -> None:
        """Extract a method, constructor or annotation element and walk its body."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        added_vars = self._push_type_vars(node)
        name = self.text(name_node)
        parameters = node.child_by_field_name("parameters")
        param_types = (
            list(owner.param_types) if node.type == "compact_constructor_declaration" else self._param_types(parameters)
        )
        modifiers, annotations = self._modifiers_of(node)
        return_node = node.child_by_field_name("type")
        qualified = f"{owner.qualified_name}.{name}"
        symbol = self._add_symbol(
            Symbol(
                id=make_symbol_id(self.result.file_path, qualified, ",".join(param_types)),
                kind=kind,
                name=name,
                qualified_name=qualified,
                file_path=self.result.file_path,
                start_line=self._line(node),
                end_line=node.end_point[0] + 1,
                container_id=owner.id,
                modifiers=modifiers,
                annotations=annotations,
                signature=self._callable_signature(node, name, parameters, return_node),
                is_test=self.is_test,
                param_types=param_types,
            )
        )
        self._emit_annotations(node, symbol.id)
        self._emit_type_refs(return_node, symbol.id)
        throws = next((c for c in node.children if c.type == "throws"), None)
        self._emit_type_refs(throws, symbol.id)
        scope = _scope_with(fields)
        body_scope = _Scope(scope)
        self._declare_parameters(parameters, body_scope, symbol.id)
        body = node.child_by_field_name("body")
        if body is not None:
            self._walk_statements(body, symbol, body_scope, owner.qualified_name)
        self._type_vars -= added_vars

    def _callable_signature(self, node: Node, name: str, parameters: Node | None, return_node: Node | None) -> str:
        """Build the one-line signature of a callable, keeping declared generics."""
        type_parameters = node.child_by_field_name("type_parameters")
        parts: list[str] = []
        if type_parameters is not None:
            parts.append(self.text(type_parameters))
        if return_node is not None:
            parts.append(self.text(return_node))
        params = self.text(parameters) if parameters is not None else "()"
        parts.append(f"{name}{params}")
        return " ".join(parts)

    # ---- statements and expressions -------------------------------------

    def _walk_statements(self, node: Node, owner: Symbol, scope: _Scope, prefix: str) -> None:
        """Walk a body, attributing every reference to the owning symbol."""
        handler = _STATEMENT_HANDLERS.get(node.type)
        if handler is not None and handler(self, node, owner, scope, prefix):
            return
        for child in node.named_children:
            self._walk_statements(child, owner, scope, prefix)

    def _on_block(self, node: Node, owner: Symbol, scope: _Scope, prefix: str) -> bool:
        """Open a nested scope for a block."""
        inner = _Scope(scope)
        for child in node.named_children:
            self._walk_statements(child, owner, inner, prefix)
        return True

    def _on_local_variable(self, node: Node, owner: Symbol, scope: _Scope, prefix: str) -> bool:
        """Record local variable types and walk their initialisers."""
        type_node = node.child_by_field_name("type")
        self._emit_type_refs(type_node, owner.id)
        erased = self._erase(type_node)
        for declarator in node.named_children:
            if declarator.type != "variable_declarator":
                continue
            name_node = declarator.child_by_field_name("name")
            if name_node is not None:
                scope.declare(self.text(name_node), erased)
            value = declarator.child_by_field_name("value")
            if value is not None:
                self._walk_statements(value, owner, scope, prefix)
        return True

    def _on_typed_binding(self, node: Node, owner: Symbol, scope: _Scope, prefix: str) -> bool:
        """Handle for-each / resource / catch / instanceof pattern bindings."""
        type_node = node.child_by_field_name("type") or next(
            (c for c in node.named_children if c.type in _TYPE_HOLDER_NODES), None
        )
        self._emit_type_refs(type_node, owner.id)
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            scope.declare(self.text(name_node), self._erase(type_node))
        for child in node.named_children:
            if child is not type_node and child is not name_node:
                self._walk_statements(child, owner, scope, prefix)
        return True

    def _on_instanceof(self, node: Node, owner: Symbol, scope: _Scope, prefix: str) -> bool:
        """Record an instanceof pattern binding and its type reference."""
        named = node.named_children
        for child in named:
            if child.type in _TYPE_NAME_NODES or child.type in ("generic_type", "array_type"):
                self._emit_type_refs(child, owner.id)
                erased = self._erase(child)
                binding = named[named.index(child) + 1] if named.index(child) + 1 < len(named) else None
                if binding is not None and binding.type == "identifier":
                    scope.declare(self.text(binding), erased)
            elif child.type != "identifier":
                self._walk_statements(child, owner, scope, prefix)
        return True

    def _on_type_position(self, node: Node, owner: Symbol, scope: _Scope, prefix: str) -> bool:
        """Emit type references for a cast or a bare type node and walk the value."""
        type_node = node.child_by_field_name("type")
        self._emit_type_refs(type_node, owner.id)
        value = node.child_by_field_name("value")
        if value is not None:
            self._walk_statements(value, owner, scope, prefix)
        return True

    def _on_method_invocation(self, node: Node, owner: Symbol, scope: _Scope, prefix: str) -> bool:
        """Emit a CALLS edge for a method invocation and walk its receiver and arguments."""
        name_node = node.child_by_field_name("name")
        arguments = node.child_by_field_name("arguments")
        object_node = node.child_by_field_name("object")
        receiver, receiver_type = self._receiver_of(object_node, scope)
        if receiver is not None and receiver == receiver_type:
            # `Foo.bar()`: the receiver text is a type name, which is a type reference.
            self._add_edge(owner.id, receiver_type, EdgeKind.REFERENCES_TYPE, self._line(node))
        if name_node is not None:
            self._add_edge(
                owner.id,
                self.text(name_node),
                EdgeKind.CALLS,
                self._line(name_node),
                arity=len(arguments.named_children) if arguments is not None else -1,
                receiver=receiver,
                receiver_type=receiver_type,
            )
        if object_node is not None:
            self._walk_statements(object_node, owner, scope, prefix)
        if arguments is not None:
            self._walk_statements(arguments, owner, scope, prefix)
        return True

    def _receiver_of(self, object_node: Node | None, scope: _Scope) -> tuple[str | None, str | None]:
        """Determine a call receiver's written text and, when known locally, its declared type."""
        if object_node is None:
            return None, None
        if object_node.type in ("this", "super"):
            return object_node.type, None
        text = self.text(object_node)
        if object_node.type == "identifier":
            declared = scope.lookup(text)
            if declared is not None:
                return text, declared
            return text, text if text[:1].isupper() else None
        if object_node.type == "object_creation_expression":
            return text, self._erase(object_node.child_by_field_name("type"))
        if object_node.type == "field_access":
            field_node = object_node.child_by_field_name("field")
            owner_node = object_node.child_by_field_name("object")
            if field_node is not None and owner_node is not None and owner_node.type == "this":
                declared = scope.lookup(self.text(field_node))
                return text, declared
            return text, text if text.rsplit(".", 1)[-1][:1].isupper() else None
        if object_node.type == "parenthesized_expression":
            inner = next((c for c in object_node.named_children), None)
            return self._receiver_of(inner, scope)
        if object_node.type == "cast_expression":
            return text, self._erase(object_node.child_by_field_name("type"))
        return text, None

    def _on_object_creation(self, node: Node, owner: Symbol, scope: _Scope, prefix: str) -> bool:
        """Emit a constructor CALLS edge and, for anonymous classes, a type symbol."""
        type_node = node.child_by_field_name("type")
        self._emit_type_refs(type_node, owner.id)
        arguments = node.child_by_field_name("arguments")
        erased = self._erase(type_node)
        if erased:
            self._add_edge(
                owner.id,
                erased,
                EdgeKind.CALLS,
                self._line(node),
                arity=len(arguments.named_children) if arguments is not None else -1,
                is_new=True,
            )
        if arguments is not None:
            self._walk_statements(arguments, owner, scope, prefix)
        body = next((c for c in node.named_children if c.type == "class_body"), None)
        if body is not None:
            self._visit_anonymous(body, node, owner, prefix, erased, scope)
        return True

    def _visit_anonymous(
        self, body: Node, node: Node, owner: Symbol, prefix: str, supertype: str | None, scope: _Scope
    ) -> None:
        """Create the symbol for an anonymous class and extract its members."""
        self._anon_counter += 1
        line = self._line(node)
        name = f"$anon@{line}"
        qualified = f"{prefix}${'anon'}@{line}"
        symbol = self._add_symbol(
            Symbol(
                id=make_symbol_id(self.result.file_path, qualified, str(self._anon_counter)),
                kind=SymbolKind.CLASS,
                name=name,
                qualified_name=qualified,
                file_path=self.result.file_path,
                start_line=line,
                end_line=node.end_point[0] + 1,
                container_id=owner.id,
                signature=f"new {supertype or '?'}() {{...}}",
                is_test=self.is_test,
            )
        )
        if supertype:
            # Emitted as EXTENDS; resolution rewrites it to IMPLEMENTS when the target is an interface.
            self._add_edge(symbol.id, supertype, EdgeKind.EXTENDS, line)
        self._visit_body(body, symbol, qualified, dict(scope.names))

    def _on_local_type(self, node: Node, owner: Symbol, scope: _Scope, prefix: str) -> bool:
        """Extract a class declared inside a method body."""
        self._visit_type(node, owner.id, owner.qualified_name)
        return True

    def _on_method_reference(self, node: Node, owner: Symbol, scope: _Scope, prefix: str) -> bool:
        """Emit a CALLS edge with unknown arity for a `Receiver::member` reference."""
        named = node.named_children
        if len(named) < 2:
            return True
        target, member = named[0], named[-1]
        receiver, receiver_type = self._receiver_of(target, scope)
        if member.type == "identifier":
            self._add_edge(
                owner.id,
                self.text(member),
                EdgeKind.CALLS,
                self._line(node),
                arity=-1,
                receiver=receiver,
                receiver_type=receiver_type,
            )
        elif target.type in _TYPE_NAME_NODES:
            self._emit_type_refs(target, owner.id)
        if target.type == "identifier" and self.text(target)[:1].isupper():
            self._add_edge(owner.id, self.text(target), EdgeKind.REFERENCES_TYPE, self._line(node))
        return True

    def _on_field_access(self, node: Node, owner: Symbol, scope: _Scope, prefix: str) -> bool:
        """Record a static field access on a type name as a type reference."""
        object_node = node.child_by_field_name("object")
        if object_node is not None and object_node.type == "identifier":
            text = self.text(object_node)
            if text[:1].isupper() and scope.lookup(text) is None:
                self._add_edge(owner.id, text, EdgeKind.REFERENCES_TYPE, self._line(node))
                return True
        if object_node is not None:
            self._walk_statements(object_node, owner, scope, prefix)
        return True

    def _on_explicit_constructor_invocation(self, node: Node, owner: Symbol, scope: _Scope, prefix: str) -> bool:
        """Emit a CALLS edge for a `this(...)` or `super(...)` chained constructor call."""
        constructor = node.child_by_field_name("constructor")
        arguments = node.child_by_field_name("arguments")
        if constructor is not None:
            self._add_edge(
                owner.id,
                constructor.type,
                EdgeKind.CALLS,
                self._line(node),
                arity=len(arguments.named_children) if arguments is not None else -1,
                receiver=constructor.type,
                is_new=True,
            )
        if arguments is not None:
            self._walk_statements(arguments, owner, scope, prefix)
        return True

    def _on_bare_type(self, node: Node, owner: Symbol, scope: _Scope, prefix: str) -> bool:
        """Emit references for a type node reached directly (throws lists, type arguments)."""
        self._emit_type_refs(node, owner.id)
        return True

    def _on_skip(self, node: Node, owner: Symbol, scope: _Scope, prefix: str) -> bool:
        """Ignore a node and everything under it (comments, literals)."""
        return True


def _dedupe_edges(edges: list[Edge]) -> list[Edge]:
    """Drop edges that repeat the same relationship on the same line.

    A single declaration mentions the same type more than once (a local variable's
    declared type and the `new` on its right hand side), which is one reference.
    """
    seen: set[tuple[str, str, str, int, int, bool]] = set()
    kept: list[Edge] = []
    for edge in edges:
        key = (edge.src_id, edge.dst_name, edge.kind.value, edge.line, edge.arity, edge.is_new)
        if key in seen:
            continue
        seen.add(key)
        kept.append(edge)
    return kept


def _scope_with(names: dict[str, str]) -> _Scope:
    """Build a root scope pre-populated with a type's field table."""
    scope = _Scope()
    scope.names.update(names)
    return scope


_STATEMENT_HANDLERS = {
    "block": _JavaExtractor._on_block,
    "local_variable_declaration": _JavaExtractor._on_local_variable,
    "enhanced_for_statement": _JavaExtractor._on_typed_binding,
    "resource": _JavaExtractor._on_typed_binding,
    "catch_formal_parameter": _JavaExtractor._on_typed_binding,
    "instanceof_expression": _JavaExtractor._on_instanceof,
    "cast_expression": _JavaExtractor._on_type_position,
    "method_invocation": _JavaExtractor._on_method_invocation,
    "object_creation_expression": _JavaExtractor._on_object_creation,
    "explicit_constructor_invocation": _JavaExtractor._on_explicit_constructor_invocation,
    "method_reference": _JavaExtractor._on_method_reference,
    "field_access": _JavaExtractor._on_field_access,
    "class_declaration": _JavaExtractor._on_local_type,
    "interface_declaration": _JavaExtractor._on_local_type,
    "enum_declaration": _JavaExtractor._on_local_type,
    "record_declaration": _JavaExtractor._on_local_type,
    "type_identifier": _JavaExtractor._on_bare_type,
    "scoped_type_identifier": _JavaExtractor._on_bare_type,
    "generic_type": _JavaExtractor._on_bare_type,
    "line_comment": _JavaExtractor._on_skip,
    "block_comment": _JavaExtractor._on_skip,
    "string_literal": _JavaExtractor._on_skip,
}

__all__ = ["FileExtraction", "FileImports", "extract_java_file", "java_parser"]
