"""Symbol and reference extraction for one Hawkeye `.hwk` template.

Like the Java extractor this is purely local: it reads the template's own text and leaves
every cross-file question - which class backs `<pl-button>`, which method backs
`String.presence` - to :mod:`zemble.graph.resolve`, which sees the whole workspace.
"""

from __future__ import annotations

from zemble.graph.java import FileExtraction
from zemble.graph.model import Edge, EdgeKind, Symbol, SymbolKind, is_test_path, make_symbol_id
from zemble.hwk import TemplateFacts, scan, template_id_path

#: Marker on a TEMPLATE symbol that declares a custom element, so the resolver can tell a
#: component apart from a page without re-reading the source.
TAG_MODIFIER = "tag"


def extract_hwk_file(source: bytes, relative_path: str) -> FileExtraction:
    """Extract the symbols and unresolved references of one Hawkeye template.

    :param source: Raw file bytes.
    :param relative_path: Path relative to the indexed root, posix style.
    :return: The file's symbols and edges; a template has no package and no imports.
    """
    return _Builder(scan(source.decode("utf-8", "replace")), relative_path).run()


class _Builder:
    """Turns one template's lexical facts into graph symbols and edges."""

    def __init__(self, facts: TemplateFacts, relative_path: str) -> None:
        """Prepare a builder for one template."""
        self.facts = facts
        self.path = relative_path
        self.is_test = is_test_path(relative_path)
        self.result = FileExtraction(file_path=relative_path, package="")
        self.file_symbol = self._file_symbol()
        self.owners: list[tuple[int, int, str]] = []

    def run(self) -> FileExtraction:
        """Build the template's symbols and edges."""
        self.result.symbols.append(self.file_symbol)
        self._tag_symbols()
        self._block_symbols()
        self._edges()
        return self.result

    def _file_symbol(self) -> Symbol:
        """Build the symbol standing for the file itself.

        A file declaring exactly one custom element IS that element, so it is named and
        looked up by the tag; anything else is named by its template id.
        """
        single = self.facts.tag if len(self.facts.tags) == 1 else None
        template_id = template_id_path(self.path)
        return Symbol(
            id=make_symbol_id(self.path, single or self.path),
            kind=SymbolKind.TEMPLATE,
            name=single or template_id.rsplit("/", 1)[-1],
            qualified_name=single or self.path,
            file_path=self.path,
            start_line=1,
            end_line=self.facts.line_count,
            modifiers=[TAG_MODIFIER] if single else [],
            signature=f"template {template_id}",
            is_test=self.is_test,
        )

    def _tag_symbols(self) -> None:
        """Give every custom element of a multi-element file its own symbol.

        Hawkeye hoists tags globally, so one file may declare a whole component family. Each
        gets a symbol so a reference lands on the element holding the markup, not on the file.
        """
        if len(self.facts.tags) < 2:
            return
        for declaration in self.facts.tags:
            symbol = Symbol(
                id=make_symbol_id(self.path, declaration.tag),
                kind=SymbolKind.TEMPLATE,
                name=declaration.tag,
                qualified_name=declaration.tag,
                file_path=self.path,
                start_line=declaration.start_line,
                end_line=declaration.end_line,
                container_id=self.file_symbol.id,
                modifiers=[TAG_MODIFIER],
                signature=f"tag <{declaration.tag}> {declaration.class_name}",
                is_test=self.is_test,
            )
            self.result.symbols.append(symbol)
            self.owners.append((declaration.start_line, declaration.end_line, symbol.id))

    def _block_symbols(self) -> None:
        """Give every `{% block "name" %}` region a symbol."""
        for block in self.facts.blocks:
            qualified = f"{self.file_symbol.qualified_name}:{block.name}"
            self.result.symbols.append(
                Symbol(
                    id=make_symbol_id(self.path, qualified),
                    kind=SymbolKind.BLOCK,
                    name=block.name,
                    qualified_name=qualified,
                    file_path=self.path,
                    start_line=block.start_line,
                    end_line=block.end_line,
                    container_id=self.file_symbol.id,
                    signature=f'block "{block.name}"',
                    is_test=self.is_test,
                )
            )

    def _owner(self, line: int) -> str:
        """Return the symbol a reference written on a line belongs to."""
        for start, end, symbol_id in self.owners:
            if start <= line <= end:
                return symbol_id
        return self.file_symbol.id

    def _edges(self) -> None:
        """Record what the template extends, renders, uses and calls."""
        if self.facts.extends is not None:
            self._add(self.file_symbol.id, self.facts.extends.target, EdgeKind.EXTENDS, self.facts.extends.line)
        for reference in self.facts.renders:
            self._add(self.file_symbol.id, reference.target, EdgeKind.IMPORTS, reference.line)
        for reference in self.facts.used_tags:
            self._add(self._owner(reference.line), reference.target, EdgeKind.REFERENCES_TYPE, reference.line)
        for call in self.facts.calls:
            # Hawkeye function arities never line up with the Java method's, which may take a
            # leading RenderContext, so arity is deliberately left unset.
            self._add(self._owner(call.line), call.name, EdgeKind.CALLS, call.line, receiver=call.namespace)

    def _add(self, src_id: str, dst_name: str, kind: EdgeKind, line: int, receiver: str | None = None) -> None:
        """Append an unresolved edge."""
        self.result.edges.append(Edge(src_id=src_id, dst_name=dst_name, kind=kind, line=line, receiver=receiver))
