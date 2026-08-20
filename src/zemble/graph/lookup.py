"""How the resolver reaches the workspace's declarations.

Resolution needs a handful of lookups over every symbol in the workspace: by id, by
qualified name, by simple name, by container, by file, plus the Hawkeye registration keys
and the resolved supertype map. Materialising all of them costs a full pass over the symbol
table, which is fine for a cold build and absurd for a one-file refresh, so they live behind
a seam with two implementations: :class:`MemoryLookup` builds the dictionaries once from a
symbol list, and :class:`SqliteLookup` answers each question with an indexed query and
caches what it touched.

Both must answer identically; `tests/test_graph_incremental.py` builds the same tree through
each and compares the resulting tables.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import orjson

from zemble.graph.hwk import TAG_MODIFIER
from zemble.graph.java import FileImports
from zemble.graph.model import CALLABLE_KINDS, TYPE_KINDS, Symbol, SymbolKind
from zemble.hwk import (
    ELEMENT_ANNOTATION,
    ELEMENT_TAG_ARGUMENT,
    FUNCTION_ANNOTATION,
    FUNCTION_NAME_ARGUMENT,
    FUNCTION_NAMESPACE_ARGUMENT,
    template_id_path,
)

#: Edge kinds that make up the type hierarchy, as stored in the `kind` column.
HIERARCHY_KINDS = ("extends", "implements")
#: Separates the namespace from the name inside a `FUNCTION_KEY` declaration key. A control
#: character, because neither half may contain one and both may contain a dot.
FUNCTION_KEY_SEPARATOR = "\x1f"


class DeclarationKey(str, Enum):
    """The registration keys a Hawkeye template resolves against.

    This is the vocabulary of the `decl_keys` table: a symbol contributes a row per key it
    carries, and every lookup of one goes through :meth:`SymbolLookup.declarations`. Adding a
    key means adding a member here and a branch in :func:`declaration_keys`, and nowhere else.
    """

    TEMPLATE_TAG = "template_tag"
    TEMPLATE_ID = "template_id"
    ELEMENT_TAG = "element_tag"
    FUNCTION_NAME = "function_name"
    FUNCTION_KEY = "function_key"


def declaration_keys(symbol: Symbol) -> Iterator[tuple[DeclarationKey, str]]:
    """Yield every Hawkeye registration key a symbol declares."""
    if symbol.kind is SymbolKind.TEMPLATE:
        if TAG_MODIFIER in symbol.modifiers:
            yield DeclarationKey.TEMPLATE_TAG, symbol.qualified_name
        if symbol.container_id is None:
            yield DeclarationKey.TEMPLATE_ID, template_id_path(symbol.file_path)
        return
    if symbol.kind is SymbolKind.METHOD and FUNCTION_ANNOTATION in symbol.annotations:
        arguments = symbol.annotation_args.get(FUNCTION_ANNOTATION, {})
        # `name` defaults to the Java method name, `namespace` to the global one.
        name = arguments.get(FUNCTION_NAME_ARGUMENT) or symbol.name
        namespace = arguments.get(FUNCTION_NAMESPACE_ARGUMENT, "")
        yield DeclarationKey.FUNCTION_NAME, name
        yield DeclarationKey.FUNCTION_KEY, f"{namespace}{FUNCTION_KEY_SEPARATOR}{name}"
        return
    if symbol.kind in TYPE_KINDS:
        tag = symbol.annotation_args.get(ELEMENT_ANNOTATION, {}).get(ELEMENT_TAG_ARGUMENT)
        if tag:
            yield DeclarationKey.ELEMENT_TAG, tag


def function_key(namespace: str, name: str) -> str:
    """Return the `FUNCTION_KEY` a namespace and name are registered under."""
    return f"{namespace}{FUNCTION_KEY_SEPARATOR}{name}"


@dataclass
class FileContext:
    """The package and imports a file's names are resolved against."""

    file_path: str
    package: str = ""
    imports: FileImports = field(default_factory=FileImports)


def symbol_from_row(row: sqlite3.Row) -> Symbol:
    """Rebuild a symbol from a database row.

    Four JSON columns per symbol, a hundred thousand symbols: this is the hot loop of every
    build that has to read the whole table, so it decodes with orjson.
    """
    return Symbol(
        id=row["id"],
        kind=SymbolKind(row["kind"]),
        name=row["name"],
        qualified_name=row["qualified_name"],
        file_path=row["file_path"],
        start_line=row["start_line"],
        end_line=row["end_line"],
        container_id=row["container_id"],
        modifiers=orjson.loads(row["modifiers"]),
        annotations=orjson.loads(row["annotations"]),
        signature=row["signature"],
        is_test=bool(row["is_test"]),
        param_types=orjson.loads(row["param_types"]),
        annotation_args=orjson.loads(row["annotation_args"] or "{}"),
    )


def context_from_row(row: sqlite3.Row) -> FileContext:
    """Rebuild a file's resolution context from its `files` row."""
    raw = json.loads(row["imports"]) if row["imports"] else {}
    return FileContext(
        file_path=row["path"],
        package=row["package"] or "",
        imports=FileImports(
            explicit=raw.get("explicit", {}),
            wildcards=raw.get("wildcards", []),
            static_members=raw.get("static_members", {}),
            static_wildcards=raw.get("static_wildcards", []),
        ),
    )


class SymbolLookup(Protocol):
    """Every question resolution asks about declarations it did not extract itself."""

    def by_id(self, symbol_id: str) -> Symbol | None:
        """Return the symbol with this id, or None."""
        ...

    def by_qualified(self, qualified_name: str) -> list[Symbol]:
        """Return every symbol carrying this exact qualified name."""
        ...

    def types_by_simple(self, name: str) -> list[Symbol]:
        """Return every type declaration with this simple name."""
        ...

    def callables_by_simple(self, name: str) -> list[Symbol]:
        """Return every method or constructor with this simple name."""
        ...

    def members(self, container_id: str) -> list[Symbol]:
        """Return the symbols declared directly inside a container."""
        ...

    def types_in_file(self, file_path: str) -> list[Symbol]:
        """Return the type declarations of one file."""
        ...

    def declarations(self, key: DeclarationKey, value: str) -> list[Symbol]:
        """Return the symbols registered under one Hawkeye declaration key."""
        ...

    def supertype_ids(self, type_id: str) -> list[str]:
        """Return the already-resolved supertype ids of a type."""
        ...

    def context(self, file_path: str) -> FileContext:
        """Return a file's package and imports."""
        ...


class MemoryLookup:
    """Answers every lookup from dictionaries built in one pass over the symbol list."""

    def __init__(
        self,
        symbols: list[Symbol],
        contexts: dict[str, FileContext],
        hierarchy: dict[str, list[str]] | None = None,
    ) -> None:
        """Index a whole workspace's symbols, contexts and already-resolved supertypes."""
        self._symbols = symbols
        self._by_id: dict[str, Symbol] = {}
        self._by_qualified: dict[str, list[Symbol]] = defaultdict(list)
        self._types_by_simple: dict[str, list[Symbol]] = defaultdict(list)
        self._callables_by_simple: dict[str, list[Symbol]] = defaultdict(list)
        self._members: dict[str, list[Symbol]] = defaultdict(list)
        self._types_in_file: dict[str, list[Symbol]] = defaultdict(list)
        self._declarations: dict[tuple[DeclarationKey, str], list[Symbol]] = defaultdict(list)
        self._contexts = contexts
        self._hierarchy = hierarchy or {}
        for symbol in symbols:
            self._by_id[symbol.id] = symbol
            self._by_qualified[symbol.qualified_name].append(symbol)
            if symbol.kind in TYPE_KINDS:
                self._types_by_simple[symbol.name].append(symbol)
                self._types_in_file[symbol.file_path].append(symbol)
            elif symbol.kind in CALLABLE_KINDS:
                self._callables_by_simple[symbol.name].append(symbol)
            if symbol.container_id is not None:
                self._members[symbol.container_id].append(symbol)
            for key, value in declaration_keys(symbol):
                self._declarations[(key, value)].append(symbol)

    def all_symbols(self) -> list[Symbol]:
        """Return every symbol this lookup was built from, in the order it was given them."""
        return self._symbols

    def by_id(self, symbol_id: str) -> Symbol | None:
        """Return the symbol with this id, or None."""
        return self._by_id.get(symbol_id)

    def by_qualified(self, qualified_name: str) -> list[Symbol]:
        """Return every symbol carrying this exact qualified name."""
        return self._by_qualified.get(qualified_name, [])

    def types_by_simple(self, name: str) -> list[Symbol]:
        """Return every type declaration with this simple name."""
        return self._types_by_simple.get(name, [])

    def callables_by_simple(self, name: str) -> list[Symbol]:
        """Return every method or constructor with this simple name."""
        return self._callables_by_simple.get(name, [])

    def members(self, container_id: str) -> list[Symbol]:
        """Return the symbols declared directly inside a container."""
        return self._members.get(container_id, [])

    def types_in_file(self, file_path: str) -> list[Symbol]:
        """Return the type declarations of one file."""
        return self._types_in_file.get(file_path, [])

    def declarations(self, key: DeclarationKey, value: str) -> list[Symbol]:
        """Return the symbols registered under one Hawkeye declaration key."""
        return self._declarations.get((key, value), [])

    def supertype_ids(self, type_id: str) -> list[str]:
        """Return the already-resolved supertype ids of a type."""
        return self._hierarchy.get(type_id, [])

    def context(self, file_path: str) -> FileContext:
        """Return a file's package and imports."""
        return self._contexts.get(file_path) or FileContext(file_path)


class SqliteLookup:
    """Answers every lookup with an indexed query, caching what one build touched.

    The database is the truth here, which is only sound because the build inserts a changed
    file's symbols and deletes a re-resolved file's edges BEFORE resolution runs: what the
    tables hold at that point is exactly the workspace minus the edges about to be rewritten.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        """Prepare the caches; nothing is read until something is asked."""
        self._connection = connection
        self._by_id: dict[str, Symbol | None] = {}
        self._by_qualified: dict[str, list[Symbol]] = {}
        self._by_name: dict[str, list[Symbol]] = {}
        self._members: dict[str, list[Symbol]] = {}
        self._types_in_file: dict[str, list[Symbol]] = {}
        self._declarations: dict[tuple[DeclarationKey, str], list[Symbol]] = {}
        self._supertypes: dict[str, list[str]] = {}
        self._contexts: dict[str, FileContext] = {}

    def _select(self, clause: str, *parameters: object) -> list[Symbol]:
        """Run a symbol query and rebuild every row it returned."""
        query = f"SELECT * FROM symbols WHERE {clause}"  # noqa: S608 - clause is a literal
        return [symbol_from_row(row) for row in self._connection.execute(query, parameters)]

    def by_id(self, symbol_id: str) -> Symbol | None:
        """Return the symbol with this id, or None."""
        if symbol_id not in self._by_id:
            found = self._select("id = ?", symbol_id)
            self._by_id[symbol_id] = found[0] if found else None
        return self._by_id[symbol_id]

    def by_qualified(self, qualified_name: str) -> list[Symbol]:
        """Return every symbol carrying this exact qualified name."""
        found = self._by_qualified.get(qualified_name)
        if found is None:
            found = self._select("qualified_name = ?", qualified_name)
            self._by_qualified[qualified_name] = found
        return found

    def _named(self, name: str) -> list[Symbol]:
        """Return every symbol with this simple name, cached across both kind filters."""
        found = self._by_name.get(name)
        if found is None:
            found = self._select("name = ?", name)
            self._by_name[name] = found
        return found

    def types_by_simple(self, name: str) -> list[Symbol]:
        """Return every type declaration with this simple name."""
        return [symbol for symbol in self._named(name) if symbol.kind in TYPE_KINDS]

    def callables_by_simple(self, name: str) -> list[Symbol]:
        """Return every method or constructor with this simple name."""
        # The memory index files a symbol under one of the two, never both, so a type that is
        # also a callable kind cannot happen and the filters stay each other's complement.
        return [symbol for symbol in self._named(name) if symbol.kind in CALLABLE_KINDS]

    def members(self, container_id: str) -> list[Symbol]:
        """Return the symbols declared directly inside a container."""
        found = self._members.get(container_id)
        if found is None:
            found = self._select("container_id = ?", container_id)
            self._members[container_id] = found
        return found

    def types_in_file(self, file_path: str) -> list[Symbol]:
        """Return the type declarations of one file."""
        found = self._types_in_file.get(file_path)
        if found is None:
            found = [symbol for symbol in self._select("file_path = ?", file_path) if symbol.kind in TYPE_KINDS]
            self._types_in_file[file_path] = found
        return found

    def declarations(self, key: DeclarationKey, value: str) -> list[Symbol]:
        """Return the symbols registered under one Hawkeye declaration key."""
        found = self._declarations.get((key, value))
        if found is None:
            rows = self._connection.execute(
                "SELECT symbols.* FROM decl_keys JOIN symbols ON symbols.id = decl_keys.symbol_id "
                "WHERE decl_keys.key = ? AND decl_keys.value = ?",
                (key.value, value),
            )
            found = [symbol_from_row(row) for row in rows]
            self._declarations[(key, value)] = found
        return found

    def supertype_ids(self, type_id: str) -> list[str]:
        """Return the already-resolved supertype ids of a type."""
        found = self._supertypes.get(type_id)
        if found is None:
            rows = self._connection.execute(
                "SELECT DISTINCT dst_id FROM edges WHERE src_id = ? AND kind IN (?, ?) AND dst_id IS NOT NULL",
                (type_id, *HIERARCHY_KINDS),
            )
            found = [row["dst_id"] for row in rows]
            self._supertypes[type_id] = found
        return found

    def context(self, file_path: str) -> FileContext:
        """Return a file's package and imports."""
        found = self._contexts.get(file_path)
        if found is None:
            row = self._connection.execute("SELECT * FROM files WHERE path = ?", (file_path,)).fetchone()
            found = context_from_row(row) if row is not None else FileContext(file_path)
            self._contexts[file_path] = found
        return found
