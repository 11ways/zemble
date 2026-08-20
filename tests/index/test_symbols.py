import random
from pathlib import Path

import pytest

from zemble.index.symbols import SymbolDefinitions, save_symbol_definitions
from zemble.ranking.boosting import _chunk_defines_symbol, apply_query_boost
from zemble.types import Chunk

_SOURCES = [
    "package be.elevenways.zemble;\n\nfinal class EventDelegationPlanner {\n    void plan() {}\n}\n",
    "class Foo:\n    def bar(self):\n        return 1\n",
    "defmodule Phoenix.Router do\n  def call(conn), do: conn\nend\n",
    "data class Point(val x: Int)\n\nabstract class Shape {}\n",
    "CREATE TABLE users (id int);\ncreate view Big_View as select 1;\n",
    "interface Session<T> extends Closeable {}\n",
    "// mentions EventDelegationPlanner without defining it\nvar planner = new EventDelegationPlanner();\n",
    "fn main() {}\nfun helper(): Unit {}\n",
    "type Alias = int;\ntypedef struct node_t node;\n",
    "namespace A\\B;\nfunction handle() {}\n",
    "class def Foo\n",
    "def type Foo\n",
    "record StateManager(int x) {}\n",
    "class a::b\nstruct a.b\n",
    "module Sinatra::Base\nend\n",
]

_NAMES = [
    "EventDelegationPlanner",
    "Foo",
    "Router",
    "Phoenix.Router",
    "Point",
    "Shape",
    "users",
    "Users",
    "Big_View",
    "Session",
    "main",
    "helper",
    "Alias",
    "node",
    "handle",
    "type",
    "def",
    "StateManager",
    "a",
    "b",
    "a::b",
    "a.b",
    "Base",
    "Sinatra::Base",
    "zemble",
    "be.elevenways.zemble",
]


def _corpus() -> list[Chunk]:
    """Build a fixture corpus that covers every definition keyword family."""
    return [
        Chunk(content=source, file_path=f"src/file{index}.java", start_line=1, end_line=source.count("\n") + 1)
        for index, source in enumerate(_SOURCES)
    ]


def test_lookup_answers_exactly_what_the_scan_answered(tmp_path: Path) -> None:
    """Every (chunk, name) pair gets the same verdict from the stored tables as from the regex scan."""
    chunks = _corpus()
    save_symbol_definitions(tmp_path, chunks)
    definitions = SymbolDefinitions.load(tmp_path)

    for name in _NAMES:
        scanned = {index for index, chunk in enumerate(chunks) if _chunk_defines_symbol(chunk, name)}
        looked_up = definitions.chunks_defining({name})
        assert looked_up is not None, f"{name} should be answerable from the tables"
        assert set(looked_up.tolist()) == scanned, f"the tables disagree with the scan about {name}"
        assert list(looked_up) == sorted(looked_up), "chunk indices come back in index order"


def test_query_boost_is_identical_with_and_without_the_lookup(tmp_path: Path) -> None:
    """The rerank pass produces the same scores whether it scans or looks names up."""
    chunks = _corpus()
    save_symbol_definitions(tmp_path, chunks)
    definitions = SymbolDefinitions.load(tmp_path)
    rng = random.Random(20260820)

    # 1. A symbol query, an embedded-symbol NL query, and a plain NL query.
    queries = ["EventDelegationPlanner", "Sinatra::Base", "how does the StateManager keep state", "where is the router"]
    for query in queries:
        candidates = {chunk: rng.random() for chunk in rng.sample(chunks, 4)}
        scanned = apply_query_boost(dict(candidates), query, chunks)
        looked_up = apply_query_boost(dict(candidates), query, chunks, definitions)
        assert looked_up == scanned, f"the boost changed for {query!r}"
        assert list(looked_up) == list(scanned), f"the boosted order changed for {query!r}"


def test_a_name_the_tables_cannot_hold_falls_back_to_scanning(tmp_path: Path) -> None:
    """A queried name that is not an identifier chain reports itself unanswerable."""
    save_symbol_definitions(tmp_path, _corpus())
    definitions = SymbolDefinitions.load(tmp_path)

    assert definitions.chunks_defining({"Foo->bar"}) is None
    assert definitions.chunks_defining({"Foo"}) is not None


def test_load_rejects_a_foreign_format(tmp_path: Path) -> None:
    """A symbol directory written by another format version is refused."""
    save_symbol_definitions(tmp_path, _corpus())
    (tmp_path / "symbols.json").write_text('{"format": 999, "n_chunks": 0}')

    with pytest.raises(ValueError, match="Unsupported symbol format"):
        SymbolDefinitions.load(tmp_path)
