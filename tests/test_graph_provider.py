"""Behaviour journeys over the graph provider's relationship queries."""

from pathlib import Path

from zemble.graph.model import EdgeKind, Resolution
from zemble.graph.provider import GraphProvider, SqliteGraphProvider


def _by_name(hits) -> set[str]:
    """Collect the qualified names of a hit list or a symbol list."""
    return {getattr(hit, "symbol", hit).qualified_name for hit in hits}


def test_provider_satisfies_its_protocol(built_graph: SqliteGraphProvider) -> None:
    """The sqlite implementation is a GraphProvider, so a javac-grade one can replace it."""
    assert isinstance(built_graph, GraphProvider), "SqliteGraphProvider implements the provider seam"


def test_definition_journey(built_graph: SqliteGraphProvider) -> None:
    """A written name is turned into declarations, most specific first."""
    # 1. A simple name finds every declaration carrying it.
    circles = built_graph.definition("Circle")
    assert "com.example.util.Circle" in _by_name(circles), "step 1: both Circles are found"

    # 2. A fully qualified name pins one down and sorts it first.
    exact = built_graph.definition("com.example.util.Circle")
    assert exact[0].qualified_name == "com.example.util.Circle", "step 2: the qualified match ranks first"

    # 3. `Type.member` finds the member, not the type.
    method = built_graph.definition("Circle.area")
    assert method[0].qualified_name == "com.example.core.Circle.area", "step 3: Type.member finds the method"

    # 4. Overloads are separate declarations with separate ids.
    scales = built_graph.definition("Circle.scale")
    assert len({symbol.id for symbol in scales}) == 2, "step 4: both overloads are returned"

    # 5. A name nobody declares returns nothing rather than guessing.
    assert built_graph.definition("NoSuchThing") == [], "step 5: unknown names return nothing"


def test_call_graph_journey(built_graph: SqliteGraphProvider) -> None:
    """Callers and callees are answered with a reason apiece."""
    twice = built_graph.definition("Helpers.twice")[0]

    # 1. Every call site of a static helper is found, across packages and source sets.
    callers = built_graph.callers(twice.id)
    assert "com.example.core.Circle.area" in _by_name(callers), "step 1: the main-source caller is found"
    assert "com.example.core.CircleTest.areaJourney" in _by_name(callers), "step 1: the test caller is found too"

    # 2. Each hit explains itself in one sentence.
    reason = next(hit.reason for hit in callers if hit.symbol.name == "area")
    assert reason.startswith("called from Circle.area (line "), f"step 2: unexpected reason {reason!r}"
    assert "exact match" in reason, "step 2: the reason states the resolution quality"

    # 3. Callees go the other way.
    area = built_graph.definition("Circle.area")[0]
    assert "com.example.util.Helpers.twice" in _by_name(built_graph.callees(area.id)), "step 3: callees invert callers"

    # 4. References cover every edge kind, not only calls.
    kinds = {hit.edge_kind for hit in built_graph.references(built_graph.definition("Helpers")[0].id)}
    assert EdgeKind.REFERENCES_TYPE in kinds, "step 4: type references are reported"
    assert EdgeKind.IMPORTS in kinds, "step 4: imports are reported"


def test_hierarchy_journey(built_graph: SqliteGraphProvider) -> None:
    """Subtypes and supertypes are walked transitively, carrying their depth."""
    shape = built_graph.definition("Shape")[0]

    # 1. Direct implementors are found.
    implementations = built_graph.implementations(shape.id)
    assert "com.example.core.Circle" in _by_name(implementations), "step 1: Circle implements Shape"

    # 2. An anonymous class counts as an implementation.
    assert any("$anon@" in name for name in _by_name(implementations)), "step 2: anonymous subtypes are included"

    # 3. Every hit knows how far it is from the queried type.
    assert {hit.depth for hit in implementations} == {1}, "step 3: all fixture subtypes are one hop away"

    # 4. Supertypes invert the walk.
    circle = built_graph.definition("com.example.core.Circle")[0]
    assert _by_name(built_graph.supertypes(circle.id)) == {"com.example.core.Shape"}, "step 4: Circle extends Shape"

    # 5. Overrides are answered from both ends.
    area = built_graph.definition("Circle.area")[0]
    assert _by_name(built_graph.overrides_of(area.id)) == {"com.example.core.Shape.area"}, (
        "step 5: Circle.area overrides Shape.area"
    )
    shape_area = built_graph.definition("Shape.area")[0]
    assert "com.example.core.Circle.area" in _by_name(built_graph.overridden_by(shape_area.id)), (
        "step 5: Shape.area is overridden by Circle.area"
    )

    # 6. An override hit is never claimed to be exact, because types were not compared.
    assert built_graph.overrides_of(area.id)[0].resolution is Resolution.UNIQUE_NAME, "step 6: overrides are by-name"


_GENERIC_HIERARCHY = {
    "src/main/java/com/example/reg/Snapshottable.java": (
        "package com.example.reg;\n\nimport java.util.List;\n\n"
        "public interface Snapshottable<E> {\n    List<E> snapshot();\n}\n"
    ),
    "src/main/java/com/example/reg/KeyedRegistry.java": (
        "package com.example.reg;\n\nimport java.util.List;\n\n"
        "public abstract class KeyedRegistry<V> implements Snapshottable<KeyedRegistry.Entry<V>> {\n"
        "    public static final class Entry<V> {\n    }\n\n"
        "    @Override\n    public List<Entry<V>> snapshot() {\n        return List.of();\n    }\n}\n"
    ),
    "src/main/java/com/example/reg/Themes.java": (
        "package com.example.reg;\n\npublic final class Themes extends KeyedRegistry<String> {\n}\n"
    ),
    "src/main/java/com/example/reg/Ordered.java": (
        "package com.example.reg;\n\n"
        "public interface Ordered<E> extends Snapshottable<E>, Comparable<Ordered<E>> {\n}\n"
    ),
    "src/main/java/com/example/reg/Tagged.java": (
        "package com.example.reg;\n\nimport java.lang.annotation.ElementType;\n"
        "import java.lang.annotation.Target;\n\n@Target(ElementType.TYPE_USE)\npublic @interface Tagged {\n}\n"
    ),
    "src/main/java/com/example/reg/Marked.java": (
        "package com.example.reg;\n\npublic abstract class Marked implements @Tagged Snapshottable<Integer> {\n}\n"
    ),
    "src/main/java/com/example/app/Feeds.java": (
        "package com.example.app;\n\nimport java.util.List;\n\n"
        "public final class Feeds implements com.example.reg.Snapshottable<String> {\n"
        "    @Override\n    public List<String> snapshot() {\n        return List.of();\n    }\n}\n"
    ),
}


def test_generic_supertypes_journey(graph_cache: Path, tmp_path: Path) -> None:
    """A supertype written with type arguments resolves like a plain one, through every hierarchy query."""
    from zemble.graph.store import build_graph

    workspace = tmp_path / "generic"
    for relative, text in _GENERIC_HIERARCHY.items():
        (workspace / relative).parent.mkdir(parents=True, exist_ok=True)
        (workspace / relative).write_text(text, encoding="utf-8")
    build_graph(str(workspace))
    provider = SqliteGraphProvider(str(workspace))
    try:
        snapshottable = provider.definition("com.example.reg.Snapshottable")[0]
        hits = provider.implementations(snapshottable.id)
        by_depth = {hit.symbol.qualified_name: hit.depth for hit in hits}

        # 1. `implements X<A>`, `extends X<A>` on an interface, a qualified `x.X<A>` and `@T X<A>` all land.
        direct = {"com.example.reg.KeyedRegistry", "com.example.reg.Ordered", "com.example.app.Feeds"}
        assert {name for name, depth in by_depth.items() if depth == 1} == direct | {"com.example.reg.Marked"}, (
            "step 1: every generic way of naming Snapshottable is one hop away"
        )

        # 2. A generic superclass carries the walk on: Themes reaches it through KeyedRegistry<String>.
        assert by_depth.get("com.example.reg.Themes") == 2, "step 2: Themes is found two hops out"
        themes = provider.definition("com.example.reg.Themes")[0]
        assert _by_name(provider.supertypes(themes.id)) == {
            "com.example.reg.KeyedRegistry",
            "com.example.reg.Snapshottable",
        }, "step 2: and the walk inverts"

        # 3. Each of those edges was resolved exactly, not guessed by name.
        assert {hit.resolution for hit in hits} == {Resolution.EXACT}, "step 3: every hop is exact"

        # 4. Overrides are derived through the generic supertype too.
        snapshot = provider.definition("Snapshottable.snapshot")[0]
        assert _by_name(provider.overridden_by(snapshot.id)) == {
            "com.example.reg.KeyedRegistry.snapshot",
            "com.example.app.Feeds.snapshot",
        }, "step 4: both implementations of snapshot() override it"

        # 5. The declaration still reads as written: only the edge is erased.
        registry = provider.definition("com.example.reg.KeyedRegistry")[0]
        assert registry.signature.endswith("implements Snapshottable<KeyedRegistry.Entry<V>>"), (
            "step 5: the signature keeps its type arguments"
        )
    finally:
        provider.close()


def test_tests_of_journey(built_graph: SqliteGraphProvider) -> None:
    """Naming matches come first, then tests that merely use the symbol."""
    circle = built_graph.definition("com.example.core.Circle")[0]

    # 1. The naming match is found.
    hits = built_graph.tests_of(circle.id)
    assert hits[0].symbol.qualified_name == "com.example.core.CircleTest", "step 1: CircleTest is the naming match"
    assert hits[0].edge_kind is EdgeKind.TESTS, "step 1: and it comes from the TESTS edge"

    # 2. A member is answered through its declaring type plus its own use.
    twice = built_graph.definition("Helpers.twice")[0]
    assert "com.example.core.CircleTest" in _by_name(built_graph.tests_of(twice.id)), "step 2: users are found too"

    # 3. Those hits are exercise edges, never naming ones.
    assert {hit.edge_kind for hit in built_graph.tests_of(twice.id)} == {EdgeKind.EXERCISES}, (
        "step 3: Helpers has no FooTest, only exercising tests"
    )

    # 4. A symbol no test touches answers empty rather than guessing.
    assert built_graph.tests_of(built_graph.definition("Marker")[0].id) == [], "step 4: untested symbols answer empty"


def test_neighbors_journey(built_graph: SqliteGraphProvider) -> None:
    """Neighbour walks go both directions and honour a kind filter."""
    circle = built_graph.definition("com.example.core.Circle")[0]

    # 1. One hop sees both what Circle points at and what points at Circle.
    one_hop = built_graph.neighbors(circle.id, hops=1)
    names = _by_name(one_hop)
    assert "com.example.core.Shape" in names, "step 1: outgoing edges are walked"
    assert "com.example.core.CircleTest" in names, "step 1: incoming edges are walked too"

    # 2. Two hops reach further than one.
    assert len(built_graph.neighbors(circle.id, hops=2)) > len(one_hop), "step 2: more hops reach more symbols"

    # 3. A kind filter restricts which edges are followed.
    only_tests = built_graph.neighbors(circle.id, hops=1, kinds=[EdgeKind.TESTS])
    assert _by_name(only_tests) == {"com.example.core.CircleTest"}, "step 3: the filter is honoured"

    # 4. The queried symbol never appears in its own neighbourhood.
    assert circle.id not in {hit.symbol.id for hit in built_graph.neighbors(circle.id, hops=3)}, (
        "step 4: a walk never revisits its origin"
    )


def test_coverage_note_names_missing_extractors(graph_cache: Path, tmp_path: Path) -> None:
    """An empty answer can say what the graph does not cover."""
    from zemble.graph.store import build_graph

    workspace = tmp_path / "mixed"
    workspace.mkdir()
    (workspace / "Main.java").write_text("package p;\npublic class Main {}\n", encoding="utf-8")
    (workspace / "app.cob").write_text("IDENTIFICATION DIVISION.\n", encoding="utf-8")
    build_graph(str(workspace))
    provider = SqliteGraphProvider(str(workspace))
    try:
        note = provider.coverage_note()
    finally:
        provider.close()
    assert "no grammar for: cobol (1)" in note, "the note names the gap"
    assert "languages" in note, "and says how wide the coverage is"


def test_capped_walk_streams_only_lookahead_hits_and_reports_a_lower_bound(tmp_path, graph_cache, monkeypatch):
    """A capped high-fanout graph query stops reading once its lookahead proves truncation."""
    from zemble.graph.mcp import answer
    from zemble.graph.store import build_graph

    source = tmp_path / "fanout.py"
    source.write_text(
        "def hub():\n    return 0\n" + "\n".join(f"def caller_{i}():\n    return hub()\n" for i in range(100))
    )
    build_graph(str(tmp_path), workers=1)
    pulled = 0
    real = SqliteGraphProvider._hits

    def counted(self, edges, **kwargs):
        nonlocal pulled
        assert len(edges) == 1, "a walk must stream, not materialize the whole adjacency list"
        pulled += 1
        return real(self, edges, **kwargs)

    monkeypatch.setattr(SqliteGraphProvider, "_hits", counted)
    result = answer(str(tmp_path), "hub", "neighbors", hops=3, limit=10)
    assert len(result["results"]) == 10
    assert result["total"] == 11 and result["total_exact"] is False
    assert "at least 11" in result["truncated"]
    assert pulled == 11, "stop at one lookahead, not after walking a hundred callers"


def test_hierarchy_cap_and_subtree_filter_are_applied_during_the_walk(tmp_path, graph_cache):
    """Capped subtree walks keep crossing siblings but stop after enough in-folder hits."""
    from zemble.graph.provider import SubtreeGraphProvider
    from zemble.graph.store import build_graph

    (tmp_path / "keep").mkdir()
    (tmp_path / "sibling").mkdir()
    (tmp_path / "keep" / "Base.java").write_text("class Base {}")
    (tmp_path / "sibling" / "Middle.java").write_text("class Middle extends Base {}")
    (tmp_path / "keep" / "Children.java").write_text(
        "\n".join(f"class Child{i} extends Middle {{}}" for i in range(100))
    )
    build_graph(str(tmp_path), workers=1)
    graph = SqliteGraphProvider(str(tmp_path))
    view = SubtreeGraphProvider(graph, "keep", str(tmp_path / "keep"))
    try:
        base = view.definition("Base")[0]
        hits = view.implementations(base.id, limit=3)
        assert len(hits) == 3 and all(hit.depth == 2 for hit in hits)
        assert all(hit.symbol.file_path == "Children.java" for hit in hits)
        assert view.traversal_truncated
        # An exhausted small traversal retains an exact count.
        assert len(view.supertypes(view.definition("Child0")[0].id, limit=10)) == 1
        assert not view.traversal_truncated
    finally:
        view.close()


def test_subtree_walk_bounds_visited_siblings_even_when_no_result_passes(tmp_path, graph_cache, monkeypatch):
    """Filtering cannot force a capped walk to retain every node of a large sibling module."""
    from zemble.graph.provider import SubtreeGraphProvider
    from zemble.graph.store import build_graph

    (tmp_path / "keep").mkdir()
    (tmp_path / "sibling").mkdir()
    (tmp_path / "keep" / "Hub.java").write_text("class Hub { static int hub() { return 0; } }")
    for batch in range(43):
        (tmp_path / "sibling" / f"Calls{batch}.java").write_text(
            "\n".join(f"class Caller{batch}_{i} {{ int call() {{ return Hub.hub(); }} }}" for i in range(100))
        )
    build_graph(str(tmp_path), workers=1)
    graph = SqliteGraphProvider(str(tmp_path))
    view = SubtreeGraphProvider(graph, "keep", str(tmp_path / "keep"))
    pulls = 0
    real = graph._hits

    def counted(edges, **kwargs):
        nonlocal pulls
        pulls += len(edges)
        return real(edges, **kwargs)

    monkeypatch.setattr(graph, "_hits", counted)
    try:
        hits = view.neighbors(view.definition("hub")[0].id, hops=3, limit=10)
        assert hits == [] and view.traversal_truncated
        assert pulls < 4300, "a filtered-out fanout still stops at the visited-node budget"
    finally:
        view.close()
