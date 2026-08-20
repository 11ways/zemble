"""Behaviour journeys over workspace-wide resolution."""

import json
from pathlib import Path

from zemble.graph.model import EdgeKind, Resolution
from zemble.graph.resolve import _subject_name
from zemble.graph.store import connect


def _edges(root: Path, **where):
    """Load edges from the built graph, filtered on exact column values."""
    connection = connect(str(root))
    clause = " AND ".join(f"{column} = ?" for column in where)
    query = f"SELECT * FROM edges WHERE {clause}" if where else "SELECT * FROM edges"
    rows = [dict(row) for row in connection.execute(query, tuple(where.values()))]
    connection.close()
    return rows


def _one(rows, dst_name: str):
    """Return the single edge with the given written destination name."""
    matching = [row for row in rows if row["dst_name"] == dst_name]
    assert len(matching) == 1, f"expected exactly one edge to {dst_name}, got {len(matching)}"
    return matching[0]


def test_type_resolution_ladder_journey(graph_fixture_root: Path, built_graph) -> None:
    """Walk a type name down every rung of the resolution ladder."""
    root = graph_fixture_root

    # 1. Same package: Shape.unit() writes `Circle` and means the one next to it.
    same_package = [
        row
        for row in _edges(root, kind=EdgeKind.REFERENCES_TYPE.value)
        if row["src_id"].endswith("Shape.unit()") and row["dst_name"] == "Circle"
    ]
    assert same_package[0]["resolution"] == Resolution.EXACT.value, "step 1: same-package types resolve exactly"
    assert same_package[0]["dst_id"].endswith("com.example.core.Circle"), "step 1: to the same-package Circle"

    # 2. Explicit import: Circle.area() writes `Helpers` and means com.example.util.Helpers.
    imported = [
        row
        for row in _edges(root, kind=EdgeKind.REFERENCES_TYPE.value)
        if row["src_id"].endswith("Circle.area()") and row["dst_name"] == "Helpers"
    ]
    assert imported[0]["dst_id"].endswith("com.example.util.Helpers"), "step 2: explicit imports win"

    # 3. Wildcard import: Registry.viaWildcard() reaches Helpers through `com.example.util.*`.
    wildcard = [
        row
        for row in _edges(root, kind=EdgeKind.REFERENCES_TYPE.value)
        if row["src_id"].endswith("Registry.viaWildcard()") and row["dst_name"] == "Helpers"
    ]
    assert wildcard[0]["resolution"] == Resolution.EXACT.value, "step 3: wildcard imports resolve exactly"

    # 4. Two packages declare `Circle`, and a file that imports neither gets AMBIGUOUS with both.
    ambiguous = [
        row
        for row in _edges(root, kind=EdgeKind.REFERENCES_TYPE.value)
        if row["src_id"].endswith("Consumer.measure(Circle)") and row["dst_name"] == "Circle"
    ]
    assert ambiguous[0]["resolution"] == Resolution.AMBIGUOUS.value, "step 4: two same-named types are ambiguous"
    assert ambiguous[0]["dst_id"] is None, "step 4: an ambiguous edge names no single destination"
    assert len(json.loads(ambiguous[0]["candidates"])) == 2, "step 4: both candidates are listed"

    # 5. A JDK type nobody declares stays honestly unresolved.
    jdk = [row for row in _edges(root, kind=EdgeKind.REFERENCES_TYPE.value) if row["dst_name"] == "String"]
    assert {row["resolution"] for row in jdk} == {Resolution.UNRESOLVED.value}, "step 5: JDK types stay unresolved"


def test_call_resolution_journey(graph_fixture_root: Path, built_graph) -> None:
    """Walk a call through the receiver, the supertype chain, static imports and the workspace."""
    root = graph_fixture_root
    calls = _edges(root, kind=EdgeKind.CALLS.value)

    # 1. A static call on a type name lands on that type's method.
    static_call = [row for row in calls if row["src_id"].endswith("Circle.area()") and row["dst_name"] == "twice"]
    assert static_call[0]["dst_id"].endswith("Helpers.twice(double)"), "step 1: Helpers.twice resolves exactly"

    # 2. An implicit-this call walks up the supertype chain into the interface.
    inherited = [row for row in calls if row["src_id"].endswith("Circle.label()")]
    assert inherited[0]["dst_id"].endswith("Shape.describe()"), "step 2: the chain reaches an inherited default method"

    # 3. A statically imported call with no receiver finds the imported owner.
    static_import = [row for row in calls if "$anon@" in row["src_id"] and row["dst_name"] == "twice"]
    assert static_import[0]["dst_id"].endswith("Helpers.twice(double)"), "step 3: static imports resolve calls"

    # 4. this(...) picks the constructor by arity.
    chained = _one([row for row in calls if row["src_id"].endswith("Circle.Circle()")], "this")
    assert chained["dst_id"].endswith("Circle.Circle(double)"), "step 4: this(1.0) picks the 1-arg constructor"

    # 5. A local variable's declared type steers the call to that type's method.
    local = [row for row in calls if row["src_id"].endswith("Registry.viaWildcard()") and row["dst_name"] != "Helpers"]
    assert local[0]["dst_id"].endswith("Helpers.instanceTwice(double)"), "step 5: declared local types steer calls"

    # 6. An ambiguous receiver narrows to the same-named members, never to one of them.
    unknown = [row for row in calls if row["src_id"].endswith("Consumer.measure(Circle)")]
    assert unknown[0]["resolution"] == Resolution.AMBIGUOUS.value, "step 6: an ambiguous receiver stays ambiguous"
    assert len(json.loads(unknown[0]["candidates"])) == 2, "step 6: both area() candidates are listed"


def test_override_journey(graph_fixture_root: Path, built_graph) -> None:
    """@Override chains are found by name and arity, and only by name and arity."""
    root = graph_fixture_root
    overrides = {row["src_id"]: row for row in _edges(root, kind=EdgeKind.OVERRIDES.value)}

    # 1. A class method overrides its interface's abstract method.
    circle = next(row for src, row in overrides.items() if src.endswith("Circle.area()"))
    assert circle["dst_id"].endswith("Shape.area()"), "step 1: Circle.area overrides Shape.area"

    # 2. The match is graded by-name because parameter types were never compared.
    assert circle["resolution"] == Resolution.UNIQUE_NAME.value, "step 2: overrides are never claimed to be exact"

    # 3. An anonymous class overrides through its declared supertype too.
    anonymous = [row for src, row in overrides.items() if "$anon@" in src]
    assert anonymous[0]["dst_id"].endswith("Shape.area()"), "step 3: anonymous classes override their supertype"

    # 4. A method inside an enum constant's body overrides the enum's own method.
    constant = next(row for src, row in overrides.items() if "Palette.GREEN.tag" in src)
    assert constant["dst_id"].endswith("Palette.tag()"), "step 4: enum constant bodies override the enum"


def test_test_edge_journey(graph_fixture_root: Path, built_graph) -> None:
    """Test-to-subject edges come from naming, and exercise edges from actual use."""
    root = graph_fixture_root

    # 1. FooTest next to Foo produces a TESTS edge.
    tests = _edges(root, kind=EdgeKind.TESTS.value)
    assert len(tests) == 1, "step 1: exactly one naming-based test edge in the fixture"
    assert tests[0]["dst_id"].endswith("com.example.core.Circle"), "step 1: CircleTest tests Circle"

    # 2. A test that touches two main types exercises both.
    exercises = _edges(root, kind=EdgeKind.EXERCISES.value)
    exercised = {row["dst_id"].split("#")[1] for row in exercises if "CircleTest" in row["src_id"]}
    assert "com.example.core.Circle" in exercised, "step 2: the test exercises Circle"
    assert "com.example.util.Helpers" in exercised, "step 2: the test exercises Helpers too"

    # 3. A browserTest source set counts as a test source set.
    browser = {row["src_id"] for row in exercises if "PaletteBrowserCheck" in row["src_id"]}
    assert browser, "step 3: browserTest files produce exercise edges"

    # 4. Main-source files never exercise anything.
    assert not [row for row in exercises if "/main/" in row["src_id"]], "step 4: only test files exercise"


def test_subject_name_conventions() -> None:
    """The four supported test naming conventions map back to their subject."""
    assert _subject_name("FooTest") == "Foo", "FooTest -> Foo"
    assert _subject_name("FooTests") == "Foo", "FooTests -> Foo"
    assert _subject_name("TestFoo") == "Foo", "TestFoo -> Foo"
    assert _subject_name("FooIT") == "Foo", "FooIT -> Foo"
    assert _subject_name("Foo") is None, "a plain name implies no subject"


def test_binding_form_journey(graph_fixture_root: Path, built_graph) -> None:
    """Every Java form that declares a variable steers the calls made on it."""
    root = graph_fixture_root
    calls = [row for row in _edges(root, kind=EdgeKind.CALLS.value) if "Traversal" in row["src_id"]]

    def landed(source: str, name: str) -> str | None:
        """The destination a named call in one method landed on."""
        matching = [row for row in calls if row["src_id"].endswith(source) and row["dst_name"] == name]
        assert len(matching) == 1, f"expected one {name} call in {source}"
        return matching[0]["dst_id"]

    # 1. A for-each binding carries its element type.
    assert landed("Traversal.sumAreas(Shape[])", "area").endswith("Shape.area()"), "step 1: for-each bindings resolve"

    # 2. An instanceof pattern binds the narrowed type.
    assert landed("Traversal.narrow(Object)", "label").endswith("Circle.label()"), "step 2: instanceof patterns resolve"

    # 3. A cast supplies the declared type of the local it initialises.
    assert landed("Traversal.narrow(Object)", "describe").endswith("Shape.describe()"), "step 3: casts resolve"

    # 4. A try-with-resources binding is a declaration like any other.
    assert landed("Traversal.withResource()", "name").endswith("Session.name()"), "step 4: resources resolve"

    # 5. A method reference is a call with unknown arity.
    reference = [
        row for row in calls if row["src_id"].endswith("Traversal.viaReference()") and row["dst_name"] == "twice"
    ]
    assert reference[0]["arity"] == -1, "step 5: a method reference has no argument list"
    assert reference[0]["dst_id"].endswith("Helpers.twice(double)"), "step 5: it still lands on the method"

    # 6. A static wildcard import resolves a bare call.
    assert landed("Traversal.staticWildcard()", "twice").endswith("Helpers.twice(double)"), "step 6: static wildcards"

    # 7. A catch binding whose type is outside the workspace leaves the call unresolved, not wrong.
    assert landed("Traversal.guarded()", "getMessage") is None, "step 7: JDK receivers stay unresolved"
