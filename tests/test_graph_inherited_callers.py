"""Callers of a method answer the call sites bound to the declarations it overrides and to its overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from zemble.graph.provider import SqliteGraphProvider, SubtreeGraphProvider
from zemble.graph.store import build_graph

_PIPELINE = "app/src/main/java/app/pipeline"
#: The quirkyquarters shape: an interface, its implementation, a deeper override, a sibling and a second branch.
JAVA = {
    f"{_PIPELINE}/GenerationHandle.java": (
        "package app.pipeline;\n\npublic interface GenerationHandle {\n    Object result();\n}\n"
    ),
    f"{_PIPELINE}/Resulting.java": "package app.pipeline;\n\npublic interface Resulting {\n    Object result();\n}\n",
    f"{_PIPELINE}/DefaultGenerationHandle.java": (
        "package app.pipeline;\n\n"
        "public class DefaultGenerationHandle implements GenerationHandle, Resulting {\n"
        "    @Override\n    public Object result() { return null; }\n"
        "    public static DefaultGenerationHandle start() { return new DefaultGenerationHandle(); }\n}\n"
    ),
    f"{_PIPELINE}/TracingHandle.java": (
        "package app.pipeline;\n\npublic class TracingHandle extends DefaultGenerationHandle {\n"
        "    @Override\n    public Object result() { return null; }\n}\n"
    ),
    f"{_PIPELINE}/OtherHandle.java": (
        "package app.pipeline;\n\npublic class OtherHandle implements GenerationHandle {\n"
        "    @Override\n    public Object result() { return null; }\n}\n"
    ),
    "app/src/main/java/app/Director.java": (
        "package app;\n\n"
        "import app.pipeline.DefaultGenerationHandle;\n"
        "import app.pipeline.GenerationHandle;\n"
        "import app.pipeline.OtherHandle;\n"
        "import app.pipeline.Resulting;\n"
        "import app.pipeline.TracingHandle;\n\n"
        "public class Director {\n"
        "    Object viaInterface(GenerationHandle handle) { return handle.result(); }\n"
        "    Object viaDefault(DefaultGenerationHandle handle) { return handle.result(); }\n"
        "    Object viaTracing(TracingHandle handle) { return handle.result(); }\n"
        "    Object viaOther(OtherHandle handle) { return handle.result(); }\n"
        "    Object viaResulting(Resulting handle) { return handle.result(); }\n"
        "    Object started() { return DefaultGenerationHandle.start(); }\n"
        "}\n"
    ),
}
#: A Python hierarchy: the generic lane derives its overrides from `class Circle(Shape)`.
PYTHON = {
    "shapes.py": (
        "class Shape:\n"
        "    def area(self):\n        return 0\n\n"
        "    def twice(self):\n        return self.area() * 2\n\n\n"
        "class Circle(Shape):\n"
        "    def area(self):\n        return 3\n\n"
        "    def describe(self):\n        return self.area()\n"
    ),
}


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text, encoding="utf-8")


@pytest.fixture
def java_graph(tmp_path: Path, graph_cache: Path) -> SqliteGraphProvider:
    """A graph over the Java workspace, built from scratch."""
    root = tmp_path / "ws"
    _write(root, JAVA)
    build_graph(str(root))
    provider = SqliteGraphProvider(str(root))
    yield provider
    provider.close()


def _one(provider, name: str):
    (symbol,) = provider.definition(name)
    return symbol


def _sites(hits) -> list[tuple[str, str, str | None]]:
    """Each hit as (caller, binding, through), in answer order."""
    return [
        (hit.symbol.name, hit.binding.value, hit.through.qualified_name if hit.through is not None else None)
        for hit in hits
    ]


def test_callers_of_an_implementation_answer_the_calls_through_its_interfaces(java_graph) -> None:
    """The implementation's callers are its own sites first, then every site bound to a related declaration."""
    default = _one(java_graph, "DefaultGenerationHandle.result")

    # 1. Its own call site, then the sites bound to both interfaces it implements, then its override's.
    assert _sites(java_graph.callers(default.id)) == [
        ("viaDefault", "direct", None),
        ("viaInterface", "overridden", "app.pipeline.GenerationHandle.result"),
        ("viaTracing", "overrider", "app.pipeline.TracingHandle.result"),
        ("viaResulting", "overridden", "app.pipeline.Resulting.result"),
    ], "step 1: direct first, then every inherited site, a sibling implementation's never"

    # 2. Each inherited site says which declaration it was bound to.
    reasons = {hit.symbol.name: hit.reason for hit in java_graph.callers(default.id)}
    assert reasons["viaInterface"].endswith(", through GenerationHandle.result)"), reasons["viaInterface"]
    assert reasons["viaTracing"].endswith(", through override TracingHandle.result)"), reasons["viaTracing"]
    assert "through" not in reasons["viaDefault"], "step 2: a direct site names no detour"

    # 3. The relation is the stored one: both supertype branches are overridden, one edge each.
    assert {hit.symbol.qualified_name for hit in java_graph.overrides_of(default.id)} == {
        "app.pipeline.GenerationHandle.result",
        "app.pipeline.Resulting.result",
    }, "step 3: a class implementing two interfaces overrides a declaration on each branch"


def test_callers_of_a_declaration_answer_the_calls_bound_to_its_overrides(java_graph) -> None:
    """The interface's callers include the sites bound to every override, at any depth, and nothing from a sibling."""
    interface = _one(java_graph, "GenerationHandle.result")
    sites = _sites(java_graph.callers(interface.id))

    # 1. Its own site, then each override's sites, the depth-two TracingHandle included.
    assert sites[0] == ("viaInterface", "direct", None), "step 1: the direct site leads"
    assert sorted(sites[1:]) == [
        ("viaDefault", "overrider", "app.pipeline.DefaultGenerationHandle.result"),
        ("viaOther", "overrider", "app.pipeline.OtherHandle.result"),
        ("viaTracing", "overrider", "app.pipeline.TracingHandle.result"),
    ], "step 1: every override's sites"

    # 2. A sibling declaration on another branch is no override of it.
    assert "viaResulting" not in {name for name, *_ in sites}, "step 2: Resulting.result is not related"

    # 3. A static method has no override relation: its callers are its own.
    start = _one(java_graph, "DefaultGenerationHandle.start")
    assert _sites(java_graph.callers(start.id)) == [("started", "direct", None)], "step 3: nothing inherited"


def test_every_surface_states_both_counts(java_graph, monkeypatch, capsys) -> None:
    """The MCP payload, its text and nib parts, the CLI and signatures count direct and inherited sites apart."""
    from tests.test_graph_cli import _run
    from zemble.evidence.answers import signatures_payload
    from zemble.graph.mcp import answer, callers_parts, callers_text

    root = java_graph.path

    # 1. The payload: `total` counts every listed site, and says how many were direct.
    payload = answer(root, "DefaultGenerationHandle.result", "callers")
    assert (payload["total"], payload["direct_total"], payload["inherited_total"]) == (4, 1, 3), "step 1: counts"
    assert payload["inherited_through"] == {
        "GenerationHandle.result": 1,
        "TracingHandle.result": 1,
        "Resulting.result": 1,
    }, "step 1: per declaration"
    via = {entry["display"]: (entry["binding"], entry["through"]) for entry in payload["results"]}
    assert via["Director.viaInterface"] == ("overridden", "GenerationHandle.result"), "step 1: marked per site"

    # 2. The text names both numbers and marks each inherited site.
    text = callers_text(payload)
    assert "of all 4: 1 direct, 1 through GenerationHandle.result" in text, text
    assert "Director viaInterface" not in text and "Director.viaInterface L" in text, text
    assert "(through GenerationHandle.result)" in text, text
    parts = callers_parts(root, payload)
    assert any("through override TracingHandle.result" in str(part.get("text", "")) for part in parts), parts

    # 3. The CLI says the same, in text and JSON.
    assert _run(monkeypatch, "graph", "callers", root, "DefaultGenerationHandle.result") == 0
    out = capsys.readouterr().out
    assert "callers: 4 result(s) (1 direct, 1 through GenerationHandle.result" in out, out
    assert _run(monkeypatch, "graph", "callers", root, "DefaultGenerationHandle.result", "--json") == 0
    cli_json = capsys.readouterr().out
    assert '"inherited_total": 3' in cli_json and '"binding": "overrider"' in cli_json, cli_json

    # 4. Signatures lists its exact direct sites and counts the inherited ones apart, never as ambiguous.
    signatures = signatures_payload(java_graph, "DefaultGenerationHandle.result")["signatures"]
    assert [caller["caller"] for caller in signatures["callers"]] == ["Director.viaDefault"], signatures
    assert signatures["ambiguous_callers"] == 0, "step 4: an inherited site is not an ambiguous one"
    assert sum(signatures["inherited_callers"].values()) == 3, signatures


def test_a_subtree_view_names_the_declaration_a_site_went_through(java_graph) -> None:
    """Inside a sub-directory the inherited sites rebase like any hit, and an out-of-folder declaration stays named."""
    root = Path(java_graph.path)
    view = SubtreeGraphProvider(SqliteGraphProvider(str(root)), "app/src/main/java/app", str(root / "app"))
    try:
        (default,) = view.definition("DefaultGenerationHandle.result")
        hits = view.callers(default.id)
        assert [hit.symbol.file_path for hit in hits] == ["Director.java"] * 4, "rebased callers"
        through = next(hit.through for hit in hits if hit.binding.value == "overridden")
        assert through is not None and through.file_path == "pipeline/GenerationHandle.java", "rebased through"
    finally:
        view.close()


def test_the_generic_lane_follows_its_own_override_relation(tmp_path: Path, graph_cache: Path) -> None:
    """Python derives overrides from `class Circle(Shape)`, so its callers follow them the same way."""
    root = tmp_path / "py"
    _write(root, PYTHON)
    build_graph(str(root))
    provider = SqliteGraphProvider(str(root))
    try:
        circle_area = _one(provider, "Circle.area")
        assert _sites(provider.callers(circle_area.id)) == [
            ("describe", "direct", None),
            ("twice", "overridden", "shapes.Shape.area"),
        ], "the call on Shape's own self is bound to Shape.area"
        shape_area = _one(provider, "Shape.area")
        assert _sites(provider.callers(shape_area.id)) == [
            ("twice", "direct", None),
            ("describe", "overrider", "shapes.Circle.area"),
        ]
    finally:
        provider.close()
