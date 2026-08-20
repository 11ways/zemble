"""Behaviour journeys over the generated-source mapper: javac facts folded back onto `.hwk`.

The fixture workspace under `tests/fixtures/generated` carries REAL Hawkeye output, copied
verbatim out of the javaweb workspace: `zenit-widget`'s `widget-display/markdown.hwk` with its
`Tpl_WidgetDisplayMarkdown.java` and `Tpl_WidgetDisplayMarkdown.sourcemap.json`. The two
generated classes that carry no source map - a tag class and `HawkeyeClassSerializers` - are
written by hand, because Hawkeye emits no map for either and there is nothing real to copy.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from zemble.cli import _cli_main
from zemble.graph.generated import camel_case_identifier, parse_generated_path
from zemble.graph.store import build_graph, connect

TEMPLATE = "src/common/templates/widget-display/markdown.hwk"
CARD = "src/common/templates/components/card.hwk"
GENERATED_ROOT = "build/generated-sources/hawkeye/common/java/be/elevenways/hawkeye/generated"
TPL = f"{GENERATED_ROOT}/zenitwidget/Tpl_WidgetDisplayMarkdown.java"
SERIALIZERS = f"{GENERATED_ROOT}/zenitwidget/HawkeyeClassSerializers.java"
TAG_IMPL = f"{GENERATED_ROOT}/tags/demo/DemoCardImpl.java"
PANEL = "src/common/templates/components/panel.hwk"
PANEL_IMPL = f"{GENERATED_ROOT}/tags/demo/DemoPanelImpl.java"

RENDER_ROOT = (
    "be.elevenways.hawkeye.generated.zenitwidget.Tpl_WidgetDisplayMarkdown"
    "#renderRoot(be.elevenways.hawkeye.common.render.RenderContext)"
)
BRANCH2 = (
    "be.elevenways.hawkeye.generated.zenitwidget.Tpl_WidgetDisplayMarkdown"
    "#branch2(be.elevenways.hawkeye.common.render.RenderContext)"
)
TAG_RENDER = "be.elevenways.hawkeye.generated.tags.demo.DemoCardImpl#render()"
PANEL_RENDER = "be.elevenways.hawkeye.generated.tags.demo.DemoPanelImpl#render()"
SERIALIZER_REGISTER = "be.elevenways.hawkeye.generated.zenitwidget.HawkeyeClassSerializers#register()"
RENDER = "com.example.ui.WidgetFunctions#render(java.lang.String)"
LOCALIZED = "com.example.ui.WidgetFunctions#localizedConfig(java.lang.Object,java.lang.String)"


@pytest.fixture
def workspace(tmp_path: Path, graph_cache: Path) -> Path:
    """A writable copy of the generated-source fixture workspace."""
    destination = tmp_path / "ws"
    shutil.copytree(Path(__file__).parent / "fixtures" / "generated", destination)
    _generated_after_templates(destination)
    return destination


def _generated_after_templates(workspace: Path) -> None:
    """Stamp the generated files as produced AFTER the templates they were compiled from."""
    for relative in (TEMPLATE, CARD, PANEL):
        os.utime(workspace / relative, (1_000_000, 1_000_000))
    for relative in (TPL, SERIALIZERS, TAG_IMPL, PANEL_IMPL):
        os.utime(workspace / relative, (2_000_000, 2_000_000))


def _sha(workspace: Path, relative: str) -> str:
    """Hex sha256 of a workspace file's current content."""
    return hashlib.sha256((workspace / relative).read_bytes()).hexdigest()


def _facts(workspace: Path) -> None:
    """Write the facts a javac emitter would produce for the fixture's generated classes."""
    lines = [
        {
            "zemble_facts": 1,
            "tool": "zemble-javac-facts",
            "tool_version": "0.1.0",
            "generated_at": "2026-08-20T09:00:00Z",
            "language": "java",
            "root": "../..",
        },
        {"t": "file", "path": TPL, "sha256": _sha(workspace, TPL)},
        # Line 74 is the discriminator: the `.java` marker above it says template line 6, the
        # sidecar - written before the compiler injected two lines - would answer 8.
        {"t": "call", "from": RENDER_ROOT, "to": RENDER, "path": TPL, "line": 74},
        {"t": "call", "from": BRANCH2, "to": LOCALIZED, "path": TPL, "line": 81},
        {"t": "call", "from": BRANCH2, "to": RENDER, "path": TPL, "line": 83},
        {"t": "override", "from": BRANCH2, "to": RENDER},
        {"t": "file", "path": TAG_IMPL, "sha256": _sha(workspace, TAG_IMPL)},
        {"t": "call", "from": TAG_RENDER, "to": RENDER, "path": TAG_IMPL, "line": 11},
        {"t": "file", "path": PANEL_IMPL, "sha256": _sha(workspace, PANEL_IMPL)},
        {"t": "call", "from": PANEL_RENDER, "to": RENDER, "path": PANEL_IMPL, "line": 7},
        {"t": "file", "path": SERIALIZERS, "sha256": _sha(workspace, SERIALIZERS)},
        {"t": "call", "from": SERIALIZER_REGISTER, "to": RENDER, "path": SERIALIZERS, "line": 7},
    ]
    path = workspace / ".zemble" / "facts" / "javac.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def _calls(workspace: Path, where: str) -> list[dict]:
    """Every stored call edge of one source file, ordered by line then target."""
    connection = connect(str(workspace))
    try:
        rows = connection.execute("SELECT * FROM edges WHERE file_path = ? AND kind = 'calls'", (where,)).fetchall()
    finally:
        connection.close()
    return sorted((dict(row) for row in rows), key=lambda edge: (edge["line"], edge["dst_name"]))


def _status(monkeypatch: pytest.MonkeyPatch, workspace: Path, capsys) -> dict:
    """Run `graph facts status --json` over a workspace and return its payload."""
    from zemble.graph import cli as graph_cli

    graph_cli._refreshed.clear()
    monkeypatch.setattr("sys.argv", ["zemble", "graph", "facts", "status", str(workspace), "--json"])
    try:
        _cli_main()
    except SystemExit:
        pass
    return json.loads(capsys.readouterr().out)


def _bucket(payload: dict, name: str) -> dict:
    """One skipped-fact bucket out of a `--json` status payload."""
    return next(bucket for bucket in payload["skipped"] if bucket["bucket"] == name)


def test_generated_facts_journey(workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """Facts about generated Hawkeye classes become exact template edges, and stay honest."""
    # 1. Without facts the template's calls are the extractor's, resolved by name alone.
    build_graph(str(workspace))
    before = _calls(workspace, TEMPLATE)
    assert [edge["dst_name"] for edge in before] == ["localizedConfig", "render"], "step 1: both calls extracted"
    assert {edge["source"] for edge in before} == {"tree-sitter"}, "step 1: and the extractor owns them"

    # 2. With facts, the generated class's calls land on the template through the source map.
    _facts(workspace)
    build_graph(str(workspace))
    mapped = _calls(workspace, TEMPLATE)
    assert {edge["source"] for edge in mapped} == {"zemble-javac-facts"}, "step 2: the facts replaced them"
    assert {edge["resolution"] for edge in mapped} == {"exact"}, "step 2: and resolution is exact"

    # 3. The `.java` markers place them, not the sidecar: line 74 is template line 6.
    positions = sorted((edge["line"], edge["dst_name"]) for edge in mapped)
    assert positions == [(6, "render"), (8, "localizedConfig"), (8, "render")], "step 3: marker-placed lines"

    # 4. Every mapped edge keeps the generated member it was actually written about.
    assert all(edge["origin_ref"] for edge in mapped), "step 4: the detour is visible on the edge"
    assert {edge["origin_ref"] for edge in mapped} == {RENDER_ROOT, BRANCH2}, "step 4: and names the real source"

    # 5. A tag class carries no source map at all, so it maps by class name, without a line.
    tag_calls = _calls(workspace, CARD)
    assert [edge["source"] for edge in tag_calls] == ["zemble-javac-facts"], "step 5: the tag class mapped too"
    assert tag_calls[0]["origin_ref"] == TAG_RENDER, "step 5: through DemoCardImpl"

    # 6. What is generated but is no template is counted, never attributed to one.
    payload = _status(monkeypatch, workspace, capsys)
    no_template = _bucket(payload, "generated_no_template")
    assert no_template["count"] == 2, "step 6: the serializer call and the override fact"
    reasons = {entry["reason"] for entry in no_template["top"]}
    assert "no template answers to generated class" in " ".join(reasons), "step 6: the class is named"
    assert "only calls map back through a template source map" in reasons, "step 6: and so is the kind"

    # 7. The status report says how much the source maps recovered.
    assert payload["coverage"]["generated_mapped"] == 5, "step 7: three template calls plus two tags'"
    assert payload["coverage"]["generated_templates"] == 3, "step 7: across three templates"


def test_a_template_edited_after_generation_is_stale(workspace: Path) -> None:
    """A template newer than the class generated from it keeps its extracted edges."""
    _facts(workspace)
    build_graph(str(workspace))
    assert {edge["source"] for edge in _calls(workspace, TEMPLATE)} == {"zemble-javac-facts"}, "mapped to begin with"

    # 1. Editing the template makes it newer than the generated class: the map cannot be trusted.
    (workspace / TEMPLATE).write_text(
        (workspace / TEMPLATE).read_text(encoding="utf-8") + "\n<p>added</p>\n", encoding="utf-8"
    )
    os.utime(workspace / TEMPLATE, (3_000_000, 3_000_000))
    build_graph(str(workspace))
    reverted = _calls(workspace, TEMPLATE)
    assert {edge["source"] for edge in reverted} == {"tree-sitter"}, "step 1: the extractor's edges came back"

    # 2. Regenerating - the class newer again - hands the template back to the facts.
    os.utime(workspace / TPL, (4_000_000, 4_000_000))
    build_graph(str(workspace))
    assert {edge["source"] for edge in _calls(workspace, TEMPLATE)} == {"zemble-javac-facts"}, "step 2: mapped again"


def test_template_extends_survives_a_mapped_call(workspace: Path) -> None:
    """The overlay owns only a mapped template's CALLS; what it renders stays the extractor's."""
    _facts(workspace)
    build_graph(str(workspace))
    assert [edge["source"] for edge in _calls(workspace, PANEL)] == ["zemble-javac-facts"], "its calls are mapped"
    connection = connect(str(workspace))
    try:
        rows = connection.execute(
            "SELECT kind, source FROM edges WHERE file_path = ? AND kind != 'calls'", (PANEL,)
        ).fetchall()
    finally:
        connection.close()
    assert [row["kind"] for row in rows] == ["references_type"], "the `<demo-card>` it renders survived"
    assert {row["source"] for row in rows} == {"tree-sitter"}, "and the extractor still owns it"


def test_generated_paths_and_class_names_are_read_by_the_compiler_rules() -> None:
    """The two naming rules are applied exactly as Hawkeye's own compiler spells them."""
    origin = parse_generated_path(f"zenit-cms/{GENERATED_ROOT}/zenitcms/Tpl_PagesResourceList.java")
    assert origin is not None, "a generated template class is recognised"
    assert (origin.module, origin.source_set) == ("zenit-cms", "common"), "module and source set are split off"
    assert origin.template_root == "zenit-cms/src/common/templates/", "which names where its templates live"
    assert not origin.is_tag_class, "and it is a template class, not a tag class"
    assert camel_case_identifier("pages/resource-list") == "PagesResourceList", "the class-name rule"

    tagged = parse_generated_path(f"plumage/{GENERATED_ROOT}/tags/plumage/PlBadgeImpl.java")
    assert tagged is not None and tagged.is_tag_class, "a tag class is recognised by its package"
    assert parse_generated_path("src/common/java/com/example/Thing.java") is None, "hand-written source is not ours"
    assert parse_generated_path(f"{GENERATED_ROOT}/zenitwidget/Tpl_X.class") is None, "and neither is a class file"
