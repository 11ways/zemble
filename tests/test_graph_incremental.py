"""Behaviour journeys proving an incremental refresh lands where a full rebuild would.

An incremental build reads a fraction of the workspace: only the changed files' symbols, only
the names their edges write, only the facts files whose coverage it is re-resolving. Every one
of those narrowings is a chance to leave a stale edge behind, and none of them is visible in a
count. So each step here applies one edit, refreshes, and compares the whole `symbols` and
`edges` tables against a from-scratch build of the very same tree.

The journey runs twice, once through each :class:`SymbolLookup`: the two must be
indistinguishable, since which one a build picks is a cost decision and nothing else.
"""

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import pytest

from zemble.graph.store import build_graph, connect

CIRCLE = "src/main/java/com/example/core/Circle.java"
HELPERS = "src/main/java/com/example/util/Helpers.java"
TRAVERSAL = "src/main/java/com/example/core/Traversal.java"
CALLER = "src/main/java/com/example/app/Caller.java"

TEMPLATE = "src/common/templates/widget-display/markdown.hwk"
CARD = "src/common/templates/components/card.hwk"
PANEL = "src/common/templates/components/panel.hwk"
GENERATED_ROOT = "build/generated-sources/hawkeye/common/java/be/elevenways/hawkeye/generated"
TPL = f"{GENERATED_ROOT}/zenitwidget/Tpl_WidgetDisplayMarkdown.java"
SERIALIZERS = f"{GENERATED_ROOT}/zenitwidget/HawkeyeClassSerializers.java"
TAG_IMPL = f"{GENERATED_ROOT}/tags/demo/DemoCardImpl.java"
PANEL_IMPL = f"{GENERATED_ROOT}/tags/demo/DemoPanelImpl.java"
FACTS = ".zemble/facts/javac.jsonl"

RENDER_ROOT = (
    "be.elevenways.hawkeye.generated.zenitwidget.Tpl_WidgetDisplayMarkdown"
    "#renderRoot(be.elevenways.hawkeye.common.render.RenderContext)"
)
TAG_RENDER = "be.elevenways.hawkeye.generated.tags.demo.DemoCardImpl#render()"
RENDER = "com.example.ui.WidgetFunctions#render(java.lang.String)"


@pytest.fixture(params=[0, 1_000_000], ids=["sqlite-lookup", "memory-lookup"])
def either_lookup(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> int:
    """Force every build of a journey through one of the two symbol lookups."""
    monkeypatch.setattr("zemble.graph.store._MEMORY_LOOKUP_TARGETS", request.param)
    return request.param


def _tables(root: Path) -> dict[str, Counter]:
    """Return the graph's derived tables as multisets, which is what identity means here.

    Row ORDER carries no meaning - an incremental build reinserts a file's rows at the end of
    the table - so the tables are compared as bags of rows and nothing else.
    """
    connection = connect(str(root))
    try:
        return {
            table: Counter(tuple(row) for row in connection.execute(f"SELECT * FROM {table}"))  # noqa: S608
            for table in ("symbols", "edges", "decl_keys")
        }
    finally:
        connection.close()


def _assert_matches_full_rebuild(workspace: Path, scratch: Path, step: str) -> None:
    """Build the same tree from nothing elsewhere and demand the same tables."""
    fresh = scratch / f"fresh-{step.replace(' ', '-')}"
    shutil.copytree(workspace, fresh)
    build_graph(str(fresh))
    incremental, rebuilt = _tables(workspace), _tables(fresh)
    for table in incremental:
        missing = rebuilt[table] - incremental[table]
        extra = incremental[table] - rebuilt[table]
        assert not missing, f"{step}: the refresh is missing {len(missing)} {table} row(s): {list(missing)[:3]}"
        assert not extra, f"{step}: the refresh kept {len(extra)} stale {table} row(s): {list(extra)[:3]}"


def _write(workspace: Path, relative: str, text: str) -> Path:
    """Write a workspace file and hand back its path."""
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_java_edit_journey_matches_a_full_rebuild(
    graph_fixture_root: Path, graph_cache: Path, tmp_path: Path, either_lookup: int
) -> None:
    """A rename, a package move, a deletion and a new caller each refresh to the full answer."""
    workspace = tmp_path / "ws"
    shutil.copytree(graph_fixture_root, workspace)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    build_graph(str(workspace))
    _assert_matches_full_rebuild(workspace, scratch, "step 0 first build")

    # 1. Renaming a method moves the name-to-declaration map, so its callers re-resolve.
    circle = workspace / CIRCLE
    circle.write_text(circle.read_text(encoding="utf-8").replace("scale", "resize"), encoding="utf-8")
    build_graph(str(workspace))
    _assert_matches_full_rebuild(workspace, scratch, "step 1 rename")

    # 2. Moving a class to another package: every importer's scope answer changes.
    helpers = (workspace / HELPERS).read_text(encoding="utf-8")
    (workspace / HELPERS).unlink()
    moved = "src/main/java/com/example/extra/Helpers.java"
    _write(workspace, moved, helpers.replace("package com.example.util;", "package com.example.extra;"))
    build_graph(str(workspace))
    _assert_matches_full_rebuild(workspace, scratch, "step 2 move")

    # 3. Deleting a file takes its symbols and edges with it, named through the change set.
    (workspace / TRAVERSAL).unlink()
    build_graph(str(workspace), changed_paths=[workspace / TRAVERSAL])
    _assert_matches_full_rebuild(workspace, scratch, "step 3 delete")

    # 4. A new file appears, and only the watcher's word says so.
    caller = _write(
        workspace,
        CALLER,
        "package com.example.app;\n\n"
        "import com.example.extra.Helpers;\n\n"
        "public class Caller {\n"
        "    public double twice(double value) {\n"
        "        return Helpers.twice(value);\n"
        "    }\n"
        "}\n",
    )
    build_graph(str(workspace), changed_paths=[caller])
    _assert_matches_full_rebuild(workspace, scratch, "step 4 add")

    # 5. Undoing the rename puts the workspace back, and the graph with it.
    circle.write_text(circle.read_text(encoding="utf-8").replace("resize", "scale"), encoding="utf-8")
    build_graph(str(workspace), changed_paths=[circle])
    _assert_matches_full_rebuild(workspace, scratch, "step 5 undo")


def _sha(workspace: Path, relative: str) -> str:
    """Hex sha256 of a workspace file's current content."""
    return hashlib.sha256((workspace / relative).read_bytes()).hexdigest()


def _facts_lines(workspace: Path) -> list[dict]:
    """The facts a javac emitter would write about the fixture's generated classes."""
    return [
        {
            "zemble_facts": 1,
            "tool": "zemble-javac-facts",
            "tool_version": "0.1.0",
            "generated_at": "2026-08-20T09:00:00Z",
            "language": "java",
            "root": "../..",
        },
        {"t": "file", "path": TPL, "sha256": _sha(workspace, TPL)},
        {"t": "call", "from": RENDER_ROOT, "to": RENDER, "path": TPL, "line": 74},
        {"t": "file", "path": TAG_IMPL, "sha256": _sha(workspace, TAG_IMPL)},
        {"t": "call", "from": TAG_RENDER, "to": RENDER, "path": TAG_IMPL, "line": 11},
    ]


def _write_facts(workspace: Path, lines: list[dict]) -> Path:
    """Write the workspace's facts file."""
    return _write(workspace, FACTS, "\n".join(json.dumps(line) for line in lines) + "\n")


@pytest.fixture
def generated_workspace(tmp_path: Path, graph_cache: Path) -> Path:  # noqa: ARG001
    """A writable copy of the generated-source fixture, stamped so its facts apply."""
    workspace = tmp_path / "ws"
    shutil.copytree(Path(__file__).parent / "fixtures" / "generated", workspace)
    for relative in (TEMPLATE, CARD, PANEL):
        os.utime(workspace / relative, (1_000_000, 1_000_000))
    for relative in (TPL, SERIALIZERS, TAG_IMPL, PANEL_IMPL):
        os.utime(workspace / relative, (2_000_000, 2_000_000))
    return workspace


def test_template_and_facts_journey_matches_a_full_rebuild(
    generated_workspace: Path, tmp_path: Path, either_lookup: int
) -> None:
    """Touching a template, editing one, and rewriting the facts all refresh to the full answer."""
    workspace = generated_workspace
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    build_graph(str(workspace))
    _assert_matches_full_rebuild(workspace, scratch, "step 0 no facts")

    # 1. A facts file appears: its generated classes' calls land on the templates.
    _write_facts(workspace, _facts_lines(workspace))
    build_graph(str(workspace))
    _assert_matches_full_rebuild(workspace, scratch, "step 1 facts appear")

    # 2. Touching the template past its generated class makes those facts stale, with no facts
    #    file moving at all: the verdict is two stats, and it must still be noticed.
    os.utime(workspace / TEMPLATE, (3_000_000, 3_000_000))
    build_graph(str(workspace), changed_paths=[workspace / TEMPLATE])
    _assert_matches_full_rebuild(workspace, scratch, "step 2 template newer")

    # 3. Recompiling puts the generated class back in front, and the facts apply again.
    os.utime(workspace / TPL, (4_000_000, 4_000_000))
    build_graph(str(workspace), changed_paths=[workspace / TPL])
    _assert_matches_full_rebuild(workspace, scratch, "step 3 regenerated")

    # 4. Editing a template's markup changes its own symbols and the edges that reach it.
    card = workspace / CARD
    card.write_text(card.read_text(encoding="utf-8").replace("</div>", "<span>extra</span></div>", 1), encoding="utf-8")
    build_graph(str(workspace), changed_paths=[card])
    _assert_matches_full_rebuild(workspace, scratch, "step 4 template edit")

    # 5. Rewriting the facts file with one fact fewer takes exactly that edge away.
    _write_facts(workspace, _facts_lines(workspace)[:3])
    build_graph(str(workspace), changed_paths=[workspace / FACTS])
    _assert_matches_full_rebuild(workspace, scratch, "step 5 facts shrink")

    # 6. Deleting the facts file hands every template back to the extractor.
    (workspace / FACTS).unlink()
    build_graph(str(workspace), changed_paths=[workspace / FACTS])
    _assert_matches_full_rebuild(workspace, scratch, "step 6 facts gone")
