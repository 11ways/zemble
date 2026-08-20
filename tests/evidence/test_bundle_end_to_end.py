"""An evidence bundle over a real index and a real graph of a small Java tree."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests.conftest import FakeEmbedder
from zemble.evidence.bundle import ItemKind, build_bundle
from zemble.graph import SqliteGraphProvider, build_graph
from zemble.index import ZembleIndex

_FILES = {
    "src/main/java/com/demo/SessionCache.java": """
        package com.demo;

        /** Holds sessions until they expire. */
        public class SessionCache {

            private final java.util.Map<String, String> entries = new java.util.HashMap<>();

            public String lookup(String token) {
                return entries.get(token);
            }

            public void store(String token, String session) {
                entries.put(token, session);
            }

            public void evictAll() {
                entries.clear();
            }
        }
        """,
    "src/main/java/com/demo/LoginHandler.java": """
        package com.demo;

        public class LoginHandler {

            private final SessionCache cache = new SessionCache();

            public String login(String token) {
                cache.store(token, "session-for-" + token);
                return cache.lookup(token);
            }
        }
        """,
    "src/test/java/com/demo/SessionCacheTest.java": """
        package com.demo;

        public class SessionCacheTest {

            public void storeAndLookupJourney() {
                SessionCache cache = new SessionCache();
                cache.store("a", "one");
                cache.lookup("a");
                cache.evictAll();
            }
        }
        """,
}


@pytest.fixture
def java_workspace(tmp_path: Path, graph_cache: Path) -> Path:
    """A three-file Java workspace with a cache, a caller and a test."""
    for relative, source in _FILES.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")
    (tmp_path / "src/main/java/com/demo/README.md").write_text(
        "# Demo package\n\nThe session cache is keyed by presented token.\n", encoding="utf-8"
    )
    return tmp_path


def test_bundle_end_to_end(java_workspace: Path) -> None:
    """A bundle carries the primary chunk, its outline and its callers, each with a reason."""
    build_graph(str(java_workspace))
    graph = SqliteGraphProvider(str(java_workspace))
    index = ZembleIndex.from_path(java_workspace, embedder=FakeEmbedder())

    # 1. A generous budget shows the searched-for code as content.
    bundle = build_bundle(index, graph, "store a session against its token", 4000)
    assert bundle.items, "step 1: the bundle is not empty"
    chunks = [item for item in bundle.items if item.kind is ItemKind.CHUNK]
    assert any("SessionCache.java" in item.file_path for item in chunks), "step 1: the subject file is a primary chunk"

    # 2. The graph hop adds the enclosing type's outline, without any file text.
    outlines = [item for item in bundle.items if item.kind is ItemKind.OUTLINE]
    assert outlines, "step 2: an outline was expanded from an anchor"
    assert any("void store(" in item.text for item in outlines), "step 2: the outline lists the type's members"

    # 3. A caller arrives with a reason saying how it was resolved.
    callers = [item for item in bundle.items if item.kind is ItemKind.CALLER]
    assert callers, "step 3: at least one call site was expanded"
    assert any("called from" in item.reason for item in callers), "step 3: the reason explains the edge"
    assert any("LoginHandler" in item.file_path or "SessionCacheTest" in item.file_path for item in callers), (
        "step 3: the caller is the file that really calls it"
    )

    # 4. Every item carries its own token cost and the total is honest.
    assert bundle.total_tokens == sum(item.tokens for item in bundle.items), "step 4: the total is the item sum"
    assert bundle.rendered_tokens <= 4000, "step 4: the whole rendered answer holds the budget"

    # 5. The markdown says what each item is and why.
    markdown = bundle.render()
    assert markdown.startswith("# Evidence for: store a session against its token"), "step 5: the query is the title"
    assert "```java" in markdown, "step 5: code is fenced with its language"
    for item in bundle.items:
        assert f"## {item.location}  ({item.reason})" in markdown, "step 5: every item is headed by location and reason"

    # 6. A small budget keeps the same shape and still names what it dropped.
    tight = build_bundle(index, graph, "store a session against its token", 500)
    assert tight.rendered_tokens <= 500, "step 6: the tight budget holds for the rendered answer"
    assert tight.omitted or tight.unlisted_omissions, "step 6: what did not fit is still accounted for"
    assert any(item.presentation is not item.presentation.CONTENT for item in tight.items), (
        "step 6: items degrade rather than vanish"
    )
    graph.close()


def test_bundle_without_results_is_empty(java_workspace: Path) -> None:
    """A query that matches nothing yields an empty bundle rather than an error."""
    build_graph(str(java_workspace))
    graph = SqliteGraphProvider(str(java_workspace))
    index = ZembleIndex.from_path(java_workspace, embedder=FakeEmbedder())

    bundle = build_bundle(index, graph, "   ", 2000)
    assert not bundle.items, "an empty query returns nothing"
    assert bundle.total_tokens == 0, "and costs nothing"
    graph.close()
