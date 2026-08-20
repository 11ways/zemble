"""A home answer over a real index and a real symbol graph of a small Java workspace."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tests.conftest import FakeEmbedder
from zemble.graph import SqliteGraphProvider, build_graph
from zemble.home.answers import build_answer, home_payload
from zemble.home.config import HomeConfig
from zemble.home.decide import Verdict
from zemble.index import ZembleIndex
from zemble.types import ContentType

_FILES = {
    "core/src/main/java/com/demo/SessionCookies.java": """
        package com.demo;

        /** Assembles the one session cookie every entry point sends. */
        public class SessionCookies {

            public String assemble(String token) {
                return "session=" + token + "; HttpOnly; Secure";
            }

            public String clear() {
                return "session=; Max-Age=0";
            }
        }
        """,
    "app/src/main/java/com/app/LoginHandler.java": """
        package com.app;

        import com.demo.SessionCookies;

        public class LoginHandler {

            private final SessionCookies cookies = new SessionCookies();

            public String login(String token) {
                return cookies.assemble(token);
            }
        }
        """,
    "ui/src/main/java/com/ui/LogoutButton.java": """
        package com.ui;

        import com.demo.SessionCookies;

        public class LogoutButton {

            private final SessionCookies cookies = new SessionCookies();

            public String logout() {
                return cookies.clear();
            }
        }
        """,
}

_CONFIG = """
    order = ["core", "ui", "app"]

    [modules]
    core = "core/**"
    ui = "ui/**"
    app = "app/**"

    [[tables]]
    file = "ARCH.md"
    capability = "Capability"
    home = "Mechanism home"
    consumers = "Consumers"

    [skills]
    ui = ["ui-components"]

    [[rules]]
    text = "Nothing lands without a wired consumer and a test"
"""

_ARCH = """
    # Architecture

    | Capability | Mechanism home | Consumers |
    | --- | --- | --- |
    | Session cookie assembly (one cookie shape for every entry point) | `core` (`session`) | app, ui |
"""


@pytest.fixture
def workspace(tmp_path: Path, graph_cache: Path) -> Path:
    """A three-module Java workspace with a declared-home table and a doc."""
    for relative, source in _FILES.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")
    (tmp_path / ".zemble").mkdir()
    (tmp_path / ".zemble" / "home.toml").write_text(textwrap.dedent(_CONFIG).strip() + "\n", encoding="utf-8")
    (tmp_path / "ARCH.md").write_text(textwrap.dedent(_ARCH).strip() + "\n", encoding="utf-8")
    (tmp_path / "core" / "README.md").write_text(
        "# core\n\nSession cookie assembly lives here; nobody hand-writes a Set-Cookie header.\n", encoding="utf-8"
    )
    build_graph(str(tmp_path))
    return tmp_path


def test_home_answer_journey(workspace: Path) -> None:
    """Walk one description from search through the graph to a verdict and its renderings."""
    index = ZembleIndex.from_path(workspace, content=[ContentType.CODE, ContentType.DOCS], embedder=FakeEmbedder())
    config = HomeConfig.load(workspace)
    graph = SqliteGraphProvider(str(workspace))
    try:
        answer = build_answer(index, graph, config, "assemble the session cookie every entry point sends")

        # 1. The existing class is found, in the module that really holds it.
        labels = [mechanism.label for mechanism in answer.mechanisms]
        assert "SessionCookies" in labels, f"step 1: the mechanism is found, got {labels}"
        found = next(mechanism for mechanism in answer.mechanisms if mechanism.label == "SessionCookies")
        assert found.module == "core", "step 1: its module comes from the declared glob"

        # 2. The graph supplies the consumer spread, its own module excluded.
        assert set(found.consumer_modules) == {"ui", "app"}, f"step 2: both consumers, got {found.consumer_modules}"
        assert found.strong, "step 2: two consuming modules make it a strong match"

        # 3. So the verdict is to extend it, not to build a second one.
        assert answer.verdict is Verdict.EXTEND_EXISTING, "step 3: an existing mechanism is extended"
        assert answer.home == "core" and answer.extend is not None, "step 3: it names what and where"

        # 4. The declared table is quoted as evidence, with the row it came from.
        assert answer.declared, "step 4: the ARCH.md row matched"
        assert answer.declared[0].row.home_modules == ("core",), "step 4: and declares core the home"

        # 5. The docs lane and find-related both contribute locations.
        assert any(entry.file_path.endswith("README.md") for entry in answer.docs), "step 5: documentation is reported"
        assert answer.similar, "step 5: the best hit's neighbours are offered"

        # 6. Candidate homes rank core first and explain themselves.
        assert answer.candidates[0].module == "core", "step 6: core leads the candidates"
        assert answer.candidates[0].reasons, "step 6: with reasons attached"

        # 7. The checklist carries the workspace's own rule.
        assert "Nothing lands without a wired consumer and a test" in answer.checklist.rules, "step 7: rules are echoed"

        # 8. Disabling the table lane removes exactly that evidence, nothing else.
        without = build_answer(
            index, graph, config, "assemble the session cookie every entry point sends", use_tables=False
        )
        assert without.declared == [], "step 8: no declared rows when the lane is off"
        assert [m.label for m in without.mechanisms] == labels, "step 8: the rest of the evidence is unchanged"

        # 9. The payload carries both renderings.
        payload = home_payload(index, graph, config, "assemble the session cookie every entry point sends")
        assert payload["home"]["verdict"] == "EXTEND_EXISTING", "step 9: the data shape carries the verdict"
        assert payload["markdown"].startswith("# Home for:"), "step 9: and the markdown its title"
    finally:
        graph.close()


def test_a_description_that_matches_nothing(workspace: Path) -> None:
    """A description with no hits is uncertain rather than wrong."""
    index = ZembleIndex.from_path(workspace, embedder=FakeEmbedder())
    graph = SqliteGraphProvider(str(workspace))
    try:
        answer = build_answer(index, graph, HomeConfig.load(workspace), "   ")
        assert answer.verdict is Verdict.UNCERTAIN, "an empty description names no home"
        assert answer.docs == [], "a code-only index has no documentation lane"
    finally:
        graph.close()
