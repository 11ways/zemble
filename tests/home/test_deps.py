"""The module dependency graph: what it reads, what it merges and what it refuses to guess."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from zemble.home.config import ConfigError, HomeConfig
from zemble.home.deps import DependencyGraph, DependencySource, EdgeOrigin, Reachability
from zemble.home.gradle import discover, is_code_configuration

_MODULES = ("protoblast", "zenit", "zenit-flow", "zenit-widget")

_BUILD = """
    plugins {{ id 'java-library' }}

    dependencies {{
    {body}
    }}
"""


def _toml(deps: dict[str, list[str]], extra: str) -> str:
    """Render a four-module home.toml, writing the modules that declare dependencies as tables."""
    lines = [f"order = {list(_MODULES)!r}".replace("'", '"'), "", "[modules]"]
    lines += [f'{module} = "{module}/**"' for module in _MODULES if module not in deps]
    for module, targets in deps.items():
        rendered = ", ".join(f'"{target}"' for target in targets)
        lines += ["", f"[modules.{module}]", f'globs = ["{module}/**"]', f"depends_on = [{rendered}]"]
    return "\n".join(lines) + "\n" + textwrap.dedent(extra)


def _workspace(
    root: Path,
    deps: dict[str, list[str]] | None = None,
    extra: str = "",
    builds: dict[str, str] | None = None,
) -> HomeConfig:
    """Write a four-module workspace, optionally with declared dependencies and build files."""
    (root / ".zemble").mkdir(parents=True, exist_ok=True)
    (root / ".zemble" / "home.toml").write_text(_toml(deps or {}, extra), encoding="utf-8")
    for module, body in (builds or {}).items():
        directory = root / module
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "settings.gradle").write_text(f"rootProject.name = '{module}'\n", encoding="utf-8")
        (directory / "build.gradle").write_text(textwrap.dedent(_BUILD).format(body=body), encoding="utf-8")
    return HomeConfig.load(root)


def test_reachability_answers_every_shape(tmp_path: Path) -> None:
    """Walk one declared graph through each Reachability value it can produce."""
    config = _workspace(
        tmp_path,
        deps={"zenit": ["protoblast"], "zenit-flow": ["zenit"], "zenit-widget": ["zenit"]},
        extra="""
        [[forbidden]]
        from = "zenit-widget"
        to = "protoblast"
        why = "the widget layer goes through zenit"
        """,
    )
    graph = config.dependencies

    # 1. A declared edge is DIRECT, and what it leads to is TRANSITIVE.
    assert config.reachable("zenit-flow", "zenit") is Reachability.DIRECT, "step 1: the declared edge"
    assert config.reachable("zenit", "protoblast") is Reachability.DIRECT, "step 1: one hop"
    assert config.reachable("zenit-flow", "protoblast") is Reachability.TRANSITIVE, "step 1: two hops"

    # 2. Siblings do not reach each other, in either direction, however close in `order`.
    assert config.reachable("zenit-flow", "zenit-widget") is Reachability.UNREACHABLE, "step 2: siblings"
    assert config.reachable("zenit-widget", "zenit-flow") is Reachability.UNREACHABLE, "step 2: the other way"
    assert config.reachable("zenit", "zenit-flow") is Reachability.UNREACHABLE, "step 2: order grants nothing"

    # 3. A forbidden pair loses whatever the edges say.
    assert config.reachable("zenit-widget", "protoblast") is Reachability.FORBIDDEN, "step 3: the refusal wins"

    # 4. A module nothing is known about is UNKNOWN, not "no".
    assert config.reachable("protoblast", "zenit") is Reachability.UNKNOWN, "step 4: no edges from protoblast"
    assert Reachability.UNKNOWN.usable is False, "step 4: and unknown is never treated as permission"

    # 5. Every module reaches itself.
    assert config.reachable("zenit", "zenit") is Reachability.DIRECT, "step 5: a module is its own home"

    # 6. The shared home of two siblings is the highest-ranked module both can reach.
    assert config.nearest_common_dependency(["zenit-flow", "zenit-widget"]) == "zenit", "step 6: their substrate"
    assert graph.known, "step 6: this workspace carries dependency information"


def test_an_empty_graph_is_unknown_rather_than_closed() -> None:
    """A workspace that declared nothing must not be read as "nobody may depend on anybody"."""
    graph = DependencyGraph(nodes=("a", "b"))
    assert not graph.known, "no edges at all"
    assert graph.reachable("a", "b") is Reachability.UNKNOWN, "so the answer is unknown"
    assert graph.nearest_common_dependency(["a", "b"], lambda module: 0) is None, "and nothing is suggested"


def test_gradle_discovery_and_declaration_override(tmp_path: Path) -> None:
    """Build files supply the edges nobody declared, and a declaration replaces them."""
    builds = {
        "protoblast": "    implementation 'org.checkerframework:checker-qual:4.2.0'",
        "zenit": "    commonCompileOnly ('be.elevenways:protoblast-client:0.1.0') { changing = true }",
        "zenit-flow": (
            "    serverImplementation ('be.elevenways:zenit-server:0.1.0') { changing = true }\n"
            "    testImplementation project(':zenit-widget')\n"
            "    annotationProcessor 'systems.manifold:manifold-preprocessor:2026.1.6'"
        ),
        "zenit-widget": "    commonCompileOnly ('be.elevenways:zenit-common:0.1.0') { changing = true }",
    }
    config = _workspace(tmp_path, builds=builds)
    graph = config.dependencies

    # 1. A published coordinate resolves to the module that publishes it, fold suffix and all.
    assert graph.targets_of("zenit") == ("protoblast",), "step 1: protoblast-client is protoblast"
    assert graph.targets_of("zenit-widget") == ("zenit",), "step 1: zenit-common is zenit"

    # 2. A test-scoped project reference is still an edge, and its configuration is recorded.
    assert set(graph.targets_of("zenit-flow")) == {"zenit", "zenit-widget"}, "step 2: both lanes"
    configurations = {edge.target: edge.configuration for edge in graph.edges if edge.source == "zenit-flow"}
    assert configurations["zenit"] == "serverImplementation", "step 2: the configuration is kept"
    assert all(edge.origin is EdgeOrigin.DISCOVERED for edge in graph.edges), "step 2: all discovered"

    # 3. A configuration that is not a code dependency declares no edge.
    assert not is_code_configuration("annotationProcessor"), "step 3: build tooling is not a dependency"
    assert graph.targets_of("protoblast") == (), "step 3: an external library is nobody's module"

    # 4. Declaring depends_on REPLACES discovery for that module, empty list included.
    declared = _workspace(tmp_path, deps={"zenit-flow": ["zenit"]}, builds=builds)
    assert declared.dependencies.targets_of("zenit-flow") == ("zenit",), "step 4: the declaration wins"
    assert declared.reachable("zenit-flow", "zenit-widget") is Reachability.UNREACHABLE, "step 4: discovery dropped"
    assert declared.dependencies.targets_of("zenit-widget") == ("zenit",), "step 4: other modules still discovered"


def test_the_dependency_source_selects_the_lanes(tmp_path: Path) -> None:
    """`[dependencies] source` decides which of the two lanes may speak."""
    builds = {"zenit-flow": "    implementation ('be.elevenways:zenit-server:0.1.0') { changing = true }"}
    declared_only = _workspace(
        tmp_path,
        deps={"zenit-widget": ["zenit"]},
        extra="""
        [dependencies]
        source = "declared"
        """,
        builds=builds,
    )
    assert declared_only.dependencies.targets_of("zenit-flow") == (), "discovery is off"
    assert declared_only.dependencies.targets_of("zenit-widget") == ("zenit",), "the declaration still counts"

    gradle_only = _workspace(
        tmp_path,
        deps={"zenit-flow": ["zenit-widget"]},
        extra="""
        [dependencies]
        source = "gradle"
        """,
        builds=builds,
    )
    assert gradle_only.dependency_source is DependencySource.GRADLE, "the source is what was written"
    assert gradle_only.dependencies.targets_of("zenit-flow") == ("zenit",), "only the build file speaks"


def test_a_settings_file_maps_included_projects(tmp_path: Path) -> None:
    """A multi-project build's `include` lines name the directories the edges point at."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "settings.gradle").write_text(
        "rootProject.name = 'app'\ninclude ':core'\ninclude ':web'\n", encoding="utf-8"
    )
    (tmp_path / "app" / "build.gradle").write_text("dependencies {\n    implementation project(':web')\n}\n", "utf-8")
    for child in ("core", "web"):
        (tmp_path / "app" / child).mkdir()
        (tmp_path / "app" / child / "build.gradle").write_text(
            "dependencies {\n    api project(':core')\n}\n", encoding="utf-8"
        )
    projects = {project.gradle_path: project for project in discover(tmp_path)}
    assert projects[":core"].directory == "app/core", "the included project is a directory"
    assert projects[":web"].refs[0].project_path == ":core", "and its dependency is read"


def test_a_depends_on_naming_an_unknown_module_is_refused(tmp_path: Path) -> None:
    """A dependency on a module the workspace never declared is a config error, not an edge."""
    with pytest.raises(ConfigError, match="undeclared module 'zenit-cms'"):
        _workspace(tmp_path, deps={"zenit-flow": ["zenit-cms"]})


def test_the_shared_home_is_the_deepest_one_not_the_most_core(tmp_path: Path) -> None:
    """`nearest_common_dependency` walks down to the modules the siblings actually sit on."""
    # 1. Two siblings on zenit, which itself sits on protoblast: zenit is the shared home.
    deep = _workspace(
        tmp_path / "deep",
        deps={"zenit": ["protoblast"], "zenit-flow": ["zenit"], "zenit-widget": ["zenit"]},
    )
    assert deep.common_dependencies(["zenit-flow", "zenit-widget"]) == ("zenit",), "step 1: protoblast is dropped"
    assert deep.nearest_common_dependency(["zenit-flow", "zenit-widget"]) == "zenit", "step 1: the deepest one wins"

    # 2. Siblings that share only the root library get the root library.
    only = _workspace(tmp_path / "only", deps={"zenit-flow": ["protoblast"], "zenit-widget": ["protoblast"]})
    assert only.nearest_common_dependency(["zenit-flow", "zenit-widget"]) == "protoblast", "step 2: nothing deeper"

    # 3. Two shared modules neither of which reaches the other are a tie, broken by `order`.
    tie = _workspace(
        tmp_path / "tie",
        deps={"zenit-flow": ["zenit", "protoblast"], "zenit-widget": ["zenit", "protoblast"]},
    )
    assert set(tie.common_dependencies(["zenit-flow", "zenit-widget"])) == {"zenit", "protoblast"}, "step 3: both"
    assert tie.nearest_common_dependency(["zenit-flow", "zenit-widget"]) == "protoblast", "step 3: most core wins"

    # 4. Siblings that share nothing are told so rather than pushed into an unrelated module.
    apart = _workspace(tmp_path / "apart", deps={"zenit-flow": ["zenit"], "zenit-widget": ["protoblast"]})
    assert apart.common_dependencies(["zenit-flow", "zenit-widget"]) == (), "step 4: no shared dependency"
    assert apart.nearest_common_dependency(["zenit-flow", "zenit-widget"]) is None, "step 4: and no suggestion"
