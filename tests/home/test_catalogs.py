"""Version catalog accessors in build files: resolved to modules where a catalog applies, reported where none does."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from zemble.home.config import ConfigError, HomeConfig
from zemble.home.decide import decide
from zemble.home.gradle import AliasProblem, discover

_CATALOG = """
    [versions]
    framework = "0.1.0"

    [libraries]
    protoblast-client = { module = "be.elevenways:protoblast-client", version.ref = "framework" }
    protoblast-server = { module = "be.elevenways:protoblast-server", version.ref = "framework" }
    zenit-common = { module = "be.elevenways:zenit-common", version.ref = "framework" }
    zenit-client = "be.elevenways:zenit-client:0.1.0"
    zenit-server = { group = "be.elevenways", name = "zenit-server", version.ref = "framework" }
    zenit-test-support = { module = "be.elevenways:zenit-test-support", version.ref = "framework" }
    zenit_flow-server = { module = "be.elevenways:zenit-flow-server", version.ref = "framework" }
    jackson-databind = { module = "com.fasterxml.jackson.core:jackson-databind", version = "2.17.0" }

    [bundles]
    flow-stack = ["zenit-flow-server", "jackson-databind"]

    [plugins]
    shadow = { id = "com.gradleup.shadow", version = "8.3.0" }
"""

_HOME = """
    order = ["protoblast", "zenit", "zenit-flow", "app", "other-app"]

    [modules]
    protoblast = "protoblast/**"
    zenit = "zenit/**"
    zenit-flow = "zenit-flow/**"
    app = "apps/app/**"
    other-app = "apps/other-app/**"
"""

_PLUGIN = "be.elevenways.protoblast.settings"


def _write(path: Path, text: str) -> None:
    """Write a dedented file, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")


def _repo(root: Path, directory: str, build: str, settings: str = "") -> None:
    """Write one single-project Gradle build: its settings file and its build file."""
    name = directory.rsplit("/", 1)[-1]
    _write(root / directory / "settings.gradle", f"{textwrap.dedent(settings)}\nrootProject.name = '{name}'\n")
    _write(root / directory / "build.gradle", build)


def _home(root: Path, extra: str = "") -> HomeConfig:
    """Write `.zemble/home.toml` and load it."""
    _write(root / ".zemble" / "home.toml", textwrap.dedent(_HOME) + textwrap.dedent(extra))
    return HomeConfig.load(root)


_APPLIES_PLUGIN = f"""
    plugins {{
        id '{_PLUGIN}' version '0.1.0-SNAPSHOT'
    }}
"""

_DECLARES_CATALOG = f"""
    [dependencies]

    [[dependencies.catalogs]]
    plugin = "{_PLUGIN}"
    path = "protoblast/versions/libs.versions.toml"
"""


def test_catalog_accessors_resolve_to_the_modules_that_publish_them(tmp_path: Path) -> None:
    """Gradle's conventional catalog turns every accessor shape a build writes into module edges."""
    _write(tmp_path / "zenit-flow" / "gradle" / "libs.versions.toml", _CATALOG)
    _repo(
        tmp_path,
        "zenit-flow",
        """
        blastModule {
            // A bare group is the trio below it; a list goes on after a trailing comma, comments and all.
            implementation libs.zenit,
                    // the root library's server fold
                    libs.protoblast.server
        }
        dependencies {
            testImplementation(libs.zenit.test.support)
            annotationProcessor libs.no.such.processor
        }
        """,
    )
    _repo(tmp_path, "zenit", "dependencies {\n    implementation libs.bundles.flow.stack\n}\n")
    _write(tmp_path / "zenit" / "gradle" / "libs.versions.toml", _CATALOG)
    config = _home(tmp_path)
    graph = config.dependencies

    # 1. A group accessor (`libs.zenit`) and a continued list both resolve, folds and all.
    assert graph.targets_of("zenit-flow") == ("zenit", "protoblast"), "step 1: the group and the continued list"

    # 2. The accessor that produced each coordinate is kept on the reference.
    refs = {project.directory: project.refs for project in discover(tmp_path)}["zenit-flow"]
    assert {ref.accessor for ref in refs} == {"libs.zenit", "libs.protoblast.server", "libs.zenit.test.support"}
    assert {ref.artifact for ref in refs if ref.accessor == "libs.zenit"} == {
        "zenit-common",
        "zenit-client",
        "zenit-server",
    }, "step 2: a group is its direct leaves, never zenit-test-support below it"

    # 3. A bundle expands to its libraries, `_` and `-` separators alike; external ones make no edge.
    assert graph.targets_of("zenit") == ("zenit-flow",), "step 3: the bundle's workspace library"

    # 4. Nothing went unresolved: a non-code configuration is never even looked up.
    assert graph.unresolved == (), "step 4: annotationProcessor accessors are not dependencies"
    assert graph.summary()["edges"] == 3, "step 4: the summary counts the edges"


def test_an_unresolvable_accessor_is_reported_not_dropped(tmp_path: Path) -> None:
    """An accessor the catalog lacks, or a build with no catalog at all, is named in the graph and the answer."""
    _write(tmp_path / "zenit" / "gradle" / "libs.versions.toml", _CATALOG)
    _repo(tmp_path, "zenit", "dependencies {\n    implementation libs.protoblast.client, libs.protoblast.parser\n}\n")
    _repo(tmp_path, "apps/app", "dependencies {\n    serverImplementation libs.zenit.server\n}\n")
    config = _home(tmp_path)
    graph = config.dependencies

    # 1. The alias the catalog has still resolves beside the one it lacks.
    assert graph.targets_of("zenit") == ("protoblast",), "step 1: the known alias is an edge"

    # 2. Both failures are reported, each with its reason and where it was written.
    problems = {(problem.build_file, problem.alias): problem.problem for problem in graph.unresolved}
    assert problems == {
        ("zenit/build.gradle", "protoblast.parser"): AliasProblem.UNKNOWN_ALIAS,
        ("apps/app/build.gradle", "zenit.server"): AliasProblem.NO_CATALOG,
    }, "step 2: an unknown alias and a build no catalog applies to"
    described = graph.summary()["unresolved"]
    assert any("no library, bundle or group" in line for line in described), "step 2: the unknown alias says why"
    assert any("no catalog named 'libs'" in line for line in described), "step 2: the missing catalog says why"

    # 3. The home answer carries the graph's counts and a note naming what is missing.
    answer = decide(config, "anything", [], [])
    assert answer.dependencies is not None and answer.dependencies["edges"] == 1, "step 3: counts in the answer"
    assert any("2 catalog accessor(s)" in note for note in answer.notes), "step 3: the note counts them"
    assert "Dependency graph: 5 modules, 1 edges" in answer.render(), "step 3: the markdown says it too"


def test_a_settings_plugin_catalog_applies_where_the_plugin_is_applied(tmp_path: Path) -> None:
    """A catalog only a settings plugin imports is declared once and reaches exactly the builds applying it."""
    _write(tmp_path / "protoblast" / "versions" / "libs.versions.toml", _CATALOG)
    # The catalog's home imports the file itself, the way a build that cannot apply its own plugin must.
    _repo(
        tmp_path,
        "protoblast",
        "dependencies {\n    implementation libs.jackson.databind\n}\n",
        settings="""
        dependencyResolutionManagement {
            versionCatalogs {
                libs {
                    from(files('versions/libs.versions.toml'))
                }
            }
        }
        """,
    )
    _repo(tmp_path, "zenit", "dependencies {\n    implementation libs.protoblast.server\n}\n", _APPLIES_PLUGIN)
    _repo(tmp_path, "apps/app", "dependencies {\n    implementation libs.zenit\n}\n", _APPLIES_PLUGIN)
    _repo(tmp_path, "apps/other-app", "dependencies {\n    implementation libs.zenit\n}\n")

    # 1. Without the declaration the plugin's catalog is invisible, and every build using it says so.
    undeclared = _home(tmp_path)
    assert undeclared.dependencies.edges == (), "step 1: no edge without a catalog"
    assert {problem.build_file for problem in undeclared.dependencies.unresolved} == {
        "zenit/build.gradle",
        "apps/app/build.gradle",
        "apps/other-app/build.gradle",
    }, "step 1: each build that writes an accessor is reported"
    assert undeclared.dependencies.unresolved[0].problem is AliasProblem.NO_CATALOG, "step 1: no catalog applies"

    # 2. Declared, it applies to the builds whose settings apply the plugin.
    config = _home(tmp_path, _DECLARES_CATALOG)
    graph = config.dependencies
    assert graph.targets_of("zenit") == ("protoblast",), "step 2: the plugin's catalog resolves"
    assert graph.targets_of("app") == ("zenit",), "step 2: an app is a consumer like any other"

    # 3. A build whose settings do not apply the plugin still gets no catalog, and is still reported.
    assert graph.targets_of("other-app") == (), "step 3: no plugin, no catalog"
    assert [(problem.build_file, problem.problem) for problem in graph.unresolved] == [
        ("apps/other-app/build.gradle", AliasProblem.NO_CATALOG)
    ], "step 3: only the build without the plugin is left unresolved"

    # 4. The catalog's own build reads it through its settings file, no declaration needed.
    assert not any(problem.build_file == "protoblast/build.gradle" for problem in undeclared.dependencies.unresolved)

    # 5. Siblings stay unreachable from each other: an app never inherits another app's code.
    assert not config.reachable("app", "other-app").usable, "step 5: sibling apps do not reach each other"


def test_a_declared_catalog_must_exist(tmp_path: Path) -> None:
    """A catalog declaration pointing at no file is a config error, not an empty catalog."""
    with pytest.raises(ConfigError, match="is not a file relative to the workspace root"):
        _home(tmp_path, _DECLARES_CATALOG)


def test_a_forbidden_rule_may_name_several_modules(tmp_path: Path) -> None:
    """Lists on either side of a [[forbidden]] rule refuse every pair except a module to itself."""
    config = _home(
        tmp_path,
        """
        [[forbidden]]
        from = ["app", "other-app"]
        to = ["app", "other-app", "zenit-flow"]
        why = "apps are leaves"
        """,
    )
    pairs = {(rule.source, rule.target) for rule in config.forbidden}
    assert pairs == {
        ("app", "other-app"),
        ("app", "zenit-flow"),
        ("other-app", "app"),
        ("other-app", "zenit-flow"),
    }, "every pair, never a module forbidden from itself"
    with pytest.raises(ConfigError, match="non-empty 'to'"):
        _home(tmp_path, '[[forbidden]]\nfrom = "app"\nto = []\n')
