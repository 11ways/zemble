"""Reading a Gradle workspace's build files for the dependency edges between its modules.

A heuristic text scan, never an evaluation: Gradle build scripts are programs, and
running them to learn which module depends on which is not something a read-only code
search may do. What the scan produces is therefore evidence, not truth - `home.toml`'s
own `depends_on` always overrides it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import tomllib

#: Settings files that declare a Gradle build and the projects it includes.
SETTINGS_NAMES = ("settings.gradle", "settings.gradle.kts")
#: Build files that declare one project's dependencies.
BUILD_NAMES = ("build.gradle", "build.gradle.kts")

#: Directories never walked: build output, caches and version control.
SKIPPED_DIRS = frozenset({"build", "out", "target", "node_modules", ".git", ".gradle", ".idea", "bin"})
#: How deep below the workspace root a build file is still looked for.
MAX_DEPTH = 4

#: The dependency configurations a code edge may be declared in.
#:
#: Matched as a SUFFIX, case-insensitively, so every source-set-prefixed variant Gradle
#: generates (`commonImplementation`, `serverApi`, `browserTestCompileOnly`) is covered by
#: the base name it ends with. A configuration matching none of these declares no edge:
#: `annotationProcessor`, `checkstyle` and every plugin-specific configuration are build
#: tooling, not something the module's code may reach into.
CODE_CONFIGURATIONS = ("implementation", "api", "compileonly", "compileonlyapi", "runtimeonly")

#: `configurationName project(':name')` or `configurationName(project(":name"))`.
_PROJECT_REF = re.compile(r"^[ \t]*([A-Za-z][A-Za-z0-9_]*)[ \t]*\(?[ \t]*project\([ \t]*['\"]([^'\"]+)['\"]", re.M)
#: `configurationName 'group:artifact:version'`, the shape a workspace of sibling repos
#: publishing to mavenLocal uses instead of a project reference.
_COORDINATE_REF = re.compile(
    r"^[ \t]*([A-Za-z][A-Za-z0-9_]*)[ \t]*\(?[ \t]*['\"]([A-Za-z0-9_.\-]+):([A-Za-z0-9_.\-]+):", re.M
)
#: `configurationName libs.a.b, libs.c`: one or more `catalog.alias` accessors after a configuration. A list
#: continues across lines after a trailing comma, line comments included, the way Groovy reads it.
_ACCESSOR_LIST = re.compile(
    r"^[ \t]*([A-Za-z][A-Za-z0-9_]*)[ \t]*\(?[ \t]*"
    r"([A-Za-z][A-Za-z0-9_]*\.[A-Za-z0-9_.]+(?:[ \t]*,(?:\s|//[^\n]*)*[A-Za-z][A-Za-z0-9_]*\.[A-Za-z0-9_.]+)*)",
    re.M,
)
#: A `//` line comment inside such a list.
_LINE_COMMENT = re.compile(r"//[^\n]*")
#: One `catalog.alias` accessor inside such a list.
_ACCESSOR = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\.([A-Za-z0-9_.]+)")
#: The catalog name Gradle gives its conventional catalog, and the one every build may use.
DEFAULT_CATALOG = "libs"
#: Where Gradle looks for the conventional catalog, relative to the settings directory.
CATALOG_PATH = "gradle/libs.versions.toml"
#: `name { from(files('path')) }` or `create("name") { from(files("path")) }` in `versionCatalogs`.
_SETTINGS_CATALOG = re.compile(
    r"(?:(?:create|register)\([ \t]*['\"]([A-Za-z][A-Za-z0-9_]*)['\"][ \t]*\)|([A-Za-z][A-Za-z0-9_]*))"
    r"[ \t]*\{\s*from[ \t]*\(?[ \t]*(?:files\([ \t]*['\"]([^'\"]+)['\"]|['\"]([^'\"]+)['\"])"
)
#: What separates the segments of a catalog alias; Gradle treats all three alike.
_ALIAS_SEPARATORS = re.compile(r"[-_.]")
#: `id 'plugin.id'` / `id("plugin.id")`, a plugin a settings file applies.
_PLUGIN_ID = re.compile(r"\bid[ \t]*\(?[ \t]*['\"]([^'\"]+)['\"]")
#: `rootProject.name = 'zenit-flow'`.
_ROOT_NAME = re.compile(r"rootProject\.name[ \t]*=[ \t]*['\"]([^'\"]+)['\"]")
#: `include ':a:b'` / `include("a", "b")`, one quoted path per match.
_INCLUDE = re.compile(r"^[ \t]*include[ \t]*\(?([^\n)]*)\)?", re.M)
_QUOTED = re.compile(r"['\"]([^'\"]+)['\"]")
#: `project(':a').projectDir = file('../elsewhere')`.
_PROJECT_DIR = re.compile(
    r"project\([ \t]*['\"]([^'\"]+)['\"][ \t]*\)\.projectDir[ \t]*=[ \t]*(?:new File\()?file\([ \t]*['\"]([^'\"]+)['\"]"
)


class RefKind(str, Enum):
    """How a build file wrote the dependency it declares."""

    PROJECT = "project"
    COORDINATE = "coordinate"


@dataclass(frozen=True)
class GradleRef:
    """One dependency a build file declares, as written."""

    configuration: str
    kind: RefKind
    #: The Gradle project path (`:zenit-flow`) for a PROJECT ref, else "".
    project_path: str = ""
    #: The group and artifact for a COORDINATE ref, else "".
    group: str = ""
    artifact: str = ""
    #: The catalog accessor the coordinate was resolved from (`libs.zenit.cms`), else "".
    accessor: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Render the reference as JSON-ready data."""
        return {
            "configuration": self.configuration,
            "kind": self.kind.value,
            "project_path": self.project_path,
            "group": self.group,
            "artifact": self.artifact,
            "accessor": self.accessor,
        }


@dataclass(frozen=True)
class CatalogDeclaration:
    """A version catalog a SETTINGS PLUGIN applies, declared by the workspace because no build file names it.

    A settings plugin imports its catalog from compiled code (`catalog.from(coordinate)`), which a text scan
    cannot follow; `home.toml` names the catalog's source file and the plugin id that applies it, and the
    catalog then applies to exactly the builds whose settings file applies that plugin.
    """

    plugin: str
    #: Workspace-relative path of the catalog's TOML source.
    path: str
    name: str = DEFAULT_CATALOG

    def to_dict(self) -> dict[str, Any]:
        """Render the declaration as JSON-ready data."""
        return {"plugin": self.plugin, "path": self.path, "name": self.name}


@dataclass(frozen=True)
class VersionCatalog:
    """One version catalog as a build sees it: its accessor name and its libraries and bundles."""

    name: str
    #: Workspace-relative path of the TOML source, or the coordinate a settings file imports it from.
    source: str
    #: False when the source is not a file this scan can read (a published coordinate, a missing file).
    readable: bool
    #: Normalised alias (`zenit.cms.server`) -> (group, artifact).
    libraries: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    #: Normalised bundle name -> the normalised library aliases it holds.
    bundles: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def resolve(self, alias: str) -> tuple[tuple[str, str], ...] | None:
        """Resolve an accessor's alias part (`zenit.cms`, `bundles.web`) to the coordinates it stands for.

        AIDEV-NOTE: an alias that names no library but is a GROUP of them (`libs.hawkeye`, while the catalog
        holds `hawkeye-common`, `-client` and `-server`) resolves to the group's DIRECT leaves only, never to
        everything below it: `libs.zenit` is zenit's own trio, not every `zenit-*` module. Gradle itself would
        refuse a bare group as a dependency; build DSLs that take module trios (protoblast's `blastModule`)
        are exactly the scripts that write one.

        :return: The coordinates, or None when the catalog holds nothing by that name.
        """
        key = normalise_alias(alias)
        if key in self.libraries:
            return (self.libraries[key],)
        if key.startswith("bundles."):
            members = self.bundles.get(key[len("bundles.") :])
            if members is not None:
                return tuple(self.libraries[member] for member in members if member in self.libraries)
        prefix = f"{key}."
        leaves = [
            coordinate
            for alias_key, coordinate in self.libraries.items()
            if alias_key.startswith(prefix) and "." not in alias_key[len(prefix) :]
        ]
        return tuple(leaves) if leaves else None


class AliasProblem(str, Enum):
    """Why a catalog accessor in a build file resolved to no coordinate."""

    NO_CATALOG = "no-catalog"
    UNREADABLE_CATALOG = "unreadable-catalog"
    UNKNOWN_ALIAS = "unknown-alias"


@dataclass(frozen=True)
class UnresolvedAlias:
    """A catalog accessor a build file uses in a code configuration that the scan could not resolve.

    Reported rather than dropped: an unresolved accessor is a dependency edge the graph is missing.
    """

    build_file: str
    configuration: str
    catalog: str
    alias: str
    problem: AliasProblem
    #: The catalog source consulted, "" when no catalog of that name applies.
    source: str = ""

    def describe(self) -> str:
        """One line naming the accessor, where it was written and why it resolved to nothing."""
        head = f"{self.build_file}: {self.configuration} {self.catalog}.{self.alias}"
        if self.problem is AliasProblem.NO_CATALOG:
            return f"{head}: no catalog named {self.catalog!r} applies to this build"
        if self.problem is AliasProblem.UNREADABLE_CATALOG:
            return f"{head}: catalog {self.catalog!r} comes from {self.source}, which this scan cannot read"
        if self.problem is AliasProblem.UNKNOWN_ALIAS:
            return f"{head}: catalog {self.catalog!r} ({self.source}) has no library, bundle or group by that name"
        raise ValueError(f"unhandled alias problem: {self.problem!r}")

    def to_dict(self) -> dict[str, Any]:
        """Render the problem as JSON-ready data."""
        return {
            "build_file": self.build_file,
            "configuration": self.configuration,
            "catalog": self.catalog,
            "alias": self.alias,
            "problem": self.problem.value,
            "source": self.source,
            "describe": self.describe(),
        }


@dataclass(frozen=True)
class GradleProject:
    """One Gradle project found under the workspace root."""

    #: Gradle project path, `:` for a build's root project.
    gradle_path: str
    name: str
    #: Workspace-relative directory, "" for the workspace root itself.
    directory: str
    build_file: str
    refs: tuple[GradleRef, ...] = ()
    #: Workspace-relative directory of the settings file that includes the project, None when none does.
    settings_dir: str | None = None
    unresolved: tuple[UnresolvedAlias, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Render the project as JSON-ready data."""
        return {
            "gradle_path": self.gradle_path,
            "name": self.name,
            "directory": self.directory,
            "build_file": self.build_file,
            "refs": [ref.to_dict() for ref in self.refs],
            "settings_dir": self.settings_dir,
            "unresolved": [problem.to_dict() for problem in self.unresolved],
        }


def is_code_configuration(name: str) -> bool:
    """Whether a Gradle configuration name declares a dependency of the module's code."""
    lowered = name.lower()
    return any(lowered.endswith(base) for base in CODE_CONFIGURATIONS)


def discover(
    root: str | Path, scan_roots: tuple[str, ...] = (), declared_catalogs: Sequence[CatalogDeclaration] = ()
) -> tuple[GradleProject, ...]:
    """Find every Gradle project under a workspace root and the dependencies it declares.

    :param root: The workspace root.
    :param scan_roots: Workspace-relative directories to look in; empty means the whole tree.
    :param declared_catalogs: Catalogs settings plugins apply, which no build file names.
    :return: One entry per project that has a build file, in path order.
    """
    base = Path(root)
    projects: dict[str, GradleProject] = {}
    cache: dict[Path, VersionCatalog] = {}
    catalogs: dict[str, dict[str, VersionCatalog]] = {}
    for settings in _settings_files(base, scan_roots):
        text = _read(settings)
        catalogs[_relative(base, settings.parent)] = settings_catalogs(
            base, settings.parent, text, declared_catalogs, cache
        )
        for project in _projects_of(base, settings, text):
            projects.setdefault(project.directory, project)
    for build in _build_files(base, scan_roots):
        directory = _relative(base, build.parent)
        if directory not in projects:
            projects[directory] = GradleProject(
                gradle_path=f":{Path(directory).name}" if directory else ":",
                name=Path(directory).name or base.name,
                directory=directory,
                build_file=_relative(base, build),
                settings_dir=_nearest_settings(directory, catalogs),
            )
    read: list[GradleProject] = []
    for project in projects.values():
        if project.settings_dir is not None:
            applied = catalogs[project.settings_dir]
        else:
            applied = settings_catalogs(base, base / project.directory, "", (), cache)
        scan = scan_build(_read(base / project.build_file), project.build_file, applied)
        read.append(replace(project, refs=scan.refs, unresolved=scan.unresolved))
    read.sort(key=lambda project: project.directory)
    return tuple(read)


@dataclass(frozen=True)
class BuildScan:
    """What one build script's text says it depends on, and the catalog accessors that said nothing."""

    refs: tuple[GradleRef, ...] = ()
    unresolved: tuple[UnresolvedAlias, ...] = ()


def scan_build(text: str, build_file: str = "", catalogs: Mapping[str, VersionCatalog] | None = None) -> BuildScan:
    """Read the dependency references a build script writes, keeping their configuration.

    AIDEV-NOTE: this is a HEURISTIC text scan, not a Gradle evaluation: a dependency built
    from a variable or a loop is invisible to it, and a reference inside a block comment is
    not. It is therefore evidence a workspace may override - a module that declares
    `depends_on` in `.zemble/home.toml` ignores everything found here. A `name.alias`
    accessor is read as a catalog reference only when `name` is a catalog this build has, or
    is `libs`, the name every Gradle build may use; a `libs` accessor nothing resolves is
    REPORTED, never dropped, because it is an edge the graph is missing.

    :param text: The build script.
    :param build_file: Its workspace-relative path, for the report.
    :param catalogs: Catalog name -> the catalog this build sees under it.
    """
    found: list[GradleRef] = []
    unresolved: list[UnresolvedAlias] = []
    applied = catalogs or {}
    for configuration, accessors in _ACCESSOR_LIST.findall(text):
        if not is_code_configuration(configuration):
            continue
        for name, alias in _ACCESSOR.findall(_LINE_COMMENT.sub("", accessors)):
            if name not in applied and name != DEFAULT_CATALOG:
                continue
            refs, problem = _resolve_accessor(applied.get(name), name, alias, configuration, build_file)
            found.extend(refs)
            if problem is not None:
                unresolved.append(problem)
    for configuration, path in _PROJECT_REF.findall(text):
        if is_code_configuration(configuration):
            found.append(GradleRef(configuration=configuration, kind=RefKind.PROJECT, project_path=_path(path)))
    for configuration, group, artifact in _COORDINATE_REF.findall(text):
        if is_code_configuration(configuration):
            found.append(
                GradleRef(configuration=configuration, kind=RefKind.COORDINATE, group=group, artifact=artifact)
            )
    return BuildScan(refs=tuple(found), unresolved=tuple(unresolved))


def _resolve_accessor(
    catalog: VersionCatalog | None, name: str, alias: str, configuration: str, build_file: str
) -> tuple[list[GradleRef], UnresolvedAlias | None]:
    """Resolve one `name.alias` accessor to coordinate refs, or to the reason it resolves to nothing."""

    def problem(kind: AliasProblem) -> UnresolvedAlias:
        return UnresolvedAlias(
            build_file=build_file,
            configuration=configuration,
            catalog=name,
            alias=alias,
            problem=kind,
            source=catalog.source if catalog is not None else "",
        )

    if catalog is None:
        return [], problem(AliasProblem.NO_CATALOG)
    if not catalog.readable:
        return [], problem(AliasProblem.UNREADABLE_CATALOG)
    coordinates = catalog.resolve(alias)
    if coordinates is None:
        return [], problem(AliasProblem.UNKNOWN_ALIAS)
    accessor = f"{name}.{alias}"
    return [
        GradleRef(
            configuration=configuration, kind=RefKind.COORDINATE, group=group, artifact=artifact, accessor=accessor
        )
        for group, artifact in coordinates
    ], None


def settings_catalogs(
    base: Path,
    settings_dir: Path,
    text: str,
    declared: Sequence[CatalogDeclaration] = (),
    cache: dict[Path, VersionCatalog] | None = None,
) -> dict[str, VersionCatalog]:
    """The catalogs one settings file gives its builds, by accessor name.

    Three sources, the first to name a catalog winning: the settings file's own
    `versionCatalogs { name { from(...) } }`, a workspace-declared catalog of a settings plugin
    the file applies, and Gradle's convention file `gradle/libs.versions.toml` as `libs`.

    :param base: The workspace root, which declared catalog paths are relative to.
    :param settings_dir: The directory holding the settings file (or the project, when none does).
    :param text: The settings file's text, "" when there is none.
    :param declared: Catalogs settings plugins apply, from `home.toml`.
    :param cache: Catalogs already read, by resolved path.
    """
    memo = cache if cache is not None else {}
    found: dict[str, VersionCatalog] = {}
    start = text.find("versionCatalogs")
    for created, named, path, coordinate in _SETTINGS_CATALOG.findall(text[start:] if start >= 0 else ""):
        name = created or named
        if name in found:
            continue
        if path:
            found[name] = read_catalog(name, settings_dir / path, base, memo)
        else:
            found[name] = VersionCatalog(name=name, source=coordinate, readable=False)
    applied = set(_PLUGIN_ID.findall(text))
    for declaration in declared:
        if declaration.plugin in applied and declaration.name not in found:
            found[declaration.name] = read_catalog(declaration.name, base / declaration.path, base, memo)
    convention = settings_dir / CATALOG_PATH
    if DEFAULT_CATALOG not in found and convention.is_file():
        found[DEFAULT_CATALOG] = read_catalog(DEFAULT_CATALOG, convention, base, memo)
    return found


def read_catalog(name: str, path: Path, base: Path, cache: dict[Path, VersionCatalog] | None = None) -> VersionCatalog:
    """Read a version catalog TOML file under an accessor name; an unreadable file is a catalog that says so.

    Aliases are normalised the way Gradle builds accessors from them: `-`, `_` and `.` all
    separate segments, so the catalog key `zenit-cms_server` is the accessor `zenit.cms.server`.
    """
    resolved = path.resolve()
    memo = cache if cache is not None else {}
    if resolved not in memo:
        memo[resolved] = _parse_catalog(path, base)
    return replace(memo[resolved], name=name)


def normalise_alias(alias: str) -> str:
    """Return a catalog alias or accessor in Gradle's accessor form, segments joined by dots."""
    return _ALIAS_SEPARATORS.sub(".", alias)


def _parse_catalog(path: Path, base: Path) -> VersionCatalog:
    """Parse one catalog file, unnamed; `read_catalog` names it."""
    source = _relative(base, path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return VersionCatalog(name="", source=source, readable=False)
    libraries: dict[str, tuple[str, str]] = {}
    section = raw.get("libraries")
    for alias, entry in (section if isinstance(section, dict) else {}).items():
        coordinate = _coordinate(entry)
        if coordinate is not None:
            libraries[normalise_alias(alias)] = coordinate
    bundles: dict[str, tuple[str, ...]] = {}
    section = raw.get("bundles")
    for alias, members in (section if isinstance(section, dict) else {}).items():
        if isinstance(members, list):
            bundles[normalise_alias(alias)] = tuple(normalise_alias(m) for m in members if isinstance(m, str))
    return VersionCatalog(name="", source=source, readable=True, libraries=libraries, bundles=bundles)


def _nearest_settings(directory: str, settings_dirs: Mapping[str, Any]) -> str | None:
    """The deepest settings directory at or above a project directory, None when there is none."""
    parts = directory.split("/") if directory else []
    for depth in range(len(parts), -1, -1):
        candidate = "/".join(parts[:depth])
        if candidate in settings_dirs:
            return candidate
    return None


def _coordinate(entry: Any) -> tuple[str, str] | None:
    """Read one catalog entry as its group and artifact."""
    if isinstance(entry, str):
        parts = entry.split(":")
        return (parts[0], parts[1]) if len(parts) >= 2 else None
    if isinstance(entry, dict):
        module = entry.get("module")
        if isinstance(module, str) and ":" in module:
            group, _, artifact = module.partition(":")
            return group, artifact
        group, artifact = entry.get("group"), entry.get("name")
        if isinstance(group, str) and isinstance(artifact, str):
            return group, artifact
    return None


def _path(written: str) -> str:
    """Normalise a Gradle project path to its leading-colon form."""
    cleaned = written.strip()
    return cleaned if cleaned.startswith(":") else f":{cleaned}"


def _settings_files(base: Path, scan_roots: tuple[str, ...]) -> list[Path]:
    """Every settings file under the workspace root."""
    return _walk(base, scan_roots, SETTINGS_NAMES)


def _build_files(base: Path, scan_roots: tuple[str, ...]) -> list[Path]:
    """Every build file under the workspace root."""
    return _walk(base, scan_roots, BUILD_NAMES)


def _walk(base: Path, scan_roots: tuple[str, ...], names: tuple[str, ...]) -> list[Path]:
    """Collect the named files below the roots to scan, pruning build output and caches."""
    starts = [base / entry for entry in scan_roots] if scan_roots else [base]
    found: list[Path] = []
    for start in starts:
        if not start.is_dir():
            continue
        start_depth = len(start.parts)
        for directory, subdirectories, files in os.walk(start):
            here = Path(directory)
            if len(here.parts) - start_depth >= MAX_DEPTH:
                subdirectories[:] = []
            subdirectories[:] = sorted(
                name for name in subdirectories if name not in SKIPPED_DIRS and not name.startswith(".")
            )
            found.extend(here / name for name in names if name in files)
    return sorted(set(found))


def _projects_of(base: Path, settings: Path, text: str) -> list[GradleProject]:
    """Read one settings file as the root project plus every project it includes."""
    directory = settings.parent
    settings_dir = _relative(base, directory)
    root_name = _ROOT_NAME.search(text)
    projects = [
        GradleProject(
            gradle_path=":",
            name=root_name.group(1) if root_name else directory.name,
            directory=_relative(base, directory),
            build_file=_relative(base, _build_file(directory)),
            settings_dir=settings_dir,
        )
    ]
    overrides = {_path(path): value for path, value in _PROJECT_DIR.findall(text)}
    for line in _INCLUDE.findall(text):
        for written in _QUOTED.findall(line):
            gradle_path = _path(written)
            override = overrides.get(gradle_path)
            child = (
                (directory / override).resolve()
                if override
                else directory.joinpath(*gradle_path.lstrip(":").split(":"))
            )
            projects.append(
                GradleProject(
                    gradle_path=gradle_path,
                    name=gradle_path.rsplit(":", 1)[-1],
                    directory=_relative(base, child),
                    build_file=_relative(base, _build_file(child)),
                    settings_dir=settings_dir,
                )
            )
    return [project for project in projects if (base / project.build_file).is_file()]


def _build_file(directory: Path) -> Path:
    """Return the build file of a project directory, the Groovy name when neither exists."""
    for name in BUILD_NAMES:
        if (directory / name).is_file():
            return directory / name
    return directory / BUILD_NAMES[0]


def _relative(base: Path, path: Path) -> str:
    """Return a path relative to the workspace root, with forward slashes."""
    try:
        relative = path.resolve().relative_to(base.resolve())
    except ValueError:
        return str(path).replace("\\", "/")
    return str(relative).replace("\\", "/") if str(relative) != "." else ""


def _read(path: Path) -> str:
    """Read a build script, or return an empty string when it cannot be read."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


__all__ = [
    "BUILD_NAMES",
    "CATALOG_PATH",
    "CODE_CONFIGURATIONS",
    "DEFAULT_CATALOG",
    "MAX_DEPTH",
    "SETTINGS_NAMES",
    "SKIPPED_DIRS",
    "AliasProblem",
    "BuildScan",
    "CatalogDeclaration",
    "GradleProject",
    "GradleRef",
    "RefKind",
    "UnresolvedAlias",
    "VersionCatalog",
    "discover",
    "is_code_configuration",
    "normalise_alias",
    "read_catalog",
    "scan_build",
    "settings_catalogs",
]
