"""Reading a Gradle workspace's build files for the dependency edges between its modules.

A heuristic text scan, never an evaluation: Gradle build scripts are programs, and
running them to learn which module depends on which is not something a read-only code
search may do. What the scan produces is therefore evidence, not truth - `home.toml`'s
own `depends_on` always overrides it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
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
#: `configurationName libs.protoblast.client`, a version catalog alias.
_CATALOG_REF = re.compile(r"^[ \t]*([A-Za-z][A-Za-z0-9_]*)[ \t]*\(?[ \t]*libs\.([A-Za-z0-9_.]+)", re.M)
#: Where a Gradle build keeps its version catalog, relative to the project directory.
CATALOG_PATH = "gradle/libs.versions.toml"
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

    def to_dict(self) -> dict[str, Any]:
        """Render the reference as JSON-ready data."""
        return {
            "configuration": self.configuration,
            "kind": self.kind.value,
            "project_path": self.project_path,
            "group": self.group,
            "artifact": self.artifact,
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

    def to_dict(self) -> dict[str, Any]:
        """Render the project as JSON-ready data."""
        return {
            "gradle_path": self.gradle_path,
            "name": self.name,
            "directory": self.directory,
            "build_file": self.build_file,
            "refs": [ref.to_dict() for ref in self.refs],
        }


def is_code_configuration(name: str) -> bool:
    """Whether a Gradle configuration name declares a dependency of the module's code."""
    lowered = name.lower()
    return any(lowered.endswith(base) for base in CODE_CONFIGURATIONS)


def discover(root: str | Path, scan_roots: tuple[str, ...] = ()) -> tuple[GradleProject, ...]:
    """Find every Gradle project under a workspace root and the dependencies it declares.

    :param root: The workspace root.
    :param scan_roots: Workspace-relative directories to look in; empty means the whole tree.
    :return: One entry per project that has a build file, in path order.
    """
    base = Path(root)
    projects: dict[str, GradleProject] = {}
    for settings in _settings_files(base, scan_roots):
        for project in _projects_of(base, settings):
            projects.setdefault(project.directory, project)
    for build in _build_files(base, scan_roots):
        directory = _relative(base, build.parent)
        if directory not in projects:
            projects[directory] = GradleProject(
                gradle_path=f":{Path(directory).name}" if directory else ":",
                name=Path(directory).name or base.name,
                directory=directory,
                build_file=_relative(base, build),
            )
    read: list[GradleProject] = []
    for project in projects.values():
        text = _read(base / project.build_file)
        read.append(replace(project, refs=refs_of(text, catalog_of(base / project.directory))))
    read.sort(key=lambda project: project.directory)
    return tuple(read)


def refs_of(text: str, catalog: Mapping[str, tuple[str, str]] | None = None) -> tuple[GradleRef, ...]:
    """Read the dependency references a build script writes, keeping their configuration.

    AIDEV-NOTE: this is a HEURISTIC text scan, not a Gradle evaluation: a dependency built
    from a variable, a loop or a version catalog alias is invisible to it, and a reference
    inside a comment is not. It is therefore evidence a workspace may override - a module
    that declares `depends_on` in `.zemble/home.toml` ignores everything found here.
    """
    found: list[GradleRef] = []
    aliases = catalog or {}
    for configuration, alias in _CATALOG_REF.findall(text):
        coordinate = aliases.get(alias.replace(".", "-"))
        if coordinate is not None and is_code_configuration(configuration):
            found.append(
                GradleRef(
                    configuration=configuration,
                    kind=RefKind.COORDINATE,
                    group=coordinate[0],
                    artifact=coordinate[1],
                )
            )
    for configuration, path in _PROJECT_REF.findall(text):
        if is_code_configuration(configuration):
            found.append(GradleRef(configuration=configuration, kind=RefKind.PROJECT, project_path=_path(path)))
    for configuration, group, artifact in _COORDINATE_REF.findall(text):
        if is_code_configuration(configuration):
            found.append(
                GradleRef(configuration=configuration, kind=RefKind.COORDINATE, group=group, artifact=artifact)
            )
    return tuple(found)


def catalog_of(directory: Path) -> dict[str, tuple[str, str]]:
    """Read a project's version catalog as alias -> (group, artifact).

    An alias a build file writes as `libs.protoblast.client` is the catalog key
    `protoblast-client`; a catalog that cannot be read resolves no alias, which drops the
    edge rather than inventing one.
    """
    try:
        raw = tomllib.loads((directory / CATALOG_PATH).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    libraries = raw.get("libraries")
    if not isinstance(libraries, dict):
        return {}
    found: dict[str, tuple[str, str]] = {}
    for alias, entry in libraries.items():
        coordinate = _coordinate(entry)
        if coordinate is not None:
            found[alias] = coordinate
    return found


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


def _projects_of(base: Path, settings: Path) -> list[GradleProject]:
    """Read one settings file as the root project plus every project it includes."""
    text = _read(settings)
    directory = settings.parent
    root_name = _ROOT_NAME.search(text)
    projects = [
        GradleProject(
            gradle_path=":",
            name=root_name.group(1) if root_name else directory.name,
            directory=_relative(base, directory),
            build_file=_relative(base, _build_file(directory)),
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
    "MAX_DEPTH",
    "SETTINGS_NAMES",
    "SKIPPED_DIRS",
    "GradleProject",
    "GradleRef",
    "RefKind",
    "catalog_of",
    "discover",
    "is_code_configuration",
    "refs_of",
]
