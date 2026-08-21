"""What a workspace declares about its own module boundaries.

Every fact `zemble home` uses about a workspace - which modules exist, which of
them sits closer to the core, which dependencies are forbidden, which markdown
tables already declare a home - lives in `<root>/.zemble/home.toml`, never in
zemble itself. A workspace without one is answered generically and told so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from functools import cached_property
from pathlib import Path
from typing import Any

from zemble.home.deps import DependencyGraph, DependencySource, Reachability, build_graph
from zemble.home.source_sets import SourceSet, classify, compatible
from zemble.workspace import HOME_CONFIG_RELATIVE_PATH

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10, which the package still supports
    tomllib = None  # type: ignore[assignment]

#: Backwards-compatible name for the workspace declaration path.
CONFIG_RELATIVE_PATH = HOME_CONFIG_RELATIVE_PATH

#: The module name reported for a file that sits at the workspace root.
ROOT_MODULE = "<root>"


class ConfigError(ValueError):
    """A `home.toml` that cannot be trusted.

    Raised loudly rather than degraded to generic mode: a config that is present but
    wrong is a mistake to fix, while an absent one is a deliberate choice.
    """


@dataclass(frozen=True)
class ForbiddenRule:
    """A dependency the workspace refuses: `source` may not depend on `target`."""

    source: str
    target: str
    why: str = ""

    def describe(self) -> str:
        """One line naming the refusal and, where it was given, its reason."""
        head = f"{self.source} must not depend on {self.target}"
        return f"{head}: {self.why}" if self.why else head

    def to_dict(self) -> dict[str, Any]:
        """Render the rule as JSON-ready data."""
        return {"from": self.source, "to": self.target, "why": self.why}


@dataclass(frozen=True)
class TableSpec:
    """A markdown table in the workspace that DECLARES capability homes."""

    file: str
    capability: str
    home: str
    consumers: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render the spec as JSON-ready data."""
        return {"file": self.file, "capability": self.capability, "home": self.home, "consumers": self.consumers}


@dataclass(frozen=True)
class WorkspaceRule:
    """A free-text rule echoed in an answer, optionally scoped to some modules."""

    text: str
    modules: tuple[str, ...] = ()

    def applies_to(self, modules: tuple[str, ...]) -> bool:
        """Return True when this rule concerns any of the given modules, or is unscoped."""
        return not self.modules or any(module in self.modules for module in modules)

    def to_dict(self) -> dict[str, Any]:
        """Render the rule as JSON-ready data."""
        return {"text": self.text, "modules": list(self.modules)}


@dataclass(frozen=True)
class HomeConfig:
    """The workspace knowledge one `home` answer is allowed to rely on."""

    root: Path
    #: Modules closest to the core first. A pure RANKING: being earlier in this list
    #: says a module is closer to the core, never that anyone may depend on it.
    order: tuple[str, ...] = ()
    #: Module name -> the globs matching its files.
    module_globs: dict[str, tuple[str, ...]] = None  # type: ignore[assignment]
    forbidden: tuple[ForbiddenRule, ...] = ()
    #: Module name -> the modules it declares a dependency on; overrides discovery.
    depends_on: dict[str, tuple[str, ...]] = None  # type: ignore[assignment]
    #: Which lanes may contribute dependency edges.
    dependency_source: DependencySource = DependencySource.BOTH
    #: Directories to scan for build files; empty means the whole workspace.
    gradle_roots: tuple[str, ...] = ()
    #: Source set -> the path globs declaring it; empty means the built-in defaults.
    source_set_globs: dict[str, tuple[str, ...]] = None  # type: ignore[assignment]
    tables: tuple[TableSpec, ...] = ()
    skills: dict[str, tuple[str, ...]] = None  # type: ignore[assignment]
    rules: tuple[WorkspaceRule, ...] = ()
    #: The file this came from; None means nothing was declared.
    source: Path | None = None

    def __post_init__(self) -> None:
        """Give the mutable defaults a value without sharing one dict between configs."""
        if self.module_globs is None:
            object.__setattr__(self, "module_globs", {})
        if self.skills is None:
            object.__setattr__(self, "skills", {})
        if self.depends_on is None:
            object.__setattr__(self, "depends_on", {})
        if self.source_set_globs is None:
            object.__setattr__(self, "source_set_globs", {})

    @property
    def generic(self) -> bool:
        """Whether this workspace declared nothing, so modules are guessed from paths."""
        return self.source is None

    @property
    def modules(self) -> tuple[str, ...]:
        """Every module name the workspace declared, ordered, globs-only ones last."""
        extra = [name for name in self.module_globs if name not in self.order]
        return (*self.order, *sorted(extra))

    def module_of(self, file_path: str) -> str:
        """Return the module a workspace-relative path belongs to.

        Falls back to the first path segment, which is the right answer for a workspace
        of sibling repositories and an honest guess everywhere else.
        """
        relative = file_path.replace("\\", "/").lstrip("./")
        for name, globs in self.module_globs.items():
            if any(fnmatch(relative, pattern) for pattern in globs):
                return name
        head, separator, _ = relative.partition("/")
        return head if separator else ROOT_MODULE

    def rank(self, module: str) -> int:
        """Return a module's distance from the core; an undeclared module ranks last."""
        try:
            return self.order.index(module)
        except ValueError:
            return len(self.order)

    def known_module(self, name: str) -> str | None:
        """Resolve a name as written to a declared module, case-insensitively."""
        cleaned = name.strip().strip("`")
        if not cleaned:
            return None
        for module in self.modules:
            if module == cleaned or module.lower() == cleaned.lower():
                return module
        return None

    def forbids(self, consumer: str, home: str) -> ForbiddenRule | None:
        """Return the rule broken by `consumer` depending on `home`, if there is one."""
        for rule in self.forbidden:
            if rule.source == consumer and rule.target == home:
                return rule
        return None

    @cached_property
    def dependencies(self) -> DependencyGraph:
        """The module dependency graph, built once from the declarations and the build files.

        Built lazily because discovery walks the workspace for build files: a config that
        is only asked for its modules never pays for it.
        """
        return build_graph(
            self.root,
            self.modules,
            self.module_of,
            self.depends_on,
            forbidden=[(rule.source, rule.target) for rule in self.forbidden],
            source=self.dependency_source,
            gradle_roots=self.gradle_roots,
        )

    def reachable(self, consumer: str, home: str) -> Reachability:
        """Answer whether a module may use code that lives in another one."""
        return self.dependencies.reachable(consumer, home)

    def nearest_common_dependency(self, modules: Sequence[str]) -> str | None:
        """Return the highest-ranked module every given module may depend on, if any."""
        return self.dependencies.nearest_common_dependency(modules, self.rank)

    def source_set_of(self, file_path: str) -> SourceSet:
        """Return the fold of a module a workspace-relative path is compiled into."""
        patterns = {SourceSet(name): globs for name, globs in self.source_set_globs.items()}
        return classify(file_path, patterns or None)

    def source_set_compatible(self, consumer_path: str, provider_path: str) -> bool:
        """Whether code at one path may use code at another, by their folds alone."""
        return compatible(self.source_set_of(consumer_path), self.source_set_of(provider_path))

    def skills_for(self, module: str) -> tuple[str, ...]:
        """Return the skills to read before designing inside a module."""
        return self.skills.get(module, ())

    def to_dict(self) -> dict[str, Any]:
        """Render the configuration as JSON-ready data."""
        return {
            "root": str(self.root),
            "source": str(self.source) if self.source else None,
            "generic": self.generic,
            "order": list(self.order),
            "modules": {name: list(globs) for name, globs in self.module_globs.items()},
            "forbidden": [rule.to_dict() for rule in self.forbidden],
            "depends_on": {name: list(targets) for name, targets in self.depends_on.items()},
            "dependency_source": self.dependency_source.value,
            "source_sets": {name: list(globs) for name, globs in self.source_set_globs.items()},
            "tables": [table.to_dict() for table in self.tables],
            "skills": {name: list(values) for name, values in self.skills.items()},
            "rules": [rule.to_dict() for rule in self.rules],
        }

    @classmethod
    def load(cls, root: str | Path) -> HomeConfig:
        """Read a workspace's `home.toml`, or return generic mode when it has none.

        :param root: The workspace root.
        :return: The parsed configuration.
        :raises ConfigError: If the file exists but is malformed.
        """
        base = Path(root)
        path = base / CONFIG_RELATIVE_PATH
        if not path.is_file():
            return cls(root=base)
        if tomllib is None:  # pragma: no cover - Python 3.10 only
            raise ConfigError(f"{path}: reading it needs Python 3.11 or newer (tomllib)")
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            raise ConfigError(f"{path}: {error}") from error
        return cls._parse(base, path, raw)

    @classmethod
    def _parse(cls, root: Path, path: Path, raw: dict[str, Any]) -> HomeConfig:
        """Turn a decoded document into a configuration, refusing anything unexpected."""
        modules_section = _table(raw, "modules", path)
        # `order` is written either at the top level or inside [modules]; a workspace
        # that spells it the second way cannot also have a module called "order".
        order_raw = raw.get("order", modules_section.pop("order", None))
        order = tuple(_strings(order_raw, "order", path)) if order_raw is not None else ()
        module_globs: dict[str, tuple[str, ...]] = {}
        depends_on: dict[str, tuple[str, ...]] = {}
        for name, value in modules_section.items():
            globs, declared_deps = _module_entry(value, name, path)
            module_globs[name] = globs
            if declared_deps is not None:
                depends_on[name] = declared_deps
        for name in module_globs:
            if not module_globs[name]:
                raise ConfigError(f"{path}: modules.{name} declares no globs")
        dependency_source, gradle_roots = _dependencies(_table(raw, "dependencies", path), path)
        source_set_globs = _source_sets(_table(raw, "source_sets", path), path)
        forbidden = tuple(_forbidden(entry, index, path) for index, entry in enumerate(_array(raw, "forbidden", path)))
        tables = tuple(_table_spec(entry, index, path) for index, entry in enumerate(_array(raw, "tables", path)))
        skills = {
            name: tuple(_strings(value, f"skills.{name}", path)) for name, value in _table(raw, "skills", path).items()
        }
        rules = tuple(_rule(entry, index, path) for index, entry in enumerate(_array(raw, "rules", path)))
        unknown = set(raw) - {
            "order", "modules", "forbidden", "tables", "skills", "rules", "dependencies", "source_sets"
        }  # fmt: skip
        if unknown:
            raise ConfigError(f"{path}: unknown section(s): {', '.join(sorted(unknown))}")
        config = cls(
            root=root,
            order=order,
            module_globs=module_globs,
            forbidden=forbidden,
            depends_on=depends_on,
            dependency_source=dependency_source,
            gradle_roots=gradle_roots,
            source_set_globs=source_set_globs,
            tables=tables,
            skills=skills,
            rules=rules,
            source=path,
        )
        _check_names(config, path)
        return config


def _check_names(config: HomeConfig, path: Path) -> None:
    """Refuse a config that names a module nothing else declares."""
    known = set(config.modules)
    for rule in config.forbidden:
        _known(rule.source, known, "forbidden rule names", path)
        _known(rule.target, known, "forbidden rule names", path)
    for name, targets in config.depends_on.items():
        _known(name, known, "depends_on declared for", path)
        for target in targets:
            _known(target, known, f"modules.{name}.depends_on names", path)
    for name in config.skills:
        _known(name, known, "skills names", path)
    for rule in config.rules:
        for name in rule.modules:
            _known(name, known, "rule scope names", path)


def _known(name: str, known: set[str], what: str, path: Path) -> None:
    """Refuse one name that is not a declared module."""
    if name not in known:
        raise ConfigError(f"{path}: {what} undeclared module {name!r}")


def _module_entry(value: Any, name: str, path: Path) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    """Read one [modules] entry as its globs and, where it declares them, its dependencies.

    A module is written either as its globs (`zenit = "zenit/**"`) or as a table
    (`[modules.zenit] globs = [...]`, `depends_on = [...]`). Declaring `depends_on`, even
    as an empty list, REPLACES Gradle discovery for that module.
    """
    if isinstance(value, dict):
        unknown = set(value) - {"globs", "depends_on"}
        if unknown:
            raise ConfigError(f"{path}: modules.{name} has unknown key(s): {', '.join(sorted(unknown))}")
        if "globs" not in value:
            raise ConfigError(f"{path}: modules.{name} declares no globs")
        globs = tuple(_strings(value["globs"], f"modules.{name}.globs", path))
        declared = value.get("depends_on")
        deps = tuple(_strings(declared, f"modules.{name}.depends_on", path)) if declared is not None else None
        return globs, deps
    return tuple(_strings(value, f"modules.{name}", path)), None


def _dependencies(section: dict[str, Any], path: Path) -> tuple[DependencySource, tuple[str, ...]]:
    """Read the [dependencies] section: which lanes may contribute edges, and where to scan."""
    unknown = set(section) - {"source", "gradle_roots"}
    if unknown:
        raise ConfigError(f"{path}: [dependencies] has unknown key(s): {', '.join(sorted(unknown))}")
    written = section.get("source", DependencySource.BOTH.value)
    try:
        source = DependencySource(written)
    except ValueError as error:
        allowed = ", ".join(member.value for member in DependencySource)
        raise ConfigError(f"{path}: [dependencies] source must be one of {allowed}, got {written!r}") from error
    roots = section.get("gradle_roots")
    return source, tuple(_strings(roots, "dependencies.gradle_roots", path)) if roots is not None else ()


def _source_sets(section: dict[str, Any], path: Path) -> dict[str, tuple[str, ...]]:
    """Read the [source_sets] section, refusing a fold zemble has no name for."""
    folds: dict[str, tuple[str, ...]] = {}
    for name, value in section.items():
        try:
            fold = SourceSet(name)
        except ValueError as error:
            allowed = ", ".join(member.value for member in SourceSet)
            raise ConfigError(f"{path}: [source_sets] {name!r} is not a source set ({allowed})") from error
        folds[fold.value] = tuple(_strings(value, f"source_sets.{name}", path))
    return folds


def _table(raw: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    """Read one table section, refusing a value of another shape."""
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: [{key}] must be a table, got {type(value).__name__}")
    return dict(value)


def _array(raw: dict[str, Any], key: str, path: Path) -> list[Any]:
    """Read one array-of-tables section, refusing a value of another shape."""
    value = raw.get(key, [])
    if not isinstance(value, list):
        raise ConfigError(f"{path}: [[{key}]] must be an array of tables, got {type(value).__name__}")
    for entry in value:
        if not isinstance(entry, dict):
            raise ConfigError(f"{path}: every [[{key}]] entry must be a table, got {type(entry).__name__}")
    return value


def _strings(value: Any, key: str, path: Path) -> list[str]:
    """Read a string or a list of strings as a list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ConfigError(f"{path}: {key} must be a string or a list of strings")


def _required(entry: dict[str, Any], key: str, where: str, path: Path) -> str:
    """Read a required string field of an array entry."""
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: {where} needs a non-empty {key!r}")
    return value.strip()


def _forbidden(entry: dict[str, Any], index: int, path: Path) -> ForbiddenRule:
    """Parse one [[forbidden]] entry."""
    where = f"[[forbidden]] #{index + 1}"
    why = entry.get("why", "")
    if not isinstance(why, str):
        raise ConfigError(f"{path}: {where} 'why' must be a string")
    return ForbiddenRule(
        source=_required(entry, "from", where, path), target=_required(entry, "to", where, path), why=why.strip()
    )


def _table_spec(entry: dict[str, Any], index: int, path: Path) -> TableSpec:
    """Parse one [[tables]] entry."""
    where = f"[[tables]] #{index + 1}"
    consumers = entry.get("consumers")
    if consumers is not None and not isinstance(consumers, str):
        raise ConfigError(f"{path}: {where} 'consumers' must be a string")
    return TableSpec(
        file=_required(entry, "file", where, path),
        capability=_required(entry, "capability", where, path),
        home=_required(entry, "home", where, path),
        consumers=consumers.strip() if isinstance(consumers, str) else None,
    )


def _rule(entry: dict[str, Any], index: int, path: Path) -> WorkspaceRule:
    """Parse one [[rules]] entry."""
    where = f"[[rules]] #{index + 1}"
    modules = entry.get("modules", [])
    return WorkspaceRule(
        text=_required(entry, "text", where, path), modules=tuple(_strings(modules, f"{where} modules", path))
    )


__all__ = [
    "CONFIG_RELATIVE_PATH",
    "ROOT_MODULE",
    "ConfigError",
    "DependencyGraph",
    "DependencySource",
    "ForbiddenRule",
    "HomeConfig",
    "Reachability",
    "SourceSet",
    "TableSpec",
    "WorkspaceRule",
]
