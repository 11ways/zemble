"""What a workspace declares about its own module boundaries.

Every fact `zemble home` uses about a workspace - which modules exist, which of
them sits closer to the core, which dependencies are forbidden, which markdown
tables already declare a home - lives in `<root>/.zemble/home.toml`, never in
zemble itself. A workspace without one is answered generically and told so.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

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
    #: Modules closest to the core first. Ranking is by position in this list.
    order: tuple[str, ...] = ()
    #: Module name -> the globs matching its files.
    module_globs: dict[str, tuple[str, ...]] = None  # type: ignore[assignment]
    forbidden: tuple[ForbiddenRule, ...] = ()
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
        module_globs = {
            name: tuple(_strings(value, f"modules.{name}", path)) for name, value in modules_section.items()
        }
        for name in module_globs:
            if not module_globs[name]:
                raise ConfigError(f"{path}: modules.{name} declares no globs")
        forbidden = tuple(_forbidden(entry, index, path) for index, entry in enumerate(_array(raw, "forbidden", path)))
        tables = tuple(_table_spec(entry, index, path) for index, entry in enumerate(_array(raw, "tables", path)))
        skills = {
            name: tuple(_strings(value, f"skills.{name}", path)) for name, value in _table(raw, "skills", path).items()
        }
        rules = tuple(_rule(entry, index, path) for index, entry in enumerate(_array(raw, "rules", path)))
        unknown = set(raw) - {"order", "modules", "forbidden", "tables", "skills", "rules"}
        if unknown:
            raise ConfigError(f"{path}: unknown section(s): {', '.join(sorted(unknown))}")
        config = cls(
            root=root,
            order=order,
            module_globs=module_globs,
            forbidden=forbidden,
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
        for name in (rule.source, rule.target):
            if name not in known:
                raise ConfigError(f"{path}: forbidden rule names undeclared module {name!r}")
    for name in config.skills:
        if name not in known:
            raise ConfigError(f"{path}: skills names undeclared module {name!r}")
    for rule in config.rules:
        for name in rule.modules:
            if name not in known:
                raise ConfigError(f"{path}: rule scope names undeclared module {name!r}")


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
    "ForbiddenRule",
    "HomeConfig",
    "TableSpec",
    "WorkspaceRule",
]
