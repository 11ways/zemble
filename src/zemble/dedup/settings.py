"""What the sub-body and vocabulary channels read from a workspace: frozen code, catalog keys and source sets.

Every key of a `home.toml` `[dupes]` section is declared here once, with the default a workspace without one gets;
an unknown key is a configuration error, never ignored.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path

from zemble.dedup.languages import KeyArguments
from zemble.home.config import ConfigError, HomeConfig
from zemble.home.source_sets import SourceSet, classify, compatible


class SettingKey(str, Enum):
    """The keys of the `[dupes]` section, each a list of strings."""

    #: Path globs of frozen code: shipped once and never edited again (schema migrations), so a literal there is the
    #: value as it was, never a stray copy of a live constant, and repeating a migration DSL is its design.
    FROZEN = "frozen"
    #: Key arguments of copy and translation calls (`<receiver glob>.<method>:<index>`); a string there is a key.
    COPY_KEYS = "copy_keys"

    @property
    def default(self) -> tuple[str, ...]:
        """What a workspace that does not set this key gets."""
        return _DEFAULTS[self]


_DEFAULTS: Mapping[SettingKey, tuple[str, ...]] = {
    SettingKey.FROZEN: ("*/migration/*", "*/migrations/*", "*/db/migrate/*"),
    SettingKey.COPY_KEYS: (
        "*Microcopy*.of:0",
        "*Microcopy*.literal:0",
        "*Microcopy*.withArg:0",
        "*Microcopy*.withFilter:0",
        "*ResourceBundle*.getString:0",
        "*essageSource*.getMessage:0",
    ),
}


@dataclass(frozen=True)
class ShapeSettings:
    """One workspace's frozen globs, key-argument patterns and source-set globs."""

    frozen: tuple[str, ...] = SettingKey.FROZEN.default
    copy_keys: tuple[str, ...] = SettingKey.COPY_KEYS.default
    #: Source set -> path globs, as `home.toml` declares them; empty takes the built-in defaults.
    source_sets: Mapping[SourceSet, tuple[str, ...]] = field(default_factory=dict)

    def is_frozen(self, file_path: str) -> bool:
        """Whether a root-relative path is frozen code."""
        rooted = "/" + file_path.replace("\\", "/").lstrip("./")
        return any(fnmatchcase(rooted, pattern) for pattern in self.frozen)

    def source_set(self, file_path: str) -> SourceSet:
        """The fold a root-relative path compiles into."""
        return _classified(file_path, tuple(sorted(self.source_sets.items())))

    def reaches(self, consumer_path: str, provider_path: str) -> bool:
        """Whether code at one path may read a declaration at another, by their source sets alone.

        AIDEV-NOTE: an unclassified side (a plain `src/main` module) says nothing here, so it reaches and is
        reached; `home` reads the same folds strictly, because it recommends, while this only refuses.
        """
        consumer, provider = self.source_set(consumer_path), self.source_set(provider_path)
        if SourceSet.UNKNOWN in (consumer, provider):
            return True
        return compatible(consumer, provider)

    @classmethod
    def load(cls, root: str | Path) -> tuple[ShapeSettings, list[str]]:
        """Read the root's `home.toml`; a broken one (an unknown `[dupes]` key too) is noted and the defaults hold."""
        try:
            return cls.of(HomeConfig.load(root)), []
        except ConfigError as error:
            return cls(), [f"home.toml error, dupes settings are the defaults: {error}"]

    @classmethod
    def of(cls, config: HomeConfig) -> ShapeSettings:
        """The settings a loaded configuration declares, defaults for every key it leaves out.

        :raises ConfigError: If `[dupes]` names an unknown key or a malformed key-argument pattern.
        """
        known = {key.value: key for key in SettingKey}
        unknown = sorted(set(config.dupes) - set(known))
        if unknown:
            raise ConfigError(f"[dupes] unknown key(s): {', '.join(unknown)} (known: {', '.join(sorted(known))})")
        values = {key: config.dupes.get(key.value, key.default) for key in SettingKey}
        try:
            KeyArguments(values[SettingKey.COPY_KEYS])
        except ValueError as error:
            raise ConfigError(f"[dupes] {error}") from error
        folds = {SourceSet(name): globs for name, globs in config.source_set_globs.items()}
        return cls(values[SettingKey.FROZEN], values[SettingKey.COPY_KEYS], folds)


@lru_cache(maxsize=65536)
def _classified(file_path: str, patterns: tuple[tuple[SourceSet, tuple[str, ...]], ...]) -> SourceSet:
    """`classify`, once per path and pattern set."""
    return classify(file_path, dict(patterns) or None)
