"""What `zemble health` reads from a workspace: frozen code and the findings the workspace explains once.

Every key of a `home.toml` `[health]` section is declared here once; an unknown key, an unknown signal, a rule
without paths or without a reason is a configuration error, never ignored. Frozen code has one home, the
`[dupes] frozen` globs (`zemble.dedup.settings`), read here as well: nobody edits it, so no finding on it is one
anyone can act on.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any

from zemble.dedup.settings import ShapeSettings
from zemble.health.model import Signal
from zemble.home.config import ConfigError, HomeConfig

if TYPE_CHECKING:  # pragma: no cover
    from zemble.health.signals import Sample

#: The one key of `[health]`: an array of tables, each one `Exemption`.
EXEMPT_KEY = "exempt"
#: The keys of one `[[health.exempt]]` entry.
_ENTRY_KEYS = frozenset({"paths", "signals", "why"})
#: Why frozen code is measured by nothing, as the report notes it.
FROZEN_REASON = "frozen code ([dupes] frozen)"


def matches(file_path: str, patterns: tuple[str, ...]) -> bool:
    """Whether a workspace-relative path matches one of some globs, `*` crossing directories as `[modules]` globs do."""
    relative = file_path.replace("\\", "/").lstrip("./")
    return any(fnmatchcase(relative, pattern) for pattern in patterns)


@dataclass(frozen=True)
class Exemption:
    """Paths some signals (every signal when none is named) do not measure, for a reason the report repeats."""

    paths: tuple[str, ...]
    signals: tuple[Signal, ...]
    why: str

    @property
    def whole(self) -> bool:
        """Whether the exemption takes its files out of every signal of every tier."""
        return not self.signals

    def covers(self, file_path: str) -> bool:
        """Whether a workspace-relative path is one of the exemption's."""
        return matches(file_path, self.paths)

    def note(self, files: int, units: Mapping[Signal, int]) -> str:
        """The report note: what the rule took out, or that it matched nothing (a stale rule stays visible)."""
        if self.whole:
            taken = f"{files} file(s) not measured" if files else "matched no file"
        elif units:
            counted = ", ".join(f"{count} {signal.value}" for signal, count in units.items())
            taken = f"{counted} unit(s) not measured"
        else:
            taken = "matched no unit"
        return f"exempt ({self.why}): {taken}"


@dataclass(frozen=True)
class HealthSettings:
    """One workspace's frozen globs and exemptions."""

    shapes: ShapeSettings = ShapeSettings()
    exemptions: tuple[Exemption, ...] = ()

    def is_frozen(self, file_path: str) -> bool:
        """Whether a workspace-relative path is frozen code."""
        return self.shapes.is_frozen(file_path)

    def whole_exemption(self, file_path: str) -> Exemption | None:
        """The first exemption taking a workspace-relative path out of every signal, if any."""
        return next((rule for rule in self.exemptions if rule.whole and rule.covers(file_path)), None)

    def exemption_of(self, signal: Signal, file_path: str) -> Exemption | None:
        """The first exemption taking one signal off a workspace-relative path, if any."""
        return next(
            (rule for rule in self.exemptions if signal in rule.signals and rule.covers(file_path)),
            None,
        )

    def apply(
        self, samples: dict[Signal, list[Sample]], respell: Callable[[str], str]
    ) -> tuple[dict[Signal, list[Sample]], dict[Exemption, Counter[Signal]]]:
        """Drop the samples a signal exemption covers, counting what each rule took out.

        :param respell: Maps a sample's path (relative to the reported directory) onto the workspace the
            config was declared for, where the globs are written.
        """
        taken: dict[Exemption, Counter[Signal]] = {rule: Counter() for rule in self.exemptions if not rule.whole}
        if not taken:
            return samples, taken
        kept: dict[Signal, list[Sample]] = {}
        for signal, found in samples.items():
            kept[signal] = []
            for sample in found:
                rule = self.exemption_of(signal, respell(sample.file_path))
                if rule is None:
                    kept[signal].append(sample)
                else:
                    taken[rule][signal] += 1
        return kept, taken

    @classmethod
    def of(cls, config: HomeConfig) -> HealthSettings:
        """The settings a loaded configuration declares.

        :raises ConfigError: If `[health]` names an unknown key, an entry an unknown signal, or `[dupes]` is wrong.
        """
        unknown = sorted(set(config.health) - {EXEMPT_KEY})
        if unknown:
            raise ConfigError(f"[health] unknown key(s): {', '.join(unknown)} (known: {EXEMPT_KEY})")
        entries = config.health.get(EXEMPT_KEY, [])
        if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
            raise ConfigError(f"[[health.{EXEMPT_KEY}]] must be an array of tables")
        return cls(ShapeSettings.of(config), tuple(_exemption(entry, index) for index, entry in enumerate(entries)))


def _exemption(entry: dict[str, Any], index: int) -> Exemption:
    """One `[[health.exempt]]` entry, refusing anything it does not declare."""
    where = f"[[health.{EXEMPT_KEY}]] #{index + 1}"
    unknown = sorted(set(entry) - _ENTRY_KEYS)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s): {', '.join(unknown)} (known: {', '.join(sorted(_ENTRY_KEYS))})")
    paths = tuple(_strings(entry.get("paths"), f"{where} paths"))
    if not paths:
        raise ConfigError(f"{where}: needs at least one path glob in 'paths'")
    why = entry.get("why")
    if not isinstance(why, str) or not why.strip():
        raise ConfigError(f"{where}: needs a non-empty 'why' (the report repeats it)")
    names = _strings(entry.get("signals", []), f"{where} signals")
    known = {signal.value: signal for signal in Signal}
    unknown_signals = sorted(set(names) - set(known))
    if unknown_signals:
        raise ConfigError(f"{where}: unknown signal(s): {', '.join(unknown_signals)} (known: {', '.join(known)})")
    return Exemption(paths, tuple(signal for signal in Signal if signal.value in names), why.strip())


def _strings(value: Any, key: str) -> list[str]:
    """Read a string or a list of strings as a list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ConfigError(f"{key} must be a string or a list of strings")


__all__ = ["EXEMPT_KEY", "FROZEN_REASON", "Exemption", "HealthSettings", "matches"]
