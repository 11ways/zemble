"""One `health` answer from an open graph and the workspace config, shared by the CLI, MCP and daemon."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from zemble.graph.provider import AnyProvider
from zemble.health.collect import collect
from zemble.health.model import Signal
from zemble.health.report import DEFAULT_LIMIT, build_report
from zemble.health.signals import measure
from zemble.home.config import HomeConfig


def parse_signals(names: Iterable[str] | None) -> tuple[Signal, ...]:
    """Every signal for None or nothing, else the named ones in declaration order.

    :raises ValueError: If a name is not a signal; the message lists the valid ones.
    """
    wanted = {name.strip().lower() for name in names or () if name.strip()}
    if not wanted:
        return tuple(Signal)
    known = {signal.value for signal in Signal}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"unknown signal(s): {', '.join(unknown)}; known: {', '.join(sorted(known))}")
    return tuple(signal for signal in Signal if signal.value in wanted)


def health_payload(
    provider: AnyProvider,
    config: HomeConfig,
    *,
    signals: Sequence[str] | None = None,
    paths: Sequence[str] = (),
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Measure, score and render one report: `{"health": <json>, "markdown": <text>}`.

    :raises ValueError: If a signal name is unknown.
    """
    selected = parse_signals(signals)
    facts = collect(provider, config, paths)
    report = build_report(
        facts.root,
        str(config.source) if config.source is not None else None,
        measure(facts, selected),
        facts.paths,
    )
    return {"health": report.to_dict(limit), "markdown": report.markdown(limit)}


__all__ = ["health_payload", "parse_signals"]
