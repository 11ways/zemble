"""Reading a ceiling off the environment, in the one spelling every guard uses.

A knob whose value is nonsense falls back to the default rather than crashing a build, and
the rejected value is LOGGED: a ceiling silently replaced by something else is how a guard
stops guarding without anybody noticing.
"""

from __future__ import annotations

import logging
import math
import os
from typing import overload

logger = logging.getLogger(__name__)


def env_float(name: str, default: float) -> float:
    """Return a finite float named by an environment variable, or the default, loudly.

    A ceiling that is not FINITE is not a ceiling: ``nan`` compares false against everything
    and ``inf`` is never exceeded, so either one would disable the guard reading it while
    looking like a configured value. Both are rejected, as is anything ``float`` refuses.

    :param name: The environment variable holding the value.
    :param default: What an unset or unusable value means.
    :return: The value, or the default.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        logger.warning("Ignoring %s=%r: not a finite number; using %s instead", name, raw, default)
        return default
    return value


@overload
def env_int(name: str, default: int) -> int: ...


@overload
def env_int(name: str, default: None) -> int | None: ...


def env_int(name: str, default: int | None) -> int | None:
    """Return an integer named by an environment variable, or the default, loudly.

    An unreadable value falls back to the default, so a caller that names a real default never
    has to handle None - which is why the overloads above carry the default's own type through.

    :param name: The environment variable holding the value.
    :param default: What an unset or unusable value means; None where nobody named a ceiling.
    :return: The value, or the default.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring %s=%r: not an integer; using %s instead", name, raw, default)
        return default


__all__ = ["env_float", "env_int"]
