"""Which code snapshot this process is running, and whether the checkout moved since."""

from zemble.runtime.identity import (
    GIT_TIMEOUT_SECONDS,
    MAX_SOURCE_FILES,
    RuntimeIdentity,
    git_revision,
    identity,
    stale_note,
    status_payload,
)

__all__ = [
    "GIT_TIMEOUT_SECONDS",
    "MAX_SOURCE_FILES",
    "RuntimeIdentity",
    "git_revision",
    "identity",
    "stale_note",
    "status_payload",
]
