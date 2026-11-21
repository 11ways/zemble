"""What the embedding server bought from its providers, per day, client key and model: the bill, attributable."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from zemble.embedding.cache import BUSY_TIMEOUT_SECONDS, today

#: The ledger's file in the server's data directory. AIDEV-NOTE: deliberately not `*.sqlite`: the
#: garbage collection and the orphan sweep treat every `*.sqlite` there as a family's vector cache.
SPEND_FILE = "spend.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spend (
    day INTEGER NOT NULL,
    label TEXT NOT NULL,
    model TEXT NOT NULL,
    kind TEXT NOT NULL,
    texts INTEGER NOT NULL,
    tokens INTEGER NOT NULL,
    PRIMARY KEY (day, label, model, kind)
) WITHOUT ROWID;
"""


class SpendKind(str, Enum):
    """What a provider was paid for."""

    DOCUMENTS = "documents"
    QUERIES = "queries"
    RERANK = "rerank"


@dataclass(frozen=True)
class SpendRow:
    """One day's purchases of one kind, by one client key, from one model."""

    day: int
    label: str
    model: str
    kind: SpendKind
    texts: int
    #: Estimated from characters (`zemble.embedding.pricing.estimate_tokens`), never the provider's count.
    tokens: int


class SpendLedger:
    """A small sqlite table of purchases, one upsert per paid provider call."""

    def __init__(self, directory: Path) -> None:
        """Open (creating if needed) the ledger in a server's data directory."""
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / SPEND_FILE
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None, timeout=BUSY_TIMEOUT_SECONDS
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)

    def add(self, label: str, model: str, kind: SpendKind, texts: int, tokens: int) -> None:
        """Count one purchase into today's row; nothing bought writes nothing."""
        if texts <= 0:
            return
        with self._lock:
            self._connection.execute(
                "INSERT INTO spend (day, label, model, kind, texts, tokens) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(day, label, model, kind) DO UPDATE SET "
                "texts = texts + excluded.texts, tokens = tokens + excluded.tokens",
                (today(), label, model, kind.value, texts, tokens),
            )

    def since(self, day: int) -> list[SpendRow]:
        """Return every row from *day* (a day number) on, oldest first."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT day, label, model, kind, texts, tokens FROM spend WHERE day >= ? "
                "ORDER BY day, label, model, kind",
                (day,),
            ).fetchall()
        return [
            SpendRow(int(row[0]), str(row[1]), str(row[2]), SpendKind(row[3]), int(row[4]), int(row[5])) for row in rows
        ]

    def close(self) -> None:
        """Close the connection."""
        with self._lock:
            self._connection.close()


__all__ = ["SPEND_FILE", "SpendKind", "SpendLedger", "SpendRow"]
