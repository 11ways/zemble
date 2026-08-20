from __future__ import annotations

import subprocess
import sys

_PROBE = """
import sys
import zemble
import zemble.cli
import zemble.rerank
from zemble.rerank.registry import parse_reranker_spec

parse_reranker_spec("cross:cross-encoder/ms-marco-MiniLM-L-6-v2")
leaked = sorted(name for name in ("torch", "transformers") if name in sys.modules)
print(",".join(leaked))
"""


def test_importing_zemble_does_not_import_torch() -> None:
    """Importing zemble, its CLI, and building a cross-encoder spec must not pull in torch."""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "", "the heavy extra is loaded lazily, on the first score() call"
