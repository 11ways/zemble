"""Pinned historical declarations and a group-isolated, never-tuned holdout."""

import json
from collections import Counter
from pathlib import Path

from benchmarks.historical_recall import duplicate_matches, rank

DATASET = Path(__file__).parents[1] / "benchmarks/local/historical_recall_pairs.json"


def test_pinned_pairs_have_exact_provenance_and_group_isolated_splits():
    """The evaluator cannot silently lose pairs, query text or original source identity."""
    dataset = json.loads(DATASET.read_text())
    pairs = dataset["pairs"]
    assert len(pairs) == len({pair["id"] for pair in pairs}) == 42
    assert Counter(pair["split"] for pair in pairs) == {"development": 30, "holdout": 12}
    groups = {}
    for pair in pairs:
        assert pair["query"] and pair["old"] and pair["replacement"] and pair["targets"]
        assert len(pair["source"]["revision"]) == 40
        assert pair["source"]["line"] > 0
        assert pair["source"]["file"].startswith(pair["source"]["repo"] + "/")
        assert groups.setdefault(pair["group"], pair["split"]) == pair["split"]


def test_scoring_requires_canonical_source_not_copy_mentions_or_one_sided_clones():
    """An unrelated table mention or clone of the removed wrapper is never a recall hit."""
    target = {"file_path": "core/canonical.java", "line": 50}
    assert rank([{"file_path": target["file_path"], "start_line": 1, "end_line": 20}], [target]) is None
    assert rank([{"file_path": target["file_path"], "start_line": 45, "end_line": 55}], [target]) == 1
    pair = {"source": {"file": "host/old.java", "line": 10, "member": "old"}, "targets": [target]}
    old = {"file_path": "host/old.java", "start_line": 8, "end_line": 12}
    replacement = {"file_path": "core/canonical.java", "start_line": 45, "end_line": 55}
    assert duplicate_matches(pair, {"classes": [{"key": "one-sided", "members": [old]}]}) == []
    assert duplicate_matches(pair, {"classes": [{"key": "pair", "members": [old, replacement]}]}) == ["pair"]
