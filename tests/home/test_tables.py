"""Reading a workspace's declared-home markdown table."""

from __future__ import annotations

import textwrap
from pathlib import Path

from zemble.home.config import HomeConfig, TableSpec
from zemble.home.tables import load_rows, match_rows

_CONFIG_BODY = """
    order = ["protoblast", "zenit", "plumage", "zenit-forms", "zenit-cms", "zenit-auth", "hohenheim"]

    [modules]
    protoblast = "protoblast/**"
    zenit = "zenit/**"
    plumage = "plumage/**"
    zenit-forms = "zenit-forms/**"
    zenit-cms = "zenit-cms/**"
    zenit-auth = "zenit-auth/**"
    hohenheim = "hohenheim/**"

    [[tables]]
    file = "ARCH.md"
    capability = "Capability"
    home = "Mechanism home"
    consumers = "Consumers (thin wiring)"
"""


def _workspace(tmp_path: Path) -> HomeConfig:
    """A workspace whose ARCH.md declares homes."""
    (tmp_path / ".zemble").mkdir()
    (tmp_path / ".zemble" / "home.toml").write_text(textwrap.dedent(_CONFIG_BODY).strip() + "\n", encoding="utf-8")
    fixture = Path(__file__).parent.parent / "fixtures" / "home" / "ARCH.md"
    (tmp_path / "ARCH.md").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    return HomeConfig.load(tmp_path)


def test_declared_table_journey(tmp_path: Path) -> None:
    """Walk the parser over a table with backticked names, prose, escaped pipes and a bad row."""
    config = _workspace(tmp_path)
    rows = load_rows(config)

    # 1. Only rows of the matching table, and only the parseable ones, come back.
    assert len(rows) == 4, "step 1: the broken row and the unrelated table are skipped"
    titles = [row.title for row in rows]
    assert titles[0] == "Pagination arithmetic", "step 1: the title is the capability up to its first parenthesis"

    # 2. A home cell yields the backticked names that are modules, and only those.
    pagination = rows[0]
    assert pagination.home_modules == ("zenit",), "step 2: `zenit` is the declared home"
    assert pagination.home_names == ("common/data",), (
        "step 2: a backticked non-module in the home cell is kept verbatim, never read as a module"
    )

    # 3. Consumers are read from backticks AND prose, since the tables spell them both ways.
    assert pagination.consumer_modules == ("zenit-cms",), "step 3: a prose consumer is found"
    assert rows[1].consumer_modules == ("zenit-auth", "hohenheim"), "step 3: several consumers keep order"

    # 4. An escaped pipe inside a code span does not split the row.
    assert "LINE|BYTES" in rows[1].capability, "step 4: the escaped pipe is unescaped, not a cell boundary"

    # 5. Two homes in one cell are both homes.
    assert rows[2].home_modules == ("zenit-forms", "plumage"), "step 5: both declared modules are the home"

    # 6. A backticked name that is not a declared module names no home.
    assert rows[3].home_modules == (), "step 6: an unknown name is not silently a module"
    assert rows[3].home_names == ("SomethingUndeclared",), "step 6: it survives as raw text"

    # 7. The row is JSON-ready and says where it was read.
    assert rows[0].to_dict()["file"] == "ARCH.md", "step 7: the row names its source file"
    assert rows[0].line > 0, "step 7: and its line"


def test_matching_a_description_against_declared_rows(tmp_path: Path) -> None:
    """A description is matched to rows by shared vocabulary, not by luck."""
    config = _workspace(tmp_path)
    rows = load_rows(config)

    matches = match_rows(rows, "I need pagination arithmetic: page offsets for a list")
    assert matches, "the pagination row is found"
    assert matches[0].row.title == "Pagination arithmetic", "the closest row leads"
    assert "pagination" in matches[0].shared, "the answer can say what it matched on"

    assert match_rows(rows, "kubernetes ingress controller") == [], "an unrelated description matches nothing"
    assert match_rows(rows, "the of a") == [], "a description with no meaningful words matches nothing"


def test_a_table_file_that_is_missing_is_skipped(tmp_path: Path) -> None:
    """Documentation that moved must not stop the answer."""
    (tmp_path / ".zemble").mkdir()
    (tmp_path / ".zemble" / "home.toml").write_text(textwrap.dedent(_CONFIG_BODY).strip() + "\n", encoding="utf-8")
    config = HomeConfig.load(tmp_path)
    assert config.tables == (TableSpec("ARCH.md", "Capability", "Mechanism home", "Consumers (thin wiring)"),)
    assert load_rows(config) == [], "a missing table file yields no rows and no exception"
