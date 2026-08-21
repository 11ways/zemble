"""Which fold a path belongs to, and the one table saying which fold may use which."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from zemble.home.config import ConfigError, HomeConfig
from zemble.home.source_sets import COMPATIBILITY, SourceSet, classify, compatible

_CONFIG = """
    order = ["zenit"]

    [modules]
    zenit = "zenit/**"
"""


def _config(root: Path, extra: str = "") -> HomeConfig:
    """A one-module workspace, optionally with its own source-set globs."""
    (root / ".zemble").mkdir(parents=True, exist_ok=True)
    (root / ".zemble" / "home.toml").write_text(
        textwrap.dedent(_CONFIG).strip() + "\n" + textwrap.dedent(extra), encoding="utf-8"
    )
    return HomeConfig.load(root)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("zenit/src/common/java/be/Texts.java", SourceSet.COMMON),
        ("src/common/java/be/Texts.java", SourceSet.COMMON),
        ("zenit/src/server/java/be/Router.java", SourceSet.SERVER),
        ("zenit/src/browser/java/be/Dom.java", SourceSet.BROWSER),
        ("zenit/src/client/java/be/Dom.java", SourceSet.BROWSER),
        ("zenit/src/test/java/be/TextsTest.java", SourceSet.TEST),
        ("zenit/src/browserTest/java/be/DomTest.java", SourceSet.TEST),
        ("zenit/src/main/java/be/Plain.java", SourceSet.UNKNOWN),
        ("README.md", SourceSet.UNKNOWN),
    ],
)
def test_a_path_is_classified_by_its_fold(path: str, expected: SourceSet) -> None:
    """Each fold is read off the path, and a test source set is a test before it is a fold."""
    assert classify(path) is expected


@pytest.mark.parametrize(
    ("consumer", "provider", "allowed"),
    [
        (SourceSet.COMMON, SourceSet.COMMON, True),
        (SourceSet.COMMON, SourceSet.SERVER, False),
        (SourceSet.COMMON, SourceSet.BROWSER, False),
        (SourceSet.SERVER, SourceSet.COMMON, True),
        (SourceSet.SERVER, SourceSet.SERVER, True),
        (SourceSet.SERVER, SourceSet.BROWSER, False),
        (SourceSet.BROWSER, SourceSet.COMMON, True),
        (SourceSet.BROWSER, SourceSet.BROWSER, True),
        (SourceSet.BROWSER, SourceSet.SERVER, False),
        (SourceSet.TEST, SourceSet.SERVER, True),
        (SourceSet.TEST, SourceSet.BROWSER, True),
        (SourceSet.TEST, SourceSet.UNKNOWN, True),
        (SourceSet.UNKNOWN, SourceSet.UNKNOWN, True),
        (SourceSet.UNKNOWN, SourceSet.COMMON, False),
        (SourceSet.COMMON, SourceSet.UNKNOWN, False),
    ],
)
def test_every_row_of_the_compatibility_table(consumer: SourceSet, provider: SourceSet, allowed: bool) -> None:
    """The whole table, one case per pair that has to hold."""
    assert compatible(consumer, provider) is allowed


def test_the_table_covers_the_vocabulary() -> None:
    """A fold nobody wrote a row for must break the build, not fall through to "allowed"."""
    assert set(COMPATIBILITY) == set(SourceSet), "every source set has exactly one row"


def test_a_workspace_may_declare_its_own_folds(tmp_path: Path) -> None:
    """Declared globs replace the defaults, and a fold zemble has no name for is refused."""
    config = _config(
        tmp_path,
        extra="""
        [source_sets]
        common = ["shared/**"]
        server = ["backend/**"]
        """,
    )
    assert config.source_set_of("zenit/shared/Texts.java") is SourceSet.COMMON, "the declared glob wins"
    assert config.source_set_of("zenit/src/common/Texts.java") is SourceSet.UNKNOWN, "the default is replaced"
    assert config.source_set_compatible("zenit/backend/Router.java", "zenit/shared/Texts.java"), "server uses common"
    assert not config.source_set_compatible("zenit/shared/Texts.java", "zenit/backend/Router.java"), "not the reverse"

    with pytest.raises(ConfigError, match="is not a source set"):
        _config(tmp_path, extra='\n[source_sets]\nnative = ["src/native/**"]\n')


def test_the_default_folds_answer_a_javaweb_shaped_path(tmp_path: Path) -> None:
    """A workspace that declares no globs still classifies the javaweb layout."""
    config = _config(tmp_path)
    assert config.source_set_of("zenit/src/common/java/X.java") is SourceSet.COMMON
    assert not config.source_set_compatible("zenit/src/common/java/X.java", "zenit/src/server/java/Y.java"), (
        "a common class may not reach into the server fold"
    )
    assert config.source_set_compatible("zenit/src/test/java/T.java", "zenit/src/server/java/Y.java"), (
        "a test may use anything"
    )
