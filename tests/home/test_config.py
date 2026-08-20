"""What a workspace declares about its modules, and what happens when it declares nonsense."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from zemble.home.config import CONFIG_RELATIVE_PATH, ROOT_MODULE, ConfigError, HomeConfig

_FULL = """
    order = ["protoblast", "hawkeye", "zenit", "zenit-cms"]

    [modules]
    protoblast = "protoblast/**"
    hawkeye = ["hawkeye/**", "hawkeye-core/**"]
    zenit = "zenit/**"
    zenit-cms = "zenit-cms/**"

    [[forbidden]]
    from = "zenit-cms"
    to = "zenit"
    why = "the cms never reaches back into core internals"

    [[tables]]
    file = "CLAUDE.md"
    capability = "Capability"
    home = "Mechanism home"
    consumers = "Consumers"

    [skills]
    zenit-cms = ["zenit-cms-resources"]
    zenit = ["zenit-framework", "zenit-forms-editing"]

    [[rules]]
    text = "Nothing lands without at least one wired consumer and a test"

    [[rules]]
    text = "Generated admin pages are the floor, not the ceiling"
    modules = ["zenit-cms"]
"""


def _write(root: Path, body: str) -> Path:
    """Write a home.toml into a workspace root."""
    path = root / CONFIG_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
    return path


def test_config_journey(tmp_path: Path) -> None:
    """Walk a full config from parsing to every question the answer asks it."""
    path = _write(tmp_path, _FULL)
    config = HomeConfig.load(tmp_path)

    # 1. Every section survives the parse.
    assert config.source == path, "step 1: the config remembers where it came from"
    assert not config.generic, "step 1: a declared workspace is not generic"
    assert config.order == ("protoblast", "hawkeye", "zenit", "zenit-cms"), "step 1: order is kept as written"
    assert config.module_globs["hawkeye"] == ("hawkeye/**", "hawkeye-core/**"), "step 1: a list of globs is kept"
    assert config.module_globs["zenit"] == ("zenit/**",), "step 1: a lone glob becomes a one-element tuple"
    assert [rule.describe() for rule in config.forbidden] == [
        "zenit-cms must not depend on zenit: the cms never reaches back into core internals"
    ], "step 1: a forbidden rule renders as one sentence"
    assert config.tables[0].file == "CLAUDE.md", "step 1: the declared table is named"
    assert config.tables[0].consumers == "Consumers", "step 1: an optional consumers column is kept"

    # 2. A path resolves to the module whose glob matches it.
    assert config.module_of("hawkeye-core/src/main/java/X.java") == "hawkeye", "step 2: globs beat path segments"
    assert config.module_of("plumage/src/Button.java") == "plumage", "step 2: undeclared falls back to the segment"
    assert config.module_of("README.md") == ROOT_MODULE, "step 2: a root file has no module"

    # 3. Distance from the core is the position in `order`; unknown modules rank last.
    assert config.rank("protoblast") < config.rank("zenit"), "step 3: protoblast is closer to the core"
    assert config.rank("plumage") == len(config.order), "step 3: an undeclared module ranks last"

    # 4. Names as written resolve case-insensitively, and only to declared modules.
    assert config.known_module("`Zenit-CMS`") == "zenit-cms", "step 4: backticks and case are stripped"
    assert config.known_module("SessionCookies") is None, "step 4: a class name is not a module"

    # 5. Forbidden dependencies and skills answer by module.
    assert config.forbids("zenit-cms", "zenit") is not None, "step 5: the declared refusal is found"
    assert config.forbids("zenit", "zenit-cms") is None, "step 5: the refusal has a direction"
    assert config.skills_for("zenit") == ("zenit-framework", "zenit-forms-editing"), "step 5: skills are per module"

    # 6. Rule scoping: an unscoped rule applies everywhere, a scoped one only to its modules.
    unscoped, scoped = config.rules
    assert unscoped.applies_to(("plumage",)), "step 6: an unscoped rule always applies"
    assert scoped.applies_to(("zenit-cms",)), "step 6: a scoped rule applies to its module"
    assert not scoped.applies_to(("zenit",)), "step 6: and to nothing else"

    # 7. The whole thing is JSON-ready for the wire.
    assert config.to_dict()["order"] == list(config.order), "step 7: to_dict carries the order"


def test_missing_config_is_generic_mode(tmp_path: Path) -> None:
    """A workspace that declares nothing is answered from path segments alone."""
    config = HomeConfig.load(tmp_path)
    assert config.generic, "an absent config is generic mode, not an error"
    assert config.order == () and config.tables == (), "nothing is assumed about the workspace"
    assert config.module_of("zenit/src/Foo.java") == "zenit", "the first path segment is the module"


def test_order_may_be_written_inside_the_modules_table(tmp_path: Path) -> None:
    """`order` under [modules] is the same key as `order` at the top level."""
    _write(
        tmp_path,
        """
        [modules]
        order = ["a", "b"]
        a = "a/**"
        b = "b/**"
    """,
    )
    config = HomeConfig.load(tmp_path)
    assert config.order == ("a", "b"), "the [modules] spelling is accepted"
    assert "order" not in config.module_globs, "and never becomes a module of its own"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("[modules\nzenit = 1", "Expected"),
        ("[modules]\nzenit = 1\n", "must be a string or a list of strings"),
        ('modules = "nope"', "must be a table"),
        ('[modules]\nzenit = "zenit/**"\n[[forbidden]]\nfrom = "zenit"', "needs a non-empty 'to'"),
        ('[modules]\nzenit = "zenit/**"\n[[tables]]\nfile = "x.md"\ncapability = "C"', "needs a non-empty 'home'"),
        ('[modules]\nzenit = "zenit/**"\n[[forbidden]]\nfrom = "ghost"\nto = "zenit"', "undeclared module 'ghost'"),
        ('[modules]\nzenit = "zenit/**"\n[skills]\nghost = ["x"]', "skills names undeclared module"),
        ('[modules]\nzenit = "zenit/**"\n[[rules]]\ntext = "t"\nmodules = ["ghost"]', "rule scope names undeclared"),
        ("[modules]\nzenit = []", "declares no globs"),
        ("[whatever]\nx = 1", "unknown section"),
    ],
)
def test_a_malformed_config_is_loud(tmp_path: Path, body: str, message: str) -> None:
    """Every shape of nonsense refuses rather than quietly becoming generic mode."""
    _write(tmp_path, body)
    with pytest.raises(ConfigError, match=message):
        HomeConfig.load(tmp_path)
