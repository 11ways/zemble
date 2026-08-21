"""Behaviour journeys over the language profiles duplication detection is driven by."""

from pathlib import Path

import pytest
from semble_grammars import get_language

from zemble.dedup.detect import DupeOptions, find_duplication
from zemble.dedup.languages import PROFILES, Visibility, body_unit_kinds, profile_for, supported_extensions
from zemble.dedup.model import BODY_KINDS, WINDOW_KIND, CloneKind
from zemble.dedup.report import format_report, report_json
from zemble.dedup.units import extract_units
from zemble.index.files import get_extensions
from zemble.types import ContentType

ZIG_FIXTURES = Path(__file__).parent / "fixtures" / "dedup_zig"


def _zig_units(name: str) -> list:
    """Extract the whole-body units of one Zig fixture file."""
    path = ZIG_FIXTURES / "src" / name
    return extract_units(path.read_bytes(), f"src/{name}", windows=False)


def _members(report, kind: CloneKind) -> list[set[str]]:
    """The member names of every class of one kind."""
    return [{member.name for member in clone.members} for clone in report.of_kind(kind)]


def test_zig_journey() -> None:
    """Zig files become units the same way Java ones do: top level, nested, and no false clone."""
    # 1. A top-level function and one nested in a `const Foo = struct` are both members.
    names = {unit.name for unit in _zig_units("alpha.zig")} | {unit.name for unit in _zig_units("beta.zig")}
    assert "sumAll" in names, "step 1: a free function is a member of the file struct"
    assert "Widget.fill" in names and "Helper.sumAll" in names, "step 1: a nested struct qualifies its members"

    # 2. Visibility is reported on the unit and is deliberately absent from every hash.
    alpha = {unit.name: unit for unit in _zig_units("alpha.zig")}
    assert alpha["sumAll"].modifiers == ("pub",), "step 2: `pub` is a modifier"
    assert alpha["scaleValues"].modifiers == (), "step 2: a private function has none"
    assert alpha["sumAll"].visibility is Visibility.PUBLIC, "step 2: and `pub` is what makes it public"
    assert alpha["scaleValues"].visibility is Visibility.PRIVATE, "step 2: everything else is file-private"
    assert alpha["sumAll"].container_visibility is Visibility.PUBLIC, "step 2: the file struct hides nothing"
    assert alpha["Widget.fill"].visibility is Visibility.PUBLIC, "step 2: the member itself is `pub`"
    assert alpha["Widget.fill"].container_visibility is Visibility.PRIVATE, "step 2: but `const Widget` is not"

    # 3. Calls come off both plain calls and method calls; builtins and control flow are seen.
    assert alpha["Widget.fill"].calls == ("alloc", "print"), "step 3: `alloc.alloc` calls `alloc`"
    assert alpha["Widget.fill"].skeleton == ("try", "while", "return"), "step 3: try is control flow"

    # 4. The exact copy and the alpha-renamed copy are found, across files and nesting levels.
    report = find_duplication(ZIG_FIXTURES, DupeOptions(kinds=(CloneKind.EXACT, CloneKind.RENAMED), windows=False))
    assert report.analyzed_files == 2 and report.failed_files == 0, "step 4: both files parsed"
    assert _members(report, CloneKind.EXACT) == [{"sumAll", "Helper.sumAll"}], "step 4: the exact copy is found"
    assert _members(report, CloneKind.RENAMED) == [{"scaleValues", "scaleAmounts"}], "step 4: locals normalize"

    # 5. The near miss differs in one call only, and is reported by neither kind.
    grouped = _members(report, CloneKind.EXACT) + _members(report, CloneKind.RENAMED)
    clustered = {name for members in grouped for name in members}
    assert "Widget.fill" not in clustered and "fill" not in clustered, "step 5: one differing call is not a clone"


def test_zig_and_java_are_scanned_by_one_run(tmp_path: Path) -> None:
    """One workspace holding both languages reports both, and says what it supports."""
    (tmp_path / "src").mkdir()
    java = (Path(__file__).parent / "fixtures" / "dedup" / "src" / "ExactA.java").read_text()
    (tmp_path / "src" / "ExactA.java").write_text(java)
    (tmp_path / "src" / "ExactB.java").write_text(java.replace("ExactA", "ExactB"))
    for name in ("alpha.zig", "beta.zig"):
        (tmp_path / "src" / name).write_text((ZIG_FIXTURES / "src" / name).read_text())
    report = find_duplication(tmp_path, DupeOptions(kinds=(CloneKind.EXACT,), windows=False))
    assert report.analyzed_files == 4, "both languages are walked in one pass"
    assert report.supported_extensions == (".java", ".zig"), "the report names what it walked"
    files = {member.file_path for clone in report.classes for member in clone.members}
    assert any(path.endswith(".java") for path in files), "the Java copy is reported"
    assert any(path.endswith(".zig") for path in files), "the Zig copy is reported"


def test_every_wire_member_carries_its_visibility() -> None:
    """Both levels ship on every JSON member, so a reader sees what a verdict was read from."""
    report = find_duplication(ZIG_FIXTURES, DupeOptions(kinds=(CloneKind.EXACT,), windows=False))
    members = [member for clone in report_json(report)["classes"] for member in clone["members"]]
    assert members, "the fixture has one exact clone class"
    for member in members:
        assert member["visibility"] == "public", "the cloned `pub fn` is public on the wire"
        assert member["container_visibility"] in {"public", "private"}, "and its container's level ships too"


def test_every_profile_names_real_grammar_nodes() -> None:
    """Drift guard: a profile may only name node kinds its grammar actually has."""
    for profile in set(PROFILES.values()):
        language = get_language(profile.name)
        missing = [
            kind
            for kind in sorted(profile.node_kinds)
            if language.id_for_node_kind(kind, True) is None and language.id_for_node_kind(kind, False) is None
        ]
        assert not missing, f"{profile.name} names node kinds its grammar does not have: {missing}"


def test_every_profile_extension_is_indexable_code() -> None:
    """Drift guard: duplication may only claim extensions zemble already treats as code."""
    code = set(get_extensions([ContentType.CODE]))
    assert set(supported_extensions()) <= code, "a dupes language must be an indexed code language"


def test_the_window_kind_is_never_a_body_kind() -> None:
    """Drift guard: a profile may not name its members `window`, the one kind that is not a body."""
    assert WINDOW_KIND not in body_unit_kinds(), "`window` is a statement run, never a declaration"
    assert set(BODY_KINDS) == set(body_unit_kinds()), "the profiles are the home of the body vocabulary"


def test_an_unclaimed_extension_fails_closed(tmp_path: Path) -> None:
    """A file no profile claims is never parsed, never walked, and never a silent clean report."""
    # 1. Asking for one by hand is refused rather than guessed at.
    with pytest.raises(ValueError, match="No duplication language profile"):
        extract_units(b"print('hi')\n", "src/thing.py")
    assert profile_for("src/thing.py") is None, "nothing claims .py"

    # 2. A workspace of only unclaimed files scans nothing and says so instead of "no duplication".
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "thing.py").write_text("print('hi')\n")
    report = find_duplication(tmp_path, DupeOptions(kinds=(CloneKind.EXACT,)))
    assert report.analyzed_files == 0 and report.failed_files == 0, "step 2: unclaimed files are not walked"
    text = format_report(report)
    assert "No duplication found." not in text, "step 2: nothing scanned is not a clean result"
    assert "Scanned 0 supported file(s)" in text and ".java, .zig" in text, "step 2: it says what it looked for"
