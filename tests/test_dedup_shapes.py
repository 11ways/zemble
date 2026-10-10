"""Behaviour journeys over the holed, idiom, re-implementation and vocabulary channels."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tests.test_dedup import BagOfWordsEmbedder
from zemble.dedup.detect import DupeOptions, find_duplication
from zemble.dedup.languages import SiteKind
from zemble.dedup.model import SITE_CAP, CloneKind, Ranking, Unit
from zemble.dedup.report import format_report, report_json

SHAPES = Path(__file__).parent / "fixtures" / "dedup_shapes"
_CLI_ENTRY = "from zemble.cli import main; main()"


def _run(folder: str | Path, kind: CloneKind, **options: object):
    """Scan one fixture folder (or a path) for one channel."""
    root = SHAPES / folder if isinstance(folder, str) else folder
    return find_duplication(root, DupeOptions(kinds=(kind,), jobs=1, **options), embedder=BagOfWordsEmbedder())


def _names(clone) -> set[str]:
    """The member names of a class."""
    return {member.name for member in clone.members}


def _text(clone) -> str:
    """Every reason line of a class, joined."""
    return " | ".join(clone.wire_reasons)


def test_every_kind_declares_its_facts() -> None:
    """Every channel is keyed, ranked and judged by a declared fact: a new kind without one fails here."""
    for kind in CloneKind:
        facts = kind.facts
        assert facts.ranking in Ranking, f"{kind} declares a ranking"
        assert facts.key_attribute in Unit.__dataclass_fields__, f"{kind} keys by a real unit attribute"
    assert {kind for kind in CloneKind if kind.facts.indexed_focus} == {
        CloneKind.EXACT,
        CloneKind.RENAMED,
        CloneKind.LOGIC,
    }, "only the clone kinds keep a per-root index"


def test_idiom_journey() -> None:
    """A call shape with a hidden constant, a wired value or a get/set pair is an idiom; a plain API call is not."""
    report = _run("idiom", CloneKind.IDIOM)
    by_shape = {clone.notes[0]: clone for clone in report.classes}

    # 1. The copy wrapper and the inline chain are one idiom: literals and once-used names are the same hole.
    copy = by_shape["idiom: Copy.of($0).withFilter(<str>, <str>)"]
    assert len(copy.members) == 8 and copy.files == 4, "step 1: four wrappers and four inline chains"
    hidden = [note for note in copy.notes if "nearly every site repeats" in note]
    assert hidden and '"scope"' in hidden[0], "step 1: the hidden constant is named"
    assert sum(member.kind == SiteKind.WRAPPER.value for member in copy.members) == 4, "step 1: wrappers are marked"

    # 2. The request cache: one receiver read and written under one key.
    pair = by_shape["idiom: getAttribute(@k) .. setAttribute(@k, _)"]
    assert _names(pair) == {"Alpha.counts", "Beta.counts", "Gamma.counts", "Delta.counts"}, "step 2: every cache"

    # 3. One value wired into two calls is data flow, kept even inside a longer chain.
    assert any("getLocales" in shape for shape in by_shape), "step 3: the resolve wiring is an idiom"

    # 4. Near miss: `.icon(Icon.of("x"))` at every site is an API used as designed: no evidence, no class.
    assert not any("icon" in shape for shape in by_shape), "step 4: a plain API call is not an idiom"

    # 5. Spread ranks: copies x files, the copies counted up to three per file.
    assert report.classes[0] is copy, "step 5: the widest idiom leads"
    assert copy.score == 8 * 4, "step 5: spread score"


def test_idiom_folds_a_prefix_extension_cut_at_the_same_sites(tmp_path: Path) -> None:
    """`_.tabs(...)` and `_.tabs(...).build()` at the same sites are one family; an extension at few of them is not."""
    chain = "b.tabs(Tabs.none().withHistory().withContributions())"

    def write(count: int, built: int) -> None:
        for index in range(count):
            tail = ".build()" if index < built else ""
            body = f"class A{index} {{ void declare(Builder b) {{ {chain}{tail}; }} }}\n"
            (tmp_path / f"A{index}.java").write_text(body)

    # 1. Three of four sites go on with `.build()`: one class, the prefix with every site, naming the extension.
    write(4, 3)
    tabs = [clone for clone in _run(tmp_path, CloneKind.IDIOM).classes if ".tabs(" in clone.notes[0]]
    assert len(tabs) == 1, "step 1: one family, one class"
    assert len(tabs[0].members) == 4, "step 1: the variant with more sites stays"
    family = [note for note in tabs[0].notes if note.startswith("one family: 3 of these sites as ")]
    assert family and family[0].endswith(".build()"), "step 1: the folded extension is named with its site count"

    # 2. Near miss: three of seven sites build; the extension is a narrower finding and stays its own class.
    write(7, 3)
    tabs = [clone for clone in _run(tmp_path, CloneKind.IDIOM).classes if ".tabs(" in clone.notes[0]]
    assert sorted(len(clone.members) for clone in tabs) == [3, 7], "step 2: prefix and extension apart"
    assert not any(note.startswith("one family") for clone in tabs for note in clone.notes), "step 2: nothing folded"


def test_holed_journey() -> None:
    """Bodies equal up to literal values and constants are one holed class; data and field differences are not."""
    report = _run("holed", CloneKind.HOLED)

    # 1. `runtimeFor` differs only in the `Egress` constant: one class of three.
    assert len(report.classes) == 1, "step 1: exactly one holed class"
    (clone,) = report.classes
    assert _names(clone) == {"DatabaseKind.runtimeFor", "ReleaseKind.runtimeFor", "ServiceKind.runtimeFor"}
    assert "<const>" in clone.notes[0], "step 1: the constant is a typed hole in the shape"

    # 2. Near misses: `return Icon.of("x");` declares data, and a differing field name is different code.
    names = {name for clone in report.classes for name in _names(clone)}
    assert not any(name.endswith(("getIcon", "pick")) for name in names), "step 2: neither is reported"


def test_holed_leaves_clone_classes_to_the_clone_kinds(tmp_path: Path) -> None:
    """Bodies equal in every literal already form an exact class; holed never reports them twice."""
    body = (
        "    int sum(int a, int b) {{ int total = a + b; total = total * 2 + {n}; log(total); log(total + a);"
        " return total - 1; }}\n"
    )
    for name in ("One", "Two"):
        (tmp_path / f"{name}.java").write_text(f"class {name} {{\n{body.format(n=7)}}}\n")
    assert _run(tmp_path, CloneKind.EXACT).classes, "the pair is exact duplication"
    assert not _run(tmp_path, CloneKind.HOLED).classes, "and holed leaves it to exact"
    (tmp_path / "Two.java").write_text(f"class Two {{\n{body.format(n=9)}}}\n")
    assert _run(tmp_path, CloneKind.HOLED).classes, "a differing literal makes it holed"


def test_reimplements_journey() -> None:
    """A private copy of a public method, and a forwarding facade, point at what to call; parallel overrides do not."""
    report = _run("reimpl", CloneKind.REIMPLEMENTS)
    lines = [_text(clone) for clone in report.classes]

    # 1. `fixOf` repeats `Health.fixCell` without calling it: copy first, API second, and the advice names it.
    fix = next(clone for clone in report.classes if "fixOf" in _text(clone))
    assert [member.name for member in fix.members] == ["AppDirectory.fixOf", "Health.fixCell"], "step 1: copy, API"
    assert fix.notes[0].startswith("AppDirectory.fixOf re-implements Health.fixCell; call Health.fixCell")

    # 2. `Slugs` only passes its parameters to `SlugText`: retire the facade.
    assert any("Slugs is a forwarding facade over SlugText" in line for line in lines), "step 2: the facade"

    # 3. Near misses: two same-named overrides are parallel implementations, and a body that calls the API
    #    already uses it.
    assert not any("specFor" in line or "firstFix" in line for line in lines), "step 3: neither is a re-implementation"
    assert len(report.classes) == 2, "step 3: nothing else"
    assert not any(clone.inferred for clone in report.classes), "step 3: both rest on code"


def test_reimplements_intent_journey() -> None:
    """A static helper stating a public method's intent is a copy of it; its twin follows; parallels do not."""
    from zemble.dedup.reimplements import INTENT_MIN_SCORE, Signal, _may_replace, _NameWords, _substitution

    report = _run("intent", CloneKind.REIMPLEMENTS)

    # 1. `blankToNull` states `Texts.blankAsNull`'s intent with other calls: an inferred class, copy then API.
    (clone,) = report.classes
    assert [member.name for member in clone.members][-1] == "Texts.blankAsNull", "step 1: the API comes last"
    assert clone.inferred, "step 1: intent evidence alone ranks after code evidence"
    assert any(note.startswith("same intent, score") for note in clone.notes), "step 1: the signals are shown"

    # 2. `nullIfBlank` is `blankToNull` again: it joins through the twin lane, not on its own name.
    assert "Courier.nullIfBlank" in _names(clone), "step 2: the twin of a copy is a copy"
    assert any(note.startswith("the same code as Mailer.blankToNull") for note in clone.notes), "step 2: says so"

    # 3. Near misses: an instance method, a helper with its own literal, a helper returning something else.
    assert not _names(clone) & {"Mailer.blankAsText", "Mailer.scoped", "Courier.blankCount"}, "step 3: none"
    shaped = {unit.name: unit for unit in _shaped("intent")}
    assert not _may_replace(shaped["Mailer.scoped"], shaped["Texts.scoped"]), "step 3: its own scope literal"
    assert _substitution(shaped["Courier.blankCount"], shaped["Texts.blankAsNull"]) == 0.0, "step 3: int is no text"
    assert _substitution(shaped["Mailer.blankToNull"], shaped["Texts.blankAsNull"]) == 1.0, "step 3: text is"

    # 4. Names: rare shared words count, a prefix matches (`int` in `integer`), a different verb does not.
    units = list(shaped.values())
    names = _NameWords(units)
    index = {unit.name: position for position, unit in enumerate(units)}
    assert names.similarity(index["Mailer.blankToNull"], index["Texts.blankAsNull"]) == 1.0, "step 4: same words"
    assert names.similarity(index["Courier.blankCount"], index["Texts.blankAsNull"]) < 1.0, "step 4: count differs"

    # 5. The weights are shares of one score: they sum to one, and the bar is a share of it.
    assert abs(sum(signal.weight for signal in Signal) - 1.0) < 1e-9 and 0 < INTENT_MIN_SCORE < 1, "step 5"


def _shaped(folder: str) -> list[Unit]:
    """Every shaped body of one fixture folder, signatures read."""
    from zemble.dedup.detect import collect_units

    return collect_units(SHAPES / folder, DupeOptions(kinds=(CloneKind.REIMPLEMENTS,), jobs=1)).shaped


def test_vocabulary_journey() -> None:
    """One vocabulary declared in several places is reported with every place and a home; coincidences are not."""
    report = _run("vocab", CloneKind.VOCABULARY)
    lines = [_text(clone) for clone in report.classes]

    def find(fragment: str):
        return next(clone for clone in report.classes if fragment in _text(clone))

    # 1. "failed" declared by three constants whose names spell it.
    failed = find('value "failed"')
    assert {member.kind for member in failed.members} == {SiteKind.CONSTANT.value}, "step 1: the word's uses unlisted"
    assert "suggested home:" in _text(failed), "step 1: a home is suggested"

    # 2. Three status sets in parallel, with `success` drifting from `succeeded`.
    family = find("value sets declare the same vocabulary")
    assert family.files == 3 and "drift: succeeded ~ success" in _text(family), "step 2: the family and its drift"

    # 3. A literal written where a constant holds it.
    devices = find('value "instance-devices"')
    assert _names(devices) == {"Attachments.DEVICES", "DevicesPage.target"}, "step 3: constant and literal"

    # 4. A regex written twice, and switches on one call restating labels in three files.
    assert find("regex '^[^@").files == 2, "step 4: the email regex"
    assert find("switches on column.name() in 3 files"), "step 4: the dispatch"

    # 5. Near misses: "hohenheim" under three unrelated names, a separator regex, a dispatch in two files.
    assert not any('"hohenheim"' in line for line in lines), "step 5: one value, three meanings"
    assert not any("\\\\s+" in line for line in lines), "step 5: a separator is not a regex worth a home"
    assert not any("switches on field.label()" in line for line in lines), "step 5: two files are not a dispatch"


def test_new_kinds_ride_the_ignore_file_lanes_and_caps(tmp_path: Path) -> None:
    """A justified ignore entry suppresses an idiom; classes are sectioned by lane and list at most SITE_CAP sites."""
    for index in range(SITE_CAP + 2):
        folder = tmp_path / ("src/test" if index % 2 else "src/main")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"C{index}.java").write_text(
            f'class C{index} {{ String t() {{ return Copy.of("k{index}").withFilter("scope", "x"); }} }}\n'
        )
    report = _run(tmp_path, CloneKind.IDIOM)
    (clone,) = report.classes

    # 1. Production and test copies of one idiom form a mixed class, and the wire caps the member list.
    assert clone.lane.value == "mixed", "step 1: lanes apply"
    payload = report_json(report)["classes"][0]
    assert payload["copies"] == SITE_CAP + 2 and len(payload["members"]) == SITE_CAP, "step 1: capped, counted"
    assert "and 2 more site(s)" in format_report(report), "step 1: the text says what it left out"

    # 2. A justified entry suppresses it; an unjustified one is a violation.
    (tmp_path / ".zemble").mkdir()
    (tmp_path / ".zemble" / "dupes.ignore").write_text(f"{clone.key}  the copy helper is being retired\n")
    suppressed = _run(tmp_path, CloneKind.IDIOM)
    assert not suppressed.classes and len(suppressed.suppressed) == 1, "step 2: suppressed"
    (tmp_path / ".zemble" / "dupes.ignore").write_text(f"{clone.key}\n")
    assert _run(tmp_path, CloneKind.IDIOM).ignore_problems, "step 2: no justification is a violation"


def test_focus_filters_the_new_kinds_to_classes_touching_it() -> None:
    """A focused run reports the new channels' classes with a member under the focus, the same as a whole run."""
    whole = _run("vocab", CloneKind.VOCABULARY)
    focused = _run("vocab", CloneKind.VOCABULARY, focus=("Attachments.java",))
    expected = [clone.key for clone in whole.classes if any(m.file_path == "Attachments.java" for m in clone.members)]
    assert (
        [clone.key for clone in focused.classes]
        == expected
        == [next(clone.key for clone in whole.classes if "instance-devices" in _text(clone))]
    ), "only the class holding the focus file, with the whole run's key"


def test_cli_and_mcp_accept_the_new_kinds(tmp_path: Path) -> None:
    """Both surfaces take the new kinds through the existing `kind` argument."""
    result = subprocess.run(
        [sys.executable, "-c", _CLI_ENTRY, "dupes", str(SHAPES / "holed"), "--kind", "holed,idiom", "--brief"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "holed production" in result.stdout, "the CLI prints the holed class"

    import asyncio

    from mcp.server.fastmcp import FastMCP

    from zemble.dedup.mcp import register_dupes_tool

    server = FastMCP("test")
    register_dupes_tool(server)
    content = asyncio.run(server.call_tool("dupes", {"repo": str(SHAPES / "holed"), "kind": "holed", "format": "json"}))
    assert '"kind": "holed"' in content[0].text, "the MCP tool answers with the holed class"


@pytest.mark.parametrize("kind", [CloneKind.IDIOM, CloneKind.VOCABULARY])
def test_a_language_without_shape_hooks_says_so(tmp_path: Path, kind: CloneKind) -> None:
    """A run of a site channel names the languages it covers, so an empty Python result is never read as clean."""
    (tmp_path / "a.py").write_text("def f(x):\n    return g(x).h(1)\n")
    report = _run(tmp_path, kind)
    assert not report.classes and any("sites cover: java" in note for note in report.notes)


def test_site_units_never_reach_the_clone_kinds() -> None:
    """The sub-body outputs live beside the clone units, so exact and renamed stay what they were."""
    report = _run("idiom", CloneKind.EXACT, min_tokens=5)
    kinds = {member.kind for clone in report.classes for member in clone.members}
    assert not kinds & {kind.value for kind in SiteKind}, "no site unit in an exact class"
    assert replace(DupeOptions(), kinds=(CloneKind.EXACT,)).kinds == (CloneKind.EXACT,)
