"""The acceptance tests for two harms, two guards, two units.

Runaway WORK is refused in bytes before a parse; a runaway BILL is refused in money once
the uncached set is known. The bug these pin is a work measurement priced as a bill: the
pre-flight said a build was affordable while the pre-parse guard refused the same tree.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from tests.conftest import make_chunk
from tests.embedding.test_pricing import PricedEmbedder
from zemble.cache import save_index_to_cache
from zemble.daemon.server import REFUSAL_TYPES
from zemble.dedup.detect import DupeOptions, logic_classes
from zemble.dedup.model import CloneKind
from zemble.dedup.units import extract_units
from zemble.embedding.base import Embedder
from zemble.embedding.cache import CachingEmbedder
from zemble.embedding.preflight import embed_status
from zemble.embedding.pricing import (
    BUDGET_ENV,
    BUDGET_USD_ENV,
    CAPSULE_OVERHEAD_HIGH,
    CAPSULE_OVERHEAD_LOW,
    DEAREST_DOCUMENTED_RATE,
    DEFAULT_BUDGET_USD,
    DEFAULT_UNPRICED_BUDGET_TOKENS,
    ESTIMATE_CHARS_PER_TOKEN,
    MAX_BUDGET_TOKENS,
    PRICES_USD_PER_MILLION_TOKENS,
    EmbeddingBudgetExceeded,
    bill_refusal,
    check_budget,
    estimate_cost,
    format_usd,
)
from zemble.embedding.registry import CACHE_ENV, ResolvedEmbedder, build_embedder, caching_enabled
from zemble.index import ScopeRefused, ZembleIndex
from zemble.index.dense import embed_chunks
from zemble.index.scope import DEFAULT_WORK_LIMIT_BYTES, WORK_LIMIT_ENV, estimate_tree
from zemble.refusal import Refused
from zemble.types import ContentType

FAMILY = "voyage:voyage-4-lite"

#: The dearest family the price table documents, named the way a cache family key is spelled.
DEAREST_FAMILY = "openai:https://api.openai.com/v1#text-embedding-3-large"

#: The FEWEST estimated tokens a build the work guard admits can carry: 180 MB of source at the
#: measured capsule floor, over the pricing density. The backstop has to sit under this one -
#: a tree of large files is the cheapest thing 180 MB can be, so binding here binds everywhere.
WORK_CEILING_TOKENS = int(DEFAULT_WORK_LIMIT_BYTES * CAPSULE_OVERHEAD_LOW / ESTIMATE_CHARS_PER_TOKEN)

#: The MOST it can carry: the same volume in files small enough for the capsule header to
#: double them. The worst real bill a build at the work ceiling can run up is priced from this.
WORST_CASE_CEILING_TOKENS = int(DEFAULT_WORK_LIMIT_BYTES * CAPSULE_OVERHEAD_HIGH / ESTIMATE_CHARS_PER_TOKEN)

#: The javaweb workspace, measured on the real tree: 73.7 MB of code and docs, 17.8M tokens from
#: bytes and +21% for capsules at that mix of file sizes. The build the backstop may never refuse.
JAVAWEB_ESTIMATED_TOKENS = 21_600_000

#: The byte volume the pre-parse guard used to refuse: it converted bytes to tokens at the
#: pricing density and compared them against the old 2M-token ceiling, so any tree fatter
#: than this was refused however little of it still had to be bought.
OLD_CEILING_BYTES = math.ceil(DEFAULT_UNPRICED_BUDGET_TOKENS * ESTIMATE_CHARS_PER_TOKEN)


def _tree(root: Path, files: int, per_file: int) -> Path:
    """Write a source tree of a given byte size, with no two functions alike."""
    (root / "src").mkdir(parents=True)
    for index in range(files):
        lines: list[str] = []
        written = 0
        line = 0
        while written < per_file:
            body = f"def f_{index}_{line}():\n    return {line}\n\n"
            lines.append(body)
            written += len(body)
            line += 1
        (root / "src" / f"module_{index}.py").write_text("".join(lines), encoding="utf-8")
    return root


def _touch_every_file(root: Path) -> None:
    """Rewrite every source file with its own content, so only its modification time moves."""
    for path in sorted(root.rglob("*.py")):
        path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def test_a_tree_whose_chunks_are_already_bought_is_not_refused(tmp_path: Path) -> None:
    """A fat tree nobody has to pay for twice rebuilds: bytes are work, and work is not a bill."""
    root = _tree(tmp_path / "workspace", files=10, per_file=800_000)
    inner = PricedEmbedder(dimensions=8)
    embedder = CachingEmbedder(inner, FAMILY, tmp_path / "vectors")

    # 1. The tree is fatter than the byte volume the old ceiling refused: this is the regression.
    assert estimate_tree(root.resolve(), (ContentType.CODE,)).bytes > OLD_CEILING_BYTES, (
        "step 1: the fixture must be a tree the old ceiling would have refused"
    )

    # 2. The first build is allowed and pays for every chunk once.
    first = ZembleIndex.from_path(root, embedder=embedder)
    save_index_to_cache(first, str(root.resolve()))
    bought = len(inner.document_batches)
    assert bought > 0, "step 2: a cold build really does buy the tree"

    # 3. Every file moves, so the manifest covers none of them and the walk prices all of it.
    _touch_every_file(root.resolve())
    estimate = estimate_tree(root.resolve(), (ContentType.CODE,), (), first._manifest)
    assert estimate.bytes > OLD_CEILING_BYTES, "step 3: the walk still sees the whole tree as work to redo"

    # 4. Neither guard refuses: not the pre-parse walk, and not the budget once the set is known.
    rebuilt = ZembleIndex.from_path(root, embedder=embedder)
    assert rebuilt.stats.indexed_files == first.stats.indexed_files, "step 4: the same tree was indexed again"

    # 5. And the provider was never contacted again, because there was nothing left to buy.
    assert len(inner.document_batches) == bought, (
        f"step 5: a fully cached rebuild sends nothing, got {len(inner.document_batches)} batches"
    )


def test_the_two_lanes_agree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever `embed-status` reports for a tree, a real build of that tree does: one answer, two surfaces.

    This is the invariant whose absence produced the bug: the report said a build was allowed
    while the guard refused the very same tree, so nobody could tell which one was lying. Both
    lanes are walked twice - with the vector cache on, and with it turned off, which is a
    documented knob and must move neither verdict.
    """
    root = _tree(tmp_path / "workspace", files=12, per_file=96_000)
    inner = PricedEmbedder(dimensions=8)
    monkeypatch.setattr(
        "zemble.embedding.preflight.build_embedder",
        lambda spec: ResolvedEmbedder(spec=spec, embedder=inner, scheme="voyage", family=FAMILY),
    )

    def _verdicts(cached: bool = True) -> tuple[bool, bool]:
        """Ask the report and a real build the same question, in that order."""
        reported = embed_status(root).would_refuse
        embedder = CachingEmbedder(inner, FAMILY) if cached else inner
        try:
            ZembleIndex.from_path(root, embedder=embedder)
        except (ScopeRefused, EmbeddingBudgetExceeded):
            return reported, True
        return reported, False

    # 1. Refused as work: the tree is over the byte ceiling, and both lanes say so.
    monkeypatch.setenv(WORK_LIMIT_ENV, "1")
    reported, built = _verdicts()
    assert (reported, built) == (True, True), f"step 1: report {reported}, build {built}"

    # 2. Refused as money: affordable work, unaffordable bill, and both lanes say so.
    monkeypatch.delenv(WORK_LIMIT_ENV)
    monkeypatch.setenv(BUDGET_USD_ENV, "0.001")
    reported, built = _verdicts()
    assert (reported, built) == (True, True), f"step 2: report {reported}, build {built}"

    # 3. Allowed: the report says a build would go through, and the build goes through.
    monkeypatch.delenv(BUDGET_USD_ENV)
    reported, built = _verdicts()
    assert (reported, built) == (False, False), f"step 3: report {reported}, build {built}"
    assert inner.document_batches, "step 3: and the allowed build is the only one that bought anything"

    # 4. With the vector cache turned off, the very same three verdicts hold. The bill guard
    #    used to live INSIDE the cache wrapper, so this one variable deleted every money
    #    ceiling while the report went on saying REFUSED.
    monkeypatch.setenv(CACHE_ENV, "0")
    assert not caching_enabled(), "step 4: the documented knob really is off"
    monkeypatch.setenv(WORK_LIMIT_ENV, "1")
    assert _verdicts(cached=False) == (True, True), "step 4: work is refused with the cache off too"
    monkeypatch.delenv(WORK_LIMIT_ENV)
    monkeypatch.setenv(BUDGET_USD_ENV, "0.001")
    assert _verdicts(cached=False) == (True, True), "step 4: and so is a bill"
    monkeypatch.delenv(BUDGET_USD_ENV)
    assert _verdicts(cached=False) == (False, False), "step 4: and an affordable build still runs"


def test_the_bill_guard_is_not_deleted_by_turning_the_cache_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ZEMBLE_EMBED_CACHE=0` is a documented knob, and it must not be a way past the budget.

    The money check lived in `CachingEmbedder`, whose wrapper `build_embedder` only installs
    while caching is on - so one documented variable removed every spending ceiling, and a
    library caller handing in a bare remote embedder had none either. Measured before the fix:
    cache on refused and sent 0 texts, cache off sent 2,032.
    """
    root = _tree(tmp_path / "workspace", files=4, per_file=48_000)
    monkeypatch.setenv(BUDGET_USD_ENV, "0.0000001")

    # 1. With the knob off, `build_embedder` really does hand back an unwrapped embedder.
    monkeypatch.setenv(CACHE_ENV, "0")
    monkeypatch.setattr(
        "zemble.embedding.registry._build", lambda spec: (PricedEmbedder(dimensions=8), "voyage", FAMILY)
    )
    resolved = build_embedder("voyage:voyage-4-lite")
    assert not isinstance(resolved.embedder, CachingEmbedder), "step 1: the wrapper is what the knob removes"

    # 2. That bare embedder is refused all the same, and buys nothing.
    bare = resolved.embedder
    with pytest.raises(EmbeddingBudgetExceeded) as raised:
        ZembleIndex.from_path(root, embedder=bare)
    assert bare.document_batches == [], "step 2: a refused build sends nothing, cache or no cache"
    assert BUDGET_USD_ENV in str(raised.value), "step 2: refused in money, naming the money knob"

    # 3. And the report agrees, which is the invariant that was inverted: it said REFUSED while
    #    the build paid.
    monkeypatch.setattr(
        "zemble.embedding.preflight.build_embedder",
        lambda spec: ResolvedEmbedder(spec=spec, embedder=bare, scheme="voyage", family=FAMILY),
    )
    assert embed_status(root).would_refuse, "step 3: one answer, two surfaces, with the cache off too"


def test_a_refusal_carries_the_knob_that_refused_it_down_every_lane(tmp_path: Path) -> None:
    """A refusal names the ceiling a caller has to raise, and it stays named as it travels.

    `knob` is declared once, on the shared `Refused` base, because the daemon reads it off a
    caught refusal it never type-checks: a refusal type without one is an AttributeError inside
    an except block. Re-raising to add a hint used to reset it to the money knob, so a VOLUME
    refusal told the reader to raise a budget that was never the one that refused.
    """
    # 1. Both shipped refusal types are the same kind of answer, and both carry a knob.
    for refusal in (ScopeRefused("refused"), EmbeddingBudgetExceeded("refused")):
        assert isinstance(refusal, Refused), f"step 1: {type(refusal).__name__} must be a Refused"
        assert refusal.knob, f"step 1: {type(refusal).__name__} carries no knob"
    assert REFUSAL_TYPES == (Refused,), "step 1: the daemon catches the base, so a new type needs no edit"

    # 2. The volume lane names the volume knob, not the money one.
    with pytest.raises(EmbeddingBudgetExceeded) as raised:
        check_budget("voyage:voyage-9-imaginary@8", "voyage:voyage-9-imaginary", 1, DEFAULT_UNPRICED_BUDGET_TOKENS + 1)
    assert raised.value.knob == BUDGET_ENV, f"step 2: refused by volume, got {raised.value.knob}"

    # 3. And the ancestor hint the index cache adds keeps it, text and knob alike.
    inner = raised.value
    outer = EmbeddingBudgetExceeded(f"{inner} plus a hint", inner.knob)
    assert outer.knob == BUDGET_ENV, "step 3: a re-raise for a hint does not re-decide the ceiling"
    assert str(inner) in str(outer), "step 3: and the original reason is still readable"


def test_the_volume_backstop_binds_under_the_work_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A price mistyped low narrows the money ceiling; the backstop is what stops it deleting it.

    The backstop is denominated in the table's own DEAREST documented rate, so the tokens it
    admits bill at most the default budget whatever any single rate claims. At 100,000,000 it
    sat ABOVE everything the 180 MB work guard can admit and therefore refused nothing at all:
    with `voyage-code-4` typed one order of magnitude low, a build at the work ceiling billed
    $7.26 for real while the guard computed $0.73 and let it through.
    """
    # 1. The ceiling is derived from the price table, not typed a second time beside it.
    dearest = max(price for prices in PRICES_USD_PER_MILLION_TOKENS.values() for price in prices.values())
    assert DEAREST_DOCUMENTED_RATE == dearest, "step 1: the dearest rate is read off the table"
    assert MAX_BUDGET_TOKENS == int(DEFAULT_BUDGET_USD / dearest * 1_000_000), "step 1: and the backstop divides it"

    # 2. Which means the dearest model in the table bills exactly the budget at the backstop.
    at_the_ceiling = estimate_cost(MAX_BUDGET_TOKENS, DEAREST_DOCUMENTED_RATE)
    assert at_the_ceiling is not None and at_the_ceiling <= DEFAULT_BUDGET_USD, (
        f"step 2: {MAX_BUDGET_TOKENS:,} tokens at the dearest rate is {at_the_ceiling}, over the budget"
    )

    # 3. It BINDS: a backstop above everything the work guard admits can never refuse anything.
    assert MAX_BUDGET_TOKENS < WORK_CEILING_TOKENS, (
        f"step 3: {MAX_BUDGET_TOKENS:,} is not under the {WORK_CEILING_TOKENS:,} tokens a 180 MB "
        "build can carry, so no build the work guard admits could ever reach it"
    )

    # 4. The counter-example: one rate mistyped 10x low. The MONEY half computes $0.73 and allows.
    monkeypatch.setitem(PRICES_USD_PER_MILLION_TOKENS["voyage"], "voyage-code-4", 0.012)
    computed = estimate_cost(WORK_CEILING_TOKENS, 0.012)
    assert computed is not None and computed < DEFAULT_BUDGET_USD, f"step 4: the wrong rate reads as {computed}"

    # 5. And the build is refused all the same, by volume, naming the knob that refused it.
    refusal = bill_refusal(WORK_CEILING_TOKENS, "voyage:voyage-code-4")
    assert refusal is not None, "step 5: a rate 10x too low must not delete the ceiling off a build"
    assert BUDGET_ENV in refusal and "no price may lift" in refusal, f"step 5: refused by volume, got {refusal}"

    # 6. Which matters because the real bill at the real rate was over the budget all along.
    real = estimate_cost(WORK_CEILING_TOKENS, 0.12)
    assert real is not None and real > DEFAULT_BUDGET_USD, f"step 6: the real bill was {real}"


def test_the_volume_backstop_admits_a_full_workspace_index() -> None:
    """The backstop bounds a runaway, and a real multi-repo workspace is not one.

    A ceiling that refuses the build it was written for is a broken ceiling: javaweb is the
    workspace this tool is developed against, and its whole code-and-docs index has to pass.
    """
    # 1. The measured full javaweb index is under the backstop, with room over it.
    assert JAVAWEB_ESTIMATED_TOKENS < MAX_BUDGET_TOKENS, (
        f"step 1: {JAVAWEB_ESTIMATED_TOKENS:,} estimated tokens must fit under {MAX_BUDGET_TOKENS:,}"
    )

    # 2. So no lane refuses it: not at the configured model ($0.43), not at the dearest ($2.81).
    assert bill_refusal(JAVAWEB_ESTIMATED_TOKENS, FAMILY) is None, "step 2: the configured model is affordable"
    assert bill_refusal(JAVAWEB_ESTIMATED_TOKENS, DEAREST_FAMILY) is None, "step 2: and so is the dearest one"


#: What each prose file quotes off the constants, and which figure it quotes. Prose cannot import,
#: so this is the binding: the numbers are rendered from the constants here and must appear in the
#: file. ``docs/plan.md`` is deliberately absent - it is a dated decision log, not current prose.
_DOCUMENTED_FIGURES: dict[str, tuple[str, ...]] = {
    "docs/embedders.md": ("low_overhead", "high_overhead", "low_tokens", "high_tokens", "backstop", "worst_bill"),
    "README.md": ("low_tokens", "high_tokens", "backstop"),
    "CLAUDE.md": ("low_overhead", "high_overhead", "low_tokens", "high_tokens", "backstop", "worst_bill"),
}


def _rendered_figures() -> dict[str, str]:
    """Render every capsule/backstop figure the prose quotes, the way the prose spells it."""
    worst_bill = estimate_cost(WORST_CASE_CEILING_TOKENS, DEAREST_DOCUMENTED_RATE)
    assert worst_bill is not None, "the dearest documented rate must price the worst case"
    return {
        "low_overhead": f"{CAPSULE_OVERHEAD_LOW:.2f}x",
        "high_overhead": f"{CAPSULE_OVERHEAD_HIGH:.2f}x",
        "low_tokens": f"{WORK_CEILING_TOKENS // 1_000_000}M",
        "high_tokens": f"{WORST_CASE_CEILING_TOKENS // 1_000_000}M",
        "backstop": f"{MAX_BUDGET_TOKENS / 1_000_000:.1f}M",
        "worst_bill": format_usd(worst_bill),
    }


def test_the_prose_quotes_the_capsule_constants_it_argues_from() -> None:
    """Every number the prose argues the guards from is derived here, so neither can drift alone.

    The binding argument ("the backstop sits under what 180 MB can carry") and the worst-case
    bill are the two figures a reader checks the design against, and both are hand-typed in
    Markdown. Moving a constant without moving the prose leaves the repo asserting the old one.
    """
    root = Path(__file__).resolve().parents[2]
    figures = _rendered_figures()
    for relative, names in _DOCUMENTED_FIGURES.items():
        prose = (root / relative).read_text(encoding="utf-8")
        for name in names:
            assert figures[name] in prose, f"{relative} does not quote {name} as {figures[name]}"


def _buy_chunks(embedder: Embedder) -> None:
    """Drive `zemble.index.dense.embed_chunks`, the seam every index build buys its vectors at."""
    embed_chunks(embedder, [make_chunk("x = 1\n" * 600)])


def _buy_logic_bodies(embedder: Embedder) -> None:
    """Drive the dupes logic lane, the seam that buys vectors without ever building an index."""
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "dedup" / "src"
    units = [
        unit
        for name in ("LogicA.java", "LogicB.java")
        for unit in extract_units((fixtures / name).read_bytes(), name, include_text=True)
        if unit.is_body
    ]
    assert len(units) >= 2, f"the logic lane needs two candidate bodies to embed anything, got {len(units)}"
    logic_classes(units, DupeOptions(kinds=(CloneKind.LOGIC,), windows=False), embedder)


#: The method every provider hands document texts to. Whoever names it is a seam that buys.
SEAM_METHOD = "embed_documents"


def _reaches_the_provider(module: Path) -> bool:
    """Return whether a module names the provider seam, in any spelling Python can reach it by.

    Enumerating by substring (`.embed_documents(`) read source as TEXT, so a space before the
    parenthesis or a `getattr` by name was never enumerated at all, and step 2 only ever drives
    what step 1 found. The parse tree does not care how a call is spaced; what it cannot see is
    a name assembled at runtime, which no buying seam has any reason to do.

    :param module: The module to read.
    :return: Whether it reaches the seam by name.
    """
    for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Attribute) and node.attr == SEAM_METHOD:
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
            named = node.args[1] if len(node.args) > 1 else None
            if isinstance(named, ast.Constant) and named.value == SEAM_METHOD:
                return True
    return False


#: Every place outside `zemble/embedding/` that hands document texts to a provider, each with the
#: call that drives it. The guard lives at the seams that buy rather than inside the optional
#: cache wrapper, so this is what keeps them in step - and naming a seam here is not enough,
#: because step 2 RUNS each probe: asserting that the string "require_affordable_bill" appears
#: in the file passed with both guard call lines deleted.
PAID_DOCUMENT_SEAMS = {"index/dense.py": _buy_chunks, "dedup/detect.py": _buy_logic_bodies}


def test_every_paid_document_seam_passes_the_bill_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A new place that buys document vectors must meet the bill guard, not discover it later.

    The money check no longer sits behind `CachingEmbedder`, which one environment variable can
    remove; it sits at the seams that buy. That only holds while every seam is one of these AND
    every one of them really refuses.
    """
    # 1. The enumeration reads the grammar rather than the text, so no spelling of the call
    #    hides from it - and a module that only talks about the seam is not one.
    spellings = {
        "spaced.py": "def go(e, t):\n    return e .embed_documents (t)\n",
        "indirect.py": 'def go(e, t):\n    return getattr(e, "embed_documents")(t)\n',
        "mentioned.py": '"""A module that says embed_documents and calls nothing."""\n',
    }
    for name, source in spellings.items():
        (tmp_path / name).write_text(source, encoding="utf-8")
    reaching = [name for name in sorted(spellings) if _reaches_the_provider(tmp_path / name)]
    assert reaching == ["indirect.py", "spaced.py"], f"step 1: the enumeration found {reaching}"

    source_root = Path(__file__).resolve().parents[2] / "src" / "zemble"
    found: list[str] = []
    for module in sorted(source_root.rglob("*.py")):
        relative = module.relative_to(source_root).as_posix()
        if relative.startswith("embedding/"):
            continue  # The providers and the cache are the implementations, not the buyers.
        if _reaches_the_provider(module):
            found.append(relative)

    # 2. Nobody bought vectors from a seam this suite has never heard of.
    assert set(found) == set(PAID_DOCUMENT_SEAMS), (
        f"step 2: the seams that buy document vectors changed, got {found}; "
        "wire require_affordable_bill into the new one and name it here with the call that drives it"
    )

    # 3. And every one of them, driven for real against a ceiling of nothing, refuses and buys
    #    nothing. Reading the source for the guard's NAME never proved this: deleting both call
    #    lines left this test - and the whole suite - green.
    monkeypatch.setenv(BUDGET_USD_ENV, "0.0000001")
    for relative, probe in sorted(PAID_DOCUMENT_SEAMS.items()):
        spy = PricedEmbedder(dimensions=8)
        with pytest.raises(EmbeddingBudgetExceeded):
            probe(spy)
        assert spy.document_batches == [], f"step 3: {relative} bought {spy.document_batches} before the guard"


def test_the_report_bills_what_the_build_announces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A repeated chunk is bought once, so the report may not bill it twice.

    The caching embedder gives duplicate texts inside one call a single provider slot. A report
    that counted every copy refused trees of near-identical vendored source over a bill nobody
    was ever going to be sent.
    """
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "repeated.py").write_text("x = 1\n" * 16_000, encoding="utf-8")
    inner = PricedEmbedder(dimensions=8)
    # The report resolves the buyer the build gets - `build_embedder` wraps while caching is on -
    # because what a repeated text costs is the BUYER's answer, not a rule the report owns.
    monkeypatch.setattr(
        "zemble.embedding.preflight.build_embedder",
        lambda spec: ResolvedEmbedder(
            spec=spec, embedder=CachingEmbedder(inner, FAMILY), scheme="voyage", family=FAMILY
        ),
    )

    # 1. The file chunks into many pieces, and they are not all different pieces.
    status = embed_status(root)
    assert status.uncached > 1, f"step 1: the fixture must chunk into several, got {status.uncached}"

    # 2. The build announces exactly the tokens the report estimated: one answer, two surfaces.
    with caplog.at_level("INFO", logger="zemble.embedding.pricing"):
        ZembleIndex.from_path(root, embedder=CachingEmbedder(inner, FAMILY))
    announced = [record.getMessage() for record in caplog.records if "uncached chunk(s)" in record.getMessage()]
    assert len(announced) == 1, f"step 2: exactly one announcement, got {caplog.records}"
    bought = int(announced[0].split()[1])
    assert bought < status.uncached, f"step 2: the fixture must repeat chunks, {bought} of {status.uncached} distinct"
    assert f"~{status.estimated_tokens} tokens" in announced[0], (
        f"step 2: the report estimated {status.estimated_tokens}, the build announced {announced[0]}"
    )


class _ProbeOnlyWidth(PricedEmbedder):
    """A remote model whose vector width only a provider request could tell."""

    #: Shadows the base property on purpose: this model publishes no width anywhere.
    declared_dimensions = None

    def __init__(self, dimensions: int = 8) -> None:
        """Initialise the fake, counting every read of the width a probe would have to pay for."""
        super().__init__(dimensions=dimensions)
        self.width_reads = 0

    @property
    def dimensions(self) -> int:
        """The width, recorded because reading it is the request a pre-flight may never make."""
        self.width_reads += 1
        return self._dimensions


def test_the_report_bills_what_a_build_buys_for_a_model_that_declares_no_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A model with no declared width is billed for what its cache buys, not for every copy.

    Asking the buyer what a repeated chunk costs reads a vector width, and reading one off such
    a model is a provider request this report may never make - so the question was skipped and
    every uncached text billed instead: measured at 3,662 tokens where the caching build bought
    184. The width a report may use is the one the family's own cache file already holds, and a
    duplicate collapses whether or not any width is known at all.
    """
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "repeated.py").write_text("x = 1\n" * 16_000, encoding="utf-8")
    inner = _ProbeOnlyWidth()
    monkeypatch.setattr(
        "zemble.embedding.preflight.build_embedder",
        lambda spec: ResolvedEmbedder(
            spec=spec, embedder=CachingEmbedder(inner, FAMILY), scheme="voyage", family=FAMILY
        ),
    )

    # 1. The report answers over an empty cache file, with no width to be had from anywhere,
    #    and it contacts nobody to get one.
    status = embed_status(root)
    assert status.dimensions is None, "step 1: the fixture must be a model that declares no width"
    assert status.uncached > 1, f"step 1: the fixture must chunk into several, got {status.uncached}"
    assert inner.width_reads == 0, "step 1: a pre-flight report may not pay for a probe"

    # 2. The build buys the distinct texts once, and announces exactly what the report estimated.
    with caplog.at_level("INFO", logger="zemble.embedding.pricing"):
        ZembleIndex.from_path(root, embedder=CachingEmbedder(inner, FAMILY))
    announced = [record.getMessage() for record in caplog.records if "uncached chunk(s)" in record.getMessage()]
    assert len(announced) == 1, f"step 2: exactly one announcement, got {caplog.records}"
    bought = int(announced[0].split()[1])
    assert bought < status.uncached, f"step 2: the fixture must repeat chunks, {bought} of {status.uncached} distinct"
    assert f"~{status.estimated_tokens} tokens" in announced[0], (
        f"step 2: the report estimated {status.estimated_tokens}, the build announced {announced[0]}"
    )

    # 3. And now that the file holds vectors, the report reads its width off it: nothing to buy.
    assert inner.width_reads > 0, "step 3: the build is what pays for the probe"
    after = embed_status(root)
    assert after.estimated_tokens == 0, f"step 3: everything is bought already, got {after.estimated_tokens} tokens"


def test_the_report_bills_what_a_cacheless_build_announces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Collapsing a repeated chunk is the CACHING buyer's behaviour, not a rule of the report's.

    `ZEMBLE_EMBED_CACHE=0` hands a build a bare remote embedder, and a bare remote embedder buys
    every copy of a repeated text. The report deduplicated unconditionally, so on a tree of
    near-identical chunks it reported a twentieth of the bill the guard would refuse over - the
    same report/build divergence as the work guard's, in the other direction.
    """
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "src" / "repeated.py").write_text("x = 1\n" * 16_000, encoding="utf-8")
    inner = PricedEmbedder(dimensions=8)

    def _resolve(spec: str) -> ResolvedEmbedder:
        """Resolve the buyer the way `build_embedder` does: wrapped only while caching is on."""
        embedder = CachingEmbedder(inner, FAMILY, tmp_path / "vectors") if caching_enabled() else inner
        return ResolvedEmbedder(spec=spec, embedder=embedder, scheme="voyage", family=FAMILY)

    monkeypatch.setattr("zemble.embedding.preflight.build_embedder", _resolve)

    # 1. With the knob off there is no cache to collapse a repeat into, on either lane.
    monkeypatch.setenv(CACHE_ENV, "0")
    assert not caching_enabled(), "step 1: the documented knob really is off"
    status = embed_status(root)
    assert status.uncached > 1, f"step 1: the fixture must chunk into several, got {status.uncached}"

    # 2. So the report bills every copy, and the build announces exactly that: one answer.
    with caplog.at_level("INFO", logger="zemble.embedding.pricing"):
        ZembleIndex.from_path(root, embedder=inner)
    announced = [record.getMessage() for record in caplog.records if "uncached chunk(s)" in record.getMessage()]
    assert len(announced) == 1, f"step 2: exactly one announcement, got {caplog.records}"
    assert int(announced[0].split()[1]) == status.uncached, (
        f"step 2: the report billed {status.uncached} chunks, the build announced {announced[0]}"
    )
    assert f"~{status.estimated_tokens} tokens" in announced[0], (
        f"step 2: the report estimated {status.estimated_tokens}, the build announced {announced[0]}"
    )

    # 3. And with the cache back on the same tree is billed for fewer, because the buyer changed:
    #    the report follows the buyer down instead of assuming a wrapper is there.
    monkeypatch.setenv(CACHE_ENV, "1")
    assert embed_status(root).estimated_tokens < status.estimated_tokens, (
        "step 3: a caching buyer gives every copy of one text a single provider slot"
    )
