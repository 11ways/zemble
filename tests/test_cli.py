import argparse
import hashlib
import json
import sys
import warnings
from importlib.resources import files
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import FakeEmbedder, make_chunk, write_index_components
from zemble.cli import _build_parser, _cli_main, _maybe_save_index, _run_clear, _via_daemon, main
from zemble.embedding.pricing import CONFIRM_ENV
from zemble.index.create import create_index_from_path
from zemble.types import ContentType, SearchResult
from zemble.version import __version__


@pytest.mark.parametrize(
    "argv",
    [
        ["zemble"],
    ],
)
def test_main_calls_asyncio_run(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """main() delegates to asyncio.run(serve(...)) when no CLI subcommand is given."""
    monkeypatch.setattr(sys, "argv", argv)
    with patch("asyncio.run") as mock_run:
        mock_run.side_effect = lambda coro: coro.close()
        main()
    mock_run.assert_called_once()


def test_embedding_confirmation_bypasses_a_running_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CLI `--yes` decision stays in-process because an existing daemon cannot inherit it."""
    monkeypatch.setenv(CONFIRM_ENV, "1")
    with patch("zemble.daemon.client.call") as call:
        assert _via_daemon("search", {"path": "/tmp/project"}, False, None) is None
    call.assert_not_called()


@pytest.mark.parametrize(
    "argv, expected_in_output",
    [
        (["zemble", "search", "query text", "/some/path"], ["query text", "0.9"]),
        (["zemble", "search", "nothing", "/some/path", "--top-k", "3"], ["No results found"]),
    ],
)
def test_cli_search(
    argv: list[str],
    expected_in_output: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_cli_main search subcommand calls index.search and prints results."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    # An unfiltered view of an index is the index itself, which is what the real one returns.
    fake_index.filtered.return_value = fake_index
    has_results = "No results" not in expected_in_output[0]
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9)] if has_results else []
    monkeypatch.setattr(sys, "argv", argv)
    with patch("zemble.cli.ZembleIndex.from_path", return_value=fake_index):
        _cli_main()
    out = capsys.readouterr().out
    for fragment in expected_in_output:
        assert fragment in out


@pytest.mark.parametrize(
    ("scenario", "expected_stdout", "expected_stderr", "expected_exit_code"),
    [
        ("with_results", ["src/bar.py", "0.8"], None, None),
        ("no_results", ["No related chunks found"], None, None),
        ("unknown_chunk", [], "No indexed file matches 'elsewhere/bar.py'. Did you mean 'src/bar.py'?", 1),
    ],
)
def test_cli_find_related(
    scenario: str,
    expected_stdout: list[str],
    expected_stderr: str | None,
    expected_exit_code: int | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """_cli_main find-related prints results, empty states, and missing-chunk errors."""
    chunk = make_chunk("class Bar: pass", "src/bar.py")
    fake_index = MagicMock()
    fake_index.filtered.return_value = fake_index
    fake_index.chunk_at.return_value = None if scenario == "unknown_chunk" else chunk
    fake_index.chunks_of.return_value = []
    fake_index.indexed_paths.return_value = ["src/bar.py"]
    fake_index.find_related.return_value = [SearchResult(chunk=chunk, score=0.8)] if scenario == "with_results" else []
    file_path = "elsewhere/bar.py" if scenario == "unknown_chunk" else "src/bar.py"
    monkeypatch.setattr(sys, "argv", ["zemble", "find-related", file_path, "1", "/some/path"])
    with patch("zemble.cli.ZembleIndex.from_path", return_value=fake_index):
        if expected_exit_code is None:
            _cli_main()
        else:
            with pytest.raises(SystemExit) as exc_info:
                _cli_main()
            assert exc_info.value.code == expected_exit_code
    captured = capsys.readouterr()
    for fragment in expected_stdout:
        assert fragment in captured.out
    if expected_stderr:
        assert expected_stderr in captured.err


@pytest.mark.parametrize("argv", [["zemble", "--version"], ["zemble", "-V"]])
def test_cli_version(argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """--version and -V print the package version and exit 0, via both _cli_main and main()."""
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == __version__


def test_main_dispatches_to_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main() routes to _cli_main when first argument is a CLI subcommand."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9)]
    monkeypatch.setattr(sys, "argv", ["zemble", "search", "query text", "/some/path"])
    with patch("zemble.cli.ZembleIndex.from_path", return_value=fake_index):
        main()
    assert "query text" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "expected_stdout", "expect_system_exit"),
    [
        (["zemble", "--help"], "find-related", True),
        (["zemble", "search", "query", "/some/path"], "query", False),
    ],
)
def test_cli_entrypoint_works_without_mcp_installed(
    argv: list[str],
    expected_stdout: str,
    expect_system_exit: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI entrypoint paths succeed even when the mcp package is not installed."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9)]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.server", None)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
    monkeypatch.setitem(sys.modules, "zemble.mcp", None)
    with patch("zemble.cli.ZembleIndex.from_path", return_value=fake_index):
        if expect_system_exit:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        else:
            main()
    assert expected_stdout in capsys.readouterr().out


def test_mcp_main_exits_with_message_when_extras_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """_mcp_main prints an actionable message and exits when mcp extras are not installed."""
    monkeypatch.setattr(sys, "argv", ["zemble"])
    with patch("zemble.cli.find_spec", return_value=None):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 1
    assert "pip install 'zemble[mcp]'" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("command", "argv"),
    [
        ("search", ["zemble", "search", "query", "/no/such/path"]),
        ("find-related", ["zemble", "find-related", "src/foo.py", "1", "/no/such/path"]),
    ],
)
def test_cli_path_not_found(
    command: str, argv: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """index, search, and find-related exit 1 with a friendly message when the path does not exist."""
    monkeypatch.setattr(sys, "argv", argv)
    with patch("zemble.cli._build_index", side_effect=FileNotFoundError("Path does not exist: /no/such/path")):
        with pytest.raises(SystemExit) as exc_info:
            _cli_main()
    assert exc_info.value.code == 1
    assert "Path does not exist" in capsys.readouterr().err


def test_include_text_files_cli_deprecated(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--include-text-files on CLI raises DeprecationWarning."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9)]
    monkeypatch.setattr(sys, "argv", ["zemble", "search", "query", "/some/path", "--include-text-files"])
    with patch("zemble.cli.ZembleIndex.from_path", return_value=fake_index):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _cli_main()
    assert any(
        "include-text-files" in str(w.message).lower() for w in caught if issubclass(w.category, DeprecationWarning)
    )


@pytest.mark.parametrize(
    ("argv_content", "expected"),
    [
        (["--content", "code"], [ContentType.CODE]),
        (["--content", "code", "docs"], [ContentType.CODE, ContentType.DOCS]),
        (["--content", "all"], [ContentType.CODE, ContentType.DOCS, ContentType.CONFIG]),
        (["--content", "code", "all"], [ContentType.CODE, ContentType.DOCS, ContentType.CONFIG]),
        ([], [ContentType.CODE]),
    ],
)
def test_cli_content_argument(
    argv_content: list[str],
    expected: list[ContentType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--content parses into the right ContentType list (including the 'all' shorthand and default)."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    fake_index.search.return_value = [SearchResult(chunk=chunk, score=0.9)]
    monkeypatch.setattr(sys, "argv", ["zemble", "search", "query", "/some/path", *argv_content])
    with patch("zemble.cli.ZembleIndex.from_path", return_value=fake_index) as mock_from_path:
        _cli_main()
    assert list(mock_from_path.call_args.kwargs["content"]) == expected


def test_maybe_save_index_logs_error_on_save_failure(capsys: pytest.CaptureFixture[str]) -> None:
    """_maybe_save_index prints to stderr when cache persistence fails."""
    fake_index = MagicMock()
    with patch("zemble.cli.save_index_to_cache", side_effect=OSError("disk full")):
        _maybe_save_index(fake_index, "/some/path")
    assert "Error saving index" in capsys.readouterr().err


def test_agent_file_tools_are_bash_only() -> None:
    """The agent file must list only Bash and Read — no MCP tools that require schema loading."""
    frontmatter = files("zemble").joinpath("agents/claude.md").read_text(encoding="utf-8").split("---")[1]
    tools_line = next(line for line in frontmatter.splitlines() if line.startswith("tools:"))
    tools = [t.strip() for t in tools_line.removeprefix("tools:").split(",")]
    assert set(tools) == {"Bash", "Read"}, f"Unexpected tools in agent file: {tools}"
    assert not any("mcp__" in t for t in tools)


def _make_valid_index_dir(
    cache_folder: Path, sha: str = "a" * 64, metadata: str = "{}", index_name: str = "index"
) -> Path:
    """Create a fake valid index directory with the expected structure."""
    index_dir = cache_folder / sha / index_name
    # Create the files that PersistencePath.non_existing checks
    write_index_components(index_dir)
    (index_dir / "metadata.json").write_text(metadata)
    return index_dir


@pytest.mark.parametrize(
    ("scenario", "expected_in_output"),
    [
        ("valid", ["Cleared index", "a" * 64, "b" * 64]),
        ("empty", ["No indexes found"]),
        ("non_sha", ["No indexes found"]),
        ("incomplete", ["No indexes found"]),
    ],
)
def test_run_clear_index(
    scenario: str, expected_in_output: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """_run_clear('index') finds valid indexes, and skips non-SHA/incomplete/empty dirs."""
    if scenario == "valid":
        _make_valid_index_dir(tmp_path, "a" * 64)
        _make_valid_index_dir(tmp_path, "b" * 64, index_name="index-docs")
    elif scenario == "non_sha":
        bad_dir = tmp_path / "not-a-sha" / "index"
        write_index_components(bad_dir)
        (bad_dir / "metadata.json").write_text("{}")
    elif scenario == "incomplete":
        index_dir = tmp_path / ("c" * 64) / "index"
        index_dir.mkdir(parents=True)

    with patch("zemble.cli.resolve_cache_folder", return_value=tmp_path):
        _run_clear("index")

    out = capsys.readouterr().out
    for fragment in expected_in_output:
        assert fragment in out

    if scenario == "valid":
        assert not (tmp_path / ("a" * 64)).exists()
        assert not (tmp_path / ("b" * 64)).exists()


@pytest.mark.parametrize(
    "scenario",
    ["orphan", "live", "mismatched_key", "no_root_path"],
)
def test_run_clear_orphans(scenario: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """_run_clear('orphans') removes entries whose local root is gone, and keeps everything else."""
    cache_folder = tmp_path / "cache"
    cache_folder.mkdir()
    root = tmp_path / "repo"
    root.mkdir()
    sha = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()
    if scenario == "orphan":
        _make_valid_index_dir(cache_folder, sha, metadata=json.dumps({"root_path": str(root)}), index_name="index-docs")
        root.rmdir()
    elif scenario == "live":
        _make_valid_index_dir(cache_folder, sha, metadata=json.dumps({"root_path": str(root)}))
    elif scenario == "mismatched_key":
        # A git-URL entry: the dir name hashes the URL, not the (missing) root_path
        _make_valid_index_dir(cache_folder, "a" * 64, metadata=json.dumps({"root_path": str(root / "clone")}))
    elif scenario == "no_root_path":
        _make_valid_index_dir(cache_folder, "b" * 64)

    with patch("zemble.cli.resolve_cache_folder", return_value=cache_folder):
        _run_clear("orphans")

    out = capsys.readouterr().out
    if scenario == "orphan":
        assert str(root) in out
        assert not (cache_folder / sha).exists()
    else:
        assert "No orphaned indexes found" in out
        assert len(list(cache_folder.iterdir())) == 1


def test_run_clear_orphans_skips_invalid_metadata(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Malformed cache entries are skipped without aborting the rest of the cleanup."""
    cache_folder = tmp_path / "cache"
    cache_folder.mkdir()
    root = tmp_path / "repo"
    root.mkdir()
    sha = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()
    _make_valid_index_dir(cache_folder, sha, metadata=json.dumps({"root_path": str(root)}))
    root.rmdir()
    _make_valid_index_dir(cache_folder, "f" * 64, metadata=json.dumps({"root_path": 123}))
    _make_valid_index_dir(cache_folder, "0" * 64, metadata="not json")
    _make_valid_index_dir(cache_folder, "1" * 64, metadata="[]")
    (cache_folder / "not-a-sha" / "index").mkdir(parents=True)

    with patch("zemble.cli.resolve_cache_folder", return_value=cache_folder):
        _run_clear("orphans")

    out = capsys.readouterr().out
    assert str(root) in out
    assert not (cache_folder / sha).exists()
    for kept in ("f" * 64, "0" * 64, "1" * 64, "not-a-sha"):
        assert (cache_folder / kept).exists()


@pytest.mark.parametrize(
    ("create_file", "expected"),
    [
        (True, "Cleared savings"),
        (False, "No savings file found"),
    ],
)
def test_run_clear_savings(
    create_file: bool, expected: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """_run_clear('savings') deletes the file when present, reports missing otherwise."""
    savings_file = tmp_path / "savings.jsonl"
    if create_file:
        savings_file.write_text('{"tokens": 100}\n')

    with patch("zemble.cli.resolve_cache_folder", return_value=tmp_path):
        _run_clear("savings")

    if create_file:
        assert not savings_file.exists()
    out = capsys.readouterr().out
    assert expected in out


@pytest.mark.parametrize(
    ("populate", "expected_fragments"),
    [
        (True, ["Cleared index", "d" * 64, "Cleared savings"]),
        (False, ["No indexes found", "No savings file found"]),
    ],
)
def test_run_clear_all(
    populate: bool, expected_fragments: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """_run_clear('all') handles both indexes and savings."""
    if populate:
        _make_valid_index_dir(tmp_path, "d" * 64)
        (tmp_path / "savings.jsonl").write_text('{"tokens": 50}\n')

    with patch("zemble.cli.resolve_cache_folder", return_value=tmp_path):
        _run_clear("all")

    out = capsys.readouterr().out
    for fragment in expected_fragments:
        assert fragment in out

    if populate:
        assert not (tmp_path / ("d" * 64)).exists()
        assert not (tmp_path / "savings.jsonl").exists()


@pytest.mark.parametrize(
    ("subcommand", "setup_index", "setup_savings", "expected_fragments"),
    [
        ("index", True, False, ["Cleared index", "e" * 64]),
        ("savings", False, True, ["Cleared savings"]),
        ("all", True, True, ["Cleared index", "Cleared savings"]),
        ("orphans", False, False, ["No orphaned indexes found"]),
    ],
)
def test_cli_clear_command(
    subcommand: str,
    setup_index: bool,
    setup_savings: bool,
    expected_fragments: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The `zemble clear <subcommand>` CLI dispatches to _run_clear correctly."""
    sha = "e" * 64
    if setup_index:
        _make_valid_index_dir(tmp_path, sha)
    savings_file = tmp_path / "savings.jsonl"
    if setup_savings:
        savings_file.write_text('{"tokens": 200}\n')

    monkeypatch.setattr(sys, "argv", ["zemble", "clear", subcommand])
    with patch("zemble.cli.resolve_cache_folder", return_value=tmp_path):
        _cli_main()

    out = capsys.readouterr().out
    for fragment in expected_fragments:
        assert fragment in out

    if setup_index:
        assert not (tmp_path / sha).exists()
    if setup_savings:
        assert not savings_file.exists()


def test_cli_search_filters_the_results(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """`--paths` and `--exclude` are parsed and applied to the index that answers."""
    chunk = make_chunk("def foo(): pass", "src/foo.py")
    fake_index = MagicMock()
    view = MagicMock()
    view.search.return_value = [SearchResult(chunk=chunk, score=0.9)]
    fake_index.filtered.return_value = view
    monkeypatch.setattr(
        sys,
        "argv",
        ["zemble", "search", "query", "/some/path", "--paths", "src", "--exclude", "vendor/", "*.min.js"],
    )
    with patch("zemble.cli.ZembleIndex.from_path", return_value=fake_index):
        _cli_main()
    fake_index.filtered.assert_called_once_with(["src"], ["vendor/", "*.min.js"])
    assert "src/foo.py" in capsys.readouterr().out, "the filtered view answered"


def test_cli_reports_a_daemon_refusal_without_rebuilding(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal ends the command: answering it in this process would refuse identically, slowly."""
    from zemble.daemon import client
    from zemble.daemon.protocol import CommandRefused

    def _refuse(*args: object, **kwargs: object) -> object:
        raise CommandRefused("Refusing to index /some/path: too big")

    monkeypatch.setattr(client, "call", _refuse)
    monkeypatch.setattr(client, "_disabled_reason", None)
    monkeypatch.delenv("ZEMBLE_DAEMON", raising=False)
    with pytest.raises(SystemExit) as exit_code:
        _via_daemon("search", {"path": "/some/path"}, False, None)
    assert exit_code.value.code == 1, "the refusal is a failed command, not a fallback"


def _leaf_commands(
    parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], argparse.ArgumentParser]:
    """Return every LEAF subcommand, keyed by the words a user types to reach it.

    Keyed by leaf because a parent is only a grouping: proving `graph build` never indexes says
    nothing about the twelve query commands beside it.
    """
    leaves: dict[tuple[str, ...], argparse.ArgumentParser] = {}
    nested = [action for action in parser._actions if isinstance(action, argparse._SubParsersAction)]
    if not nested:
        return {prefix: parser} if prefix else {}
    for action in nested:
        for name, child in action.choices.items():
            leaves.update(_leaf_commands(child, (*prefix, name)))
    return leaves


def _takes(subparser: argparse.ArgumentParser, option: str) -> bool:
    """Return whether a subcommand declares that argument itself."""
    return any(option in action.option_strings or action.dest == option for action in subparser._actions)


#: One runnable invocation per LEAF subcommand that names a workspace root but is claimed never
#: to build an index. The claim is PROVED by running it with the build seam trip-wired, so a
#: command that grows a build fails here instead of telling a user to pass a flag it lacks.
_NEVER_BUILDS_PROOFS: dict[tuple[str, ...], list[str]] = {
    ("outline",): ["outline", "{root}", "auth.py", "--no-daemon"],
    ("signatures",): ["signatures", "{root}", "authenticate", "--no-daemon"],
    ("graph", "build"): ["graph", "build", "{root}", "--no-daemon"],
    ("graph", "facts", "status"): ["graph", "facts", "status", "{root}", "--no-daemon"],
    **{
        ("graph", command): ["graph", command, "{root}", "authenticate", "--no-daemon"]
        for command in ("definition", "callers", "callees", "references", "implementations")
    },
    **{
        ("graph", command): ["graph", command, "{root}", "authenticate", "--no-daemon"]
        for command in ("supertypes", "overrides-of", "overridden-by", "tests-of", "neighbors")
    },
    ("dupes",): ["dupes", "{root}"],
    ("embed-status",): ["embed-status", "{root}"],
}


def test_every_subcommand_that_can_build_accepts_the_yes_its_refusals_advertise(
    tmp_project: Path,
    graph_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every refusal ends with "or pass --yes", so every command that can be refused must take it.

    `zemble home ... --yes` exited 2 with `unrecognized arguments` for as long as the flag was
    on three subcommands only. Nothing here is asserted from a list of names: a leaf subcommand
    is exempt only when it declares no filesystem root through `add_root_arg` - the marker, not
    the spelling, so a command naming its tree `repo` is classified like every other - or when
    running it proves it never reaches the one seam every index build passes.
    """
    from zemble.cli import names_a_root
    from zemble.embedding.registry import ResolvedEmbedder
    from zemble.index import ZembleIndex

    built: list[str] = []
    real_create = create_index_from_path

    def _trip_wire(path, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        built.append(str(path))
        return real_create(path, *args, **kwargs)

    monkeypatch.setattr("zemble.index.create.create_index_from_path", _trip_wire)
    monkeypatch.setattr("zemble.index.index.create_index_from_path", _trip_wire)
    monkeypatch.setattr(
        "zemble.embedding.preflight.build_embedder",
        lambda spec: ResolvedEmbedder(spec=spec, embedder=FakeEmbedder(), scheme="fake", family="fake:test"),
    )

    # 1. The trip wire really is on the seam a build passes, or every proof below is vacuous.
    ZembleIndex.from_path(tmp_project, embedder=FakeEmbedder())
    assert built, "step 1: the trip wire must fire on a real build"
    built.clear()

    # 2. Every leaf is classified, and a new one cannot slip through unclassified.
    leaves = _leaf_commands(_build_parser())
    assert ("graph", "callers") in leaves and ("home",) in leaves, (
        f"step 2: leaves not enumerated, got {sorted(leaves)}"
    )
    for words, subparser in sorted(leaves.items()):
        if _takes(subparser, "confirm_embedding"):
            continue
        if not names_a_root(subparser):
            continue  # It names no tree, so it cannot build one.
        assert words in _NEVER_BUILDS_PROOFS, (
            f"step 2: {' '.join(words)!r} names a workspace root but neither takes --yes nor proves it never builds"
        )

    # 3. Every proof names a leaf that still exists, so a renamed command cannot leave a
    #    proof standing over nothing.
    assert set(_NEVER_BUILDS_PROOFS) <= set(leaves), (
        f"step 3: proofs for commands that no longer exist: {sorted(set(_NEVER_BUILDS_PROOFS) - set(leaves))}"
    )

    # 4. Each claimed non-builder is run for real: it must answer, and the seam must stay untouched.
    for words, template in sorted(_NEVER_BUILDS_PROOFS.items()):
        argv = ["zemble", *(part.format(root=str(tmp_project)) for part in template)]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            _cli_main()
        streams = capsys.readouterr()
        name = " ".join(words)
        assert "usage:" not in streams.err, f"step 4: {name!r} never ran, argparse rejected {argv}: {streams.err}"
        assert streams.out.strip(), f"step 4: {name!r} answered nothing, so it proves nothing"
        assert built == [], f"step 4: {name!r} built an index, so it must accept --yes; built {built}"
