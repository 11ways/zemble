from pathlib import Path

import pytest

from zemble.index.file_walker import walk_entries, walk_files


def _touch(path: Path, content: str = "x = 1\n") -> None:
    """Create path (and any missing parents) and write content to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.mark.parametrize(
    ("files", "gitignore", "zembleignore", "expected"),
    [
        # Default-ignored dirs (.venv, node_modules, .cache) are always skipped.
        (
            ["src/a.py", ".venv/lib/b.py", "node_modules/pkg/c.py", ".cache/uv/d.py"],
            None,
            None,
            {"src/a.py"},
        ),
        # Root .gitignore excludes both directories and files.
        (
            ["src/keep.py", "local/ignored.py", "generated.py"],
            "local/\ngenerated.py\n# comment",
            None,
            {"src/keep.py"},
        ),
        # Negation (`!`) patterns re-include previously ignored files.
        (
            ["out/a.py", "out/keep.py"],
            "out/*\n!out/keep.py\n",
            None,
            {"out/keep.py"},
        ),
        # Allow-list style gitignore (`*` + `!*/` + `!*.py`) must not prune subdirs.
        (
            ["main.py", "internal/pkg/foo.py", "internal/pkg/bar.py"],
            "*\n!*/\n!*.py\n",
            None,
            {"main.py", "internal/pkg/foo.py", "internal/pkg/bar.py"},
        ),
        # Ignored-parent negation: out/* prunes out/deep/, so out/deep/keep.py must not leak.
        (
            ["out/deep/keep.py"],
            "out/*\n!out/deep/keep.py\n",
            None,
            set(),
        ),
        # Ignored-parent negation: out/* prunes out/deep/, so out/deep/keep.py must not leak.
        (
            ["out/deep/keep.py"],
            None,
            "out/*\n!out/deep/keep.py\n",
            set(),
        ),
        # Explicit file negation bypasses extension filter: !special.kjs is yielded even if .kjs is not in extensions.
        (
            ["special.kjs", "other.kjs", "main.py"],
            None,
            "*.kjs\n!special.kjs\n",
            {"main.py", "special.kjs"},
        ),
        # Glob negation without suffix does NOT bypass extension filter.
        (
            [".github/workflows/ci.yaml", "src/main.py"],
            None,
            "!.github/*\n",
            {"src/main.py"},
        ),
        # Directory negation does NOT bypass extension filter: files inside vendor/ still need a matching extension.
        (
            ["vendor/special.kjs", "vendor/main.py"],
            None,
            "*\n!vendor/\n",
            {"vendor/main.py"},
        ),
    ],
)
def test_walk_files_filtering(
    tmp_path: Path, files: list[str], gitignore: str | None, zembleignore: str | None, expected: set[str]
) -> None:
    """Directory defaults, gitignore patterns, and negations filter the yielded files."""
    for rel in files:
        _touch(tmp_path / rel)
    if gitignore is not None:
        (tmp_path / ".gitignore").write_text(gitignore)
    if zembleignore is not None:
        (tmp_path / ".zembleignore").write_text(zembleignore)

    found = {p.relative_to(tmp_path).as_posix() for p in walk_files(tmp_path, [".py"])}
    assert found == expected


def test_walk_files_prunes_ignored_dirs(tmp_path: Path) -> None:
    """Ignored directories are pruned so os.walk never descends into them."""
    _touch(tmp_path / "src" / "a.py")
    _touch(tmp_path / "node_modules" / "deep" / "deeper" / "b.js")

    visited = list(walk_files(tmp_path, [".py", ".js"]))
    assert not any("node_modules" in str(v) for v in visited), visited


def _reference_walk(root: Path, extensions: list[str], ignore: list[str] | None = None) -> list[Path]:  # noqa: C901
    """The pre-scandir walker, kept verbatim as the oracle for the fast one."""
    from pathspec import GitIgnoreSpec

    from zemble.index.file_walker import _DEFAULT_IGNORED_DIRS, _load_ignore_for_dir

    def is_ignored(path: Path, specs: list[tuple[Path, GitIgnoreSpec]]) -> tuple[bool, bool]:
        is_dir = path.is_dir()
        ignored = False
        found = False
        for base, spec in specs:
            try:
                relative = path.relative_to(base)
            except ValueError:
                continue
            relative_str = relative.as_posix() + ("/" if is_dir else "")
            for pattern in spec.patterns:
                if pattern.include is None:
                    continue
                if pattern.match_file(relative_str) is not None:
                    ignored = pattern.include
                    raw = pattern.pattern
                    found = not ignored and isinstance(raw, str) and bool(Path(raw.rstrip("/")).suffix)
        return ignored, found

    def walk(directory: Path, specs: list[tuple[Path, GitIgnoreSpec]]) -> list[Path]:
        loaded = _load_ignore_for_dir(str(directory))
        if loaded is not None:
            specs = [*specs, (directory, loaded[0])]
        out: list[Path] = []
        for item in sorted(directory.iterdir()):
            if item.is_symlink():
                continue
            ignored, found = is_ignored(item, specs)
            if ignored:
                continue
            if item.is_dir():
                out.extend(walk(item, specs))
            elif item.is_file() and (found or item.suffix.lower() in set(extensions)):
                out.append(item)
        return out

    base = GitIgnoreSpec.from_lines(sorted(_DEFAULT_IGNORED_DIRS) + list(ignore or []), backend="simple")
    return walk(root, [(root, base)])


def test_walk_matches_the_reference_walker(tmp_path: Path) -> None:
    """The scandir walker yields exactly the files, in the order, the path-based one did."""
    # 1. A tree with nested ignore files, negations, default-ignored dirs and a symlink.
    for rel in [
        "src/main.py",
        "src/nested/deep/keep.py",
        "src/nested/deep/skip.py",
        "src/nested/vendor/lib.py",
        "docs/readme.md",
        "build/out.py",
        "node_modules/pkg/index.js",
        "tools/special.kjs",
        "tools/other.kjs",
        "tools/run.py",
    ]:
        _touch(tmp_path / rel)
    (tmp_path / ".gitignore").write_text("*.md\ndocs/\n")
    (tmp_path / "src" / "nested" / ".gitignore").write_text("deep/skip.py\nvendor/\n")
    (tmp_path / "tools" / ".zembleignore").write_text("*.kjs\n!special.kjs\n")
    (tmp_path / "linked.py").symlink_to(tmp_path / "src" / "main.py")

    # 2. Both walkers must agree exactly, order included: the chunk order depends on it.
    fast = list(walk_files(tmp_path, [".py", ".md"]))
    reference = _reference_walk(tmp_path, [".py", ".md"])
    assert fast == reference, "the walkers disagree about which files exist or in what order"
    assert {p.name for p in fast} == {"main.py", "keep.py", "special.kjs", "run.py"}

    # 3. The relative paths handed out alongside are the ones a manual relative_to would give.
    entries = list(walk_entries(tmp_path, [".py", ".md"]))
    assert [entry.relative_path for entry in entries] == [str(p.relative_to(tmp_path)) for p in reference]


def test_walk_files_skips_symlinks(tmp_path: Path) -> None:
    """Symlinked files and directories are skipped; real paths are still walked."""
    # Real directory with a file
    real_dir = tmp_path / "real_pkg" / "src"
    _touch(real_dir / "mod.py")

    # A symlink to that directory from another location
    link_parent = tmp_path / "wrapper" / "src"
    link_parent.mkdir(parents=True)
    (link_parent / "linked").symlink_to(real_dir)

    # A symlink to a single file
    _touch(tmp_path / "original.py")
    (tmp_path / "link_to_original.py").symlink_to(tmp_path / "original.py")

    found = {p.relative_to(tmp_path).as_posix() for p in walk_files(tmp_path, [".py"])}

    # Real paths are present
    assert "real_pkg/src/mod.py" in found
    assert "original.py" in found

    # Symlink-based paths are absent
    assert "wrapper/src/linked/mod.py" not in found
    assert "link_to_original.py" not in found
