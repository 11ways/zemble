"""Replay pinned pre-removal declarations without modifying any source checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def git(repo: Path, *args: str) -> str:
    """Run a checked, noninteractive git operation in one owning checkout."""
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def prepare(workspace: Path, replay: Path, dataset: dict[str, Any]) -> None:
    """Restore exact recorded blobs over the module-fit-final tag corpus."""
    if not str(replay.resolve()).startswith("/tmp/"):
        raise ValueError("replay fixtures belong under /tmp, never in a source checkout")
    replay.parent.mkdir(parents=True, exist_ok=True)
    if not replay.exists():
        git(workspace, "worktree", "add", "--detach", str(replay), dataset["workspace_tag"])
    projects = json.loads((workspace / "workspace.json").read_text())["projects"]
    for project in projects:
        name = project["name"]
        if not (replay / name).exists():
            git(workspace / name, "worktree", "add", "--detach", str(replay / name), "module-fit-final-" + name)
    restored = {}
    for pair in dataset["pairs"]:
        source = pair["source"]
        repo, file = source["file"].split("/", 1)
        key = (repo, file)
        if key in restored and restored[key] != source["revision"]:
            raise ValueError(f"conflicting historical versions of {repo}/{file}")
        restored[key] = source["revision"]
        git(replay / repo, "restore", "--source=" + source["revision"], "--staged", "--worktree", "--", file)
        expected = git(workspace / repo, "rev-parse", source["revision"] + ":" + file)
        actual = git(replay / repo, "hash-object", file)
        if expected != actual:
            raise ValueError(f"historical blob mismatch: {source['file']}")
        if not (replay / source["file"]).read_text().splitlines()[source["line"] - 1].strip():
            raise ValueError(f"seed is not a declaration line: {source['file']}")


def rpc(runtime: Path, command: str, args: dict[str, Any], timeout: float = 600) -> dict[str, Any]:
    """Keep completed misses separate from availability errors."""
    started = time.perf_counter()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(str(runtime / "daemon.sock"))
        connection.sendall(
            (
                json.dumps(
                    {
                        "id": 1,
                        "cmd": command,
                        "args": args,
                        "accepts_busy": True,
                        "deadline_ms": min(900000, int(timeout * 1000 - 1000)),
                    }
                )
                + "\n"
            ).encode()
        )
        response = json.loads(connection.makefile("rb").readline())
    if not response.get("ok"):
        raise RuntimeError(f"{command} unavailable: {response.get('error')}")
    return {"seconds": time.perf_counter() - started, "response": response}


def rank(hits: list[dict[str, Any]], targets: list[dict[str, Any]]) -> int | None:
    """Score the canonical declaration rather than mere mentions in copy or rules."""
    for position, hit in enumerate(hits, 1):
        start = hit.get("start_line", hit.get("line", 0))
        end = hit.get("end_line", start)
        if any(hit.get("file_path") == target["file_path"] and start <= target["line"] <= end for target in targets):
            return position
    return None


def main() -> None:
    """Prepare immutable fixtures, run one private daemon, export split-scoped evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, required=True, help="Owning checkouts with module-fit-final tags/history"
    )
    parser.add_argument("--replay", type=Path, required=True, help="Historical corpus under /tmp")
    parser.add_argument("--out", type=Path, required=True, help="Evidence directory under /tmp")
    parser.add_argument("--split", choices=["development", "holdout", "all"], default="development")
    parser.add_argument("--pairs", type=Path, default=Path(__file__).parent / "local" / "historical_recall_pairs.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dupes-report", type=Path, help="Reuse a complete scan of this identical historical corpus")
    parser.add_argument("--dupes-kind", default="all", help="Clone/candidate channel to evaluate; none skips scanning")
    args = parser.parse_args()
    dataset = json.loads(args.pairs.read_text())
    prepare(args.workspace.resolve(), args.replay.resolve(), dataset)
    if args.prepare_only:
        return
    if not str(args.out.resolve()).startswith("/tmp/"):
        raise ValueError("measurement evidence belongs under /tmp")
    args.out.mkdir(parents=True, exist_ok=True)
    check_resources()
    runtime = args.out / "runtime"
    runtime.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "ZEMBLE_DAEMON_DIR": str(runtime),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2",
    }
    env.setdefault("ZEMBLE_CACHE_LOCATION", str(args.out / "cache"))
    log = (args.out / "daemon.log").open("a")
    process = subprocess.Popen(
        [sys.executable, "-m", "zemble.daemon", "run", "--no-watch", "--idle-minutes", "0"],
        env=env,
        stdout=log,
        stderr=log,
    )
    results = []
    try:
        for _ in range(200):
            if (runtime / "daemon.sock").exists():
                break
            if process.poll() is not None:
                raise RuntimeError("measurement daemon failed to start")
            time.sleep(0.1)
        root = str(args.replay.resolve())
        stats = rpc(runtime, "stats", {"path": root, "content": ["code"]}, 900)
        rpc(runtime, "graph", {"path": root, "command": "ensure"}, 900)
        for pair in dataset["pairs"]:
            if args.split != "all" and pair["split"] != args.split:
                continue
            home = rpc(runtime, "home", {"path": root, "description": pair["query"], "top_k": 40})
            related = rpc(
                runtime,
                "find_related",
                {
                    "path": root,
                    "file_path": pair["source"]["file"],
                    "line": pair["source"]["line"],
                    "top_k": 5,
                    "max_snippet_lines": 0,
                },
            )
            home_hits = home["response"]["result"].get("home", {}).get("mechanisms", [])
            related_hits = related["response"]["result"].get("results", [])
            result = {
                "id": pair["id"],
                "split": pair["split"],
                "home_rank": rank(home_hits, pair["targets"]),
                "related_rank": rank(related_hits, pair["targets"]),
                "home": home,
                "related": related,
            }
            results.append(result)
            (args.out / "queries.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({key: result[key] for key in ["id", "split", "home_rank", "related_rank"]}), flush=True)
        summary = {
            "split": args.split,
            "pairs": len(results),
            "home_hits": sum(bool(row["home_rank"]) for row in results),
            "related_hits": sum(bool(row["related_rank"]) for row in results),
            "stats": stats,
            "dataset_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        }
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        subprocess.run([sys.executable, "-m", "zemble.daemon", "stop"], env=env, check=True, timeout=120)
        process.wait(timeout=120)
        log.close()

    if args.dupes_kind != "none":
        evaluate_dupes(args, dataset, results, summary, root)


def check_resources() -> None:
    """Refuse measurement startup below disk/RAM floors or beside another daemon."""
    import shutil

    from zemble.daemon.protocol import socket_path

    available = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < 8 * 1024 * 1024:
        raise RuntimeError("memory floor: wait until at least 8 GiB is available")
    if shutil.disk_usage("/").free < 15 * 1024**3:
        raise RuntimeError("disk floor: need at least 15 GiB free before starting a measurement daemon")
    if socket_path().exists():
        raise RuntimeError("another daemon socket exists; stop it cleanly before this serial measurement")


def evaluate_dupes(args: argparse.Namespace, dataset: dict, results: list, summary: dict, root: str) -> None:
    """Scan after closing the daemon so construction and clone scratch never multiply."""
    if args.dupes_report:
        report = json.loads(args.dupes_report.read_text())
    else:
        from zemble.dedup.detect import DupeOptions, find_duplication
        from zemble.dedup.model import CloneKind
        from zemble.dedup.report import report_json
        from zemble.userenv import load_user_env

        load_user_env()
        kinds = tuple(CloneKind) if args.dupes_kind == "all" else (CloneKind(args.dupes_kind),)
        report = report_json(find_duplication(root, DupeOptions(kinds=kinds, jobs=1)), 1000000)
    (args.out / "dupes.json").write_text(json.dumps(report, indent=2) + "\n")
    by_id = {pair["id"]: pair for pair in dataset["pairs"]}
    for row in results:
        row["dupes_hits"] = duplicate_matches(by_id[row["id"]], report)
    summary["dupes_hits"] = sum(bool(row.get("dupes_hits")) for row in results)
    summary["dupes_kind"] = args.dupes_kind
    (args.out / "queries.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def duplicate_matches(pair: dict[str, Any], report: dict[str, Any]) -> list[str]:
    """Require both removed and replacement declarations in one returned class."""
    source = pair["source"]

    def covers(member: dict[str, Any], file: str, line: int) -> bool:
        return member.get("file_path") == file and member.get("start_line", 0) <= line <= member.get("end_line", 0)

    def old(member: dict[str, Any]) -> bool:
        if source["member"] is None:
            return member.get("file_path") == source["file"]
        return covers(member, source["file"], source["line"])

    return [
        clone["key"]
        for clone in report.get("classes", [])
        if any(old(member) for member in clone["members"])
        and any(
            any(covers(member, target["file_path"], target["line"]) for target in pair["targets"])
            for member in clone["members"]
        )
    ]


if __name__ == "__main__":
    main()
