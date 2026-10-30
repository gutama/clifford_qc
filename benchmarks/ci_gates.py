"""One CI gate registry, conservative change routing and receipt-producing runs.

No campaign result is produced here. Every command invokes its existing checker
or report checker, with the same record-local environment and timeout policy.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
from functools import lru_cache
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

from benchmarks.experiment_artifacts import local_path, read_json
from benchmarks.experiment_runner import Task, run

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "benchmarks/ci_gates.json"
TIERS = {"core", "structural", "sampled", "environment"}
TRANSPORT = {"benchmarks/experiment_runner.py", "benchmarks/experiment_artifacts.py",
             "benchmarks/ci_gates.py", "benchmarks/ci_gates.json"}


def execution_environment(relative, *, root=ROOT):
    value = read_json(local_path(root, relative))["execution_environment"]
    if (set(value) != {"python", "numpy", "scipy", "threads"}
            or type(value["threads"]) is not int or value["threads"] != 1
            or any(not isinstance(value[p], str) or not re.fullmatch(r"\d+\.\d+\.\d+", value[p])
                   for p in ("python", "numpy", "scipy"))):
        raise ValueError("invalid declared execution environment")
    return value


def verify_execution_environment(value):
    observed = {"python": platform.python_version(),
                **{p: importlib.metadata.version(p) for p in ("numpy", "scipy")}}
    if any(observed[p] != value[p] for p in observed):
        raise ValueError("declared replay runtime differs")
    if any(os.environ.get(p) != str(value["threads"]) for p in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")):
        raise ValueError("declared replay thread settings differ")


def load_registry(path=REGISTRY, *, root=ROOT):
    registry = read_json(path)
    if set(registry) != {"schema", "gates"} or registry["schema"] != "clifford_qc.ci_gates.v1":
        raise ValueError("invalid CI registry")
    gates = registry["gates"]
    if not isinstance(gates, list) or not gates or len({g["gate"] for g in gates}) != len(gates):
        raise ValueError("empty or duplicate CI gates")
    for gate in gates:
        required = {"gate", "tier", "commands", "environment_record", "timeout", "watch"}
        if (set(gate) not in (required, required | {"execution_declaration"})
                or not re.fullmatch(r"[a-z][a-z0-9_]*", gate["gate"])
                or gate["tier"] not in TIERS or type(gate["timeout"]) is not int
                or not 1 <= gate["timeout"] <= 350 or not gate["commands"]
                or not gate["watch"] or any(not isinstance(p, str) or not p for p in gate["watch"])):
            raise ValueError("invalid gate contract")
        if "execution_declaration" in gate:
            execution_environment(gate["execution_declaration"], root=root)
        record = local_path(root, "benchmarks/reference_results/" + gate["environment_record"])
        if not record.is_file():
            raise ValueError("missing record-local environment")
        for command in gate["commands"]:
            if (not isinstance(command, list) or not command
                    or any(not isinstance(p, str) or not p or "\0" in p for p in command)
                    or not command[0].endswith(".py") or not local_path(root, command[0]).is_file()):
                raise ValueError("invalid checker argv")
    return gates


@lru_cache(maxsize=2048)
def _imports(path, modified_ns, size):
    """Parse imports without executing a checker or importing optional extras."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend((alias.name, 0, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append((node.module or "", node.level, tuple(a.name for a in node.names)))
    return tuple(result)


@lru_cache(maxsize=2048)
def _edges(root, path, modified_ns, size):
    root, path = Path(root), Path(path)
    edges = set()
    for module, level, aliases in _imports(str(path), modified_ns, size):
        if level:
            parent = path.parent
            for _ in range(level-1):
                parent = parent.parent
            bases = [parent]
        else:
            bases = [root, path.parent]
        for base in bases:
            prefix = base.joinpath(*module.split(".")) if module else base
            candidates = [prefix.with_suffix(".py"), prefix/"__init__.py"]
            candidates.extend(prefix/str(alias).replace(".", "/") for alias in aliases)
            for candidate in candidates:
                for target in (candidate, candidate.with_suffix(".py"), candidate/"__init__.py"):
                    if target.is_file() and target.suffix == ".py" and root in target.resolve().parents:
                        edges.add(target.resolve())
    return tuple(edges)


def dependencies(root, commands):
    """Conservative transitive local-import closure, including imports in functions.

    Explicit watch paths still cover data, declarations, dynamic source snapshots
    and deleted files. Unknown dependencies retain the all-campaign fallback.
    """
    root = Path(root).resolve()
    pending = [local_path(root, command[0]) for command in commands]
    found = set()
    while pending:
        path = pending.pop()
        relative = path.relative_to(root).as_posix()
        if relative in found:
            continue
        found.add(relative)
        stat = path.stat()
        pending.extend(_edges(str(root), str(path), stat.st_mtime_ns, stat.st_size))
    return found


def affected(gates, changed, *, full=False, root=ROOT):
    """Unknown non-document paths fail closed to every campaign, including shots.

    Shared production sources affect all campaigns. Transport and workflow
    edits select all deterministic gates; their invariants are checked by the
    always-running suite. Frozen scientific replay remains byte-bound to its
    historical runner and is not rewritten by a transport change.
    """
    selected = {g["gate"] for g in gates if g["tier"] == "core"}
    for path in changed:
        if (Path(path).is_absolute() or ".." in Path(path).parts or "\\" in path or not path):
            raise ValueError("invalid changed path")
    if full:
        return [g for g in gates if g["tier"] != "environment"]
    # Resolve the dependency graph once per selection, not once per changed
    # path. A deleted source selects full replay so removing a helper cannot
    # prune the very dependency edge that should select its consumers.
    if any(path.endswith(".py") and not (Path(root)/path).is_file() for path in changed):
        return [g for g in gates if g["tier"] != "environment"]
    try:
        closures = {g["gate"]: dependencies(root, g["commands"]) for g in gates}
    except (OSError, SyntaxError, ValueError):
        return [g for g in gates if g["tier"] != "environment"]
    for path in changed:
        matches = [g for g in gates if path in closures[g["gate"]]
                   or any(fnmatch.fnmatchcase(path, p) for p in g["watch"])]
        selected.update(g["gate"] for g in matches if g["tier"] != "environment")
        transport = (path in TRANSPORT or path.startswith(".github/") or path == ".gitignore"
                     or path in {"tests/test_experiment_runner.py", "tests/test_ci_gates.py"})
        shared = (path.startswith(("clifford_qc/", "molecular/", "benchmarks/provenance/"))
                  or path in {"pyproject.toml", "setup.cfg", "setup.py", "conftest.py",
                              "docs/restart/blueprint_2026-10-24.md",
                              "docs/restart/v1_evidence_snapshot.json"})
        if transport:
            selected.update(g["gate"] for g in gates if g["tier"] == "structural")
        elif shared:
            selected.update(g["gate"] for g in gates if g["tier"] in {"structural", "sampled"})
        elif path.startswith("tests/") and not matches:
            selected.update(g["gate"] for g in gates if g["tier"] == "structural")
        elif not matches and not (path.endswith((".md", ".rst", ".txt"))
                                  and not path.startswith("benchmarks/")):
            selected.update(g["gate"] for g in gates if g["tier"] in {"structural", "sampled"})
    return [g for g in gates if g["gate"] in selected]


def changed_paths(root, base, head):
    if any(not re.fullmatch(r"[0-9a-f]{40,64}", value or "") for value in (base, head)):
        raise ValueError("diff requires full commit identities")
    ancestor = subprocess.check_output(["git", "merge-base", base, head], cwd=root, text=True).strip()
    raw = subprocess.check_output(["git", "diff", "--no-renames", "--name-only", "-z", ancestor, head], cwd=root)
    return raw.decode("utf-8").rstrip("\0").split("\0") if raw else []


def plan(event, payload, *, gates=None, root=ROOT):
    gates = load_registry() if gates is None else gates
    if event == "pull_request":
        try:
            paths = changed_paths(root, payload["pull_request"]["base"]["sha"],
                                  payload["pull_request"]["head"]["sha"])
        except (KeyError, ValueError, OSError, subprocess.CalledProcessError):
            # Missing ancestry must spend checks, never silently skip them.
            chosen = affected(gates, [], full=True)
        else:
            chosen = affected(gates, paths)
    elif event == "push":
        # Every main update keeps the full deterministic safety net.
        chosen = [g for g in gates if g["tier"] in {"core", "structural"}]
        try:
            paths = changed_paths(root, payload["before"], payload["after"])
            chosen += [g for g in affected(gates, paths) if g["tier"] == "sampled"]
        except (KeyError, ValueError, OSError, subprocess.CalledProcessError):
            chosen = affected(gates, [], full=True)
    elif event in {"schedule", "workflow_dispatch"}:
        chosen = affected(gates, [], full=True)
    else:
        raise ValueError("unsupported CI event")
    result = {}
    for tier in ("structural", "sampled"):
        rows = [{"gate": g["gate"], "timeout": g["timeout"] + 2,
                 "execution_declaration": g.get("execution_declaration", ""),
                 "environment_record": g["environment_record"]} for g in chosen if g["tier"] == tier]
        result[tier] = {"include": rows}
        result["has_" + tier] = bool(rows)
    return result


def execute(gates, *, output, root=ROOT):
    tasks = tuple(Task(g["gate"] + (f"_{i}" if len(g["commands"]) > 1 else ""),
                       (sys.executable, *command), g["timeout"] * 60)
                  for g in gates for i, command in enumerate(g["commands"]))
    # Bind current tracked bytes, including evidence and helper dependencies.
    # Receipts record maintenance inputs; archived evidence identities stay in
    # their independent checkers, not inferred from this source closure.
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    sources = tuple(p for p in raw.decode().rstrip("\0").split("\0") if p)
    return run(tasks, root=root, destination=output, sources=sources)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--plan", action="store_true")
    choice.add_argument("--gate")
    choice.add_argument("--tier", choices=sorted(TIERS))
    choice.add_argument("--environment", help="read a frozen declaration's execution environment")
    parser.add_argument("--emit", choices=("python", "constraints", "verify"), default="verify")
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME"))
    parser.add_argument("--event-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.environment:
        value = execution_environment(args.environment)
        if args.emit == "python":
            print(value["python"])
        elif args.emit == "constraints":
            print("\n".join(p + "==" + value[p] for p in ("numpy", "scipy")))
        else:
            verify_execution_environment(value)
            print("declared execution environment matches")
        return 0
    gates = load_registry()
    if args.plan:
        payload = read_json(args.event_file) if args.event_file else {}
        for key, value in plan(args.event, payload, gates=gates).items():
            print(key + "=" + json.dumps(value, separators=(",", ":")))
        return 0
    selected = [g for g in gates if g["gate"] == args.gate or g["tier"] == args.tier]
    if not selected or args.output is None:
        parser.error("select an existing gate/tier and provide --output")
    receipt = execute(selected, output=args.output)
    return int(receipt["summary"]["failed"] != 0)


if __name__ == "__main__":
    sys.exit(main())
