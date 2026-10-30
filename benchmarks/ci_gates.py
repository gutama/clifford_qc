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
TIERS = {"core", "structural", "sampled"}
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
        if (not required <= set(gate) or not set(gate) <= required | {"execution_declaration", "inputs", "manual_only", "max_output_bytes"}
                or not re.fullmatch(r"[a-z][a-z0-9_]*", gate["gate"])
                or gate["tier"] not in TIERS or type(gate["timeout"]) is not int
                or not 1 <= gate["timeout"] <= 350 or not gate["commands"]
                or not gate["watch"] or any(not isinstance(p, str) or not p for p in gate["watch"])):
            raise ValueError("invalid gate contract")
        if "execution_declaration" in gate:
            execution_environment(gate["execution_declaration"], root=root)
        if gate['gate'] == 'check_r3d_qr3_refinement':
            frozen = read_json(local_path(root, 'benchmarks/configs/r3d_qr3_refinement.json'))
            if (frozen['reporting_contract']['sampled_ci_is_manual_dispatch_only']
                    and gate.get('manual_only') is not True):
                raise ValueError('R3d frozen reporting contract requires manual dispatch')
        if (type(gate.get("manual_only", False)) is not bool
                or type(gate.get("max_output_bytes", 16 * 2**20)) is not int
                or not 0 < gate.get("max_output_bytes", 16 * 2**20) <= 64 * 2**20
                or not isinstance(gate.get("inputs", []), list)):
            raise ValueError("invalid gate execution limits")
        for relative in gate.get("inputs", []):
            local_path(root, relative)
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


def gate_inputs(gate, *, root=ROOT):
    """Declared environment and non-imported byte bindings are dependencies."""
    paths = {'benchmarks/reference_results/' + gate['environment_record']}
    paths.update(gate.get('inputs', []))
    if gate.get('execution_declaration'):
        paths.add(gate['execution_declaration'])
    # Only declarations describe live inputs. Paths mentioned in an old
    # observation/snapshot refer to historical execution, not today's files.
    for relative in tuple(paths):
        if not relative.endswith('_declaration.json'):
            continue
        try:
            value = read_json(local_path(root, relative))
        except (OSError, ValueError):
            continue
        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for key, child in value.items():
                    yield key
                    yield from strings(child)
            elif isinstance(value, list):
                for child in value:
                    yield from strings(child)
        for candidate in strings(value):
            if '/' not in candidate:
                continue
            try:
                if local_path(root, candidate).is_file():
                    paths.add(candidate)
            except (OSError, ValueError):
                pass

    return paths


def affected(gates, changed, *, full=False, root=ROOT):
    """Keep the deterministic safety net; select only relevant sampled work.

    Literal data reads are not a complete import graph, so every deterministic
    gate still runs on every PR. Legacy sampled gates retain manual dispatch.
    The two new replay gates use explicit data/environment and import closures.
    """
    for path in changed:
        if Path(path).is_absolute() or '..' in Path(path).parts or '\\' in path or not path:
            raise ValueError('invalid changed path')
    if full:
        return list(gates)
    selected = {g['gate'] for g in gates if g['tier'] in {'core', 'structural'}}
    try:
        closures = {g['gate']: dependencies(root, g['commands']) | gate_inputs(g, root=root)
                    for g in gates}
    except (OSError, ValueError, SyntaxError):
        return list(gates)
    for path in changed:
        # Test/document/example edits cannot change the sampled instrument.
        routine = (path.startswith(('tests/', 'examples/', 'paper/', 'paper_', 'docs/'))
                   or path.endswith(('.md', '.rst', '.txt', '.tex', '.cff'))
                   or path in {'PHASE_STATUS.json', '.gitignore'})
        if not path.startswith('tests/'):
            selected.update(g['gate'] for g in gates
                            if path in closures[g['gate']]
                            or any(fnmatch.fnmatchcase(path, p) for p in g['watch']))
        transport = path in TRANSPORT or path.startswith('.github/')
        shared = (path == 'benchmarks/check_record_environment.py'
                  or path.startswith('benchmarks/provenance/')
                  or path in {'pyproject.toml', 'setup.cfg', 'setup.py'})
        production = path.startswith(('clifford_qc/', 'molecular/', 'benchmarks/')) and not routine
        if transport or shared or production and not any(path in c for c in closures.values()):
            selected.update(g['gate'] for g in gates)
    return [g for g in gates if g['gate'] in selected]


def changed_paths(root, base, head):
    if any(not re.fullmatch(r"[0-9a-f]{40,64}", value or "") for value in (base, head)):
        raise ValueError("diff requires full commit identities")
    ancestor = subprocess.check_output(["git", "merge-base", base, head], cwd=root, text=True).strip()
    raw = subprocess.check_output(["git", "diff", "--no-renames", "--name-only", "-z", ancestor, head], cwd=root)
    return raw.decode("utf-8").rstrip("\0").split("\0") if raw else []


def plan(event, payload, *, gates=None, root=ROOT):
    gates = load_registry() if gates is None else gates
    if event == 'pull_request':
        try:
            paths = changed_paths(root, payload['pull_request']['base']['sha'],
                                  payload['pull_request']['head']['sha'])
            chosen = affected(gates, paths, root=root)
        except (KeyError, ValueError, OSError, subprocess.CalledProcessError):
            chosen = affected(gates, [], full=True)
    elif event == 'push':
        try:
            paths = changed_paths(root, payload['before'], payload['after'])
            chosen = affected(gates, paths, root=root)
        except (KeyError, ValueError, OSError, subprocess.CalledProcessError):
            chosen = affected(gates, [], full=True)
    elif event in {'schedule', 'workflow_dispatch'}:
        chosen = affected(gates, [], full=True)
    else:
        raise ValueError('unsupported CI event')
    manual = [g['gate'] for g in chosen if g.get('manual_only', False)]
    if event != 'workflow_dispatch':
        chosen = [g for g in chosen if not g.get('manual_only', False)]
    result = {'manual_required': manual if event in {'pull_request', 'push'} else []}
    for tier in ('structural', 'sampled'):
        rows = []
        for g in chosen:
            if g['tier'] != tier:
                continue
            job_minutes = min(g['timeout'] + 15, 350)
            rows.append({'gate': g['gate'], 'timeout': job_minutes,
                         'job_timeout_seconds': job_minutes * 60,
                         'execution_declaration': g.get('execution_declaration', ''),
                         'environment_record': g['environment_record']})
        result[tier] = {'include': rows}
        result['has_' + tier] = bool(rows)
    return result


def execute(gates, *, output, root=ROOT):
    tasks = tuple(Task(g["gate"] + (f"_{i}" if len(g["commands"]) > 1 else ""),
                       (sys.executable, *command), g["timeout"] * 60,
                       max_output_bytes=g.get("max_output_bytes", 16 * 2**20))
                  for g in gates for i, command in enumerate(g["commands"]))
    if 'CI_JOB_STARTED_EPOCH' in os.environ:
        deadline = float(os.environ['CI_JOB_STARTED_EPOCH']) + float(os.environ['CI_JOB_TIMEOUT_SECONDS']) - 180
    else:
        deadline = None
    # Bind current tracked bytes, including evidence and helper dependencies.
    # Receipts record maintenance inputs; archived evidence identities stay in
    # their independent checkers, not inferred from this source closure.
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    sources = tuple(p for p in raw.decode().rstrip("\0").split("\0") if p)
    return run(tasks, root=root, destination=output, sources=sources, deadline_epoch=deadline)


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
