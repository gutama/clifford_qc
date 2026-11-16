#!/usr/bin/env python3
"""Verify the committed interval-coverage record against its declaration.

Four things are checked, in this order:

1. the record embeds exactly the committed declaration, by content and digest;
2. the rows are complete: every declared (cell, arm, replica) once, with the
   seeds the declaration derives and the shots it prescribes;
3. every summary, label and headline count is recomputed from the rows;
4. a spot replay regenerates the first declared replicas of every cell and
   compares them with the committed rows.

The replay runs only where it can mean something: under the library versions
that drew the record's shots, and while the recorded implementation digests
still match the live sources. Otherwise the reason is printed (``EVOLVED`` for
a changed source) and the structural checks stand alone; ``--require-replay``
turns that into a failure. ``--verify-history`` additionally proves from Git
that the declaration was committed before the record existed.

    python benchmarks/check_interval_coverage.py --verify-history
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_interval_coverage as producer  # noqa: E402
from clifford_qc.reproducibility import (  # noqa: E402
    compare_json_records, sampling_stream_mismatch,
)

# Replayed floats pass through an eigensolve or an SLSQP search, so they agree
# across machines only to rounding amplified by conditioning. This is far below
# any interval width in the study and far above BLAS-kernel noise.
REPLAY_TOLERANCE = 1e-7
# Below this stratification value a row's floats amplify rounding too much to
# compare; its discrete fields are still exact.
ILL_CONDITIONED = 1e-6
SUMMARY_TOLERANCE = 1e-12


def structural_problems(record: dict, declaration_path: Path) -> list[str]:
    problems = []
    declaration = producer.load_declaration(declaration_path)
    if record.get("schema") != producer.RESULTS_SCHEMA:
        return ["unsupported interval-coverage record schema"]
    if record.get("declaration") != declaration:
        problems.append("record does not embed the committed declaration")
    if record.get("declaration_sha256") != producer.sha256(declaration_path):
        problems.append("declaration digest differs from the committed file")
    designs = {system["key"]: producer.build_system(system)[1]
               for system in declaration["systems"]}
    problems.extend(compare_json_records(
        designs, record.get("systems"), "systems",
        atol=SUMMARY_TOLERANCE, rtol=SUMMARY_TOLERANCE))

    expected = {}
    for cell in producer.cells(declaration):
        shared, _ = producer.build_system(
            next(s for s in declaration["systems"] if s["key"] == cell["system"]))
        for index, arm in enumerate(declaration["arms"]):
            if cell["system"] not in arm["systems"]:
                continue
            for replica in range(arm["replicas"]):
                expected[(cell["cell"], arm["key"], replica)] = (
                    producer.measurement_seed(declaration, cell["cell"], replica),
                    producer.bootstrap_seed(declaration, cell["cell"], replica, index),
                    cell["shots_per_group"] * len(shared.groups))
    seen = set()
    for row in record.get("rows", []):
        key = (row.get("cell"), row.get("arm"), row.get("replica"))
        if key in seen:
            problems.append(f"duplicate row {key}")
        elif key not in expected:
            problems.append(f"undeclared row {key}")
        else:
            measurement, bootstrap, shots = expected[key]
            if (row.get("measurement_seed"), row.get("bootstrap_seed")) != (measurement,
                                                                            bootstrap):
                problems.append(f"row {key} does not carry its declared seeds")
            if row.get("shots") != shots:
                problems.append(f"row {key} spent {row.get('shots')} shots, declared {shots}")
        seen.add(key)
    missing = sorted(set(expected) - seen)
    if missing:
        problems.append(f"{len(missing)} declared row(s) are missing, first {missing[0]}")
    if problems:
        return problems

    recomputed = producer.summarize(declaration, record["rows"])
    for key in ("cells", "headline"):
        problems.extend(compare_json_records(
            recomputed[key], record.get(key), key,
            atol=SUMMARY_TOLERANCE, rtol=SUMMARY_TOLERANCE))
    return problems


def _discrete(value):
    """A row with every float removed: the part that must replay exactly."""
    if isinstance(value, dict):
        return {key: _discrete(item) for key, item in value.items()
                if not isinstance(item, float) and key != "covered"}
    if isinstance(value, list):
        return [_discrete(item) for item in value]
    return value


def replay_problems(record: dict, *, replicas: int | None) -> tuple[list[str], int]:
    """Regenerate rows and compare; ``replicas=None`` replays every row."""
    declaration = record["declaration"]
    committed = {(row["cell"], row["arm"], row["replica"]): row for row in record["rows"]}
    problems: list[str] = []
    compared = 0
    for task in producer.expected_tasks(declaration, replicas=replicas):
        for fresh in producer._work(task):
            key = (fresh["cell"], fresh["arm"], fresh["replica"])
            stored = committed[key]
            value = stored.get("stratification_value")
            if value is not None and value < ILL_CONDITIONED:
                found = compare_json_records(_discrete(stored), _discrete(fresh), f"row{key}")
            else:
                found = compare_json_records(stored, fresh, f"row{key}",
                                             atol=REPLAY_TOLERANCE, rtol=REPLAY_TOLERANCE)
            problems.extend(found)
            compared += 1
    return problems, compared


def replay_blockers(record: dict) -> list[str]:
    """Reasons a replay could not reproduce the record even if nothing is wrong."""
    blockers = list(sampling_stream_mismatch(record, packages=("numpy", "scipy")))
    live = producer.implementation_digests()
    for relative, digest in sorted(record.get("implementation", {}).items()):
        if live.get(relative) != digest:
            blockers.append(f"EVOLVED {relative}")
    return blockers


def history_problems(record: dict, declaration_path: Path, results_path: Path) -> list[str]:
    """The declaration must predate the record in the commit graph."""
    commit = record.get("declaration_commit", "")
    declaration = declaration_path.resolve().relative_to(ROOT).as_posix()
    results = results_path.resolve().relative_to(ROOT).as_posix()

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True)

    if git("cat-file", "-e", f"{commit}^{{commit}}").returncode:
        return [f"declaration commit {commit!r} is not in this repository's history"]
    problems = []
    if git("merge-base", "--is-ancestor", commit, "HEAD").returncode:
        problems.append("declaration commit is not an ancestor of HEAD")
    declared = git("show", f"{commit}:{declaration}")
    if declared.returncode or declared.stdout != declaration_path.read_bytes():
        problems.append("the declaration changed after its declaration commit")
    if git("cat-file", "-e", f"{commit}:{results}").returncode == 0:
        problems.append("the record already existed at the declaration commit")
    return problems


def _percent(proportion: dict) -> str:
    return "n/a" if proportion["rate"] is None else f"{100.0 * proportion['rate']:.1f}%"


def report(record: dict) -> str:
    """The record's results as Markdown, in the form the study document embeds.

    Generated rather than typed, so the document cannot drift from the record:
    a test holds ``INTERVAL_COVERAGE.md`` to this exact text.
    """
    declaration = record["declaration"]
    methods = declaration["methods"]
    levels = [producer.level_key(level) for level in declaration["levels"]]
    lines = ["| Arm | Method | Under | Consistent | Over | Solve failures | Interval failures |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in declaration["arms"]:
        for method in methods:
            tally = record["headline"].get(arm["key"], {}).get(method)
            if tally is not None:
                lines.append(
                    f"| `{arm['key']}` | `{method}` | {tally['under']} | {tally['consistent']} "
                    f"| {tally['over']} | {tally['solve_failures']} "
                    f"| {tally['interval_failures']} |")
    for level in levels:
        nominal = f"{100.0 * float(level):.0f}%"
        lines += ["", f"Coverage of the exact subspace energy at the nominal {nominal} level, "
                      "over all replicas:", "",
                  "| System | Shots per group | Arm | " + " | ".join(f"`{m}`" for m in methods)
                  + " |", "| --- | ---: | --- | " + " | ".join("---" for _ in methods) + " |"]
        for cell in record["cells"]:
            entries = []
            for method in methods:
                found = cell["summary"]["methods"][method]
                if not found["applicable"]:
                    entries.append("n/a")
                else:
                    entries.append(f"{_percent(found['levels'][level]['all_replicas'])} "
                                   f"{cell['labels'][method][level]}")
            lines.append(f"| `{cell['system']}` | {cell['shots_per_group']:,} | `{cell['arm']}` | "
                         + " | ".join(entries) + " |")
    lines += ["", "Point-estimate error against the exact subspace energy:", "",
              "| System | Shots per group | Arm | Replicas | Solve failures | Median abs. error "
              "| 90th pct. abs. error | Max abs. error | Mean error | Below target |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for cell in record["cells"]:
        summary = cell["summary"]
        estimate = summary["estimate"]
        if estimate is None:
            numbers = "n/a | n/a | n/a | n/a | n/a"
        else:
            numbers = (f"{estimate['median_abs_error']:.4f} | {estimate['p90_abs_error']:.4f} "
                       f"| {estimate['max_abs_error']:.4f} | {estimate['mean_error']:+.4f} "
                       f"| {100.0 * estimate['below_target'] / estimate['n']:.1f}%")
        lines.append(f"| `{cell['system']}` | {cell['shots_per_group']:,} | `{cell['arm']}` "
                     f"| {summary['replicas']} | {summary['solve_failures']} | {numbers} |")
    top = levels[-1]
    lines += ["", f"Coverage at the nominal {100.0 * float(top):.0f}% level by stratification "
                  "value, pooled over cells (replicas whose interval exists):", "",
              "| Arm | Stratum | " + " | ".join(f"`{m}`" for m in methods) + " |",
              "| --- | --- | " + " | ".join("---" for _ in methods) + " |"]
    for arm in declaration["arms"]:
        cells = [cell for cell in record["cells"] if cell["arm"] == arm["key"]]
        for index in range(len(declaration["strata_edges"]) + 1):
            strata = [cell["summary"]["strata"][index] for cell in cells]
            if not any(stratum["n"] for stratum in strata):
                continue
            lower, upper = strata[0]["lower"], strata[0]["upper"]
            name = (f"below {upper:g}" if lower is None else
                    f"at least {lower:g}" if upper is None else f"{lower:g} to {upper:g}")
            entries = []
            for method in methods:
                found = [stratum["methods"][method]["levels"][top]["successful_intervals"]
                         for stratum in strata
                         if stratum["n"] and stratum["methods"][method]["applicable"]]
                trials = sum(item["n"] for item in found)
                covered = sum(item["covered"] for item in found)
                entries.append("n/a" if not trials
                               else f"{100.0 * covered / trials:.1f}% of {trials}")
            lines.append(f"| `{arm['key']}` | {name} | " + " | ".join(entries) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--declaration", type=Path, default=producer.DECLARATION)
    parser.add_argument("--results", type=Path, default=producer.RESULTS)
    parser.add_argument("--verify-history", action="store_true")
    parser.add_argument("--full-replay", action="store_true")
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--require-replay", action="store_true")
    parser.add_argument("--report", action="store_true",
                        help="print the results tables as Markdown and exit")
    args = parser.parse_args(argv)
    if args.no_replay and (args.full_replay or args.require_replay):
        parser.error("--no-replay excludes --full-replay and --require-replay")

    record = producer.read_record(args.results)
    if args.report:
        print(report(record), end="")
        return 0
    problems = structural_problems(record, args.declaration)
    if args.verify_history and not problems:
        problems.extend(history_problems(record, args.declaration, args.results))
    replayed = "not requested"
    if not problems and not args.no_replay:
        blockers = replay_blockers(record)
        if blockers:
            for blocker in blockers:
                print(f"NOTE replay unavailable: {blocker}")
            replayed = "skipped"
            if args.require_replay:
                problems.append("replay was required but cannot reproduce this record here")
        else:
            count = None if args.full_replay else record["declaration"]["spot_replay_replicas"]
            found, compared = replay_problems(record, replicas=count)
            problems.extend(found)
            replayed = f"{compared} row(s) regenerated"
    for problem in problems[:20]:
        print(f"FAIL {problem}")
    if len(problems) > 20:
        print(f"FAIL ... {len(problems) - 20} more")
    if problems:
        print(f"{len(problems)} interval-coverage problem(s)")
        return 1
    print(f"OK ({len(record['rows'])} rows match the declaration; replay: {replayed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
