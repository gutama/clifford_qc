#!/usr/bin/env python3
"""Verify the committed target-acquisition record against its declaration.

Five things are checked, in this order:

1. the record embeds exactly the committed declaration, by content and digest;
2. the rows are complete: every declared (system, allocation, replica) once,
   with the seed the declaration derives;
3. every run obeys the schedule it was declared under: the shots at its last
   look, a stopped run that is the same seeded run ended at the first look at
   or below the target, and a stopped interval widened for every look;
4. every summary, comparison, label and headline count is recomputed from the
   rows;
5. a spot replay regenerates the first declared replicas of every system and
   compares them with the committed rows.

The replay runs only where it can mean something: under the library versions
that drew the record's shots, and while the recorded implementation digests
still match the live sources. Otherwise the reason is printed (``EVOLVED`` for
a changed source) and the structural checks stand alone; ``--require-replay``
turns that into a failure. ``--verify-history`` additionally proves from Git
that the declaration was committed before the record existed.

    python benchmarks/check_target_acquisition.py --verify-history
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_target_acquisition as producer  # noqa: E402
from benchmarks.run_interval_coverage import build_system, read_record, sha256  # noqa: E402
from clifford_qc.reproducibility import (  # noqa: E402
    compare_json_records, sampling_stream_mismatch,
)

# Replayed floats pass through an eigensolve, so they agree across machines
# only to rounding amplified by conditioning. Across BLAS kernels the
# development banks agreed to 2e-11; this is far below any interval width in
# the study.
REPLAY_TOLERANCE = 1e-7
# A look whose smallest retained overlap eigenvalue is below this amplifies
# rounding too much to compare its floats. In an adaptive schedule it also
# decides the next round's shots, so nothing after it is comparable either.
ILL_CONDITIONED = 1e-6
# Structural checks also run on supported stacks other than the record's.
# NumPy 2.2.6 and 2.5.2 recompute a paired-bootstrap endpoint about 3e-12
# apart from the same stored rows. Allow that summation/quantile roundoff;
# integer counts and labels are still compared exactly.
SUMMARY_TOLERANCE = 1e-10
# A stopped run and the full schedule are the same seeded computation up to
# the stopping look, so they agree to rounding, not merely to tolerance.
PREFIX_TOLERANCE = 1e-12


LEDGER = ("energies", "std_errors", "retained_ranks", "smallest_retained_overlaps")


def _close(left: float, right: float, tolerance: float = PREFIX_TOLERANCE) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _run_problems(run: dict, name: str, *, cumulative: list[int], groups: int,
                  target: float, goal: float | None, delta: float) -> list[str]:
    """One run against the schedule and the stopping rule it was declared under."""
    looks = len(cumulative)
    taken = run.get("looks_taken")
    if run.get("outcome") == "solve_failed":
        if not isinstance(taken, int) or not 0 <= taken < looks:
            return [f"{name} failed at an undeclared look"]
        problems = []
        if run.get("shots") != cumulative[taken] * groups:
            problems.append(f"{name} failed after shots the schedule does not contain")
        if not run.get("solve_error") or run.get("estimate") is not None:
            problems.append(f"{name} is a failure with an estimate or without a reason")
        return problems
    if run.get("outcome") != "ok":
        return [f"{name} has an unknown outcome"]
    if not isinstance(taken, int) or not 1 <= taken <= looks:
        return [f"{name} took an undeclared number of looks"]
    problems = []
    if run["shots"] != cumulative[taken - 1] * groups:
        problems.append(f"{name} spent {run['shots']} shots, the schedule has "
                        f"{cumulative[taken - 1] * groups} at look {taken - 1}")
    if run["stopped"] not in producer.STOP_REASONS:
        problems.append(f"{name} ended for an undeclared reason")
    elif run["stopped"] == "schedule_exhausted" and taken != looks:
        problems.append(f"{name} claims an exhausted schedule before its last look")
    elif run["stopped"] == "target_met" and (goal is None or run["std_error"] > goal):
        problems.append(f"{name} claims a target its standard error does not meet")
    elif run["stopped"] == "schedule_exhausted" and goal is not None \
            and run["std_error"] <= goal:
        problems.append(f"{name} met its target but is recorded as exhausted")
    for kind in ("interval", "naive_interval"):
        bounds = run[kind]
        if not bounds["lower"] <= run["estimate"] <= bounds["upper"]:
            problems.append(f"{name} {kind} does not contain its estimate")
        if bounds["covered"] != (bounds["lower"] <= target <= bounds["upper"]):
            problems.append(f"{name} {kind} coverage flag contradicts its bounds")
    # The stopped interval splits delta over every look; the naive one does not.
    family = looks if goal is not None else 1
    normal = NormalDist()
    widening = (normal.inv_cdf(1.0 - delta / (2.0 * family))
                / normal.inv_cdf(1.0 - delta / 2.0))
    naive = run["naive_interval"]["upper"] - run["naive_interval"]["lower"]
    valid = run["interval"]["upper"] - run["interval"]["lower"]
    if not math.isclose(valid, widening * naive, rel_tol=1e-9, abs_tol=1e-12):
        problems.append(f"{name} interval is not widened for {family} look(s)")
    return problems


def _row_problems(row: dict, declaration: dict, design: dict) -> list[str]:
    name = f"row ({row['system']}, {row['allocation']}, {row['replica']})"
    cumulative = declaration["cumulative_shots_per_group"]
    common = {"cumulative": cumulative, "groups": design["qwc_groups"],
              "target": design["target"], "delta": declaration["delta"]}
    fixed = row["fixed"]
    problems = _run_problems(fixed, f"{name} fixed", goal=None, **common)
    ledger = fixed.get("ledger", {})
    lengths = {len(ledger.get(key, [])) for key in LEDGER}
    if lengths != {fixed.get("looks_taken")}:
        return problems + [f"{name} fixed ledger does not have one entry per look"]
    if fixed["outcome"] == "ok":
        if fixed["stopped"] == "target_met":
            problems.append(f"{name} fixed run stopped on a target it was not given")
        if (fixed["estimate"], fixed["std_error"]) != (ledger["energies"][-1],
                                                       ledger["std_errors"][-1]):
            problems.append(f"{name} fixed estimate is not its ledger's last look")
    if set(row["stopping"]) != {producer.target_key(goal)
                                for goal in declaration["target_std_errors"]}:
        return problems + [f"{name} does not have one stopped run per declared target"]
    for goal in declaration["target_std_errors"]:
        run = row["stopping"][producer.target_key(goal)]
        label = f"{name} stopping at {goal}"
        problems.extend(_run_problems(run, label, goal=goal, **common))
        taken = run.get("looks_taken")
        if run.get("outcome") != "ok":
            # The stopped run walks the same looks, so it can only fail where
            # the full schedule failed.
            if fixed["outcome"] == "ok" or taken != fixed["looks_taken"]:
                problems.append(f"{label} failed where the full schedule did not")
            continue
        if not isinstance(taken, int) or not 1 <= taken <= len(ledger["energies"]):
            problems.append(f"{label} outran the full schedule's ledger")
            continue
        # The stopped run is the full schedule's seeded run, ended early.
        if not (_close(run["estimate"], ledger["energies"][taken - 1])
                and _close(run["std_error"], ledger["std_errors"][taken - 1])):
            problems.append(f"{label} is not a prefix of the full schedule")
        if any(sigma <= goal for sigma in ledger["std_errors"][:taken - 1]):
            problems.append(f"{label} ran past the first look that met its target")
    return problems


def structural_problems(record: dict, declaration_path: Path) -> list[str]:
    problems = []
    declaration = producer.load_declaration(declaration_path)
    if record.get("schema") != producer.RESULTS_SCHEMA:
        return ["unsupported target-acquisition record schema"]
    if record.get("declaration") != declaration:
        problems.append("record does not embed the committed declaration")
    if record.get("declaration_sha256") != sha256(declaration_path):
        problems.append("declaration digest differs from the committed file")
    designs = {system["key"]: build_system(system)[1] for system in declaration["systems"]}
    problems.extend(compare_json_records(
        designs, record.get("systems"), "systems",
        atol=SUMMARY_TOLERANCE, rtol=SUMMARY_TOLERANCE))

    expected = {(system["key"], allocation, replica):
                producer.measurement_seed(declaration, index, replica)
                for index, system in enumerate(declaration["systems"])
                for allocation in declaration["allocations"]
                for replica in range(declaration["replicas"])}
    seen = set()
    for row in record.get("rows", []):
        key = (row.get("system"), row.get("allocation"), row.get("replica"))
        if key in seen:
            problems.append(f"duplicate row {key}")
        elif key not in expected:
            problems.append(f"undeclared row {key}")
        elif row.get("measurement_seed") != expected[key]:
            problems.append(f"row {key} does not carry its declared seed")
        seen.add(key)
    missing = sorted(set(expected) - seen)
    if missing:
        problems.append(f"{len(missing)} declared row(s) are missing, first {missing[0]}")
    if problems:
        return problems

    for row in record["rows"]:
        problems.extend(_row_problems(row, declaration, designs[row["system"]]))
    if problems:
        return problems
    recomputed = producer.summarize(declaration, record["rows"])
    for key in ("summaries", "headline"):
        problems.extend(compare_json_records(
            recomputed[key], record.get(key), key,
            atol=SUMMARY_TOLERANCE, rtol=SUMMARY_TOLERANCE))
    return problems


def comparable_looks(row: dict) -> int:
    """How many leading looks of a committed row replay to rounding on any machine."""
    values = row["fixed"]["ledger"]["smallest_retained_overlaps"]
    return next((index for index, value in enumerate(values)
                 if value is not None and value < ILL_CONDITIONED), len(values))


def _comparable(row: dict, looks: int) -> dict:
    """The part of a row decided before its first ill-conditioned look.

    The ill-conditioned look itself keeps its retained rank, which is discrete;
    a stopped run is kept only if it ended before that look.
    """
    ledger = row["fixed"]["ledger"]
    return {
        **{key: row[key] for key in ("system", "allocation", "replica", "measurement_seed")},
        "ledger": {key: ledger[key][:looks + (key == "retained_ranks")] for key in LEDGER},
        "stopping": {key: run for key, run in row["stopping"].items()
                     if run["looks_taken"] <= looks}}


def replay_problems(record: dict, *, replicas: int | None) -> tuple[list[str], int]:
    """Regenerate rows and compare; ``replicas=None`` replays every row."""
    declaration = record["declaration"]
    committed = {(row["system"], row["allocation"], row["replica"]): row
                 for row in record["rows"]}
    problems: list[str] = []
    compared = 0
    for task in producer.expected_tasks(declaration, replicas=replicas):
        for fresh in producer._work(task):
            key = (fresh["system"], fresh["allocation"], fresh["replica"])
            stored = committed[key]
            looks = comparable_looks(stored)
            if looks < len(stored["fixed"]["ledger"]["energies"]):
                stored, fresh = _comparable(stored, looks), _comparable(fresh, looks)
            problems.extend(compare_json_records(
                stored, fresh, f"row{key}", atol=REPLAY_TOLERANCE, rtol=REPLAY_TOLERANCE))
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


# --------------------------------------------------------------------- report

def _percent(proportion: dict) -> str:
    return f"{100.0 * proportion['rate']:.1f}% {proportion['label']}"


def _ratio(found: dict) -> str:
    if found["ratio"] is None:
        return "n/a"
    if found["interval"] is None:
        return f"{found['ratio']:.2f}"
    return f"{found['ratio']:.2f} [{found['interval'][0]:.2f}, {found['interval'][1]:.2f}]"


def _number(value: float | None, style: str = ".4f") -> str:
    return "n/a" if value is None else format(value, style)


def report(record: dict) -> str:
    """The record's results as Markdown, in the form the study document embeds.

    Generated rather than typed, so the document cannot drift from the record:
    a test holds ``TARGET_ACQUISITION.md`` to this exact text.
    """
    declaration = record["declaration"]
    uniform, targeted = declaration["allocations"]
    keys = [producer.target_key(goal) for goal in declaration["target_std_errors"]]
    nominal = f"{100.0 * (1.0 - declaration['delta']):.0f}%"
    headline = record["headline"]

    lines = ["Cost of reaching a declared standard error. Ratios are uniform over "
             "targeted, so above one means targeting is cheaper:", "",
             "| System | Role | Target | Mean shots per setting (U / T) | Shot ratio "
             "| RMSE at stop (U / T) | Shots at matched error | Exhausted (U / T) | Verdict |",
             "| --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- |"]
    for system in record["summaries"]:
        for key in keys:
            found = system["comparison"]["stopping"][key]
            runs = {name: system["allocations"][name]["stopping"][key]
                    for name in (uniform, targeted)}
            errors = " / ".join(
                _number(None if runs[name]["errors"] is None else runs[name]["errors"]["rmse"])
                for name in (uniform, targeted))
            lines.append(
                f"| `{system['system']}` | {system['role'].replace('_', ' ')} | {key} "
                f"| {runs[uniform]['mean_shots_per_group']:,.0f} "
                f"/ {runs[targeted]['mean_shots_per_group']:,.0f} | {_ratio(found['shots'])} "
                f"| {errors} | {_ratio(found['matched_error'])} "
                f"| {found['exhausted'][uniform]} / {found['exhausted'][targeted]} "
                f"| `{found['verdict']}` |")

    lines += ["", "Verdict counts over systems:", "",
              "| Target | " + " | ".join(f"`{verdict}`" for verdict in producer.VERDICTS) + " |",
              "| ---: | " + " | ".join("---:" for _ in producer.VERDICTS) + " |"]
    for key in keys:
        lines.append(f"| {key} | " + " | ".join(
            str(headline["stopping"][key][verdict]) for verdict in producer.VERDICTS) + " |")

    lines += ["", "Mean-squared error at the last look of the full schedule, "
                  "uniform over targeted at equal shots:", "",
              "| System | Shots | Uniform RMSE | Targeted RMSE | MSE ratio | Label |",
              "| --- | ---: | ---: | ---: | --- | --- |"]
    for system in record["summaries"]:
        found = system["comparison"]["matched_budget"]
        last = {name: system["allocations"][name]["fixed"]["looks"][-1]["errors"]
                for name in (uniform, targeted)}
        lines.append(
            f"| `{system['system']}` | {found['shots']:,} "
            f"| {_number(None if last[uniform] is None else last[uniform]['rmse'], '.5f')} "
            f"| {_number(None if last[targeted] is None else last[targeted]['rmse'], '.5f')} "
            f"| {_ratio(found)} | `{found['label']}` |")

    lines += ["", f"Coverage of the exact subspace energy at the nominal {nominal} level, "
                  "over all replicas:", "",
              "| System | Allocation | Full schedule | "
              + " | ".join(f"Stopped at {key}, schedule-valid | Stopped at {key}, naive"
                           for key in keys) + " |",
              "| --- | --- | --- | " + " | ".join("--- | ---" for _ in keys) + " |"]
    for system in record["summaries"]:
        for name in (uniform, targeted):
            found = system["allocations"][name]
            entries = [_percent(found["fixed"]["final_coverage"])]
            for key in keys:
                coverage = found["stopping"][key]["coverage"]
                entries += [_percent(coverage["schedule_valid"]), _percent(coverage["naive"])]
            lines.append(f"| `{system['system']}` | `{name}` | " + " | ".join(entries) + " |")

    lines += ["", "Coverage label counts:", "",
              "| Interval | Under | Consistent | Over |", "| --- | ---: | ---: | ---: |"]
    for kind, name in (("fixed_final", "Full schedule, single look"),
                       ("schedule_valid", "Stopped, schedule-valid"),
                       ("naive", "Stopped, naive")):
        tally = headline["coverage"][kind]
        lines.append(f"| {name} | {tally['under']} | {tally['consistent']} | {tally['over']} |")
    lines += ["", f"Solve failures over all runs: {headline['solve_failures']}."]

    for system in record["summaries"]:
        lines += ["", f"Error at every look of the full schedule on `{system['system']}`:", "",
                  "| Shots per setting | Uniform RMSE | Uniform median abs. error "
                  "| Uniform worst abs. error | Targeted RMSE | Targeted median abs. error "
                  "| Targeted worst abs. error | Mean retained rank (U / T) |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        looks = {name: system["allocations"][name]["fixed"]["looks"]
                 for name in (uniform, targeted)}
        for index, look in enumerate(looks[uniform]):
            cells = []
            for name in (uniform, targeted):
                errors = looks[name][index]["errors"]
                cells += ["n/a"] * 3 if errors is None else [
                    f"{errors['rmse']:.4f}", f"{errors['median_abs_error']:.4f}",
                    f"{errors['max_abs_error']:.4f}"]
            ranks = " / ".join(_number(looks[name][index]["mean_retained_rank"], ".2f")
                               for name in (uniform, targeted))
            lines.append(f"| {look['shots_per_group']:,} | " + " | ".join(cells)
                         + f" | {ranks} |")
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

    record = read_record(args.results)
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
        print(f"{len(problems)} target-acquisition problem(s)")
        return 1
    print(f"OK ({len(record['rows'])} rows match the declaration; replay: {replayed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
