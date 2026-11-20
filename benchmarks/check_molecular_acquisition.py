#!/usr/bin/env python3
"""Verify the committed molecular-acquisition record against its declaration.

Checked, in this order:

1. the record embeds exactly the committed declaration, by content and digest,
   and its exact system facts are the ones the declaration's banks have;
2. the rows are complete: every declared system and replica once, with the
   seed the declaration derives;
3. every row obeys its schedule: the shots at every look, and for every arm and
   target a stop at the first look whose reported standard error meets the
   target, or the last look if none did, with the declared intervals there and
   a bootstrap exactly where the plan asks for one;
4. every summary, comparison, verdict and criterion is recomputed from the
   rows;
5. with ``--replay N``, the first ``N`` replicas of every system are rerun with
   ``acquire_ritz`` itself, one run per arm and target, and must reproduce each
   stop, estimate, standard error, interval and bootstrap spread.

The replay is meaningful only under the NumPy and SciPy that produced the
record; under others it is skipped with a note, or fails with
``--require-replay``. Building an eight-qubit bank's outcome distributions
takes several minutes, so the replay is not part of the CI gate.
``--verify-history`` proves from Git that the declaration was committed before
the record existed, and ``--report`` prints the Markdown the study document
embeds.

    python benchmarks/check_molecular_acquisition.py --verify-history
    python benchmarks/check_molecular_acquisition.py --replay 1
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import run_molecular_acquisition as producer  # noqa: E402
from benchmarks.run_interval_coverage import read_record  # noqa: E402
from clifford_qc.backends.finite_shot import FiniteShotBackend  # noqa: E402
from clifford_qc.reproducibility import (  # noqa: E402
    compare_json_records, sampling_stream_mismatch,
)
from clifford_qc.subspace.acquisition import AcquisitionError, acquire_ritz  # noqa: E402

DOCUMENT = ROOT / "docs" / "restart" / "MOLECULAR_ACQUISITION.md"
# A replayed float passes through an eigensolve and, for the corrected arms, a
# least-squares fit; across BLAS kernels it agrees to rounding amplified by
# conditioning, far below any interval width here.
REPLAY_TOLERANCE = 1e-7
# Recomputed summaries pass through means and quantiles of the same rows.
SUMMARY_TOLERANCE = 1e-10
# Exact facts are recomputed here, possibly under another LAPACK.
DESIGN_TOLERANCE = 1e-9


def _close(left, right, tolerance: float = REPLAY_TOLERANCE) -> bool:
    return (isinstance(left, (int, float)) and isinstance(right, (int, float))
            and math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance))


# ------------------------------------------------------------------- structure

def declaration_problems(record: dict, declaration_path: Path) -> list[str]:
    problems = []
    if record.get("schema") != producer.RESULTS_SCHEMA:
        problems.append(f"schema is {record.get('schema')!r}")
    declaration = producer.load_declaration(declaration_path)
    if record.get("declaration") != declaration:
        problems.append("the record does not embed the committed declaration")
    if record.get("declaration_sha256") != producer.sha256(declaration_path):
        problems.append("the record names a different declaration digest")
    return problems


def design_problems(record: dict, declaration: dict, systems: dict) -> list[str]:
    problems = []
    for spec in declaration["systems"]:
        stored = record.get("systems", {}).get(spec["key"])
        fresh = systems[spec["key"]]["design"]
        if not isinstance(stored, dict) or set(stored) != set(fresh) or not all(
                _close(stored[name], fresh[name], DESIGN_TOLERANCE) for name in fresh):
            problems.append(f"{spec['key']}: recorded exact facts are not the bank's")
    return problems


def row_problems(record: dict, declaration: dict) -> list[str]:
    """Completeness, seeds and every arm's stop against its own ledger."""
    problems: list[str] = []
    expected = {(spec["key"], replica): producer.measurement_seed(declaration, index, replica)
                for index, spec in enumerate(declaration["systems"])
                for replica in range(spec["replicas"])}
    seen = set()
    designs = record.get("systems", {})
    for row in record.get("rows", []):
        key = (row.get("system"), row.get("replica"))
        name = "{}/{}".format(*key)
        if key not in expected or key in seen:
            problems.append(f"row {name} is undeclared or repeated")
            continue
        seen.add(key)
        if row.get("measurement_seed") != expected[key]:
            problems.append(f"row {name} has the wrong seed")
        design = designs.get(key[0])
        if not isinstance(design, dict):
            problems.append(f"row {name} has no recorded design")
            continue
        problems.extend(_row_schedule_problems(row, declaration, design, name))
    problems.extend(f"row {key} is missing" for key in sorted(set(expected) - seen))
    return problems


def _row_schedule_problems(row: dict, declaration: dict, design: dict, name: str) -> list[str]:
    problems = []
    groups = design["settings"]
    cumulative = [shots * groups for shots in declaration["cumulative_shots_per_group"]]
    z = {count: producer.simultaneous_z_radius(1.0, declaration["delta"], count)
         for count in (1, producer.looks(declaration))}
    target = design["target"]
    for run in declaration["runs"]:
        found = row.get("runs", {}).get(run["key"])
        if not isinstance(found, dict):
            problems.append(f"{name} has no run {run['key']}")
            continue
        ledger, failure = found["ledger"], found["failure"]
        if [entry["shots"] for entry in ledger] != cumulative[:len(ledger)]:
            problems.append(f"{name} {run['key']}: shots are not the schedule's")
        if failure is None and len(ledger) != len(cumulative):
            problems.append(f"{name} {run['key']}: ended early without a failure")
        if failure is not None and (failure["look"] != len(ledger)
                                    or failure["shots"] != cumulative[failure["look"]]):
            problems.append(f"{name} {run['key']}: the failure is not at the next look")
        for arm in (arm for arm in declaration["arms"] if arm["run"] == run["key"]):
            sigma_key = "std_error" if arm["corrected"] else "raw_std_error"
            energy_key = "energy" if arm["corrected"] else "raw_energy"
            sigmas = [entry[sigma_key] for entry in ledger]
            for goal_index, goal in enumerate(declaration["target_std_errors"]):
                stop = row["arms"][arm["key"]]["stopping"][producer.target_key(goal)]
                label = f"{name} {arm['key']} at {goal}"
                first = next((index for index, sigma in enumerate(sigmas) if sigma <= goal),
                             None)
                if first is None and failure is not None:
                    if stop.get("outcome") != "solve_failed" or stop.get("look") != failure["look"]:
                        problems.append(f"{label}: should have failed with its run")
                    continue
                look = first if first is not None else len(ledger) - 1
                reason = "target_met" if first is not None else "schedule_exhausted"
                entry = ledger[look]
                if (stop.get("outcome"), stop.get("stopped"), stop.get("look"),
                        stop.get("shots")) != ("ok", reason, look, entry["shots"]) \
                        or stop.get("estimate") != entry[energy_key] \
                        or stop.get("std_error") != entry[sigma_key]:
                    problems.append(f"{label}: the stop is not the ledger's first met look")
                    continue
                for kind, count in (("interval", producer.looks(declaration)),
                                    ("naive_interval", 1)):
                    bounds = stop[kind]
                    centre = (bounds["lower"] + bounds["upper"]) / 2
                    if not _close(centre, stop["estimate"], 1e-12) or bounds["covered"] != (
                            bounds["lower"] <= target <= bounds["upper"]):
                        problems.append(f"{label}: its {kind} is not centred or labelled")
                    if arm["corrected"] and not _close(bounds["upper"] - stop["estimate"],
                                                       z[count] * stop["std_error"], 1e-9):
                        problems.append(f"{label}: its {kind} is not z standard errors wide")
                    if not arm["corrected"] and (bounds["upper"] - stop["estimate"]
                                                 < z[count] * stop["std_error"] * (1 - 1e-9)):
                        problems.append(f"{label}: its {kind} is narrower than z sigma")
                problems.extend(_bootstrap_problems(stop, declaration, row, arm, goal_index,
                                                    target, z, label))
    return problems


def _bootstrap_problems(stop, declaration, row, arm, goal_index, target, z, label) -> list[str]:
    draws = producer.bootstrap_plan(declaration, row["system"], arm["key"], row["replica"])
    entry = stop.get("bootstrap")
    if draws is None:
        return [] if entry is None else [f"{label}: a bootstrap the plan does not ask for"]
    index = [spec["key"] for spec in declaration["systems"]].index(row["system"])
    arm_index = [a["key"] for a in declaration["arms"]].index(arm["key"])
    seed = producer.interval_seed(declaration, index, row["replica"], arm_index, goal_index)
    if not isinstance(entry, dict) or entry.get("draws") != draws or entry.get("seed") != seed:
        return [f"{label}: the bootstrap is not the planned draws and seed"]
    if "failure" in entry:
        return []
    radius = z[producer.looks(declaration)] * entry["spread"]
    bounds = entry["interval"]
    if not (_close(bounds["lower"], stop["estimate"] - radius, 1e-9)
            and _close(bounds["upper"], stop["estimate"] + radius, 1e-9)
            and bounds["covered"] == (bounds["lower"] <= target <= bounds["upper"])):
        return [f"{label}: the bootstrap interval is not E +- z r"]
    return []


def summary_problems(record: dict, declaration: dict) -> list[str]:
    fresh = producer.summarize(declaration, record["rows"], record["systems"])
    problems = []
    for name in ("summaries", "criteria"):
        problems.extend(compare_json_records(record.get(name), fresh[name], name,
                                             atol=SUMMARY_TOLERANCE, rtol=SUMMARY_TOLERANCE))
    return problems


# ---------------------------------------------------------------------- replay

def replay_row(declaration: dict, system: dict, system_index: int, stored: dict) -> list[str]:
    """Rerun every arm and target of one stored row with ``acquire_ritz``."""
    spec = declaration["systems"][system_index]
    shared, family = system["shared"], system["family"]
    pilot, budgets = producer.schedule(declaration, len(shared.groups))
    runs = {run["key"]: run for run in declaration["runs"]}
    name = f"{spec['key']}/{stored['replica']}"
    problems = []
    for arm_index, arm in enumerate(declaration["arms"]):
        run = runs[arm["run"]]
        for goal_index, goal in enumerate(declaration["target_std_errors"]):
            stop = stored["arms"][arm["key"]]["stopping"][producer.target_key(goal)]
            draws = producer.bootstrap_plan(declaration, spec["key"], arm["key"],
                                            stored["replica"])
            options = {}
            if draws is not None:
                options = {"bootstrap_replicates": draws, "bootstrap_seed": producer.interval_seed(
                    declaration, system_index, stored["replica"], arm_index, goal_index)}
            label = f"{name} {arm['key']} at {goal}"
            # A prepared backend reseeded draws the stream a fresh one would,
            # without recompiling every setting's outcome distribution.
            backend = system.get("backend")
            if backend is None:
                backend = FiniteShotBackend(seed=stored["measurement_seed"])
            else:
                backend.reseed(stored["measurement_seed"])
            try:
                found = acquire_ritz(
                    shared, backend,
                    pilot_shots_per_group=pilot, round_budgets=budgets,
                    allocation=run["allocation"], target_std_error=goal,
                    solver_kwargs=dict(declaration["solver_kwargs"]), delta=declaration["delta"],
                    control_variates=family if arm["corrected"] else None, **options)
            except AcquisitionError as error:
                if stop["outcome"] != "solve_failed" or len(error.looks) != stop["look"]:
                    problems.append(f"{label}: acquire_ritz failed where the row did not")
                continue
            checks = [stop["outcome"] == "ok", found.stopped == stop.get("stopped"),
                      len(found.looks) - 1 == stop.get("look"), found.shots == stop.get("shots"),
                      _close(found.energy, stop.get("estimate")),
                      _close(found.std_error, stop.get("std_error")),
                      _close(found.interval.lower, stop["interval"]["lower"]),
                      _close(found.interval.upper, stop["interval"]["upper"]),
                      _close(found.naive_interval.lower, stop["naive_interval"]["lower"])]
            if draws is not None:
                entry = stop["bootstrap"]
                if "failure" in entry:
                    checks.append(found.bootstrap_interval is None)
                else:
                    checks += [_close(found.looks[-1].get("bootstrap_spread"), entry["spread"]),
                               _close(found.bootstrap_interval.lower, entry["interval"]["lower"])]
            if not all(checks):
                problems.append(f"{label}: acquire_ritz does not reproduce the row")
    return problems


def replay_blockers(record: dict) -> list[str]:
    return sampling_stream_mismatch(record, packages=("numpy", "scipy"))


def replay_problems(record: dict, declaration: dict, systems: dict, replicas: int) -> list[str]:
    stored = {(row["system"], row["replica"]): row for row in record["rows"]}
    problems = []
    for index, spec in enumerate(declaration["systems"]):
        for replica in range(min(replicas, spec["replicas"])):
            problems.extend(replay_row(declaration, systems[spec["key"]], index,
                                       stored[(spec["key"], replica)]))
    return problems


# --------------------------------------------------------------------- history

def history_problems(record: dict, declaration_path: Path, results_path: Path) -> list[str]:
    """The declaration must predate the record in the commit graph."""
    commit = record.get("declaration_commit", "")
    declaration = declaration_path.resolve().relative_to(ROOT).as_posix()
    results = results_path.resolve().relative_to(ROOT).as_posix()

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True)

    if git("rev-parse", "--is-shallow-repository").stdout.strip() != b"false":
        return ["full Git history is required to verify the declaration order"]
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


# ---------------------------------------------------------------------- report

def _rate(proportion: dict | None) -> str:
    if proportion is None:
        return "–"
    low, high = proportion["wilson"]
    return f"{proportion['rate']:.3f} [{low:.3f}, {high:.3f}] {proportion['label']}"


def _ratio(found: dict) -> str:
    if found["ratio"] is None:
        return "n/a"
    if found["interval"] is None:
        return f"{found['ratio']:.2f}x"
    low, high = found["interval"]
    return f"{found['ratio']:.2f}x [{low:.2f}, {high:.2f}]"


def report(record: dict) -> str:
    """The record's results as Markdown, in the form the study document embeds."""
    declaration = record["declaration"]
    lines = []
    for entry in record["summaries"]:
        system = entry["system"]
        design = record["systems"][system]
        lines += [f"### `{system}` ({entry['role'].replace('_', ' ')})", "",
                  "| Target | Arm | Exhausted | Mean shots per setting | RMSE at stop "
                  "| Delta interval | Bootstrap interval | r / sigma |",
                  "| ---: | --- | ---: | ---: | ---: | --- | --- | ---: |"]
        for goal in declaration["target_std_errors"]:
            key = producer.target_key(goal)
            for arm in declaration["arms"]:
                cell = entry["arms"][arm["key"]]["stopping"][key]
                boot = cell["bootstrap"]
                ratio = (f"{boot['median_spread_over_std_error']:.2f}"
                         if boot and boot["median_spread_over_std_error"] is not None else "–")
                rmse = f"{cell['errors']['rmse']:.5f}" if cell["errors"] else "n/a"
                lines.append(
                    f"| {goal:g} | {arm['key']} | {cell['stopped']['schedule_exhausted']} | "
                    f"{cell['mean_shots_per_setting']:,.0f} | {rmse} | "
                    f"{_rate(cell['coverage']['schedule_valid'])} | "
                    f"{_rate(boot['coverage'] if boot else None)} | {ratio} |")
        lines += ["", "| Comparison | Equal shots (MSE ratio) | "
                  + " | ".join(f"Matched error at {goal:g}"
                               for goal in declaration["target_std_errors"]) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in declaration["target_std_errors"])
                  + " |"]
        for reference, arm in declaration["comparison"]["pairs"]:
            found = entry["comparisons"][f"{reference}/{arm}"]
            cells = [f"{_ratio(found['stopping'][producer.target_key(goal)]['matched_error'])} "
                     f"{found['stopping'][producer.target_key(goal)]['verdict']}"
                     for goal in declaration["target_std_errors"]]
            lines.append(f"| {found['reference']} over {found['arm']} | "
                         f"{_ratio(found['equal_shots'])} {found['equal_shots']['label']} | "
                         + " | ".join(cells) + " |")
        lines += ["", f"Exact rank {design['exact_rank']}, {design['settings']:,} settings, "
                  f"{entry['arms']['uniform']['replicas']} replicas.", ""]
    criteria = record["criteria"]
    lines += ["### Criteria", "",
              "| System | Target | Matched-error verdict | Delta interval | Bootstrap interval |",
              "| --- | ---: | --- | --- | --- |"]
    for cell in criteria["cells"]:
        lines.append(f"| `{cell['system']}` | {float(cell['target']):g} | {cell['verdict']} | "
                     f"{cell['delta_label']} | {cell['bootstrap_label']} |")
    lines += ["", " ".join(f"{name}: **{'PASS' if criteria[name] else 'FAIL'}**."
                           for name in ("C1", "C2", "C3"))]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ main

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--declaration", type=Path, default=producer.DECLARATION)
    parser.add_argument("--results", type=Path, default=producer.RESULTS)
    parser.add_argument("--replay", type=int, default=0, metavar="N")
    parser.add_argument("--require-replay", action="store_true")
    parser.add_argument("--verify-history", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    record = read_record(args.results)
    if args.report:
        print(report(record), end="")
        return 0
    declaration = producer.load_declaration(args.declaration)
    problems = declaration_problems(record, args.declaration)
    systems = {spec["key"]: producer.build_system(spec) for spec in declaration["systems"]}
    problems += design_problems(record, declaration, systems)
    problems += row_problems(record, declaration)
    if not problems:
        problems += summary_problems(record, declaration)
    if args.verify_history:
        problems += history_problems(record, args.declaration, args.results)
    replayed = False
    if args.replay:
        blockers = replay_blockers(record)
        for blocker in blockers:
            print(f"NOTE replay unavailable: {blocker}")
        if not blockers:
            for spec in declaration["systems"]:
                producer.prepare(systems[spec["key"]])
            problems += replay_problems(record, declaration, systems, args.replay)
            replayed = True
        elif args.require_replay:
            problems += [f"replay required but unavailable: {b}" for b in blockers]
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print(f"{len(record.get('rows', []))} rows; "
          f"{'replayed ' + str(args.replay) + ' replica(s) per system; ' if replayed else ''}"
          f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
