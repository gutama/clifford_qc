#!/usr/bin/env python3
"""Verify the bootstrap-at-the-stop records and report them.

Checked, in this order:

1. the criteria file has the digest the fresh record carries, and each record
   names a packaged producer whose bytes match the digest it records;
2. the rows are complete: every declared system, allocation and replica once,
   fresh seeds derived from the fresh root, and every screen run identical to
   the declared target-acquisition record's;
3. every fresh row obeys its arms' rules: arm A stops at a look whose standard
   error meets the target, arm B never earlier and only where the bootstrap
   spread meets it too, and every spread is ``(q84 - q16) / 2`` of its quantiles;
4. the arm-A criterion of ``bootstrap_stop_criteria.md`` is recomputed from the
   rows and compared with the verdict the study document states.

``--replay N`` regenerates the first ``N`` replicas of every system with the
live producer and compares them with the committed rows. It also reruns
``acquire_ritz`` with ``bootstrap_replicates`` at each replayed arm-A stop and
requires the spread it reports to be the recorded one. ``--report`` prints the
Markdown the study document embeds.

    python benchmarks/check_bootstrap_stop.py --replay 1
    python benchmarks/check_bootstrap_stop.py --report
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from benchmarks import run_bootstrap_stop as producer  # noqa: E402
from benchmarks.run_interval_coverage import derived_seed  # noqa: E402
from benchmarks.run_target_acquisition import load_declaration  # noqa: E402

DOCUMENT = ROOT / "docs" / "restart" / "BOOTSTRAP_STOP.md"
CRITERIA_SHA256 = "8f23ed6215e544eed7b496d9ca3e374871b42d24b31c01274c127aa62110f6bf"
# A replayed float passes through an eigensolve; across BLAS kernels it agrees
# to rounding amplified by conditioning, far below any interval width here.
REPLAY_TOLERANCE = 1e-7
WIDE = 2.0  # "wide": a half-width above WIDE * z * target, as the criteria report it
WILSON_Z = 1.959963984540054


def load(path: Path) -> dict:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def wilson(successes: int, trials: int) -> tuple[float, float]:
    """The Wilson 95% score interval of a proportion."""
    p = successes / trials
    scale = 1 + WILSON_Z ** 2 / trials
    centre = (p + WILSON_Z ** 2 / (2 * trials)) / scale
    half = WILSON_Z * math.sqrt(p * (1 - p) / trials + WILSON_Z ** 2 / (4 * trials ** 2)) / scale
    return centre - half, centre + half


def _coverage(successes: int, trials: int) -> str:
    low, high = wilson(successes, trials)
    return f"{successes / trials:.3f} [{low:.3f}, {high:.3f}]"


# ------------------------------------------------------------------- structure

def provenance_problems(record: dict, schema: str) -> list[str]:
    problems = []
    if record.get("schema") != schema:
        problems.append(f"schema is {record.get('schema')!r}, not {schema!r}")
    stamp = record.get("provenance", {})
    path = ROOT / str(stamp.get("producer", ""))
    if not stamp.get("producer") or not path.is_file():
        problems.append(f"{schema}: producer {stamp.get('producer')!r} is not packaged")
    elif producer.sha256(path) != stamp.get("producer_sha256"):
        problems.append(f"{schema}: packaged producer bytes differ from the recorded digest")
    return problems


def fresh_problems(record: dict, declaration: dict) -> list[str]:
    problems = provenance_problems(record, producer.RESULTS_SCHEMA)
    if producer.sha256(producer.CRITERIA) != CRITERIA_SHA256:
        problems.append("the criteria file changed after the fresh run")
    if record.get("criteria_sha256") != CRITERIA_SHA256:
        problems.append("the fresh record names different criteria")
    if (record.get("seed_root"), record.get("replicas"), record.get("bootstrap_replicates")) \
            != (producer.FRESH_ROOT, producer.REPLICAS, producer.BOOTSTRAP_REPLICATES):
        problems.append("the fresh record is not the declared root, replicas and draws")
    keys = [spec["key"] for spec in declaration["systems"]]
    expected = {(system, allocation, replica): derived_seed(producer.FRESH_ROOT, s, replica)
                for s, system in enumerate(keys) for allocation in declaration["allocations"]
                for replica in range(record.get("replicas", 0))}
    seen = set()
    for row in record.get("rows", []):
        key = (row.get("system"), row.get("allocation"), row.get("replica"))
        name = "fresh row {}/{}/{}".format(*key)
        if key not in expected or key in seen:
            problems.append(f"{name} is undeclared or repeated")
            continue
        seen.add(key)
        if "failed" in row:
            continue  # a solve failure is counted, never dropped: the report shows it
        if row.get("seed") != expected[key]:
            problems.append(f"{name} has the wrong seed")
        problems.extend(_arm_problems(row, declaration, name))
    problems.extend(f"fresh row {key} is missing" for key in sorted(set(expected) - seen))
    return problems


def _arm_problems(row: dict, declaration: dict, name: str) -> list[str]:
    problems = []
    last = len(declaration["cumulative_shots_per_group"]) - 1
    for goal in declaration["target_std_errors"]:
        a, b = row["A"].get(repr(goal)), row["B"].get(repr(goal))
        if a is None or b is None:
            problems.append(f"{name} has no arm for target {goal}")
            continue
        for arm, found in (("A", a), ("B", b)):
            q = found["quantiles"]
            if q != sorted(q) or found["robust"] != (q[3] - q[1]) / 2:
                problems.append(f"{name} arm {arm} spread is not its quantiles'")
            met = found["stopped"] == "target_met"
            if found["stopped"] not in ("target_met", "schedule_exhausted"):
                problems.append(f"{name} arm {arm} has an unknown stop")
            if met and found["sigma"] > goal:
                problems.append(f"{name} arm {arm} stopped above the target")
            if not met and found["look"] > last:
                problems.append(f"{name} arm {arm} looked past the schedule")
        if b["stopped"] == "target_met" and b["robust"] > goal:
            problems.append(f"{name} arm B stopped with its spread above the target")
        if b["look"] < a["look"] or b["shots"] < a["shots"]:
            problems.append(f"{name} arm B stopped before arm A")
        if a["stopped"] == "schedule_exhausted" and b != a:
            problems.append(f"{name} an exhausted arm A differs from arm B")
        if b["look"] == a["look"] and b["stopped"] == a["stopped"] and b != a:
            problems.append(f"{name} arms differ at the same look")
    return problems


def screen_problems(record: dict, declaration: dict) -> list[str]:
    problems = provenance_problems(record, producer.SCREEN_SCHEMA)
    if record.get("seed_root") != declaration["seed_root"]:
        problems.append("the screen did not use the declared seed root")
    if record.get("quantile_levels") != list(producer.screen_levels(declaration)):
        problems.append("the screen kept different quantiles")
    stops = producer._declared_stops()
    expected = {(spec["key"], allocation, replica, goal)
                for spec in declaration["systems"] for allocation in declaration["allocations"]
                for replica in range(record.get("replicas", 0))
                for goal in declaration["target_std_errors"]}
    seen = set()
    for row in record.get("rows", []):
        key = (row.get("system"), row.get("allocation"), row.get("replica"),
               row.get("target_std_error"))
        if key not in expected or key in seen:
            problems.append(f"screen row {key} is undeclared or repeated")
            continue
        seen.add(key)
        stored = stops[key[:3]][producer.target_key(key[3])]
        if not row.get("matches") or stored.get("outcome") != row.get("outcome"):
            problems.append(f"screen row {key} is not the declared record's run")
        elif row["outcome"] == "ok" and (stored["estimate"] != row["estimate"]
                                         or stored["looks_taken"] != row["looks"]):
            problems.append(f"screen row {key} differs from the declared record")
        if row.get("outcome") == "ok" and "quantiles" not in row:
            problems.append(f"screen row {key} has no bootstrap")
    problems.extend(f"screen row {key} is missing" for key in sorted(expected - seen))
    return problems


# ---------------------------------------------------------------------- report

def fresh_cells(record: dict, declaration: dict) -> dict:
    z = producer.schedule_z(declaration)
    cells: dict = defaultdict(lambda: defaultdict(list))
    for row in record["rows"]:
        if "failed" in row:
            for goal in declaration["target_std_errors"]:
                cells[(row["system"], row["allocation"], goal)]["failed"].append(1)
            continue
        for goal in declaration["target_std_errors"]:
            cell = cells[(row["system"], row["allocation"], goal)]
            a, b, truth = row["A"][repr(goal)], row["B"][repr(goal)], row["target"]
            cell["delta"].append(abs(a["estimate"] - truth) <= z * a["sigma"])
            cell["A"].append(abs(a["estimate"] - truth) <= z * a["robust"])
            cell["B"].append(abs(b["estimate"] - truth) <= z * b["robust"])
            cell["ratio"].append(a["robust"] / a["sigma"])
            cell["A_wide"].append(a["robust"] > WIDE * goal)
            cell["B_wide"].append(b["robust"] > WIDE * goal)
            cell["A_shots"].append(a["shots"])
            cell["B_shots"].append(b["shots"])
            cell["B_exhausted"].append(b["stopped"] == "schedule_exhausted")
    return cells


def criterion(cells: dict) -> bool:
    """Arm A, as fixed before the run: no cell resolved below 0.95, tfim6 at least 0.95."""
    for (system, _, _), cell in cells.items():
        if cell["failed"]:
            return False
        covered, n = sum(cell["A"]), len(cell["A"])
        if wilson(covered, n)[1] < 0.95 or (system == "tfim6_mixed" and covered / n < 0.95):
            return False
    return True


def fresh_report(record: dict, declaration: dict) -> str:
    cells = fresh_cells(record, declaration)
    lines = [
        "| Bank | Allocation | Target | n | Delta at A stop | A: bootstrap at A stop | "
        "A median r/σ | A wide | B: bootstrap-confirmed stop | B wide | B shots / A shots | "
        "B exhausted |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for key in sorted(cells):
        cell, n = cells[key], len(cells[key]["A"])
        lines.append(
            f"| `{key[0]}` | {key[1]} | {key[2]:g} | {n} | {_coverage(sum(cell['delta']), n)} | "
            f"{_coverage(sum(cell['A']), n)} | {np.median(cell['ratio']):.2f} | "
            f"{np.mean(cell['A_wide']):.2f} | {_coverage(sum(cell['B']), n)} | "
            f"{np.mean(cell['B_wide']):.2f} | "
            f"{np.mean(cell['B_shots']) / np.mean(cell['A_shots']):.2f} | "
            f"{np.mean(cell['B_exhausted']):.2f} |")
    failed = sum(len(cell["failed"]) for cell in cells.values()) // len(
        declaration["target_std_errors"])
    lines += ["", f"{len(record['rows'])} runs, {failed} solve failures. "
              f"Arm A criterion: **{'PASS' if criterion(cells) else 'FAIL'}**."]
    return "\n".join(lines)


SCREEN_CONSTRUCTIONS = ("delta", "percentile", "basic", "spread")


def screen_intervals(row: dict, z: float) -> dict:
    tail, _, p16, _, p84, _, upper_tail = row["quantiles"]
    estimate, sigma = row["estimate"], row["sigma"]
    spread = (p84 - p16) / 2
    return {"delta": (estimate - z * sigma, estimate + z * sigma),
            "percentile": (tail, upper_tail),
            "basic": (2 * estimate - upper_tail, 2 * estimate - tail),
            "spread": (estimate - z * spread, estimate + z * spread)}


def screen_report(record: dict, declaration: dict) -> str:
    z = producer.schedule_z(declaration)
    cells: dict = defaultdict(list)
    for row in record["rows"]:
        cells[(row["system"], row["allocation"], row["target_std_error"])].append(row)
    lines = [
        "| Bank | Allocation | Target | n | Truncated stops | Delta | Percentile | Basic | "
        "Bootstrap spread | Median width: percentile / delta | spread / delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(cells):
        rows = [row for row in cells[key] if row["outcome"] == "ok"]
        covered = dict.fromkeys(SCREEN_CONSTRUCTIONS, 0)
        widths: dict = defaultdict(list)
        for row in rows:
            truth = row["estimate"] - row["error"]
            for name, (low, high) in screen_intervals(row, z).items():
                covered[name] += low <= truth <= high
                widths[name].append(high - low)
        delta_width = np.asarray(widths["delta"])
        ratio = {name: float(np.median(np.asarray(widths[name]) / delta_width))
                 for name in ("percentile", "spread")}
        truncated = sum(row["rank"] < row["exact_rank"] for row in rows)
        lines.append(
            f"| `{key[0]}` | {key[1]} | {key[2]:g} | {len(rows)} | {truncated} | "
            + " | ".join(f"{covered[name] / len(rows):.3f}" for name in SCREEN_CONSTRUCTIONS)
            + f" | {ratio['percentile']:.2f} | {ratio['spread']:.2f} |")
    return "\n".join(lines)


def report(fresh: dict, screen: dict, declaration: dict) -> str:
    return ("### Fresh seeds\n\n" + fresh_report(fresh, declaration)
            + "\n\n### Development screen\n\n" + screen_report(screen, declaration) + "\n")


# ---------------------------------------------------------------------- replay

def replay_problems(record: dict, declaration: dict, replicas: int) -> list[str]:
    from clifford_qc.backends.finite_shot import FiniteShotBackend
    from clifford_qc.subspace.acquisition import acquire_ritz
    from benchmarks.run_interval_coverage import build_system
    from benchmarks.run_target_acquisition import schedule

    from clifford_qc.reproducibility import compare_json_records

    stored = {(row["system"], row["allocation"], row["replica"]): row for row in record["rows"]}
    problems: list[str] = []
    for s, spec in enumerate(declaration["systems"]):
        shared, _ = build_system(spec)
        pilot, budgets = schedule(declaration, len(shared.groups))
        for replica in range(replicas):
            for a, allocation in enumerate(declaration["allocations"]):
                fresh = json.loads(json.dumps(producer.fresh_row(declaration, s, replica, a)))
                old = stored[(spec["key"], allocation, replica)]
                problems.extend(compare_json_records(
                    old, fresh, f"{spec['key']}/{allocation}/{replica}",
                    atol=REPLAY_TOLERANCE, rtol=REPLAY_TOLERANCE))
                if "failed" in old:
                    continue
                for t, goal in enumerate(declaration["target_std_errors"]):
                    arm = old["A"][repr(goal)]
                    found = acquire_ritz(
                        shared, FiniteShotBackend(seed=old["seed"]), pilot_shots_per_group=pilot,
                        round_budgets=budgets, allocation=allocation, target_std_error=goal,
                        solver_kwargs=dict(declaration["solver_kwargs"]),
                        delta=declaration["delta"],
                        bootstrap_replicates=record["bootstrap_replicates"],
                        bootstrap_seed=derived_seed(record["seed_root"],
                                                    producer.BOOTSTRAP_STREAM + s, replica,
                                                    10 * a + t, arm["look"]))
                    spread = found.looks[-1].get("bootstrap_spread")
                    if len(found.looks) - 1 != arm["look"] or spread is None or not math.isclose(
                            spread, arm["robust"], rel_tol=REPLAY_TOLERANCE,
                            abs_tol=REPLAY_TOLERANCE):
                        problems.append(f"{spec['key']}/{allocation}/{replica} target {goal}: "
                                        "acquire_ritz's bootstrap is not arm A's")
    return problems


def stated_verdict(document: str) -> str | None:
    for verdict in ("PASS", "FAIL"):
        if f"Arm A criterion: **{verdict}**" in document:
            return verdict
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=producer.RESULTS)
    parser.add_argument("--screen", type=Path, default=producer.SCREEN)
    parser.add_argument("--replay", type=int, default=0, metavar="N")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    declaration = load_declaration()
    fresh, screen = load(args.results), load(args.screen)
    if args.report:
        print(report(fresh, screen, declaration), end="")
        return 0
    problems = fresh_problems(fresh, declaration) + screen_problems(screen, declaration)
    verdict = "PASS" if criterion(fresh_cells(fresh, declaration)) else "FAIL"
    if DOCUMENT.is_file() and stated_verdict(DOCUMENT.read_text()) != verdict:
        problems.append(f"the study document does not state the recomputed verdict {verdict}")
    if args.replay:
        problems += replay_problems(fresh, declaration, args.replay)
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print(f"arm A criterion recomputed: {verdict}; "
          f"{'replayed ' + str(args.replay) + ' replica(s) per system; ' if args.replay else ''}"
          f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
