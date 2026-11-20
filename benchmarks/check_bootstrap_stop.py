#!/usr/bin/env python3
"""Verify the bootstrap-at-the-stop records and report them.

Checked, in this order:

1. the criteria file has the digest the fresh record carries, and each record
   names a packaged producer whose bytes match the digest it records, and the
   dependency versions it was produced under;
2. the rows are complete: the declared replicas and draws, every system,
   allocation and replica once, fresh seeds derived from the fresh root, each
   bank's exact energy and rank, and every screen run identical to the declared
   target-acquisition record's in estimate, looks, shots, standard error and stop;
3. every fresh row obeys its arms' rules: each look is in the schedule with the
   schedule's shots; arm A stops at the first look whose standard error meets
   the target and exhausts only at the last look having never met it; arm B
   never stops earlier, and stops only where the bootstrap spread meets the
   target too; every spread is half its ``q84 - q16``;
4. the arm-A criterion of ``bootstrap_stop_criteria.md`` is recomputed from the
   rows and compared with the verdict the study document states. A failed solve
   or bootstrap leaves no interval and counts as a miss.

``--replay N`` regenerates the first ``N`` replicas of every system with the
live producer and compares them with the committed rows within
``REPLAY_TOLERANCE``. It also reruns ``acquire_ritz`` with
``bootstrap_replicates`` at each replayed arm-A stop and requires the same
stop, shots, estimate, spread and interval. Finite-shot values are comparable
only under the NumPy and SciPy that produced the record, so under others the
replay is skipped with a note, or fails with ``--require-replay``. ``--report``
prints the Markdown the study document embeds. ``--reported-intervals``
reruns every arm-A stop without the bootstrap and compares the interval
``acquire_ritz`` itself reports with the ``E +- z sigma`` of the criteria.

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
from benchmarks.run_interval_coverage import build_system, derived_seed  # noqa: E402
from benchmarks.run_target_acquisition import load_declaration, schedule  # noqa: E402
from clifford_qc.backends.finite_shot import FiniteShotBackend  # noqa: E402
from clifford_qc.reproducibility import (  # noqa: E402
    compare_json_records, sampling_stream_mismatch,
)
from clifford_qc.subspace.acquisition import acquire_ritz  # noqa: E402

DOCUMENT = ROOT / "docs" / "restart" / "BOOTSTRAP_STOP.md"
CRITERIA_SHA256 = "8f23ed6215e544eed7b496d9ca3e374871b42d24b31c01274c127aa62110f6bf"
# A replayed float passes through an eigensolve; across BLAS kernels it agrees
# to rounding amplified by conditioning, far below any interval width here.
REPLAY_TOLERANCE = 1e-7
# The exact subspace energy is recomputed here, possibly on another BLAS kernel.
TARGET_TOLERANCE = 1e-9
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

_SHAPES: dict = {}


def system_shape(declaration: dict, key: str) -> tuple[float, int, tuple[int, ...]]:
    """A bank's exact subspace energy, exact rank and cumulative shots at every look."""
    if key not in _SHAPES:
        spec = next(spec for spec in declaration["systems"] if spec["key"] == key)
        shared, design = build_system(spec)
        groups = len(shared.groups)
        pilot, budgets = schedule(declaration, groups)
        cumulative = [pilot * groups]
        for budget in budgets:
            cumulative.append(cumulative[-1] + budget)
        _SHAPES[key] = (float(design["target"]), int(design["exact_rank"]), tuple(cumulative))
    return _SHAPES[key]


def _finite_sorted(values, length: int) -> bool:
    return (isinstance(values, list) and len(values) == length
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
            and values == sorted(values))


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
    if not isinstance(stamp.get("dependencies"), dict):
        problems.append(f"{schema}: the record does not name its dependency versions")
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
                for replica in range(producer.REPLICAS)}
    seen = set()
    for row in record.get("rows", []):
        key = (row.get("system"), row.get("allocation"), row.get("replica"))
        name = "fresh row {}/{}/{}".format(*key)
        if key not in expected or key in seen:
            problems.append(f"{name} is undeclared or repeated")
            continue
        seen.add(key)
        target, rank, cumulative = system_shape(declaration, key[0])
        if (row.get("seed"), row.get("exact_rank")) != (expected[key], rank) \
                or not math.isclose(row.get("target", math.nan), target, abs_tol=TARGET_TOLERANCE):
            problems.append(f"{name} has the wrong seed, exact rank or target")
        problems.extend(_arm_problems(row, declaration, name, cumulative))
    problems.extend(f"fresh row {key} is missing" for key in sorted(set(expected) - seen))
    return problems


def _arm_problems(row: dict, declaration: dict, name: str,
                  cumulative: tuple[int, ...]) -> list[str]:
    """One fresh row against the two arms' rules, as ``fresh_row`` applies them."""
    problems = []
    last = len(cumulative) - 1
    for goal in declaration["target_std_errors"]:
        a, b = row.get("A", {}).get(repr(goal)), row.get("B", {}).get(repr(goal))
        if not isinstance(a, dict) or not isinstance(b, dict):
            problems.append(f"{name} has no arm for target {goal}")
            continue
        broken = False
        for arm, found in (("A", a), ("B", b)):
            label = f"{name} arm {arm} target {goal}"
            look = found.get("look")
            if not isinstance(look, int) or not 0 <= look <= last:
                problems.append(f"{label} has no look in the schedule")
                broken = True
                continue
            if found.get("shots") != cumulative[look]:
                problems.append(f"{label} does not have the schedule's shots at its look")
            if "failed" in found:
                continue  # a solve failure: counted as a miss, nothing else to check
            stopped = found.get("stopped")
            if stopped not in ("target_met", "schedule_exhausted"):
                problems.append(f"{label} has an unknown stop")
            if "boot_failed" not in found and (
                    not _finite_sorted(found.get("quantiles"), len(producer.FRESH_LEVELS))
                    or found.get("robust") != (found["quantiles"][3] - found["quantiles"][1]) / 2):
                problems.append(f"{label} spread is not half its q84 - q16")
            sigma = found.get("sigma")
            if not isinstance(sigma, float) or not math.isfinite(sigma) or sigma < 0:
                problems.append(f"{label} has no standard error")
                broken = True
                continue
            if stopped == "target_met" and sigma > goal:
                problems.append(f"{label} stopped above the target")
            if stopped == "schedule_exhausted" and look != last:
                problems.append(f"{label} exhausted the schedule before its last look")
        if broken:
            continue
        if "failed" in a:
            if b != a:
                problems.append(f"{name} target {goal}: arm A failed but arm B did not fail there")
            continue
        if "failed" not in b:
            spread_met = b.get("robust", math.inf) <= goal
            if b["stopped"] == "target_met" and not spread_met:
                problems.append(f"{name} arm B stopped with its spread above the target")
            if b["stopped"] == "schedule_exhausted" and b["sigma"] <= goal and spread_met:
                problems.append(f"{name} arm B exhausted although both measures met the target")
        if a["stopped"] == "schedule_exhausted" and a["sigma"] <= goal:
            problems.append(f"{name} arm A exhausted although its standard error met the target")
        if b["look"] < a["look"] or ("failed" in b and b["look"] == a["look"]):
            problems.append(f"{name} arm B stopped before arm A")
        if a["stopped"] == "schedule_exhausted" and b != a:
            problems.append(f"{name} an exhausted arm A differs from arm B")
        if b["look"] == a["look"] and b.get("stopped") == a["stopped"] and b != a:
            problems.append(f"{name} arms differ at the same look")
    return problems


def screen_problems(record: dict, declaration: dict) -> list[str]:
    problems = provenance_problems(record, producer.SCREEN_SCHEMA)
    if record.get("seed_root") != declaration["seed_root"]:
        problems.append("the screen did not use the declared seed root")
    levels = list(producer.screen_levels(declaration))
    if record.get("quantile_levels") != levels:
        problems.append("the screen kept different quantiles")
    if (record.get("replicas"), record.get("bootstrap_replicates")) \
            != (producer.REPLICAS, producer.BOOTSTRAP_REPLICATES):
        problems.append("the screen is not the declared replicas and draws")
    stops = producer._declared_stops()
    expected = {(spec["key"], allocation, replica, goal)
                for spec in declaration["systems"] for allocation in declaration["allocations"]
                for replica in range(producer.REPLICAS)
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
            continue
        if row["outcome"] != "ok":
            continue
        if (stored["estimate"], stored["looks_taken"], stored["shots"], stored["std_error"],
                stored["stopped"]) != (row.get("estimate"), row.get("looks"), row.get("shots"),
                                       row.get("sigma"), row.get("stopped")):
            problems.append(f"screen row {key} differs from the declared record")
            continue
        target, rank, _ = system_shape(declaration, key[0])
        if row.get("exact_rank") != rank or not isinstance(row.get("error"), float) \
                or not math.isclose(row["estimate"] - row["error"], target,
                                    abs_tol=TARGET_TOLERANCE):
            problems.append(f"screen row {key} has the wrong error or exact rank")
        if "boot_failed" in row:
            continue  # counted as a miss of every bootstrap construction
        if not _finite_sorted(row.get("quantiles"), len(levels)) \
                or row.get("draws") != record.get("bootstrap_replicates"):
            problems.append(f"screen row {key} does not keep its quantiles")
    problems.extend(f"screen row {key} is missing" for key in sorted(expected - seen))
    return problems


# ---------------------------------------------------------------------- report

def fresh_cells(record: dict, declaration: dict) -> dict:
    """Per cell, one entry per run. A failed arm has no interval: it is a miss."""
    z = producer.schedule_z(declaration)
    cells: dict = defaultdict(lambda: defaultdict(list))

    def covers(found: dict, spread: str, truth: float) -> bool:
        width = found.get(spread)
        return width is not None and abs(found["estimate"] - truth) <= z * width

    for row in record["rows"]:
        for goal in declaration["target_std_errors"]:
            cell = cells[(row["system"], row["allocation"], goal)]
            a, b, truth = row["A"][repr(goal)], row["B"][repr(goal)], row["target"]
            cell["delta"].append(covers(a, "sigma", truth))
            cell["A"].append(covers(a, "robust", truth))
            cell["B"].append(covers(b, "robust", truth))
            for arm, found in (("A", a), ("B", b)):
                cell[f"{arm}_shots"].append(found["shots"])
                if "robust" in found:
                    cell[f"{arm}_wide"].append(found["robust"] > WIDE * goal)
            if "robust" in a:
                cell["ratio"].append(a["robust"] / a["sigma"])
            cell["B_exhausted"].append(b.get("stopped") == "schedule_exhausted")
            cell["solve_failed"].append("failed" in a or "failed" in b)
            cell["boot_failed"].append("boot_failed" in a or "boot_failed" in b)
    return cells


def criterion(cells: dict) -> bool:
    """Arm A, as fixed before the run: no cell resolved below 0.95, tfim6 at least 0.95."""
    for (system, _, _), cell in cells.items():
        covered, n = sum(cell["A"]), len(cell["A"])
        if wilson(covered, n)[1] < 0.95 or (system == "tfim6_mixed" and covered / n < 0.95):
            return False
    return True


def _mean(values) -> str:
    return f"{np.mean(values):.2f}" if len(values) else "n/a"


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
        ratio = f"{np.median(cell['ratio']):.2f}" if cell["ratio"] else "n/a"
        lines.append(
            f"| `{key[0]}` | {key[1]} | {key[2]:g} | {n} | {_coverage(sum(cell['delta']), n)} | "
            f"{_coverage(sum(cell['A']), n)} | {ratio} | {_mean(cell['A_wide'])} | "
            f"{_coverage(sum(cell['B']), n)} | {_mean(cell['B_wide'])} | "
            f"{np.mean(cell['B_shots']) / np.mean(cell['A_shots']):.2f} | "
            f"{np.mean(cell['B_exhausted']):.2f} |")
    solve = sum(sum(cell["solve_failed"]) for cell in cells.values())
    boot = sum(sum(cell["boot_failed"]) for cell in cells.values())
    lines += ["", f"{len(record['rows'])} runs; {solve} stops with a solve failure and "
              f"{boot} with a bootstrap failure, each counted as a miss. "
              f"Arm A criterion: **{'PASS' if criterion(cells) else 'FAIL'}**."]
    return "\n".join(lines)


SCREEN_CONSTRUCTIONS = ("delta", "percentile", "basic", "spread")


def screen_intervals(row: dict, z: float) -> dict:
    """Each construction's interval; one whose bootstrap failed has none."""
    estimate, sigma = row["estimate"], row["sigma"]
    out: dict = {"delta": (estimate - z * sigma, estimate + z * sigma)}
    if "quantiles" not in row:
        return {**out, "percentile": None, "basic": None, "spread": None}
    tail, _, p16, _, p84, _, upper_tail = row["quantiles"]
    spread = (p84 - p16) / 2
    return {**out, "percentile": (tail, upper_tail),
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
        ratios: dict = defaultdict(list)
        for row in rows:
            truth = row["estimate"] - row["error"]
            found = screen_intervals(row, z)
            for name, bounds in found.items():
                if bounds is not None:
                    covered[name] += bounds[0] <= truth <= bounds[1]
                    ratios[name].append((bounds[1] - bounds[0])
                                        / (found["delta"][1] - found["delta"][0]))
        ratio = {name: f"{np.median(ratios[name]):.2f}" if ratios[name] else "n/a"
                 for name in ("percentile", "spread")}
        truncated = sum(row["rank"] < row["exact_rank"] for row in rows)
        lines.append(
            f"| `{key[0]}` | {key[1]} | {key[2]:g} | {len(rows)} | {truncated} | "
            + " | ".join(f"{covered[name] / len(rows):.3f}" for name in SCREEN_CONSTRUCTIONS)
            + f" | {ratio['percentile']} | {ratio['spread']} |")
    return "\n".join(lines)


def report(fresh: dict, screen: dict, declaration: dict) -> str:
    return ("### Fresh seeds\n\n" + fresh_report(fresh, declaration)
            + "\n\n### Development screen\n\n" + screen_report(screen, declaration) + "\n")


# ---------------------------------------------------------------------- replay

def replay_blockers(record: dict) -> list[str]:
    """Why values here could not be the record's: a different NumPy or SciPy."""
    return sampling_stream_mismatch(record, packages=("numpy", "scipy"))


def _close(left, right) -> bool:
    return (isinstance(left, (int, float)) and isinstance(right, (int, float))
            and math.isclose(left, right, rel_tol=REPLAY_TOLERANCE, abs_tol=REPLAY_TOLERANCE))


def replay_problems(record: dict, declaration: dict, replicas: int) -> list[str]:
    """The first replicas regenerated by the producer and by ``acquire_ritz``.

    Each committed row is regenerated with the record's seed root and draw
    count and compared within ``REPLAY_TOLERANCE``. At every arm-A stop that
    bootstrapped, ``acquire_ritz(bootstrap_replicates=...)`` with the same seed
    must stop at the same look with the same shots and estimate and report the
    recorded spread and the interval ``E +- z r``; where the bootstrap failed it
    must report no interval.
    """
    z = producer.schedule_z(declaration)
    stored = {(row["system"], row["allocation"], row["replica"]): row for row in record["rows"]}
    problems: list[str] = []
    for s, spec in enumerate(declaration["systems"]):
        shared, _ = build_system(spec)
        pilot, budgets = schedule(declaration, len(shared.groups))
        for replica in range(replicas):
            for a, allocation in enumerate(declaration["allocations"]):
                name = f"{spec['key']}/{allocation}/{replica}"
                fresh = json.loads(json.dumps(producer.fresh_row(
                    declaration, s, replica, a, root=record["seed_root"],
                    replicates=record["bootstrap_replicates"])))
                old = stored[(spec["key"], allocation, replica)]
                problems.extend(compare_json_records(old, fresh, name, atol=REPLAY_TOLERANCE,
                                                     rtol=REPLAY_TOLERANCE))
                for t, goal in enumerate(declaration["target_std_errors"]):
                    arm = old["A"][repr(goal)]
                    if "failed" in arm:
                        continue
                    found = acquire_ritz(
                        shared, FiniteShotBackend(seed=old["seed"]), pilot_shots_per_group=pilot,
                        round_budgets=budgets, allocation=allocation, target_std_error=goal,
                        solver_kwargs=dict(declaration["solver_kwargs"]),
                        delta=declaration["delta"],
                        bootstrap_replicates=record["bootstrap_replicates"],
                        bootstrap_seed=derived_seed(record["seed_root"],
                                                    producer.BOOTSTRAP_STREAM + s, replica,
                                                    10 * a + t, arm["look"]))
                    interval = found.bootstrap_interval
                    if "robust" in arm:
                        radius = z * arm["robust"]
                        agrees = interval is not None and all(_close(*pair) for pair in (
                            (found.looks[-1].get("bootstrap_spread"), arm["robust"]),
                            (interval.lower, arm["estimate"] - radius),
                            (interval.upper, arm["estimate"] + radius)))
                    else:
                        agrees = interval is None and "bootstrap_failure" in found.looks[-1]
                    if not (agrees and len(found.looks) - 1 == arm["look"]
                            and found.shots == arm["shots"]
                            and _close(found.energy, arm["estimate"])):
                        problems.append(f"{name} target {goal}: acquire_ritz's bootstrap "
                                        "stop is not arm A's")
    return problems


def _reported_intervals(task: tuple) -> list[tuple]:
    declaration, row = task
    spec = next(spec for spec in declaration["systems"] if spec["key"] == row["system"])
    shared, _ = build_system(spec)
    pilot, budgets = schedule(declaration, len(shared.groups))
    z = producer.schedule_z(declaration)
    out = []
    for goal in declaration["target_std_errors"]:
        arm = row["A"][repr(goal)]
        if "failed" in arm:
            continue
        found = acquire_ritz(shared, FiniteShotBackend(seed=row["seed"]),
                             pilot_shots_per_group=pilot, round_budgets=budgets,
                             allocation=row["allocation"], target_std_error=goal,
                             solver_kwargs=dict(declaration["solver_kwargs"]),
                             delta=declaration["delta"])
        same = len(found.looks) - 1 == arm["look"] and _close(found.energy, arm["estimate"])
        truth, interval = row["target"], found.interval
        out.append((same, (interval.upper - found.energy) / (z * arm["sigma"]),
                    abs(arm["estimate"] - truth) <= z * arm["sigma"],
                    interval.lower <= truth <= interval.upper))
    return out


def reported_interval_comparison(record: dict, declaration: dict, jobs: int = 1) -> dict:
    """``acquire_ritz``'s own interval at every arm-A stop, against ``E +- z sigma``.

    The criteria compare the bootstrap interval with ``E +- z sigma``, ``sigma``
    being the standard error the stopping rule tested. ``acquire_ritz`` reports
    a radius built from per-setting statistics that floor each setting's
    variance at ``range**2 / (4 (N + 1))``
    (``GroupedWordCache.candidate_group_statistics``), so it is never narrower.
    This reruns every stop without the bootstrap and measures by how much, and
    whether the two ever cover differently.
    """
    from concurrent.futures import ProcessPoolExecutor

    tasks = [(declaration, row) for row in record["rows"]]
    with ProcessPoolExecutor(jobs) as pool:
        found = [item for chunk in pool.map(_reported_intervals, tasks, chunksize=8)
                 for item in chunk]
    ratios = [ratio for _, ratio, _, _ in found]
    return {"stops": len(found), "different_stops": sum(not same for same, *_ in found),
            "min_ratio": min(ratios), "max_ratio": max(ratios),
            "different_coverage": sum(a != b for *_, a, b in found)}


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
    parser.add_argument("--require-replay", action="store_true",
                        help="fail instead of skipping a replay this environment cannot make")
    parser.add_argument("--reported-intervals", action="store_true",
                        help="rerun every arm-A stop and compare acquire_ritz's own interval")
    parser.add_argument("--jobs", type=int, default=1)
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
    replayed = False
    if args.replay:
        blockers = replay_blockers(fresh)
        for blocker in blockers:
            print(f"NOTE replay unavailable: {blocker}")
        if not blockers:
            problems += replay_problems(fresh, declaration, args.replay)
            replayed = True
        elif args.require_replay:
            problems += [f"replay required but unavailable: {b}" for b in blockers]
    if args.reported_intervals:
        blockers = replay_blockers(fresh)
        for blocker in blockers:
            print(f"NOTE reported intervals unavailable: {blocker}")
        if not blockers:
            found = reported_interval_comparison(fresh, declaration, args.jobs)
            print(f"acquire_ritz interval / (z sigma) over {found['stops']} stops: "
                  f"{found['min_ratio']:.4f} to {found['max_ratio']:.4f}; covers differently "
                  f"in {found['different_coverage']} run(s)")
            if found["different_stops"]:
                problems.append(f"{found['different_stops']} rerun stop(s) differ from arm A")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print(f"arm A criterion recomputed: {verdict}; "
          f"{'replayed ' + str(args.replay) + ' replica(s) per system; ' if replayed else ''}"
          f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
