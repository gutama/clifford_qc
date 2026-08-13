"""Aggregate benchmark JSONL into per-(model, method) tables.

Reports median and IQR for relative error, shots, and circuits, plus
near-optimality and ambiguity rates — the resource-accounting quantities
of PLAN.md section 9.5. Writes CSV when --csv is given, always
prints a markdown table.

Run: python benchmarks/summarize.py results.jsonl [--csv summary.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def model_family(name: str) -> str:
    """Pool disorder realizations: random_ising(n=4,seed=7,obc) ->
    random_ising(n=4,obc). Each seed is a different Hamiltonian instance,
    but the aggregate row should describe the ensemble."""
    return re.sub(r"seed=\d+,?", "", name).replace(",)", ")")


def quartiles(values):
    values = sorted(values)
    if not values:
        return (None, None, None)
    med = statistics.median(values)
    half = len(values) // 2
    lo = statistics.median(values[:half]) if half else values[0]
    hi = statistics.median(values[-half:]) if half else values[-1]
    return lo, med, hi


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(model_family(row["model"]), row["method"])].append(row)
    out = []
    for (model, method), runs in sorted(groups.items()):
        rel = quartiles([r["relative_error"] for r in runs if r["relative_error"] is not None])
        shots = quartiles([r["total_shots"] for r in runs])
        circuits = quartiles([r["total_circuits"] for r in runs])
        near = [r["near_optimal_rate"] for r in runs if r["near_optimal_rate"] is not None]
        ambiguous = sum(r["status_counts"].get("budget_exhausted_ambiguous", 0) for r in runs)
        steps = sum(r["selection_steps"] for r in runs)
        out.append({
            "model": model, "method": method, "runs": len(runs),
            "rel_err_q1": rel[0], "rel_err_median": rel[1], "rel_err_q3": rel[2],
            "shots_q1": shots[0], "shots_median": shots[1], "shots_q3": shots[2],
            "circuits_median": circuits[1],
            "operators_median": statistics.median([r["operators"] for r in runs]),
            "near_optimal_rate": statistics.mean(near) if near else None,
            "ambiguous_step_rate": ambiguous / steps if steps else None,
        })
    return out


def fmt(x, spec=".2e"):
    if x is None:
        return "-"
    return format(x, spec)


def print_markdown(summary):
    header = ("| model | method | runs | rel err (med [IQR]) | shots (med) "
              "| circuits (med) | ops (med) | near-opt | ambiguous |")
    print(header)
    print("|" + "---|" * 9)
    for row in summary:
        print(f"| {row['model']} | {row['method']} | {row['runs']} "
              f"| {fmt(row['rel_err_median'])} [{fmt(row['rel_err_q1'])}, {fmt(row['rel_err_q3'])}] "
              f"| {fmt(row['shots_median'], ',.0f')} "
              f"| {fmt(row['circuits_median'], ',.0f')} "
              f"| {fmt(row['operators_median'], '.0f')} "
              f"| {fmt(row['near_optimal_rate'], '.2f')} "
              f"| {fmt(row['ambiguous_step_rate'], '.2f')} |")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="JSONL file from run_benchmark.py")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line]
    if not rows:
        raise SystemExit(f"no benchmark rows in {args.results}")
    summary = aggregate(rows)
    print_markdown(summary)
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        # stderr, not stdout: stdout is the Markdown table, and callers
        # redirect it into *_summary.md. Printing the notice there appended a
        # stray "wrote <path>" line to every committed Markdown summary and
        # made the artifact depend on the --csv path used to generate it.
        print(f"wrote {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
