"""Aggregate an A-CASE ladder JSONL into the §7 comparison table.

``summarize.py`` understands the ``run_benchmark.py`` schema (per-seed ADAPT
runs, aggregated by median and IQR). A ladder record is a different object: one
deterministic run per (system, method), and the interesting columns are the §6
resource metrics beside the energy, not the spread over seeds. Hence a second
summarizer rather than a schema compromise that would serve neither.

Every row keeps its ``evidence`` label. A certified finite-shot energy can come
out *below* the exact one -- thresholding a noisy overlap matrix is a PSD repair
with no variational guarantee -- so a table that sorted on energy alone would
report a noisy run as the winner. The label is what stops that reading.

Run: python benchmarks/summarize_ladder.py RECORD.jsonl [--csv summary.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

COLUMNS = [
    ("rung", "rung"), ("system", "system"), ("n", "n"), ("method", "method"),
    ("evidence", "evidence"), ("energy", "energy"), ("error", "error"),
    ("chem_acc", "chemical_accuracy"), ("basis", "basis_size"),
    ("ops", "operators"), ("W", "word_universe"), ("S_H", "max_element_support"),
    ("rank", "retained_rank"), ("kappa_S", "condition_number"),
    ("shots", "total_shots"), ("circuits", "total_circuits"),
    ("abstain", "abstentions"),
]

# The sampled-subspace arms spend a different budget, and the main table has no
# column for most of it. Left in the main table alone a QSCI row reads as a
# method that costs nothing -- W really is zero, and every other resource column
# is blank -- which is the exact misreading `LITERATURE_ROADMAP.md` §0.1 warns
# against. These are the §8E Pareto axes, reported beside the energy.
SAMPLED_COLUMNS = [
    ("rung", "rung_name"), ("input", "sampling_state"),
    ("evidence", "input_category"), ("mode", "sampling_mode"),
    ("M", "subspace_dimension"), ("error", "error"),
    ("draws", "raw_shots"), ("unique", "unique_configurations"),
    ("dup", "duplicate_fraction"), ("discard", "discarded_fraction"),
    ("kept_p", "retained_probability"), ("preps", "state_preparations"),
    ("W", "projected_matrix_words"), ("nnz", "matrix_nonzeros"),
    ("bytes", "matrix_bytes"), ("build_s", "build_seconds"),
    ("solve_s", "solve_seconds"),
]


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} has no records")
    missing = [key for key in ("system", "method", "energy", "reference_energy")
               if any(key not in row for row in rows)]
    if missing:
        raise SystemExit(f"{path} is not an A-CASE ladder record "
                         f"(missing {', '.join(sorted(set(missing)))})")
    return rows


def cell(row: dict, key: str) -> str:
    value = row.get(key)
    if value is None:
        # A missing support column means one of two different things: the method
        # has no projected matrix (the reference determinant), or the row was
        # solved through the cyclic contraction because its generators were too
        # wide to track. Rendering both as "-" would let a deliberate fallback
        # read as an inapplicable column.
        if key in ("word_universe", "max_element_support") \
                and row.get("support_tracked") is False:
            return "n/t"
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if key in ("energy",):
            return f"{value:.8f}"
        if key in ("error",):
            return f"{value:+.2e}"
        if key in ("condition_number",):
            return f"{value:.2e}"
        return f"{value:.4g}"
    if isinstance(value, int):
        return f"{value:,}" if key in ("total_shots", "total_circuits") else str(value)
    return str(value)


def ordered(rows: list[dict]) -> list[dict]:
    """Ladder order, then the method order the config declared per rung."""
    order: dict[tuple, int] = {}
    for index, row in enumerate(rows):
        order.setdefault((row.get("rung_name", row["system"]), row["method"]), index)
    return sorted(rows, key=lambda row: order[(row.get("rung_name", row["system"]),
                                               row["method"])])


def sampled_subspace_section(rows: list[dict]) -> list[str]:
    """The §8E resource table for QSCI-family rows, or nothing if there are none."""
    sampled = [row for row in ordered(rows) if row.get("family") == "qsci"]
    if not sampled:
        return []
    lines = ["", "### Sampled subspaces (QSCI): the resources that replace `W`", ""]
    lines.append("| " + " | ".join(name for name, _ in SAMPLED_COLUMNS) + " |")
    lines.append("|" + "|".join("---" for _ in SAMPLED_COLUMNS) + "|")
    for row in sampled:
        lines.append("| " + " | ".join(cell(row, key)
                                       for _, key in SAMPLED_COLUMNS) + " |")
    categories = {row.get("input_category") for row in sampled}
    lines.append("")
    lines.append("`evidence` here is the *input* category, not the arithmetic. "
                 "`oracle` rows sample an exact eigenvector no device can "
                 "prepare: they validate the method and bound what sampling "
                 "could achieve, and they carry no `preps` because there is no "
                 "preparation to count. Reading an oracle row on the same "
                 "resource axis as an `implementable` one advertises a frontier "
                 "nothing can reach.")
    if categories == {"oracle", "implementable"}:
        lines.append("")
        lines.append("Both categories are present above, so this table is a "
                     "record, not a comparison. Any Pareto frontier drawn from "
                     "it must be drawn within one category.")
    return lines


def markdown(rows: list[dict]) -> str:
    lines = ["| " + " | ".join(name for name, _ in COLUMNS) + " |",
             "|" + "|".join("---" for _ in COLUMNS) + "|"]
    for row in ordered(rows):
        lines.append("| " + " | ".join(cell(row, key) for _, key in COLUMNS) + " |")

    lines.append("")
    lines.append("### Reached chemical accuracy (1.6 mHa) against the sector reference")
    lines.append("")
    # The reference determinant is excluded from the "most compact" column, and
    # reported on its own: on a small active space HF can already sit inside
    # 1.6 mHa of the sector FCI, and a table that ranked it by basis size would
    # name the reference state the most compact subspace method. That is not a
    # result about subspaces, it is a rung that does not discriminate -- so the
    # table says so instead.
    excluded = {"exact", "reference_state"}
    lines.append("| rung | reference determinant | methods reaching accuracy "
                 "| most compact exact-arithmetic |")
    lines.append("|---|---|---|---|")
    for name in dict.fromkeys(row.get("rung_name", row["system"]) for row in rows):
        rung = [row for row in rows if row.get("rung_name", row["system"]) == name]
        reference = next((row for row in rung if row["method"] == "reference_state"),
                         None)
        discriminating = ("already accurate (rung does not discriminate)"
                          if reference is not None
                          and reference.get("chemical_accuracy") else "not accurate")
        accurate = [row["method"] for row in rung
                    if row.get("chemical_accuracy") and row["method"] not in excluded]
        sized = [(row.get("basis_size") or row.get("operators") or 0, row["method"])
                 for row in rung
                 if row.get("chemical_accuracy") and row["evidence"] == "exact"
                 and row["method"] not in excluded]
        winner = min(sized)[1] + f" (M={min(sized)[0]})" if sized else "none"
        lines.append(f"| {name} | {discriminating} | "
                     f"{', '.join(accurate) or 'none'} | {winner} |")

    lines.extend(sampled_subspace_section(rows))

    lines.append("")
    lines.append("### Evidence labels")
    lines.append("")
    lines.append("`reference` is the exact diagonalization every error column is "
                 "measured against; `exact` is noiseless arithmetic; "
                 "`finite_sample` is a shot-based run whose growth decisions carry "
                 "an empirical-Bernstein certificate. A `finite_sample` energy "
                 "below the reference is not an improvement -- thresholding a "
                 "noisy overlap matrix has no variational guarantee (open "
                 "question Q3).")
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict], path: Path) -> None:
    # The sampled-subspace keys ride along so the CSV stays a superset of both
    # tables; they are empty on rows whose method does not sample.
    fields = list(dict.fromkeys([key for _, key in COLUMNS]
                                + [key for _, key in SAMPLED_COLUMNS]))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in ordered(rows):
            writer.writerow({key: row.get(key) for key in fields})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args(argv)
    rows = load(Path(args.record))
    if args.csv:
        write_csv(rows, Path(args.csv))
    sys.stdout.write(markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
