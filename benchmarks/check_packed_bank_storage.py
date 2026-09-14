"""Recompute and gate the Phase 2M-B packed-storage record.

Two jobs, as the project's other checkers do. The record is rebuilt from
scratch and compared field by field, and the contracts it must satisfy are
re-derived from its own contents rather than trusted:

* the equivalence gate holds on every priced bank, in both layouts: ``S`` and
  ``H`` bitwise equal, identical labels, and no mismatched structural field. A
  byte reduction quoted from a bank whose answers moved is worth nothing, so
  this is checked before any byte comparison is believed;
* every byte figure closes against its own parts -- packed rows, word table and
  the shared word-code integers sum to the total, per-coefficient and per-word
  rates are the quotients they claim to be, reserved bytes never understate live
  bytes, and the packed rows land on exactly the modelled bytes per coefficient;
* the shared word-code integers are charged on the packed side too. The numpy
  table does not reference them, but they stay resident through the bank's own
  word universe under either backend and the object side charges them, so
  omitting them here would credit packing with an allocation it never removed;
* the two layouts hold identical bytes on every bank, which is what makes the
  reduction unambiguous about which representation produced it;
* the priced set spans the declared factor in coefficient reuse, and the
  threshold it locates is separated -- the lowest reuse that clears the gate is
  above the highest that fails it, or the verdict says it is indeterminate;
* banks where packing costs more than it saves are named rather than dropped;
* the verdict follows mechanically from the measured ladder and the committed
  banks' reuse, so no outcome is writable by hand;
* exactly one go/no-go clause is graded, the other two are named as ungraded,
  and no cost-record field appears anywhere -- this tier prices nothing.

Measured byte counts are interpreter-dependent, so they are compared under a
declared relative tolerance rather than exactly; see
``INTERPRETER_DEPENDENT_FIELDS``. Every count, pair and packed byte stays exact.

    python benchmarks/check_packed_bank_storage.py
"""

from __future__ import annotations

import json
import math
import sys

from clifford_qc.reproducibility import compare_json_records, guarded_contract_problems
from clifford_qc.subspace.packed import LAYOUTS, PACKED_BYTES_PER_COEFFICIENT

try:  # package import in tests versus direct script execution
    from benchmarks.run_packed_bank_storage import (
        GO_NO_GO_THRESHOLD,
        GRADED_CLAUSE,
        REFERENCE,
        SCHEMA,
        build_record,
        load_config,
    )
except ImportError:  # pragma: no cover - direct script execution
    from run_packed_bank_storage import (
        GO_NO_GO_THRESHOLD,
        GRADED_CLAUSE,
        REFERENCE,
        SCHEMA,
        build_record,
        load_config,
    )

# Fields that would make this a cost record. The tier is structural; if one of
# these ever appears the record has quietly changed what it claims. Word
# universes and byte counts are structural quantities and are deliberately not
# on it: this record is *about* bytes.
COST_RECORD_KEYS = {
    "C_time",
    "cost_bracket",
    "device_costs",
    "k_star",
    "qr3_accuracy_matched",
    "shot_to_target",
}

# Byte counts taken from live Python objects, and everything derived from them.
# Dict capacity, integer boxing and object headers are CPython implementation
# details rather than portable constants, so these compare under their own
# relative tolerance while every count and pair below stays exact. The
# environment contract pins the interpreter's minor version but not its patch
# release, which is the gap this covers.
INTERPRETER_DEPENDENT_FIELDS = frozenset({
    "banks_where_packing_costs_more",
    "committed_bank_ratio_floor",
    "highest_reuse_failing_threshold",
    "ladder",
    "largest_monotonicity_violation",
    "lowest_reuse_clearing_threshold",
    "measured_reuse_range",
    "monotone_in_reuse",
    "object_bytes",
    "object_bytes_per_coefficient",
    "ratio_range",
    "reduction",
    "reserved_bytes",
    "reuse_span_factor",
    "row_only_reduction",
    "shared_word_code_bytes",
    "total_bytes",
    "total_bytes_per_coefficient",
    "word_table_bytes",
    "word_table_bytes_per_word",
    "word_table_bytes_per_word_range",
})

# Metadata that is not a measurement of anything this record claims, and is
# therefore compared under no tolerance at all. ``producer_seconds`` is wall
# clock: it belongs in the record because it says what the gate costs to run,
# but it is a property of the machine and its load rather than of the packed
# representation, so gating on it would make the check resolve on which runner
# drew it -- the failure mode ``CROSS_MACHINE_ATOL`` exists to document. It was
# briefly in the interpreter-tolerance set below and failed the gate on a 2.3%
# rerun difference, which is the right outcome for a byte count and the wrong
# question to ask of a stopwatch.
VOLATILE_FIELDS = frozenset({"provenance", "producer_seconds"})

# What a CPython patch release may move a live-object byte count by before the
# gate calls it a regression. Same 2% Phase 2M-A declared, for the same reason
# and with the same standing: a change in the retained representation moves
# these by tens of percent, while object-layout drift is far below it.
INTERPRETER_RTOL = 0.02


def _cost_key_paths(node, path: str = "$") -> list[str]:
    if isinstance(node, dict):
        found = []
        for key, value in node.items():
            if key in COST_RECORD_KEYS:
                found.append(f"{path}.{key}")
            found += _cost_key_paths(value, f"{path}.{key}")
        return found
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in _cost_key_paths(v, f"{path}[{i}]")]
    return []


def _leaves(node, path: str) -> dict[str, float]:
    if isinstance(node, bool) or node is None:
        return {}
    if isinstance(node, (int, float)):
        return {path: float(node)}
    if isinstance(node, dict):
        return {k: v for key, value in node.items()
                for k, v in _leaves(value, f"{path}.{key}").items()}
    if isinstance(node, list):
        return {k: v for i, value in enumerate(node)
                for k, v in _leaves(value, f"{path}[{i}]").items()}
    return {}


def _numbers(node, path: str = "$") -> dict[str, float]:
    """Every numeric leaf under an interpreter-dependent key, by path."""
    out: dict[str, float] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}"
            if key in VOLATILE_FIELDS:
                continue
            if key in INTERPRETER_DEPENDENT_FIELDS:
                out.update(_leaves(value, here))
            else:
                out.update(_numbers(value, here))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            out.update(_numbers(value, f"{path}[{index}]"))
    return out


def interpreter_field_problems(expected: dict, actual: dict) -> list[str]:
    """Compare the measured byte fields under their own declared tolerance."""
    left, right = _numbers(expected), _numbers(actual)
    problems = []
    for path in sorted(set(left) - set(right)):
        problems.append(f"{path}: present in the committed record, absent on rebuild")
    for path in sorted(set(right) - set(left)):
        problems.append(f"{path}: produced by the rebuild, absent from the record")
    for path in sorted(set(left) & set(right)):
        if not math.isclose(left[path], right[path],
                            rel_tol=INTERPRETER_RTOL, abs_tol=0.0):
            problems.append(
                f"{path}: committed {left[path]:.17g}, rebuilt {right[path]:.17g}, "
                f"outside the declared {INTERPRETER_RTOL:.0%} interpreter "
                "tolerance. These are live objects' sizes, so a difference this "
                "large is a change in the retained representation rather than a "
                "patch-release layout drift")
    return problems


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


@guarded_contract_problems
def equivalence_problems(record: dict) -> list[str]:
    """The gate that has to hold before any byte figure means anything."""
    problems: list[str] = []
    for row in record["banks"]:
        verdict = row["equivalence"]
        where = f"{row['bank']}/{row['kind']}"
        if not verdict["overlap_matrix_bitwise_equal"]:
            problems.append(f"{where}: the overlap matrix is not bitwise equal")
        if not verdict["hamiltonian_matrix_bitwise_equal"]:
            problems.append(f"{where}: the Hamiltonian matrix is not bitwise equal")
        if not verdict["basis_labels_equal"]:
            problems.append(f"{where}: the two backends selected different labels")
        if verdict["mismatched_fields"]:
            problems.append(
                f"{where}: structural fields disagree between backends "
                f"({verdict['mismatched_fields']}); packing changes where "
                "coefficients live, not what the bank computes")
        if not verdict["structural_fields_compared"]:
            problems.append(
                f"{where}: records an equivalence verdict over no fields at all")
    return problems


@guarded_contract_problems
def byte_problems(record: dict) -> list[str]:
    """Each bank's byte arithmetic, against its own parts."""
    problems: list[str] = []
    for row in record["banks"]:
        occurrences = row["coefficient_occurrences"]
        universe = row["resident_word_universe"]
        if not _close(row["coefficient_reuse"], occurrences / universe):
            problems.append(f"{row['bank']}: coefficient reuse is not T_coeff/W")
        if not _close(row["object_bytes_per_coefficient"],
                      row["object_bytes"] / occurrences):
            problems.append(f"{row['bank']}: object bytes per coefficient is not the quotient")
        for layout, priced in row["layouts"].items():
            where = f"{row['bank']}/{layout}"
            parts = (priced["row_bytes"] + priced["word_table_bytes"]
                     + priced["shared_word_code_bytes"])
            if parts != priced["total_bytes"]:
                problems.append(
                    f"{where}: packed rows, word table and shared word-code "
                    f"integers sum to {parts}, not the reported "
                    f"{priced['total_bytes']}")
            if priced["shared_word_code_bytes"] <= 0:
                problems.append(
                    f"{where}: no shared word-code integers are charged. They are "
                    "resident under either backend and the object side charges "
                    "them, so omitting them here credits packing with an "
                    "allocation it never removed")
            if priced["row_bytes"] != PACKED_BYTES_PER_COEFFICIENT * occurrences:
                problems.append(
                    f"{where}: packed rows are {priced['row_bytes']} bytes, not "
                    f"{PACKED_BYTES_PER_COEFFICIENT} * {occurrences}. The modelled "
                    "per-coefficient cost is what this phase claims to reach")
            if not _close(priced["row_bytes_per_coefficient"],
                          PACKED_BYTES_PER_COEFFICIENT):
                problems.append(f"{where}: row bytes per coefficient is not the model")
            if not _close(priced["word_table_bytes_per_word"],
                          priced["word_table_bytes"] / universe):
                problems.append(f"{where}: word-table bytes per word is not the quotient")
            if not _close(priced["total_bytes_per_coefficient"],
                          priced["total_bytes"] / occurrences):
                problems.append(f"{where}: total bytes per coefficient is not the quotient")
            if not _close(priced["reduction"], row["object_bytes"] / priced["total_bytes"]):
                problems.append(f"{where}: the reduction is not the byte quotient")
            if not _close(priced["row_only_reduction"],
                          row["object_bytes"] / priced["row_bytes"]):
                problems.append(f"{where}: the row-only reduction is not the quotient")
            if priced["reserved_bytes"] < priced["total_bytes"]:
                problems.append(
                    f"{where}: reserved bytes understate live bytes, so the growth "
                    "slack is unaccounted rather than merely small")
            if priced["clears_threshold"] != (priced["reduction"] >= GO_NO_GO_THRESHOLD):
                problems.append(
                    f"{where}: records clears_threshold="
                    f"{priced['clears_threshold']} against a measured "
                    f"{priced['reduction']:.4f}x; the flag follows from the "
                    "threshold and is not writable by hand")
    return problems


@guarded_contract_problems
def layout_problems(record: dict) -> list[str]:
    """Both layouts must hold identical bytes, and the record must check it."""
    problems: list[str] = []
    comparison = record["layout_comparison"]
    if tuple(comparison["layouts"]) != LAYOUTS:
        problems.append(
            f"the record compares {comparison['layouts']}, not the store's "
            f"{list(LAYOUTS)}")
    disagreeing = [row["bank"] for row in record["banks"]
                   if row["layouts"]["interleaved"]["total_bytes"]
                   != row["layouts"]["soa"]["total_bytes"]]
    if comparison["banks_where_bytes_differ"] != disagreeing:
        problems.append(
            "the layout comparison's disagreement list is not what the priced "
            "banks say")
    if comparison["byte_identical_on_every_bank"] != (not disagreeing):
        problems.append("the layout byte-equality verdict is not the measured one")
    for row in record["banks"]:
        missing = set(LAYOUTS) - set(row["layouts"])
        if missing:
            problems.append(f"{row['bank']}: not priced under {sorted(missing)}")
    return problems


@guarded_contract_problems
def model_problems(record: dict) -> list[str]:
    """The reduction ladder, re-derived from the banks it summarizes."""
    problems: list[str] = []
    model = record["reduction_model"]
    config = load_config()
    expected = sorted(
        ({"bank": row["bank"],
          "coefficient_reuse": row["coefficient_reuse"],
          "reduction": row["layouts"]["interleaved"]["reduction"],
          "clears_threshold": row["layouts"]["interleaved"]["clears_threshold"]}
         for row in record["banks"]),
        key=lambda entry: entry["coefficient_reuse"])
    if [entry["bank"] for entry in model["ladder"]] != [e["bank"] for e in expected]:
        problems.append("the ladder is not the priced banks ordered by reuse")
        return problems
    for recorded, derived in zip(model["ladder"], expected):
        if not _close(recorded["reduction"], derived["reduction"]):
            problems.append(f"{derived['bank']}: ladder reduction disagrees with its row")
        if recorded["clears_threshold"] != derived["clears_threshold"]:
            problems.append(f"{derived['bank']}: ladder threshold flag disagrees")
    reuses = [entry["coefficient_reuse"] for entry in expected]
    if not (_close(model["measured_reuse_range"][0], min(reuses))
            and _close(model["measured_reuse_range"][1], max(reuses))):
        problems.append("the measured reuse range is not the range over the banks")
    if not _close(model["reuse_span_factor"], max(reuses) / min(reuses)):
        problems.append("the reuse span factor is not the quotient of the range")
    required = float(config["banks"]["reuse_span_requirement"])
    if model["reuse_span_factor"] < required:
        problems.append(
            f"the priced banks span {model['reuse_span_factor']:.1f}x in reuse, "
            f"under the declared {required}x; a set that does not span reuse "
            "cannot locate the threshold this record reports")
    clearing = [e["coefficient_reuse"] for e in expected if e["clears_threshold"]]
    failing = [e["coefficient_reuse"] for e in expected if not e["clears_threshold"]]
    for field, values, reduce_to in (
            ("lowest_reuse_clearing_threshold", clearing, min),
            ("highest_reuse_failing_threshold", failing, max)):
        wanted = reduce_to(values) if values else None
        recorded = model[field]
        if wanted is None:
            if recorded is not None:
                problems.append(f"{field} is recorded against an empty set")
        elif recorded is None or not _close(recorded, wanted):
            problems.append(f"{field} is not what the ladder gives")
    losses = [e["bank"] for e in expected if e["reduction"] < 1.0]
    if model["banks_where_packing_costs_more"] != losses:
        problems.append(
            "the net-loss list disagrees with the measured reductions; a bank "
            "where packing costs more may not be dropped from it")
    return problems


@guarded_contract_problems
def verdict_problems(record: dict) -> list[str]:
    """The graded clause follows from the ladder, and only one clause is graded."""
    problems: list[str] = []
    verdict = record["go_no_go"]
    model = record["reduction_model"]
    if verdict["graded_clause"] != GRADED_CLAUSE:
        problems.append(f"the record grades {verdict['graded_clause']!r}")
    if verdict["threshold"] != GO_NO_GO_THRESHOLD:
        problems.append(
            f"the record grades against {verdict['threshold']!r}, not Phase 2M's "
            f"declared {GO_NO_GO_THRESHOLD}")
    lowest = model["lowest_reuse_clearing_threshold"]
    highest = model["highest_reuse_failing_threshold"]
    floor = min(record["committed_bank_reuse"]["ratio_range"])
    expected = ("not_reached_on_any_measured_bank" if lowest is None else
                "reached_on_measured_banks_historical_banks_ungraded")
    if verdict.get("historical_banks_status") != "ungraded_resident_word_universe_unknown":
        problems.append("historical upper bounds cannot grade resident reuse")
    if verdict["outcome"] != expected:
        problems.append(
            f"the record records outcome {verdict['outcome']!r}, but the measured "
            f"ladder gives {expected!r}. The verdict follows from the threshold "
            "on directly measured banks; historical upper bounds cannot establish it")
    if not _close(verdict["committed_bank_ratio_floor"], floor):
        problems.append("the committed ratio floor is not the minimum it quotes")
    ungraded = verdict.get("ungraded_clauses") or {}
    if len(ungraded) != 2:
        problems.append(
            f"Phase 2M's go/no-go has three clauses and this record grades one, so "
            f"exactly two must be named ungraded; {len(ungraded)} are")
    for clause, reason in ungraded.items():
        if not reason:
            problems.append(f"ungraded clause {clause} carries no reason")
    if not verdict.get("what_this_does_not_establish"):
        problems.append(
            "the verdict records no boundary; a graded clause is the quantity here "
            "most likely to be re-read as Phase 2M passing")
    return problems


@guarded_contract_problems
def committed_reuse_problems(record: dict) -> list[str]:
    """The committed banks are quoted, not rebuilt, and say so."""
    problems: list[str] = []
    committed = record["committed_bank_reuse"]
    if not committed.get("rows"):
        problems.append("no committed bank is quoted, so the threshold is unanchored")
        return problems
    ratios = [row["coefficient_occurrences_per_selected_element_word"]
              for row in committed["rows"]]
    if not (_close(committed["ratio_range"][0], min(ratios))
            and _close(committed["ratio_range"][1], max(ratios))):
        problems.append("the committed ratio range is not the range over its rows")
    if not committed.get("caveat"):
        problems.append(
            "the committed rows are quoted with no caveat, but they carry "
            "selected-subspace W rather than resident W and so are an upper "
            "bound on reuse rather than reuse")
    if not committed.get("source_record"):
        problems.append("the committed rows name no source record")
    return problems


@guarded_contract_problems
def contract_problems(record: dict) -> list[str]:
    problems: list[str] = []
    if record.get("schema") != SCHEMA:
        problems.append(f"unexpected schema {record.get('schema')!r}")
    if record.get("evidence_tier") != "structural":
        problems.append(
            f"evidence tier is {record.get('evidence_tier')!r}; this record samples "
            "nothing and must not claim a stronger tier")
    if record.get("default_storage_backend") != "object":
        problems.append(
            "the record does not record the object backend as the package "
            "default; Phase 2M-A's committed baseline prices it, and a phase that "
            "changed the default before its gate was graded would invalidate the "
            "baseline it is graded against")
    if record.get("packed_bytes_per_coefficient") != PACKED_BYTES_PER_COEFFICIENT:
        problems.append("the record's packed model disagrees with the store's")
    if not record.get("claim_boundary"):
        problems.append("no claim boundary recorded")
    if not record.get("interpreter_dependence"):
        problems.append(
            "the record publishes measured byte counts without saying they are a "
            "property of the interpreter that measured them")
    for path in _cost_key_paths(record):
        problems.append(
            f"cost-record field at {path}: this tier prices nothing, so its "
            "presence means the record has changed what it claims")
    problems += equivalence_problems(record)
    problems += byte_problems(record)
    problems += layout_problems(record)
    problems += model_problems(record)
    problems += committed_reuse_problems(record)
    problems += verdict_problems(record)
    return problems


def main() -> int:
    load_config()  # the config's own declaration gates
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    actual = build_record()
    problems = compare_json_records(
        expected, actual, atol=1e-12, rtol=1e-12,
        ignored_keys=VOLATILE_FIELDS | INTERPRETER_DEPENDENT_FIELDS)
    problems += interpreter_field_problems(expected, actual)
    problems += contract_problems(expected)
    problems += [f"rebuilt record: {p}" for p in contract_problems(actual)]
    for problem in problems:
        print(f"  {problem}")
    print("packed bank storage:", "FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
