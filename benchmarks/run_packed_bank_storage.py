"""Phase 2M-B -- what the packed CSR/SoA row representation actually retains.

Phase 2M-A measured the baseline and found ``85.24``--``96.19`` bytes per
retained coefficient against a packed model of ``24``. That established
headroom. This record measures the reduction an implementation delivers, and
separates the part that is a property of the representation from the part that
is a property of the bank.

**The reduction is not a constant, and that is the finding.** Packed rows cost
exactly ``24`` bytes per coefficient on every bank. The word table they index
into costs its own bytes per distinct *word*, so the total is
``24 + table/reuse`` where reuse is ``T_coeff/W`` -- and the five frozen
mapping-axis banks sit at reuse ``1.4``--``2.5`` while the committed molecular
banks that actually ran out of memory sit at ``60.7``--``154.9``. Grading Phase
2M's ``3x`` gate on the convenient banks would understate it by more than a
factor of two, so the priced set is required to span reuse and the record
reports where the gate crosses.

**Equivalence is checked before any byte is reported.** Every bank is built
under both backends and compared bitwise -- ``S``, ``H``, word universe,
``T_coeff``, labels -- and the producer refuses to emit a row that fails. A
storage change that moved a matrix element would not be a storage change, and a
byte reduction quoted from a bank whose answers drifted is worth nothing.

    python benchmarks/run_packed_bank_storage.py
    python benchmarks/check_packed_bank_storage.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from clifford_qc.backends import ExactMVBackend
from clifford_qc.models import fcidump_model
from clifford_qc.reproducibility import stamp_record
from clifford_qc.subspace.elements import MatrixElementBank
from clifford_qc.subspace.fermionic_generators import (
    determinant_excitations,
    occupied_spin_orbitals,
)
from clifford_qc.subspace.generators import identity_generator
from clifford_qc.subspace.packed import LAYOUTS, PACKED_BYTES_PER_COEFFICIENT
from clifford_qc.subspace.projection import STORAGE_BACKENDS

try:  # package import in tests versus direct ``python benchmarks/...`` execution
    from benchmarks.run_bank_storage_ledger import (
        REFERENCE as LEDGER_REFERENCE,
    )
    from benchmarks.run_mapping_axis import (
        _build_model,
        _selected_generators,
        load_config as load_mapping_config,
    )
except ImportError:  # pragma: no cover - direct script execution
    from run_bank_storage_ledger import REFERENCE as LEDGER_REFERENCE
    from run_mapping_axis import (
        _build_model,
        _selected_generators,
        load_config as load_mapping_config,
    )

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = HERE / "configs" / "packed_bank_storage.json"
REFERENCE = HERE / "reference_results" / "packed_bank_storage.json"
SCHEMA = "clifford_qc.packed_bank_storage.v1"
CONFIG_SCHEMA = "clifford_qc.packed_bank_storage_config.v1"

# Phase 2M's first go/no-go clause, and the only one an implementation-side
# measurement can reach. Declared here so a record that quietly relaxed it fails
# at the boundary rather than reporting a pass against a softer number.
GO_NO_GO_THRESHOLD = 3.0
GRADED_CLAUSE = "packed_storage_reduction_at_least_3x"

# Structural fields that must be identical under both backends before any byte
# comparison is meaningful. Packing changes where coefficients live, not what
# the bank computes, so every one of these is an equality rather than a
# tolerance.
EQUIVALENCE_FIELDS = (
    "word_universe",
    "resident_word_universe",
    "coefficient_occurrences",
    "cached_operator_bytes",
    "pairs_built",
    "resident_operator_rows",
    "max_overlap_element_support",
    "max_hamiltonian_element_support",
)


def load_config(path: Path | None = None) -> dict:
    """Read the frozen configuration and check its own declarations."""
    config = json.loads((CONFIG if path is None else path).read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"unexpected config schema {config.get('schema')!r}")
    if config.get("packed_bytes_per_coefficient") != PACKED_BYTES_PER_COEFFICIENT:
        raise ValueError(
            f"the config declares {config.get('packed_bytes_per_coefficient')!r} "
            f"packed bytes per coefficient against the store's "
            f"{PACKED_BYTES_PER_COEFFICIENT}")
    if tuple(config.get("layouts", ())) != LAYOUTS:
        raise ValueError(
            f"the config declares layouts {config.get('layouts')!r} but the store "
            f"implements {list(LAYOUTS)}")
    if config["equivalence_gate"].get("required") is not True:
        raise ValueError(
            "the equivalence gate is not asserted; a byte reduction measured "
            "without it says nothing about a bank whose answers may have moved")
    if config["go_no_go"].get("threshold") != GO_NO_GO_THRESHOLD:
        raise ValueError(
            f"the config grades against {config['go_no_go'].get('threshold')!r}, "
            f"not Phase 2M's declared {GO_NO_GO_THRESHOLD}")
    if config["go_no_go"].get("graded_clause") != GRADED_CLAUSE:
        raise ValueError("the config grades a clause this producer does not measure")
    if not config.get("claim_boundary"):
        raise ValueError("config declares no claim boundary")
    return config


def _extended_bank(spec: dict):
    """Generators for a declared pool prefix, in the builder's own order."""
    model = fcidump_model(ROOT / spec["source"], name=spec["key"])
    pool = determinant_excitations(
        model.n, occupied_spin_orbitals(model),
        max_rank=int(spec["max_rank"]), conserve_sz=bool(spec["conserve_sz"]))
    size = int(spec["basis_size"])
    if size > len(pool) + 1:
        raise ValueError(
            f"{spec['key']}: basis size {size} exceeds the {len(pool) + 1} the "
            "declared pool can supply")
    return model, [identity_generator(model.n), *pool[:size - 1]]


def _build(model, generators, **kwargs) -> MatrixElementBank:
    bank = MatrixElementBank(ExactMVBackend().state(model.reference, ()),
                             model.hamiltonian, generators, **kwargs)
    bank.matrices()
    return bank


def _equivalence(reference: MatrixElementBank, packed: MatrixElementBank) -> dict:
    """Compare the two backends before either one's bytes are believed."""
    S0, H0 = reference.matrices()
    S1, H1 = packed.matrices()
    left, right = reference.resources(), packed.resources()
    mismatched = [field for field in EQUIVALENCE_FIELDS if left[field] != right[field]]
    return {
        "overlap_matrix_bitwise_equal": bool(np.array_equal(S0, S1)),
        "hamiltonian_matrix_bitwise_equal": bool(np.array_equal(H0, H1)),
        "basis_labels_equal": list(reference.labels) == list(packed.labels),
        "structural_fields_compared": list(EQUIVALENCE_FIELDS),
        "mismatched_fields": mismatched,
    }


def _holds(equivalence: dict) -> bool:
    return (equivalence["overlap_matrix_bitwise_equal"]
            and equivalence["hamiltonian_matrix_bitwise_equal"]
            and equivalence["basis_labels_equal"]
            and not equivalence["mismatched_fields"])


def _price(key: str, kind: str, model, generators) -> dict:
    """One bank under every backend, with equivalence checked first."""
    reference = _build(model, generators)
    packed = {layout: _build(model, generators, storage="packed", layout=layout)
              for layout in LAYOUTS}
    equivalence = {layout: _equivalence(reference, bank)
                   for layout, bank in packed.items()}
    for layout, verdict in equivalence.items():
        if not _holds(verdict):
            raise ValueError(
                f"{key}/{kind}/{layout}: the packed backend does not reproduce the "
                f"object backend ({verdict}); no byte reduction may be reported "
                "for a bank whose answers moved")

    resources = reference.resources()
    occurrences = resources["coefficient_occurrences"]
    universe = resources["resident_word_universe"]
    object_bytes = reference.measured_storage_bytes()
    row = {
        "bank": key,
        "kind": kind,
        "n_qubits": model.n,
        "basis_size": resources["basis_size"],
        "pairs_built": resources["pairs_built"],
        "coefficient_occurrences": occurrences,
        "resident_word_universe": universe,
        "coefficient_reuse": resources["coefficient_reuse"],
        "object_bytes": object_bytes["measured_operator_bytes"],
        "object_bytes_per_coefficient":
            object_bytes["measured_bytes_per_coefficient"],
        "equivalence": equivalence["interleaved"],
        "layouts": {},
    }
    for layout, bank in packed.items():
        measured = bank.measured_storage_bytes()
        table_bytes = measured["word_table_bytes"]
        rows_bytes = measured["measured_container_bytes"] - table_bytes
        # Three terms, not two. The packed buffers and the word table are pure
        # numpy; the third is the distinct word-code integers, which this
        # backend does not reference but which stay resident through the bank's
        # own universe under either backend -- so both sides charge them once
        # and the comparison stays like-for-like.
        row["layouts"][layout] = {
            "total_bytes": measured["measured_operator_bytes"],
            "row_bytes": rows_bytes,
            "word_table_bytes": table_bytes,
            "shared_word_code_bytes": measured["measured_boxed_bytes"],
            "reserved_bytes": measured["reserved_operator_bytes"],
            "row_bytes_per_coefficient": rows_bytes / occurrences,
            "word_table_bytes_per_word": table_bytes / universe,
            "total_bytes_per_coefficient":
                measured["measured_bytes_per_coefficient"],
            "reduction": object_bytes["measured_operator_bytes"]
            / measured["measured_operator_bytes"],
            "row_only_reduction":
                object_bytes["measured_operator_bytes"] / rows_bytes,
            "clears_threshold": (object_bytes["measured_operator_bytes"]
                                 / measured["measured_operator_bytes"]
                                 ) >= GO_NO_GO_THRESHOLD,
        }
    return row


def _banks(config: dict) -> list[dict]:
    mapping = load_mapping_config()
    by_key = {spec["key"]: spec for spec in mapping["systems"]}
    rows = []
    for key in config["banks"]["frozen_mapping_axis"]["systems"]:
        spec = by_key.get(key)
        if spec is None:
            raise ValueError(
                f"{key!r} is declared here but absent from mapping_axis.json; this "
                "record prices the mapping instances and may not invent one")
        model, _ = _build_model(spec)
        _, selected = _selected_generators(model, spec)
        rows.append(_price(key, "frozen_mapping_axis", model, selected))
    for spec in config["banks"]["extended"]:
        model, generators = _extended_bank(spec)
        sz = "sz" if spec["conserve_sz"] else "full"
        rows.append(_price(f"{spec['key']}_M{spec['basis_size']}_{sz}",
                           "extended_pool", model, generators))
    return rows


def _layout_comparison(rows: list[dict]) -> dict:
    """Interleaved against structure-of-arrays, on identical banks."""
    disagreeing = [row["bank"] for row in rows
                   if row["layouts"]["interleaved"]["total_bytes"]
                   != row["layouts"]["soa"]["total_bytes"]]
    return {
        "layouts": list(LAYOUTS),
        "byte_identical_on_every_bank": not disagreeing,
        "banks_where_bytes_differ": disagreeing,
        "statement": (
            "Interleaved keeps one complex128 array and structure-of-arrays keeps "
            "separate float64 real and imaginary arrays; both are sixteen bytes per "
            "coefficient, so the choice between them is one of access cost and not "
            "of size. The byte equality is checked here rather than assumed, "
            "because a layout that silently changed the footprint would make every "
            "reduction below ambiguous about which representation produced it."),
    }


def _reduction_model(rows: list[dict]) -> dict:
    """Total bytes per coefficient against reuse, and where 3x is reached.

    ``24 + table/reuse`` is not fitted: the row term is the packed model,
    measured to hold exactly on every bank, and the table term is the measured
    per-word cost divided by that bank's own reuse. What the record reports is
    the *measured* ladder and the threshold read off it, so the shape is a
    description of the measurement rather than a curve laid over it.
    """
    ladder = sorted(
        ({"bank": row["bank"],
          "coefficient_reuse": row["coefficient_reuse"],
          "reduction": row["layouts"]["interleaved"]["reduction"],
          "clears_threshold": row["layouts"]["interleaved"]["clears_threshold"]}
         for row in rows),
        key=lambda entry: entry["coefficient_reuse"])
    clearing = [entry for entry in ladder if entry["clears_threshold"]]
    failing = [entry for entry in ladder if not entry["clears_threshold"]]
    reuses = [entry["coefficient_reuse"] for entry in ladder]
    return {
        "ladder": ladder,
        "measured_reuse_range": [min(reuses), max(reuses)],
        "reuse_span_factor": max(reuses) / min(reuses),
        "lowest_reuse_clearing_threshold": (
            min(entry["coefficient_reuse"] for entry in clearing) if clearing else None),
        "highest_reuse_failing_threshold": (
            max(entry["coefficient_reuse"] for entry in failing) if failing else None),
        "monotone_in_reuse": all(
            a["reduction"] <= b["reduction"] for a, b in zip(ladder, ladder[1:])),
        # A bare boolean would overstate a dip of a few thousandths. The
        # reduction is not a function of reuse alone -- the object backend's own
        # bytes per coefficient vary across banks too -- so exact monotonicity
        # was never expected, and the size of the violation is what says whether
        # the ladder is still readable as one.
        "largest_monotonicity_violation": max(
            (a["reduction"] - b["reduction"]
             for a, b in zip(ladder, ladder[1:]) if a["reduction"] > b["reduction"]),
            default=0.0),
        # Below some reuse the shared word table costs more than the dict slots
        # packing removed, and the representation is a net loss. Naming those
        # banks is the point: a record that reported only the wins would be
        # describing a different phase.
        "banks_where_packing_costs_more": [
            entry["bank"] for entry in ladder if entry["reduction"] < 1.0],
        "word_table_bytes_per_word_range": [
            min(row["layouts"]["interleaved"]["word_table_bytes_per_word"]
                for row in rows),
            max(row["layouts"]["interleaved"]["word_table_bytes_per_word"]
                for row in rows)],
        "statement": (
            "Packed rows cost exactly the modelled bytes per coefficient on every "
            "bank; the shared word table costs its own bytes per distinct word. So "
            "the total per coefficient is the row cost plus the table cost divided "
            "by coefficient reuse, and the reduction rises with reuse rather than "
            "being a property of the representation alone. At the bottom of the "
            "measured ladder that sum exceeds what it replaces and packing is a "
            "net loss, which is the same mechanism read from the other end rather "
            "than a separate effect."),
    }


def _committed_bank_reuse() -> dict:
    """Where the banks that actually failed sit on that ladder.

    Read from Phase 2M-A's committed record rather than rebuilt. Those runs
    reached 10.75 GiB and two of them died; re-running them to measure a
    storage ratio is neither affordable nor necessary, because reuse is already
    recorded there and reuse is what the threshold is read against.
    """
    ledger = json.loads(LEDGER_REFERENCE.read_text(encoding="utf-8"))
    rows = []
    for row in ledger["committed_records"]:
        universe = row["word_universe"]
        rows.append({
            "record": row["record"],
            "coefficient_occurrences": row["coefficient_occurrences"],
            "selected_word_universe": universe,
            "coefficient_occurrences_per_selected_element_word":
                row["coefficient_occurrences_per_selected_element_word"],
        })
    ratios = [row["coefficient_occurrences_per_selected_element_word"] for row in rows]
    return {
        "source_record": LEDGER_REFERENCE.name,
        "rows": rows,
        "ratio_range": [min(ratios), max(ratios)],
        "caveat": (
            "These are resident coefficients per *selected-subspace* word, not "
            "true coefficient reuse: the committed records preserved the selected "
            "subspace's W and not the word union of the rejected rows still in "
            "their caches, as Phase 2M-A's own reuse_recovery_boundary records. "
            "The resident universe is the larger of the two, so true reuse is no "
            "greater than the ratio quoted here. That direction matters: it means "
            "these numbers may overstate where those banks sit on the ladder, and "
            "the comparison below is made with that stated rather than assumed "
            "away."),
    }


def _verdict(config: dict, model: dict, committed: dict) -> dict:
    """Grade Phase 2M's first clause, and only that one."""
    lowest_clearing = model["lowest_reuse_clearing_threshold"]
    highest_failing = model["highest_reuse_failing_threshold"]
    committed_low = min(committed["ratio_range"])
    if lowest_clearing is None:
        outcome = "not_reached_on_any_measured_bank"
    elif highest_failing is not None and highest_failing > lowest_clearing:
        outcome = "indeterminate_threshold_not_separated"
    elif committed_low >= lowest_clearing:
        outcome = "reached_above_a_measured_reuse_threshold_committed_banks_exceed_it"
    else:
        outcome = "reached_only_above_the_committed_banks_reuse"
    return {
        "graded_clause": GRADED_CLAUSE,
        "threshold": GO_NO_GO_THRESHOLD,
        "outcome": outcome,
        "lowest_reuse_clearing_threshold": lowest_clearing,
        "highest_reuse_failing_threshold": highest_failing,
        "committed_bank_ratio_floor": committed_low,
        "ungraded_clauses": config["go_no_go"]["ungraded_clauses"],
        "what_this_does_not_establish": (
            "Not that Phase 2M passes. Its go/no-go has three clauses and this "
            "record reaches one: no streaming policy exists to bound live operator "
            "rows, and no end-to-end run has completed the stretched-H2O M=31 "
            "configuration under a memory ceiling. A storage ratio measured on "
            "declared banks is also not a process-level reduction: it prices the "
            "retained rows, not the transient allocation a build passes through, "
            "and 2M-D's feasibility test stays the test precisely because no ratio "
            "can stand in for it."),
    }


def build_record() -> dict:
    config = load_config()
    started = time.perf_counter()
    rows = _banks(config)
    model = _reduction_model(rows)
    required = float(config["banks"]["reuse_span_requirement"])
    if model["reuse_span_factor"] < required:
        raise ValueError(
            f"the priced banks span {model['reuse_span_factor']:.1f}x in "
            f"coefficient reuse, under the declared {required}x. The reduction is "
            "a function of reuse, so a set that does not span it cannot locate "
            "the threshold this record exists to report")
    committed = _committed_bank_reuse()
    record = {
        "schema": SCHEMA,
        "phase": config["phase"],
        "evidence_tier": "structural",
        "estimand": config["estimand"],
        "storage_backends": list(STORAGE_BACKENDS),
        "default_storage_backend": "object",
        "default_backend_statement": (
            "The object backend stays the package default. Phase 2M-A's committed "
            "record prices it and Phase 2M-D's equivalence matrix has to rebuild "
            "it, so a phase that changed the default before its own gate was "
            "graded would invalidate the baseline it is graded against."),
        "packed_bytes_per_coefficient": PACKED_BYTES_PER_COEFFICIENT,
        "packed_model_statement": config["packed_model_statement"],
        "equivalence_gate": config["equivalence_gate"],
        "interpreter_dependence": (
            "Byte counts of live Python objects are a property of the CPython "
            "build that measured them -- dict capacity, integer boxing and object "
            "headers are implementation details, not portable constants. The "
            "record declares its interpreter in provenance and the checker "
            "compares those fields under a declared relative tolerance, while "
            "every count, pair and packed byte stays exact."),
        "banks": rows,
        "layout_comparison": _layout_comparison(rows),
        "reduction_model": model,
        "committed_bank_reuse": committed,
        "go_no_go": _verdict(config, model, committed),
        "producer_seconds": time.perf_counter() - started,
        "claim_boundary": config["claim_boundary"],
    }
    return stamp_record(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REFERENCE)
    args = parser.parse_args()
    record = build_record()
    args.out.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{'bank':26s} {'n':>2s} {'M':>3s} {'T_coeff':>9s} {'reuse':>7s} "
          f"{'obj B/c':>8s} {'pack B/c':>8s} {'reduction':>9s}")
    for row in record["banks"]:
        packed = row["layouts"]["interleaved"]
        print(f"{row['bank']:26s} {row['n_qubits']:2d} {row['basis_size']:3d} "
              f"{row['coefficient_occurrences']:9d} {row['coefficient_reuse']:7.2f} "
              f"{row['object_bytes_per_coefficient']:8.2f} "
              f"{packed['total_bytes_per_coefficient']:8.2f} "
              f"{packed['reduction']:8.2f}x"
              f"{'  *' if packed['clears_threshold'] else ''}")
    model = record["reduction_model"]
    print(f"\n  * clears the {GO_NO_GO_THRESHOLD}x threshold; reuse spanned "
          f"{model['measured_reuse_range'][0]:.2f}-{model['measured_reuse_range'][1]:.2f} "
          f"({model['reuse_span_factor']:.1f}x), monotone: {model['monotone_in_reuse']}")
    print(f"  lowest reuse clearing: {model['lowest_reuse_clearing_threshold']}")
    print(f"  highest reuse failing: {model['highest_reuse_failing_threshold']}")
    if model["banks_where_packing_costs_more"]:
        print(f"  packing is a NET LOSS on: "
              f"{', '.join(model['banks_where_packing_costs_more'])}")
    committed = record["committed_bank_reuse"]
    print(f"  committed banks sit at {committed['ratio_range'][0]:.1f}"
          f"-{committed['ratio_range'][1]:.1f}")
    print(f"\n{record['go_no_go']['graded_clause']}: {record['go_no_go']['outcome']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
