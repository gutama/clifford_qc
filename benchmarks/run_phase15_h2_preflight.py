"""Run the preregistered Phase 15 H² support/cost preflight (Q16), once.

Counts the second-moment rows ``K_ij = A_i† (H² A_j)`` over the retained block
of the five frozen mapping-axis banks, exactly as
``benchmarks/configs/phase15_h2_preflight.json`` froze them. See
``benchmarks/PHASE15_PREFLIGHT_PREREGISTRATION.md``.

The producer refuses before it forms ``H²`` unless all of the following hold:

* the preregistration gate passes, both its static clauses and its
  recomputation of every frozen number from the committed inputs;
* the run is the declared one: no bank subset is written to the committed
  record path;
* no record already exists there, because the preflight runs once;
* the working tree is clean, so the record's provenance names the code that
  produced it;
* the environment is one the committed records declare. The stamp's guard is
  tripped before counting rather than after it.

For each bank it rebuilds the ``(S, H)`` bank and forms ``H2 = H * H`` once.
Then, one column at a time, it forms ``H2 * A_j`` and each row
``A_i.dagger() * (H2 * A_j)`` for ``i <= j``, in the declared product order.
Each row is counted and discarded, so memory holds one column and the word
sets, never the second-moment bank itself. The strict counts drop, from those
same objects, every coefficient of magnitude at most the strict threshold.
They are not recomputed from a pruned ``H2``, so strict pruning can only remove
words.

Three deterministic checks run on every row. Each row paired with the
reference must equal ``<psi|A_i† H² A_j|psi>``, recomputed as the Gram matrix
of dense statevectors ``H A_j |psi>`` with no multivector product involved.
That block must be positive semidefinite. Every word must lie in the
spin-parity sector the declaration's ceiling rests on. A failed check makes
the bank INVALID. The frozen rule then gives each bank a status, and the
verdict follows.

    python benchmarks/run_phase15_h2_preflight.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:  # pragma: no cover - script execution
        sys.path.insert(0, _path)

try:
    from benchmarks import check_phase15_preregistration as gate
except ImportError:  # pragma: no cover - script execution
    import check_phase15_preregistration as gate

SCHEMA = "clifford_qc.phase15_h2_preflight.v1"
RECORD = gate.RECORD
PACKED_BYTES_PER_COEFFICIENT = 24
DENSE_TOLERANCE = 1e-8
CLAIM_BOUNDARY = (
    "This record reports the one preflight benchmarks/configs/phase15_h2_preflight.json "
    "froze: exact word and coefficient counts of the second-moment rows "
    "A_i^dagger (H^2 A_j) over the retained block of each of the five frozen "
    "mapping-axis banks, both clauses read at both pruning thresholds, and the "
    "verdict the frozen rule gives. The counts price the words a measured "
    "second-moment bank must estimate and the coefficients it must hold. They do "
    "not price grouping, shots, circuits, wall time or hardware, and they bound no "
    "bank outside the five.")


def reference_vector(model) -> np.ndarray:
    """The determinant an X-gate-only reference prepares, in the full space.

    Qubit ``j`` is bit ``n - 1 - j`` of the basis index, the convention
    ``PauliLinearOperator`` applies operators in.
    """
    index = 0
    for operation in model.reference.ops:
        if getattr(operation, "name", None) != "X":
            raise ValueError("the dense check needs an X-gate determinant reference")
        for qubit in operation.qubits:
            index ^= 1 << (model.n - 1 - qubit)
    vector = np.zeros(2 ** model.n, dtype=complex)
    vector[index] = 1.0
    return vector


def dense_second_moments(model, selected, hamiltonian) -> np.ndarray:
    """``<psi|A_i† H² A_j|psi>`` as the Gram matrix of ``H A_j |psi>``."""
    from clifford_qc.pauli_action import PauliLinearOperator

    psi = reference_vector(model)
    h = PauliLinearOperator(hamiltonian)
    columns = np.column_stack([h.matvec(PauliLinearOperator(g.mv).matvec(psi))
                               for g in selected])
    return columns.conj().T @ columns


def _digest(words) -> str:
    """SHA-256 of a word set, as sorted little-endian unsigned 64-bit codes."""
    codes = np.fromiter(sorted(words), dtype="<u8", count=len(words))
    return hashlib.sha256(codes.tobytes()).hexdigest()


def count_bank(config: dict, model, selected, *, progress=None) -> dict:
    """Every count, check value and row of one bank's second-moment block."""
    started = time.perf_counter()
    strict = float(config["pruning"]["strict_tolerance"])
    bank = gate.first_moment_bank(model, selected)
    resources = bank.resources()
    sh_universe = set(bank.word_set())
    hamiltonian, rho, n = bank.hamiltonian, bank.reference, bank.n
    scale = float(2 ** n)

    h2 = hamiltonian * hamiltonian
    h2_strict = sum(1 for value in h2.terms.values() if abs(value) > strict)
    adjoints = [generator.mv.dagger() for generator in selected]
    size = len(selected)
    pairing = np.zeros((size, size), dtype=complex)
    rows = []
    k_words: set[int] = set()
    k_words_strict: set[int] = set()
    k_total = k_total_strict = largest = 0
    for j, generator in enumerate(selected):
        column = h2 * generator.mv
        for i in range(j + 1):
            row = adjoints[i] * column
            terms = row.terms
            kept = [code for code, value in terms.items() if abs(value) > strict]
            k_words.update(terms)
            k_words_strict.update(kept)
            k_total += len(terms)
            k_total_strict += len(kept)
            largest = max(largest, len(terms))
            value = scale * row.trace_pairing(rho)
            pairing[i, j] = value
            rows.append([i, j, len(terms), len(kept), value.real, value.imag])
        del column
        if progress is not None:
            progress(f"    column {j + 1}/{size}: {len(k_words)} K words so far")

    dense = dense_second_moments(model, selected, hamiltonian)
    upper = np.triu_indices(size)
    deviation = np.abs(pairing[upper] - dense[upper]) / (1.0 + np.abs(dense[upper]))
    block = np.triu(pairing) + np.triu(pairing, 1).conj().T
    eigenvalues = np.linalg.eigvalsh(0.5 * (block + block.conj().T))
    magnitude = float(np.max(np.abs(block)))

    combined = sh_universe | k_words
    combined_strict = sh_universe | k_words_strict
    sh_total = int(resources["coefficient_occurrences"])
    violations = (sum(not gate.in_spin_parity_sector(n, code) for code in k_words)
                  + sum(not gate.in_spin_parity_sector(n, code) for code in h2.terms))
    return {
        "counts": {
            "hamiltonian_terms": hamiltonian.nnz(),
            "h2_terms": h2.nnz(),
            "h2_terms_strict": h2_strict,
            "h2_words_outside_h": len(set(h2.terms) - set(hamiltonian.terms)),
            "block_pairs": len(rows),
            "largest_row_terms": largest,
            "sh_word_universe": int(resources["word_universe"]),
            "sh_coefficient_occurrences": sh_total,
            "k_word_universe": len(k_words),
            "k_word_universe_strict": len(k_words_strict),
            "combined_word_universe": len(combined),
            "combined_word_universe_strict": len(combined_strict),
            "additional_words": len(combined) - len(sh_universe),
            "k_coefficient_occurrences": k_total,
            "k_coefficient_occurrences_strict": k_total_strict,
            "k_near_threshold_coefficients": k_total - k_total_strict,
            "h2_near_threshold_coefficients": h2.nnz() - h2_strict,
            "total_coefficient_occurrences": sh_total + k_total,
            "total_coefficient_occurrences_strict": sh_total + k_total_strict,
            "packed_bytes": PACKED_BYTES_PER_COEFFICIENT * (sh_total + k_total),
            "word_ratio": len(combined) / len(sh_universe),
            "word_ratio_strict": len(combined_strict) / len(sh_universe),
        },
        "digests": {
            "k_words_sha256": _digest(k_words),
            "combined_words_sha256": _digest(combined),
            "combined_strict_words_sha256": _digest(combined_strict),
        },
        "dense_check": {
            "max_relative_deviation": float(np.max(deviation)),
            "min_eigenvalue": float(eigenvalues[0]),
            "max_abs_entry": magnitude,
        },
        "sector_violations": violations,
        "rows": rows,
        "seconds": round(time.perf_counter() - started, 2),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def deterministic_checks(frozen: dict, counted: dict) -> dict:
    """The declared checks, as booleans the checker re-derives."""
    counts, dense = counted["counts"], counted["dense_check"]
    return {
        "baseline_matches_freeze": (
            counts["sh_word_universe"] == frozen["sh_word_universe"]
            and counts["sh_coefficient_occurrences"] == frozen["sh_coefficient_occurrences"]
            and counts["hamiltonian_terms"] == frozen["hamiltonian_terms"]),
        "rows_complete": counts["block_pairs"] == frozen["block_pairs"],
        "sector_closed": counted["sector_violations"] == 0,
        "rows_reproduce_dense_second_moments": (
            dense["max_relative_deviation"] <= DENSE_TOLERANCE),
        "gram_psd": dense["min_eigenvalue"] >= -DENSE_TOLERANCE * max(dense["max_abs_entry"], 1.0),
        "strict_not_above_declared": all(
            counts[f"{key}_strict"] <= counts[key] for key in (
                "h2_terms", "k_word_universe", "combined_word_universe",
                "k_coefficient_occurrences", "total_coefficient_occurrences")),
    }


def evaluate(config: dict, counts: dict, checks: dict) -> dict:
    """The frozen bank rule, read off one bank's counts."""
    if not all(checks.values()):
        return {"status": "INVALID",
                "failed_checks": sorted(key for key, ok in checks.items() if not ok)}
    sh = counts["sh_word_universe"]
    declared = gate.clause_passes(
        config, sh_words=sh, combined_words=counts["combined_word_universe"],
        total_coefficients=counts["total_coefficient_occurrences"])
    strict = gate.clause_passes(
        config, sh_words=sh, combined_words=counts["combined_word_universe_strict"],
        total_coefficients=counts["total_coefficient_occurrences_strict"])
    return {
        "status": gate.status_of(declared, strict),
        "clauses": {"declared": {"word": declared[0], "storage": declared[1]},
                    "strict": {"word": strict[0], "storage": strict[1]}},
    }


def run_preflight(config: dict, *, config_bytes: bytes, banks=None, inputs=None,
                  structural: dict | None = None, progress=None) -> dict:
    """The declared preflight as a record (unstamped). ``inputs`` is for tests."""
    inputs = gate.bank_inputs if inputs is None else inputs
    declared = list(config["banks"]["systems"])
    names = declared if banks is None else list(banks)
    started = time.perf_counter()
    record = {
        "schema": SCHEMA,
        "config_path": str(gate.CONFIG.relative_to(ROOT)),
        "config_digest": hashlib.sha256(config_bytes).hexdigest(),
        "claim_boundary": CLAIM_BOUNDARY,
        "preregistration": {
            "gate": "benchmarks/check_phase15_preregistration.py",
            "revision": config["revisions"][-1]["revision"],
            "config_claim_boundary_at_landing": config["claim_boundary"],
        },
        "quantum_advantage_claim": False,
        "evidence": config["evidence"],
        "clauses": config["clauses"],
        "pruning": config["pruning"],
        "row_definition": config["row_definition"],
        "dense_tolerance": DENSE_TOLERANCE,
        "banks": {},
    }
    for name in names:
        model, selected = inputs(name)
        frozen = config["measured_before_freezing"][name]
        measured = (structural or {}).get(name)
        if measured is None:
            measured = gate.quantities_for(config, model, selected)
        if progress is not None:
            progress(f"{name}: n={model.n} M={len(selected)}")
        counted = count_bank(config, model, selected, progress=progress)
        checks = deterministic_checks(frozen, counted)
        decision = evaluate(config, counted["counts"], checks)
        record["banks"][name] = {
            "measured_at_execution": {key: measured[key] for key in gate.MEASURED_FIELDS},
            **{key: counted[key] for key in ("counts", "digests", "dense_check",
                                             "sector_violations", "rows")},
            "deterministic_checks": checks,
            "decision": decision,
            "diagnostics": {"seconds": counted["seconds"],
                            "process_peak_rss_bytes": counted["peak_rss_bytes"]},
        }
        if progress is not None:
            counts = counted["counts"]
            progress(f"{name}: ratio {counts['word_ratio']:.3f}, total coefficients "
                     f"{counts['total_coefficient_occurrences']}, {decision['status']}")
    statuses = {name: record["banks"][name]["decision"]["status"]
                for name in declared if name in record["banks"]}
    complete = len(statuses) == len(declared)
    verdict = gate.verdict_of(statuses.values()) if complete else "INCOMPLETE"
    record["decision"] = {
        "bank_statuses": statuses,
        "eligible_banks": [name for name, status in statuses.items()
                           if status == "ELIGIBLE"],
        "verdict": verdict,
        "consequence": config["consequences"].get(verdict),
        "rule": "frozen in the config; see decision_rule",
    }
    record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    return record


def refusals(args, *, reduced: bool) -> list[str]:
    """Reasons to stop before any computation; each keeps the one run honest."""
    problems = []
    if args.out.resolve() == RECORD.resolve():
        if reduced:
            problems.append("a bank subset (--banks) is not the declared preflight "
                            "and may not write the committed record")
        if RECORD.exists() and not args.overwrite:
            problems.append(f"{RECORD.name} exists: the declared preflight runs once "
                            "(pass --overwrite only to regenerate it deliberately)")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=RECORD,
                        help="record path (default: the committed record)")
    parser.add_argument("--banks", nargs="*", default=None,
                        help="bank subset; refused for the committed path")
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate an existing committed record deliberately")
    args = parser.parse_args(argv)
    problems = refusals(args, reduced=args.banks is not None)
    if problems:
        for problem in problems:
            print(f"REFUSE {problem}")
        return 1

    from clifford_qc.reproducibility import execution_provenance, stamp_record

    config_bytes = gate.CONFIG.read_bytes()
    config = gate.load_config()
    problems = gate.static_problems(config)
    notes: list[str] = []
    computed: dict = {}
    if not problems:
        problems = gate.structural_problems(config, notes, computed)
    if problems:
        for problem in problems:
            print(f"REFUSE preregistration gate: {problem}")
        return 1
    for note in notes:
        print(note)
    provenance = execution_provenance()
    if args.out.resolve() == RECORD.resolve() and provenance["git_dirty"]:
        print("REFUSE the working tree is dirty; commit first so the record's "
              "provenance names the code that produced it")
        return 1
    # The stamp enforces the record-environment contract. Trip it before H²
    # is formed, so an undeclared environment costs nothing, not the run.
    stamp_record({}, provenance)

    record = run_preflight(config, config_bytes=config_bytes, banks=args.banks,
                           structural=computed,
                           progress=lambda line: print(line, flush=True))
    stamped = stamp_record(record, provenance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stamped, indent=1) + "\n", encoding="utf-8")
    decision = record["decision"]
    print(f"\nverdict: {decision['verdict']}  statuses: {decision['bank_statuses']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
