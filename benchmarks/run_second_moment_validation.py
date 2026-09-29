"""Validate SecondMomentBank on the banks the Phase 15 preflight licensed.

The preflight's verdict, FULL, says what follows: build ``SecondMomentBank``
for the retained block and validate it on all five frozen mapping-axis banks
against a dense residual oracle. This producer does that, and nothing else.
It reads the committed preflight record and refuses any bank that record
does not list as eligible, so the licence is enforced here rather than
restated.

Each bank is checked on three independent routes:

* **The block.** ``K`` from the bank's projected-observable rows must equal the
  Gram matrix of ``H A_j|psi>``, computed from dense statevectors through
  ``PauliLinearOperator``, which uses no multivector product.
* **The residuals.** For every Ritz root, the variance from ``K`` must equal
  ``||(H - E)|Psi>||^2 / <Psi|Psi>`` computed matrix-free from the
  reconstructed state, within the module's resolution times the root's
  cancellation scale.
* **The rows.** The word universe, the coefficient counts, the largest row and
  the word-set digest must equal the preflight record's. These are the rows
  the preflight priced.

The record reports each root's true residual norm, which no committed record
has carried before, and whether it is resolved above rounding. It makes no
claim about finite-shot estimation of ``K``: the values are exact pairings.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      python benchmarks/run_second_moment_validation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    from benchmarks import run_phase15_h2_preflight as preflight_producer
except ImportError:  # pragma: no cover - script execution
    import check_phase15_preregistration as gate
    import run_phase15_h2_preflight as preflight_producer

SCHEMA = "clifford_qc.second_moment_validation.v1"
RECORD = HERE / "reference_results" / "second_moment_validation.json"
PREFLIGHT = gate.RECORD
BLOCK_TOLERANCE = 1e-10
LICENSED_VERDICTS = ("FULL", "RESTRICTED")
CLAIM_BOUNDARY = (
    "This record validates clifford_qc.subspace.SecondMomentBank on the banks the "
    "Phase 15 preflight record lists as eligible. The second-moment block equals the "
    "dense Gram matrix of H A_j|psi>, every Ritz root's variance equals the "
    "matrix-free dense residual, and the rows are the ones the preflight priced. "
    "The residual norms are exact properties of each frozen basis's Ritz states. "
    "They say nothing about finite-shot estimation of the block, grouping its words, "
    "or any bank outside the preflight's.")
COUNT_FIELDS = (
    ("second_moment_word_universe", "k_word_universe"),
    ("second_moment_coefficient_occurrences", "k_coefficient_occurrences"),
    ("combined_word_universe", "combined_word_universe"),
    ("sh_coefficient_occurrences", "sh_coefficient_occurrences"),
    ("largest_row_terms", "largest_row_terms"),
    ("hamiltonian_square_terms", "h2_terms"),
)


def load_preflight(path: Path = PREFLIGHT) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def licensed_banks(preflight: dict) -> list[str]:
    decision = preflight.get("decision", {})
    if decision.get("verdict") not in LICENSED_VERDICTS:
        raise ValueError(f"the preflight verdict {decision.get('verdict')!r} licenses "
                         "no SecondMomentBank")
    return list(decision["eligible_banks"])


def _digest(words) -> str:
    codes = np.fromiter(sorted(words), dtype="<u8", count=len(words))
    return hashlib.sha256(codes.tobytes()).hexdigest()


def dense_residuals(model, selected, hamiltonian, result) -> list[float]:
    """``||(H - E_k)|Psi_k>|| / ||Psi_k||`` from reconstructed statevectors."""
    from clifford_qc.pauli_action import PauliLinearOperator

    psi = preflight_producer.reference_vector(model)
    basis = np.column_stack([PauliLinearOperator(g.mv).matvec(psi) for g in selected])
    h = PauliLinearOperator(hamiltonian)
    out = []
    for k, energy in enumerate(result.energies):
        state = basis[:, list(result.indices)] @ result.coefficients[:, k]
        norm = float(np.linalg.norm(state))
        out.append(float(np.linalg.norm(h.matvec(state) - energy * state)) / norm)
    return out


def validate_bank(model, selected, frozen_counts: dict, frozen_digest: str) -> dict:
    """Every quantity and check for one bank."""
    from clifford_qc.subspace import SecondMomentBank
    from clifford_qc.subspace.second_moment import RESOLUTION

    started = time.perf_counter()
    bank = gate.first_moment_bank(model, selected)
    moments = SecondMomentBank(bank)
    block = moments.matrix()
    dense = preflight_producer.dense_second_moments(model, selected, bank.hamiltonian)
    deviation = float(np.max(np.abs(block - dense) / (1.0 + np.abs(dense))))
    lowest = float(np.linalg.eigvalsh(block)[0])
    magnitude = float(np.max(np.abs(block)))
    resources = moments.resources()
    digest = _digest(moments.word_set())
    result = bank.solve()
    residuals = moments.residuals(result)
    oracle = dense_residuals(model, selected, bank.hamiltonian, result)
    roots = []
    for residual, reference in zip(residuals, oracle):
        row = residual.as_dict()
        row["dense_residual_norm"] = reference
        row["variance_minus_dense"] = residual.variance - reference * reference
        roots.append(row)
    checks = {
        "block_matches_dense": deviation <= BLOCK_TOLERANCE,
        "block_psd": lowest >= -BLOCK_TOLERANCE * max(magnitude, 1.0),
        "variances_match_dense": all(
            abs(row["variance_minus_dense"]) <= RESOLUTION * row["cancellation_scale"]
            for row in roots),
        "counts_match_preflight": all(
            resources[mine] == frozen_counts[theirs] for mine, theirs in COUNT_FIELDS),
        "digest_matches_preflight": digest == frozen_digest,
    }
    return {
        "n_qubits": model.n,
        "basis_size": len(selected),
        "resources": resources,
        "k_words_sha256": digest,
        "block_check": {"max_relative_deviation": deviation, "min_eigenvalue": lowest,
                        "max_abs_entry": magnitude},
        "roots": roots,
        "effective_rank": int(result.effective_rank),
        "checks": checks,
        "status": "VALIDATED" if all(checks.values()) else "FAILED",
        "diagnostics": {"seconds": round(time.perf_counter() - started, 2)},
    }


def run_validation(preflight: dict, *, banks=None, inputs=None, progress=None,
                   preflight_bytes: bytes = b"") -> dict:
    """The validation as a record (unstamped). ``inputs`` is for tests."""
    from clifford_qc.subspace.second_moment import RESOLUTION

    inputs = gate.bank_inputs if inputs is None else inputs
    licensed = licensed_banks(preflight)
    names = licensed if banks is None else list(banks)
    refused = sorted(set(names) - set(licensed))
    if refused:
        raise ValueError(f"the preflight does not license {refused}")
    started = time.perf_counter()
    record = {
        "schema": SCHEMA,
        "claim_boundary": CLAIM_BOUNDARY,
        "preflight": {
            "record": str(PREFLIGHT.relative_to(ROOT)),
            "sha256": hashlib.sha256(preflight_bytes).hexdigest(),
            "verdict": preflight["decision"]["verdict"],
            "eligible_banks": licensed,
        },
        "quantum_advantage_claim": False,
        "evidence": {"label": "exact",
                     "statement": "Exact second-moment pairings against dense statevectors."},
        "tolerances": {"block_relative": BLOCK_TOLERANCE,
                       "variance_resolution": RESOLUTION},
        "banks": {},
    }
    for name in names:
        model, selected = inputs(name)
        entry = preflight["banks"][name]
        if progress is not None:
            progress(f"{name}: n={model.n} M={len(selected)}")
        record["banks"][name] = validate_bank(model, selected, entry["counts"],
                                              entry["digests"]["k_words_sha256"])
        if progress is not None:
            ground = record["banks"][name]["roots"][0]
            progress(f"{name}: {record['banks'][name]['status']}, ground residual "
                     f"{ground['residual_norm']:.3e} (dense {ground['dense_residual_norm']:.3e})")
    statuses = {name: entry["status"] for name, entry in record["banks"].items()}
    record["summary"] = {
        "bank_statuses": statuses,
        "all_validated": (len(statuses) == len(licensed)
                          and all(s == "VALIDATED" for s in statuses.values())),
        "ground_residual_norms": {name: entry["roots"][0]["residual_norm"]
                                  for name, entry in record["banks"].items()},
    }
    record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    return record


def refusals(args, *, reduced: bool) -> list[str]:
    problems = []
    if args.out.resolve() == RECORD.resolve():
        if reduced:
            problems.append("a bank subset (--banks) may not write the committed record")
        if RECORD.exists() and not args.overwrite:
            problems.append(f"{RECORD.name} exists (pass --overwrite to regenerate it "
                            "deliberately)")
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

    if not PREFLIGHT.exists():
        print(f"REFUSE the preflight record {PREFLIGHT.name} does not exist")
        return 1
    preflight_bytes = PREFLIGHT.read_bytes()
    preflight = json.loads(preflight_bytes)
    try:
        licensed_banks(preflight)
    except ValueError as exc:
        print(f"REFUSE {exc}")
        return 1
    provenance = execution_provenance()
    if args.out.resolve() == RECORD.resolve() and provenance["git_dirty"]:
        print("REFUSE the working tree is dirty; commit first so the record's "
              "provenance names the code that produced it")
        return 1
    stamp_record({}, provenance)
    record = run_validation(preflight, banks=args.banks, preflight_bytes=preflight_bytes,
                            progress=lambda line: print(line, flush=True))
    stamped = stamp_record(record, provenance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stamped, indent=1) + "\n", encoding="utf-8")
    print(f"\nall validated: {record['summary']['all_validated']}  "
          f"statuses: {record['summary']['bank_statuses']}")
    print(f"wrote {args.out}")
    return 0 if record["summary"]["all_validated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
