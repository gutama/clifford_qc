"""Does the 2/4/6-operator warm-start anomaly replicate beyond one system?

``reference_results/warm_start_h4.json`` reports something odd: on H4 the
downstream A-CASE error gets *worse* as the ADAPT warm start gets *better*.
Two operators leave ADAPT at 27.09 mHa and A-CASE at 0.342 mHa; six operators
take ADAPT to 6.79 mHa and A-CASE to 0.768 mHa.  That single system, single
budget, three-point trend is the whole evidential basis for treating reference
optimisation as an accuracy lever, and it has never been explained -- all three
rows report full effective rank and ``kappa(S) ~ 1.02-1.06``, so overlap
collapse does not account for it.

This driver replicates the experiment across the five Phase 12 primary systems
and, on each, over a denser operator ladder than three points.  It is a
replication, so the FCIDUMP H4 row is a **positive control**: it must reproduce
the committed numbers, and the record fails to write if it does not.

What decides the question
-------------------------

The anomaly is a claim about *rank correlation*, not about three numbers, so
the record reports Kendall's tau between the ADAPT reference error and the
downstream A-CASE error over each system's ladder.  A negative tau is the
anomaly: better reference, worse subspace.

Two diagnostics are carried alongside, because a rank correlation on its own
does not say what broke:

``reference_capture``
    ``|<psi_ref|psi_exact>|^2`` -- how much of the answer the reference already
    holds.

``subspace_capture``
    ``||P_span |psi_exact>||^2`` over the *retained* A-CASE span.  This is the
    quantity that actually bounds the achievable Ritz error, so if it falls as
    the reference improves, the anomaly is a span problem rather than a
    conditioning or optimisation artifact.

Claim boundary
--------------

Every energy here is exact statevector simulation and every ADAPT cost is an
exact algorithmic count.  The ADAPT prelude is additional hybrid work that this
record prices in rotors, gradient evaluations, and optimizer evaluations, never
in shots.  A warm reference is sector-mixed unless it is projected, and this
driver does not project: the committed H4 record's sector-projected row leaves
its QND projection circuit unpriced, and nothing here removes that caveat.

Run from the repository root, for example::

    python -m benchmarks.run_warm_start_replication --systems hubbard_2x2
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import resource
import sys
import time
from pathlib import Path

# Exact ADAPT optimization contains BLAS reductions.  One thread removes their
# completion-order freedom so operator choices and resource counts reproduce.
_THREADS = os.environ.get("CLIFFORD_QC_BENCHMARK_THREADS", "1")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = _THREADS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import run_acase_ladder as ladder
from benchmarks import run_phase10_hybrid as phase10
from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.reproducibility import execution_provenance, stamp_record
from clifford_qc.subspace import adapt_warm_start, run_acase
from clifford_qc.pauli_action import apply_pauli_sum

ROOT = Path(__file__).resolve().parent
PRIMARY_SYSTEMS = phase10.PRIMARY_SYSTEMS

# The committed warm-start budget: eight adaptive additions beside the identity.
ADAPTIVE_ADDITIONS = 8

# The committed ladder is (2, 4, 6).  Three points cannot distinguish a trend
# from two accidents, so the default here is denser and contains the committed
# rungs as a subset.
DEFAULT_OPERATOR_LADDER = (1, 2, 3, 4, 5, 6, 8)
COMMITTED_LADDER = (2, 4, 6)

CHEMICAL_ACCURACY_HARTREE = 1.6e-3

# The exact MV backend stores a state by its Pauli-word support, and that
# support grows multiplicatively with non-commuting rotors.  On 12 qubits an
# eight-rotor ADAPT state can reach the 4^12 ceiling, which does not fail
# gracefully -- an unbounded run is simply killed, taking the whole ladder with
# it.  A declared address-space ceiling converts that into a catchable
# MemoryError, so the rung is recorded as a failed row with its reason and the
# remaining systems still run.  This is a real limit of the exact backend, not
# a defect to tune away.
DEFAULT_MEMORY_LIMIT_GB = 8.0

# The positive control: the committed record this replication has to reproduce
# before any of its other rows mean anything.
CONTROL_RECORD = ROOT / "reference_results" / "warm_start_h4.json"
CONTROL_SYSTEM = "fcidump_h4_equilibrium"
CONTROL_TOLERANCE_MHA = 1e-6


def kendall_tau(first, second) -> float | None:
    """Kendall's tau-b over two short sequences, ties included.

    Written out rather than pulled from SciPy because the ladders here are
    seven points long and the tie correction is the entire subtlety: a ladder
    whose A-CASE errors are flat should report tau near zero, not a spurious
    +/-1 from an arbitrary tie-break.
    """
    x = [float(value) for value in first]
    y = [float(value) for value in second]
    if len(x) != len(y):
        raise ValueError("kendall_tau needs two equal-length sequences")
    if len(x) < 2:
        return None
    concordant = discordant = tied_x = tied_y = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            tied_x += 1
        elif dy == 0:
            tied_y += 1
        elif (dx > 0) == (dy > 0):
            concordant += 1
        else:
            discordant += 1
    denominator = ((concordant + discordant + tied_x)
                   * (concordant + discordant + tied_y)) ** 0.5
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def _sector_ground_vector(backend: SectorStatevectorBackend, model,
                          exact_energy: float) -> np.ndarray:
    """The exact sector ground state embedded in the full ``2^n`` space.

    The capture diagnostics compare against the A-CASE span, which lives in the
    dense full space, so the sector vector has to be scattered into it.
    """
    values, vectors = backend.ground_state(model.hamiltonian, k=1)
    if abs(float(values[0]) - exact_energy) > 1e-9:
        raise ValueError("sector ground state disagrees with the exact energy")
    full = np.zeros(2 ** model.n, dtype=complex)
    full[np.asarray(backend.basis, dtype=np.int64)] = vectors[:, 0]
    return full / np.linalg.norm(full)


def pure_vector(rho, n: int, *, tol: float = 1e-9) -> np.ndarray:
    """Matrix-free state vector of a pure density multivector, phase-fixed.

    ``subspace.reference.pure_statevector`` diagonalizes the dense ``2^n x 2^n``
    matrix, which is a small-``n`` oracle and costs more than the benchmark it
    is diagnosing by 12 qubits.  Since ``rho = |psi><psi|``, applying it to any
    probe with nonzero overlap returns ``|psi>`` up to a scale, so one
    matrix-free matvec suffices.  Purity is then *verified* -- ``rho|psi> =
    |psi>`` -- so this keeps the check that made the dense route safe rather
    than trading it away for speed.
    """
    dimension = 2 ** n
    generator = np.random.default_rng(0)
    for attempt in range(4):
        probe = (np.zeros(dimension, dtype=complex) if attempt == 0
                 else generator.normal(size=dimension)
                 + 1j * generator.normal(size=dimension))
        if attempt == 0:
            probe[0] = 1.0
        candidate = apply_pauli_sum(rho, probe)
        norm = float(np.linalg.norm(candidate))
        if norm > tol:
            break
    else:                                            # pragma: no cover
        raise ValueError("reference density annihilated every probe")
    psi = candidate / norm
    residual = float(np.linalg.norm(apply_pauli_sum(rho, psi) - psi))
    if residual > 1e-7:
        raise ValueError(
            f"reference density is not a pure state (residual {residual:.3e})")
    lead = psi[int(np.argmax(np.abs(psi)))]
    return psi * (np.conjugate(lead) / abs(lead))


def generator_basis(rho, generators, n: int) -> np.ndarray:
    """Columns ``A_i|psi>``, matrix-free."""
    psi = pure_vector(rho, n)
    return np.column_stack([apply_pauli_sum(generator.mv, psi)
                            for generator in generators])


def _capture(vectors: np.ndarray, target: np.ndarray) -> float:
    """``||P_span target||^2`` for a span given by (not necessarily orthonormal) columns."""
    Q, _ = np.linalg.qr(vectors)
    return float(np.abs(Q.conj().T @ target) ** 2 @ np.ones(Q.shape[1]))


def _row(result, exact_energy: float, rho, target: np.ndarray,
         n: int) -> dict:
    subspace = result.result
    error = result.energy - exact_energy
    basis = generator_basis(
        rho, [result.bank.generator(index) for index in result.indices], n)
    return {
        "basis_size": len(subspace.basis_labels),
        "labels": list(subspace.basis_labels),
        "ground_energy": float(result.energy),
        "error_hartree": float(error),
        "error_millihartree": float(error * 1e3),
        "chemical_accuracy": bool(abs(error) < CHEMICAL_ACCURACY_HARTREE),
        "condition_number": float(subspace.condition_number),
        "effective_rank": int(subspace.effective_rank),
        "word_universe": subspace.resources.get("word_universe"),
        "reference_capture": float(np.abs(
            np.vdot(pure_vector(rho, n), target)) ** 2),
        "subspace_capture": _capture(basis, target),
    }


def apply_memory_limit(limit_gb: float) -> dict:
    """Cap address space so a support blow-up raises instead of being killed."""
    if limit_gb <= 0:
        return {"applied": False, "note": "no memory ceiling requested"}
    limit = int(limit_gb * 1024 ** 3)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    return {"applied": True, "limit_bytes": limit, "limit_gb": limit_gb,
            "note": "an ADAPT rung whose Pauli-word support exceeds this "
                    "raises MemoryError and is recorded as a failed row"}


def run_system(name: str, *, additions: int = ADAPTIVE_ADDITIONS,
               operator_ladder=DEFAULT_OPERATOR_LADDER) -> dict:
    """Cold and warm A-CASE over one system's ADAPT operator ladder."""
    if name not in PRIMARY_SYSTEMS:
        raise ValueError(f"unknown primary system {name!r}")

    model, construction = phase10.build_system(name)
    backend = SectorStatevectorBackend(
        model.n, int(model.metadata["n_electrons"]), float(model.metadata["sz"]))
    exact_energy = float(backend.ground_state(model.hamiltonian, k=1)[0][0])
    target = _sector_ground_vector(backend, model, exact_energy)

    candidates = ladder.build_candidates(model)
    pool = ladder.word_pool(model)

    started = time.perf_counter()
    cold_rho = ExactMVBackend().state(model.reference, ())
    cold = run_acase(cold_rho, model.hamiltonian, candidates,
                     max_size=additions, leakage_tol=1e-10,
                     exact_ground_energy=exact_energy)
    cold_row = _row(cold, exact_energy, cold_rho, target, model.n)
    cold_row.update({"reference": "reference determinant",
                     "adapt_operators": 0,
                     "adapt_error_millihartree": None,
                     "wall_seconds": time.perf_counter() - started})
    print(f"  cold: {cold_row['error_millihartree']:.6f} mHa "
          f"M={cold_row['basis_size']} capture={cold_row['subspace_capture']:.6f}",
          flush=True)

    warm_rows = []
    for operators in operator_ladder:
        started = time.perf_counter()
        try:
            rho, adapt = adapt_warm_start(model, pool, max_operators=operators,
                                          compute_exact_reference=False)
            warm = run_acase(rho, model.hamiltonian, candidates,
                             max_size=additions, leakage_tol=1e-10,
                             exact_ground_energy=exact_energy)
            row = _row(warm, exact_energy, rho, target, model.n)
        except (ValueError, RuntimeError, MemoryError) as error:
            # A rung the exact backend cannot carry is a datum about the
            # backend, not a reason to lose the rest of the ladder.
            warm_rows.append({
                "requested_operators": int(operators),
                "adapt_operators": int(operators),
                "failed": f"{type(error).__name__}: {error}",
                "wall_seconds": time.perf_counter() - started})
            print(f"  warm k={operators}: FAILED "
                  f"({type(error).__name__})", flush=True)
            continue
        adapt_error = float(adapt.energy - exact_energy)
        row.update({
            "reference": f"ADAPT-VQE state, {operators} operators",
            "requested_operators": int(operators),
            "adapt_operators": len(adapt.labels),
            "adapt_labels": list(adapt.labels),
            "adapt_energy": float(adapt.energy),
            "adapt_error_hartree": adapt_error,
            "adapt_error_millihartree": adapt_error * 1e3,
            "adapt_chemical_accuracy":
                bool(abs(adapt_error) < CHEMICAL_ACCURACY_HARTREE),
            "adapt_selection_mode": "exact statevector",
            "adapt_selection_shots": adapt.total_shots,
            "adapt_selection_circuits": adapt.total_circuits,
            "adapt_gradient_evaluations":
                sum(record.active_candidates for record in adapt.records),
            "adapt_optimizer_evaluations": adapt.optimizer_evaluations,
            "adapt_support_peak": adapt.support_peak,
            "adapt_state_preparation_operators": len(adapt.labels),
            "wall_seconds": time.perf_counter() - started,
        })
        warm_rows.append(row)
        print(f"  warm k={operators}: adapt={row['adapt_error_millihartree']:.4f} "
              f"-> acase={row['error_millihartree']:.6f} mHa "
              f"kappa={row['condition_number']:.4g} "
              f"capture={row['subspace_capture']:.6f}", flush=True)

    return {
        "system": name,
        "construction": construction,
        "n_qubits": int(model.n),
        "n_electrons": int(model.metadata["n_electrons"]),
        "sz": float(model.metadata["sz"]),
        "sector_dimension": int(backend.dimension),
        "spin_ordering": str(backend.spin_ordering),
        "exact_energy": exact_energy,
        "predeclared_additions": int(additions),
        "operator_ladder": [int(k) for k in operator_ladder],
        "candidate_generators": len(candidates),
        "adapt_pool_size": len(pool),
        "cold": cold_row,
        "warm": warm_rows,
        "anomaly": analyse_anomaly(cold_row, warm_rows),
    }


def analyse_anomaly(cold_row: dict, warm_rows: dict) -> dict:
    """Does a better ADAPT reference buy a worse A-CASE subspace here?

    The committed observation is a rank statement, so it is tested as one.  A
    negative ``tau_adapt_vs_acase`` means the downstream error moves *against*
    the reference error -- the anomaly.  ``tau_adapt_vs_capture`` says whether
    the retained span is what moved, which is the difference between a span
    problem and a solver artifact.
    """
    live = [row for row in warm_rows if "failed" not in row]
    if len(live) < 2:
        return {"replicated": None,
                "reason": "fewer than two usable warm rows"}
    adapt_errors = [abs(row["adapt_error_millihartree"]) for row in live]
    acase_errors = [abs(row["error_millihartree"]) for row in live]
    captures = [row["subspace_capture"] for row in live]

    tau = kendall_tau(adapt_errors, acase_errors)
    committed = [row for row in live
                 if row.get("requested_operators") in COMMITTED_LADDER]
    committed_tau = (kendall_tau(
        [abs(row["adapt_error_millihartree"]) for row in committed],
        [abs(row["error_millihartree"]) for row in committed])
        if len(committed) >= 2 else None)

    best_adapt = min(live, key=lambda row: abs(row["adapt_error_millihartree"]))
    best_acase = min(live, key=lambda row: abs(row["error_millihartree"]))
    return {
        "tau_adapt_vs_acase": tau,
        "tau_adapt_vs_acase_committed_rungs": committed_tau,
        "tau_adapt_vs_capture": kendall_tau(adapt_errors,
                                            [-value for value in captures]),
        "replicated": None if tau is None else bool(tau < 0.0),
        "interpretation": (
            "tau < 0 is the anomaly: a lower ADAPT reference error goes with a "
            "higher downstream A-CASE error. tau_adapt_vs_capture < 0 says the "
            "retained span itself got worse, which is a span problem rather "
            "than a conditioning or optimizer artifact"),
        "best_adapt_operators": best_adapt.get("requested_operators"),
        "best_acase_operators": best_acase.get("requested_operators"),
        "best_reference_is_best_subspace": bool(
            best_adapt.get("requested_operators")
            == best_acase.get("requested_operators")),
        "acase_error_range_millihartree": [min(acase_errors), max(acase_errors)],
        "adapt_error_range_millihartree": [min(adapt_errors), max(adapt_errors)],
        "cold_error_millihartree": cold_row["error_millihartree"],
        "warm_rows_beating_cold": int(sum(
            1 for value in acase_errors
            if value < abs(cold_row["error_millihartree"]))),
        "condition_numbers": [row["condition_number"] for row in live],
        "effective_ranks": [row["effective_rank"] for row in live],
    }


def check_control(record: dict, *, tolerance: float = CONTROL_TOLERANCE_MHA
                  ) -> dict:
    """The FCIDUMP H4 rows must reproduce the committed warm-start record.

    A replication whose positive control drifts is measuring a different
    experiment, so this is a gate rather than a report.
    """
    if not CONTROL_RECORD.exists():
        return {"available": False,
                "note": f"{CONTROL_RECORD.name} is not present; the control "
                        "cannot be checked in this tree"}
    committed = json.loads(CONTROL_RECORD.read_text(encoding="utf-8"))
    expected = {int(row["adapt_operators"]): float(row["error_millihartree"])
                for row in committed["warm"]}
    expected_cold = float(committed["cold"]["error_millihartree"])

    observed = {int(row["adapt_operators"]): float(row["error_millihartree"])
                for row in record["warm"] if "failed" not in row}
    problems, compared = [], {}
    if abs(record["cold"]["error_millihartree"] - expected_cold) > tolerance:
        problems.append(
            f"cold row: expected {expected_cold:.9f} mHa, got "
            f"{record['cold']['error_millihartree']:.9f} mHa")
    for operators, want in sorted(expected.items()):
        got = observed.get(operators)
        if got is None:
            problems.append(f"k={operators}: missing from the replication")
            continue
        compared[str(operators)] = {"committed": want, "replicated": got,
                                    "difference": got - want}
        if abs(got - want) > tolerance:
            problems.append(
                f"k={operators}: expected {want:.9f} mHa, got {got:.9f} mHa")
    return {"available": True, "source": CONTROL_RECORD.name,
            "tolerance_millihartree": tolerance, "compared": compared,
            "cold_committed": expected_cold,
            "cold_replicated": record["cold"]["error_millihartree"],
            "problems": problems, "passed": not problems}


def check_invariants(document: dict) -> None:
    """Fail at the producer boundary rather than write a misleading record."""
    problems: list[str] = []
    for record in document["systems"]:
        exact = float(record["exact_energy"])
        for row in [record["cold"]] + record["warm"]:
            if "failed" in row:
                continue
            if row["ground_energy"] < exact - 1e-9:
                problems.append(
                    f"{record['system']} k={row.get('adapt_operators')} is "
                    f"below the variational bound")
            capture = row["subspace_capture"]
            if not -1e-9 <= capture <= 1.0 + 1e-9:
                problems.append(
                    f"{record['system']} k={row.get('adapt_operators')} has "
                    f"an out-of-range subspace capture {capture}")
            # The retained span always contains the reference direction, so it
            # can never capture less of the target than the reference alone.
            if capture < row["reference_capture"] - 1e-8:
                problems.append(
                    f"{record['system']} k={row.get('adapt_operators')} "
                    f"captures less ({capture:.6g}) than its own reference "
                    f"({row['reference_capture']:.6g})")
    control = document["control"]
    if control.get("available") and not control["passed"]:
        problems.append("the positive control failed: "
                        + "; ".join(control["problems"]))
    if problems:
        raise AssertionError("warm-start replication invariants failed:\n  "
                             + "\n  ".join(problems))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--systems", default=",".join(PRIMARY_SYSTEMS))
    parser.add_argument("--additions", type=int, default=ADAPTIVE_ADDITIONS,
                        help="adaptive A-CASE additions beside the identity")
    parser.add_argument("--operator-ladder", default=",".join(
        str(k) for k in DEFAULT_OPERATOR_LADDER))
    parser.add_argument("--memory-limit-gb", type=float,
                        default=DEFAULT_MEMORY_LIMIT_GB,
                        help="address-space ceiling; 0 disables it")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "warm_start_replication.json")
    args = parser.parse_args(argv)
    memory_limit = apply_memory_limit(args.memory_limit_gb)

    systems = [name.strip() for name in args.systems.split(",") if name.strip()]
    unknown = sorted(set(systems) - set(PRIMARY_SYSTEMS))
    if unknown:
        raise SystemExit(f"unknown systems: {unknown}")
    operator_ladder = tuple(int(k) for k in args.operator_ladder.split(","))

    records = []
    for name in systems:
        print(f"{name}:", flush=True)
        records.append(run_system(name, additions=args.additions,
                                  operator_ladder=operator_ladder))

    control = next((check_control(record) for record in records
                    if record["system"] == CONTROL_SYSTEM),
                   {"available": False,
                    "note": f"{CONTROL_SYSTEM} was not in this run, so the "
                            "committed warm-start record was not replicated"})
    document = stamp_record({
        "schema": "clifford_qc.warm_start_replication.v1",
        "systems": records,
        "control": control,
        "committed_ladder": list(COMMITTED_LADDER),
        "memory_limit": memory_limit,
        "claim_boundary": (
            "Exact statevector energies and exact algorithmic ADAPT counts. "
            "The ADAPT prelude is additional hybrid work priced in rotors, "
            "gradient evaluations, and optimizer evaluations, never in shots. "
            "Warm references here are sector-mixed and unprojected: the "
            "committed H4 record's sector-projected row leaves its QND "
            "projection circuit unpriced, and nothing here removes that."),
        "evidence": {
            "energies": "exact statevector simulation",
            "resource_counts": "exact algorithmic counts",
            "physical_shot_budget": "not estimated",
            "quantum_advantage_claim": False},
    }, execution_provenance())
    check_invariants(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")

    print()
    for record in records:
        anomaly = record["anomaly"]
        print(f"{record['system']:<24s} tau={anomaly['tau_adapt_vs_acase']} "
              f"committed-rungs tau={anomaly['tau_adapt_vs_acase_committed_rungs']} "
              f"replicated={anomaly['replicated']}")
    if control.get("available"):
        print(f"positive control: {'PASS' if control['passed'] else 'FAIL'}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
