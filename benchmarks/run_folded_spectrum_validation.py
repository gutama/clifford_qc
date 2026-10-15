"""Validate folded-spectrum roots on the banks the Phase 15 preflight licensed.

Folded-spectrum roots read only the three blocks ``SecondMomentBank``
already pairs, so the preflight's licence covers them on the same five banks.
This producer refuses any bank that record does not license. On each bank it
checks the new quantities against routes that share none of their code:

* **Refined Ritz roots.** For every Ritz root ``theta_k`` of the frozen basis,
  the lowest folded root at ``theta_k``. Its folded value, Rayleigh energy and
  variance must equal ``||(H - theta_k)Psi||^2``, ``<H>`` and
  ``||(H - E)Psi||^2`` of the state reconstructed from dense statevectors
  through ``PauliLinearOperator``, to each value's resolution floor. Its folded
  value must not exceed the Ritz root's own variance.
* **Variance minimization.** The self-consistent iteration ``w <- E`` from the
  ground Ritz value must be non-increasing, reach its fixed point, and match
  the dense state at the end.
* **Inclusion.** Every target and Weinstein interval must contain an exact
  eigenvalue of the reference's ``(N, S_z)`` sector. That holds because every
  basis vector's image of the reference lies in the sector, which the producer
  checks to zero first.

The record also reports what the intervals say. Each Weinstein interval that
holds exactly one distinct sector level isolates that level. For each bank it
gives the ground refined residual against the Ritz residual, and the
minimized variance. The values are exact pairings. Nothing here estimates the
block from samples or prices a measurement.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      python benchmarks/run_folded_spectrum_validation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    from benchmarks import run_second_moment_validation as second_moment
except ImportError:  # pragma: no cover - script execution
    import check_phase15_preregistration as gate
    import run_phase15_h2_preflight as preflight_producer
    import run_second_moment_validation as second_moment

SCHEMA = "clifford_qc.folded_spectrum_validation.v1"
RECORD = HERE / "reference_results" / "folded_spectrum_validation.json"
PREFLIGHT = second_moment.PREFLIGHT
# Two sector eigenvalues closer than this, relative to max(1, |E|), are one level.
LEVEL_TOLERANCE = 1e-9
# The dense and bank energies of one state agree to this, per unit |E|.
ENERGY_TOLERANCE = 1e-9
# A Krylov vector or the Hamiltonian leaving the sector by more than this
# invalidates the sector spectrum as the inclusion reference.
SECTOR_LEAKAGE_TOLERANCE = 1e-12
CLAIM_BOUNDARY = (
    "This record validates clifford_qc.subspace.FoldedSpectrum on the banks the Phase 15 "
    "preflight record lists as eligible. For every Ritz root of each frozen basis, the "
    "refined Ritz root's folded value, energy and variance equal those of the state "
    "reconstructed from dense statevectors, its folded value does not exceed the Ritz "
    "root's variance, and its target and Weinstein intervals contain an exact sector "
    "eigenvalue. The variance-minimizing iteration from the ground Ritz value is "
    "non-increasing and reaches its fixed point. Values are exact pairings: they say "
    "nothing about finite-shot estimation of the second-moment block, grouping, or any "
    "bank outside the preflight's.")
EVIDENCE = {
    "folded_roots_and_intervals": {
        "tier": "exact_small_sector_simulation",
        "scope": "Exact second-moment pairings on the five licensed frozen bases, checked "
                 "against dense statevectors and the exact (N, S_z) sector spectrum; no "
                 "finite-shot estimate, grouping or device cost."},
}


def sector_levels(model) -> tuple[list[float], int]:
    """Every eigenvalue of the reference's sector, and its distinct levels."""
    from benchmarks.check_phase15_extrapolation_preregistration import sector_spectrum

    values, dimension = sector_spectrum(model, count=4 ** model.n)
    levels: list[float] = []
    for value in values:
        if not levels or value - levels[-1] > LEVEL_TOLERANCE * max(1.0, abs(value)):
            levels.append(float(value))
    return levels, int(dimension)


def contains(levels, interval) -> list[int]:
    """Indices of the distinct levels inside a closed interval."""
    lo, hi = interval
    return [i for i, level in enumerate(levels) if lo <= level <= hi]


def dense_state_values(basis, hamiltonian, indices, coefficients, target) -> dict:
    """``<H>``, ``||(H - w)Psi||^2`` and ``||(H - E)Psi||^2`` of one dense state."""
    state = basis[:, list(indices)] @ coefficients
    state = state / np.linalg.norm(state)
    acted = hamiltonian.matvec(state)
    energy = float(np.vdot(state, acted).real)
    return {"energy": energy,
            "folded_value": float(np.linalg.norm(acted - target * state) ** 2),
            "variance": float(np.linalg.norm(acted - energy * state) ** 2)}


def root_row(root, dense: dict, levels) -> dict:
    """One folded root, its dense twin, and what its intervals contain."""
    from clifford_qc.subspace.second_moment import RESOLUTION

    row = root.as_dict()
    row.pop("schema")
    row.pop("evidence")
    target_levels = contains(levels, row["target_interval"])
    weinstein_levels = contains(levels, row["weinstein_interval"])
    row.update({
        "dense_energy": dense["energy"],
        "dense_folded_value": dense["folded_value"],
        "dense_variance": dense["variance"],
        "target_levels": target_levels,
        "weinstein_levels": weinstein_levels,
        "isolated_level": weinstein_levels[0] if len(weinstein_levels) == 1 else None,
        "matches_dense": bool(
            abs(row["energy"] - dense["energy"]) <= ENERGY_TOLERANCE * max(1.0, abs(row["energy"]))
            and abs(row["folded_value"] - dense["folded_value"])
            <= RESOLUTION * row["folded_scale"]
            and abs(row["variance"] - dense["variance"]) <= RESOLUTION * row["variance_scale"]),
    })
    return row


def validate_bank(model, selected, *, replay=None) -> dict:
    """Every refined root, the minimization and every check for one bank.

    ``replay`` lists targets at which to re-solve the lowest folded root on
    the same span, as trace steps under the key ``replay``. The checker passes
    a recorded trace's targets so each step is restated without building the
    second moments twice; the record itself carries no replay.
    """
    from benchmarks.phase15_certifying_screen import sector_premises
    from clifford_qc.pauli_action import PauliLinearOperator
    from clifford_qc.subspace import FoldedSpectrum, SecondMomentBank
    from clifford_qc.subspace.second_moment import RESOLUTION

    started = time.perf_counter()
    premises = sector_premises(model, [selected])
    levels, dimension = sector_levels(model)
    bank = gate.first_moment_bank(model, selected)
    moments = SecondMomentBank(bank)
    result = bank.solve()
    spectrum = FoldedSpectrum(moments, result.indices)
    psi = preflight_producer.reference_vector(model)
    basis = np.column_stack([PauliLinearOperator(g.mv).matvec(psi) for g in selected])
    hamiltonian = PauliLinearOperator(bank.hamiltonian)

    roots = []
    for k, theta in enumerate(result.energies):
        ritz = moments.residual(result, k)
        refined = spectrum.refined_ritz(float(theta))
        dense = dense_state_values(basis, hamiltonian, result.indices, refined.coefficients,
                                   float(theta))
        row = root_row(refined, dense, levels)
        row.update({
            "ritz_root": k,
            "ritz_variance": ritz.variance,
            "ritz_residual_norm": ritz.residual_norm,
            "ritz_variance_resolved": ritz.resolved,
            "refined_not_above_ritz": bool(
                refined.folded_value <= ritz.variance + RESOLUTION * refined.folded_scale),
        })
        roots.append(row)

    run = spectrum.minimize_variance(float(result.energies[0]))
    final = run.root
    dense = dense_state_values(basis, hamiltonian, result.indices, final.coefficients,
                               final.target)
    minimization = {
        "start": run.start, "iterations": run.iterations, "converged": run.converged,
        "monotone": run.monotone, "trace": [list(step) for step in run.trace],
        "root": root_row(final, dense, levels),
    }
    every = roots + [minimization["root"]]
    checks = {
        "sector_premise": bool(
            premises["reference_image_sector_leakage"] <= SECTOR_LEAKAGE_TOLERANCE
            and premises["hamiltonian_sector_leakage"] <= SECTOR_LEAKAGE_TOLERANCE),
        "roots_match_dense": all(row["matches_dense"] for row in every),
        "refined_not_above_ritz": all(row["refined_not_above_ritz"] for row in roots),
        "target_intervals_contain_a_level": all(row["target_levels"] for row in every),
        "weinstein_intervals_contain_a_level": all(row["weinstein_levels"] for row in every),
        "minimization_monotone_and_converged": run.monotone and run.converged,
    }
    ground = roots[0]
    out = {
        "n_qubits": model.n,
        "basis_size": len(selected),
        "effective_rank": int(result.effective_rank),
        "sector": {"dimension": dimension, "levels": levels,
                   "reference_image_sector_leakage": premises["reference_image_sector_leakage"],
                   "hamiltonian_sector_leakage": premises["hamiltonian_sector_leakage"]},
        "roots": roots,
        "minimization": minimization,
        "checks": checks,
        "status": "VALIDATED" if all(checks.values()) else "FAILED",
        "reading": {
            "ground_ritz_residual_norm": ground["ritz_residual_norm"],
            "ground_refined_target_bound": math.sqrt(max(ground["folded_value"], 0.0)),
            "minimized_residual_norm": minimization["root"]["residual_norm"],
            "minimized_energy": minimization["root"]["energy"],
            "isolated_levels": sorted({row["isolated_level"] for row in roots
                                       if row["isolated_level"] is not None}),
        },
        "diagnostics": {"seconds": round(time.perf_counter() - started, 2)},
    }
    if replay is not None:
        out["replay"] = []
        for target in replay:
            step = spectrum.roots(float(target), count=1).lowest
            out["replay"].append([step.target, step.folded_value, step.energy, step.variance,
                                  step.folded_scale])
    return out


def run_validation(preflight: dict, *, banks=None, inputs=None, progress=None,
                   preflight_bytes: bytes = b"") -> dict:
    """The validation as an unstamped record. ``inputs`` is for tests."""
    from clifford_qc.evidence import declare_evidence
    from clifford_qc.subspace.second_moment import RESOLUTION

    inputs = gate.bank_inputs if inputs is None else inputs
    licensed = second_moment.licensed_banks(preflight)
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
        "tolerances": {"variance_resolution": RESOLUTION, "energy": ENERGY_TOLERANCE,
                       "level": LEVEL_TOLERANCE, "sector_leakage": SECTOR_LEAKAGE_TOLERANCE},
        "banks": {},
    }
    for name in names:
        model, selected = inputs(name)
        if progress is not None:
            progress(f"{name}: n={model.n} M={len(selected)}")
        entry = validate_bank(model, selected)
        record["banks"][name] = entry
        if progress is not None:
            reading = entry["reading"]
            progress(f"{name}: {entry['status']} in {entry['diagnostics']['seconds']} s; ground "
                     f"residual {reading['ground_ritz_residual_norm']:.3e} -> refined "
                     f"{reading['ground_refined_target_bound']:.3e} -> minimized "
                     f"{reading['minimized_residual_norm']:.3e}")
    statuses = {name: entry["status"] for name, entry in record["banks"].items()}
    record["summary"] = {
        "bank_statuses": statuses,
        "all_validated": (len(statuses) == len(licensed)
                          and all(s == "VALIDATED" for s in statuses.values())),
    }
    record["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    return declare_evidence(record, EVIDENCE)


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
        second_moment.licensed_banks(preflight)
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
    args.out.write_text(json.dumps(stamped, indent=1, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(f"\nall validated: {record['summary']['all_validated']}  "
          f"statuses: {record['summary']['bank_statuses']}")
    print(f"wrote {args.out}")
    return 0 if record["summary"]["all_validated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
