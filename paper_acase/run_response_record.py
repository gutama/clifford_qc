"""Regenerate the paper's finite-shot nonlinear-response records.

The records are deliberately separate from the figure.  They contain the full
replica accounting and the fixed seeds needed to reproduce the reported
percentile intervals.  The intervals remain heuristic and conditional on
replicas that preserve the thresholded rank and ordered-root identity.

Two records are written, and the second exists because the first cannot make
the paper's own point.  The determinant-excitation basis is perfectly
conditioned, every replica survives, and the failure accounting that Secs.
III C and V C describe at length never fires -- so the machinery is described
but not demonstrated.

The ill-conditioned record fixes that as a controlled comparison rather than a
different experiment.  Same dimer, same staggered-spin observable, same 63
words, same 25 QWC groups, same 8,000 shots per group, same basis size M=4,
same seeds.  The single change is the generator family: determinant
excitations become the Krylov powers ``I, H, H^2, H^3``, which span the same
sector while carrying an overlap condition number four orders larger.  Any
difference in the replica accounting is therefore attributable to conditioning
and to nothing else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clifford_qc.backends import ExactMVBackend, FiniteShotBackend
from clifford_qc.models import load_effective_hamiltonian, magnetization
from clifford_qc.subspace import (
    MatrixElementBank,
    ResponseMeasurement,
    bootstrap_response,
    determinant_excitations,
    identity_generator,
    krylov_response,
    occupied_spin_orbitals,
)
from clifford_qc.subspace.response import broaden_response

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "examples" / "data" / "wannier_hubbard_dimer.json"
DATA = Path(__file__).resolve().parent / "data"
DEFAULT_OUT = DATA / "response_bootstrap.json"
ILL_CONDITIONED_OUT = DATA / "response_bootstrap_illconditioned.json"

SHOTS_PER_GROUP = 8_000
BOOTSTRAP_REPLICATES = 200
MEASUREMENT_SEED = 2026
BOOTSTRAP_SEED = 17
BROADENING_EV = 0.1
# The Krylov order that matches the determinant family's basis size: I, H, H^2,
# H^3 is M=4, exactly the four vectors the determinant arm uses.
ILL_CONDITIONED_KRYLOV_ORDER = 3
# bootstrap_response refuses a run whose acceptance falls below this fraction.
# The well-conditioned arm accepts every replica; the ill-conditioned arm is
# the measurement being made, so it must be allowed to report a low rate
# rather than raise.
MINIMUM_SUCCESS_FRACTION = 0.01


def build_record(*, generators: str = "determinant") -> dict:
    model = load_effective_hamiltonian(MODEL)
    rho = ExactMVBackend().state(model.reference, ())
    if generators == "determinant":
        family = determinant_excitations(
            model.n, occupied_spin_orbitals(model))
        family_label = "determinant excitations"
    elif generators == "krylov":
        family = krylov_response(
            model.hamiltonian, ILL_CONDITIONED_KRYLOV_ORDER)
        family_label = (
            f"Hamiltonian powers to order {ILL_CONDITIONED_KRYLOV_ORDER}")
    else:
        raise ValueError(f"unknown generator family {generators!r}")
    bank = MatrixElementBank(
        rho, model.hamiltonian, [identity_generator(model.n), *family])
    staggered = magnetization(model, 0) + (-1.0) * magnetization(model, 1)
    measurement = ResponseMeasurement(
        bank, staggered, label="staggered_spin")
    cache = measurement.measure(
        FiniteShotBackend(seed=MEASUREMENT_SEED), SHOTS_PER_GROUP)

    frequencies = np.linspace(0.0, 2.0, 401)
    result = bootstrap_response(
        measurement,
        cache,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        min_weight=1e-3,
        frequencies=frequencies,
        broadening=BROADENING_EV,
        minimum_success_fraction=MINIMUM_SUCCESS_FRACTION,
    )
    exact = measurement.exact_spectrum(min_weight=1e-12)
    exact_broadened = broaden_response(
        exact.lines, frequencies, BROADENING_EV)

    line_records = []
    for item in result.lines:
        line_records.append({
            "initial_state": item.line.initial_state,
            "final_state": item.line.final_state,
            "gap_ev": {
                "estimate": item.excitation_energy.estimate,
                "lower": item.excitation_energy.lower,
                "upper": item.excitation_energy.upper,
            },
            "weight": {
                "estimate": item.weight.estimate,
                "lower": item.weight.lower,
                "upper": item.weight.upper,
            },
        })

    succeeded = result.replicates_succeeded
    requested = result.replicates_requested
    failures = {
        "rank": result.rank_failures,
        "root_collision": result.root_collision_failures,
        "solver": result.solver_failures,
    }
    if succeeded + sum(failures.values()) != requested:
        raise RuntimeError("bootstrap replica accounting does not close")

    # One row per replica, in draw order.  This is the fingerprint that makes a
    # census disagreement between two environments diagnosable: acceptance
    # turns on the sign of the resampled overlap's smallest eigenvalue, so a
    # run accepting a different number of replicas can be diffed against this
    # record replica by replica, and each mover's distance from zero says
    # whether the decision was marginal or the pipeline differs.
    census = [{
        "index": item.index,
        "outcome": item.outcome,
        "overlap_eigenvalue_min": item.overlap_eigenvalue_min,
        "effective_rank": item.effective_rank,
    } for item in result.replica_census]
    if len(census) != requested:
        raise RuntimeError("replica census does not cover every replica")

    return {
        "schema": "clifford_qc.acase_response_bootstrap.v2",
        "source_main_commit": "4b1c636953e4ebe9aa7541d4260cfe95aa18674e",
        "basis": {
            "generators": generators,
            "family": family_label,
            "size": len(family) + 1,
            "condition_number": result.spectrum.result.condition_number,
            "effective_rank": result.spectrum.result.effective_rank,
        },
        "system": {
            "name": model.name,
            "n_qubits": model.n,
            "observable": "staggered spin",
            "energy_unit": model.metadata["energy_unit"],
        },
        "measurement": {
            "words": len(measurement.words),
            "qwc_groups": len(measurement.groups),
            "shots_per_group": SHOTS_PER_GROUP,
            "total_shots": cache.total_shots,
            "measurement_seed": MEASUREMENT_SEED,
        },
        "bootstrap": {
            "method": "grouped nonparametric percentile bootstrap",
            "replicates_requested": requested,
            "replicates_succeeded": succeeded,
            "acceptance_rate": succeeded / requested,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "failures": failures,
            "conditional_on_surviving_replicas": True,
            "evidence": result.evidence,
            "certified": result.certified,
            "pointwise_not_simultaneous": True,
            "replica_census": census,
        },
        "exact": {
            "lines": [{
                "initial_state": line.initial_state,
                "final_state": line.final_state,
                "gap_ev": line.excitation_energy,
                "weight": line.weight,
            } for line in exact.lines],
            "susceptibility_per_ev": exact.susceptibility,
        },
        "measured": {
            "lines": line_records,
            "susceptibility_per_ev": {
                "estimate": result.susceptibility.estimate,
                "lower": result.susceptibility.lower,
                "upper": result.susceptibility.upper,
            },
        },
        "spectrum": {
            "broadening_ev": BROADENING_EV,
            "frequency_ev": frequencies.tolist(),
            "exact": exact_broadened.tolist(),
            "estimate": result.broadened_estimate.tolist(),
            "lower": result.broadened_lower.tolist(),
            "upper": result.broadened_upper.tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ill-conditioned-out", type=Path,
                        default=ILL_CONDITIONED_OUT)
    args = parser.parse_args()
    for path, family in ((args.out, "determinant"),
                         (args.ill_conditioned_out, "krylov")):
        record = build_record(generators=family)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        bootstrap = record["bootstrap"]
        print(f"{path}  kappa={record['basis']['condition_number']:.4g}  "
              f"acceptance={bootstrap['replicates_succeeded']}/"
              f"{bootstrap['replicates_requested']}")


if __name__ == "__main__":
    main()
