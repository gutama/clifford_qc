"""Regenerate the paper's finite-shot nonlinear-response record.

The record is deliberately separate from the figure.  It contains the full
replica accounting and the fixed seeds needed to reproduce the reported
percentile intervals.  The intervals remain heuristic and conditional on
replicas that preserve the thresholded rank and ordered-root identity.
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
    occupied_spin_orbitals,
)
from clifford_qc.subspace.response import broaden_response

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "examples" / "data" / "wannier_hubbard_dimer.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "response_bootstrap.json"

SHOTS_PER_GROUP = 8_000
BOOTSTRAP_REPLICATES = 200
MEASUREMENT_SEED = 2026
BOOTSTRAP_SEED = 17
BROADENING_EV = 0.1


def build_record() -> dict:
    model = load_effective_hamiltonian(MODEL)
    rho = ExactMVBackend().state(model.reference, ())
    candidates = determinant_excitations(
        model.n, occupied_spin_orbitals(model))
    bank = MatrixElementBank(
        rho, model.hamiltonian, [identity_generator(model.n), *candidates])
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

    return {
        "schema": "clifford_qc.acase_response_bootstrap.v1",
        "source_main_commit": "4b1c636953e4ebe9aa7541d4260cfe95aa18674e",
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
    args = parser.parse_args()
    record = build_record()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
