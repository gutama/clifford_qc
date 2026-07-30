"""Grouped-bootstrap uncertainty for a measured A-CASE response spectrum.

The exact dimer showcase has one staggered-spin line.  Here ``S``, ``H``, and
the projected staggered-spin operator are reconstructed from one shared QWC
measurement cache, then every bootstrap replica resamples the group
histograms and reruns thresholding, the generalized eigensolve, transition
weights, susceptibility, and Lorentzian broadening.

The intervals are labelled ``heuristic``.  They are useful finite-shot
uncertainty diagnostics, not finite-sample coverage certificates.
"""

from __future__ import annotations

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

DATA = Path(__file__).with_name("data") / "wannier_hubbard_dimer.json"
SHOTS_PER_GROUP = 8_000
BOOTSTRAP_REPLICATES = 200


def main() -> None:
    model = load_effective_hamiltonian(DATA)
    rho = ExactMVBackend().state(model.reference, ())
    candidates = determinant_excitations(
        model.n, occupied_spin_orbitals(model))
    bank = MatrixElementBank(
        rho, model.hamiltonian, [identity_generator(model.n), *candidates])
    staggered = magnetization(model, 0) + (-1.0) * magnetization(model, 1)
    measurement = ResponseMeasurement(
        bank, staggered, label="staggered_spin")
    cache = measurement.measure(
        FiniteShotBackend(seed=2026), SHOTS_PER_GROUP)

    frequencies = np.linspace(0.0, 2.0, 401)
    uncertainty = bootstrap_response(
        measurement,
        cache,
        replicates=BOOTSTRAP_REPLICATES,
        seed=17,
        min_weight=1e-3,
        frequencies=frequencies,
        broadening=0.1,
    )

    print("finite-shot staggered-spin response")
    print(f"  words={len(measurement.words)}, groups={len(measurement.groups)}, "
          f"shots={cache.total_shots}")
    print(f"  evidence={uncertainty.evidence}, "
          f"certified={uncertainty.certified}")
    print(f"  usable bootstrap replicas="
          f"{uncertainty.replicates_succeeded}/"
          f"{uncertainty.replicates_requested}"
          f" (acceptance {uncertainty.acceptance_rate:.0%};"
          f" intervals below are conditional on these)")
    for item in uncertainty.lines:
        gap, weight = item.excitation_energy, item.weight
        print(f"  root {item.line.final_state}:")
        print(f"    gap    = {gap.estimate:.9f} "
              f"[{gap.lower:.9f}, {gap.upper:.9f}] eV")
        print(f"    weight = {weight.estimate:.9f} "
              f"[{weight.lower:.9f}, {weight.upper:.9f}]")
    chi = uncertainty.susceptibility
    print(f"  chi(0) = {chi.estimate:.9f} "
          f"[{chi.lower:.9f}, {chi.upper:.9f}] 1/eV")
    peak = int(np.argmax(uncertainty.broadened_estimate))
    print(f"  broadened peak near {frequencies[peak]:.3f} eV; "
          "reported grid intervals are pointwise, not simultaneous")


if __name__ == "__main__":
    main()
