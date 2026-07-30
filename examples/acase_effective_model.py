"""Smallest complete DFT/Wannier/embedding -> A-CASE showcase.

The input is a versioned effective-Hamiltonian JSON record.  Its one-body
matrix is what a Wannier/downfolding step hands over; ``onsite_u`` is the local
interaction supplied by cRPA, DMFT, or another embedding model.  The bundled
record is synthetic and canonical, not a claimed DFT calculation.

The two-site Hubbard dimer is only four qubits, but it is not trivial: hopping
competes with onsite repulsion, the ground state is a correlated singlet,
double occupancy is suppressed, and staggered-spin response has a resolved
excitation.  Its four-dimensional ``(N=2, Sz=0)`` sector gives an independent
exact oracle for every result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from clifford_qc.backends import ExactMVBackend
from clifford_qc.models.effective import load_effective_hamiltonian
from clifford_qc.models.observables import (double_occupancy, magnetization,
                                            spin_correlation,
                                            total_spin_squared)
from clifford_qc.sparse import sparse_ground_in_sector
from clifford_qc.subspace import (MatrixElementBank, determinant_excitations,
                                  identity_generator, lehmann_spectrum,
                                  occupied_spin_orbitals, run_acase,
                                  static_susceptibility)


def run(path: Path) -> None:
    model = load_effective_hamiltonian(path)
    unit = model.metadata["energy_unit"]
    n_electrons = model.metadata["n_electrons"]
    sz = model.metadata["sz"]
    occupied = occupied_spin_orbitals(model)
    rho = ExactMVBackend().state(model.reference, ())
    candidates = determinant_excitations(model.n, occupied, max_rank=2)

    exact_values, _ = sparse_ground_in_sector(
        model.hamiltonian, n_electrons=n_electrons, sz=sz, k=4)
    adaptive = run_acase(
        rho, model.hamiltonian, candidates, max_size=len(candidates),
        leakage_tol=1e-10, exact_ground_energy=float(exact_values[0]))

    # The complete determinant-response basis supplies every state in this
    # four-dimensional sector.  Adaptive ground-state growth is reported
    # separately above; response needs the low-lying roots together.
    bank = MatrixElementBank(
        rho, model.hamiltonian,
        [identity_generator(model.n), *candidates])
    spectrum = bank.solve()

    double = spectrum.expectation(double_occupancy(model))
    spin = spectrum.expectation(spin_correlation(model, 0, 1))
    spin2 = spectrum.expectation(total_spin_squared(model))
    staggered = (magnetization(model, 0, axis="z")
                 + (-1.0) * magnetization(model, 1, axis="z"))
    lines = lehmann_spectrum(spectrum, staggered, min_weight=1e-12)

    print(f"input: {path}")
    print(f"model: {model.name}")
    print(f"mapping: {model.metadata['sites']} Wannier sites -> {model.n} qubits; "
          f"sector (N={n_electrons}, Sz={sz:+g})")
    print(f"reference occupations: {occupied}")
    print(f"exact sector E0: {exact_values[0]:+.9f} {unit}")
    print(f"adaptive A-CASE E0: {adaptive.result.ground_energy:+.9f} {unit}  "
          f"error={adaptive.result.ground_energy - exact_values[0]:+.2e}")
    print(f"selected basis: {', '.join(adaptive.labels)}")
    print(f"complete response basis: M={len(spectrum.basis_labels)}, "
          f"rank={spectrum.effective_rank}, kappa(S)={spectrum.condition_number:.3g}, "
          f"words={spectrum.resources['word_universe']}")
    print("ground-state coefficients:")
    for label, coefficient in zip(spectrum.basis_labels, spectrum.ritz_vector(0)):
        if abs(coefficient) > 1e-10:
            print(f"  {label:16s} {coefficient.real:+.9f}{coefficient.imag:+.9f}j")
    print("correlations:")
    print(f"  average double occupancy = {double:.9f}")
    print(f"  <S0.S1>                  = {spin:.9f}")
    print(f"  <S^2>                    = {spin2:.3e}")
    print("staggered-spin Lehmann response:")
    for line in lines:
        print(f"  root {line.final_state}: omega={line.excitation_energy:.9f} {unit}, "
              f"weight={line.weight:.9f}")
    print(f"  chi(0) = {static_susceptibility(lines):.9f} 1/{unit}")


def main() -> None:
    default = Path(__file__).with_name("data") / "wannier_hubbard_dimer.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=default)
    args = parser.parse_args()
    run(args.input)


if __name__ == "__main__":
    main()
