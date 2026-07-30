"""End-to-end checks for the downfolded effective-Hamiltonian boundary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from clifford_qc.backends import ExactMVBackend, SectorStatevectorBackend
from clifford_qc.models.effective import (EFFECTIVE_HAMILTONIAN_SCHEMA,
                                          effective_hamiltonian,
                                          load_effective_hamiltonian)
from clifford_qc.models.lattice import hubbard
from clifford_qc.models.observables import (magnetization,
                                            total_spin_squared)
from clifford_qc.subspace import (MatrixElementBank, broaden_response,
                                  determinant_excitations, identity_generator,
                                  lehmann_spectrum, occupied_spin_orbitals,
                                  run_acase, static_susceptibility)

DATA = Path(__file__).parents[1] / "examples" / "data" / "wannier_hubbard_dimer.json"


def test_bundled_dimer_matches_the_native_hubbard_hamiltonian():
    imported = load_effective_hamiltonian(DATA)
    native = hubbard(2, t=1.0, U=4.0, mu=0.0)
    assert (imported.hamiltonian.to_mv()
            - native.hamiltonian.to_mv()).is_zero(1e-12)
    assert imported.metadata["n_electrons"] == 2
    assert imported.metadata["sz"] == pytest.approx(0.0)
    assert imported.metadata["energy_unit"] == "eV"


def test_complex_hermitian_one_body_matrix_is_supported():
    payload = {
        "schema": EFFECTIVE_HAMILTONIAN_SCHEMA,
        "one_body": [[0.0, [0.0, 1.0]], [[0.0, -1.0], 0.0]],
        "onsite_u": 2.0,
        "reference_occupied_spin_orbitals": [0, 3],
    }
    model = effective_hamiltonian(payload)
    assert model.hamiltonian.is_hermitian()
    assert any(isinstance(value, list)
               for row in model.metadata["one_body"] for value in row)


@pytest.mark.parametrize("update, message", [
    ({"schema": "unknown"}, "schema"),
    ({"one_body": [[0.0, 1.0], [0.0, 0.0]]}, "Hermitian"),
    ({"onsite_u": [4.0]}, "onsite_u"),
    ({"reference_occupied_spin_orbitals": [0, 0]}, "unique"),
    ({"reference_occupied_spin_orbitals": [0.5, 3]}, "integers"),
    ({"sector": {"n_electrons": 3}}, "disagrees"),
    ({"source": "not provenance metadata"}, "mapping"),
])
def test_effective_input_rejects_ambiguous_or_inconsistent_records(update, message):
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    payload.update(update)
    with pytest.raises((TypeError, ValueError), match=message):
        effective_hamiltonian(payload)


def test_dimer_is_a_complete_energy_state_correlation_response_showcase():
    model = load_effective_hamiltonian(DATA)
    rho = ExactMVBackend().state(model.reference, ())
    occupied = occupied_spin_orbitals(model)
    candidates = determinant_excitations(model.n, occupied)
    values, _ = SectorStatevectorBackend(
        model.n, n_electrons=2, sz=0.0).ground_state(
            model.hamiltonian, k=4, method="dense")

    adaptive = run_acase(
        rho, model.hamiltonian, candidates, max_size=len(candidates),
        leakage_tol=1e-10, exact_ground_energy=float(values[0]))
    assert adaptive.result.ground_energy == pytest.approx(values[0], abs=1e-10)
    assert adaptive.result.ground_energy == pytest.approx(
        0.5 * (4.0 - np.sqrt(4.0 ** 2 + 16.0)), abs=1e-10)

    bank = MatrixElementBank(
        rho, model.hamiltonian, [identity_generator(model.n), *candidates])
    result = bank.solve()
    assert result.effective_rank == 4
    assert np.allclose(result.energies, values, atol=1e-10)
    assert result.condition_number == pytest.approx(1.0, abs=1e-10)

    spin2 = total_spin_squared(model)
    assert result.expectation(spin2) == pytest.approx(0.0, abs=1e-10)
    staggered = (magnetization(model, 0)
                 + (-1.0) * magnetization(model, 1))
    lines = lehmann_spectrum(result, staggered, min_weight=1e-12)
    assert lines
    assert all(line.excitation_energy > 0.0 and line.weight > 0.0
               for line in lines)
    susceptibility = static_susceptibility(lines)
    assert susceptibility > 0.0
    # Lehmann completeness: on the full sector, inelastic response weight is
    # exactly the variance of the observable in the ground state.
    observable = staggered.to_mv()
    variance = (result.expectation(observable * observable)
                - abs(result.transition(staggered, 0, 0)) ** 2)
    assert sum(line.weight for line in lines) == pytest.approx(
        variance, abs=1e-10)
    omega = np.linspace(0.0, 8.0, 100)
    broadened = broaden_response(lines, omega, broadening=0.1)
    assert broadened.shape == omega.shape
    assert np.all(broadened >= 0.0)
