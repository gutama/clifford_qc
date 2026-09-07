"""Unit contracts for the result-free R4a contextual restriction API."""

from __future__ import annotations

import pytest

from clifford_qc.pauli import I, X, Z
from clifford_qc.states import ket_density
from clifford_qc.subspace import (
    Generator,
    Restriction,
    compile_contextual_restriction,
    project_contextual_problem,
    select_contextual_stabilizers,
)


def test_selection_uses_packed_word_code_as_the_equal_weight_tiebreak():
    reference = ket_density(3, "000")
    hamiltonian = -1.0 * Z(3, 1) - 1.0 * Z(3, 0) - 0.5 * Z(3, 2)

    selected = select_contextual_stabilizers(hamiltonian, reference, 1)

    assert selected.stabilizers[0].word == next_word(Z(3, 0))
    assert selected.stabilizers[0].visit_index == 0


def test_selection_requires_reference_eigenstate_and_energy_lowering_sign():
    reference = ket_density(2, "00")
    hamiltonian = -2.0 * X(2, 0) + 1.5 * Z(2, 0) - 1.0 * Z(2, 1)

    selected = select_contextual_stabilizers(hamiltonian, reference, 1)

    assert selected.stabilizers[0].word == next_word(Z(2, 1))
    assert selected.stabilizers[0].eigenvalue == 1
    assert selected.visited_terms == 3
    assert selected.skipped_reference_not_eigenstate == 1
    assert selected.skipped_reference_sign_mismatch == 1


def test_selection_does_not_use_reference_tolerance_as_a_coefficient_cutoff():
    reference = ket_density(2, "00")
    hamiltonian = -1e-10 * Z(2, 0)

    selected = select_contextual_stabilizers(
        hamiltonian, reference, 1, tol=1e-9
    )

    assert selected.stabilizers[0].word == next_word(Z(2, 0))
    assert selected.stabilizers[0].coefficient == pytest.approx(-1e-10)
    assert selected.visited_terms == 1


def test_dependent_hamiltonian_words_do_not_consume_fixed_qubits():
    reference = ket_density(4, "0000")
    hamiltonian = (
        -5.0 * Z(4, 0)
        - 4.0 * Z(4, 1)
        - 3.0 * (Z(4, 0) * Z(4, 1))
        - 2.0 * Z(4, 2)
    )

    selected = select_contextual_stabilizers(hamiltonian, reference, 3)

    assert [row.word for row in selected.stabilizers] == [
        next_word(Z(4, 0)),
        next_word(Z(4, 1)),
        next_word(Z(4, 2)),
    ]
    assert selected.skipped_dependent == 1
    assert selected.visited_terms == 4


def test_signed_stabilizer_is_compiled_to_positive_z_on_the_fixed_lane():
    pytest.importorskip("stim")
    reference = ket_density(2, "10")
    hamiltonian = 2.0 * Z(2, 0) - 0.5 * Z(2, 1)

    plan = compile_contextual_restriction(hamiltonian, reference, 1)
    row = plan.selection.stabilizers[0]
    rotated = plan.restriction.rotate(row.word.to_mv())

    assert row.eigenvalue == -1
    assert rotated.is_close(-1.0 * Z(2, 0), 1e-12)
    assert plan.restriction.signs == (1,)
    assert plan.restriction.state(reference).trace() == pytest.approx(1.0)


def test_contextual_projection_exposes_what_exact_transport_must_refuse():
    pytest.importorskip("stim")
    reference = ket_density(3, "000")
    hamiltonian = -3.0 * Z(3, 0) + 0.5 * X(3, 0) - 0.2 * Z(3, 2)
    plan = compile_contextual_restriction(hamiltonian, reference, 1)
    generators = [
        Generator("I", I(3)),
        Generator("kept", Z(3, 2)),
        Generator("annihilated", X(3, 0)),
    ]

    with pytest.raises(ValueError, match="does not commute"):
        plan.restriction.transport(
            hamiltonian=hamiltonian,
            reference=reference,
            generators=(row.mv for row in generators),
        )

    contextual = project_contextual_problem(
        plan,
        hamiltonian=hamiltonian,
        reference=reference,
        generators=generators,
    )

    assert contextual.problem.n == 2
    assert contextual.hamiltonian_removed_hs_fraction > 0.0
    assert contextual.annihilated_indices == (2,)
    assert contextual.problem.reference.trace() == pytest.approx(1.0)

    shifted = project_contextual_problem(
        plan,
        hamiltonian=hamiltonian + 1_000_000.0 * I(3),
        reference=reference,
        generators=generators,
    )
    assert shifted.hamiltonian_removed_hs_fraction == pytest.approx(
        contextual.hamiltonian_removed_hs_fraction
    )
    assert shifted.problem.hamiltonian.is_close(
        contextual.problem.hamiltonian + 1_000_000.0 * I(2), 1e-9
    )


def test_contextual_projection_is_exact_when_the_hamiltonian_preserves_the_sector():
    pytest.importorskip("stim")
    reference = ket_density(3, "000")
    hamiltonian = -3.0 * Z(3, 0) - 0.2 * Z(3, 2)
    plan = compile_contextual_restriction(hamiltonian, reference, 1)

    contextual = project_contextual_problem(
        plan,
        hamiltonian=hamiltonian,
        reference=reference,
        generators=[Generator("I", I(3))],
    )

    assert contextual.hamiltonian_removed_hs_fraction == pytest.approx(0.0)


@pytest.mark.parametrize("count", [0, 3, True])
def test_invalid_fixed_qubit_counts_are_rejected(count):
    with pytest.raises(ValueError, match="count must be"):
        select_contextual_stabilizers(
            -1.0 * Z(3, 0), ket_density(3, "000"), count
        )


def test_unavailable_requested_count_is_an_explicit_failure():
    with pytest.raises(ValueError, match="requested 2.*found 1"):
        select_contextual_stabilizers(
            -1.0 * Z(3, 0), ket_density(3, "000"), 2
        )


def next_word(operator):
    code, = operator.terms
    from clifford_qc.ir import PauliWord
    return PauliWord(operator.n, code)
