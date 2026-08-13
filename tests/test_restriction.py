"""Invariants for the shared Clifford-rotate-then-fix restriction primitive.

The three oracle checks the plan requires before any restriction feeds a
benchmark are here: reference energy equality, spectrum equality on the fixed
sector, and the word-multiset bijection under a pure encoding change. Each
compares the restricted route against arithmetic that does not share code with
it -- dense matrices and eigenvalues -- because comparing one projected
quantity against another agrees with itself under a systematic error.
"""

from __future__ import annotations

import numpy as np
import pytest

from clifford_qc.dense_reference import to_matrix
from clifford_qc.ir import Program
from clifford_qc.multivector import MV
from clifford_qc.pauli import I, X, Y, Z
from clifford_qc.states import ket_density
from clifford_qc.subspace.restriction import (
    Restriction,
    restricted_sector_operators,
)

pytest.importorskip("stim")

from clifford_qc.bridges.stim_bridge import CliffordMap  # noqa: E402


def _z2_hamiltonian(n: int, fixed: int) -> MV:
    """A Hermitian operator that commutes with ``Z_fixed`` but is not trivial."""
    terms = I(n) * 0.0
    terms = terms + 0.7 * Z(n, 0)
    terms = terms + 1.3 * Z(n, fixed)
    terms = terms + 0.5 * (X(n, 0) * X(n, 1))
    terms = terms + 0.9 * (Z(n, 0) * Z(n, fixed))
    terms = terms + 0.4 * (Y(n, 0) * Y(n, 1))
    return terms


def _sector_block(operator: MV, qubit: int, sign: int) -> np.ndarray:
    """Dense block of ``operator`` on the ``Z_qubit = sign`` eigenspace."""
    n = operator.n
    matrix = to_matrix(operator)
    # Basis index bit for `qubit` under the package's little-endian packing:
    # to_matrix builds the Kronecker product with qubit 0 leftmost.
    bit = n - 1 - qubit
    want = 0 if sign == 1 else 1
    keep = [k for k in range(2 ** n) if (k >> bit) & 1 == want]
    return matrix[np.ix_(keep, keep)]


class TestTaperingWithoutRotation:
    def test_spectrum_matches_the_fixed_sector_block(self):
        n, fixed = 3, 2
        H = _z2_hamiltonian(n, fixed)
        for sign in (+1, -1):
            restriction = Restriction(
                n=n, clifford=None, fixed_qubits=(fixed,), signs=(sign,)
            )
            restricted = restriction.operator(H, require_commuting=True)
            assert restricted.n == n - 1
            got = np.linalg.eigvalsh(to_matrix(restricted))
            want = np.linalg.eigvalsh(_sector_block(H, fixed, sign))
            assert np.allclose(got, want, atol=1e-12)

    def test_reference_energy_is_preserved(self):
        n, fixed = 3, 2
        H = _z2_hamiltonian(n, fixed)
        # |001> has Z_2 = -1, so it belongs to the sign = -1 sector.
        rho = ket_density(n, "001")
        restriction = Restriction(
            n=n, clifford=None, fixed_qubits=(fixed,), signs=(-1,)
        )
        restricted_rho = restriction.state(rho)
        restricted_H = restriction.operator(H, require_commuting=True)
        before = (H * rho).trace().real
        after = (restricted_H * restricted_rho).trace().real
        assert after == pytest.approx(before, abs=1e-12)

    def test_in_sector_state_keeps_unit_trace_without_renormalization(self):
        # The Z_q partner of every surviving word merges into it, which is what
        # restores the trace; a rescale here would hide an out-of-sector state.
        restriction = Restriction(
            n=2, clifford=None, fixed_qubits=(1,), signs=(1,)
        )
        restricted = restriction.state(ket_density(2, "00"))
        assert restricted.n == 1
        assert restricted.trace().real == pytest.approx(1.0, abs=1e-12)
        assert restricted.is_close(ket_density(1, "0"), 1e-12)

    def test_removing_a_middle_qubit_repacks_the_lanes_above_it(self):
        # Fixing the *last* qubit leaves every surviving lane where it was, so
        # it cannot distinguish a correct re-pack from one that keeps original
        # qubit indices. Fixing a middle qubit is what pins the shift.
        n, fixed = 3, 1
        operator = 1.0 * Z(n, 0) + 3.0 * Z(n, fixed) + 2.0 * Z(n, 2)
        restriction = Restriction(
            n=n, clifford=None, fixed_qubits=(fixed,), signs=(1,)
        )
        restricted = restriction.operator(operator, require_commuting=True)
        assert restricted.n == 2
        # Z_0 -> Z_0, Z_2 -> Z_1, and Z_1 folds into the identity at its sign.
        assert restricted.to_labels() == {
            "II": pytest.approx(3.0), "ZI": pytest.approx(1.0),
            "IZ": pytest.approx(2.0),
        }

    def test_middle_qubit_removal_preserves_the_sector_spectrum(self):
        n, fixed = 3, 1
        operator = (
            0.7 * (X(n, 0) * X(n, 2)) + 1.3 * Z(n, fixed)
            + 0.5 * Z(n, 0) + 0.9 * (Y(n, 0) * Y(n, 2))
        )
        for sign in (+1, -1):
            restriction = Restriction(
                n=n, clifford=None, fixed_qubits=(fixed,), signs=(sign,)
            )
            restricted = restriction.operator(operator, require_commuting=True)
            assert np.allclose(
                np.linalg.eigvalsh(to_matrix(restricted)),
                np.linalg.eigvalsh(_sector_block(operator, fixed, sign)),
                atol=1e-12,
            )

    def test_wrong_declared_sign_is_rejected_not_rescaled(self):
        restriction = Restriction(
            n=2, clifford=None, fixed_qubits=(1,), signs=(-1,)
        )
        with pytest.raises(ValueError, match="not in the fixed sector"):
            restriction.state(ket_density(2, "00"))

    def test_a_word_and_its_z_partner_merge_rather_than_survive_separately(self):
        # The transport preserves products, not coefficients term by term.
        # Both words below land on the same restricted word, so pinning this
        # keeps the docstring honest about what the homomorphism does.
        restriction = Restriction(
            n=2, clifford=None, fixed_qubits=(1,), signs=(1,)
        )
        merged = restriction.operator(
            1.0 * Z(2, 0) + 3.0 * (Z(2, 0) * Z(2, 1)), require_commuting=True
        )
        assert merged.to_labels() == {"Z": pytest.approx(4.0)}
        # ...and at the other declared sign the partner subtracts instead.
        flipped = Restriction(
            n=2, clifford=None, fixed_qubits=(1,), signs=(-1,)
        ).operator(1.0 * Z(2, 0) + 3.0 * (Z(2, 0) * Z(2, 1)), require_commuting=True)
        assert flipped.to_labels() == {"Z": pytest.approx(-2.0)}

    def test_sector_check_rejects_a_complex_trace_with_the_right_real_part(self):
        # A phase error is one of the failures this primitive exists to catch,
        # so it must not be the one that slips past by having Re(tr) == 1.
        restriction = Restriction(
            n=2, clifford=None, fixed_qubits=(1,), signs=(1,)
        )
        phase_broken = ket_density(2, "00") + 0.125j * I(2)
        assert restriction.operator(phase_broken).trace().real == pytest.approx(1.0)
        with pytest.raises(ValueError, match="not in the fixed sector"):
            restriction.state(phase_broken)

    def test_non_commuting_operator_is_rejected_only_when_required(self):
        n, fixed = 3, 2
        leaky = X(n, fixed) * 1.0
        restriction = Restriction(
            n=n, clifford=None, fixed_qubits=(fixed,), signs=(1,)
        )
        with pytest.raises(ValueError, match="does not commute"):
            restriction.operator(leaky, require_commuting=True)
        # As a candidate generator the same operator is projected, not an error:
        # it moves the state to an orthogonal sector and correctly vanishes.
        assert restriction.operator(leaky).is_zero(1e-12)
        assert restriction.leakage(leaky) == pytest.approx(1.0)


class TestEncodingChange:
    def test_word_multiset_is_a_bijection_so_W_is_invariant(self):
        rng = np.random.default_rng(7)
        n = 4
        program = Program(n).clifford("H", 0).clifford("CX", 0, 1)
        program = program.clifford("CX", 1, 2).clifford("S", 3)
        restriction = Restriction.encoding(CliffordMap.from_program(program))
        codes = rng.choice(4 ** n, size=40, replace=False)
        operator = MV(n, {int(c): float(rng.normal()) for c in codes})
        rotated = restriction.operator(operator)
        assert len(rotated.terms) == len(operator.terms)
        assert restriction.n_restricted == n
        # A Clifford maps one word to one word, so the *multiset* of magnitudes
        # is preserved exactly; only which word carries each one changes.
        assert sorted(np.round(np.abs(list(rotated.terms.values())), 12).tolist()) == (
            sorted(np.round(np.abs(list(operator.terms.values())), 12).tolist())
        )

    def test_spectrum_is_invariant_under_a_pure_rotation(self):
        n = 3
        program = Program(n).clifford("H", 0).clifford("CX", 0, 2)
        restriction = Restriction.encoding(CliffordMap.from_program(program))
        H = _z2_hamiltonian(n, 2)
        rotated = restriction.operator(H, require_commuting=True)
        assert np.allclose(
            np.linalg.eigvalsh(to_matrix(rotated)),
            np.linalg.eigvalsh(to_matrix(H)),
            atol=1e-12,
        )


class TestRotateThenFix:
    def test_stabilizer_rotated_to_a_single_z_then_tapered(self):
        # The symmetry is X_2; H on qubit 2 sends X_2 -> Z_2, which is exactly
        # the two-step shape every restriction in the plan shares.
        n = 3
        H_op = 0.6 * Z(n, 0) + 0.8 * (X(n, 0) * X(n, 1)) + 1.1 * X(n, 2)
        symmetry = X(n, 2)
        assert (H_op * symmetry - symmetry * H_op).is_zero(1e-12)

        program = Program(n).clifford("H", 2)
        cmap = CliffordMap.from_program(program)
        restriction = Restriction(
            n=n, clifford=cmap, fixed_qubits=(2,), signs=(1,)
        )
        # The rotation must actually take the stabilizer to +Z on the fixed qubit.
        assert restriction.rotate(symmetry).is_close(Z(n, 2), 1e-12)

        restricted = restriction.operator(H_op, require_commuting=True)
        assert restricted.n == n - 1
        want = np.linalg.eigvalsh(
            _sector_block(restriction.rotate(H_op), 2, +1)
        )
        assert np.allclose(np.linalg.eigvalsh(to_matrix(restricted)), want, atol=1e-12)


class TestCongruentTransport:
    def test_transport_moves_all_four_objects_and_reports_leakage(self):
        n, fixed = 3, 2
        H = _z2_hamiltonian(n, fixed)
        rho = ket_density(n, "000")
        keeper = Z(n, 0) * 1.0            # commutes with Z_2, survives
        leaker = Y(n, fixed) * 1.0        # anticommutes, is annihilated
        restriction = Restriction(
            n=n, clifford=None, fixed_qubits=(fixed,), signs=(1,)
        )
        problem = restriction.transport(
            hamiltonian=H, reference=rho,
            generators=(keeper, leaker), observables=(Z(n, 1),),
        )
        assert problem.n == n - 1
        assert problem.hamiltonian.n == n - 1
        assert problem.reference.trace().real == pytest.approx(1.0, abs=1e-12)
        assert problem.observables[0].n == n - 1
        assert problem.generator_leakage[0] == pytest.approx(0.0)
        assert problem.generator_leakage[1] == pytest.approx(1.0)
        assert problem.annihilated_indices() == (1,)
        assert len(problem.surviving_generators()) == 1

    def test_identity_restriction_is_a_faithful_control_arm(self):
        n = 3
        H = _z2_hamiltonian(n, 2)
        rho = ket_density(n, "000")
        problem = Restriction.identity(n).transport(
            hamiltonian=H, reference=rho, generators=(Z(n, 0),),
        )
        assert problem.hamiltonian.is_close(H, 1e-12)
        assert problem.reference.is_close(rho, 1e-12)
        assert problem.generator_leakage == (pytest.approx(0.0),)


class TestSectorLayerTransport:
    def test_number_and_sz_survive_a_pure_encoding_change(self):
        # The JW-only sector layer is the quiet failure R2 is gated on: under a
        # rotated encoding the untransported operators still return numbers.
        n = 4
        program = Program(n).clifford("CX", 0, 1).clifford("CX", 2, 3)
        restriction = Restriction.encoding(CliffordMap.from_program(program))
        number, sz = restricted_sector_operators(restriction)
        from clifford_qc.fermion import total_number_op, total_sz_op

        assert np.allclose(
            np.linalg.eigvalsh(to_matrix(number)),
            np.linalg.eigvalsh(to_matrix(total_number_op(n))),
            atol=1e-12,
        )
        assert np.allclose(
            np.linalg.eigvalsh(to_matrix(sz)),
            np.linalg.eigvalsh(to_matrix(total_sz_op(n))),
            atol=1e-12,
        )
        # And the transported number operator is genuinely different from the
        # one the JW-specific helper would have handed a non-JW caller.
        assert not number.is_close(total_number_op(n), 1e-9)


class TestConstructionContracts:
    def test_rejects_inconsistent_or_degenerate_restrictions(self):
        with pytest.raises(ValueError, match="equal length"):
            Restriction(n=3, clifford=None, fixed_qubits=(0, 1), signs=(1,))
        with pytest.raises(ValueError, match="distinct"):
            Restriction(n=3, clifford=None, fixed_qubits=(1, 1), signs=(1, 1))
        with pytest.raises(ValueError, match="out of range"):
            Restriction(n=2, clifford=None, fixed_qubits=(5,), signs=(1,))
        with pytest.raises(ValueError, match=r"\+1 or -1"):
            Restriction(n=3, clifford=None, fixed_qubits=(0,), signs=(0,))
        with pytest.raises(ValueError, match="nothing would remain"):
            Restriction(n=2, clifford=None, fixed_qubits=(0, 1), signs=(1, 1))
