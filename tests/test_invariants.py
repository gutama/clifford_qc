"""Pytest port of the rich invariant suite from ``clifford_qc.verify``.

This file is the canonical home of the invariants; ``python -m
clifford_qc.verify`` remains as a dependency-light smoke entry point.
"""

import math

import numpy as np
import pytest

from clifford_qc import (
    MV, I, X, Y, Z, P, comm, anticomm, tensor, gamma, blade_from_mask,
    c_op, cdag_op, number_op, number_op_pauli,
    H, S, T, RX, RY, RZ, CNOT, CZ, SWAP, TOFFOLI,
    ket_density, evolve, expectation, probability, purity,
    partial_trace, partial_transpose, z_projectors, measure,
    computational_probabilities, bell_density, ghz_density,
    depolarizing, dephasing, amplitude_damping, check_kraus, apply_channel,
    to_matrix, from_matrix, negativity, vn_entropy,
    trotter_unitary, trotter2_unitary, expm_taylor, expm_matrix,
)

N = 3


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(7)


@pytest.fixture(scope="module")
def random_mvs(rng):
    def rand_mv():
        return MV(N, {int(rng.integers(0, 4 ** N)): complex(rng.normal(), rng.normal())
                      for _ in range(8)})
    return rand_mv(), rand_mv(), rand_mv()


class TestAlgebraKernel:
    def test_associativity(self, random_mvs):
        A, B, C = random_mvs
        assert ((A * B) * C).is_close(A * (B * C), 1e-8)

    def test_dagger_antihomomorphism(self, random_mvs):
        A, B, _ = random_mvs
        assert (A * B).dagger().is_close(B.dagger() * A.dagger(), 1e-8)

    def test_trace_cyclicity(self, random_mvs):
        A, B, C = random_mvs
        assert abs((A * B * C).trace() - (B * C * A).trace()) < 1e-8

    def test_local_pauli_table(self):
        assert (X(N, 1) * Y(N, 1)).is_close(1j * Z(N, 1))

    def test_different_qubit_paulis_commute(self):
        assert comm(X(N, 0), X(N, 1)).norm_hs() < 1e-12

    def test_label_roundtrip(self):
        assert P("XIZ").to_labels() == {"XIZ": 1 + 0j}


class TestCliffordJordanWigner:
    def test_gamma_squares_to_one(self):
        assert all((g * g).is_close(I(N)) for g in (gamma(N, i) for i in range(2 * N)))

    def test_gamma_anticommute(self):
        gs = [gamma(N, i) for i in range(2 * N)]
        assert all(anticomm(gs[i], gs[j]).norm_hs() < 1e-12
                   for i in range(2 * N) for j in range(i + 1, 2 * N))

    def test_gamma_grade_one(self):
        assert all(gamma(N, i).grades() == {1} for i in range(2 * N))

    def test_z_as_bivector(self):
        assert Z(N, 1).is_close(-1j * gamma(N, 2) * gamma(N, 3))
        assert Z(N, 1).grades() == {2}

    def test_blade_mask_decode_roundtrip(self):
        for mask in range(1 << (2 * N)):
            blade = blade_from_mask(N, mask)
            assert blade.blade_mask(next(iter(blade.terms))) == mask


class TestFermionWitt:
    def test_car_and_nilpotency(self):
        cs = [c_op(N, j) for j in range(N)]
        cds = [cdag_op(N, j) for j in range(N)]
        for i in range(N):
            for j in range(N):
                target = I(N) if i == j else MV(N)
                assert (cs[i] * cds[j] + cds[j] * cs[i]).is_close(target, 1e-12)
                assert (cs[i] * cs[j] + cs[j] * cs[i]).norm_hs() < 1e-12
        assert all((c * c).norm_hs() < 1e-12 for c in cs)

    def test_number_operator(self):
        assert all(number_op(N, j).is_close(number_op_pauli(N, j), 1e-12) for j in range(N))


class TestMatrixRepresentation:
    def test_product_homomorphism(self, random_mvs):
        A, B, _ = random_mvs
        assert np.allclose(to_matrix(A * B), to_matrix(A) @ to_matrix(B), atol=1e-8)

    def test_trace_normalization(self, random_mvs):
        A, _, _ = random_mvs
        assert abs(np.trace(to_matrix(A)) - A.trace()) < 1e-8

    def test_dagger_is_conjugate_transpose(self, random_mvs):
        A, _, _ = random_mvs
        assert np.allclose(to_matrix(A.dagger()), to_matrix(A).conj().T, atol=1e-8)

    def test_matrix_roundtrip(self, random_mvs):
        A, _, _ = random_mvs
        assert from_matrix(to_matrix(A), N).is_close(A, 1e-8)

    def test_tensor_matches_kron(self):
        assert np.allclose(to_matrix(tensor(X(1, 0), Z(1, 0))),
                           np.kron(to_matrix(X(1, 0)), to_matrix(Z(1, 0))))


class TestGatesAndRotors:
    def test_standard_gates_unitary(self):
        gates = [H(N, 0), S(N, 1), T(N, 2), RX(N, 0, 0.7), RY(N, 1, 1.1), RZ(N, 2, 2.3),
                 CNOT(N, 0, 1), CZ(N, 1, 2), SWAP(N, 0, 2), TOFFOLI(N, 0, 1, 2)]
        assert all(U.is_unitary(1e-9) for U in gates)

    def test_rotor_identity(self):
        assert RZ(N, 0, 1.3).is_close(math.cos(0.65) * I(N) - 1j * math.sin(0.65) * Z(N, 0))

    def test_gate_relations(self):
        assert (H(N, 0) * H(N, 0)).is_close(I(N))
        assert (T(N, 2) * T(N, 2)).is_close(S(N, 2))
        assert (S(N, 2) * S(N, 2)).is_close(Z(N, 2))

    def test_cnot_preserves_pauli_words(self):
        U = CNOT(N, 0, 1)
        for s in ("XII", "YII", "ZII", "IXI", "IYI", "IZI"):
            assert (U * P(s) * U.dagger()).nnz() == 1


class TestStatesAndEntanglement:
    def test_density_validity(self):
        rho0 = ket_density(2, "00")
        bell = bell_density()
        assert rho0.is_density() and bell.is_density()
        assert abs(purity(bell) - 1) < 1e-9

    def test_bell_pauli_expansion(self):
        target = 0.25 * (I(2) + X(2, 0) * X(2, 1) - Y(2, 0) * Y(2, 1) + Z(2, 0) * Z(2, 1))
        assert bell_density().is_close(target, 1e-9)

    def test_bell_reduced_state_maximally_mixed(self):
        red = partial_trace(bell_density(), {1})
        assert abs(red.trace() - 1) < 1e-9
        assert abs(purity(red) - 0.5) < 1e-9

    def test_ppt_witness(self):
        ev = np.linalg.eigvalsh(to_matrix(partial_transpose(bell_density(), {1})))
        assert abs(min(ev) + 0.5) < 1e-9

    def test_negativity_and_entropy(self):
        bell = bell_density()
        assert abs(negativity(bell, {1}) - 0.5) < 1e-9
        assert abs(vn_entropy(partial_trace(bell, {1})) - 1.0) < 1e-9

    def test_bell_z_measurement(self):
        meas = measure(bell_density(), z_projectors(2, 0), check_projectors=True)
        assert abs(meas[0][0] - 0.5) < 1e-9 and abs(meas[1][0] - 0.5) < 1e-9

    def test_ghz(self):
        ghz = ghz_density(3)
        assert abs(purity(ghz) - 1) < 1e-9
        assert abs(purity(partial_trace(ghz, {2})) - 0.5) < 1e-9


class TestProtocols:
    def test_chsh_tsirelson(self):
        bell = bell_density()
        A0, A1 = Z(2, 0), X(2, 0)
        B0 = (1 / math.sqrt(2)) * (Z(2, 1) + X(2, 1))
        B1 = (1 / math.sqrt(2)) * (Z(2, 1) - X(2, 1))
        chsh = sum(s * expectation(bell, Aa * Bb).real
                   for s, Aa, Bb in [(1, A0, B0), (1, A0, B1), (1, A1, B0), (-1, A1, B1)])
        assert abs(chsh - 2 * math.sqrt(2)) < 1e-9

    def test_grover_two_qubit(self):
        Hs = H(2, 0) * H(2, 1)
        Gop = (Hs * (2 * ket_density(2, "00") - I(2)) * Hs) * CZ(2, 0, 1)
        grover = evolve(evolve(ket_density(2, "00"), Hs), Gop)
        assert abs(probability(grover, ket_density(2, "11")) - 1) < 1e-9
        assert abs(sum(computational_probabilities(grover).values()) - 1.0) < 1e-9


class TestChannelsAndDynamics:
    def test_kraus_completeness(self):
        for ks in [depolarizing(2, 0, 0.3), dephasing(2, 0, 0.2), amplitude_damping(2, 0, 0.4)]:
            assert check_kraus(ks)

    def test_depolarizing_fixed_point(self):
        noisy = bell_density()
        for _ in range(60):
            noisy = apply_channel(noisy, depolarizing(2, 0, 0.5))
            noisy = apply_channel(noisy, depolarizing(2, 1, 0.5))
        assert abs(purity(noisy) - 0.25) < 1e-6

    def test_trotter_and_expm(self):
        H_terms = [(1.0, X(2, 0) * X(2, 1)), (0.5, Z(2, 0)), (0.5, Z(2, 1))]
        Hmat = sum(c * to_matrix(Pw) for c, Pw in H_terms)
        t = 0.9
        w, V = np.linalg.eigh(Hmat)
        Uex = V @ np.diag(np.exp(-1j * w * t)) @ V.conj().T
        Hmv = sum((c * Pw for c, Pw in H_terms), MV(2))
        assert np.linalg.norm(to_matrix(trotter_unitary(H_terms, t, 400)) - Uex) < 2e-2
        assert np.linalg.norm(to_matrix(trotter2_unitary(H_terms, t, 40)) - Uex) < 2e-3
        assert np.linalg.norm(to_matrix(expm_taylor((-1j * t) * Hmv)) - Uex) < 1e-9
        assert np.linalg.norm(to_matrix(expm_matrix((-1j * t) * Hmv)) - Uex) < 1e-9

    def test_expm_handles_a_defective_jordan_block(self):
        # N = X + iY is nilpotent but not diagonalizable, so exp(N)=I+N.
        N = X(1, 0) + 1j * Y(1, 0)
        assert np.allclose(to_matrix(expm_matrix(N)),
                           np.eye(2) + to_matrix(N), atol=1e-12)


class TestStructure:
    def test_sparsity(self):
        assert bell_density().nnz() == 4
        assert ghz_density(3).nnz() == 8
        assert TOFFOLI(3, 0, 1, 2).nnz() == 8

    def test_grade_structure(self):
        assert sorted(RZ(2, 0, 0.8).grades()) == [0, 2]
        assert sorted(CNOT(2, 0, 1).grades()) == [0, 1, 2, 3]
