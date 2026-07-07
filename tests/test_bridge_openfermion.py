"""OpenFermion converters: QubitOperator/FermionOperator <-> MV."""

import numpy as np
import pytest

openfermion = pytest.importorskip("openfermion")
from openfermion import FermionOperator, QubitOperator, jordan_wigner
from openfermion.linalg import get_sparse_operator

from clifford_qc import MV, P, to_matrix
from clifford_qc.bridges.openfermion_bridge import (
    fermion_operator_to_mv, mv_to_qubit_operator, pauli_sum_to_qubit_operator,
    qubit_operator_to_mv, qubit_operator_to_pauli_sum,
)


class TestQubitOperator:
    def test_simple_roundtrip(self):
        op = QubitOperator("X0 Z2", 0.5) + QubitOperator("Y1", -1.5j) + QubitOperator("", 2.0)
        A = qubit_operator_to_mv(op)
        assert A.n == 3
        assert A.to_labels() == {"III": 2.0, "XIZ": 0.5, "IYI": -1.5j}
        assert mv_to_qubit_operator(A) == op

    def test_random_roundtrip(self):
        rng = np.random.default_rng(3)
        A = MV(3, {int(rng.integers(0, 64)): complex(rng.normal(), rng.normal())
                   for _ in range(10)})
        assert qubit_operator_to_mv(mv_to_qubit_operator(A), 3).is_close(A, 1e-12)

    def test_pauli_sum_variant(self):
        op = QubitOperator("X0 X1", 1.0) + QubitOperator("Z0", 0.5)
        ps = qubit_operator_to_pauli_sum(op)
        assert ps.to_labels() == {"XX": 1.0, "ZI": 0.5}
        assert pauli_sum_to_qubit_operator(ps) == op

    def test_explicit_n_padding(self):
        A = qubit_operator_to_mv(QubitOperator("Z0", 1.0), n=4)
        assert A.to_labels() == {"ZIII": 1.0}
        with pytest.raises(ValueError):
            qubit_operator_to_mv(QubitOperator("Z3", 1.0), n=2)

    def test_matrix_agreement(self):
        op = QubitOperator("X0 Y1", 0.3) + QubitOperator("Z0 Z2", -0.7)
        A = qubit_operator_to_mv(op, 3)
        dense = get_sparse_operator(op, n_qubits=3).toarray()
        assert np.allclose(to_matrix(A), dense, atol=1e-12)


class TestFermionOperator:
    def test_number_operator(self):
        n_op = fermion_operator_to_mv(FermionOperator("1^ 1"), n=2)
        assert n_op.is_close(0.5 * (P("II") - P("IZ")), 1e-12)

    def test_matches_openfermion_jordan_wigner(self):
        ferm = (FermionOperator("0^ 1", 0.5) + FermionOperator("1^ 0", 0.5)
                + FermionOperator("2^ 2", 1.0) + FermionOperator("0^ 1^ 2 0", 0.25))
        ours = fermion_operator_to_mv(ferm, n=3)
        theirs = qubit_operator_to_mv(jordan_wigner(ferm), n=3)
        assert ours.is_close(theirs, 1e-12)

    def test_car_via_conversion(self):
        c0 = fermion_operator_to_mv(FermionOperator("0"), n=2)
        c1d = fermion_operator_to_mv(FermionOperator("1^"), n=2)
        assert (c0 * c1d + c1d * c0).norm_hs() < 1e-12
        c0d = fermion_operator_to_mv(FermionOperator("0^"), n=2)
        assert (c0 * c0d + c0d * c0).is_close(P("II"), 1e-12)
