"""OpenFermion converters (Phase 2).

``QubitOperator`` maps losslessly to/from ``MV``/``PauliSum`` — both are
sparse Pauli-word sums. ``FermionOperator`` lowers one-way through the
package's own Jordan-Wigner Witt operators ``c_op``/``cdag_op``, which use
the same convention as OpenFermion's ``jordan_wigner`` transform
(annihilator ``a_j = Z_0..Z_{j-1} (X_j + iY_j)/2``).
"""

from __future__ import annotations

from openfermion import FermionOperator, QubitOperator

from ..fermion import c_op, cdag_op
from ..ir import PauliSum
from ..multivector import MV, TOL, label_to_code
from ..pauli import I


def _qubit_operator_n(op: QubitOperator, n: int | None) -> int:
    max_index = max((idx for term in op.terms for idx, _ in term), default=-1)
    if n is None:
        n = max_index + 1
    if n <= max_index:
        raise ValueError(f"operator touches qubit {max_index}, but n={n}")
    return n


def qubit_operator_to_mv(op: QubitOperator, n: int | None = None) -> MV:
    n = _qubit_operator_n(op, n)
    terms: dict[int, complex] = {}
    for term, coeff in op.terms.items():
        label = ["I"] * n
        for idx, letter in term:
            label[idx] = letter
        code = label_to_code("".join(label), n)
        terms[code] = terms.get(code, 0j) + complex(coeff)
    return MV(n, terms)


def mv_to_qubit_operator(A: MV, tol: float = TOL) -> QubitOperator:
    out = QubitOperator()
    for label, coeff in A.to_labels(tol=tol).items():
        term = tuple((j, letter) for j, letter in enumerate(label) if letter != "I")
        out += QubitOperator(term, complex(coeff))
    return out


def qubit_operator_to_pauli_sum(op: QubitOperator, n: int | None = None) -> PauliSum:
    return PauliSum.from_mv(qubit_operator_to_mv(op, n))


def pauli_sum_to_qubit_operator(ps: PauliSum, tol: float = TOL) -> QubitOperator:
    return mv_to_qubit_operator(ps.to_mv(), tol)


def fermion_operator_to_mv(op: FermionOperator, n: int | None = None) -> MV:
    """Jordan-Wigner lowering of a FermionOperator into Cl(2n,C)."""
    max_mode = max((idx for term in op.terms for idx, _ in term), default=-1)
    if n is None:
        n = max_mode + 1
    if n <= max_mode:
        raise ValueError(f"operator touches mode {max_mode}, but n={n}")
    out = MV(n)
    for term, coeff in op.terms.items():
        prod = complex(coeff) * I(n)
        for idx, dagger in term:
            prod = prod * (cdag_op(n, idx) if dagger else c_op(n, idx))
        out = out + prod
    return out
