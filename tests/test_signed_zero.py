"""Stored coefficients carry no negative zeros, on every supported interpreter.

Python 3.14 changed mixed real/complex arithmetic: ``0.0 + z`` and ``-1 * z``
no longer promote the real operand to ``complex`` first, so a ``-0.0``
component that every earlier interpreter rounded to ``+0.0`` now survives.
``(1j*Y)*X`` then stored ``1-0j`` where 3.13 stored ``1+0j``. The two compare
equal, which is why nothing numerical noticed -- but record digests, packed
rows and the ``float.hex`` comparisons in the replay checkers read bytes, and
ten replay tests failed on 3.14 for that reason alone.

The invariant pinned here is the one that makes the bytes interpreter
independent: an ``MV`` never stores a negative zero, whatever arithmetic built
it. It is checked on the bit pattern, not with ``==``.
"""

import math

import numpy as np
import pytest

from clifford_qc import MV, X, Y, Z, PauliSum, c_op, cdag_op
from clifford_qc.gates import CNOT, H, S, rotor
from clifford_qc.pauli import tensor
from clifford_qc.states import evolve, ket_density, partial_trace, partial_transpose
from clifford_qc.subspace.ga_restriction import reference_action


def negative_zeros(values) -> int:
    """Components whose bit pattern is ``-0.0``."""
    return sum(
        part == 0.0 and math.copysign(1.0, part) < 0.0
        for value in values
        for part in (complex(value).real, complex(value).imag)
    )


# Exact units and halves rather than Gaussian draws: a negative zero only
# appears where a component is *exactly* zero, which random reals never are.
UNITS = (1, -1, 1j, -1j, 0.5, -0.5, 0.5j, -0.5j)


def random_mv(rng, n=3, terms=6):
    return MV(n, {int(rng.integers(0, 4 ** n)): UNITS[int(rng.integers(0, len(UNITS)))]
                  for _ in range(terms)})


def test_the_product_that_differed_between_interpreters():
    """``(1j*Y)*X``: the coefficient is ``(1j)(1)(-1j)``, whose imaginary part is -0.0."""
    assert negative_zeros([1j * 1 * -1j]) == 1  # the raw product really carries one
    product = (1j * Y(1, 0)) * X(1, 0)
    (coefficient,) = product.terms.values()
    assert coefficient == 1
    assert (coefficient.real.hex(), coefficient.imag.hex()) == ("0x1.0000000000000p+0", "0x0.0p+0")


def test_the_constructor_normalizes_what_it_is_handed():
    assert negative_zeros(MV(1, {1: complex(0.5, -0.0)}).terms.values()) == 0
    assert negative_zeros(MV(1, {1: complex(-0.0, 0.5)}).terms.values()) == 0
    # ``PauliSum`` is the other public coefficient container.
    assert negative_zeros(PauliSum(1, {1: complex(0.5, -0.0)}).terms.values()) == 0
    assert negative_zeros((-1 * PauliSum(1, {1: 1.0})).terms.values()) == 0


@pytest.mark.parametrize("seed", range(20))
def test_no_operation_stores_a_negative_zero(seed):
    rng = np.random.default_rng(seed)
    a, b = random_mv(rng), random_mv(rng)
    results = {
        "product": a * b,
        "sum": a + b,
        "difference": a - b,
        "negation": -a,
        "real scaling": -1 * a,
        "right scaling": a * -1.0,
        "division": a / -2,
        "adjoint": a.dagger(),
        "reversion": a.reverse(),
        "wedge": a ^ b,
        "left contraction": a.lcontract(b),
        "right contraction": a.rcontract(b),
        "grade part": (a * b).grade(2),
        "tensor": tensor(a, b),
        "partial transpose": partial_transpose(a * b, {0, 2}),
        "partial trace": partial_trace(a * b, {1}),
    }
    for name, value in results.items():
        assert negative_zeros(value.terms.values()) == 0, name


def test_gates_states_and_fermionic_operators_store_no_negative_zero():
    n = 3
    operators = [
        H(n, 0), S(n, 1), S(n, 1).dagger(), CNOT(n, 0, 2),
        rotor(X(n, 0) * Y(n, 1) * Z(n, 2), -0.7),
        cdag_op(n, 0) * c_op(n, 2) - cdag_op(n, 2) * c_op(n, 0),
    ]
    state = ket_density(n, "010")
    for unitary in operators[:5]:
        state = evolve(state, unitary)
    for value in (*operators, state):
        assert negative_zeros(value.terms.values()) == 0


def test_reference_action_amplitudes_carry_no_negative_zero():
    """A plain ``dict`` of amplitudes, so the constructor is not there to help."""
    operator = (1j * Y(2, 0)) * X(2, 0) + Y(2, 1) * (1j * X(2, 1))
    for index in range(4):
        assert negative_zeros(reference_action(operator, index).values()) == 0


def test_the_independent_more_oracle_agrees_bit_for_bit():
    """Its own letter product, compared with the kernel through ``float.hex``.

    ``check_phase17_more`` makes exactly this comparison against the committed
    record, so a sign-of-zero difference between the two is a replay failure.
    """
    from benchmarks.phase17_more_oracle import product

    left = {2: 1j}                      # 1j * Y
    right = {1: 1 + 0j}                 # X
    oracle = product(left, right, 1)
    kernel = (MV(1, left) * MV(1, right)).terms
    assert negative_zeros(oracle.values()) == 0

    def as_hex(terms):
        return {code: (value.real.hex(), value.imag.hex()) for code, value in terms.items()}

    assert as_hex(oracle) == as_hex(kernel)
