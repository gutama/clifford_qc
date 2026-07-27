"""Gate constructors must reject inputs for which their closed form is false.

Both helpers here are exact only under a precondition, and both used to accept
violations silently at their default settings -- returning a multivector that
is neither the intended operator nor unitary, with nothing to distinguish it
from a valid result. A silently invalid propagator is worse than an exception,
so the preconditions are checked by default and these tests pin that.
"""

import math

import pytest

from clifford_qc import I, MV
from clifford_qc.gates import (
    CNOT, CZ, TOFFOLI, H, RX, RY, RZ, controlled, rotor, trotter_unitary,
)
from clifford_qc.pauli import X, Y, Z


# ------------------------------------------------------------------- rotor


def test_rotor_rejects_generators_that_are_not_involutions():
    """``P^2 = 1`` is what resums the series; without it the formula is false."""
    n = 1
    bad = X(n, 0) + Z(n, 0)                      # P^2 = 2I
    assert not (bad * bad).is_close(I(n))
    with pytest.raises(ValueError, match="P\\^2 = 1"):
        rotor(bad, 0.7)
    # and the value it used to return really was not unitary
    unchecked = rotor(bad, 0.7, check_word=False)
    assert not unchecked.is_unitary()
    assert (unchecked.dagger() * unchecked).scalar_part().real == pytest.approx(
        1 + math.sin(0.35) ** 2, abs=1e-12)


def test_rotor_rejects_a_scaled_word_despite_it_being_a_single_term():
    """Single-term is not the criterion: ``2X`` has one term and ``P^2 = 4``."""
    two_x = 2 * X(1, 0)
    assert two_x.nnz() == 1
    with pytest.raises(ValueError, match="P\\^2 = 1"):
        rotor(two_x, 0.4)


def test_rotor_accepts_a_multi_term_involution():
    """The other direction: ``H = (X+Z)/sqrt(2)`` is a legitimate generator.

    Guards the fix against over-tightening -- a single-Pauli-word test would
    reject this, even though the closed form is exact and unitary for it.
    """
    n = 1
    h = H(n, 0)
    assert h.nnz() == 2 and (h * h).is_close(I(n))
    assert rotor(h, 0.7).is_unitary()


@pytest.mark.parametrize("gen", ["X", "Y", "Z", "I"])
def test_rotor_accepts_every_pauli_generator(gen):
    n = 2
    P = {"X": X(n, 0), "Y": Y(n, 0), "Z": Z(n, 0), "I": I(n)}[gen]
    assert rotor(P, 0.9).is_unitary()


def test_named_rotations_still_build_and_are_unitary():
    for gate in (RX, RY, RZ):
        assert gate(2, 1, 0.6).is_unitary()


def test_trotter_raises_instead_of_returning_an_invalid_propagator():
    """The reported blast radius: Trotter validated only the algebra size."""
    n = 1
    with pytest.raises(ValueError, match="P\\^2 = 1"):
        trotter_unitary([(0.5, X(n, 0) + Z(n, 0))], 1.0, 4)
    good = trotter_unitary([(0.5, X(n, 0)), (0.3, Z(n, 0))], 1.0, 4)
    assert good.is_unitary()


# -------------------------------------------------------------- controlled


def test_controlled_rejects_a_target_acting_on_the_control():
    with pytest.raises(ValueError, match="identity on ctrl"):
        controlled(X(1, 0), 0)
    bad = controlled(X(1, 0), 0, check_identity_on_control=False)
    assert not bad.is_unitary()
    assert (bad.dagger() * bad).is_close(I(1) + Z(1, 0))


def test_controlled_gates_remain_correct():
    assert CNOT(2, 0, 1).is_unitary()
    assert CZ(2, 0, 1).is_unitary()
    assert TOFFOLI(3, 0, 1, 2).is_unitary()
    with pytest.raises(ValueError, match="identity on ctrl"):
        CNOT(2, 0, 0)
