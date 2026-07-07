"""Property-based tests (hypothesis) for the algebra kernel and layers above it."""

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

from clifford_qc import (
    MV, rotor, label_to_code, code_to_label, blade_from_mask,
    to_matrix, from_matrix, tensor,
    depolarizing, dephasing, amplitude_damping, apply_channel,
    partial_trace, partial_transpose, ket_density,
)

MAX_N = 3


def coeffs():
    part = st.floats(min_value=-2, max_value=2, allow_nan=False, allow_infinity=False)
    return st.tuples(part, part).map(lambda t: complex(*t))


@st.composite
def mvs(draw, n=None, max_terms=6):
    if n is None:
        n = draw(st.integers(min_value=1, max_value=MAX_N))
    k = draw(st.integers(min_value=0, max_value=max_terms))
    terms = draw(st.dictionaries(st.integers(min_value=0, max_value=4 ** n - 1), coeffs(),
                                 min_size=0, max_size=k))
    return MV(n, terms)


@st.composite
def mv_pairs(draw):
    n = draw(st.integers(min_value=1, max_value=MAX_N))
    return draw(mvs(n=n)), draw(mvs(n=n))


@st.composite
def mv_triples(draw):
    n = draw(st.integers(min_value=1, max_value=MAX_N))
    return draw(mvs(n=n)), draw(mvs(n=n)), draw(mvs(n=n))


@st.composite
def pauli_words(draw):
    n = draw(st.integers(min_value=1, max_value=MAX_N))
    label = draw(st.text(alphabet="IXYZ", min_size=n, max_size=n))
    return n, label


@settings(deadline=None)
@given(mv_triples())
def test_associativity_and_distributivity(triple):
    A, B, C = triple
    assert ((A * B) * C).is_close(A * (B * C), 1e-7)
    assert (A * (B + C)).is_close(A * B + A * C, 1e-7)


@settings(deadline=None)
@given(mv_pairs())
def test_dagger_and_trace(pair):
    A, B = pair
    assert (A * B).dagger().is_close(B.dagger() * A.dagger(), 1e-7)
    assert abs((A * B).trace() - (B * A).trace()) < 1e-7
    assert A.dagger().dagger().is_close(A, 1e-9)


@settings(deadline=None)
@given(mv_pairs())
def test_matrix_bridge_homomorphism(pair):
    A, B = pair
    assert np.allclose(to_matrix(A * B), to_matrix(A) @ to_matrix(B), atol=1e-7)
    assert from_matrix(to_matrix(A), A.n).is_close(A, 1e-7)
    assert abs(np.trace(to_matrix(A)) - A.trace()) < 1e-7


@settings(deadline=None)
@given(pauli_words())
def test_label_code_roundtrip(word):
    n, label = word
    assert code_to_label(n, label_to_code(label, n)) == label


@settings(deadline=None)
@given(st.integers(min_value=1, max_value=MAX_N), st.data())
def test_blade_mask_roundtrip(n, data):
    mask = data.draw(st.integers(min_value=0, max_value=(1 << (2 * n)) - 1))
    blade = blade_from_mask(n, mask)
    assert blade.nnz() == 1
    assert blade.blade_mask(next(iter(blade.terms))) == mask


@settings(deadline=None)
@given(pauli_words(),
       st.floats(min_value=-6, max_value=6, allow_nan=False),
       st.floats(min_value=-6, max_value=6, allow_nan=False))
def test_rotor_group_law(word, a, b):
    n, label = word
    P = MV.from_label(label)
    assert rotor(P, a).is_unitary(1e-9)
    assert (rotor(P, a) * rotor(P, b)).is_close(rotor(P, a + b), 1e-9)


@settings(deadline=None, max_examples=40)
@given(st.integers(min_value=1, max_value=2),
       st.floats(min_value=0, max_value=1, allow_nan=False),
       st.integers(min_value=0, max_value=2))
def test_channels_preserve_density(n, p, which):
    n = max(n, 1)
    rho = ket_density(n, "0" * n)
    make = [depolarizing, dephasing, amplitude_damping][which]
    out = apply_channel(rho, make(n, 0, p))
    assert abs(out.trace() - 1) < 1e-9
    assert out.is_hermitian(1e-9)
    ev = np.linalg.eigvalsh(to_matrix(out))
    assert float(min(ev)) > -1e-9


@settings(deadline=None, max_examples=40)
@given(mvs(n=MAX_N))
def test_partial_operations(A):
    traced = partial_trace(A, {0})
    assert abs(traced.trace() - A.trace()) < 1e-8
    assert partial_transpose(partial_transpose(A, {1}), {1}).is_close(A, 1e-9)


@settings(deadline=None, max_examples=30)
@given(mvs(n=1), mvs(n=2))
def test_tensor_matches_kron(A, B):
    assert np.allclose(to_matrix(tensor(A, B)),
                       np.kron(to_matrix(A), to_matrix(B)), atol=1e-8)
