"""Exterior-algebra layer: reversion, wedge, and the k-vector dot product.

The oracle throughout is the Khosravi-Taylor construction of Lambda^k R^n from
oriented parallelepipeds: a simple k-vector's pairing with another is the Gram
determinant ``det(a_i . b_j)``, and that determinant equals the GA scalar
product ``<a ~b>_0`` -- *with* reversion. Dropping the reversion flips grades
``k = 2, 3 mod 4``, which is the sign trap these tests pin down.

Reference: M. Khosravi and M. D. Taylor, "The Wedge Product and Analytic
Geometry", Amer. Math. Monthly 115 (2008) 623-644.
"""

import itertools

import numpy as np
import pytest

from clifford_qc.clifford import blade_from_mask, gamma
from clifford_qc.multivector import MV, blade_mask

N_QUBITS = 3
N_GEN = 2 * N_QUBITS


def rev_sign(k: int) -> int:
    return (-1) ** (k * (k - 1) // 2)


def one_vector(coeffs) -> MV:
    """A grade-1 element: a real combination of the JW Clifford generators."""
    out = MV(N_QUBITS)
    for c, i in zip(coeffs, range(N_GEN)):
        out = out + c * gamma(N_QUBITS, i)
    return out


def wedge_all(vectors):
    out = vectors[0]
    for v in vectors[1:]:
        out = out.wedge(v)
    return out


# ---------------------------------------------------------------- reversion


def test_generators_form_a_euclidean_clifford_algebra():
    """JW images of the generators square to +1 and pairwise anticommute."""
    g = [gamma(N_QUBITS, i) for i in range(N_GEN)]
    for i in range(N_GEN):
        assert (g[i] * g[i]).is_close(MV.scalar(N_QUBITS, 1.0))
    for i, j in itertools.combinations(range(N_GEN), 2):
        assert (g[i] * g[j] + g[j] * g[i]).is_zero()


@pytest.mark.parametrize("mask", range(1 << N_GEN))
def test_reversion_normalizes_every_basis_blade(mask):
    """``<B ~B>_0 == 1`` on every blade, and ``<B B>_0`` carries the sign."""
    B = blade_from_mask(N_QUBITS, mask)
    k = mask.bit_count()
    assert (B * (~B)).scalar_part() == pytest.approx(1.0, abs=1e-12)
    assert (B * B).scalar_part() == pytest.approx(rev_sign(k), abs=1e-12)


def test_reversion_is_an_involution_and_grade_wise():
    rng = np.random.default_rng(0)
    codes = rng.choice(4 ** N_QUBITS, size=8, replace=False)
    A = MV(N_QUBITS, {int(c): complex(rng.normal(), rng.normal()) for c in codes})
    assert (~(~A)).is_close(A)
    for code, v in A.terms.items():
        k = blade_mask(N_QUBITS, code).bit_count()
        assert (~A).terms[code] == pytest.approx(rev_sign(k) * v)


# -------------------------------------------------------------------- wedge


@pytest.mark.parametrize("ma", range(1 << N_GEN))
def test_wedge_is_the_disjoint_geometric_product(ma):
    """Blades wedge to their geometric product iff their generator sets are
    disjoint, and vanish otherwise; grades add when they do not."""
    a = blade_from_mask(N_QUBITS, ma)
    for mb in range(1 << N_GEN):
        b = blade_from_mask(N_QUBITS, mb)
        w = a.wedge(b)
        if ma & mb:
            assert w.is_zero()
        else:
            assert w.is_close(a * b)
            assert w.grades() == {ma.bit_count() + mb.bit_count()}


def test_wedge_is_bilinear_and_operator_form_agrees():
    rng = np.random.default_rng(1)
    g = [gamma(N_QUBITS, i) for i in range(N_GEN)]
    a, b, c = g[0], g[1], g[2]
    lam = complex(rng.normal(), rng.normal())
    assert a.wedge(b + lam * c).is_close(a.wedge(b) + lam * a.wedge(c))
    assert (a ^ b).is_close(a.wedge(b))
    assert a.wedge(b).is_close(-b.wedge(a))       # antisymmetry on 1-vectors
    assert a.wedge(a).is_zero()                   # repeated factor kills it


# ------------------------------------------- scalar product vs Gram oracle


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_scalar_product_equals_gram_determinant(k):
    """``<a ~b>_0 == det(a_i . b_j)`` -- the Khosravi-Taylor pairing."""
    rng = np.random.default_rng(100 + k)
    for _ in range(20):
        A = rng.normal(size=(k, N_GEN))
        B = rng.normal(size=(k, N_GEN))
        a = wedge_all([one_vector(row) for row in A])
        b = wedge_all([one_vector(row) for row in B])
        det = np.linalg.det(A @ B.T)
        assert a.scalar_product(b) == pytest.approx(det, abs=1e-9)


@pytest.mark.parametrize("k", [2, 3])
def test_dropping_reversion_flips_grades_2_and_3_mod_4(k):
    """The trap: ``<a b>_0`` differs from the Gram determinant by rev_sign(k).

    Guards against anyone "simplifying" scalar_product to the plain scalar
    part of the geometric product.
    """
    rng = np.random.default_rng(200 + k)
    A = rng.normal(size=(k, N_GEN))
    B = rng.normal(size=(k, N_GEN))
    a = wedge_all([one_vector(row) for row in A])
    b = wedge_all([one_vector(row) for row in B])
    det = np.linalg.det(A @ B.T)
    assert rev_sign(k) == -1
    assert (a * b).scalar_part() == pytest.approx(rev_sign(k) * det, abs=1e-9)
    assert abs((a * b).scalar_part() - det) > 1e-6   # genuinely different


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_squared_norm_is_the_gram_determinant(k):
    """``|a|^2 = <a ~a>_0 = det(a_i . a_j)``, which is positive here."""
    rng = np.random.default_rng(300 + k)
    A = rng.normal(size=(k, N_GEN))
    a = wedge_all([one_vector(row) for row in A])
    det = np.linalg.det(A @ A.T)
    assert a.scalar_product(a).real == pytest.approx(det, abs=1e-9)
    assert det > 0                       # Euclidean: independent rows


@pytest.mark.parametrize("k", [2, 3])
def test_dropping_reversion_also_breaks_the_squared_norm(k):
    """Norms are *not* immune to the reversion convention.

    At ``k = 2, 3`` the sign is ``-1``, so omitting reversion reports a
    negative squared norm for a Euclidean blade. Only comparing absolute
    magnitudes would hide the error.
    """
    rng = np.random.default_rng(400 + k)
    A = rng.normal(size=(k, N_GEN))
    a = wedge_all([one_vector(row) for row in A])
    det = np.linalg.det(A @ A.T)
    assert (a * a).scalar_part().real == pytest.approx(-det, abs=1e-9)
    assert (a * a).scalar_part().real < 0 < det


def test_scalar_product_is_bilinear_and_hs_product_is_sesquilinear():
    """The two pairings this class exposes must not be confused."""
    rng = np.random.default_rng(4)
    codes = [int(c) for c in rng.choice(4 ** N_QUBITS, size=5, replace=False)]
    A = MV(N_QUBITS, {c: complex(rng.normal(), rng.normal()) for c in codes})
    B = MV(N_QUBITS, {c: complex(rng.normal(), rng.normal()) for c in codes})
    lam = 2 + 3j
    assert A.scalar_product(lam * B) == pytest.approx(lam * A.scalar_product(B))
    assert (lam * A).hs_product(B) == pytest.approx(lam.conjugate() * A.hs_product(B))
    assert A.hs_product(A).real == pytest.approx(
        A.norm_hs() ** 2 / 2 ** N_QUBITS, abs=1e-12)
    assert A.scalar_product(B) == pytest.approx((A * (~B)).scalar_part(), abs=1e-12)


def test_binet_cauchy_via_lambda_k_dot_product():
    """``det(A B) = sum over k-subsets of minor products`` is the Lambda^k dot
    product computed in two bases; check it against the blade pairing."""
    k, m = 3, 5
    rng = np.random.default_rng(5)
    A = rng.normal(size=(k, m))
    B = rng.normal(size=(m, k))
    expanded = sum(np.linalg.det(A[:, list(S)]) * np.linalg.det(B[list(S), :])
                   for S in itertools.combinations(range(m), k))
    assert np.linalg.det(A @ B) == pytest.approx(expanded, abs=1e-10)
    # and the same number as a blade pairing in this algebra
    a = wedge_all([one_vector(np.pad(row, (0, N_GEN - m))) for row in A])
    b = wedge_all([one_vector(np.pad(col, (0, N_GEN - m))) for col in B.T])
    assert a.scalar_product(b) == pytest.approx(np.linalg.det(A @ B), abs=1e-9)


# --------------------------------------------------------------- simplicity


def test_odd_grade_sums_are_rejected():
    """``A ^ A == 0`` is vacuous at odd grade, so it cannot be the test.

    Graded commutativity gives ``A ^ A = (-1)^{k^2} A ^ A``, forcing
    ``A ^ A = 0`` for every odd-grade element whether or not it is simple.
    Screening on it alone accepted every homogeneous odd-grade multivector.
    """
    g = [gamma(N_QUBITS, i) for i in range(N_GEN)]
    witness = (g[0] ^ g[1] ^ g[2]) + (g[3] ^ g[4] ^ g[5])
    assert witness.grades() == {3}
    assert witness.wedge(witness).is_zero()        # vacuously, the trap
    assert not witness.is_blade()                  # but it is not decomposable
    # Plucker witness: a contraction that escapes the subspace
    v = (blade_from_mask(N_QUBITS, 0b000011) * witness).grade(1)
    assert not v.wedge(witness).is_zero()


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_every_decomposable_k_vector_is_accepted(k):
    """The other direction: no genuine blade may be rejected."""
    rng = np.random.default_rng(500 + k)
    for _ in range(10):
        a = wedge_all([one_vector(row) for row in rng.normal(size=(k, N_GEN))])
        assert a.is_blade()


def test_simplicity_screen():
    g = [gamma(N_QUBITS, i) for i in range(N_GEN)]
    assert MV(N_QUBITS).is_blade()                       # zero
    assert MV.scalar(N_QUBITS, 2.5).is_blade()           # grade 0
    assert g[0].is_blade()                               # grade 1
    assert all(blade_from_mask(N_QUBITS, m).is_blade() for m in range(1 << N_GEN))
    assert (g[0] ^ g[1] ^ g[2]).is_blade()
    # the standard non-simple witness, and it is genuinely nonzero
    non_simple = (g[0] ^ g[1]) + (g[2] ^ g[3])
    assert not non_simple.is_blade()
    assert not non_simple.wedge(non_simple).is_zero()
    assert not (MV.scalar(N_QUBITS, 1.0) + g[0]).is_blade()   # inhomogeneous


def test_blade_mask_round_trips_and_composes_by_xor():
    """The Pauli-word <-> Clifford-blade dictionary the wedge relies on."""
    for ma in range(1 << N_GEN):
        A = blade_from_mask(N_QUBITS, ma)
        (code_a,) = A.terms
        assert blade_mask(N_QUBITS, code_a) == ma
        for mb in range(1 << N_GEN):
            C = A * blade_from_mask(N_QUBITS, mb)
            (code_c,) = C.terms
            assert blade_mask(N_QUBITS, code_c) == ma ^ mb
