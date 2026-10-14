"""Executable fixtures for §3.6(iv): eligibility before transport (GA-3).

§3.6(iv) proves that a computational determinant, or any pure stabilizer
state, is not an eigenstate of a unit clique combination
``S = sum_i c_i A_i`` once two coefficients are nonzero: at most one member has
a nonzero expectation, so ``<S> = sign_d c_d`` and
``Var(S) = sum_(i != d) c_i**2``. These fixtures execute that proof's lemma and
closed forms, the numerical selector's tolerance boundary, the scope limit for
correlated non-stabilizer references and the transport caveats. They build no
transport or projection primitive and establish no eligibility outside the
theorem's scope.

Dense oracles use explicit tensor matrices and dense statevector enumeration,
not the packed-code anticommutation predicate or sparse word multiplication.
Determinant and integer-amplitude fixtures use exact rational arithmetic:
products of Pauli matrices are monomial with entries in ``{0, ±1, ±i}``, so
their floating-point entries on small integer vectors are exact.
"""

import functools
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from clifford_qc.dense_reference import density_from_statevector, from_matrix
from clifford_qc.ir import PauliWord
from clifford_qc.measurement import compile_anticommuting_clique
from clifford_qc.pauli import I
from clifford_qc.states import evolve, expectation, ket_density
from clifford_qc.subspace import select_contextual_stabilizers


MATS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}
TOL = 1e-9  # select_contextual_stabilizers' default reference tolerance
# Rational unit vectors with every entry nonzero; cyclic shifts move each
# value onto the clique's (possibly) diagonal member.
UNIT = {
    2: [(Fraction(3, 5), Fraction(4, 5)), (Fraction(-5, 13), Fraction(12, 13))],
    3: [(Fraction(2, 7), Fraction(3, 7), Fraction(6, 7)),
        (Fraction(-1, 3), Fraction(2, 3), Fraction(-2, 3))],
    4: [(Fraction(2, 9), Fraction(4, 9), Fraction(5, 9), Fraction(6, 9)),
        (Fraction(1, 2), Fraction(-1, 2), Fraction(1, 2), Fraction(-1, 2))],
    5: [(Fraction(1, 6), Fraction(1, 6), Fraction(3, 6), Fraction(3, 6), Fraction(4, 6)),
        (Fraction(-1, 6), Fraction(1, 6), Fraction(-3, 6), Fraction(3, 6), Fraction(-4, 6))],
}
# The n = 3 Majorana frame plus its Hermitian pseudoscalar, written out rather
# than built by the package; only ZZZ is diagonal.
FRAME7 = ("XII", "YII", "ZXI", "ZYI", "ZZX", "ZZY", "ZZZ")
FRAME7_COEFFICIENTS = tuple(Fraction(k, 4) for k in (1, -1, 1, -1, 2, -2, 2))


@functools.lru_cache(maxsize=None)
def dense(label):
    out = np.ones((1, 1), dtype=complex)
    for letter in label:
        out = np.kron(out, MATS[letter])
    out.flags.writeable = False  # cached and shared between tests
    return out


def labels(n):
    return ["".join(p) for p in itertools.product("IXYZ", repeat=n)][1:]


def words(members):
    return [PauliWord.from_label(label) for label in members]


@functools.lru_cache(maxsize=None)
def anticommute(a, b):
    A, B = dense(a), dense(b)
    return np.array_equal(A @ B, -B @ A)


def shifts(vector):
    return [vector[k:] + vector[:k] for k in range(len(vector))]


def gaussian_integer(value):
    re, im = float(value.real), float(value.imag)
    assert re.is_integer() and im.is_integer(), value
    return int(re), int(im)


def exact_moments(members, coefficients, vector):
    """Exact <v|S|v>/<v|v> and <v|S^2|v>/<v|v> for an integer-amplitude ket."""
    v = np.asarray(vector, dtype=complex)
    norm = Fraction(int((v.conj() @ v).real))
    mats = [dense(label) for label in members]
    first = second = imaginary = Fraction(0)
    for c, A in zip(coefficients, mats):
        re, im = gaussian_integer(v.conj() @ A @ v)
        assert im == 0
        first += c * re
    for (ci, A), (cj, B) in itertools.product(zip(coefficients, mats), repeat=2):
        re, im = gaussian_integer(v.conj() @ A @ B @ v)
        second += ci * cj * re
        imaginary += ci * cj * im
    assert imaginary == 0  # anticommuting cross terms cancel exactly
    return first / norm, second / norm


def is_diagonal(label):
    return set(label) <= {"I", "Z"}


def local(n, qubit, gate):
    out = np.ones((1, 1), dtype=complex)
    for j in range(n):
        out = np.kron(out, gate if j == qubit else MATS["I"])
    return out


def cx(n, control, target):
    size = 2**n
    gate = np.zeros((size, size), dtype=complex)
    for col in range(size):
        row = col ^ (1 << (n - 1 - target)) if col & (1 << (n - 1 - control)) else col
        gate[row, col] = 1
    return gate


@functools.lru_cache(maxsize=None)
def stabilizer_states(n):
    """Every n-qubit stabilizer state: the H/S/CX orbit of |0...0>, mod phase."""
    hadamard = (MATS["X"] + MATS["Z"]) / math.sqrt(2)
    phase = np.diag([1, 1j])
    gates = ([local(n, q, hadamard) for q in range(n)]
             + [local(n, q, phase) for q in range(n)]
             + [cx(n, c, t) for c in range(n) for t in range(n) if c != t])

    def key(psi):
        k = next(i for i, a in enumerate(psi) if abs(a) > 1e-9)
        psi = psi * abs(psi[k]) / psi[k]
        return tuple(np.round(np.concatenate([psi.real, psi.imag]), 9))

    start = np.zeros(2**n, dtype=complex)
    start[0] = 1
    seen = {key(start): start}
    frontier = [start]
    while frontier:
        found = []
        for psi in frontier:
            for gate in gates:
                image = gate @ psi
                if key(image) not in seen:
                    seen[key(image)] = image
                    found.append(image)
        frontier = found
    return tuple(seen.values())


@functools.lru_cache(maxsize=None)
def two_qubit_cliques():
    """Every pairwise anticommuting set of at least two two-qubit words."""
    members = labels(2)
    cliques = []
    for size in range(2, len(members) + 1):
        found = [group for group in itertools.combinations(members, size)
                 if all(anticommute(a, b) for a, b in itertools.combinations(group, 2))]
        if not found:
            break
        cliques.extend(found)
    return tuple(cliques)


def rotate(operator, rotations):
    """Sparse exact conjugation by emitted rotors, in circuit order."""
    for rotation in rotations:
        operator = evolve(operator, rotation.to_mv())
    return operator


def givens_matrix(n, rotations):
    """Dense product of the emitted rotors, independent of sparse multiplication."""
    V = np.eye(2**n, dtype=complex)
    for rotation in rotations:
        V = (math.cos(rotation.angle / 2) * np.eye(2**n)
             - 1j * math.sin(rotation.angle / 2) * dense(rotation.word.label)) @ V
    return V


# --- Lemma: determinants and stabilizer states ------------------------------

@pytest.mark.parametrize("n", [1, 2, 3])
def test_determinant_word_expectation_is_a_signed_diagonal_indicator(n):
    for label in labels(n):
        A = dense(label)
        for b in range(2**n):
            entry = A[b, b]
            if is_diagonal(label):
                bits = format(b, f"0{n}b")
                sign = (-1) ** sum(bit == "1" for bit, letter in zip(bits, label)
                                   if letter == "Z")
                assert entry == sign
            else:
                assert entry == 0
    diagonal = [label for label in labels(n) if is_diagonal(label)]
    assert not any(anticommute(a, b) for a, b in itertools.combinations(diagonal, 2))


def test_every_two_qubit_clique_is_enumerated():
    sizes = [len(group) for group in two_qubit_cliques()]
    assert len(sizes) == 176
    assert {size: sizes.count(size) for size in set(sizes)} == {2: 60, 3: 80, 4: 30, 5: 6}


@pytest.mark.parametrize("n", [1, 2, 3])
def test_stabilizer_state_expectations_form_one_signed_commuting_group(n):
    states = stabilizer_states(n)
    assert len(states) == 2**n * math.prod(2**k + 1 for k in range(1, n + 1))
    members = labels(n)
    mats = np.array([dense(label) for label in members])
    anti = {(a, b): anticommute(a, b) for a, b in itertools.combinations(members, 2)}
    for psi in states:
        means = np.einsum("i,wij,j->w", psi.conj(), mats, psi)
        np.testing.assert_allclose(means.imag, 0, atol=1e-12)
        rounded = np.round(means.real)
        np.testing.assert_allclose(means.real, rounded, atol=1e-12)
        assert set(rounded) <= {-1.0, 0.0, 1.0}
        group = [label for label, value in zip(members, rounded) if value]
        assert len(group) == 2**n - 1
        # No two nonzero-expectation words anticommute, so no clique contains
        # two of them: the lemma §3.6(iv) uses for general stabilizer states.
        assert not any(anti[a, b] for a, b in itertools.combinations(group, 2))


# --- Closed forms -----------------------------------------------------------

def test_determinant_closed_forms_on_every_two_qubit_clique():
    branches = {"no diagonal word": 0, "one diagonal word": 0}
    for group in two_qubit_cliques():
        diagonal = [i for i, label in enumerate(group) if is_diagonal(label)]
        assert len(diagonal) <= 1
        for vector in UNIT[len(group)]:
            assert sum(c * c for c in vector) == 1
            for coefficients in shifts(vector):
                for b in range(4):
                    ket = [int(i == b) for i in range(4)]
                    mean, square = exact_moments(group, coefficients, ket)
                    assert square == 1  # S**2 = I
                    variance = square - mean * mean
                    if not diagonal:
                        branches["no diagonal word"] += 1
                        assert mean == 0 and variance == 1
                        continue
                    branches["one diagonal word"] += 1
                    d, = diagonal
                    sign = int(dense(group[d])[b, b].real)
                    assert mean == sign * coefficients[d]
                    assert variance == sum(c * c for i, c in enumerate(coefficients) if i != d)
                    # Two or more nonzero coefficients: never an eigenstate.
                    assert abs(mean) < 1 and variance > 0
    assert all(branches.values())


def test_stabilizer_closed_forms_on_every_two_qubit_clique():
    states = np.array(stabilizer_states(2)).T  # one column per state
    members = labels(2)
    means = {label: np.round(np.einsum("is,ij,js->s", states.conj(), dense(label), states).real)
             for label in members}
    smallest_variance = math.inf
    branches = {"no stabilizer member": 0, "one stabilizer member": 0}
    for group in two_qubit_cliques():
        nonzero = np.array([means[label] != 0 for label in group])
        assert (nonzero.sum(axis=0) <= 1).all()
        for vector in UNIT[len(group)]:
            for coefficients in shifts(vector):
                c = np.array([float(x) for x in coefficients])
                S = sum(ci * dense(label) for ci, label in zip(c, group))
                image = S @ states
                mean = np.einsum("is,is->s", states.conj(), image).real
                variance = np.einsum("is,is->s", image.conj(), image).real - mean**2
                predicted = sum(ci * means[label] for ci, label in zip(c, group))
                np.testing.assert_allclose(mean, predicted, atol=1e-12)
                np.testing.assert_allclose(variance, 1 - predicted**2, atol=1e-12)
                none = ~nonzero.any(axis=0)
                np.testing.assert_allclose(mean[none], 0, atol=1e-12)
                np.testing.assert_allclose(variance[none], 1, atol=1e-12)
                branches["no stabilizer member"] += int(none.sum())
                branches["one stabilizer member"] += int((~none).sum())
                smallest_variance = min(smallest_variance, float(variance.min()))
    assert all(branches.values())
    # Every coefficient is nonzero, so no stabilizer state is an eigenstate; the
    # closest case puts the largest coefficient, 12/13, on the stabilizer member.
    assert smallest_variance == pytest.approx(25 / 169, abs=1e-12)


def test_clifford_transformed_determinant_clique_keeps_the_closed_form():
    n = 3
    hadamard = (MATS["X"] + MATS["Z"]) / math.sqrt(2)
    U = np.eye(2**n, dtype=complex)
    for gate in (local(n, 0, hadamard), cx(n, 0, 1), local(n, 2, np.diag([1, 1j])),
                 local(n, 1, hadamard), cx(n, 1, 2), local(n, 2, hadamard)):
        U = gate @ U
    paulis = {label: dense(label) for label in labels(n)}
    transformed = []
    for label in FRAME7:
        image = U @ dense(label) @ U.conj().T
        weights = {other: np.vdot(P, image).real / 2**n for other, P in paulis.items()}
        support = [(other, w) for other, w in weights.items() if abs(w) > 1e-12]
        assert len(support) == 1 and abs(abs(support[0][1]) - 1) < 1e-12
        transformed.append(image)
    for i, A in enumerate(transformed):
        for B in transformed[:i]:
            np.testing.assert_allclose(A @ B, -B @ A, atol=1e-12)
    c = [float(x) for x in FRAME7_COEFFICIENTS]
    S = sum(ci * A for ci, A in zip(c, transformed))
    tail = float(sum(x * x for x in FRAME7_COEFFICIENTS[:-1]))
    for b in range(2**n):
        psi = U[:, b]
        assert np.count_nonzero(np.abs(psi) > 1e-12) > 1  # not a determinant
        member_means = [np.vdot(psi, A @ psi).real for A in transformed]
        np.testing.assert_allclose(member_means[:-1], 0, atol=1e-12)
        sign = (-1) ** bin(b).count("1")  # <ZZZ> on |b>
        assert abs(member_means[-1] - sign) < 1e-12
        mean = np.vdot(psi, S @ psi).real
        assert abs(mean - sign * c[-1]) < 1e-12
        assert abs(np.vdot(S @ psi, S @ psi).real - mean**2 - tail) < 1e-12


# --- The numerical selector after a simultaneous rotation -------------------

REFERENCES = {"determinant": (1, 0, 0, 0), "bell": (1, 0, 0, 1)}
# (reference, ordered clique with its stabilizer member first, an independent
# commuting stabilizer that also commutes with the clique's Givens generator)
SELECTOR_CASES = [
    ("determinant", ("ZI", "XI"), "IZ"),
    ("bell", ("XX", "ZI"), "ZZ"),
]


def selection_after_rotation(vector, members, coefficients, fallback):
    reference = density_from_statevector(np.array(vector, dtype=complex))
    setting = compile_anticommuting_clique(words(members), [float(c) for c in coefficients])
    rotated = rotate(reference, setting.rotations)
    # The fallback must survive the rotation unchanged to be a clean control.
    assert not any(anticommute(fallback, rotation.word.label) for rotation in setting.rotations)
    pivot = setting.pivot.to_mv()
    hamiltonian = -1.0 * pivot - 0.25 * PauliWord.from_label(fallback).to_mv()
    selection = select_contextual_stabilizers(hamiltonian, rotated, 1, tol=TOL)
    return setting, rotated, pivot, selection


@pytest.mark.parametrize("name,members,fallback", SELECTOR_CASES)
@pytest.mark.parametrize("t,admitted", [
    (Fraction(1, 2_000_000), True),   # off-diagonal coefficient ~1e-6, gap ~5e-13
    (Fraction(1, 50_000), True),      # gap 8e-10, just inside the tolerance
    (Fraction(3, 100_000), False),    # gap 1.8e-9, just outside it
])
def test_near_singleton_tolerance_pass_is_not_exact_preservation(
        name, members, fallback, t, admitted):
    coefficients = ((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t))
    mean, square = exact_moments(members, coefficients, REFERENCES[name])
    gap = 1 - abs(mean)
    variance = square - mean * mean
    assert mean == coefficients[0] and gap == 2 * t * t / (1 + t * t)
    assert variance == coefficients[1] ** 2 > 0  # never an exact eigenstate
    assert (gap <= Fraction(TOL)) is admitted

    setting, rotated, pivot, selection = selection_after_rotation(
        REFERENCES[name], members, coefficients, fallback)
    assert len(setting.rotations) == 1
    # Reference and observable rotate together: <A_1> after V is <S> before it.
    assert abs(expectation(rotated, pivot) - float(mean)) < 1e-15
    chosen = selection.stabilizers[0]
    if admitted:
        assert chosen.word == setting.pivot and chosen.visit_index == 0
        assert selection.skipped_reference_not_eigenstate == 0
    else:
        assert chosen.word.label == fallback
        assert selection.skipped_reference_not_eigenstate == 1
    # Projecting onto the +1 eigenspace keeps 1 - gap/2 of the trace, admitted
    # or not: a tolerance pass is an unpriced approximation, not preservation.
    kept = expectation(rotated, 0.5 * (I(2) + pivot)).real
    assert abs(kept - float(1 - gap / 2)) < 1e-15


@pytest.mark.parametrize("name,members,coefficients,fallback,outcome", [
    # Singleton stabilizer word: the ordinary Pauli-stabilizer case.
    ("determinant", ("ZI",), (1,), "IZ", "admitted"),
    ("bell", ("XX",), (1,), "ZZ", "admitted"),
    # Singleton outside the reference's stabilizer group: <S> = 0, Var = 1.
    ("determinant", ("XY",), (1,), "IZ", "not eigenstate"),
    ("bell", ("ZI",), (1,), "ZZ", "not eigenstate"),
    # One nonzero coefficient in a multiword clique is still ordinary.
    ("determinant", ("ZI", "XI", "YI"), (1, 0, 0), "IZ", "admitted"),
    ("determinant", ("ZI", "XI", "YI"), (0, 1, 0), "IZ", "not eigenstate"),
    ("bell", ("XX", "ZI"), (1, 0), "ZZ", "admitted"),
    # A negative pivot fires its pi rotation; the exact eigenstate then has the
    # unfavourable sign, which the selector checks independently.
    ("determinant", ("ZI", "XI", "YI"), (-1, 0, 0), "IZ", "sign mismatch"),
])
def test_singleton_and_one_nonzero_coefficient_cases(
        name, members, coefficients, fallback, outcome):
    mean, square = exact_moments(members, coefficients, REFERENCES[name])
    assert square == 1
    eigenstate = square - mean * mean == 0
    assert eigenstate is (outcome != "not eigenstate")
    if not eigenstate:
        assert mean == 0
    setting, rotated, pivot, selection = selection_after_rotation(
        REFERENCES[name], members, coefficients, fallback)
    assert abs(expectation(rotated, pivot) - float(mean)) < 1e-15
    chosen = selection.stabilizers[0]
    if outcome == "admitted":
        assert chosen.word == setting.pivot and chosen.eigenvalue == 1
    else:
        assert chosen.word.label == fallback
        assert selection.skipped_reference_not_eigenstate == int(outcome == "not eigenstate")
        assert selection.skipped_reference_sign_mismatch == int(outcome == "sign mismatch")


# --- Scope: correlated non-stabilizer references -----------------------------

def test_correlated_non_stabilizer_reference_is_outside_the_theorem_scope():
    psi = (2, 0, 0, 1)  # (2|00> + |11>)/sqrt(5): even parity, entangled
    members, coefficients = ("ZI", "XX"), (Fraction(3, 5), Fraction(4, 5))
    assert anticommute(*members)
    zi, zz, iz = (exact_moments((label,), (1,), psi)[0] for label in ("ZI", "ZZ", "IZ"))
    assert zz == 1 and zi == iz == Fraction(3, 5) and zz != zi * iz
    # A Pauli expectation of 3/5 rules out every stabilizer state.
    assert zi not in (-1, 0, 1)
    mean, square = exact_moments(members, coefficients, psi)
    assert mean == 1 and square - mean * mean == 0  # exact eigenstate, two words

    # Eligibility is genuine here: V carries this reference to a determinant and
    # the ordinary selector admits the rotated pivot. Using it would still need
    # every other operator transported by the non-Clifford V -- the separately
    # declared, unbuilt step whose support growth is pinned below.
    setting, rotated, pivot, selection = selection_after_rotation(
        psi, members, coefficients, "ZZ")
    assert rotated.is_close(ket_density(2, "00"), 1e-12)
    assert selection.stabilizers[0].word == setting.pivot


# --- Transport caveats ------------------------------------------------------

def frame_rotation(members, coefficients):
    """The SO(m) action of the emitted Givens product on the clique frame."""
    setting = compile_anticommuting_clique(words(members), [float(c) for c in coefficients])
    n = len(members[0])
    V = givens_matrix(n, setting.rotations)
    mats = [dense(label) for label in members]
    R = np.array([[np.trace(A @ V @ B @ V.conj().T).real / 2**n for B in mats] for A in mats])
    return setting, R


@pytest.mark.parametrize("members,coefficients,rank", [
    (("X", "Y"), UNIT[2][0], 2),
    (("X", "Y", "Z"), UNIT[3][0], 2),            # every Spin(3) rotor is simple
    (("XI", "YI", "ZX", "ZY"), (Fraction(3, 5), 0, Fraction(4, 5), 0), 2),
    (("XI", "YI", "ZX", "ZY"), UNIT[4][0], 4),    # a double rotation
    (("XI", "YI", "ZX", "ZY"), UNIT[4][1], 4),
    (FRAME7, FRAME7_COEFFICIENTS, 6),             # three rotation planes
])
def test_givens_product_is_a_spin_rotor_not_necessarily_a_simple_one(
        members, coefficients, rank):
    setting, R = frame_rotation(members, coefficients)
    m = len(members)
    np.testing.assert_allclose(R @ R.T, np.eye(m), atol=1e-12)
    assert abs(np.linalg.det(R) - 1) < 1e-12
    e1 = np.eye(m)[0]
    np.testing.assert_allclose(R @ np.array([float(c) for c in coefficients]), e1, atol=1e-12)
    # A simple rotor fixes an (m - 2)-dimensional subspace pointwise.
    assert np.linalg.matrix_rank(R - np.eye(m), tol=1e-9) == rank


def test_sparse_word_transport_expands_support_across_rotations():
    n = 3
    setting = compile_anticommuting_clique(words(FRAME7), [float(c) for c in FRAME7_COEFFICIENTS])
    assert len(setting.rotations) == 6
    angles = [rotation.angle for rotation in setting.rotations]
    assert all(abs(math.sin(a)) > 1e-3 and abs(math.cos(a)) > 1e-3 for a in angles)
    first = setting.rotations[0]
    P = dense(first.word.label)
    U = givens_matrix(n, setting.rotations[:1])
    V = givens_matrix(n, setting.rotations)
    supports = {}
    for label in labels(n):
        W = dense(label)
        word = PauliWord.from_label(label).to_mv()
        once = rotate(word, setting.rotations[:1])
        # One rotation: a commuting word is fixed and an anticommuting word
        # becomes exactly two, cos(theta) W - i sin(theta) P W.
        if anticommute(label, first.word.label):
            expected = math.cos(first.angle) * W - 1j * math.sin(first.angle) * P @ W
            assert len(once.terms) == 2
        else:
            expected = W
            assert len(once.terms) == 1
        np.testing.assert_allclose(U @ W @ U.conj().T, expected, atol=1e-12)
        sparse = rotate(word, setting.rotations)
        oracle = from_matrix(V @ W @ V.conj().T, tol=1e-12)
        support = {code for code, value in sparse.terms.items() if abs(value) > 1e-12}
        assert support == set(oracle.terms)
        supports[label] = len(support)
    # Composition is not bounded by the single-rotation two-word identity.
    assert max(supports.values()) == 21
    assert sum(size > 2 for size in supports.values()) > len(supports) // 2
