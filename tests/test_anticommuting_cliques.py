"""Phase 19's algebraic gate and fixed-functional compilation contract.

Dense oracles below use explicit tensor matrices, not the packed-code
anticommutation predicate or sparse word multiplication used by compilation.
The rational ball fixtures compute expectations without floating-point sums.
"""

import itertools
import math
from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest

from clifford_qc.clifford import gamma
from clifford_qc.dense_reference import to_matrix
from clifford_qc.ir import PauliSum, PauliWord
from clifford_qc.measurement import (
    anticommuting_clique_cover, compile_anticommuting_clique,
    compile_clique_measurement_plan,
)
from clifford_qc.measurement.cliques import COVER_RULE, _setting_resources
from clifford_qc.measurement.cost import SettingResources
from clifford_qc.models import hubbard, extended_hubbard, tfim, random_ising
from clifford_qc.subspace.ga_restriction import hermitian_majorana_monomial


MATS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}


def dense(label):
    out = np.ones((1, 1), dtype=complex)
    for letter in label:
        out = np.kron(out, MATS[letter])
    return out


def words(labels):
    return tuple(PauliWord.from_label(label) for label in labels)


def rotor_matrix(word, angle):
    return math.cos(angle / 2) * np.eye(2**word.n) - 1j * math.sin(angle / 2) * dense(word.label)


def emitted_matrix(n, ops):
    """Independent gate interpreter, including explicit computational CX action."""
    size = 2**n
    out = np.eye(size, dtype=complex)
    for op in ops:
        name = op[0]
        if name == "cx":
            _, control, target = op
            gate = np.zeros((size, size), dtype=complex)
            for col in range(size):
                row = col ^ (1 << (n - 1 - target)) if col & (1 << (n - 1 - control)) else col
                gate[row, col] = 1
        else:
            if name == "rz":
                _, angle, q = op
                local = np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
            else:
                _, q = op
                local = {"h": (MATS["X"] + MATS["Z"]) / math.sqrt(2),
                         "s": np.diag([1, 1j]), "sdg": np.diag([1, -1j])}[name]
            gate = np.ones((1, 1), dtype=complex)
            for j in range(n):
                gate = np.kron(gate, local if j == q else MATS["I"])
        out = gate @ out
    return out


def reconstructed(setting):
    unitary = emitted_matrix(setting.pivot.n, setting.ops)
    readout = dense("".join("Z" if q in setting.readout_qubits else "I"
                            for q in range(setting.pivot.n)))
    return setting.weight * unitary.conj().T @ readout @ unitary


@pytest.mark.parametrize("coefficients", list(itertools.product((-0.6, 0.0, 0.8), repeat=3)))
def test_closure_and_signed_givens_against_dense(coefficients):
    if not any(coefficients):
        with pytest.raises(ValueError, match="nonzero"):
            compile_anticommuting_clique(words(["X", "Y", "Z"]), coefficients)
        return
    setting = compile_anticommuting_clique(words(["X", "Y", "Z"]), coefficients)
    matrices = [dense(word.label) for word in setting.words]
    B = sum(c * A for c, A in zip(setting.normalized_coefficients, matrices))
    np.testing.assert_allclose(B, B.conj().T, atol=1e-14)
    np.testing.assert_allclose(B @ B, np.eye(2), atol=1e-14)
    V = np.eye(2, dtype=complex)
    for rotation in setting.rotations:
        V = rotor_matrix(rotation.word, rotation.angle) @ V
    np.testing.assert_allclose(V @ B @ V.conj().T, matrices[0], atol=1e-14)
    np.testing.assert_allclose(reconstructed(setting), sum(c * A for c, A in zip(coefficients, matrices)),
                               atol=1e-14)


def test_negative_pivot_with_zero_tail_requires_pi_rotation():
    setting = compile_anticommuting_clique(words(["X", "Y", "Z"]), [-1, 0, 0])
    assert len(setting.rotations) == 1
    assert setting.rotations[0].word.label == "Z"
    assert setting.rotations[0].angle == -math.pi  # i X Y = -Z
    assert setting.weight == 1
    np.testing.assert_allclose(reconstructed(setting), -MATS["X"], atol=1e-14)


@pytest.mark.parametrize("label,coefficient,ops", [
    ("YI", -2.5, (("sdg", 0), ("h", 0))),
    ("IZ", -0.25, ()), ("XY", 3.0, (("h", 0), ("sdg", 1), ("h", 1))),
])
def test_singleton_sign_and_local_readout(label, coefficient, ops):
    setting = compile_anticommuting_clique(words([label]), [coefficient])
    assert setting.rotations == ()
    assert setting.weight == coefficient
    assert setting.ops == ops
    assert setting.readout_qubits == setting.pivot.support()
    np.testing.assert_allclose(reconstructed(setting), coefficient * dense(label), atol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_majorana_ceiling_construction_and_rotation(n):
    frame = [gamma(n, i) for i in range(2*n)] + [hermitian_majorana_monomial(n, range(2*n))]
    matrices = [to_matrix(A) for A in frame]
    identity = np.eye(2**n)
    for i, A in enumerate(matrices):
        np.testing.assert_allclose(A.conj().T, A, atol=0)
        np.testing.assert_allclose(A @ A, identity, atol=0)
        for B in matrices[:i]:
            np.testing.assert_allclose(A @ B + B @ A, np.zeros_like(A), atol=0)
    # The phase of the Hermitian pseudoscalar is carried in its coefficient.
    members, signs = zip(*(next(iter(A.terms.items())) for A in frame))
    coefficients = [(-1)**i * (i+1) * float(sign.real) for i, sign in enumerate(signs)]
    setting = compile_anticommuting_clique([PauliWord(n, code) for code in members], coefficients)
    expected = sum(c * dense(word.label) for c, word in zip(coefficients, setting.words))
    normalized = expected / math.hypot(*coefficients)
    np.testing.assert_allclose(normalized @ normalized, identity, atol=1e-14)
    np.testing.assert_allclose(reconstructed(setting), expected, atol=2e-13)
    # Intrinsic bound on dense random pure/mixed states; not a finite-shot gate.
    rng = np.random.default_rng(1900+n)
    for _ in range(20):
        X = rng.normal(size=(2**n, 2**n)) + 1j*rng.normal(size=(2**n, 2**n))
        rho = X @ X.conj().T
        rho /= np.trace(rho)
        means = [np.trace(rho @ A).real for A in matrices]
        assert sum(x*x for x in means) <= 1 + 1e-14


def test_two_qubit_maximality_by_exhaustion():
    labels = ["".join(pair) for pair in itertools.product("IXYZ", repeat=2)][1:]
    matrices = [dense(label) for label in labels]
    compatible = [[np.array_equal(A @ B, -B @ A) for B in matrices] for A in matrices]
    # Exhaust all 2**15 subsets, with early rejection only after a witnessed pair.
    maximum = 0
    for mask in range(1 << len(matrices)):
        indices = [i for i in range(len(matrices)) if mask & (1 << i)]
        if all(compatible[i][j] for i, j in itertools.combinations(indices, 2)):
            maximum = max(maximum, len(indices))
    assert maximum == 5
    majoranas = [to_matrix(gamma(2, i)) for i in range(4)]
    extensions = [label for label, A in zip(labels, matrices)
                  if all(np.array_equal(A @ B, -B @ A) for B in majoranas)]
    assert extensions == ["ZZ"]


def exact_mean(label, vector):
    """Rational expectation on an integer-amplitude ket (normalization exact)."""
    real = imaginary = 0
    n = len(label)
    for col, amplitude in enumerate(vector):
        row, phase = col, 1
        for q, letter in enumerate(label):
            bit = 1 << (n-1-q)
            if letter in "XY":
                row ^= bit
            if letter == "Y":
                phase *= -1j if col & bit else 1j
            if letter == "Z" and col & bit:
                phase *= -1
        real += int(complex(phase).real) * amplitude * vector[row]
        imaginary += int(complex(phase).imag) * amplitude * vector[row]
    assert imaginary == 0
    return Fraction(real, sum(a*a for a in vector))


@pytest.mark.parametrize("n", [1, 2, 3])
def test_ball_and_variance_floor_in_exact_rational_arithmetic(n):
    frame = [gamma(n, i) for i in range(2*n)] + [hermitian_majorana_monomial(n, range(2*n))]
    labels = [PauliWord(n, next(iter(A.terms))).label for A in frame]
    # Pure integer-amplitude states and a rational convex mixture are PSD by construction.
    basis = [int(i == 0) for i in range(2**n)]
    vectors = [basis, [1] * 2**n, [i-2 for i in range(2**n)]]
    pure_means = [[exact_mean(label, vector) for label in labels] for vector in vectors]
    mixture = [sum(Fraction(w, 6)*row[i] for w, row in zip((1, 2, 3), pure_means))
               for i in range(len(labels))]
    for means in pure_means + [mixture]:
        squared = sum(x*x for x in means)
        assert squared <= 1
        assert sum(1-x*x for x in means) >= len(labels)-1
    assert sum(x*x for x in pure_means[0]) == 1


@pytest.mark.parametrize("model", [hubbard(2), extended_hubbard(2, V=0.3)])
def test_odd_grade_absent_only_in_parity_conserving_fermionic_models(model):
    H = model.hamiltonian.to_mv()
    parity = dense("Z" * H.n)
    np.testing.assert_array_equal(to_matrix(H) @ parity, parity @ to_matrix(H))
    assert all(H.blade_grade(code) % 2 == 0 for code in H.terms)


@pytest.mark.parametrize("model", [tfim(4), random_ising(4, seed=19)])
def test_spin_fields_are_outside_the_fermionic_odd_grade_gate(model):
    H = model.hamiltonian.to_mv()
    assert sum(H.blade_grade(code) % 2 for code in H.terms) == 4


def test_odd_monomial_means_vanish_in_parity_commuting_states():
    n = 3
    parity = dense("ZZZ")
    rng = np.random.default_rng(19)
    X = rng.normal(size=(8, 8)) + 1j*rng.normal(size=(8, 8))
    rho = X @ X.conj().T
    rho = (rho + parity @ rho @ parity) / (2*np.trace(rho))
    for k in (1, 3, 5):
        for indices in itertools.combinations(range(2*n), k):
            A = to_matrix(hermitian_majorana_monomial(n, indices))
            np.testing.assert_array_equal(A @ parity, -parity @ A)
            assert abs(np.trace(rho @ A)) < 1e-14


def test_cover_rule_and_input_permutations():
    members = words(["XI", "YI", "ZI", "IX", "IY", "IZ", "XX", "YY", "ZZ"])
    cover = anticommuting_clique_cover(members)
    # Pin an actual greedy output, rather than merely comparing the function to itself.
    assert [[w.label for w in group] for group in cover] == [
        ["XI", "YI", "ZI"], ["IX", "IY", "IZ"], ["XX"], ["YY"], ["ZZ"],
    ]
    rng = np.random.default_rng(19)
    for _ in range(20):
        assert anticommuting_clique_cover([members[i] for i in rng.permutation(len(members))]) == cover
    assert sorted(w.code for group in cover for w in group) == sorted(w.code for w in members)
    for group in cover:
        assert list(group) == sorted(group, key=lambda word: word.code)
        for a, b in itertools.combinations(group, 2):
            A, B = dense(a.label), dense(b.label)
            np.testing.assert_array_equal(A @ B, -B @ A)


def test_fixed_functional_plan_exact_reconstruction_and_snapshot():
    H = PauliSum.from_labels({"II": -1.5, "XI": -0.3, "YI": 0.7, "ZI": 0.6,
                              "IX": 0.2, "XY": -0.4, "ZY": 0.8})
    plan = compile_clique_measurement_plan(H)
    assert plan.cover_rule == COVER_RULE
    assert plan.resources == tuple(setting.resources for setting in plan.settings)
    actual = plan.identity_offset * np.eye(4) + sum(reconstructed(setting) for setting in plan.settings)
    np.testing.assert_allclose(actual, to_matrix(H.to_mv()), atol=2e-14)
    H.terms.clear()
    assert plan.identity_offset == -1.5 and plan.settings
    with pytest.raises(FrozenInstanceError):
        plan.settings[0].weight = 0


@pytest.mark.parametrize("labels", [("XII", "ZII"), ("XYZ", "ZIZ"), ("IYX", "IZX")])
def test_multiqubit_generator_lowering_and_readout(labels):
    setting = compile_anticommuting_clique(words(labels), [-0.6, 0.8])
    assert len(setting.rotations) == 1
    expected = -0.6*dense(labels[0]) + 0.8*dense(labels[1])
    np.testing.assert_allclose(reconstructed(setting), expected, atol=2e-14)
    assert setting.resources.n_2q == sum(op[0] == "cx" for op in setting.ops)
    assert setting.resources.n_1q == sum(op[0] != "cx" for op in setting.ops)


def test_resources_follow_emitted_support_and_preserve_interleaved_dependencies():
    short = compile_anticommuting_clique(words(["XII", "ZII"]), [0.6, 0.8])
    long = compile_anticommuting_clique(words(["XXX", "ZII"]), [0.6, 0.8])
    assert len(short.words) == len(long.words) == 2
    assert short.resources.n_2q == 0 and long.resources.n_2q == 4
    # Disjoint H pulses share one layer; a CX then an H then CX cannot share clocks.
    ops = (("h", 0), ("h", 1), ("cx", 0, 1), ("h", 0), ("cx", 0, 1))
    assert _setting_resources(ops) == SettingResources(3, 2, 2, 2)
    assert _setting_resources((("cx", 0, 1), ("cx", 2, 3))) == SettingResources(0, 2, 0, 1)
    assert compile_anticommuting_clique(words(["XY"]), [1]).resources == SettingResources(3, 0, 2, 0)


@pytest.mark.parametrize("scale", [1e-300, 1e300, 5e-324])
def test_scaled_normalization_avoids_squared_norm_underflow_and_overflow(scale):
    setting = compile_anticommuting_clique(words(["X", "Y"]), [scale, -scale])
    assert math.isfinite(setting.weight) and setting.weight > 0
    assert math.isclose(math.hypot(*setting.normalized_coefficients), 1, abs_tol=1e-15)
    U = emitted_matrix(1, setting.ops)
    B = sum(c*dense(w.label) for c, w in zip(setting.normalized_coefficients, setting.words))
    np.testing.assert_allclose(U @ B @ U.conj().T, MATS["Z"], atol=1e-14)


@pytest.mark.parametrize("H", [PauliSum(2), PauliSum.from_labels({"II": -2.0})])
def test_zero_and_identity_only_functionals_need_no_settings(H):
    plan = compile_clique_measurement_plan(H)
    assert plan.settings == () and plan.resources == ()
    assert plan.identity_offset == H.terms.get(0, 0).real
    assert anticommuting_clique_cover([]) == ()


def test_plan_drops_only_exact_zeros():
    H = PauliSum.from_labels({"X": 1e-300})
    H.terms[PauliWord.from_label("Z").code] = 0j
    plan = compile_clique_measurement_plan(H)
    assert len(plan.settings) == 1
    assert plan.settings[0].coefficients == (1e-300,)


@pytest.mark.parametrize("members", [words(["I"]), words(["X", "X"]), words(["X", "XX"])])
def test_cover_rejects_identity_duplicates_and_mixed_sizes(members):
    with pytest.raises(ValueError):
        anticommuting_clique_cover(members)


@pytest.mark.parametrize("members,coefficients", [
    ([], []), (words(["X", "Z"]), [1]), (words(["XI", "IX"]), [1, 1]),
    (words(["X"]), [float("nan")]), (words(["X"]), [float("inf")]),
    (words(["X", "Y"]), [1.7e308, 1.7e308]),
])
def test_invalid_compilation_inputs(members, coefficients):
    with pytest.raises(ValueError):
        compile_anticommuting_clique(members, coefficients)


@pytest.mark.parametrize("coefficient", [1j, 1+0j, True, "1"])
def test_direct_clique_requires_real_coefficients(coefficient):
    with pytest.raises(TypeError):
        compile_anticommuting_clique(words(["X"]), [coefficient])


@pytest.mark.parametrize("coefficient", [1+1e-20j, float("nan"), float("inf")])
def test_plan_rejects_nonreal_or_nonfinite_coefficients(coefficient):
    with pytest.raises(ValueError):
        compile_clique_measurement_plan(PauliSum.from_labels({"X": coefficient}))


def test_wrong_object_types_rejected():
    with pytest.raises(TypeError):
        compile_clique_measurement_plan({"X": 1})
    with pytest.raises(TypeError):
        anticommuting_clique_cover(["X"])
