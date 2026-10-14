"""Named bilinear contractions against grade-projected reference products."""
import numpy as np
import pytest

from clifford_qc import MV, blade_from_mask, blade_mask
from clifford_qc import multivector
from clifford_qc.pauli_kernel import word_mul_reference


def reference(a, b, *, left):
    out = {}
    for ca, va in a.terms.items():
        r = blade_mask(a.n, ca).bit_count()
        for cb, vb in b.terms.items():
            s = blade_mask(b.n, cb).bit_count()
            if (r > s if left else s > r):
                continue
            phase, code = word_mul_reference(a.n, ca, cb)
            if blade_mask(a.n, code).bit_count() != abs(s - r):
                continue
            reversed_grade = r if left else s
            sign = (-1) ** (reversed_grade * (reversed_grade - 1) // 2)
            out[code] = out.get(code, 0j) + sign * phase * va * vb
    return MV(a.n, out)


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_every_small_basis_pair_matches_projected_products(n):
    for a in range(4 ** n):
        for b in range(4 ** n):
            left, right = MV(n, {a: 2j}), MV(n, {b: 1 - 3j})
            assert left.lcontract(right).is_close(reference(left, right, left=True))
            assert left.rcontract(right).is_close(reference(left, right, left=False))


def test_mixed_grades_complex_bilinearity_and_operator_aliases():
    rng = np.random.default_rng(3001)
    for _ in range(30):
        a, b, c = [MV(3, {int(code): complex(rng.normal(), rng.normal())
                          for code in rng.choice(64, size=8, replace=False)}) for _ in range(3)]
        assert (a << b).is_close(reference(a, b, left=True))
        assert (a >> b).is_close(reference(a, b, left=False))
        assert ((a + 2j * b) << c).is_close((a << c) + 2j * (b << c))
        assert (a >> (b + 2j * c)).is_close((a >> b) + 2j * (a >> c))


def test_reversion_sign_and_scalar_conventions():
    bivector = blade_from_mask(2, 3)
    assert (bivector << bivector).is_close(MV.scalar(2, 1))
    assert (bivector >> bivector).is_close(MV.scalar(2, 1))
    assert ((1j * bivector) << bivector).is_close(MV.scalar(2, 1j))
    assert (2 << bivector).is_close(2 * bivector)
    assert (bivector >> 2).is_close(2 * bivector)
    assert (bivector << 2).is_zero()
    assert (2 >> bivector).is_zero()


def test_mismatched_algebra_and_operand_type_are_rejected():
    for operation in (MV.scalar(1).lcontract, MV.scalar(1).rcontract):
        with pytest.raises(ValueError, match="mixed algebra sizes"):
            operation(MV.scalar(2))
        with pytest.raises(TypeError):
            operation(object())


@pytest.mark.parametrize("left", [True, False])
def test_non_subset_pairs_are_rejected_before_multiplication(monkeypatch, left):
    a, b = blade_from_mask(3, 0b000011), blade_from_mask(3, 0b001100)
    def forbidden(*args):
        pytest.fail("non-subset contractions must not multiply words")
    monkeypatch.setattr(multivector, "_word_mul_unchecked", forbidden)
    assert (a.lcontract(b) if left else a.rcontract(b)).is_zero()
