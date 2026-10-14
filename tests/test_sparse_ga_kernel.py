"""Sparse support decisions against exhaustive small-algebra Plücker tests."""
import builtins
import itertools

import numpy as np
import pytest

from clifford_qc import MV, blade_from_mask, blade_mask, mask_to_code
from clifford_qc import multivector as kernel
from clifford_qc.pauli_kernel import word_mul_reference


def exhaustive_blade(value, tol=1e-9):
    """Old exhaustive condition, with independent O(n) word products."""
    if value.is_zero(tol):
        return True
    grades = value.grades()
    if len(grades) != 1:
        return False
    k = next(iter(grades))
    if k <= 1 or k >= 2 * value.n - 1:
        return True
    for code in range(4 ** value.n):
        if blade_mask(value.n, code).bit_count() != k - 1:
            continue
        terms = {}
        for other, coefficient in value.terms.items():
            phase, product = word_mul_reference(value.n, code, other)
            if blade_mask(value.n, product).bit_count() == 1:
                terms[product] = terms.get(product, 0j) + phase * coefficient
        if not MV(value.n, terms).wedge(value).is_zero(tol):
            return False
    return True


def determinant_blade(rows, n):
    """Exterior coordinates are minors, independent of the wedge routine."""
    k = len(rows)
    out = MV(n)
    for columns in itertools.combinations(range(2 * n), k):
        coefficient = np.linalg.det(rows[:, columns])
        mask = sum(1 << i for i in columns)
        out += coefficient * blade_from_mask(n, mask)
    return out


@pytest.mark.parametrize("n", range(6))
def test_inverse_subset_word_map_matches_ordered_majorana_products(n):
    for mask in range(1 << (2 * n)):
        code = mask_to_code(n, mask)
        assert blade_mask(n, code) == mask
        assert mask_to_code(n, blade_mask(n, code)) == code
        assert next(iter(blade_from_mask(n, mask).terms)) == code
    # Canonical phases remain the responsibility of the ordered constructor.
    if n:
        assert blade_from_mask(n, 3).terms == {3: 1j}


@pytest.mark.parametrize("n,mask", [(-1, 0), (1.5, 0), ("2", 0),
                                    (2, -1), (2, 16), (2, 1.5), (2, "1"), (0, 1)])
def test_subset_inverse_rejects_invalid_domains(n, mask):
    with pytest.raises(ValueError):
        mask_to_code(n, mask)


@pytest.mark.parametrize("k", range(2, 6))
def test_complex_decomposable_blades_from_independent_minors(k):
    rng = np.random.default_rng(1800 + k)
    for _ in range(8):
        rows = rng.normal(size=(k, 6)) + 1j * rng.normal(size=(k, 6))
        blade = determinant_blade(rows, 3)
        assert blade.is_blade(1e-7)
        assert exhaustive_blade(blade, 1e-7)


@pytest.mark.parametrize("n", [2, 3])
def test_active_decisions_match_exhaustive_homogeneous_small_algebras(n):
    rng = np.random.default_rng(2100 + n)
    for k in range(2, 2 * n - 1):
        codes = [code for code in range(4 ** n) if blade_mask(n, code).bit_count() == k]
        for _ in range(35):
            selected = rng.choice(codes, size=min(7, len(codes)), replace=False)
            value = MV(n, {int(code): complex(rng.normal(), rng.normal()) for code in selected})
            assert value.is_blade() == exhaustive_blade(value)


@pytest.mark.parametrize("scale,tol", [(1, 1e-9), (1j, 1e-9), (1e-6, 1e-9), (1e-6, 1e-13)])
def test_odd_witness_and_absolute_tolerance_match_previous_decision(scale, tol):
    value = scale * (blade_from_mask(3, 0b000111) + blade_from_mask(3, 0b111000))
    assert value.wedge(value).is_zero()
    assert value.is_blade(tol) == exhaustive_blade(value, tol)
    if abs(scale) == 1:
        assert not value.is_blade(tol)


@pytest.mark.parametrize("k", [3, 12])
def test_n12_uses_only_active_contractions_without_algebra_enumeration(monkeypatch, k):
    n = 12
    common = (1 << (k - 1)) - 1
    masks = [common | (1 << j) for j in (k - 1, 22, 23)]
    value = sum((blade_from_mask(n, mask) * coefficient
                 for mask, coefficient in zip(masks, (1, 2j, -3))), MV(n))
    candidates = kernel._blade_contraction_masks(masks)
    assert len(candidates) <= k * value.nnz()
    seen = []
    original = kernel.mask_to_code

    def encode(size, mask):
        seen.append(mask)
        return original(size, mask)

    def limited_range(*args):
        assert args != (4 ** n,), "a whole algebra must never be enumerated"
        return builtins.range(*args)

    def forbidden(*args):
        pytest.fail("a whole-grade cache must never be constructed")

    monkeypatch.setattr(kernel, "range", limited_range, raising=False)
    monkeypatch.setattr(kernel, "codes_of_grade", forbidden)
    monkeypatch.setattr(kernel, "mask_to_code", encode)
    assert value.is_blade()
    assert set(seen) == candidates


def test_large_support_is_decoded_once_and_bucketed_in_linear_memberships(monkeypatch):
    masks = [sum(1 << bit for bit in bits)
             for bits in itertools.islice(itertools.combinations(range(24), 3), 1000)]
    value = MV(12, {mask_to_code(12, mask): 1 + 1j for mask in masks})
    original = kernel.blade_mask
    calls = []
    def decode(n, code):
        calls.append(code)
        return original(n, code)
    buckets = kernel._blade_contraction_buckets(value.terms, {code: original(12, code) for code in value.terms})
    assert sum(map(len, buckets.values())) == 3 * value.nnz()
    monkeypatch.setattr(kernel, "blade_mask", decode)
    value.is_blade()
    assert len(calls) == value.nnz()


def test_explicit_grade_helper_has_no_persistent_whole_grade_cache():
    assert not hasattr(kernel.codes_of_grade, "cache_info")
    assert kernel.codes_of_grade(2, 2) == tuple(code for code in range(16)
                                             if blade_mask(2, code).bit_count() == 2)
