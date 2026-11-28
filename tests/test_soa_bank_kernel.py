"""The ``kernel="soa"`` product route through the exact projection layer.

``kernel="mv"`` stays the default and the only route committed records were
produced on; these tests hold the SoA route to the same words, coefficients,
matrices and adaptive decisions, to floating-point summation order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clifford_qc import MV, clear_kernel_caches, kernel_cache_info, ket_density
from clifford_qc.backends import ExactMVBackend
from clifford_qc.models import fcidump_model
from clifford_qc.reference_states import DeterminantReference
from clifford_qc.soa_kernel import PauliSoA
from clifford_qc.subspace import MatrixElementBank, identity_generator
from clifford_qc.subspace.adaptive import run_acase
from clifford_qc.subspace.fermionic_generators import (
    determinant_excitations,
    occupied_spin_orbitals,
)
from clifford_qc.subspace.projection import (
    DEFAULT_KERNEL,
    KERNELS,
    projected_matrices,
    solve_subspace,
)
from clifford_qc.subspace.streaming import StreamingMatrixElementBank

ROOT = Path(__file__).resolve().parents[1]
H4 = ROOT / "benchmarks" / "data" / "h4_sto3g_r0.9.FCIDUMP"


@pytest.fixture(scope="module")
def h4():
    model = fcidump_model(H4, name="h4")
    rho = ExactMVBackend().state(model.reference, ())
    pool = determinant_excitations(model.n, occupied_spin_orbitals(model), max_rank=2)
    return model, rho, pool


def _assert_same_operator(a: MV, b: MV) -> None:
    assert set(a.terms) == set(b.terms)
    for code, value in a.terms.items():
        assert abs(value - b.terms[code]) <= 1e-12


def test_default_kernel_is_the_record_route():
    assert DEFAULT_KERNEL == "mv"
    assert KERNELS == ("mv", "soa", "auto")


def test_unknown_kernel_is_rejected(h4):
    model, rho, pool = h4
    with pytest.raises(ValueError, match="kernel"):
        MatrixElementBank(rho, model.hamiltonian, kernel="dict")
    with pytest.raises(ValueError, match="kernel"):
        projected_matrices(rho, model.hamiltonian, pool[:2], kernel="gpu")


@pytest.mark.parametrize("n", [3, 12, 33])
def test_to_mv_matches_constructor_bytes(n):
    rng = np.random.default_rng(n)
    limit = 1 << min(2 * n, 62)
    codes = [int(c) for c in rng.integers(0, limit, size=40)]
    if n > 31:
        codes = [c | (1 << (2 * n - 1)) for c in codes]
    terms = {c: complex(rng.normal(), rng.normal()) for c in codes}
    terms[codes[0]] = 5e-13 + 0j  # below MV.TOL: must be pruned by to_mv
    terms[codes[1]] = complex(-0.0, 2.0)  # negative zero must not survive
    soa = PauliSoA.from_terms(n, terms, tol=0.0)
    fast = soa.to_mv()
    slow = MV(n, soa.to_terms())
    assert list(fast.terms) == sorted(fast.terms)
    assert set(fast.terms) == set(slow.terms)
    for code, value in slow.terms.items():
        got = fast.terms[code]
        assert type(got) is complex
        assert got.real.hex() == value.real.hex()
        assert got.imag.hex() == value.imag.hex()


def test_projected_matrices_agree_across_kernels(h4):
    model, rho, pool = h4
    gens = [identity_generator(model.n), *pool[:9]]
    S0, H0, r0 = projected_matrices(rho, model.hamiltonian, gens)
    S1, H1, r1 = projected_matrices(rho, model.hamiltonian, gens, kernel="soa")
    np.testing.assert_allclose(S1, S0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(H1, H0, rtol=0, atol=1e-12)
    assert "product_kernel" not in r0
    assert r1.pop("product_kernel") == "soa"
    for key in ("assemble_seconds",):
        r0.pop(key), r1.pop(key)
    assert r0 == r1

    fast = solve_subspace(rho, model.hamiltonian, gens, kernel="soa")
    slow = solve_subspace(rho, model.hamiltonian, gens)
    np.testing.assert_allclose(fast.energies, slow.energies, rtol=0, atol=1e-10)


@pytest.mark.parametrize("storage", ["object", "packed"])
def test_bank_rows_and_matrices_agree(h4, storage):
    model, rho, pool = h4
    gens = [identity_generator(model.n), *pool[:8]]
    ref = MatrixElementBank(rho, model.hamiltonian, gens)
    soa = MatrixElementBank(rho, model.hamiltonian, gens, storage=storage, kernel="soa")
    assert soa.kernel == "soa" and ref.kernel == "mv"
    S0, H0 = ref.matrices()
    S1, H1 = soa.matrices()
    np.testing.assert_allclose(S1, S0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(H1, H0, rtol=0, atol=1e-12)
    assert soa.word_set() == ref.word_set()
    for j in range(len(gens)):
        _assert_same_operator(soa._h_acted[j], ref._h_acted[j])
        for i in range(j + 1):
            _assert_same_operator(soa.overlap_operator(i, j), ref.overlap_operator(i, j))
            _assert_same_operator(soa.element_operator(i, j), ref.element_operator(i, j))
    # The word ledger is built from the same supports, so it is identical.
    assert soa._new_words == ref._new_words
    assert soa._reused_words == ref._reused_words
    assert soa.solve().resources["product_kernel"] == "soa"
    assert "product_kernel" not in ref.solve().resources


def test_soa_pair_products_do_not_touch_the_word_memo(h4):
    model, rho, pool = h4
    bank = MatrixElementBank(rho, model.hamiltonian, kernel="soa")
    clear_kernel_caches()
    bank.extend(pool[:6])
    bank.matrices()
    info = kernel_cache_info()["word_products"]
    assert info["hits"] == 0 and info["misses"] == 0 and info["currsize"] == 0


@pytest.mark.parametrize("bank_class", [MatrixElementBank, StreamingMatrixElementBank])
def test_adaptive_selection_agrees_across_kernels(h4, bank_class):
    model, rho, pool = h4
    kwargs = {"frontier_pairs": 8} if bank_class is StreamingMatrixElementBank else {}

    def grow(kernel):
        bank = bank_class(rho, model.hamiltonian, (), kernel=kernel, **kwargs)
        return run_acase(rho, model.hamiltonian, pool, bank=bank,
                         initial=[identity_generator(model.n)], max_size=5)

    reference, fast = grow("mv"), grow("soa")
    assert fast.labels == reference.labels
    assert abs(fast.energy - reference.energy) <= 1e-10
    assert fast.resources["product_kernel"] == "soa"
    assert "product_kernel" not in reference.resources


def test_to_mv_matches_python_cutoff_and_revalidates_mutated_codes():
    import math

    tol = 1e-12
    # Cover immediately below, at, and above the boundary in multiple quadrants.
    for theta in (0.0, 0.0006283185307179586, 0.4, 1.0, 2.4, 5.5):
        x, y = tol * math.cos(theta), tol * math.sin(theta)
        for real in (x, math.nextafter(x, -math.inf), math.nextafter(x, math.inf)):
            soa = PauliSoA.from_terms(1, {0: complex(real, y)}, tol=0.0)
            assert soa.to_mv().terms == MV(1, soa.to_terms()).terms
    z = complex(9.999998026079186e-13, 6.283184893762571e-16)
    soa = PauliSoA.from_terms(1, {0: z}, tol=0.0)
    assert soa.to_mv().terms == MV(1, soa.to_terms()).terms

    soa = PauliSoA.from_terms(1, {0: 1, 1: 2})
    soa.codes[0] = 4
    with pytest.raises(ValueError):
        soa.to_mv()
    # Last-write-wins is part of the historical to_terms conversion.
    soa = PauliSoA.from_terms(1, {0: 1, 1: 2})
    soa.codes[1] = 0
    soa.coeffs[1] = 0
    assert soa.to_mv().terms == MV(1, soa.to_terms()).terms == {}


@pytest.mark.parametrize("track_support", [True, False])
def test_complex_generators_and_support_free_projection(track_support):
    rho = ket_density(2, "00")
    h = MV.from_terms(2, {"ZI": 1.0, "XX": 0.2, "YY": -0.3})
    gens = [identity_generator(2), ("complex", MV.from_terms(
        2, {"XI": 0.7 + 0.3j, "IY": -0.4j, "XX": 0.1}))]
    expected_s, expected_h, _ = projected_matrices(
        rho, h, gens, track_support=track_support, kernel="mv")
    clear_kernel_caches()
    found_s, found_h, res = projected_matrices(
        rho, h, gens, track_support=track_support, kernel="soa")
    np.testing.assert_allclose(found_s, expected_s, atol=1e-12)
    np.testing.assert_allclose(found_h, expected_h, atol=1e-12)
    assert res["product_kernel"] == "soa"
    assert kernel_cache_info()["word_products"]["currsize"] == 0


def test_multiword_object_bank():
    n = 33
    rho = DeterminantReference("0" * n)
    h = MV(n, {0: 1, 3 << 64: 0.4, 1: 0.7})
    gens = [identity_generator(n), ("wide", MV(n, {1 << 64: 0.6,
                                                   2: 0.5j}))]
    ref = MatrixElementBank(rho, h, gens, kernel="mv")
    soa = MatrixElementBank(rho, h, gens, kernel="soa")
    for a, b in zip(ref.matrices(), soa.matrices()):
        np.testing.assert_allclose(a, b, atol=1e-12)
    for j in range(len(gens)):
        for i in range(j + 1):
            _assert_same_operator(soa.overlap_operator(i, j), ref.overlap_operator(i, j))
            _assert_same_operator(soa.element_operator(i, j), ref.element_operator(i, j))


def test_auto_policy_and_workflow_provenance(h4):
    rho = ket_density(1, "0")
    h = MV.from_label("Z")
    gens = [identity_generator(1), ("X", MV.from_label("X"))]
    small = MatrixElementBank(rho, h, gens, kernel="auto")
    assert small.requested_kernel == "auto" and small.kernel == "mv"
    assert small.solve().resources["requested_product_kernel"] == "auto"
    assert small.solve().resources["product_kernel"] == "mv"
    model, molecular_rho, pool = h4
    large = MatrixElementBank(molecular_rho, model.hamiltonian, kernel="auto")
    from clifford_qc.subspace.projection import AUTO_MIN_HAMILTONIAN_WORDS
    expected = "soa" if large.hamiltonian.nnz() >= AUTO_MIN_HAMILTONIAN_WORDS else "mv"
    assert large.kernel == expected
    result = run_acase(molecular_rho, model.hamiltonian, pool[:6],
                       kernel="auto", max_size=2)
    assert result.resources["requested_product_kernel"] == "auto"
    assert result.resources["product_kernel"] == expected
    with pytest.raises(ValueError, match="supplied bank"):
        run_acase(rho, h, gens[1:], bank=small, kernel="soa")
