"""The ``kernel="soa"`` product route through the exact projection layer.

``kernel="mv"`` stays the default and the only route committed records were
produced on; these tests hold the SoA route to the same words, coefficients,
matrices and adaptive decisions, to floating-point summation order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from clifford_qc import MV, clear_kernel_caches, kernel_cache_info
from clifford_qc.backends import ExactMVBackend
from clifford_qc.models import fcidump_model
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
    assert KERNELS == ("mv", "soa")


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
