"""Unit, property, and oracle-parity tests for clifford_qc.soa_kernel."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from clifford_qc import MV, clear_kernel_caches, kernel_cache_info
from clifford_qc.models import fcidump_model
from clifford_qc.pauli_kernel import word_mul_reference
from clifford_qc.reference_states import DeterminantReference
from clifford_qc.soa_kernel import (
    GlobalSoABank,
    PauliSoA,
    SoAWordTable,
    array_to_int_codes,
    available_gpu_backends,
    build_soa_bank,
    code_dtype_for_n,
    int_codes_to_array,
    popcount_array,
    soa_add,
    soa_anticommute_mask,
    soa_commutator,
    soa_commutator_terms,
    soa_determinant_expectation,
    soa_hs_product,
    soa_mul,
    soa_mul_terms,
    soa_trace_pairing,
    word_width_for_n,
)
from clifford_qc.dense_reference import density_from_statevector
from clifford_qc.subspace import MatrixElementBank, identity_generator
from clifford_qc.subspace.packed import GlobalWordTable
from clifford_qc.subspace.symmetry_pools import (
    closed_shell_occupied_spatial,
    spin_adapted_excitations,
)

ROOT = Path(__file__).resolve().parents[1]


def _random_terms(rng: np.random.Generator, n: int, count: int) -> dict[int, complex]:
    if n == 0:
        return {0: complex(rng.normal(), rng.normal())}
    if n <= 12:
        codes = rng.choice(1 << (2 * n), size=min(count, 1 << (2 * n)), replace=False).tolist()
    else:
        seen: set[int] = set()
        while len(seen) < count:
            # Generate 2*n random bits across 32-bit chunks for arbitrary n.
            val = 0
            rem = 2 * n
            shift = 0
            while rem > 0:
                bits = min(30, rem)
                val |= int(rng.integers(0, 1 << bits)) << shift
                shift += bits
                rem -= bits
            seen.add(val)
        codes = list(seen)
    coeffs = rng.normal(size=len(codes)) + 1j * rng.normal(size=len(codes))
    return {int(c): complex(v) for c, v in zip(codes, coeffs)}


def _assert_terms_close(actual: dict[int, complex], expected: dict[int, complex], atol: float = 1e-10) -> None:
    assert set(actual) == set(expected)
    for k, exp_v in expected.items():
        act_v = actual[k]
        assert abs(act_v - exp_v) <= atol, f"code {k}: got {act_v}, expected {exp_v}"


def _assert_no_negative_zeros(arr: np.ndarray) -> None:
    assert not bool(np.any(np.signbit(arr.real) & (arr.real == 0.0)))
    assert not bool(np.any(np.signbit(arr.imag) & (arr.imag == 0.0)))


@pytest.mark.parametrize(
    "n,expected_width,expected_dtype",
    [
        (0, 0, np.uint16),
        (1, 0, np.uint16),
        (8, 0, np.uint16),
        (9, 0, np.uint32),
        (16, 0, np.uint32),
        (17, 0, np.uint64),
        (31, 0, np.uint64),
        (32, 0, np.uint64),
        (33, 2, np.uint64),
        (64, 2, np.uint64),
        (65, 3, np.uint64),
    ],
)
def test_width_and_dtype_selection(n: int, expected_width: int, expected_dtype: type) -> None:
    assert word_width_for_n(n) == expected_width
    assert code_dtype_for_n(n) == np.dtype(expected_dtype)


@pytest.mark.parametrize("n", [0, 1, 4, 8, 9, 16, 17, 31, 32, 33, 40, 64, 70])
def test_code_roundtrip_and_canonical_sort_order(n: int) -> None:
    rng = np.random.default_rng(100 + n)
    terms = _random_terms(rng, n, 25 if n > 0 else 1)
    # Include extreme boundary codes when n > 0.
    if n > 0:
        terms[0] = 1.25 - 0.5j
        terms[(1 << (2 * n)) - 1] = -0.75 + 0.25j
    soa = PauliSoA.from_terms(n, terms)
    ints = array_to_int_codes(n, soa.codes)
    assert ints == sorted(terms)
    assert soa.to_terms() == {k: terms[k] for k in sorted(terms)}
    _assert_no_negative_zeros(soa.coeffs)


def test_negative_zero_canonicalization_and_pruning() -> None:
    codes = np.array([0, 1, 2, 3], dtype=np.uint16)
    coeffs = np.array(
        [
            complex(-0.0, -0.0),
            complex(1.0, -0.0),
            complex(-0.0, -2.0),
            complex(1e-13, -1e-13),
        ],
        dtype=np.complex128,
    )
    soa = PauliSoA.from_arrays(2, codes, coeffs)
    assert soa.nnz() == 2
    assert soa.codes.tolist() == [1, 2]
    _assert_no_negative_zeros(soa.coeffs)
    # Also check via multiplication producing exact real/imag zeros.
    a = PauliSoA.from_labels(2, {"XI": 1.0, "YI": 1j})
    b = PauliSoA.from_labels(2, {"XI": -1.0, "ZI": -1j})
    prod = a * b
    _assert_no_negative_zeros(prod.coeffs)


def test_popcount_fallback_matches_bitwise_count(monkeypatch: pytest.MonkeyPatch) -> None:
    import clifford_qc.soa_kernel as sk

    rng = np.random.default_rng(7)
    for dtype in (np.uint16, np.uint32, np.uint64):
        info = np.iinfo(dtype)
        arr = rng.integers(0, info.max, size=(17, 13), dtype=dtype)
        ref = np.array([[bin(int(v)).count("1") for v in row] for row in arr], dtype=np.int16)
        assert np.array_equal(popcount_array(arr), ref)
        monkeypatch.setattr(sk, "_HAS_BITWISE_COUNT", False)
        assert np.array_equal(popcount_array(arr), ref)
        monkeypatch.setattr(sk, "_HAS_BITWISE_COUNT", hasattr(np, "bitwise_count"))


@pytest.mark.parametrize("n", [0, 1, 4, 8, 9, 12, 16, 17, 31, 32, 33, 40, 64])
@pytest.mark.parametrize("chunk_pairs", [5, 1000])
def test_soa_mul_matches_mv_oracle_across_widths_and_chunks(n: int, chunk_pairs: int) -> None:
    rng = np.random.default_rng(1000 + n * 17 + chunk_pairs)
    t = 1 if n == 0 else (18 if n <= 16 else 12)
    ta = _random_terms(rng, n, t)
    tb = _random_terms(rng, n, t)
    mv_a = MV(n, ta)
    mv_b = MV(n, tb)
    expected = (mv_a * mv_b).terms

    sa = PauliSoA.from_mv(mv_a)
    sb = PauliSoA.from_mv(mv_b)

    # Clear kernel memo and verify SoA never populates it.
    clear_kernel_caches()
    prod_sort = soa_mul(sa, sb, accumulate="sort", chunk_pairs=chunk_pairs)
    assert kernel_cache_info()["word_products"]["currsize"] == 0
    _assert_terms_close(prod_sort.to_terms(), expected)
    _assert_no_negative_zeros(prod_sort.coeffs)

    if n <= 12:
        prod_dense = soa_mul(sa, sb, accumulate="dense", chunk_pairs=chunk_pairs)
        _assert_terms_close(prod_dense.to_terms(), expected)
        _assert_no_negative_zeros(prod_dense.coeffs)

    prod_auto = soa_mul(sa, sb, accumulate="auto", chunk_pairs=chunk_pairs)
    _assert_terms_close(prod_auto.to_terms(), expected)
    _assert_terms_close(soa_mul_terms(n, ta, tb), expected)


def test_soa_mul_single_word_exhaustive_2q_against_word_mul_reference() -> None:
    n = 2
    for a in range(16):
        for b in range(16):
            ref_ph, ref_c = word_mul_reference(n, a, b)
            prod = PauliSoA.from_terms(n, {a: 1.0}) * PauliSoA.from_terms(n, {b: 1.0})
            assert prod.to_terms() == {ref_c: ref_ph}


@pytest.mark.parametrize("n", [2, 6, 8, 12, 17, 32, 33, 48, 64])
@pytest.mark.parametrize("chunk_pairs", [7, 500])
def test_soa_commutator_matches_mv_ab_minus_ba(n: int, chunk_pairs: int) -> None:
    rng = np.random.default_rng(2000 + n * 13 + chunk_pairs)
    t = 16 if n <= 16 else 12
    ta = _random_terms(rng, n, t)
    tb = _random_terms(rng, n, t)
    mv_a = MV(n, ta)
    mv_b = MV(n, tb)
    expected = (mv_a * mv_b - mv_b * mv_a).terms

    sa = PauliSoA.from_mv(mv_a)
    sb = PauliSoA.from_mv(mv_b)

    # Check anticommutation mask directly against word_mul_reference.
    mask = soa_anticommute_mask(n, sa.codes, sb.codes)
    ia_list = array_to_int_codes(n, sa.codes)
    ib_list = array_to_int_codes(n, sb.codes)
    for i, ca in enumerate(ia_list[:5]):
        for j, cb in enumerate(ib_list[:5]):
            p1, _ = word_mul_reference(n, ca, cb)
            p2, _ = word_mul_reference(n, cb, ca)
            assert bool(mask[i, j]) == (p1 != p2)

    clear_kernel_caches()
    comm_sort = soa_commutator(sa, sb, accumulate="sort", chunk_pairs=chunk_pairs)
    assert kernel_cache_info()["word_products"]["currsize"] == 0
    _assert_terms_close(comm_sort.to_terms(), expected)
    if n <= 12:
        comm_dense = soa_commutator(sa, sb, accumulate="dense", chunk_pairs=chunk_pairs)
        _assert_terms_close(comm_dense.to_terms(), expected)
    _assert_terms_close(soa_commutator_terms(n, ta, tb, tol=1e-12), expected)


@pytest.mark.parametrize("n", [2, 8, 17, 33, 64])
def test_sorted_intersections_add_and_norms(n: int) -> None:
    rng = np.random.default_rng(3000 + n)
    shared = _random_terms(rng, n, 8)
    only_a = _random_terms(rng, n, 10)
    only_b = _random_terms(rng, n, 15)
    ta = {**shared, **only_a}
    tb = {**shared, **only_b}
    mv_a = MV(n, ta)
    mv_b = MV(n, tb)
    sa = PauliSoA.from_mv(mv_a)
    sb = PauliSoA.from_mv(mv_b)

    assert abs(soa_trace_pairing(sa, sb) - mv_a.trace_pairing(mv_b)) < 1e-11
    assert abs(soa_trace_pairing(sb, sa) - mv_b.trace_pairing(mv_a)) < 1e-11
    assert abs(soa_hs_product(sa, sb) - mv_a.hs_product(mv_b)) < 1e-11
    assert abs(soa_hs_product(sb, sa) - mv_b.hs_product(mv_a)) < 1e-11
    assert abs(sa.scalar_part() - mv_a.scalar_part()) < 1e-12
    assert abs(sa.trace() - mv_a.trace()) < 1e-10
    assert math.isclose(sa.norm_hs(), mv_a.norm_hs(), rel_tol=1e-12, abs_tol=1e-12)
    _assert_terms_close(soa_add(sa, sb).to_terms(), (mv_a + mv_b).terms)
    _assert_terms_close((sa - sb).to_terms(), (mv_a - mv_b).terms)
    _assert_terms_close(sa.dagger().to_terms(), mv_a.dagger().terms)


@pytest.mark.parametrize("n,bits", [(4, "1010"), (8, "11110000"), (33, "1" * 10 + "0" * 23), (40, "10" * 20)])
def test_determinant_expectation_matches_determinant_reference(n: int, bits: str) -> None:
    rng = np.random.default_rng(4000 + n)
    # Include diagonal Z-only strings alongside general strings.
    terms = _random_terms(rng, n, 20)
    for _ in range(10):
        z_code = 0
        for q in range(n):
            if rng.random() < 0.4:
                z_code |= 3 << (2 * q)
        terms[z_code] = complex(rng.normal(), rng.normal())
    ref = DeterminantReference(bits)
    mv = MV(n, terms)
    expected = ref.evaluate_terms(mv.terms.items(), row_nnz=mv.nnz()).value
    soa = PauliSoA.from_mv(mv)
    assert abs(soa_determinant_expectation(soa, ref) - expected) < 1e-11
    assert abs(soa_determinant_expectation(soa, bits) - expected) < 1e-11
    mask = sum(1 << q for q, ch in enumerate(bits) if ch == "1")
    assert abs(soa_determinant_expectation(soa, mask) - expected) < 1e-11


def test_soa_word_table_and_global_word_table_31q_guard() -> None:
    # GlobalWordTable explicitly refuses codes > 2**63 - 1 (n > 31).
    legacy = GlobalWordTable()
    assert legacy.intern((1 << 62) - 1) == 0
    with pytest.raises(OverflowError, match="SoAWordTable"):
        legacy.intern(1 << 63)
    with pytest.raises(OverflowError, match="SoAWordTable"):
        legacy.intern_many([0, 1 << 63])
    with pytest.raises(OverflowError, match="SoAWordTable"):
        legacy.intern(-1)

    # SoAWordTable handles n = 40 and n = 64 seamlessly.
    for n in (32, 40, 64):
        c1 = int_codes_to_array(n, [0, (1 << (2 * n)) - 1, 1 << (2 * (n - 1))])
        c2 = int_codes_to_array(n, [5, 0, (1 << (2 * n)) - 1])
        table = SoAWordTable.from_code_arrays(n, [c1, c2])
        assert len(table) == 4
        assert table.int_codes() == sorted({0, 5, 1 << (2 * (n - 1)), (1 << (2 * n)) - 1})
        assert table.lookup((1 << (2 * n)) - 1) == 3
        assert table.lookup(7) is None


def test_build_soa_bank_matches_matrix_element_bank_on_h4() -> None:
    model = fcidump_model(ROOT / "benchmarks/data/h4_sto3g_r0.9.FCIDUMP")
    n_spatial = model.n // 2
    occ_spin = tuple(model.metadata["reference_occupied_spin_orbitals"])
    hf_bits = "".join("1" if q in occ_spin else "0" for q in range(model.n))
    occ_spatial = closed_shell_occupied_spatial(
        occ_spin,
        n_spatial,
        spin_ordering="interleaved",
    )
    gens = [identity_generator(model.n), *spin_adapted_excitations(n_spatial, occ_spatial)[:5]]

    # 1. Exact determinant reference parity
    det_ref = DeterminantReference(hf_bits)
    legacy_bank = MatrixElementBank(det_ref, model.hamiltonian, gens)
    S_ref, H_ref = legacy_bank.matrices()

    clear_kernel_caches()
    soa_bank: GlobalSoABank = build_soa_bank(model.hamiltonian, gens)
    assert kernel_cache_info()["word_products"]["currsize"] == 0
    assert soa_bank.word_universe_size == len(legacy_bank.word_set())
    assert set(soa_bank.word_table.int_codes()) == set(legacy_bank.word_set())

    S_soa, H_soa = soa_bank.evaluate_determinant(det_ref)
    assert np.allclose(S_soa, S_ref, atol=1e-11, rtol=1e-11)
    assert np.allclose(H_soa, H_ref, atol=1e-11, rtol=1e-11)

    # 2. Correlated superposition density reference parity (8-qubit ket)
    psi = np.zeros(1 << model.n, dtype=np.complex128)
    idx_a = int(hf_bits, 2)
    idx_b = int("00111100", 2)
    psi[idx_a] = math.cos(0.3)
    psi[idx_b] = np.exp(0.2j) * math.sin(0.3)
    rho = density_from_statevector(psi)
    legacy_rho = MatrixElementBank(rho, model.hamiltonian, gens)
    S_rho_ref, H_rho_ref = legacy_rho.matrices()
    S_rho_soa, H_rho_soa = soa_bank.evaluate_density_soa(PauliSoA.from_mv(rho))
    assert np.allclose(S_rho_soa, S_rho_ref, atol=1e-11, rtol=1e-11)
    assert np.allclose(H_rho_soa, H_rho_ref, atol=1e-11, rtol=1e-11)


def test_gpu_backends_match_numpy_when_available() -> None:
    gpus = available_gpu_backends()
    if not gpus:
        with pytest.raises(RuntimeError, match="unavailable"):
            soa_mul(PauliSoA.scalar(2, 1.0), PauliSoA.scalar(2, 1.0), backend="cupy")
        return

    rng = np.random.default_rng(555)
    for backend in gpus:
        for n, t in [(8, 80), (12, 80), (16, 60)]:
            sa = PauliSoA.from_terms(n, _random_terms(rng, n, t))
            sb = PauliSoA.from_terms(n, _random_terms(rng, n, t))
            ref_mul = soa_mul(sa, sb, backend="numpy")
            ref_comm = soa_commutator(sa, sb, backend="numpy")
            for acc in ("sort", "dense") if n <= 12 else ("sort",):
                gpu_mul = soa_mul(sa, sb, backend=backend, accumulate=acc, chunk_pairs=2000)
                gpu_comm = soa_commutator(sa, sb, backend=backend, accumulate=acc, chunk_pairs=2000)
                _assert_terms_close(gpu_mul.to_terms(), ref_mul.to_terms(), atol=1e-10)
                _assert_terms_close(gpu_comm.to_terms(), ref_comm.to_terms(), atol=1e-10)
        if backend == "cupy":
            # Multiword GPU parity for n > 32
            for n in (40, 64):
                sa = PauliSoA.from_terms(n, _random_terms(rng, n, 40))
                sb = PauliSoA.from_terms(n, _random_terms(rng, n, 40))
                ref_mul = soa_mul(sa, sb, backend="numpy")
                ref_comm = soa_commutator(sa, sb, backend="numpy")
                gpu_mul = soa_mul(sa, sb, backend="cupy", chunk_pairs=600)
                gpu_comm = soa_commutator(sa, sb, backend="cupy", chunk_pairs=600)
                _assert_terms_close(gpu_mul.to_terms(), ref_mul.to_terms(), atol=1e-10)
                _assert_terms_close(gpu_comm.to_terms(), ref_comm.to_terms(), atol=1e-10)
