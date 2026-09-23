"""The simulated group readouts the Phase 16B v3 producer prices grouping with.

v2 gave every Pauli word its own independent Gaussian. Once words share a
measurement setting that is an unlabelled approximation, because they are read
from the same shots. v3 replaces it with a simulation, and the simulation is
exact rather than asymptotic: for a computational-basis reference the per-shot
sign of every word in a commuting group is a fixed sign times a product of `k`
independent fair coins, so the whole group's `n`-shot sample means come from one
Walsh-transformed multinomial draw whatever `n` is.

That is a claim about the operator algebra, and these tests are what stops it
from being a claim about the producer's intentions. The central one is
`test_construction_matches_literal_readouts`: it draws literal bitstrings in a
group's product basis and requires the construction's predicted sign to match
the drawn one shot for shot, not on average. The rest check the pieces it rests
on -- the transform, the closed-form expectations, the covariance structure, the
exact split that keeps a 1e20-shot budget inside int64 -- and that the
verification has teeth when the construction is wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from benchmarks.run_phase16b_v3_feasibility import (
    MAX_SHOTS,
    ReadoutPlan,
    basis_state_expectation,
    least_favourable,
    partition_words,
    verify_group_construction,
    walsh,
    xz_parts,
)
from clifford_qc.dense_reference import to_matrix
from clifford_qc.multivector import MV

N = 4
BITS = 0b0011  # |1100> in qubit order, the two-electron Aufbau determinant


def reference_density(n, bits):
    psi = np.zeros(1 << n, dtype=complex)
    index = 0
    for q in range(n):
        if (bits >> q) & 1:
            index |= 1 << (n - 1 - q)
    psi[index] = 1.0
    return np.outer(psi, psi.conj())


def all_words(n):
    return list(range(4 ** n))


def test_walsh_matches_its_definition():
    rng = np.random.default_rng(0)
    for k in range(1, 5):
        counts = rng.integers(0, 97, size=(3, 2, 1 << k))
        want = np.zeros_like(counts)
        for a in range(1 << k):
            for v in range(1 << k):
                want[..., a] += counts[..., v] * (-1) ** bin(a & v).count("1")
        assert np.array_equal(walsh(counts), want)


def test_basis_state_expectation_matches_the_dense_trace():
    density = reference_density(N, BITS)
    for code in all_words(N):
        dense = float(np.trace(density @ to_matrix(MV(N, {code: 1.0}))).real)
        assert basis_state_expectation(N, code, BITS) == pytest.approx(dense, abs=1e-12)


def test_x_part_decides_whether_a_word_is_deterministic():
    for code in all_words(N):
        x, _ = xz_parts(N, code)
        value = basis_state_expectation(N, code, BITS)
        assert (value == 0.0) == bool(x)


@pytest.mark.parametrize("scheme", ["ungrouped", "qwc", "fully_commuting"])
def test_plan_covers_every_traceless_word_once(scheme):
    codes = [code for code in all_words(N) if code]
    plan = ReadoutPlan(N, codes, partition_words(N, codes, scheme), BITS)
    assigned = [code for group in plan.groups for code in group]
    assert sorted(assigned) == sorted(codes)
    assert plan.settings == len(plan.groups)


@pytest.mark.parametrize("scheme", ["ungrouped", "qwc"])
def test_construction_matches_literal_readouts(scheme):
    """Shot for shot, not on average: the identity the simulation rests on."""
    codes = [code for code in all_words(N) if code]
    plan = ReadoutPlan(N, codes, partition_words(N, codes, scheme), BITS)
    report = verify_group_construction(plan, 128, np.random.default_rng(3),
                                       literal_groups=len(plan.groups))
    assert report["covariance_mismatches"] == 0
    assert report["literal_groups_checked"] == len(plan.groups)
    assert report["literal_words_checked"] == len(codes)


def test_fully_commuting_groups_satisfy_the_covariance_identity():
    codes = [code for code in all_words(N) if code]
    plan = ReadoutPlan(N, codes, partition_words(N, codes, "fully_commuting"), BITS)
    report = verify_group_construction(plan, 0, np.random.default_rng(4))
    assert report["covariance_pairs_checked"] > 0
    assert report["covariance_mismatches"] == 0


def test_literal_check_rejects_a_corrupted_random_sign():
    """The teeth are in the shot-for-shot comparison, not in the second moments.

    Flipping `c` on a word that is alone in its x-class leaves every second
    moment alone -- `chi` is symmetric, and no other word shares its class -- so
    the closed-form covariance identity cannot see it. The literal readout can,
    because it re-derives the sign from drawn outcomes.
    """
    codes = [code for code in all_words(N) if code]
    plan = ReadoutPlan(N, codes, partition_words(N, codes, "qwc"), BITS)
    victim = next(index for index in range(len(plan.codes))
                  if plan.word_index[index] != 0)
    plan.word_sign[victim] *= -1
    with pytest.raises(RuntimeError, match="disagree shot for shot"):
        verify_group_construction(plan, 64, np.random.default_rng(5),
                                  literal_groups=len(plan.groups))


def test_closed_form_check_rejects_a_corrupted_deterministic_sign():
    """A Z-type word's sign is its expectation, and that is checked directly."""
    codes = [code for code in all_words(N) if code]
    plan = ReadoutPlan(N, codes, partition_words(N, codes, "fully_commuting"), BITS)
    victim = next(index for index in range(len(plan.codes))
                  if plan.word_index[index] == 0)
    plan.word_sign[victim] *= -1
    with pytest.raises(RuntimeError, match="covariance identities failed"):
        verify_group_construction(plan, 0, np.random.default_rng(6))


def test_sampled_means_are_unbiased_and_carry_the_group_covariance():
    codes = [code for code in all_words(N) if code]
    plan = ReadoutPlan(N, codes, partition_words(N, codes, "qwc"), BITS)
    shots, replicas = 4096, 6000
    draws = plan.sample(shots, replicas, np.random.default_rng(11))
    assert np.max(np.abs(draws.mean(axis=0) - plan.exact)) < 6 / np.sqrt(shots * replicas)

    predicted = np.zeros((len(plan.codes), len(plan.codes)))
    for i in range(len(plan.codes)):
        for j in range(len(plan.codes)):
            same_group = plan.word_group[i] == plan.word_group[j]
            random = plan.word_index[i] != 0
            if same_group and random and plan.word_index[i] == plan.word_index[j]:
                predicted[i, j] = plan.word_sign[i] * plan.word_sign[j]
    empirical = np.cov(draws, rowvar=False) * shots
    assert np.max(np.abs(empirical - predicted)) < 8 / np.sqrt(replicas)


def test_deterministic_words_are_read_without_noise():
    """A Z-type word costs shots but carries none: its readout cannot miss."""
    codes = [code for code in all_words(N) if code]
    plan = ReadoutPlan(N, codes, partition_words(N, codes, "qwc"), BITS)
    draws = plan.sample(64, 32, np.random.default_rng(13))
    deterministic = plan.word_index == 0
    assert deterministic.any()
    assert np.array_equal(draws[:, deterministic],
                          np.tile(plan.exact[deterministic], (32, 1)))


@pytest.mark.parametrize("shots", [1, 10 ** 4, 10 ** 18, 10 ** 19, 10 ** 20,
                                   MAX_SHOTS, MAX_SHOTS + 1])
def test_shot_chunks_split_exactly(shots):
    chunks = ReadoutPlan.shot_chunks(shots)
    assert sum(chunks) == shots
    assert all(0 < part <= MAX_SHOTS for part in chunks)


def test_a_budget_beyond_int64_still_samples_with_the_right_scale():
    """The 1e20 end of the declared grid, where one multinomial would overflow."""
    codes = [code for code in all_words(N) if code]
    plan = ReadoutPlan(N, codes, partition_words(N, codes, "fully_commuting"), BITS)
    shots = 10 ** 20
    assert len(ReadoutPlan.shot_chunks(shots)) > 1
    draws = plan.sample(shots, 400, np.random.default_rng(17))
    random = plan.word_index != 0
    spread = draws[:, random].std(axis=0)
    assert np.all(np.isfinite(draws))
    assert np.allclose(spread, 1 / math.sqrt(shots), rtol=0.3)
    assert np.array_equal(draws[:, ~random],
                          np.tile(plan.exact[~random], (400, 1)))


def test_grouping_reduces_settings_monotonically():
    codes = [code for code in all_words(N) if code]
    counts = {scheme: len(partition_words(N, codes, scheme))
              for scheme in ("ungrouped", "qwc", "fully_commuting")}
    assert counts["ungrouped"] > counts["qwc"] > counts["fully_commuting"]


def test_least_favourable_scheme_decides_the_verdict():
    assert least_favourable({"qwc": "GO", "fully_commuting": "CONDITIONAL"}) == (
        "fully_commuting", "CONDITIONAL")
    assert least_favourable({"qwc": "NO_GO", "fully_commuting": "CONDITIONAL"}) == (
        "qwc", "NO_GO")
    assert least_favourable({"qwc": "GO", "fully_commuting": "GO"})[1] == "GO"


# --------------------------------------------------------------------------
# The v3 producer reuses v2's solvers unchanged. It did not, at first: three of
# them were rewritten from memory, and two of the rewrites were wrong. `whiten`
# intersected an eigenvector mask with a basis mask, and `solve_unitary` dropped
# the rotation of the correlators by the energy shift and built an (m-1) pencil
# instead of an m one. The cost was not subtle -- rt_unitary's *zero-noise*
# error went from 6e-10 to 0.67 Hartree, so the candidate never reached the
# target and the first v3 run returned NO_GO everywhere, ungrouped baseline
# included.
#
# What caught it was the continuity check against v2, which is why the design
# keeps an ungrouped scheme it has no other use for. These tests catch it
# earlier and without a two-hour run: the shared solvers must agree with v2's
# bit for bit on the same input.
# --------------------------------------------------------------------------

from benchmarks import run_phase16b_v2_feasibility as v2  # noqa: E402
from benchmarks import run_phase16b_v3_feasibility as v3  # noqa: E402


def _pencil(seed, size, rank=None):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    if rank is not None:
        A[:, rank:] = A[:, :1]
    S = A.conj().T @ A
    B = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    H = 0.5 * (B + B.conj().T)
    return S, H


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("policy", [
    {"cutoff_relative": 1e-13}, {"cutoff_relative": 1e-6},
    {"floor_absolute": 1e-8}, {"cutoff_relative": 1e-13, "ridge": 1e-8}])
def test_whiten_matches_the_v2_producer(seed, policy):
    S, _ = _pencil(seed, 6, rank=4 if seed % 2 else None)
    want, want_rank = v2.whiten(S, **policy)
    got, got_rank = v3.whiten(S, **policy)
    assert got_rank == want_rank
    assert (got is None) == (want is None)
    if want is not None:
        assert np.allclose(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("seed", range(6))
def test_solve_hermitian_matches_the_v2_producer(seed):
    S, H = _pencil(seed, 6, rank=4 if seed % 2 else None)
    assert v3.solve_hermitian(S, H, cutoff_relative=1e-13) == v2.solve_hermitian(
        S, H, cutoff_relative=1e-13)


@pytest.mark.parametrize("seed", range(6))
def test_solve_unitary_matches_the_v2_producer(seed):
    rng = np.random.default_rng(seed)
    m, dt, shift = 6, 0.55, 0.67
    phases = rng.uniform(-1.0, 1.0, size=4)
    weights = rng.uniform(0.1, 1.0, size=4)
    weights /= weights.sum()
    c = {0: complex(1.0)}
    for k in range(1, m + 2):
        c[k] = complex(sum(w * np.exp(-1j * p * k * dt) for w, p in zip(weights, phases)))
        c[-k] = c[k].conjugate()
    want = v2.solve_unitary(c, m, dt, shift, cutoff_relative=1e-13)
    got = v3.solve_unitary(c, m, dt, shift, cutoff_relative=1e-13)
    assert got == want


def test_solve_unitary_recovers_a_planted_ground_energy():
    """A correctness anchor, not just agreement: the two could be wrong together."""
    m, dt, shift = 8, 0.55, 0.67
    energies = np.array([-2.18, -1.4, 0.3, 1.9])
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    c = {0: complex(1.0)}
    for k in range(1, m + 2):
        # The correlator carries the unshifted energies, as `exact_lags`
        # produces it; `solve_unitary` is what rotates by the shift.
        c[k] = complex(sum(w * np.exp(-1j * e * k * dt)
                           for w, e in zip(weights, energies)))
        c[-k] = c[k].conjugate()
    energy, rank = v3.solve_unitary(c, m, dt, shift, cutoff_relative=1e-13)
    assert rank >= len(energies)
    assert energy == pytest.approx(energies.min(), abs=1e-8)
