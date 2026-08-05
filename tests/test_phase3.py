"""Phase 3: QWC grouping (backlog 12), layered/subpool ADAPT (13), and the
random-selection baseline."""

import random

import numpy as np
import pytest

from clifford_qc.ir import PauliWord
from clifford_qc.pauli import comm
from clifford_qc.states import expectation
from clifford_qc.models import tfim, random_ising
from clifford_qc.backends import FiniteShotBackend
from clifford_qc.measurement import (
    CommutatorBank, UniformDoubling, qubit_wise_commute, qwc_groups, shared_basis,
)
from clifford_qc.algorithms import (
    ConfidenceSelector, RandomSelector, build_layer, local_pool, run_adapt,
    words_commute,
)
from clifford_qc.selection import SelectionStatus


# ---------------------------------------------------------------------------
# QWC grouping


def test_qubit_wise_commute_definition():
    w = PauliWord.from_label
    assert qubit_wise_commute(w("XIZ"), w("XYZ"))
    assert qubit_wise_commute(w("XII"), w("IYZ"))
    assert not qubit_wise_commute(w("XIZ"), w("ZIZ"))
    # qwc is strictly stronger than commutation: XX and YY commute but not qwc
    assert words_commute(w("XX"), w("YY"))
    assert not qubit_wise_commute(w("XX"), w("YY"))


def test_qwc_groups_partition_is_valid():
    m = tfim(5)
    pool = local_pool(5, periodic_context=False)
    bank = CommutatorBank(m.hamiltonian, [op.word for op in pool])
    groups = qwc_groups(bank.words)
    assert sum(len(g) for g in groups) == len(bank.words)
    assert len(groups) < len(bank.words)  # actual compression happens
    for group in groups:
        for a in group:
            for b in group:
                assert qubit_wise_commute(a, b)
        shared_basis(group)  # consistent basis must exist


def test_shared_basis_rejects_non_qwc_group():
    with pytest.raises(ValueError, match="not qubit-wise"):
        shared_basis([PauliWord.from_label("XI"), PauliWord.from_label("ZI")])


def test_grouped_sampling_matches_exact_marginals():
    """Backlog 12 acceptance: grouped and ungrouped expectations agree."""
    m = tfim(4, J=1.3, h=0.8)
    pool = local_pool(4, periodic_context=False)
    bank = CommutatorBank(m.hamiltonian, [op.word for op in pool])
    prog = m.reference
    rho = FiniteShotBackend(seed=0).state(prog, ())
    groups = qwc_groups(bank.words)
    batch = FiniteShotBackend(seed=1).sample_grouped_from_state(rho, groups, 300_000)
    assert batch.circuits == len(groups)
    for w in bank.words:
        exact = expectation(rho, w.to_mv()).real
        assert batch.mean(w.code) == pytest.approx(exact, abs=0.012)


def test_grouped_sampling_deterministic_by_seed():
    m = tfim(3)
    pool = local_pool(3, periodic_context=False)
    bank = CommutatorBank(m.hamiltonian, [op.word for op in pool])
    rho = m.reference.state()
    groups = qwc_groups(bank.words)
    a = FiniteShotBackend(seed=7).sample_grouped_from_state(rho, groups, 500)
    b = FiniteShotBackend(seed=7).sample_grouped_from_state(rho, groups, 500)
    assert a.plus_counts == b.plus_counts


# ---------------------------------------------------------------------------
# Commutation and layering


def test_words_commute_matches_mv_commutator():
    rng = random.Random(0)
    for _ in range(50):
        n = rng.choice([2, 3, 4])
        a = PauliWord(n, rng.randrange(4 ** n))
        b = PauliWord(n, rng.randrange(4 ** n))
        mv_commutes = not comm(a.to_mv(), b.to_mv()).terms
        assert words_commute(a, b) == mv_commutes


def test_build_layer_respects_alpha_and_commutation():
    pool = local_pool(4, periodic_context=False)
    scores = {j: 1.0 / (j + 1) for j in range(len(pool))}
    layer = build_layer(pool, scores, 0, list(range(len(pool))), alpha=0.4)
    assert layer[0] == 0
    for j in layer[1:]:
        assert abs(scores[j]) >= 0.4 * abs(scores[0])
    for a in layer:
        for b in layer:
            assert words_commute(pool[a].word, pool[b].word)
    with pytest.raises(ValueError, match="alpha"):
        build_layer(pool, scores, 0, [0, 1], alpha=0.0)


def test_layered_exact_adapt_energy_non_increasing():
    """Backlog 13 acceptance: exact energy remains non-increasing, and
    layering converges in fewer selection rounds (with repeats allowed)."""
    m = tfim(4, 1.0, 1.0)
    res = run_adapt(m, local_pool(4, periodic_context=False),
                    max_operators=14, layer_alpha=0.5, allow_repeats=True)
    energies = [r.energy for r in res.records if r.energy is not None]
    assert all(b <= a + 1e-9 for a, b in zip(energies, energies[1:]))
    steps = [r for r in res.records if r.selected_label]
    assert len(res.labels) > len(steps)   # layers actually formed
    assert any(r.layer_labels for r in steps)
    assert res.relative_error < 1e-9
    assert len(steps) < 6                 # sequential ADAPT needs 6 rounds


def test_layered_adapt_without_repeats_stalls_at_symmetric_point():
    """Known behavior worth pinning: alpha-layering pulls in mirror partners,
    and without operator repeats the symmetric ansatz reaches a stationary
    point where every remaining candidate gradient vanishes."""
    m = tfim(4, 1.0, 1.0)
    res = run_adapt(m, local_pool(4, periodic_context=False),
                    max_operators=14, layer_alpha=0.5, allow_repeats=False)
    assert res.stopped_reason == "exact gradient below threshold"
    assert 1e-3 < res.relative_error < 1e-1
    energies = [r.energy for r in res.records if r.energy is not None]
    assert all(b <= a + 1e-9 for a, b in zip(energies, energies[1:]))


# ---------------------------------------------------------------------------
# Subpool exploration


def test_subpool_adapt_converges_with_redraw():
    m = tfim(4, 1.0, 1.0)
    res = run_adapt(m, local_pool(4, periodic_context=False),
                    max_operators=10, subpool_size=6, subpool_seed=1)
    assert res.relative_error < 1e-9
    for rec in res.records:
        assert rec.active_candidates <= 6


def test_subpool_size_validation():
    m = tfim(3)
    with pytest.raises(ValueError, match="subpool_size"):
        run_adapt(m, local_pool(3), subpool_size=0)


def test_noisy_subpool_grouped_run():
    m = tfim(3, 1.0, 1.0)
    res = run_adapt(m, local_pool(3, periodic_context=False),
                    backend=FiniteShotBackend(seed=4),
                    selector=ConfidenceSelector(delta=0.05, near_tol=0.05),
                    allocator=UniformDoubling(base=128, max_factor=32),
                    max_operators=4, grouping=True, subpool_size=4)
    assert res.relative_error < 1e-2
    assert res.total_circuits > 0
    ungrouped = run_adapt(m, local_pool(3, periodic_context=False),
                          backend=FiniteShotBackend(seed=4),
                          selector=ConfidenceSelector(delta=0.05, near_tol=0.05),
                          allocator=UniformDoubling(base=128, max_factor=32),
                          max_operators=4, subpool_size=4)
    assert res.total_circuits < ungrouped.total_circuits


# ---------------------------------------------------------------------------
# Random baseline


def test_random_selector_baseline_reproducible():
    m = random_ising(3, seed=2)
    pool = local_pool(3, periodic_context=False)
    a = run_adapt(m, pool, selector=RandomSelector(seed=9), max_operators=5)
    b = run_adapt(m, pool, selector=RandomSelector(seed=9), max_operators=5)
    assert a.labels == b.labels
    assert a.total_shots == 0
    assert all(r.status.value == "random" for r in a.records if r.selected_label)
    energies = [r.energy for r in a.records if r.energy is not None]
    assert all(y <= x + 1e-9 for x, y in zip(energies, energies[1:]))


def test_random_baseline_does_not_use_exact_gradients_to_stop():
    m = random_ising(3, seed=2)
    result = run_adapt(
        m, local_pool(3, periodic_context=False),
        selector=RandomSelector(seed=9), max_operators=1,
        threshold=1e9, track_exact_scores=False)
    assert len(result.labels) == 1
    assert result.records[0].status is SelectionStatus.RANDOM
    assert result.records[0].exact_gradient is None
