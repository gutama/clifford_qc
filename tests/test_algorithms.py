"""Algorithm layer: pools (H1), fixed-depth VQE, exact and finite-shot
ADAPT-VQE (backlog items 5-7, 11)."""

import math

import numpy as np
import pytest

from clifford_qc.matrix import exact_ground
from clifford_qc.models import tfim, random_ising
from clifford_qc.backends import FiniteShotBackend
from clifford_qc.measurement import CommutatorBank, UniformDoubling, WordCache
from clifford_qc.algorithms import (
    ConfidenceSelector, SelectionStatus, all_words_pool, hva_program, is_odd_y,
    local_pool, minimize_energy, odd_y_filter, run_adapt, run_vqe,
)
from clifford_qc.algorithms.adapt import _ansatz_program
from clifford_qc.ir import PauliWord


# ---------------------------------------------------------------------------
# Pools / H1


def test_is_odd_y():
    assert is_odd_y(PauliWord.from_label("IYI"))
    assert is_odd_y(PauliWord.from_label("YYY"))
    assert not is_odd_y(PauliWord.from_label("YYI"))
    assert not is_odd_y(PauliWord.from_label("XZI"))


def test_local_pool_is_odd_y_and_matches_paper_size():
    pool = local_pool(4, periodic_context=True)
    assert len(pool) == 16  # 4 contexts per site at n=4, the paper's test case
    assert all(is_odd_y(op.word) for op in pool)


def test_even_y_gradients_vanish_for_real_hamiltonian_and_state():
    """The algebraic core of H1: even-Y candidates score exactly zero."""
    m = tfim(3, J=1.1, h=0.9)
    pool = all_words_pool(3, max_weight=2)
    bank = CommutatorBank(m.hamiltonian, [op.word for op in pool])
    rho = m.reference.state()
    for j, op in enumerate(pool):
        score = bank.exact_score(j, rho)
        if not is_odd_y(op.word):
            assert score == pytest.approx(0.0, abs=1e-12), op.label


def test_odd_y_restriction_preserves_exact_adapt_trajectory():
    """H1: restricted and unrestricted pools give the same exact trajectory."""
    m = random_ising(3, seed=3)  # disorder breaks ties between candidates
    full = all_words_pool(3, max_weight=2)
    restricted = odd_y_filter(full)
    assert len(restricted) < len(full)
    res_full = run_adapt(m, full, max_operators=4)
    res_restricted = run_adapt(m, restricted, max_operators=4)
    assert res_full.labels == res_restricted.labels
    assert res_full.energy == pytest.approx(res_restricted.energy, abs=1e-9)


# ---------------------------------------------------------------------------
# Optimizers


def _quadratic(x):
    g = [2 * (x[0] - 1.0), 2 * (x[1] + 2.0)]
    return (x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2, g


@pytest.mark.parametrize("method", ["lbfgsb", "adam"])
def test_minimize_energy_methods(method):
    if method == "lbfgsb":
        pytest.importorskip("scipy")
    res = minimize_energy(_quadratic, (0.0, 0.0), method=method,
                          maxiter=2000, gtol=1e-7)
    assert res.x[0] == pytest.approx(1.0, abs=1e-4)
    assert res.x[1] == pytest.approx(-2.0, abs=1e-4)
    assert res.evaluations > 0


def test_minimize_energy_bound_violation_raises():
    with pytest.raises(RuntimeError, match="variational bound"):
        minimize_energy(_quadratic, (0.0, 0.0), bound=5.0)


# ---------------------------------------------------------------------------
# Fixed-depth VQE (backlog item 5: reproduces the n=4 TFIM result)


def test_vqe_tfim_n4_reaches_ground_state():
    m = tfim(4, 1.0, 1.0)
    E0, _ = exact_ground(m.hamiltonian.to_mv())
    rng = np.random.default_rng(1)
    result = run_vqe(m, depth=4, x0=0.1 * rng.standard_normal(8), bound=E0)
    # 1e-5 accommodates the pure-Python Adam fallback used when SciPy is
    # absent (the core CI matrix); L-BFGS-B reaches ~1e-15.
    assert abs(result.energy - E0) / abs(E0) < 1e-5
    assert result.support_peak > 0
    assert result.evaluations > 0


def test_hva_program_structure():
    m = tfim(4)
    prog = hva_program(m, depth=2)
    assert prog.parameters.names == ("zz_1", "x_1", "zz_2", "x_2")
    # reference prep (4 H gates) + 2 layers * (3 bonds + 4 fields)
    assert len(prog.ops) == 4 + 2 * 7


# ---------------------------------------------------------------------------
# Exact ADAPT (backlog item 6: six-operator convergence)


def test_exact_adapt_tfim_n4_six_operator_convergence():
    m = tfim(4, 1.0, 1.0)
    res = run_adapt(m, local_pool(4, periodic_context=False), max_operators=10)
    assert res.relative_error < 1e-9
    assert len(res.labels) == 6
    assert res.stopped_reason == "exact gradient below threshold"
    energies = [r.energy for r in res.records if r.energy is not None]
    assert all(b <= a + 1e-9 for a, b in zip(energies, energies[1:]))
    assert res.total_shots == 0


# ---------------------------------------------------------------------------
# Finite-shot ADAPT with confidence-certified selection


def test_noisy_adapt_converges_and_accounts_resources():
    m = tfim(4, 1.0, 1.0)
    res = run_adapt(m, local_pool(4, periodic_context=False),
                    backend=FiniteShotBackend(seed=11),
                    selector=ConfidenceSelector(delta=0.05, near_tol=0.05),
                    allocator=UniformDoubling(base=256, max_factor=64),
                    max_operators=8)
    assert res.relative_error < 1e-3
    assert res.total_shots > 0
    assert res.total_circuits > 0
    assert all(r.cumulative_shots >= r.shots_added for r in res.records)
    assert all(isinstance(r.status, SelectionStatus) for r in res.records)


def test_noisy_adapt_is_seed_reproducible():
    m = tfim(3, 1.0, 1.0)
    kwargs = dict(selector=ConfidenceSelector(delta=0.05, near_tol=0.05),
                  allocator=UniformDoubling(base=128, max_factor=16),
                  max_operators=4)
    a = run_adapt(m, local_pool(3, periodic_context=False),
                  backend=FiniteShotBackend(seed=5), **kwargs)
    b = run_adapt(m, local_pool(3, periodic_context=False),
                  backend=FiniteShotBackend(seed=5), **kwargs)
    assert a.labels == b.labels
    assert a.total_shots == b.total_shots


def test_selector_calibration_wrong_selection_rate_below_delta():
    """H3 calibration: with distinct gradients, the resolved selection is the
    exact argmax at least 1-delta of the time.

    At the symmetric product reference state mirror-pair candidates tie
    exactly, so the state is displaced by generic rotor angles to obtain a
    unique argmax (ties make RESOLVED_BEST unreachable by design).
    """
    m = random_ising(3, seed=1)  # distinct couplings -> distinct gradients
    pool = local_pool(3, periodic_context=False)
    bank = CommutatorBank(m.hamiltonian, [op.word for op in pool],
                          [op.label for op in pool])
    prog = _ansatz_program(m, [pool[1], pool[4]])
    rho = FiniteShotBackend(seed=99).state(prog, (0.37, -0.51))
    scores = [abs(bank.exact_score(j, rho)) for j in range(len(bank))]
    ranked = sorted(scores, reverse=True)
    assert ranked[0] - ranked[1] > 0.01  # unique argmax with a real gap
    best = max(range(len(bank)), key=lambda j: scores[j])

    delta = 0.1
    selector = ConfidenceSelector(delta=delta, threshold=1e-4)
    wrong = resolved = 0
    for seed in range(60):
        backend = FiniteShotBackend(seed=seed)
        cache = WordCache(3)
        sampler = lambda words, plan: backend.sample_words_from_state(rho, words, plan)
        idx, status, _ = selector.select(bank, cache, sampler,
                                         UniformDoubling(base=256, max_factor=1024),
                                         list(range(len(bank))))
        if status is SelectionStatus.RESOLVED_BEST:
            resolved += 1
            wrong += (idx != best)
    assert resolved > 30  # the budget resolves most runs
    assert wrong / max(resolved, 1) <= delta + 0.05


def test_selector_below_threshold_on_zero_gradient_state():
    """At the exact ADAPT fixed point every candidate is statistically zero."""
    m = tfim(3, 1.0, 1.0)
    exact = run_adapt(m, local_pool(3, periodic_context=False), max_operators=8)
    assert exact.relative_error < 1e-9
    pool = local_pool(3, periodic_context=False)
    by_label = {op.label: op for op in pool}
    prog = _ansatz_program(m, [by_label[lbl] for lbl in exact.labels])
    rho = FiniteShotBackend(seed=2).state(prog, exact.parameters)

    bank = CommutatorBank(m.hamiltonian, [op.word for op in pool])
    backend = FiniteShotBackend(seed=3)
    cache = WordCache(3)
    sampler = lambda words, plan: backend.sample_words_from_state(rho, words, plan)
    idx, status, _ = ConfidenceSelector(delta=0.05, threshold=0.1).select(
        bank, cache, sampler, UniformDoubling(base=2048, max_factor=32),
        list(range(len(pool))))
    assert status is SelectionStatus.BELOW_THRESHOLD
    assert idx is None


# ---------------------------------------------------------------------------
# Post-hoc certificate-strength diagnostics


def test_rank_and_gap_reports_argmax_and_ties():
    """rank 1 for a maximizer (ties share it); top_gap is 0 on an exact tie."""
    from clifford_qc.algorithms.adapt import _rank_and_gap

    scores = {0: 1.0, 1: -1.0, 2: 0.25}          # 0 and 1 tie at the maximum
    assert _rank_and_gap(scores, 0) == (1, 0.0)
    assert _rank_and_gap(scores, 1) == (1, 0.0)
    assert _rank_and_gap(scores, 2) == (3, 0.0)  # two strictly larger

    scores = {0: 0.9, 1: 0.4, 2: -0.1}
    rank, gap = _rank_and_gap(scores, 1)
    assert rank == 2 and gap == pytest.approx(0.5)
    assert _rank_and_gap({}, 0) == (None, None)
    assert _rank_and_gap(scores, None) == (None, pytest.approx(0.5))


def test_eta_required_is_the_tightest_valid_multiplicative_tolerance():
    """eta_req is the smallest eta with L_best >= (1-eta) * U_rival."""
    from clifford_qc.algorithms.adapt import _eta_required

    # exact-best resolution: leader's lower bound already clears every rival
    assert _eta_required(0.8, 0.5) == 0.0
    # overlapping intervals: L = 0.6, U_rival = 1.0 -> eta = 0.4
    assert _eta_required(0.6, 1.0) == pytest.approx(0.4)
    # a zero lower bound supports no nontrivial multiplicative statement
    assert _eta_required(0.0, 1.0) == 1.0
    # no rival above zero: nothing to compare against
    assert _eta_required(0.5, 0.0) == 0.0


def test_eps_best_trajectory_records_certificate_strength():
    """A strict eps-best run records rank, top gap, and eta for every step."""
    m = tfim(3, 1.0, 1.0)
    res = run_adapt(m, local_pool(3, periodic_context=False),
                    backend=FiniteShotBackend(seed=0),
                    allocator=UniformDoubling(base=256, max_factor=32),
                    selector=ConfidenceSelector(delta=0.05, bound="eb",
                                                near_tol=0.3),
                    max_operators=3, grouping=True, accept_ambiguous=False)
    appended = [r for r in res.records if r.selected_label]
    assert appended, "eps-best should resolve the reference-state tie"
    for r in appended:
        assert r.status is SelectionStatus.RESOLVED_EPS_BEST
        assert r.certification == "finite_sample" and r.certified
        assert r.exact_rank is not None and r.exact_rank >= 1
        assert r.exact_top_gap is not None and r.exact_top_gap >= 0.0
        assert r.eta_required is not None and 0.0 <= r.eta_required <= 1.0
        # the certified pick must actually satisfy the eps guarantee
        assert abs(r.exact_gradient) >= r.exact_gradient_max - 0.3
