"""The cached group-sampling plan must change cost, not results.

``FiniteShotBackend.sample_grouped_from_state`` rotates the state onto each
group's shared basis, traces out the untouched qubits, and reads off the
computational distribution. None of that depends on the shot count, and the
certified A-CASE loop measures the same never-changing reference twice per
growth step -- so it was redone every batch, which is what confined certified
growth to four qubits.

Caching it is only legitimate if the sampler's output is bit-identical: the
committed records were produced under a particular RNG stream, and the stream is
part of the record. These tests compare the cached sampler against the
uncached reference loop it replaced, verbatim, and check that the plan table is
keyed by the state rather than reused across states.
"""

import numpy as np
import pytest

from clifford_qc import gates as _gates
from clifford_qc.backends import ExactMVBackend, FiniteShotBackend
from clifford_qc.ir import PauliWord
from clifford_qc.measurement.grouping import qwc_groups, shared_basis
from clifford_qc.models.spin import tfim
from clifford_qc.states import computational_probabilities, evolve, partial_trace


def _reference_batch(rng, rho, groups, shots):
    """The pre-cache loop, verbatim, so equality is a real regression check."""
    out_shots, plus, hists = {}, {}, []
    circuits = 0
    for group in groups:
        N = shots
        if N == 0:
            continue
        circuits += 1
        basis = shared_basis(group)
        rho_rot = rho
        for j, letter in basis.items():
            if letter == "X":
                rho_rot = evolve(rho_rot, _gates.H(rho.n, j))
            elif letter == "Y":
                rho_rot = evolve(rho_rot,
                                 _gates.H(rho.n, j) * _gates.S(rho.n, j).dagger())
        keep = tuple(sorted({j for w in group for j in w.support()}))
        traced = rho_rot if len(keep) == rho.n else \
            partial_trace(rho_rot, {j for j in range(rho.n) if j not in keep})
        outcomes = sorted(computational_probabilities(traced).items())
        probs = np.clip([p for _, p in outcomes], 0.0, None)
        probs = probs / probs.sum()
        counts = rng.multinomial(N, probs)
        position = {q: i for i, q in enumerate(keep)}
        hist = {}
        for (bits, _), c in zip(outcomes, counts):
            if c:
                key = "".join(bits[position[q]] for q in keep)
                hist[key] = hist.get(key, 0) + int(c)
        hists.append((keep, tuple(sorted(basis.items())), hist, N))
        for w in group:
            positions = [position[j] for j in w.support()]
            n_plus = sum(int(c) for (bits, _), c in zip(outcomes, counts)
                         if sum(bits[pos] == "1" for pos in positions) % 2 == 0)
            out_shots[w.code] = out_shots.get(w.code, 0) + N
            plus[w.code] = plus.get(w.code, 0) + n_plus
    return out_shots, plus, circuits, hists


def _setup(n=4):
    model = tfim(n, 1.0, 0.7)
    rho = ExactMVBackend().state(model.reference, ())
    words = [PauliWord.from_label(label) for label in
             ("XIII", "IXII", "ZZII", "IZZI", "XYII", "YXII", "ZIZI", "IYYI")]
    return rho, qwc_groups(words)


@pytest.mark.parametrize("shots", [1, 37, 500])
def test_cached_sampler_matches_the_uncached_reference(shots):
    rho, groups = _setup()
    backend = FiniteShotBackend(seed=11)
    reference_rng = np.random.default_rng(11)
    # Several batches in a row: the first fills the plan table, the rest hit it,
    # and every one has to stay on the same RNG stream as the reference.
    for _ in range(4):
        batch = backend.sample_grouped_from_state(rho, groups, shots)
        want_shots, want_plus, want_circuits, want_hists = _reference_batch(
            reference_rng, rho, groups, shots)
        assert batch.shots == want_shots
        assert batch.plus_counts == want_plus
        assert batch.circuits == want_circuits
        assert len(batch.groups) == len(want_hists)
        for sample, (keep, basis, hist, n) in zip(batch.groups, want_hists):
            assert sample.support == keep
            assert sample.basis == basis
            assert sample.hist == hist
            assert sample.shots == n


def test_repeated_batches_are_not_identical_to_each_other():
    """The plan is reused; the draw is not. A cache that also froze the outcomes
    would make every batch the same sample and silently destroy the statistics
    the certified bounds are computed from."""
    rho, groups = _setup()
    backend = FiniteShotBackend(seed=3)
    first = backend.sample_grouped_from_state(rho, groups, 200)
    second = backend.sample_grouped_from_state(rho, groups, 200)
    assert first.plus_counts != second.plus_counts
    assert first.shots == second.shots  # same allocation, different outcomes


def test_plans_are_keyed_by_state_not_carried_across_states():
    rho, groups = _setup()
    # The TFIM reference is |++++>, which X_0 leaves alone -- rotate instead, so
    # the two states really do differ.
    turned = evolve(rho, _gates.RY(4, 0, 0.6))
    backend = FiniteShotBackend(seed=5)
    table = backend._group_plans(rho)
    turned_table = backend._group_plans(turned)
    assert table is not turned_table

    group = next(g for g in groups if any(w.label == "ZZII" for w in g))
    plan = backend._plan(rho, group, table)
    turned_plan = backend._plan(turned, group, turned_table)
    # |+> is uniform in Z; RY(0.6)|+> is not, so the Z-basis group differs
    assert not np.allclose(plan.probs, turned_plan.probs)

    # an equal state rebuilt from the same program must hit the same table
    rebuilt = ExactMVBackend().state(tfim(4, 1.0, 0.7).reference, ())
    assert backend._group_plans(rebuilt) is table


def test_plan_table_is_bounded():
    """ADAPT moves the state every step, so the table must not grow without
    bound over a trajectory."""
    rho, groups = _setup()
    backend = FiniteShotBackend(seed=9)
    for qubit in range(4):
        state = evolve(rho, _gates.RX(4, qubit, 0.3 * (qubit + 1)))
        backend.sample_grouped_from_state(state, groups, 5)
    assert len(backend._plans) <= FiniteShotBackend._PLAN_STATES


def test_parity_masks_reproduce_single_word_expectations():
    """The cached parity mask per word is what turns group counts into per-word
    plus counts; at high shot count it must reproduce the exact expectation."""
    from clifford_qc.states import expectation

    rho, groups = _setup()
    batch = FiniteShotBackend(seed=2).sample_grouped_from_state(rho, groups, 200_000)
    for group in groups:
        for w in group:
            estimate = 2.0 * batch.plus_counts[w.code] / batch.shots[w.code] - 1.0
            assert estimate == pytest.approx(expectation(rho, w.to_mv()).real, abs=0.02)
