"""The floor of the control family: does the arm beat chance at the same size?

``run_control`` already carried the strong classical comparators --
``matched_selected_ci`` answers "what does smart classical selection reach on
this determinant budget, without ever seeing the quantum sample". What it
lacked was the weak one. A QSCI arm reports the energy of the Hamiltonian
restricted to the ``M`` configurations its state produced, and that number is
evidence about *which* configurations were found only if it beats ``M``
configurations drawn without looking at any state at all.

Without that row, "this subspace is good" and "a subspace of this size is
good" are the same number, and the arm gets credit for the second.

The tests here anchor to what is true of a uniform draw by construction rather
than to the implementation:

- the draw is without replacement, inside the operator's space, and of exactly
  the requested size;
- it consults the sampled set's *size* and nothing else, so permuting or
  replacing the sample's contents at fixed size cannot move the control;
- it does no selection work, which is what distinguishes it from every other
  control in the family;
- drawing the whole space recovers the exact energy, so the control's ceiling
  is fixed by the sector rather than chosen.

``test_a_sampled_arm_beats_chance_at_matched_size`` states the scientific
claim and is deliberately weak: the oracle's subspace must be *no worse* than
the mean of several chance draws. The margin is the quantity the control exists
to measure, so hardcoding one would defeat the purpose.
"""

from __future__ import annotations

import numpy as np
import pytest

from clifford_qc.backends.sector_statevector import SectorStatevectorBackend
from clifford_qc.models.lattice import hubbard
from clifford_qc.pauli_action import PauliLinearOperator
from clifford_qc.subspace.qsci import (exact_ground_state_oracle,
                                       sample_state_input)
from clifford_qc.subspace.selected_ci import run_control

SITES, ELECTRONS, SHOTS = 4, 4, 120


@pytest.fixture(scope="module")
def sector():
    model = hubbard(SITES, t=1.0, U=4.0, periodic=True)
    backend = SectorStatevectorBackend(model.n, n_electrons=ELECTRONS, sz=0.0)
    operator = backend.operator(model.hamiltonian)
    values, _ = backend.ground_state(model.hamiltonian)
    exact = float(np.real(np.atleast_1d(values)[0]))
    return model, backend, operator, exact


@pytest.fixture(scope="module")
def sampled(sector):
    """Configurations from the exact ground state: a genuinely good subspace."""
    model, backend, _, _ = sector
    oracle = exact_ground_state_oracle(backend, model.hamiltonian)
    indices, _ = sample_state_input(oracle, shots=SHOTS, seed=1)
    return indices


def chance(operator, sampled, *, seed, exact, budget=None):
    return run_control(operator, sampled, name="chance", kind="random",
                       seed=seed, max_determinants=budget, exact_energy=exact)


# ---------------------------------------------------------------- the draw
def test_the_draw_is_uniform_without_replacement_inside_the_space(sector, sampled):
    _, backend, operator, exact = sector
    control = chance(operator, sampled, seed=0, exact=exact)
    determinants = control.determinants
    assert determinants.size == np.unique(sampled).size
    assert determinants.size == np.unique(determinants).size
    assert 0 <= determinants.min() and determinants.max() < backend.dimension
    assert np.array_equal(determinants, np.sort(determinants))
    assert control.metadata["drawn_from"] == backend.dimension
    # Not ``sample_independent``: summarize_m7_replication.py reads that flag
    # as "does not move with the draw", and this arm moves with its own seed.
    assert "sample_independent" not in control.metadata
    assert control.metadata["sample_contents_independent"] is True
    assert control.metadata["draw_dependent"] is True


def test_the_control_does_no_selection_work(sector, sampled):
    """What separates the null from every other control in the family.

    ``selection_work`` counts candidates *scored*. A uniform draw scores none,
    so a zero here is the honest report of how it found its determinants.
    """
    _, _, operator, exact = sector
    assert chance(operator, sampled, seed=0, exact=exact).selection_work == 0
    selected = run_control(operator, sampled, name="sel", kind="budget_matched",
                           max_determinants=np.unique(sampled).size,
                           exact_energy=exact)
    assert selected.selection_work > 0


def test_only_the_size_of_the_sample_reaches_the_control(sector, sampled):
    """Replace every determinant in the sample; the control must not move.

    The replacement has to be *disjoint* for this to test anything.  Sized off
    the full sample it is not: 21 unique determinants in a 36-dimensional
    sector leave only 15 to replace them with, and a fallback that reuses part
    of the sample would let an implementation consulting the shared
    determinants pass.  A half-size slice leaves room for a genuine disjoint
    set, and the assertion below states that rather than assuming it.
    """
    _, backend, operator, exact = sector
    half = np.unique(sampled)[:np.unique(sampled).size // 2]
    disjoint = np.setdiff1d(np.arange(backend.dimension), np.unique(sampled))
    other = disjoint[:half.size]
    assert other.size == half.size
    assert np.intersect1d(other, half).size == 0
    first = chance(operator, half, seed=3, exact=exact)
    second = chance(operator, other, seed=3, exact=exact)
    assert np.array_equal(first.determinants, second.determinants)
    assert first.energy == pytest.approx(second.energy, abs=1e-12)


def test_the_budget_overrides_the_sample_size(sector, sampled):
    _, _, operator, exact = sector
    control = chance(operator, sampled, seed=0, exact=exact, budget=7)
    assert control.determinants.size == 7


def test_the_draw_is_reproducible_and_seed_dependent(sector, sampled):
    _, _, operator, exact = sector
    first = chance(operator, sampled, seed=5, exact=exact)
    again = chance(operator, sampled, seed=5, exact=exact)
    other = chance(operator, sampled, seed=6, exact=exact)
    assert np.array_equal(first.determinants, again.determinants)
    assert not np.array_equal(first.determinants, other.determinants)


def test_an_unseeded_draw_is_refused(sector, sampled):
    _, _, operator, exact = sector
    with pytest.raises(ValueError, match="needs an explicit seed"):
        run_control(operator, sampled, name="chance", kind="random",
                    exact_energy=exact)


def test_a_budget_larger_than_the_space_is_refused(sector, sampled):
    _, backend, operator, exact = sector
    with pytest.raises(ValueError, match="cannot draw"):
        chance(operator, sampled, seed=0, exact=exact,
               budget=backend.dimension + 1)


def test_an_unknown_kind_still_names_the_vocabulary(sector, sampled):
    _, _, operator, exact = sector
    with pytest.raises(ValueError, match="random"):
        run_control(operator, sampled, name="x", kind="uniform",
                    exact_energy=exact)


# -------------------------------------------------------------- the physics
def test_the_control_respects_the_variational_bound(sector, sampled):
    _, _, operator, exact = sector
    assert chance(operator, sampled, seed=0, exact=exact).energy >= exact - 1e-9


def test_drawing_the_whole_space_recovers_the_exact_energy(sector, sampled):
    """The control's ceiling is the sector's, not a tuned parameter."""
    _, backend, operator, exact = sector
    control = chance(operator, sampled, seed=0, exact=exact,
                     budget=backend.dimension)
    assert control.determinants.size == backend.dimension
    assert control.energy == pytest.approx(exact, abs=1e-9)


def test_a_sampled_arm_beats_chance_at_matched_size(sector, sampled):
    """The claim this control exists to make checkable."""
    _, _, operator, exact = sector
    arm = run_control(operator, sampled, name="qsci", kind="qsci",
                      exact_energy=exact)
    draws = [chance(operator, sampled, seed=seed, exact=exact,
                    budget=arm.determinants.size).energy for seed in range(5)]
    assert all(energy >= exact - 1e-9 for energy in draws)
    # weak on purpose: the margin is what the control measures
    assert arm.energy <= float(np.mean(draws)) + 1e-9


def test_an_arbitrary_index_block_does_not_beat_chance(sector):
    """The null is a real null, not one the arms win by construction.

    A contiguous block of sector positions is not a sampled state, and it does
    not reliably beat a uniform draw of the same size. A control that every
    input beats would be measuring nothing.
    """
    _, _, operator, exact = sector
    block = np.arange(12, dtype=np.int64)
    arm = run_control(operator, block, name="block", kind="qsci",
                      exact_energy=exact)
    draws = [chance(operator, block, seed=seed, exact=exact).energy
             for seed in range(5)]
    assert min(draws) < arm.energy


def test_the_control_record_carries_its_overlap_with_the_sample(sector, sampled):
    """Reported, not asserted: a large overlap explains a small advantage."""
    _, _, operator, exact = sector
    control = chance(operator, sampled, seed=0, exact=exact)
    overlap = control.metadata["overlap_with_sample"]
    expected = np.intersect1d(control.determinants, np.unique(sampled)).size
    assert overlap == expected
    assert 0 <= overlap <= control.determinants.size


# ------------------------------------------------------------ the sector
def test_a_full_space_draw_stays_in_the_samples_sector(sector, sampled):
    """The pool is the arm's sector, not the whole Fock space.

    Over a ``PauliLinearOperator`` the operator's space is all ``2**n`` words,
    and a draw from that lands most of its determinants in particle-number
    sectors the Hamiltonian never couples to the arm's subspace.  The "chance
    floor" would then be the energy of a multi-sector junk subspace, and an
    arm's margin over it would mostly report particle-number conservation.
    """
    model, backend, _, exact = sector
    full = PauliLinearOperator(model.hamiltonian)
    words = backend.basis[np.unique(sampled)]
    control = run_control(full, words, name="chance", kind="random", seed=0,
                          n=model.n, exact_energy=exact)
    assert control.determinants.size == words.size
    assert np.all(np.isin(control.determinants, backend.basis))
    assert control.metadata["drawn_from"] == backend.dimension


def test_a_full_space_draw_without_the_qubit_count_is_refused(sector, sampled):
    model, backend, _, exact = sector
    full = PauliLinearOperator(model.hamiltonian)
    words = backend.basis[np.unique(sampled)]
    with pytest.raises(ValueError, match="needs the qubit count"):
        run_control(full, words, name="chance", kind="random", seed=0,
                    exact_energy=exact)


def test_a_sample_spanning_sectors_has_no_matched_pool(sector):
    """No single sector to draw from, so no matched control -- say so."""
    model, backend, _, exact = sector
    full = PauliLinearOperator(model.hamiltonian)
    mixed = np.array([int(backend.basis[0]), 0], dtype=np.int64)
    with pytest.raises(ValueError, match="particle-number sectors"):
        run_control(full, mixed, name="chance", kind="random", seed=0,
                    n=model.n, exact_energy=exact)


# ------------------------------------------------------------- the budget
def test_the_nonzero_budget_is_honoured_and_recorded(sector, sampled):
    """The one arm whose whole purpose is to be budget-matched.

    ``selected_ci`` and ``budget_matched`` trim to a measured matrix-nonzero
    budget and record it; a null that silently kept nine times the requested
    nonzeros would be unmatched exactly where matching is the point.
    """
    _, _, operator, exact = sector
    control = run_control(operator, sampled, name="chance", kind="random",
                          seed=0, max_nonzeros=10, exact_energy=exact)
    assert control.matrix_nonzeros <= 10
    assert control.metadata["nonzero_budget"] == 10
    assert control.determinants.size < np.unique(sampled).size


def test_the_nonzero_budget_trims_the_draw_not_the_low_indices(sector, sampled):
    """Trimming has to shrink the draw, not bias it toward the space's floor.

    A prefix of the *sorted* draw would preferentially keep small indices, so
    the trimmed control would no longer be uniform over the sector.
    """
    _, backend, operator, exact = sector
    trimmed = run_control(operator, sampled, name="chance", kind="random",
                          seed=0, max_nonzeros=40, exact_energy=exact)
    full = run_control(operator, sampled, name="chance", kind="random",
                       seed=0, exact_energy=exact)
    assert np.all(np.isin(trimmed.determinants, full.determinants))
    assert trimmed.determinants.size < full.determinants.size
    # What a sorted-prefix trim would have produced, and must not have.
    biased = np.sort(full.determinants)[:trimmed.determinants.size]
    assert not np.array_equal(trimmed.determinants, biased)


# -------------------------------------------------------------- the record
def test_a_seed_reaching_another_kind_is_refused(sector, sampled):
    """A seed the kind cannot draw with is a mislabelled row, not a no-op."""
    _, _, operator, exact = sector
    with pytest.raises(ValueError, match="only used by the random control"):
        run_control(operator, sampled, name="q", kind="qsci", seed=123,
                    exact_energy=exact)


def test_a_seed_object_numpy_accepts_does_not_break_the_record(sector, sampled):
    """``default_rng`` takes more than an int; the record must survive them.

    Coercing the seed with ``int()`` raised *after* the draw had succeeded --
    a crash in the record-keeping rather than in the control.
    """
    _, _, operator, exact = sector
    control = run_control(operator, sampled, name="chance", kind="random",
                          seed=np.random.SeedSequence(5), exact_energy=exact)
    assert control.determinants.size == np.unique(sampled).size
    assert "SeedSequence" in control.metadata["seed"]


def test_the_record_pins_the_draw_that_produced_it(sector, sampled):
    """The seed alone does not pin it across NumPy versions.

    ``ControlResult.to_record`` drops ``determinants`` by design, and
    ``Generator`` carries no cross-version bit-stream guarantee, so a reader
    rebuilding this record on a newer NumPy could compare a different subspace
    without anything saying so.  The digest is what says so.
    """
    import json

    _, _, operator, exact = sector
    first = chance(operator, sampled, seed=0, exact=exact)
    again = chance(operator, sampled, seed=0, exact=exact)
    other = chance(operator, sampled, seed=1, exact=exact)
    assert first.metadata["draw_sha256"] == again.metadata["draw_sha256"]
    assert first.metadata["draw_sha256"] != other.metadata["draw_sha256"]
    json.dumps(first.to_record())


@pytest.mark.parametrize("qubits,electrons,sz",
                         [(8, 4, 0.0), (6, 3, 0.5), (6, 2, 0.0),
                          (8, 2, 1.0), (6, 4, 0.0)])
def test_the_pool_is_exactly_the_backends_sector(qubits, electrons, sz):
    """The full-space pool and the sector backend must agree word for word.

    Two independent derivations of the same set: the backend enumerates the
    sector from its own symmetry rule, and the control filters the Fock space
    by particle number and ``S_z``.  Checking them against each other is what
    makes "drawn uniformly from the arm's sector" a statement about the arm's
    sector rather than about this function's idea of one.
    """
    from clifford_qc.subspace.selected_ci import _sector_pool

    backend = SectorStatevectorBackend(qubits, n_electrons=electrons, sz=sz)
    pool = _sector_pool(backend.basis[:3], n=qubits, conserve_sz=True)
    assert np.array_equal(np.sort(pool), np.sort(backend.basis))
