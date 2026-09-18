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

pytest.importorskip("scipy")

from clifford_qc.backends.sector_statevector import SectorStatevectorBackend
from clifford_qc.models.lattice import hubbard
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
    assert control.metadata["sample_independent"] is True


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
    """Replace every determinant in the sample; the control must not move."""
    _, backend, operator, exact = sector
    size = np.unique(sampled).size
    disjoint = np.setdiff1d(np.arange(backend.dimension), np.unique(sampled))
    other = disjoint[:size] if disjoint.size >= size else np.arange(size)
    assert other.size == size
    first = chance(operator, sampled, seed=3, exact=exact)
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
