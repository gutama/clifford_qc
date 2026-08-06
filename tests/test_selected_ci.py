"""Phase 9 of ``LITERATURE_ROADMAP.md``: the classical selected-CI controls.

The controls exist so a hybrid gain cannot be confused with ordinary
determinant-space expansion, which means the controls themselves have to be
trustworthy in a specific direction: they must never be accidentally *weaker*
than the thing they police. A closure that misses reachable determinants, or a
score that silently retains nothing, would make the hybrid look better than it
is -- so the tests below check the controls are strong, not merely that they
run.
"""

from pathlib import Path

import numpy as np
import pytest

from clifford_qc.backends.sector_statevector import SectorStatevectorBackend
from clifford_qc.models import fcidump_model
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace import determinant_excitations, occupied_spin_orbitals
from clifford_qc.subspace.selected_ci import (
    excitation_closure, principal_angles, run_control, score_candidates,
    span_comparison,
)

H4_FCIDUMP = (Path(__file__).parents[1] / "benchmarks" / "data"
              / "h4_sto3g_r0.9.FCIDUMP")


def _case(model):
    backend = SectorStatevectorBackend(model.n,
                                       int(model.metadata["n_electrons"]),
                                       float(model.metadata["sz"]))
    operator = backend.operator(model.hamiltonian)
    exact = float(np.linalg.eigvalsh(
        operator.restrict(np.arange(backend.dimension)))[0])
    return backend, operator, exact, model


@pytest.fixture(scope="module")
def hubbard_case():
    return _case(hubbard(shape=(2, 2), t=1.0, U=4.0))


@pytest.fixture(scope="module")
def h4_case():
    return _case(fcidump_model(H4_FCIDUMP, name="H4 CAS(4e,4o)"))


@pytest.fixture(params=["hubbard", "h4"])
def sector_case(request, hubbard_case, h4_case):
    return hubbard_case if request.param == "hubbard" else h4_case


def _sample(backend, size, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(backend.dimension, size=size, replace=False))


# ------------------------------------------------------------ excitation closure

def test_closure_contains_its_input(sector_case):
    backend, _, _, model = sector_case
    sampled = _sample(backend, 4)
    closed = excitation_closure(backend.basis[sampled], n=model.n)
    assert set(backend.basis[sampled].tolist()) <= set(closed.tolist())


def test_closure_stays_in_the_sector(sector_case):
    """A symmetry-preserving closure cannot leave the sector it started in."""
    backend, _, _, model = sector_case
    sampled = _sample(backend, 5)
    closed = excitation_closure(backend.basis[sampled], n=model.n)
    assert set(closed.tolist()) <= set(backend.basis.tolist())


def test_closure_is_monotone_in_its_input(sector_case):
    """More sampled determinants can only reach more, never fewer."""
    backend, _, _, model = sector_case
    small = _sample(backend, 3, seed=1)
    large = np.unique(np.concatenate([small, _sample(backend, 3, seed=2)]))
    closed_small = excitation_closure(backend.basis[small], n=model.n)
    closed_large = excitation_closure(backend.basis[large], n=model.n)
    assert set(closed_small.tolist()) <= set(closed_large.tolist())


def test_singles_only_closure_is_contained_in_singles_and_doubles(sector_case):
    backend, _, _, model = sector_case
    sampled = _sample(backend, 4)
    singles = excitation_closure(backend.basis[sampled], n=model.n, max_rank=1)
    both = excitation_closure(backend.basis[sampled], n=model.n, max_rank=2)
    assert set(singles.tolist()) <= set(both.tolist())


def test_closure_matches_the_generator_family_it_controls(sector_case):
    """The control has to close over the *same* excitations the dressing uses.

    Built independently here: apply each `determinant_excitations` generator to
    each sampled determinant and collect whichever sector words acquire weight.
    If the hand-written bitmask rule and the generator algebra disagree, the
    control is policing a different family than the one under test.
    """
    backend, _, _, model = sector_case
    sampled = _sample(backend, 3, seed=4)
    reached = set(backend.basis[sampled].tolist())
    for generator in determinant_excitations(model.n,
                                             occupied_spin_orbitals(model)):
        compiled = backend.operator(generator.mv, validate_sector=False)
        for index in sampled:
            probe = np.zeros(backend.dimension, dtype=complex)
            probe[index] = 1.0
            hit = np.flatnonzero(np.abs(compiled.matvec(probe)) > 1e-12)
            reached.update(backend.basis[hit].tolist())
    closed = set(excitation_closure(backend.basis[sampled],
                                    n=model.n).tolist())
    assert reached <= closed


def test_closure_rejects_bad_input(sector_case):
    _, _, _, model = sector_case
    with pytest.raises(ValueError, match="empty"):
        excitation_closure([], n=model.n)
    with pytest.raises(ValueError, match="max_rank"):
        excitation_closure([0b11110000], n=model.n, max_rank=3)


# ------------------------------------------------------------------- controls

def test_controls_are_variational_and_ordered(sector_case):
    """Each control contains the previous one, so energies only fall."""
    backend, operator, exact, _ = sector_case
    sampled = _sample(backend, 6, seed=3)
    bare = run_control(operator, sampled, name="qsci", kind="qsci",
                       exact_energy=exact)
    selected = run_control(operator, sampled, name="sci", kind="selected_ci",
                           exact_energy=exact,
                           max_determinants=sampled.size + 5)
    closure = run_control(operator, sampled, name="closure",
                          kind="excitation_closure", exact_energy=exact)
    for control in (bare, selected, closure):
        assert control.error >= -1e-9
    assert selected.energy <= bare.energy + 1e-10
    assert set(bare.determinants.tolist()) <= set(closure.determinants.tolist())
    assert bare.determinant_count <= selected.determinant_count


def test_variance_falls_as_the_space_grows(sector_case):
    """Variance is the convergence signal §9 asks for beside the energy."""
    backend, operator, exact, _ = sector_case
    sampled = _sample(backend, 5, seed=7)
    bare = run_control(operator, sampled, name="qsci", kind="qsci",
                       exact_energy=exact)
    closure = run_control(operator, sampled, name="closure",
                          kind="excitation_closure", exact_energy=exact)
    assert bare.variance >= -1e-9
    assert closure.variance <= bare.variance + 1e-9


def test_full_space_control_is_exact(sector_case):
    """Selecting everything must reproduce the sector, variance included."""
    backend, operator, exact, _ = sector_case
    whole = np.arange(backend.dimension)
    control = run_control(operator, whole, name="all", kind="qsci",
                          exact_energy=exact)
    assert control.energy == pytest.approx(exact, abs=1e-10)
    assert control.variance == pytest.approx(0.0, abs=1e-8)


def test_budget_matched_respects_its_budget(sector_case):
    backend, operator, exact, _ = sector_case
    sampled = _sample(backend, 5, seed=8)
    budget = sampled.size + 4
    control = run_control(operator, sampled, name="budget",
                          kind="budget_matched", exact_energy=exact,
                          max_determinants=budget)
    assert control.determinant_count <= budget


def test_budget_matched_requires_a_declared_budget(sector_case):
    """Without a budget it is the unbudgeted control wearing another name."""
    backend, operator, exact, _ = sector_case
    with pytest.raises(ValueError, match="declared budget"):
        run_control(operator, _sample(backend, 4), name="budget",
                    kind="budget_matched", exact_energy=exact)


def test_nonzero_budget_is_converted_and_recorded(sector_case):
    backend, operator, exact, _ = sector_case
    sampled = _sample(backend, 4, seed=9)
    control = run_control(operator, sampled, name="budget",
                          kind="budget_matched", exact_energy=exact,
                          max_nonzeros=100)
    assert control.metadata["budget_from_nonzeros"] == 100
    assert control.determinant_count <= 10  # floor(sqrt(100))


def test_selection_work_counts_candidates_not_keeps(sector_case):
    """A control that scored the whole sector has not found a cheap route."""
    backend, operator, exact, _ = sector_case
    sampled = _sample(backend, 4, seed=10)
    control = run_control(operator, sampled, name="sci", kind="selected_ci",
                          exact_energy=exact, max_determinants=sampled.size + 3)
    assert control.selection_work == backend.dimension - sampled.size
    assert control.determinant_count < control.selection_work


@pytest.mark.parametrize("score", ["epstein_nesbet", "first_order", "coupling"])
def test_every_declared_score_runs_and_stays_variational(sector_case, score):
    backend, operator, exact, _ = sector_case
    sampled = _sample(backend, 5, seed=11)
    control = run_control(operator, sampled, name="sci", kind="selected_ci",
                          exact_energy=exact, score=score,
                          max_determinants=sampled.size + 4)
    assert control.error >= -1e-9
    assert control.metadata["score"] == score


def test_unknown_score_and_kind_are_refused(sector_case):
    backend, operator, _, _ = sector_case
    sampled = _sample(backend, 3)
    with pytest.raises(ValueError, match="score must be"):
        run_control(operator, sampled, name="x", kind="selected_ci",
                    score="handwave")
    with pytest.raises(ValueError, match="kind must be"):
        run_control(operator, sampled, name="x", kind="vibes")


def test_scoring_ranks_a_strongly_coupled_determinant_first(sector_case):
    """The score must order candidates, not merely return them."""
    backend, operator, _, _ = sector_case
    sampled = _sample(backend, 4, seed=12)
    matrix = operator.restrict(sampled)
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    candidates, scores, work = score_candidates(
        operator, sampled, vectors[:, 0], energy=float(values[0]))
    assert work == backend.dimension - sampled.size
    assert np.all(np.diff(scores) <= 1e-12)  # descending
    assert not set(candidates.tolist()) & set(sampled.tolist())


# --------------------------------------------------------- span diagnostic

def test_principal_angles_of_identical_spans_are_zero():
    basis = np.linalg.qr(np.random.default_rng(0).normal(size=(9, 4)))[0]
    angles = principal_angles(basis, basis)
    assert np.allclose(angles, 0.0, atol=1e-10)
    assert not np.isnan(angles).any()


def test_principal_angles_of_orthogonal_spans_are_right_angles():
    identity = np.eye(6, dtype=complex)
    angles = principal_angles(identity[:, :2], identity[:, 3:5])
    assert np.allclose(angles, np.pi / 2, atol=1e-10)


def test_dressed_states_live_inside_the_determinant_closure(sector_case):
    """The §9 question, on the family the hybrid actually proposes to use.

    If `A_mu|D_k>` never leaves the closure of the same `D_k`, the operator
    form is a representation choice; the roadmap then forbids claiming it is a
    richer variational space.
    """
    backend, _, _, model = sector_case
    sampled = _sample(backend, 4, seed=13)
    columns = []
    for generator in determinant_excitations(model.n,
                                             occupied_spin_orbitals(model))[:8]:
        compiled = backend.operator(generator.mv, validate_sector=False)
        for index in sampled:
            probe = np.zeros(backend.dimension, dtype=complex)
            probe[index] = 1.0
            dressed = compiled.matvec(probe)
            if np.linalg.norm(dressed) > 1e-12:
                columns.append(dressed)
    operator_basis = np.column_stack(columns)

    closed = excitation_closure(backend.basis[sampled], n=model.n)
    positions = np.searchsorted(backend.basis, closed)
    determinant_basis = np.zeros((backend.dimension, positions.size),
                                 dtype=complex)
    determinant_basis[positions, np.arange(positions.size)] = 1.0

    comparison = span_comparison(operator_basis, determinant_basis)
    assert comparison.contained
    assert comparison.max_angle < 1e-9
    assert comparison.operator_rank <= comparison.determinant_rank
    assert "outside closure" not in comparison.verdict


def test_span_comparison_detects_a_direction_outside_the_closure():
    """The diagnostic has to be able to fail, or containment means nothing."""
    identity = np.eye(7, dtype=complex)
    comparison = span_comparison(identity[:, 4:6], identity[:, :3])
    assert not comparison.contained
    assert comparison.max_angle == pytest.approx(np.pi / 2, abs=1e-10)
    assert "outside closure" in comparison.verdict


def test_equal_spans_report_representation_only():
    rng = np.random.default_rng(3)
    basis = np.linalg.qr(rng.normal(size=(8, 3)))[0].astype(complex)
    mixed = basis @ rng.normal(size=(3, 3))
    comparison = span_comparison(mixed, basis)
    assert comparison.contained
    assert comparison.operator_rank == comparison.determinant_rank == 3
    assert "representation only" in comparison.verdict


def test_proper_subspace_does_not_claim_compactness():
    """A smaller span is a weaker space, and must not be sold as efficiency."""
    identity = np.eye(6, dtype=complex)
    comparison = span_comparison(identity[:, :2], identity[:, :5])
    assert comparison.contained
    assert comparison.operator_rank < comparison.determinant_rank
    assert comparison.verdict.startswith("proper subspace")
    # The word "compactness" may appear, but only to say a compactness claim
    # has to be earned elsewhere -- never as this diagnostic's own verdict.
    assert "must come from energy at matched size" in comparison.verdict


def test_span_comparison_rejects_mismatched_ambient_spaces():
    with pytest.raises(ValueError, match="ambient"):
        span_comparison(np.eye(5, dtype=complex), np.eye(6, dtype=complex))
    with pytest.raises(ValueError, match="at least one column"):
        span_comparison(np.zeros((5, 0), dtype=complex), np.eye(5, dtype=complex))


def test_control_record_is_json_serializable(sector_case):
    import json

    backend, operator, exact, _ = sector_case
    control = run_control(operator, _sample(backend, 4), name="closure",
                          kind="excitation_closure", exact_energy=exact)
    record = json.loads(json.dumps(control.to_record()))
    assert record["control"] == "closure"
    assert record["determinant_count"] == control.determinant_count
    assert "variance" in record and "selection_work" in record
