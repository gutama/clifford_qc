"""Phase 8 of ``LITERATURE_ROADMAP.md``: the sampled-subspace baseline.

The go/no-go for the QSCI arm is stated as five invariants on the restricted
Hamiltonian plus a demand that they hold on H4, Hubbard, and at least one spin
model.  This file is that gate, so the invariants are tested directly rather
than inferred from an energy agreeing with a reference:

1. the sampled Hamiltonian is Hermitian;
2. nested sampled sets give non-increasing Ritz energies;
3. the k-th sampled Ritz value is not below the k-th exact sector eigenvalue
   (Cauchy interlacing), which is the statement that actually constrains the
   excited roots -- ``E_0 <= E_QSCI`` alone would not;
4. selecting the entire sector reproduces the sector spectrum;
5. permuting the sampled order moves no eigenvalue.

The three models are parametrized over one body of assertions so a future
model cannot quietly acquire a weaker gate than the others.
"""

from pathlib import Path

import numpy as np
import pytest

from clifford_qc.backends.sector_statevector import SectorStatevectorBackend
from clifford_qc.models import fcidump_model
from clifford_qc.models.lattice import hubbard, kitaev_honeycomb
from clifford_qc.pauli_action import PauliLinearOperator
from clifford_qc.subspace.qsci import (
    recover_configurations, run_qsci, sample_configurations,
)

# The committed CAS(4e,4o) active space, so the H4 leg of the gate needs no
# PySCF or OpenFermion install to run in CI.
H4_FCIDUMP = (Path(__file__).parents[1] / "benchmarks" / "data"
              / "h4_sto3g_r0.9.FCIDUMP")


def _sector_case(model):
    """A (backend, compiled operator, exact spectrum) triple for a fermionic model."""
    metadata = model.metadata
    n_electrons = int(metadata["n_electrons"])
    sz = metadata.get("sz")
    backend = SectorStatevectorBackend(model.n, n_electrons,
                                       None if sz is None else float(sz))
    operator = backend.operator(model.hamiltonian)
    exact = np.linalg.eigvalsh(operator.restrict(np.arange(backend.dimension)))
    return backend, operator, exact, model


@pytest.fixture(scope="module")
def h4_case():
    return _sector_case(fcidump_model(H4_FCIDUMP, name="H4 CAS(4e,4o)"))


@pytest.fixture(scope="module")
def hubbard_case():
    return _sector_case(hubbard(shape=(2, 2), t=1.0, U=4.0))


@pytest.fixture(scope="module")
def kitaev_case():
    """The spin arm: no particle-number sector, so the restriction is full-space."""
    model = kitaev_honeycomb(rows=2, cols=2)
    operator = PauliLinearOperator(model.hamiltonian)
    exact = np.linalg.eigvalsh(operator.restrict(np.arange(operator.dimension)))
    return operator, exact


# --------------------------------------------------------------- §8B invariants

@pytest.fixture(params=["h4", "hubbard", "kitaev"])
def restriction_case(request, h4_case, hubbard_case, kitaev_case):
    """(operator, exact spectrum, dimension) for each of the three gate models."""
    if request.param == "kitaev":
        operator, exact = kitaev_case
        return operator, exact, operator.dimension
    _, operator, exact, _ = h4_case if request.param == "h4" else hubbard_case
    return operator, exact, operator.dimension


def test_restriction_is_hermitian(restriction_case):
    """Invariant 1, on a subset large enough to carry off-diagonal structure."""
    operator, _, dimension = restriction_case
    rng = np.random.default_rng(11)
    indices = np.sort(rng.choice(dimension, size=min(12, dimension), replace=False))
    matrix = operator.restrict(indices)
    assert np.abs(matrix - matrix.conj().T).max() < 1e-12


def test_nested_sets_lower_the_ritz_energy(restriction_case):
    """Invariant 2: growth is monotone, because each set spans the previous one."""
    operator, _, dimension = restriction_case
    rng = np.random.default_rng(5)
    order = rng.permutation(dimension)
    energies = [float(np.linalg.eigvalsh(operator.restrict(order[:size]))[0])
                for size in range(1, min(14, dimension) + 1)]
    assert np.all(np.diff(energies) <= 1e-10)


def test_cauchy_interlacing_holds_for_every_root(restriction_case):
    """Invariant 3, on every retained root rather than only the ground state."""
    operator, exact, dimension = restriction_case
    rng = np.random.default_rng(7)
    size = min(10, dimension)
    indices = np.sort(rng.choice(dimension, size=size, replace=False))
    sampled = np.linalg.eigvalsh(operator.restrict(indices))
    assert np.all(sampled >= exact[:size] - 1e-9)


def test_full_sector_reproduces_the_spectrum(restriction_case):
    """Invariant 4: the restriction is exact, so the limit is not approximate."""
    operator, exact, dimension = restriction_case
    whole = np.linalg.eigvalsh(operator.restrict(np.arange(dimension)))
    assert np.allclose(whole, exact, atol=1e-10)


def test_h4_full_sector_reproduces_the_committed_fci_energy(h4_case):
    """An external anchor, not a self-consistency check.

    Invariant 4 says the full-sector limit reproduces *the sector spectrum*,
    which a restriction bug shared with the operator would satisfy vacuously.
    The committed CAS(4e,4o) FCI value comes from an independent determinant
    solver, so agreeing with it convicts the whole path.
    """
    _, _, exact, _ = h4_case
    assert float(exact[0]) == pytest.approx(-2.180316614323862, abs=1e-12)


def test_permutation_moves_no_eigenvalue(restriction_case):
    """Invariant 5: reordering is a similarity transform, not a new problem."""
    operator, _, dimension = restriction_case
    rng = np.random.default_rng(3)
    indices = rng.choice(dimension, size=min(11, dimension), replace=False)
    straight = np.linalg.eigvalsh(operator.restrict(np.sort(indices)))
    shuffled = np.linalg.eigvalsh(operator.restrict(rng.permutation(indices)))
    assert np.allclose(straight, shuffled, atol=1e-10)


def test_restriction_matches_the_submatrix_entrywise(restriction_case):
    """Entries, not just eigenvalues, and on a deliberately unsorted index set.

    Membership is resolved by binary search on a sorted copy, so the caller's
    ordering has to be carried back through that sort. An eigenvalue check
    cannot see a slip there -- any consistent row/column permutation leaves the
    spectrum alone -- but comparing against ``full[ix_(I, I)]`` can.
    """
    operator, _, dimension = restriction_case
    rng = np.random.default_rng(23)
    full = operator.restrict(np.arange(dimension))
    indices = rng.permutation(dimension)[:min(9, dimension)]
    assert np.allclose(operator.restrict(indices),
                       full[np.ix_(indices, indices)], atol=1e-12)


def test_restriction_rejects_malformed_index_sets(restriction_case):
    operator, _, dimension = restriction_case
    with pytest.raises(ValueError, match="empty"):
        operator.restrict([])
    with pytest.raises(ValueError, match="distinct"):
        operator.restrict([0, 0])
    with pytest.raises(ValueError, match=r"\[0, "):
        operator.restrict([dimension])


# ------------------------------------------------------------ §8A sampling contract

def test_sampling_counts_reconcile(hubbard_case):
    """Every shot is accounted for, and unique <= accepted <= raw."""
    backend, _, _, model = hubbard_case
    _, ground = backend.ground_state(model.hamiltonian)
    indices, record = sample_configurations(
        ground[:, 0], shots=500, seed=0, basis=backend.basis,
        input_label="exact_ground_oracle")
    assert record.raw_shots == 500
    assert record.accepted_shots == 500
    assert record.discarded_shots == 0
    assert 0 < record.unique_configurations <= record.accepted_shots
    assert record.unique_configurations == indices.size
    assert 0.0 <= record.duplicate_fraction < 1.0
    assert 0.0 < record.retained_probability <= 1.0 + 1e-12


def test_sampling_is_seed_reproducible(hubbard_case):
    backend, _, _, _ = hubbard_case
    state = np.full(backend.dimension, 1.0 / np.sqrt(backend.dimension))
    first, _ = sample_configurations(state, shots=200, seed=17, basis=backend.basis)
    second, _ = sample_configurations(state, shots=200, seed=17, basis=backend.basis)
    assert np.array_equal(first, second)


def test_more_shots_retain_more_probability(hubbard_case):
    """The yield curve the ladder needs: retained mass grows with shots."""
    backend, _, _, model = hubbard_case
    _, ground = backend.ground_state(model.hamiltonian)
    masses = []
    for shots in (20, 200, 2000):
        _, record = sample_configurations(ground[:, 0], shots=shots, seed=2,
                                          basis=backend.basis)
        masses.append(record.retained_probability)
    assert masses[0] <= masses[1] + 1e-12 <= masses[2] + 1e-12


@pytest.mark.parametrize("n", [7, 9])
def test_mismatched_qubit_count_is_rejected(n):
    """A wrong `n` would silently post-select on the wrong bits, so it raises.

    This is the one input error the sampling contract cannot absorb: every
    other bad argument fails loudly, but a plausible-looking `n` just shifts
    the occupation mask and returns confidently wrong configurations.
    """
    state = np.full(2 ** 8, 2.0 ** -4)
    with pytest.raises(ValueError, match="2\\^n-length state"):
        sample_configurations(state, shots=10, seed=0, n=n, n_electrons=4,
                              sz=0.0)


def test_zero_state_is_rejected(hubbard_case):
    backend, _, _, _ = hubbard_case
    with pytest.raises(ValueError, match="zero state"):
        sample_configurations(np.zeros(backend.dimension), shots=10,
                              basis=backend.basis)


# ------------------------------------------------------------------ end to end

def test_run_qsci_is_variational_and_converges_to_exact(hubbard_case):
    """The oracle input is a validation category, and it must reach the exact value."""
    backend, operator, exact, model = hubbard_case
    _, ground = backend.ground_state(model.hamiltonian)
    partial, record = sample_configurations(
        ground[:, 0], shots=40, seed=1, basis=backend.basis,
        input_label="exact_ground_oracle")
    result = run_qsci(operator, partial, sampling=record, exact_energy=float(exact[0]))
    assert result.variational_gap >= -1e-9
    assert result.subspace_dimension == partial.size
    assert result.to_record()["projected_matrix_words"] == 0
    assert result.matrix_nonzeros > 0
    assert result.hermiticity_residual < 1e-12

    whole = run_qsci(operator, np.arange(backend.dimension), sampling=record,
                     exact_energy=float(exact[0]))
    assert whole.energy == pytest.approx(float(exact[0]), abs=1e-10)
    assert whole.variational_gap == pytest.approx(0.0, abs=1e-10)


def test_run_qsci_spin_arm_exists(kitaev_case):
    """§8C: the Kitaev cluster gets a raw computational-basis arm, not an excuse."""
    operator, exact = kitaev_case
    _, ground = operator.ground_state()
    indices, record = sample_configurations(
        ground[:, 0], shots=300, seed=4, input_label="exact_ground_oracle")
    assert record.discarded_shots == 0  # no sector to fall out of
    result = run_qsci(operator, indices, sampling=record,
                      exact_energy=float(exact[0]))
    assert result.variational_gap >= -1e-9
    assert result.subspace_dimension == indices.size


def test_record_is_json_serializable(hubbard_case):
    import json

    backend, operator, exact, model = hubbard_case
    _, ground = backend.ground_state(model.hamiltonian)
    indices, record = sample_configurations(ground[:, 0], shots=64, seed=9,
                                            basis=backend.basis)
    result = run_qsci(operator, indices, sampling=record,
                      exact_energy=float(exact[0]))
    round_tripped = json.loads(json.dumps(result.to_record()))
    assert round_tripped["method"] == "qsci"
    assert round_tripped["subspace_dimension"] == indices.size


# --------------------------------------------------------- §8C post-selection

def test_post_selection_discards_out_of_sector_samples():
    """A full-space state with weight outside the sector must report the loss."""
    model = hubbard(shape=(2, 2), t=1.0, U=4.0)
    n, n_electrons = model.n, int(model.metadata["n_electrons"])
    rng = np.random.default_rng(0)
    state = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    _, record = sample_configurations(
        state, shots=800, seed=3, n=n, n_electrons=n_electrons, sz=0.0,
        input_label="unphysical_random_state")
    assert record.discarded_shots > 0
    assert record.accepted_shots + record.discarded_shots == record.raw_shots
    assert 0.0 < record.discarded_fraction < 1.0
    assert record.repaired_shots == 0


def test_recovery_repairs_into_the_sector_and_reports_it():
    """Recovery is optional, deterministic, and never silent."""
    model = hubbard(shape=(2, 2), t=1.0, U=4.0)
    n, n_electrons = model.n, int(model.metadata["n_electrons"])
    rng = np.random.default_rng(0)
    state = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    words, record = sample_configurations(
        state, shots=400, seed=3, n=n, n_electrons=n_electrons, sz=0.0,
        recovery="occupancy", input_label="unphysical_random_state")
    assert record.repaired_shots > 0
    assert record.discarded_shots == 0
    assert record.recovery == "occupancy"
    # Every retained word must now sit in the target sector.
    backend = SectorStatevectorBackend(n, n_electrons, 0.0)
    assert set(words.tolist()) <= set(backend.basis.tolist())


def test_recovery_is_deterministic_given_occupancy():
    occupancy = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
    words = np.array([0b11111111, 0b00000000], dtype=np.int64)
    first = recover_configurations(words, occupancy, n=8, n_electrons=4, sz=0.0)
    second = recover_configurations(words, occupancy, n=8, n_electrons=4, sz=0.0)
    assert np.array_equal(first, second)
    backend = SectorStatevectorBackend(8, 4, 0.0)
    assert set(first.tolist()) <= set(backend.basis.tolist())


def test_sector_mode_rejects_recovery(hubbard_case):
    """Nothing to recover in sector mode, and pretending otherwise would lie."""
    backend, _, _, _ = hubbard_case
    state = np.full(backend.dimension, 1.0 / np.sqrt(backend.dimension))
    with pytest.raises(ValueError, match="nothing to recover"):
        sample_configurations(state, shots=10, basis=backend.basis,
                              recovery="occupancy")
