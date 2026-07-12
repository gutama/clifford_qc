"""Phase 5 chemistry layer (backlog 17): molecule models via the OpenFermion
bridge, the JW excitation pool, and the FAST-inspired baseline."""

import pytest

pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")

from clifford_qc.matrix import exact_ground
from clifford_qc.backends import ExactMVBackend
from clifford_qc.algorithms import FastInspiredSelector, is_odd_y, run_adapt
from clifford_qc.models.chemistry import excitation_pool, h2, lih

CHEMICAL_ACCURACY = 1.6e-3  # Hartree


@pytest.fixture(scope="module")
def h2_model():
    return h2()


@pytest.fixture(scope="module")
def lih_model():
    return lih()


def test_h2_hamiltonian_reproduces_pyscf_energies(h2_model):
    """Backlog 17 acceptance: reference energies and mappings recorded.

    Tolerance 1e-7 Ha is ~4 orders tighter than chemical accuracy (so it
    still catches any JW-mapping bug) while tolerating last-decimal PySCF
    drift across BLAS backends.
    """
    E0, _ = exact_ground(h2_model.hamiltonian.to_mv())
    assert E0 == pytest.approx(h2_model.metadata["fci_energy"], abs=1e-7)
    ref = ExactMVBackend().expectation(h2_model.reference, h2_model.hamiltonian, ())
    assert ref == pytest.approx(h2_model.metadata["hf_energy"], abs=1e-7)
    assert h2_model.n == 4
    assert h2_model.metadata["n_electrons"] == 2


def test_lih_active_space_reduces_to_four_qubits(lih_model):
    assert lih_model.n == 4
    assert lih_model.metadata["n_electrons"] == 2
    ref = ExactMVBackend().expectation(lih_model.reference, lih_model.hamiltonian, ())
    # frozen-core constant lands in the identity term: <H>_HF matches pyscf HF
    assert ref == pytest.approx(lih_model.metadata["hf_energy"], abs=1e-7)
    E0, _ = exact_ground(lih_model.hamiltonian.to_mv())
    assert E0 < ref  # correlation energy within the active space
    # active-space FCI sits between HF and the full FCI
    assert lih_model.metadata["fci_energy"] < E0 < ref


def test_excitation_pool_is_odd_y_and_covers_h2(h2_model):
    pool = excitation_pool(4, 2)
    assert pool
    assert all(is_odd_y(op.word) for op in pool)
    assert len({op.word.code for op in pool}) == len(pool)
    res = run_adapt(h2_model, pool, max_operators=4, threshold=1e-7)
    E0, _ = exact_ground(h2_model.hamiltonian.to_mv())
    assert abs(res.energy - E0) < 1e-8
    assert len(res.labels) == 1  # H2 needs a single double-excitation word


def test_fast_inspired_selector_reaches_chemical_accuracy(h2_model):
    E0, _ = exact_ground(h2_model.hamiltonian.to_mv())
    pool = excitation_pool(4, 2)
    res = run_adapt(h2_model, pool,
                    selector=FastInspiredSelector(shots=2048, seed=1),
                    max_operators=4, threshold=1e-6)
    assert abs(res.energy - E0) < CHEMICAL_ACCURACY
    assert res.total_shots == 2048 * len([r for r in res.records if r.shots_added])
    assert all(r.status.value in ("fast_proxy", "below_threshold")
               for r in res.records)


def test_fast_inspired_selector_is_seed_deterministic(h2_model):
    pool = excitation_pool(4, 2)
    a = run_adapt(h2_model, pool, selector=FastInspiredSelector(shots=512, seed=7),
                  max_operators=3)
    b = run_adapt(h2_model, pool, selector=FastInspiredSelector(shots=512, seed=7),
                  max_operators=3)
    assert a.labels == b.labels


def test_fast_inspired_selector_validates_shots():
    with pytest.raises(ValueError, match="shots"):
        FastInspiredSelector(shots=0, seed=0)
