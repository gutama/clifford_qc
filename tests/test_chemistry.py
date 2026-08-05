"""Phase 5 chemistry layer (backlog 17): molecule models via the OpenFermion
bridge, the JW excitation pool, and the FAST-inspired baseline."""

import itertools
import pytest

pytest.importorskip("openfermion")
pytest.importorskip("openfermionpyscf")

from openfermion import (FermionOperator, commutator, normal_ordered,
                         number_operator, sz_operator)

from clifford_qc.matrix import exact_ground
from clifford_qc.backends import ExactMVBackend
from clifford_qc.algorithms import FastInspiredSelector, is_odd_y, run_adapt
from clifford_qc.models.chemistry import (_excitation_generators, _hf_reference,
                                          excitation_pool, h2, lih)

CHEMICAL_ACCURACY = 1.6e-3  # Hartree


def test_open_shell_hf_reference_respects_interleaved_spin_ordering():
    reference = _hf_reference(6, n_electrons=3, ms2=3)
    occupied = {op.qubits[0] for op in reference.ops}
    assert occupied == {0, 2, 4}
    with pytest.raises(ValueError, match="parity"):
        _hf_reference(6, n_electrons=2, ms2=1)


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


@pytest.mark.parametrize("n_qubits, n_electrons", [(4, 2), (6, 4), (8, 4)])
def test_excitation_generators_conserve_particle_number_and_sz(n_qubits, n_electrons):
    """Every singles/doubles generator must commute with the number and S_z
    operators. The earlier ``(i+j)%2 == (a+b)%2`` doubles filter admitted
    Delta S_z = +/-2 excitations (e.g. two spin-up into two spin-down); this
    is the physics-level guard against that regression."""
    zero = FermionOperator()
    number = number_operator(n_qubits)
    spin_z = sz_operator(n_qubits // 2)  # interleaved up/down spin-orbitals
    gens = _excitation_generators(n_qubits, n_electrons)
    assert gens
    for gen in gens:
        assert normal_ordered(commutator(gen, number)) == zero
        assert normal_ordered(commutator(gen, spin_z)) == zero


def test_excitation_pool_word_counts_follow_correct_generator_filter():
    """Lock the word counts after correcting the source-generator filter.

    The individual word rotors are not themselves symmetry preserving; the
    chemistry benchmark reports their sector leakage separately. The buggy doubles filter
    produced 176 words for H4 (16 extra from two Delta S_z = +/-2
    generators); the corrected S_z-conserving pool is 160. H2 has no
    Delta S_z = +/-2 doubles, so its 12-word pool is unchanged."""
    assert len(excitation_pool(4, 2)) == 12
    assert len(excitation_pool(8, 4)) == 160


def test_no_spin_changing_double_excitation_is_generated():
    """The specific spurious generator a†_5 a†_7 a_2 a_0 - h.c. (two spin-up
    occupied into two spin-down virtual) must not appear for H4."""
    spurious = normal_ordered(
        FermionOperator(((5, 1), (7, 1), (2, 0), (0, 0)))
        - FermionOperator(((0, 1), (2, 1), (7, 0), (5, 0))))
    gens = [normal_ordered(g) for g in _excitation_generators(8, 4)]
    # neither the generator nor its negation (opposite h.c. sign convention)
    assert all(g != spurious and g != -spurious for g in gens)


def test_strict_h4_selector_uses_finite_sample_trajectory_budget():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "benchmarks" / "run_chemistry_repair.py"
    spec = importlib.util.spec_from_file_location("run_chemistry_repair", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    selector, per_call = module.strict_h4_selector(10, 0.05)
    assert per_call == pytest.approx(0.005)
    assert selector.delta == pytest.approx(per_call)
    assert selector.bound == "eb"
    assert selector.method == "bonferroni"
    assert selector.near_tol is None


# ------------------------------------------------------ FCIDUMP ingestion (§5)


@pytest.fixture(scope="module")
def h2_fcidump(tmp_path_factory):
    """An FCIDUMP written by PySCF, as a downfolding step would hand over."""
    from pyscf import gto, scf
    from pyscf.tools import fcidump

    mol = gto.M(atom="H 0 0 0; H 0 0 0.7414", basis="sto-3g", verbose=0)
    mean_field = scf.RHF(mol).run()
    path = tmp_path_factory.mktemp("fcidump") / "h2.fcidump"
    fcidump.from_scf(mean_field, str(path))
    return path


@pytest.fixture(scope="module")
def h4_fcidump(tmp_path_factory):
    from pyscf import gto, scf
    from pyscf.tools import fcidump

    mol = gto.M(atom="H 0 0 0; H 0 0 0.9; H 0 0 1.8; H 0 0 2.7",
                basis="sto-3g", verbose=0)
    mean_field = scf.RHF(mol).run()
    path = tmp_path_factory.mktemp("fcidump") / "h4.fcidump"
    fcidump.from_scf(mean_field, str(path))
    return path


def test_fcidump_reproduces_the_pyscf_path_term_by_term(h2_fcidump, h2_model):
    """The only check that catches a wrong chemist/physicist reindexing.

    Eight of the 24 possible four-index permutations agree (the permutation
    symmetry of real two-electron integrals); the rest give a Hamiltonian that
    looks perfectly reasonable and has the wrong correlation energy. Comparing
    against the same molecule through ``openfermionpyscf`` is what pins it.
    """
    from clifford_qc.models.chemistry import fcidump_model

    ingested = fcidump_model(h2_fcidump, name="h2-fcidump")
    assert ingested.n == h2_model.n
    left, right = ingested.hamiltonian.to_labels(), h2_model.hamiltonian.to_labels()
    for label in set(left) | set(right):
        assert left.get(label, 0.0) == pytest.approx(right.get(label, 0.0), abs=1e-10)
    assert exact_ground(ingested.hamiltonian.to_mv())[0] == pytest.approx(
        h2_model.metadata["fci_energy"], abs=1e-7)


def _orbital_sign_gauge(labels, flipped_orbitals):
    """``c_p -> -c_p`` on the given spatial orbitals, applied to a Pauli image.

    The sign of a molecular orbital is not physics: SCF fixes each eigenvector
    only up to sign. Under Jordan-Wigner, negating mode ``p`` is conjugation by
    ``Z`` on its two spin-orbital qubits, and ``Z_j W Z_j = -W`` exactly when
    ``W`` carries an ``X`` or ``Y`` at ``j`` -- so the whole gauge orbit is
    reachable by flipping signs of coefficients, with no rebuild.
    """
    qubits = [q for p in flipped_orbitals for q in (2 * p, 2 * p + 1)]
    return {label: (-value if sum(1 for q in qubits if label[q] in "XY") % 2 else value)
            for label, value in labels.items()}


def _agreement_up_to_orbital_signs(left, right, n_orbitals):
    """Worst coefficient gap, minimized over the ``2^n_orbitals`` sign choices."""
    keys = set(left) | set(right)
    return min(
        max(abs(left.get(k, 0.0) - candidate.get(k, 0.0)) for k in keys)
        for size in range(n_orbitals + 1)
        for flips in itertools.combinations(range(n_orbitals), size)
        for candidate in (_orbital_sign_gauge(right, flips),))


def test_fcidump_ingestion_on_a_larger_active_space(h4_fcidump):
    """Term-by-term against an independent PySCF path, quotiented by MO sign.

    Comparing the two constructions coefficient-by-coefficient was flaky about
    once in a dozen runs, and not by a hair: the gap was either 1e-15 or
    7.4e-02, with the SCF energy identical to 1e-15 either way. The cause is
    that the two SCF runs occasionally settle on opposite signs for a molecular
    orbital, which is a gauge choice with no physical content -- flipping one MO
    of this H4 chain by hand reproduces the failing signature exactly, term
    count and all, and leaves the sector spectrum fixed to 1e-14.

    Quotienting by that gauge rather than loosening the tolerance keeps the
    check strict, and keeps its teeth: of the 24 possible chemist-to-physicist
    reindexings, 8 agree with the correct one (the permutation symmetry of real
    two-electron integrals) and the other 16 are still caught here -- no sign
    pattern rescues a wrong transpose.
    """
    from clifford_qc.models.chemistry import fcidump_model, h4_chain

    ingested = fcidump_model(h4_fcidump)
    reference = h4_chain(0.9)
    left, right = ingested.hamiltonian.to_labels(), reference.hamiltonian.to_labels()
    assert len(left) == len(right) == 185
    assert _agreement_up_to_orbital_signs(left, right, ingested.n // 2) < 1e-10


def test_orbital_sign_gauge_is_a_gauge_and_not_a_loophole(h4_fcidump):
    """The quotient must absorb an MO sign flip and nothing else.

    Without the second half, the first would be satisfiable by any comparison
    weak enough to pass -- so this pins that a flipped orbital is forgiven while
    a perturbed coefficient is not.
    """
    from clifford_qc.models.chemistry import fcidump_model

    labels = fcidump_model(h4_fcidump).hamiltonian.to_labels()
    n_orbitals = 4
    for orbital in range(n_orbitals):
        flipped = _orbital_sign_gauge(labels, (orbital,))
        raw = max(abs(labels[k] - flipped[k]) for k in labels)
        assert raw > 1e-3, "flipping an orbital should visibly move coefficients"
        assert _agreement_up_to_orbital_signs(labels, flipped, n_orbitals) < 1e-12

    perturbed = dict(labels)
    worst_label = max(labels, key=lambda k: abs(labels[k]))
    perturbed[worst_label] += 1e-3
    assert _agreement_up_to_orbital_signs(labels, perturbed, n_orbitals) >= 1e-4


def test_fcidump_model_carries_orbital_and_sector_metadata(h2_fcidump):
    from clifford_qc.models.chemistry import fcidump_model

    model = fcidump_model(h2_fcidump)
    metadata = model.metadata
    assert metadata["source"] == "fcidump"
    assert metadata["n_spatial_orbitals"] == 2
    assert metadata["spin_orbitals"] == 4
    assert metadata["n_electrons"] == 2
    assert metadata["sz"] == 0.0
    assert metadata["spin_convention"] == "interleaved"
    assert metadata["core_energy"] > 0.0  # nuclear repulsion lands in the constant
    # the reference determinant fills the advertised electron count
    assert sum(1 for op in model.reference.ops) == 2


def test_pyscf_models_carry_the_same_metadata_keys(h2_model, lih_model):
    for model in (h2_model, lih_model):
        for key in ("kind", "source", "n_spatial_orbitals", "spin_orbitals",
                    "spin_convention", "n_electrons", "sz"):
            assert key in model.metadata
        assert model.metadata["spin_orbitals"] == model.n
