"""Phase 4: stim stabilizer backend, Clifford-point search (backlog 15),
stabilizer Hamiltonian approximation, and stabilizer-seeded residual ADAPT."""

import itertools
import math
import random

import pytest

from clifford_qc.ir import PauliSum, Program
from clifford_qc.states import expectation
from clifford_qc.matrix import exact_ground
from clifford_qc.models import tfim, random_ising
from clifford_qc.backends import ExactMVBackend
from clifford_qc.algorithms import (
    CLIFFORD_ANGLES, bound_hva_program, clifford_point_search, hva_program,
    local_pool, run_adapt, seed_model, stabilizer_ground_program,
    stabilizer_hamiltonian_approximation,
)

stim = pytest.importorskip("stim")

from clifford_qc.backends.stabilizer import StimBackend  # noqa: E402


# ---------------------------------------------------------------------------
# StimBackend


def test_stim_backend_matches_exact_mv_on_clifford_programs():
    rng = random.Random(0)
    obs = PauliSum.from_labels({"ZZI": 0.7, "XII": -0.4, "IYY": 0.25})
    for _ in range(10):
        prog = Program(3)
        for _ in range(6):
            if rng.random() < 0.5:
                label = "".join(rng.choice("IXYZ") for _ in range(3))
                if set(label) != {"I"}:
                    prog.rotor(label, rng.choice(CLIFFORD_ANGLES))
            else:
                prog.clifford(rng.choice(["H", "S"]), rng.randrange(3))
        exact = ExactMVBackend().expectation(prog, obs)
        assert StimBackend().expectation(prog, obs) == pytest.approx(exact, abs=1e-9)


def test_stim_backend_rejects_initial_state():
    with pytest.raises(ValueError, match="bake the"):
        StimBackend().expectation(Program(2), PauliSum.from_labels({"ZZ": 1.0}),
                                  initial_state=object())


# ---------------------------------------------------------------------------
# Clifford-point search (backlog 15: reproducible by seed)


def test_clifford_point_search_matches_exhaustive_and_is_reproducible():
    m = tfim(3, 1.0, 0.7)
    prog = hva_program(m, depth=1)  # 2 parameters -> 16 grid points
    backend = StimBackend()
    exhaustive = min(backend.expectation(prog, m.hamiltonian, v)
                     for v in itertools.product(CLIFFORD_ANGLES, repeat=2))
    a = clifford_point_search(prog, m.hamiltonian, backend, seed=3, restarts=4)
    b = clifford_point_search(prog, m.hamiltonian, backend, seed=3, restarts=4)
    assert a.energy == pytest.approx(exhaustive, abs=1e-9)
    assert a.values == b.values and a.energy == b.energy
    assert a.evaluations == b.evaluations > 0


def test_bound_hva_program_is_clifford_at_clifford_angles():
    m = tfim(3)
    prep = bound_hva_program(m, depth=1, values=[math.pi / 2, math.pi])
    assert prep.is_clifford_only()
    assert not len(prep.parameters)
    rho_a = prep.state()
    rho_b = ExactMVBackend().state(hva_program(m, 1), [math.pi / 2, math.pi])
    diff = rho_a + (-1.0) * rho_b
    assert max((abs(c) for c in diff.terms.values()), default=0.0) < 1e-12


# ---------------------------------------------------------------------------
# Stabilizer Hamiltonian approximation


def test_stabilizer_approximation_picks_phase_appropriate_cluster():
    ordered = stabilizer_hamiltonian_approximation(tfim(6, 1.0, 0.5).hamiltonian)
    assert all(w.label.count("Z") == 2 for w, _ in ordered.selected)  # bonds
    para = stabilizer_hamiltonian_approximation(tfim(6, 1.0, 1.5).hamiltonian)
    assert all(w.label.count("X") == 1 for w, _ in para.selected)     # fields
    assert para.scaffold_energy == pytest.approx(-9.0)


def test_stabilizer_approximation_keeps_dependent_consistent_words():
    """Periodic TFIM n=3: the three bonds multiply to identity, so only two
    are independent but the third is sign-consistent and stays selected."""
    approx = stabilizer_hamiltonian_approximation(
        tfim(3, 1.0, 0.1, periodic=True).hamiltonian)
    bond_selected = [w for w, _ in approx.selected if w.label.count("Z") == 2]
    bond_generators = [w for w, _ in approx.generators if w.label.count("Z") == 2]
    assert len(bond_selected) == 3
    assert len(bond_generators) == 2


def test_stabilizer_approximation_excludes_inconsistent_signs():
    """ZZ frustration triangle: two antiferromagnetic bonds force the third
    word's sign, so the frustrated third bond is excluded."""
    H = PauliSum.from_labels({"ZZI": -1.0, "IZZ": -0.9, "ZIZ": 0.8})
    approx = stabilizer_hamiltonian_approximation(H)
    assert [w.label for w in approx.excluded] == ["ZIZ"]
    assert approx.scaffold_energy == pytest.approx(-1.9)


def test_stabilizer_approximation_excludes_anticommuting_words():
    H = PauliSum.from_labels({"ZI": -1.0, "XI": -0.5})
    approx = stabilizer_hamiltonian_approximation(H)
    assert [w.label for w in approx.excluded] == ["XI"]


def test_stabilizer_ground_program_realizes_target_signs():
    for model in (tfim(5, 1.0, 0.5), random_ising(5, seed=2)):
        approx = stabilizer_hamiltonian_approximation(model.hamiltonian)
        prep = stabilizer_ground_program(approx)
        assert prep.is_clifford_only()
        rho = prep.state()
        for word, sign in approx.generators:
            assert expectation(rho, word.to_mv()).real == pytest.approx(sign, abs=1e-9)
        # scaffold energy accounting matches the prepared state on H_stab
        e_stab = sum(model.hamiltonian.terms[w.code].real
                     * expectation(rho, w.to_mv()).real
                     for w, _ in approx.selected)
        assert e_stab == pytest.approx(approx.scaffold_energy, abs=1e-9)


# ---------------------------------------------------------------------------
# Stabilizer-seeded residual ADAPT


def test_seeded_residual_adapt_runs_and_pins_the_tfim_negative_result():
    """The h=0.5 TFIM scaffold (Z-bond cluster) starts well below the |+...+>
    reference yet residual ADAPT from it converges *worse* — raw scaffold
    energy does not imply fewer non-Clifford corrections when the scaffold
    is not symmetry-aligned with the ground sector. Pinned as the Paper B
    go/no-go datum for this family."""
    m = tfim(4, 1.0, 0.5)
    pool = local_pool(4, periodic_context=False)

    approx = stabilizer_hamiltonian_approximation(m.hamiltonian)
    seeded = seed_model(m, stabilizer_ground_program(approx), "stab")
    assert seeded.n == m.n and "stab" in seeded.name

    base_start = ExactMVBackend().expectation(m.reference, m.hamiltonian, ())
    seed_start = ExactMVBackend().expectation(seeded.reference, m.hamiltonian, ())
    assert seed_start < base_start  # the scaffold starts lower...

    res = run_adapt(seeded, pool, max_operators=6, allow_repeats=True)
    baseline = run_adapt(m, pool, max_operators=6, allow_repeats=True)
    assert baseline.relative_error < 1e-9      # ...but the baseline converges
    assert res.relative_error > baseline.relative_error
    assert res.optimizer_evaluations > 0 and baseline.optimizer_evaluations > 0
    energies = [r.energy for r in res.records if r.energy is not None]
    assert all(b <= a + 1e-9 for a, b in zip(energies, energies[1:]))


def test_seed_model_rejects_wrong_qubit_count():
    m = tfim(3)
    with pytest.raises(ValueError, match="qubit counts differ"):
        seed_model(m, Program(4))
