"""The matched-contract comparison's two structural claims.

``benchmarks/run_matched_h4.py`` reports a table; two of its rows carry claims
that are not obvious and would be easy to break silently, so they are checked
here on a small system rather than only asserted on H4:

1. generator *resolution* changes the measured word universe without changing
   the retained subspace, so a width quoted without its resolution is not a
   property of the method;
2. the sector-leakage rejection is an operator-level test, so it can discard a
   generator family whose Ritz vector is nonetheless exactly in sector.

Both run on the 2x2 Hubbard plaquette, which is eight qubits like H4 but far
cheaper, and both are structural rather than numerical: they would survive a
change of tolerances and fail on a change of meaning.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from clifford_qc.algorithms.pools import is_odd_y
from clifford_qc.backends import ExactMVBackend
from clifford_qc.ir import PauliWord
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace import (dense_basis, determinant_excitations,
                                  occupied_spin_orbitals, pauli_orbit,
                                  run_acase, sector_leakage)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))


@pytest.fixture(scope="module")
def plaquette():
    model = hubbard((2, 2), t=1.0, U=4.0, periodic=False)
    rho = ExactMVBackend().state(model.reference, ())
    determinants = determinant_excitations(
        model.n, occupied_spin_orbitals(model), max_rank=2)
    codes: dict[int, PauliWord] = {}
    for generator in determinants:
        for code in sorted(generator.mv.terms):
            word = PauliWord(model.n, code)
            if code and is_odd_y(word) and code not in codes:
                codes[code] = word
    return model, rho, determinants, pauli_orbit(list(codes.values()))


def _retained_span(rho, result, bank) -> np.ndarray:
    basis = dense_basis(rho, [bank._generators[i] for i in result.result.indices])
    q, _ = np.linalg.qr(basis)
    return q


def test_resolution_changes_width_without_changing_the_subspace(plaquette):
    """Same span, same energy, different measured width.

    This is the finding that makes W representation-dependent. Comparing
    principal angles rather than energies is what makes it a statement about
    the subspace: two different subspaces can share a Ritz value by accident,
    but they cannot share every direction.
    """
    model, rho, determinants, words = plaquette
    coarse = run_acase(rho, model.hamiltonian, determinants, max_size=6,
                       leakage_tol=1e-10)
    fine = run_acase(rho, model.hamiltonian, words, max_size=6)

    assert coarse.energy == pytest.approx(fine.energy, abs=1e-12)
    assert len(coarse.result.basis_labels) == len(fine.result.basis_labels)

    singular = np.linalg.svd(
        _retained_span(rho, coarse, coarse.bank).conj().T
        @ _retained_span(rho, fine, fine.bank), compute_uv=False)
    assert np.allclose(singular, 1.0, atol=1e-9), "spans differ"

    coarse_width = coarse.result.resources["word_universe"]
    fine_width = fine.result.resources["word_universe"]
    assert fine_width != coarse_width, (
        "the two resolutions have become equally wide; the benchmark's "
        "headline comparison is then vacuous")


def test_selection_width_exceeds_the_retained_width(plaquette):
    """Scoring a rejected candidate still costs its words.

    The manuscript's ledger reports the retained subspace's universe. The cache
    is larger whenever any candidate was scored and not kept, and the benchmark
    reports both.
    """
    model, rho, determinants, _ = plaquette
    result = run_acase(rho, model.hamiltonian, determinants, max_size=6,
                       leakage_tol=1e-10)
    bank = result.bank
    retained = len(bank.word_set(result.result.indices))
    cache = len(bank.word_set(range(len(bank._generators))))
    assert cache >= retained
    assert retained == result.result.resources["word_universe"]


def test_word_generators_break_symmetry_as_operators(plaquette):
    """Every odd-Y word leaks maximally, which is why the rule rejects them."""
    _, _, _, words = plaquette
    leakage = [sector_leakage(g)["particle_number"] for g in words]
    assert min(leakage) > 1.0


def test_but_their_ritz_vector_stays_in_sector(plaquette):
    """The operator-level test is stronger than the state-level question.

    A word acting on a determinant returns a determinant, so the span built on
    a determinant reference is symmetric even though its generators are not.
    """
    model, rho, _, words = plaquette
    result = run_acase(rho, model.hamiltonian, words, max_size=6)
    basis = dense_basis(
        rho, [result.bank._generators[i] for i in result.result.indices])
    psi = basis @ np.asarray(result.result.coefficients)[:, 0]
    psi = psi / np.linalg.norm(psi)

    index = np.arange(2 ** model.n)
    electrons = np.array([bin(i).count("1") for i in index])
    probability = np.abs(psi) ** 2
    weight = probability[electrons == model.metadata["n_electrons"]].sum()
    assert weight == pytest.approx(1.0, abs=1e-10)


def test_leakage_rejection_discards_the_whole_word_pool(plaquette):
    """With the declared tolerance the fine pool yields nothing at all."""
    model, rho, _, words = plaquette
    result = run_acase(rho, model.hamiltonian, words, max_size=6,
                       leakage_tol=1e-10)
    assert len(result.result.basis_labels) == 1


@pytest.fixture()
def committed_record():
    path = (Path(__file__).resolve().parents[1] / "benchmarks"
            / "reference_results" / "matched_h4.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _gate_with(monkeypatch, fresh) -> int:
    """Run the CI gate against a supplied 'fresh' record.

    ``check_matched_h4`` binds ``build_record`` at import, so the name to
    replace is the checker's, not the generator module's.
    """
    import check_matched_h4

    monkeypatch.setattr(check_matched_h4, "build_record", lambda: fresh)
    return check_matched_h4.main()


def test_gate_accepts_an_unchanged_record(monkeypatch, committed_record):
    assert _gate_with(monkeypatch, copy.deepcopy(committed_record)) == 0


def test_gate_tolerates_the_cross_blas_drift_that_failed_ci(
        monkeypatch, committed_record):
    """The reason this gate replaced a byte diff.

    These are the exact perturbations the CI runner produced: a sector weight
    of 1.0 serialized as 0.9999999999999999 and energies moved by ~9e-16 Ha,
    with every count identical. A byte comparison called that a regression.
    """
    fresh = copy.deepcopy(committed_record)
    for row in fresh["rows"]:
        if row.get("sector_weight") == 1.0:
            row["sector_weight"] = 0.9999999999999999
        if isinstance(row.get("error_hartree"), float):
            row["error_hartree"] += 8.9e-16
            row["error_millihartree"] += 8.9e-13
        if isinstance(row.get("prelude"), dict):
            row["prelude"]["error_millihartree"] += 8.9e-13
    assert _gate_with(monkeypatch, fresh) == 0


def test_gate_scales_condition_tolerance_to_the_condition_number(
        monkeypatch, committed_record):
    """kappa is only accurate to kappa*eps, and the gate asks for exactly that.

    CI failed on a 4e-10 relative wobble in the Krylov arm's kappa_S of 6.6e10,
    whose own accuracy bound is 1.5e-5. A flat tolerance either accepts that
    drift and goes blind on the well-conditioned arms, or holds those tight and
    fails on noise. Both directions are pinned here.
    """
    rows = committed_record["rows"]
    large = next(i for i, r in enumerate(rows)
                 if (r.get("condition_number") or 0) > 1e6)
    small = next(i for i, r in enumerate(rows)
                 if 1.0 < (r.get("condition_number") or 0) < 10.0)

    tolerated = copy.deepcopy(committed_record)
    tolerated["rows"][large]["condition_number"] *= 1 + 4e-10
    assert _gate_with(monkeypatch, tolerated) == 0, "kappa*eps drift rejected"

    rejected = copy.deepcopy(committed_record)
    rejected["rows"][large]["condition_number"] *= 1 + 1e-4
    assert _gate_with(monkeypatch, rejected) == 1, "real kappa change accepted"

    # the loosening must not leak onto arms whose kappa is well determined
    tight = copy.deepcopy(committed_record)
    tight["rows"][small]["condition_number"] *= 1 + 1e-6
    assert _gate_with(monkeypatch, tight) == 1, (
        "a well-conditioned arm inherited the large-kappa tolerance")


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda r: r["rows"][3].__setitem__("state_preparations", 91),
                 id="preparation-count"),
    pytest.param(lambda r: r["rows"][6].__setitem__("final_words", 2241),
                 id="word-universe"),
    pytest.param(lambda r: r["rows"][4]["labels"].__setitem__(1, "BOGUS"),
                 id="selected-label"),
    pytest.param(lambda r: r["rows"][4].__setitem__("error_hartree", 3.1e-3),
                 id="energy-beyond-tolerance"),
    pytest.param(lambda r: r["rows"][6].__setitem__("effective_rank", 8),
                 id="effective-rank"),
])
def test_gate_rejects_a_real_regression(monkeypatch, committed_record, mutate):
    fresh = copy.deepcopy(committed_record)
    mutate(fresh)
    assert _gate_with(monkeypatch, fresh) == 1
