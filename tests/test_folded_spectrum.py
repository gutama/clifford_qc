"""Folded-spectrum roots against dense oracles on small Hubbard spans.

Every quantity is held to an explicit statevector construction that uses no
multivector product: the folded block to ``B'(H - w)^2 B``, the folded values
to the dense generalized eigenproblem, the intervals to the full spectrum.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from benchmarks.run_mapping_axis import _raw_pool
from clifford_qc.backends import ExactMVBackend
from clifford_qc.dense_reference import to_matrix
from clifford_qc.models.lattice import hubbard
from clifford_qc.multivector import MV
from clifford_qc.subspace import (
    FoldedSpectrum,
    MatrixElementBank,
    SecondMomentBank,
    folded_spectrum_roots,
    refined_ritz_root,
)
from clifford_qc.subspace.generators import Generator, identity_generator
from clifford_qc.subspace.reference import dense_basis
from clifford_qc.subspace.second_moment import RESOLUTION


def _chain(size=5, shift=0.0):
    """Hubbard 1x3 with the reference and the first raw-pool generators."""
    model = hubbard(3)
    hamiltonian = model.hamiltonian.to_mv() + MV.scalar(model.n, shift)
    generators = [identity_generator(model.n), *_raw_pool(model)[:size - 1]]
    rho = ExactMVBackend().state(model.reference, ())
    bank = MatrixElementBank(rho, hamiltonian, generators)
    return hamiltonian, generators, rho, bank


@pytest.fixture(scope="module")
def chain():
    hamiltonian, generators, rho, bank = _chain()
    dense_h = to_matrix(hamiltonian)
    return {"bank": bank, "moments": SecondMomentBank(bank), "rho": rho,
            "basis": dense_basis(rho, generators), "h": dense_h,
            "spectrum": np.linalg.eigvalsh(dense_h), "result": bank.solve()}


def _dense_folded(chain, target):
    shifted = chain["h"] - target * np.eye(chain["h"].shape[0])
    B = chain["basis"]
    return B.conj().T @ shifted @ shifted @ B


def _contains(spectrum, interval, slack=1e-12):
    lo, hi = interval
    return bool(np.any((spectrum >= lo - slack) & (spectrum <= hi + slack)))


def _state(chain, root):
    psi = chain["basis"] @ root.coefficients
    return psi / np.linalg.norm(psi)


# ------------------------------------------------------------------ the block

@pytest.mark.parametrize("target", [-8.0, -6.3, -2.0, 0.5, 4.0])
def test_the_folded_block_is_the_dense_projection(chain, target):
    F = FoldedSpectrum(chain["moments"]).folded_matrix(target)
    dense = _dense_folded(chain, target)
    assert np.allclose(F, dense, rtol=0, atol=1e-11 * max(1.0, np.max(np.abs(dense))))
    assert np.allclose(F, F.conj().T)


def test_the_span_is_proper_so_the_test_is_not_trivial(chain):
    residuals = [chain["moments"].residual(chain["result"], k).residual_norm
                 for k in range(len(chain["result"].energies))]
    assert min(residuals) > 1e-3


@pytest.mark.parametrize("target", [-7.5, -5.0, -1.0, 2.0])
def test_folded_values_are_the_dense_generalized_roots(chain, target):
    result = FoldedSpectrum(chain["moments"]).roots(target)
    B = chain["basis"]
    gram = B.conj().T @ B
    whiten = np.linalg.inv(np.linalg.cholesky(gram))
    dense = np.linalg.eigvalsh(whiten @ _dense_folded(chain, target) @ whiten.conj().T)
    got = [root.folded_value for root in result.roots]
    assert result.effective_rank == len(dense)
    assert np.allclose(got, dense, rtol=1e-9, atol=1e-11)
    assert all(a <= b + 1e-12 for a, b in zip(got, got[1:]))


@pytest.mark.parametrize("target", [-7.5, -5.0, -1.0, 2.0])
def test_each_root_reports_its_own_state(chain, target):
    for root in FoldedSpectrum(chain["moments"]).roots(target).roots:
        psi = _state(chain, root)
        h_psi = chain["h"] @ psi
        energy = float(np.vdot(psi, h_psi).real)
        assert root.energy == pytest.approx(energy, abs=1e-11)
        assert root.folded_value == pytest.approx(
            float(np.linalg.norm(h_psi - target * psi) ** 2), abs=1e-10)
        assert root.variance == pytest.approx(
            float(np.linalg.norm(h_psi - energy * psi) ** 2), abs=1e-10)
        # mu = sigma^2 + (E - w)^2, to the folded value's resolution floor.
        assert abs(root.folded_value - root.variance - (root.energy - target) ** 2) <= (
            10 * RESOLUTION * root.folded_scale)
        norm = float((root.coefficients.conj() @ chain["bank"].matrices()[0]
                      @ root.coefficients).real)
        assert norm == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------------ the intervals

@pytest.mark.parametrize("target", [-9.0, -7.2, -6.0, -3.3, 0.0, 1.7, 6.0])
def test_both_intervals_contain_an_eigenvalue_for_every_root(chain, target):
    for root in FoldedSpectrum(chain["moments"]).roots(target).roots:
        assert _contains(chain["spectrum"], root.target_interval())
        assert _contains(chain["spectrum"], root.weinstein_interval())
        assert root.weinstein_bound <= root.target_bound + 1e-12


def test_an_interior_target_isolates_an_interior_level(chain):
    """Somewhere above the ground level, a folded root pins a single level."""
    spectrum = chain["spectrum"]
    levels = np.unique(np.round(spectrum, 9))
    isolated = 0
    for theta in chain["result"].energies[1:]:
        root = refined_ritz_root(chain["result"], list(chain["result"].energies).index(theta),
                                 moments=chain["moments"])
        lo, hi = root.weinstein_interval()
        inside = levels[(levels >= lo) & (levels <= hi)]
        isolated += int(len(inside) == 1 and inside[0] > levels[0] + 1e-9)
    assert isolated >= 1


# ------------------------------------------------------------------ refinement

def test_refined_ritz_never_has_a_larger_residual_than_its_ritz_vector(chain):
    result, moments = chain["result"], chain["moments"]
    tighter = 0
    for k, theta in enumerate(result.energies):
        ritz = moments.residual(result, k)
        refined = refined_ritz_root(result, k, moments=moments)
        assert refined.target == theta
        assert refined.folded_value <= ritz.variance + RESOLUTION * refined.folded_scale
        assert refined.residual_norm <= math.sqrt(max(refined.folded_value, 0.0)) + 1e-12
        tighter += int(refined.folded_value < 0.99 * ritz.variance)
    assert tighter >= 1


def test_refined_ritz_builds_its_own_moments_when_none_are_given(chain):
    a = refined_ritz_root(chain["result"], 0)
    b = refined_ritz_root(chain["result"], 0, moments=chain["moments"])
    assert a.folded_value == pytest.approx(b.folded_value, abs=1e-12)


def test_a_far_lower_target_selects_the_ritz_ground_state(chain):
    result = chain["result"]
    root = FoldedSpectrum(chain["moments"]).roots(-1e4, count=1).lowest
    # mu(w) = <H^2> - 2w<H> + w^2: as w -> -inf the lowest root minimizes <H>.
    assert root.energy == pytest.approx(result.energies[0], abs=1e-3)


# ------------------------------------------------------------------ minimization

def test_variance_minimization_is_monotone_and_reaches_a_fixed_point(chain):
    result, moments = chain["result"], chain["moments"]
    run = FoldedSpectrum(moments).minimize_variance(result.energies[0])
    assert run.converged and run.monotone
    values = [step[1] for step in run.trace]
    assert all(b <= a + RESOLUTION * run.root.folded_scale for a, b in zip(values, values[1:]))
    assert run.root.energy == pytest.approx(run.root.target, abs=1e-10)
    assert run.root.folded_value == pytest.approx(run.root.variance, abs=1e-10)
    assert run.root.variance <= moments.residual(result, 0).variance + 1e-12
    assert _contains(chain["spectrum"], run.root.weinstein_interval())


def test_the_fixed_point_is_a_variance_stationary_point(chain):
    """At E = w, F c = mu S c is the Lagrange condition for the variance."""
    bank = chain["bank"]
    run = FoldedSpectrum(chain["moments"]).minimize_variance(chain["result"].energies[0])
    c, energy = run.root.coefficients, run.root.energy
    S, H = bank.matrices()
    K = chain["moments"].matrix()
    gradient = K @ c - 2.0 * energy * (H @ c) - (run.root.variance - energy ** 2) * (S @ c)
    assert np.linalg.norm(gradient) <= 1e-8 * max(1.0, np.linalg.norm(K @ c))


def test_minimization_stops_at_its_iteration_cap(chain):
    run = FoldedSpectrum(chain["moments"]).minimize_variance(
        chain["result"].energies[0] + 1.0, max_iterations=1)
    assert run.iterations == 1 and not run.converged


# ------------------------------------------------------------------ invariance

def test_a_constant_shift_moves_target_and_energy_together():
    shift = 37.5
    _, _, _, plain = _chain()
    _, _, _, moved = _chain(shift=shift)
    a = FoldedSpectrum(SecondMomentBank(plain)).roots(-5.0).roots
    b = FoldedSpectrum(SecondMomentBank(moved)).roots(-5.0 + shift).roots
    for x, y in zip(a, b):
        assert y.energy == pytest.approx(x.energy + shift, abs=1e-9)
        assert y.folded_value == pytest.approx(x.folded_value, abs=1e-8)
        assert y.variance == pytest.approx(x.variance, abs=1e-8)
        overlap = abs(np.vdot(x.coefficients, y.coefficients))
        assert overlap == pytest.approx(
            np.linalg.norm(x.coefficients) * np.linalg.norm(y.coefficients), rel=1e-7)


# ------------------------------------------------------------------ edge cases

def test_a_dependent_generator_is_truncated_and_the_intervals_survive(chain):
    hamiltonian, generators, rho, _ = _chain()
    twin = Generator("twin", generators[1].mv * 2.0 + generators[2].mv * 1e-14)
    bank = MatrixElementBank(rho, hamiltonian, [*generators, twin])
    result = FoldedSpectrum(SecondMomentBank(bank)).roots(-6.0)
    assert result.effective_rank < len(bank)
    for root in result.roots:
        assert _contains(chain["spectrum"], root.target_interval())


def test_an_eigenstate_span_reports_unresolved_but_valid_intervals():
    model = hubbard(2)
    generators = [identity_generator(model.n), *_raw_pool(model)]
    rho = ExactMVBackend().state(model.reference, ())
    bank = MatrixElementBank(rho, model.hamiltonian, generators)
    spectrum = np.linalg.eigvalsh(to_matrix(model.hamiltonian.to_mv()))
    ground = bank.solve().energies[0]
    root = refined_ritz_root(bank.solve(), 0)
    assert not root.resolved and not root.variance_resolved
    lo, hi = root.target_interval()
    assert lo <= ground <= hi and _contains(spectrum, (lo, hi))
    assert root.target_bound > 0.0


def test_inputs_are_validated(chain):
    spectrum = FoldedSpectrum(chain["moments"])
    with pytest.raises(TypeError):
        FoldedSpectrum(chain["bank"])
    with pytest.raises(ValueError):
        spectrum.roots(float("nan"))
    with pytest.raises(ValueError):
        spectrum.roots(-5.0, count=0)
    with pytest.raises(ValueError):
        spectrum.roots(-5.0, count=len(chain["bank"]) + 1)
    with pytest.raises(ValueError):
        spectrum.minimize_variance(-5.0, max_iterations=0)
    with pytest.raises(ValueError):
        spectrum.minimize_variance(-5.0, tolerance=-1.0)
    with pytest.raises(IndexError):
        refined_ritz_root(chain["result"], len(chain["result"].energies))
    _, _, _, other = _chain()
    with pytest.raises(ValueError):
        refined_ritz_root(chain["result"], 0, moments=SecondMomentBank(other))


def test_a_metric_changing_ridge_is_refused():
    """A ridge solves the pencil in ``S + Delta``, so its vectors are not folded roots."""
    rho = MV.scalar(1, 0.5) + MV.word(1, "Z", 0.5)
    hamiltonian = MV.word(1, "Z") + MV.word(1, "X", 0.4)
    generators = [Generator("I", MV.scalar(1)),
                  Generator("I+0.2X", MV.scalar(1) + MV.word(1, "X", 0.2))]
    bank = MatrixElementBank(rho, hamiltonian, generators)
    moments = SecondMomentBank(bank)
    for ridge in (0.1, [0.0, 0.1]):
        with pytest.raises(ValueError, match="overlap_ridge"):
            FoldedSpectrum(moments, overlap_ridge=ridge)
        with pytest.raises(ValueError, match="overlap_ridge"):
            folded_spectrum_roots(moments, 0.3, overlap_ridge=ridge)
        with pytest.raises(ValueError, match="overlap_ridge"):
            refined_ritz_root(bank.solve(), 0, moments=moments, overlap_ridge=ridge)
    spectrum = FoldedSpectrum(moments, overlap_ridge=0.0)
    S, _ = bank.matrices(range(len(bank)))
    pencil = np.linalg.solve(S, spectrum.folded_matrix(0.3))
    dense = np.sort(np.linalg.eigvals(pencil).real)
    values = [root.folded_value for root in spectrum.roots(0.3).roots]
    assert values == pytest.approx(dense, abs=1e-12)
    assert values[0] == pytest.approx(0.6037802231, abs=1e-10)


def test_a_subset_of_generators_is_its_own_span(chain):
    full = folded_spectrum_roots(chain["moments"], -6.0)
    part = folded_spectrum_roots(chain["moments"], -6.0, indices=[0, 1, 2])
    assert part.indices == (0, 1, 2) and len(part.roots) == 3
    assert part.lowest.folded_value >= full.lowest.folded_value - 1e-12


def test_records_serialize(chain):
    spectrum = FoldedSpectrum(chain["moments"])
    root = spectrum.roots(-6.0).lowest
    row = json.loads(json.dumps(root.as_dict(), allow_nan=False))
    assert row["schema"] == "clifford_qc.folded_spectrum_root.v1"
    assert row["target_interval"][0] <= -6.0 <= row["target_interval"][1]
    run = json.loads(json.dumps(spectrum.minimize_variance(-6.0).as_dict(), allow_nan=False))
    assert run["schema"] == "clifford_qc.variance_minimization.v1"
    assert len(run["trace"]) == run["iterations"]
