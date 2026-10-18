"""Ritz cluster hulls against dense oracles on small Hubbard and Ising spans.

Each hull is held to the full spectrum: it must contain at least as many
eigenvalues as it has roots. The block residual is held to an explicit
statevector construction, ``||HQ - QM||_2`` on an orthonormalized basis,
which uses no multivector product and no second-moment row.
"""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest

from benchmarks.run_mapping_axis import _raw_pool
from clifford_qc import PauliSum, PauliWord, ket_density
from clifford_qc.backends import ExactMVBackend
from clifford_qc.dense_reference import to_matrix
from clifford_qc.models import tfim
from clifford_qc.models.lattice import hubbard
from clifford_qc.subspace import (
    RITZ_CLUSTERS_SCHEMA,
    MatrixElementBank,
    SecondMomentBank,
    pauli_orbit,
    ritz_clusters,
    solve_subspace,
)
from clifford_qc.subspace.generators import identity_generator
from clifford_qc.subspace.reference import dense_basis
from clifford_qc.subspace.second_moment import RESOLUTION


def _case(rho, hamiltonian, generators):
    bank = MatrixElementBank(rho, hamiltonian, generators)
    dense_h = to_matrix(hamiltonian)
    return {"bank": bank, "moments": SecondMomentBank(bank), "result": bank.solve(),
            "basis": dense_basis(rho, generators), "h": dense_h,
            "spectrum": np.linalg.eigvalsh(dense_h), "rho": rho,
            "hamiltonian": hamiltonian, "generators": generators}


def _hubbard(size):
    model = hubbard(3)
    generators = [identity_generator(model.n), *_raw_pool(model)[:size - 1]]
    return _case(ExactMVBackend().state(model.reference, ()),
                 model.hamiltonian.to_mv(), generators)


def _ising(drop_top=False):
    """Periodic 4-site TFIM from |0000> with X strings: the full space, or all
    but ``XXXX``, which leaves a proper span with near-degenerate pairs."""
    n = 4
    labels = ["".join("X" if q in subset else "I" for q in range(n))
              for weight in range(1, n + 1)
              for subset in itertools.combinations(range(n), weight)]
    if drop_top:
        labels = labels[:-1]
    generators = [identity_generator(n),
                  *pauli_orbit([PauliWord.from_label(label) for label in labels])]
    model = tfim(n, J=1.0, h=0.3, periodic=True)
    return _case(ket_density(n, "0" * n), model.hamiltonian.to_mv(), generators)


CASES = {"hubbard5": lambda: _hubbard(5), "hubbard7": lambda: _hubbard(7),
         "ising_full": _ising, "ising_proper": lambda: _ising(drop_top=True)}


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request):
    return CASES[request.param]()


@pytest.fixture(scope="module")
def ising_full():
    return _ising()


@pytest.fixture(scope="module")
def ising_proper():
    return _ising(drop_top=True)


def _count(spectrum, interval, slack=1e-12):
    lo, hi = interval
    return int(np.sum((spectrum >= lo - slack) & (spectrum <= hi + slack)))


def _dense_block_residual(case, roots):
    Q, _ = np.linalg.qr(case["basis"] @ case["result"].coefficients[:, list(roots)])
    M = Q.conj().T @ case["h"] @ Q
    return float(np.linalg.norm(case["h"] @ Q - Q @ M, 2))


# ------------------------------------------------------------------ the theorem

def test_every_hull_holds_at_least_as_many_eigenvalues_as_roots(case):
    certificate = ritz_clusters(case["result"], moments=case["moments"])
    for cluster in certificate.clusters:
        assert _count(case["spectrum"], cluster.interval()) >= cluster.count


def test_the_radius_is_the_dense_block_residual(case):
    certificate = ritz_clusters(case["result"], moments=case["moments"])
    for cluster in certificate.clusters:
        dense = _dense_block_residual(case, cluster.roots)
        # Unresolved residuals are rounding on both sides; only the floor binds.
        tolerance = 1e-9 * max(1.0, dense) if cluster.resolved else cluster.radius
        assert abs(cluster.residual_norm - dense) <= tolerance
        assert cluster.radius >= dense - 1e-12


def test_a_single_root_carries_its_weinstein_half_width(case):
    result, moments = case["result"], case["moments"]
    for k in range(len(result.energies)):
        single = ritz_clusters(result, [k], moments=moments).clusters[0]
        residual = moments.residual(result, k)
        weinstein = math.sqrt(max(residual.variance, 0.0)
                              + RESOLUTION * residual.cancellation_scale)
        assert single.ritz_values == pytest.approx((result.energies[k],), abs=1e-10)
        # Two roundings of one variance: they may differ by the floor, no more.
        floor = RESOLUTION * residual.cancellation_scale
        assert abs(single.radius ** 2 - weinstein ** 2) <= floor
        assert single.resolution == pytest.approx(floor, rel=1e-9)


def test_hulls_are_disjoint_ascending_and_partition_the_roots(case):
    certificate = ritz_clusters(case["result"], moments=case["moments"])
    intervals = [cluster.interval() for cluster in certificate.clusters]
    assert all(left[1] < right[0] for left, right in zip(intervals, intervals[1:]))
    covered = [root for cluster in certificate.clusters for root in cluster.roots]
    assert covered == list(certificate.roots) == list(range(len(case["result"].energies)))
    # Ascending Ritz values cluster into contiguous runs of roots.
    for cluster in certificate.clusters:
        assert cluster.roots == tuple(range(cluster.roots[0], cluster.roots[-1] + 1))


def test_merging_never_shrinks_a_member_interval(case):
    result, moments = case["result"], case["moments"]
    for cluster in ritz_clusters(result, moments=moments).clusters:
        lo, hi = cluster.interval()
        for root in cluster.roots:
            alone = ritz_clusters(result, [root], moments=moments).clusters[0]
            assert cluster.radius >= alone.radius * (1.0 - 1e-12)
            assert lo <= alone.interval()[0] + 1e-12
            assert hi >= alone.interval()[1] - 1e-12


# ------------------------------------------------------------------ what it adds

def test_an_exact_degenerate_level_is_counted_with_its_multiplicity(ising_full):
    """Per-root intervals say 'an eigenvalue' four times; the cluster says four."""
    certificate = ritz_clusters(ising_full["result"], moments=ising_full["moments"])
    multiplicities = {}
    for cluster in certificate.clusters:
        lo, hi = cluster.interval()
        assert hi - lo < 1e-4
        assert _count(ising_full["spectrum"], (lo, hi)) == cluster.count
        multiplicities[round(float(np.mean(cluster.ritz_values)), 6)] = cluster.count
    assert multiplicities[0.0] == 4
    assert multiplicities[0.6] == multiplicities[-0.6] == 2


def test_overlapping_weinstein_intervals_are_counted_not_conflated(ising_proper):
    result, moments = ising_proper["result"], ising_proper["moments"]
    assert ising_proper["basis"].shape[1] < ising_proper["h"].shape[0]
    top = len(result.energies) - 1
    a, b = (ritz_clusters(result, [k], moments=moments).clusters[0].interval()
            for k in (top - 1, top))
    assert a[1] >= b[0]  # each holds an eigenvalue, possibly the same one
    cluster = ritz_clusters(result, moments=moments).clusters[-1]
    assert cluster.roots == (top - 1, top) and cluster.resolved
    assert _count(ising_proper["spectrum"], cluster.interval()) >= 2


def test_a_root_subset_is_clustered_on_its_own(case):
    result, moments = case["result"], case["moments"]
    subset = [len(result.energies) - 1, 0]
    certificate = ritz_clusters(result, subset, moments=moments)
    assert certificate.roots == (0, len(result.energies) - 1)
    for cluster in certificate.clusters:
        assert _count(case["spectrum"], cluster.interval()) >= cluster.count


def test_it_builds_its_own_moments_when_none_are_given():
    case = _hubbard(5)
    built = ritz_clusters(case["result"])
    shared = ritz_clusters(case["result"], moments=case["moments"])
    assert built.as_dict() == shared.as_dict()


# ------------------------------------------------------------------ contracts

def test_inputs_are_validated():
    case = _hubbard(5)
    result, moments = case["result"], case["moments"]
    bankless = solve_subspace(case["rho"], case["hamiltonian"], case["generators"])
    with pytest.raises(ValueError, match="MatrixElementBank"):
        ritz_clusters(bankless)
    for bad in ([], [0, 0]):
        with pytest.raises(ValueError):
            ritz_clusters(result, bad, moments=moments)
    with pytest.raises(IndexError):
        ritz_clusters(result, [len(result.energies)], moments=moments)
    with pytest.raises(TypeError):
        ritz_clusters(result, [0.5], moments=moments)
    with pytest.raises(ValueError, match="storage_policy"):
        ritz_clusters(result, moments=moments, storage_policy=object())
    with pytest.raises(TypeError):
        ritz_clusters(result, moments=object())
    other = _hubbard(5)
    with pytest.raises(ValueError, match="different bank"):
        ritz_clusters(result, moments=other["moments"])


def test_a_metric_changing_ridge_is_refused():
    case = _hubbard(5)
    ridged = case["bank"].solve(overlap_ridge=1e-3)
    with pytest.raises(ValueError, match="overlap_ridge"):
        ritz_clusters(ridged, moments=case["moments"])


def test_a_mixed_reference_is_refused_because_it_repeats_eigenvalues():
    """Maximally mixed qubit, H = Z, generators I, X, Y, Z: the four Ritz values
    are -1, -1, 1, 1 with zero residual, but Z has each eigenvalue once, so a
    hull of count two at -1 would hold one eigenvalue."""
    rho = ket_density(1, "0") * 0.5 + ket_density(1, "1") * 0.5
    hamiltonian = PauliSum.from_labels({"Z": 1.0}).to_mv()
    generators = [identity_generator(1),
                  *pauli_orbit([PauliWord.from_label(label) for label in "XYZ"])]
    bank = MatrixElementBank(rho, hamiltonian, generators)
    result = bank.solve()
    assert np.allclose(result.energies, [-1.0, -1.0, 1.0, 1.0], atol=1e-12)
    assert np.allclose(np.linalg.eigvalsh(to_matrix(hamiltonian)), [-1.0, 1.0])
    with pytest.raises(ValueError, match="pure reference"):
        ritz_clusters(result)


def test_inconsistent_blocks_are_refused(monkeypatch):
    """A second moment below H^2 on the span cannot come from one Hermitian H."""
    case = _hubbard(5)
    moments, result = case["moments"], case["result"]
    S, _ = case["bank"].matrices(result.indices)
    K = moments.matrix(result.indices)
    monkeypatch.setattr(moments, "matrix", lambda indices=None: K - 10.0 * S)
    with pytest.raises(ValueError, match="negative block residual"):
        ritz_clusters(result, moments=moments)


def test_records_serialize(case):
    record = ritz_clusters(case["result"], moments=case["moments"]).as_dict()
    restored = json.loads(json.dumps(record))
    assert restored["schema"] == RITZ_CLUSTERS_SCHEMA
    assert restored["evidence"] == "exact"
    assert sum(cluster["count"] for cluster in restored["clusters"]) == len(restored["roots"])
