"""Storage approval remains attached to the exact block used by H-squared users."""

from __future__ import annotations

import json

import numpy as np
import pytest

from clifford_qc.dense_reference import to_matrix
from clifford_qc.multivector import MV
from clifford_qc.pauli import X, Y, Z
from clifford_qc.states import ket_density
from clifford_qc.subspace import (
    ConvergenceConfig,
    MatrixElementBank,
    SecondMomentBank,
    SecondMomentStorageError,
    SecondMomentStoragePolicy,
    StreamingMatrixElementBank,
    convergence_report,
    dense_residual_norm,
    identity_generator,
    refined_ritz_root,
    run_acase,
    second_moment_storage_preflight,
)
from clifford_qc.subspace.reference import dense_basis


def _bank(storage="object", streaming=False):
    cls = StreamingMatrixElementBank if streaming else MatrixElementBank
    options = {"frontier_pairs": 8} if streaming else {}
    bank = cls(
        ket_density(2, "00"), Z(2, 0) + X(2, 0) + 0.2 * X(2, 1),
        [identity_generator(2), ("X0", X(2, 0)), ("X1", X(2, 1))],
        storage=storage, **options,
    )
    # This is first-moment setup; the gate must need no additional product.
    bank.matrices()
    return bank


def _observable_pairs(bank):
    return {
        pair
        for record in bank._observables.values()
        if record["label"] == "H^2"
        for pair in record["operators"]
    }


@pytest.mark.parametrize("storage, streaming", [
    ("object", False), ("packed", False), ("packed", True),
])
@pytest.mark.parametrize("entry", ["constructor", "report", "refinement", "acase"])
def test_all_automatic_entries_refuse_before_products_or_registration(
        monkeypatch, storage, streaming, entry):
    bank = _bank(storage, streaming)
    result = bank.solve((0,))
    policy = SecondMomentStoragePolicy(max_coefficient_occurrences=1)
    config = ConvergenceConfig(0.1, storage_policy=policy)

    def forbidden_product(*args, **kwargs):
        raise AssertionError("the storage refusal must precede H squared and new rows")

    monkeypatch.setattr(MV, "__mul__", forbidden_product)
    with pytest.raises(SecondMomentStorageError) as refused:
        if entry == "constructor":
            SecondMomentBank(bank, result.indices, storage_policy=policy)
        elif entry == "report":
            convergence_report(result, config)
        elif entry == "refinement":
            refined_ritz_root(result, storage_policy=policy)
        else:
            run_acase(
                bank.reference, bank.hamiltonian,
                [bank.generator(1), bank.generator(2)], bank=bank,
                max_size=0, convergence=config,
            )
    assert not refused.value.report.allowed
    assert bank._observables == {}


def test_a_scoped_block_preserves_its_order_and_dense_second_moment():
    bank = _bank()
    indices = (1, 0)
    result = bank.solve(indices)
    moments = SecondMomentBank(bank, indices)
    assert moments.indices == indices
    assert moments.storage_preflight.indices == indices
    assert moments.storage_preflight.allowed
    assert moments.storage_preflight.matches_inputs(bank)
    json.dumps(moments.storage_preflight.as_dict(), allow_nan=False)

    generators = [bank.generator(i) for i in indices]
    basis = dense_basis(bank.reference, generators)
    h = to_matrix(bank.hamiltonian)
    np.testing.assert_allclose(moments.matrix(), basis.conj().T @ h @ h @ basis,
                               rtol=0, atol=1e-12)
    residual = moments.residual(result)
    assert residual.residual_norm == pytest.approx(0.2)
    assert residual.residual_norm == pytest.approx(dense_residual_norm(
        bank.reference, bank.hamiltonian, generators, result))
    # Repeated use must not charge cached approved rows a second time.
    assert moments.resources() == moments.resources()
    assert _observable_pairs(bank) == {(0, 0), (0, 1), (1, 1)}


@pytest.mark.parametrize("method", ["row", "matrix", "resources", "word_set"])
@pytest.mark.parametrize("outside", [2, 3])
def test_access_cannot_widen_the_approved_block_even_after_growth(method, outside):
    bank = _bank()
    moments = SecondMomentBank(bank, (1, 0))
    assert bank.add(("Y0", Y(2, 0))) == 3
    assert moments.storage_preflight.matches_inputs(bank)
    with pytest.raises(ValueError, match="storage-preflight block"):
        if method == "row":
            moments.row(0, outside)
        else:
            getattr(moments, method)((0, outside))
    assert bank._observables == {}


def test_omitted_indices_are_frozen_at_construction():
    bank = _bank()
    moments = SecondMomentBank(bank)
    assert moments.indices == (0, 1, 2)
    bank.add(("Y0", Y(2, 0)))
    assert moments.matrix().shape == (3, 3)
    assert moments.resources()["basis_size"] == 3
    assert all(i < 3 and j < 3 for i, j in _observable_pairs(bank))


@pytest.mark.parametrize("input_name", ["hamiltonian", "reference", "generator"])
def test_mutated_inputs_invalidate_approval_before_any_new_product(monkeypatch, input_name):
    bank = _bank()
    moments = SecondMomentBank(bank, (0, 1))
    operator = (bank.generator(1).mv if input_name == "generator"
                else getattr(bank, input_name))
    code = next(iter(operator.terms))
    operator.terms[code] += 0.125
    assert not moments.storage_preflight.matches_inputs(bank)

    def forbidden_product(*args, **kwargs):
        raise AssertionError("changed inputs must be rejected before row construction")

    monkeypatch.setattr(MV, "__mul__", forbidden_product)
    with pytest.raises(ValueError, match="inputs changed"):
        moments.row(0, 1)
    assert bank._observables == {}


@pytest.mark.parametrize("entry", ["report", "refinement"])
def test_automatic_diagnostics_price_only_the_retained_result_block(entry):
    bank = _bank()
    result = bank.solve((1, 0))
    bound = second_moment_storage_preflight(bank, result.indices).coefficient_occurrence_bound
    policy = SecondMomentStoragePolicy(max_coefficient_occurrences=bound)
    assert not second_moment_storage_preflight(bank, policy=policy).allowed

    if entry == "report":
        report = convergence_report(result, ConvergenceConfig(0.1, storage_policy=policy))
        assert report.residual.residual_norm == pytest.approx(0.2)
    else:
        refined = refined_ritz_root(result, storage_policy=policy)
        assert refined.residual_norm == pytest.approx(0.2)
    assert _observable_pairs(bank) == {(0, 0), (0, 1), (1, 1)}


def test_acase_uses_the_final_retained_block_for_its_storage_policy():
    bank = _bank()
    bound = second_moment_storage_preflight(bank, (0, 1)).coefficient_occurrence_bound
    policy = SecondMomentStoragePolicy(max_coefficient_occurrences=bound)
    assert not second_moment_storage_preflight(bank, policy=policy).allowed
    actual = run_acase(
        bank.reference, bank.hamiltonian, [bank.generator(1), bank.generator(2)],
        bank=bank, max_size=1, convergence=ConvergenceConfig(0.1, storage_policy=policy),
    )
    assert actual.indices == (0, 1)
    report, = actual.convergence_reports
    assert report.residual.residual_norm == pytest.approx(0.2)
    assert _observable_pairs(bank) == {(0, 0), (0, 1), (1, 1)}


@pytest.mark.parametrize("entry", ["report", "refinement"])
def test_supplied_moments_cannot_approve_a_wider_result(entry):
    bank = _bank()
    result = bank.solve()
    moments = SecondMomentBank(bank, (0, 1))
    with pytest.raises(ValueError, match="storage-preflight block"):
        if entry == "report":
            convergence_report(result, ConvergenceConfig(0.1), moments=moments)
        else:
            refined_ritz_root(result, moments=moments)
    assert bank._observables == {}


def test_mutated_hamiltonian_square_is_rejected_before_new_rows(monkeypatch):
    bank = _bank()
    moments = SecondMomentBank(bank)
    square = moments.hamiltonian_square
    square.terms[0] += 1

    def forbidden_product(*args, **kwargs):
        raise AssertionError("a changed square must not materialize a row")

    monkeypatch.setattr(MV, "__mul__", forbidden_product)
    with pytest.raises(ValueError, match="Hamiltonian square changed"):
        moments.matrix()
    assert bank._observables == {}


def test_nonhermitian_square_cannot_expand_the_triangular_storage_bound():
    # The first-moment bank's absolute Hermiticity tolerance accepts H, but
    # its imaginary component grows above that tolerance when H is squared.
    bank = MatrixElementBank(ket_density(1, "0"), (1e5 + 1e-10j) * Z(1, 0),
                             [identity_generator(1), ("X", X(1, 0))])
    with pytest.raises(ValueError, match="H squared must be Hermitian"):
        SecondMomentBank(bank)
    assert bank._observables == {}


def test_reversed_resource_reporting_holds_only_one_adjoint_temporary(monkeypatch):
    bank = _bank()
    moments = SecondMomentBank(bank, (2, 1, 0))
    live = peak = 0
    original_dagger = MV.dagger

    class TrackedAdjoint(MV):
        def __init__(self, operator):
            nonlocal live, peak
            super().__init__(operator.n, operator.terms)
            live += 1
            peak = max(peak, live)

        def __del__(self):
            nonlocal live
            live -= 1

    def tracked_dagger(operator):
        return TrackedAdjoint(original_dagger(operator))

    monkeypatch.setattr(MV, "dagger", tracked_dagger)
    assert moments.resources()["block_pairs"] == 6
    assert peak == 1
    assert live == 0
