"""Product-free, finite and scoped second-moment storage admission."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from clifford_qc.multivector import MV
from clifford_qc.subspace.packed import PackedRowStore, SegmentedPackedRowStore
from clifford_qc.subspace.projection import MatrixElementBank
from clifford_qc.subspace.second_moment_storage import (
    DEFAULT_MAX_COEFFICIENT_OCCURRENCES,
    SecondMomentStorageError,
    SecondMomentStoragePolicy,
    refresh_second_moment_storage,
    require_second_moment_storage,
    second_moment_storage_preflight,
)
from clifford_qc.subspace.streaming import StreamingMatrixElementBank


def _bank(*, storage="object", streaming=False):
    kind = StreamingMatrixElementBank if streaming else MatrixElementBank
    return kind(MV(1, {0: .5, 3: .5}), MV(1, {3: 1}),
                [("I", MV(1, {0: 1})), ("X", MV(1, {1: 1}))], storage=storage)


def _forbidden(*args, **kwargs):
    raise AssertionError("preflight must not multiply or materialize a row")


def _snapshot(bank):
    return (dict(bank._overlap_ops), dict(bank._element_ops),
            {key: (dict(value["operators"]), dict(value["acted"]),
                   frozenset(value["universe"])) for key, value in bank._observables.items()},
            tuple((kind, tuple(store.keys()), store.total_coefficients())
                  for kind, store in bank._packed.items()),
            frozenset(bank._universe), dict(bank._entries), bank._pairs_built,
            bank._cache_hits, bank._seconds, bank._peak_resident_rows)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1., float("inf"), "100", None])
def test_budget_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        SecondMomentStoragePolicy(value)


def test_policy_is_finite_immutable_and_accepts_integer_scalars():
    policy = SecondMomentStoragePolicy()
    assert policy.max_coefficient_occurrences == DEFAULT_MAX_COEFFICIENT_OCCURRENCES == 96_107_619
    assert SecondMomentStoragePolicy(np.int64(12)).max_coefficient_occurrences == 12
    with pytest.raises(FrozenInstanceError):
        policy.max_coefficient_occurrences = 1


def test_boundary_is_inclusive_and_refusal_carries_the_report():
    bank = _bank()
    report = second_moment_storage_preflight(bank)
    # Three upper-triangle rows each for S/H/K, two K actions, H²,
    # nine already held input coefficients and one output-row temporary.
    assert report.coefficient_occurrence_bound == 22
    assert report.overlap_coefficient_occurrences_bound == 3
    assert report.hamiltonian_coefficient_occurrences_bound == 3
    assert report.second_moment_coefficient_occurrences_bound == 3
    assert report.second_moment_action_coefficient_occurrences_bound == 2
    exact = SecondMomentStoragePolicy(22)
    assert require_second_moment_storage(bank, policy=exact).allowed
    small = SecondMomentStoragePolicy(21)
    assert not second_moment_storage_preflight(bank, policy=small).allowed
    with pytest.raises(SecondMomentStorageError, match="exceeds.*21") as caught:
        require_second_moment_storage(bank, policy=small)
    assert caught.value.report.coefficient_occurrence_bound == 22
    assert caught.value.report.max_coefficient_occurrences == 21


@pytest.mark.parametrize("storage", ["object", "packed"])
@pytest.mark.parametrize("streaming", [False, True])
def test_preflight_never_builds_or_materializes_any_rows(monkeypatch, storage, streaming):
    bank = _bank(storage=storage, streaming=streaming)
    bank.matrices()
    bank.project_observable(MV(1, {1: 1}), label="existing")
    before = _snapshot(bank)
    monkeypatch.setattr(MV, "__mul__", _forbidden)
    monkeypatch.setattr(MatrixElementBank, "_build_pair", _forbidden)
    monkeypatch.setattr(MatrixElementBank, "_row", _forbidden)
    monkeypatch.setattr(MatrixElementBank, "_resident_operators", _forbidden)
    monkeypatch.setattr(PackedRowStore, "materialize", _forbidden)
    monkeypatch.setattr(SegmentedPackedRowStore, "materialize", _forbidden)
    report = second_moment_storage_preflight(bank, [0])
    assert report.indices == (0,)
    # Two existing non-target pairs with S and H coefficients apiece,
    # plus three existing observable rows and two observable actions.
    assert report.existing_non_target_coefficient_occurrences == 9
    with pytest.raises(SecondMomentStorageError):
        require_second_moment_storage(bank, [0], policy=SecondMomentStoragePolicy(1))
    assert _snapshot(bank) == before


def test_affine_xor_rank_tightens_count_without_enumerating_pauli_space(monkeypatch):
    # All four words on one active qubit in a much larger algebra. Their
    # affine span has four elements, though |H|² is sixteen and 4**n enormous.
    bank = MatrixElementBank(MV(20, {0: 1 / 2**20}),
                             MV(20, {0: 1, 1: 2, 2: 3, 3: 4}),
                             [("I", MV(20, {0: 1}))])
    monkeypatch.setattr(MV, "__mul__", _forbidden)
    report = second_moment_storage_preflight(bank)
    assert report.hamiltonian_square_terms_bound == 4
    assert report.second_moment_coefficient_occurrences_bound == 4


def test_individual_generator_supports_price_the_requested_block():
    bank = MatrixElementBank(MV(2, {0: .25}), MV(2, {0: 1, 3: .5}),
                             [("I", MV(2, {0: 1})),
                              ("wide", MV(2, {0: 1, 1: 2, 4: 3, 5: 4}))])
    narrow = second_moment_storage_preflight(bank, [0])
    wide = second_moment_storage_preflight(bank, [1])
    full = second_moment_storage_preflight(bank)
    assert narrow.generator_supports == (1,)
    assert wide.generator_supports == (4,)
    assert narrow.coefficient_occurrence_bound < wide.coefficient_occurrence_bound
    assert wide.coefficient_occurrence_bound < full.coefficient_occurrence_bound
    assert wide.existing_input_coefficient_occurrences == full.existing_input_coefficient_occurrences


@pytest.mark.parametrize("storage", ["object", "packed"])
def test_bounds_cover_actual_materialized_payload(storage):
    bank = MatrixElementBank(MV(2, {0: .25}), MV(2, {0: 1, 1: .5, 3: .25}),
                             [("I", MV(2, {0: 1})),
                              ("wide", MV(2, {0: 1, 1: 2, 4: 3, 5: 4}))],
                             storage=storage)
    report = second_moment_storage_preflight(bank)
    square = bank.hamiltonian * bank.hamiltonian
    bank.matrices()
    bank.project_observable(square)
    record = next(iter(bank._observables.values()))
    assert square.nnz() <= report.hamiltonian_square_terms_bound
    assert sum(row.nnz() for row in record["operators"].values()) <= report.second_moment_coefficient_occurrences_bound
    assert sum(row.nnz() for row in record["acted"].values()) <= report.second_moment_action_coefficient_occurrences_bound
    actual = (bank._resident_coefficients() + sum(row.nnz() for row in record["acted"].values())
              + square.nnz() + report.existing_input_coefficient_occurrences)
    assert actual <= report.coefficient_occurrence_bound


def test_fingerprints_bind_coefficients_reference_and_scoped_generators():
    bank = _bank()
    report = second_moment_storage_preflight(bank, [0])
    assert report.matches_inputs(bank)
    original = bank.hamiltonian.terms[3]
    bank.hamiltonian.terms[3] = 2 * original
    assert not report.matches_inputs(bank)
    assert second_moment_storage_preflight(bank, [0]).input_fingerprint != report.input_fingerprint
    bank.hamiltonian.terms[3] = original
    assert report.matches_inputs(bank)
    bank.reference.terms[3] = .4
    assert not report.matches_inputs(bank)
    bank.reference.terms[3] = .5
    bank.generator(0).mv.terms[0] = 2
    assert not report.matches_inputs(bank)


def test_new_generators_require_a_new_scope_and_do_not_change_original_report():
    bank = _bank()
    report = second_moment_storage_preflight(bank, [0])
    bank.add(("Y", MV(1, {2: 1})))
    assert report.indices == (0,)
    assert report.matches_inputs(bank)
    expanded = second_moment_storage_preflight(bank)
    assert expanded.indices == (0, 1, 2)
    assert expanded.coefficient_occurrence_bound > report.coefficient_occurrence_bound
    assert expanded.input_fingerprint != report.input_fingerprint
    with pytest.raises(FrozenInstanceError):
        report.indices = (0, 1, 2)


def test_empty_block_and_bad_indices_do_not_trigger_products(monkeypatch):
    bank = _bank()
    monkeypatch.setattr(MV, "__mul__", _forbidden)
    report = second_moment_storage_preflight(bank, [])
    assert report.indices == ()
    assert report.block_pairs == 0
    assert report.second_moment_coefficient_occurrences_bound == 0
    assert report.second_moment_action_coefficient_occurrences_bound == 0
    for indices, error in [([2], IndexError), ([-1], IndexError), ([0, 0], ValueError)]:
        with pytest.raises(error):
            second_moment_storage_preflight(bank, indices)


def test_report_explicitly_names_the_coefficient_scope_and_object_k_backend():
    report = second_moment_storage_preflight(_bank(storage="packed"))
    details = report.as_dict()
    assert details["metric"] == "sparse_operator_coefficient_occurrences"
    assert details["bounds_peak_rss"] is False
    assert details["second_moment_storage_backend"] == "object"
    assert details["indices"] == [0, 1]
    assert "not peak RSS" in details["reason"]


@pytest.mark.parametrize("name", ["h4", "h4_converged", "beh2", "h2o_cas8e6o", "hubbard_2x2"])
def test_five_historical_blocks_fit_the_same_product_free_analytic_budget(monkeypatch, name):
    from benchmarks.check_phase15_preregistration import bank_inputs
    from clifford_qc.backends import ExactMVBackend

    model, selected = bank_inputs(name)
    bank = MatrixElementBank(ExactMVBackend().state(model.reference, ()),
                             model.hamiltonian, selected)
    before = _snapshot(bank)
    monkeypatch.setattr(MV, "__mul__", _forbidden)
    report = require_second_moment_storage(bank)
    assert report.allowed
    assert report.indices == tuple(range(len(bank)))
    assert _snapshot(bank) == before


def test_default_analytic_budget_refuses_a_larger_unlicensed_block_before_products(monkeypatch):
    n = 24
    hamiltonian = MV(n, {1 << bit: 1 for bit in range(20)})
    generators = [(f"A{j}", MV(n, {1 << (20 + bit): 1 + j * (bit + 1) / 100
                                  for bit in range(20)})) for j in range(40)]
    bank = MatrixElementBank(MV(n, {0: 1 / 2**n}), hamiltonian, generators)
    before = _snapshot(bank)
    monkeypatch.setattr(MV, "__mul__", _forbidden)
    with pytest.raises(SecondMomentStorageError) as caught:
        require_second_moment_storage(bank)
    assert caught.value.report.coefficient_occurrence_bound > DEFAULT_MAX_COEFFICIENT_OCCURRENCES
    assert not caught.value.report.allowed
    assert _snapshot(bank) == before


def test_bad_bank_and_policy_types_are_rejected():
    with pytest.raises(TypeError, match="MatrixElementBank"):
        second_moment_storage_preflight(object())
    with pytest.raises(TypeError, match="SecondMomentStoragePolicy"):
        second_moment_storage_preflight(_bank(), policy=100)


def test_refresh_covers_only_its_exact_square_and_approved_scope(monkeypatch):
    bank = _bank()
    policy = SecondMomentStoragePolicy(15)
    first = require_second_moment_storage(bank, [0], policy=policy)
    second = require_second_moment_storage(bank, [1], policy=policy)
    square = bank.hamiltonian * bank.hamiltonian
    bank.project_observable(square, [0], label="H^2")
    before = _snapshot(bank)
    monkeypatch.setattr(MV, "__mul__", _forbidden)
    # Its own row and action are already budgeted, exactly at the boundary.
    assert refresh_second_moment_storage(bank, first, square).coefficient_occurrence_bound == 15
    # Those same caches belong to a different block for the second request.
    with pytest.raises(SecondMomentStorageError) as caught:
        refresh_second_moment_storage(bank, second, square)
    assert caught.value.report.existing_non_target_coefficient_occurrences == 2
    assert caught.value.report.coefficient_occurrence_bound == 17
    # Public preflight never assumes an arbitrary existing observable is covered.
    assert not second_moment_storage_preflight(bank, [0], policy=policy).allowed
    assert _snapshot(bank) == before


def test_refresh_counts_new_other_observables_and_non_target_sh_rows(monkeypatch):
    bank = _bank(storage="packed")
    first = require_second_moment_storage(bank, [0], policy=SecondMomentStoragePolicy(15))
    square = bank.hamiltonian * bank.hamiltonian
    bank.matrices([1])
    bank.project_observable(MV(1, {1: 1}), [0], label="H^2")
    monkeypatch.setattr(MV, "__mul__", _forbidden)
    monkeypatch.setattr(PackedRowStore, "materialize", _forbidden)
    with pytest.raises(SecondMomentStorageError) as caught:
        refresh_second_moment_storage(bank, first, square)
    assert caught.value.report.existing_non_target_coefficient_occurrences == 4


def test_refresh_rejects_input_changes_before_products(monkeypatch):
    bank = _bank()
    report = require_second_moment_storage(bank, [0])
    square = bank.hamiltonian * bank.hamiltonian
    bank.hamiltonian.terms[3] = 2
    monkeypatch.setattr(MV, "__mul__", _forbidden)
    with pytest.raises(ValueError, match="inputs changed"):
        refresh_second_moment_storage(bank, report, square)
