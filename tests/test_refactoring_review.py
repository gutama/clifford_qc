"""Regressions for the PR #97 review: API, lifetime and accounting contracts."""
from collections.abc import ItemsView, ValuesView
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

from clifford_qc.backends import SectorStatevectorBackend
from clifford_qc.ir import PauliWord
from clifford_qc.measurement.functionals import PackedCoefficients
from clifford_qc.measurement.session import SharedMeasurement
from clifford_qc.multivector import MV
from clifford_qc.prepared import PreparedProblem, implementation_fingerprint, prepare_fcidump
from clifford_qc.subspace import MatrixElementBank, StreamingMatrixElementBank, pauli_orbit
from clifford_qc.subspace.packed import GlobalWordTable, SegmentedPackedRowStore, _Buffer


def bank_fixture(cls=StreamingMatrixElementBank, storage='packed', n=4):
    rho = MV(n, {0: 2.0 ** -n})
    h = MV(n, {0: .25, 1 << (2 * n - 1): 1.0})
    generators = pauli_orbit([PauliWord(n, code) for code in (0, 1, 2, 4)])
    options = dict(frontier_pairs=1) if cls is StreamingMatrixElementBank else {}
    return cls(rho, h, generators, storage=storage, **options)


@pytest.mark.parametrize('cls', [MatrixElementBank, StreamingMatrixElementBank])
@pytest.mark.parametrize('storage', ['object', 'packed'])
def test_entry_rejects_invalid_indices_without_mutating_cache(cls, storage):
    bank = bank_fixture(cls, storage)
    for invalid in (-1, len(bank), 1.5, '1'):
        for pair in ((invalid, 0), (0, invalid)):
            with pytest.raises(IndexError):
                bank.entry(*pair)
    assert bank._entries == {}
    assert bank.resources()['pairs_built'] == 0
    assert bank.entry(np.int64(1), 2) == tuple(v.conjugate() for v in bank.entry(2, 1))


@pytest.mark.parametrize('storage', ['object', 'packed'])
def test_incremental_resident_union_and_logical_ownership(storage):
    bank = bank_fixture(storage=storage)
    SharedMeasurement(bank)  # Structural compilation without retain_basis.
    for retained in ((), (0, 1), (0, 1, 2), (2,)):
        bank.retain_basis(retained)
        for j in range(len(bank)):
            bank.element_operator(0, j)
        bank.observable_operator(MV(bank.n, {3: 1}), 0, 0, label='extra')
        live = {code for op in bank._resident_operators() for code in op.terms}
        for order in (None, (0, 2), retained):
            r = bank.resources(order)
            assert r['resident_word_universe'] == len(live)
            assert r['retained_block_pairs'] + r['selection_pairs'] == r['pairs_built']
            assert (r['resident_retained_block_pairs'] + r['resident_selection_pairs']) * 2 == r['resident_sh_rows']
        words = {int(code) for key, pair in bank._supports.items()
                 if key[0] in retained and key[1] in retained for codes in pair for code in codes}
        assert bank.resources(retained)['word_universe'] == len(words)


@pytest.mark.parametrize('n', [4, 32])
def test_resource_snapshots_do_not_walk_support_occurrences(n):
    # The CSR word table is int64; the wide-code history path uses object rows.
    bank = bank_fixture(n=n, storage='object' if n > 31 else 'packed')
    bank.retain_basis((0, 1))
    bank.matrices()
    bank.retain_basis((0, 1))
    expected = bank.resources((0, 1))
    payload = sum(codes.nbytes for pair in bank._supports.values() for codes in pair)
    if n > 31:
        payload += sum(sys.getsizeof(code) for pair in bank._supports.values()
                       for codes in pair for code in codes)
    assert expected['support_history_payload_bytes'] == payload

    class NoWalk(np.ndarray):
        def __iter__(self):
            raise AssertionError('resource query walked coefficient history')

    bank._supports = {key: tuple(codes.view(NoWalk) for codes in pair)
                      for key, pair in bank._supports.items()}
    for _ in range(3):
        actual = bank.resources((0, 1))
        for name in ('resident_word_universe', 'word_universe', 'support_history_payload_bytes'):
            assert actual[name] == expected[name]


def test_empty_word_table_reserved_capacity_is_reported():
    bank = bank_fixture()
    assert len(bank._word_table) == 0
    assert bank.resources()['word_table_reserved_bytes'] == bank._word_table.nbytes() > 0


def test_packed_mapping_views_are_reiterable_sized_and_support_membership():
    mapping = PackedCoefficients({12: 1.5, 2: -3., 7: .25})
    items, values = mapping.items(), mapping.values()
    assert isinstance(items, ItemsView) and isinstance(values, ValuesView)
    assert len(items) == len(values) == 3
    assert list(items) == list(items) == [(12, 1.5), (2, -3.), (7, .25)]
    assert list(values) == list(values) == [1.5, -3., .25]
    assert (12, 1.5) in items and -3. in values
    assert items & {(2, -3.)} == {(2, -3.)}


def test_large_packed_word_codes_charge_boxes_once():
    codes = [2**80, 2**90]
    mapping = PackedCoefficients(dict.fromkeys(codes, 1.))
    arrays = (mapping._codes, mapping._values, mapping._order, mapping._sorted)
    assert mapping.nbytes == sum(a.nbytes for a in arrays) + sum(map(sys.getsizeof, codes))


@pytest.mark.parametrize('layout', ['interleaved', 'soa'])
def test_segment_allocates_exact_capacity_without_trimming(layout):
    store = SegmentedPackedRowStore(2, GlobalWordTable(), layout=layout)
    capacities = []
    original = _Buffer.__init__

    def record(self, dtype, capacity=1024):
        capacities.append(capacity)
        original(self, dtype, capacity)

    with patch.object(_Buffer, '__init__', record):
        for j, terms in enumerate(({}, {1: 1}, {1: 1, 2: 2, 3: 3})):
            capacities.clear()
            store.add((0, j), MV(2, terms))
            assert capacities and set(capacities) == {len(terms)}
            assert store.nbytes() == store.reserved_bytes()


def test_prepared_payload_is_validated_once_without_exposing_mutable_aliases(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'benchmarks/data/h4_sto3g_r0.9.FCIDUMP'
    prepared, path, _ = prepare_fcidump(source, tmp_path)
    loaded = PreparedProblem.load(path)
    with patch('clifford_qc.prepared.json.loads', side_effect=AssertionError('reparse')), patch(
            'clifford_qc.prepared._digest', side_effect=AssertionError('rehash')):
        a = loaded.model()
        a.metadata['nested'] = {'mutated': True}
        a.reference.ops.clear()
        specification = loaded.preparation
        specification['name'] = 'changed'
        b = loaded.model()
        assert b.reference.ops
        assert 'nested' not in b.metadata
        assert loaded.preparation == prepared.preparation
    data = json.loads(path.read_text())
    data['problem']['hamiltonian'][0][1] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='digest'):
        PreparedProblem.load(path)


def test_implementation_fingerprint_scans_once_per_process():
    implementation_fingerprint.cache_clear()
    try:
        first = implementation_fingerprint()
        with patch.object(Path, 'rglob', side_effect=AssertionError('source rescan')):
            assert implementation_fingerprint() == first
    finally:
        implementation_fingerprint.cache_clear()


@pytest.mark.parametrize('method', ['dense', 'lanczos', 'eigsh'])
def test_validation_reuses_one_sector_operator(method):
    backend = SectorStatevectorBackend(4, 2, 0.)
    h = MV(4, {3: 1., 12: 2., 48: 3., 192: 4.})
    with patch.object(backend, 'operator', wraps=backend.operator) as build:
        values, vectors, operator = backend.ground_state(h, method=method, return_operator=True)
        assert build.call_count == 1
        assert np.linalg.norm(operator.matvec(vectors[:, 0]) - values[0] * vectors[:, 0]) < 1e-10
    assert len(backend.ground_state(h, method='dense')) == 2


@pytest.mark.parametrize('highest', [9.6, 12.])
def test_overlapping_measured_thresholds_remain_indeterminate(highest):
    from benchmarks.run_packed_bank_storage import REFERENCE, load_config, _verdict
    from benchmarks.check_packed_bank_storage import verdict_problems
    record = json.loads(REFERENCE.read_text())
    model = record['reduction_model']
    model['lowest_reuse_clearing_threshold'] = 9.6
    model['highest_reuse_failing_threshold'] = highest
    record['go_no_go'] = _verdict(load_config(), model, record['committed_bank_reuse'])
    assert record['go_no_go']['outcome'] == 'indeterminate_threshold_not_separated'
    assert verdict_problems(record) == []
    record['go_no_go']['outcome'] = 'reached_on_measured_banks_historical_banks_ungraded'
    assert verdict_problems(record)


def test_reduction_statement_does_not_deny_a_measured_loss():
    from benchmarks.run_packed_bank_storage import REFERENCE, _reduction_model
    rows = deepcopy(json.loads(REFERENCE.read_text())['banks'])
    rows[0]['layouts']['interleaved'].update(reduction=.9, clears_threshold=False)
    model = _reduction_model(rows)
    assert rows[0]['bank'] in model['banks_where_packing_costs_more']
    assert 'no net losses' not in model['statement']
