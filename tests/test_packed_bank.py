"""Contracts for the Phase 2M-B packed CSR/SoA matrix-element store.

2M-B swaps the representation under an unchanged public surface, so its failure
modes are all silent. A packed row that sorted its words would still return
plausible matrix elements -- measured across the frozen banks, sorting moves
4--17% of ``S``/``H`` entries by up to ``3.1e-13`` -- and every committed record
would drift without anything raising. A store that lost its emission order would
break the same promise more quietly still, because a materialized ``MV`` looks
correct until something pairs it. And a byte comparison taken against a
different population, or one that charges the packed side for an allocation the
object side also pays, would report a reduction that is an accounting artifact.

These tests pin emission order, bitwise agreement between the two backends on
real banks, the resource ledger's backend-neutrality, and the per-coefficient
packing the go/no-go is stated against.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from clifford_qc.backends import ExactMVBackend
from clifford_qc.models import fcidump_model
from clifford_qc.multivector import MV
from clifford_qc.subspace.adaptive import run_acase
from clifford_qc.subspace.elements import MatrixElementBank
from clifford_qc.subspace.fermionic_generators import (
    determinant_excitations,
    occupied_spin_orbitals,
)
from clifford_qc.subspace.generators import identity_generator
from clifford_qc.subspace.packed import (
    LAYOUTS,
    PACKED_BYTES_PER_COEFFICIENT,
    GlobalWordTable,
    PackedRowStore,
)
from clifford_qc.subspace.projection import STORAGE_BACKENDS

from benchmarks.run_bank_storage_ledger import ROOT

H4 = ROOT / "benchmarks" / "data" / "h4_sto3g_r0.9.FCIDUMP"
BEH2 = ROOT / "benchmarks" / "data" / "beh2_sto3g_r1.3264.FCIDUMP"


def _bank(path, *, size=6, **kwargs) -> MatrixElementBank:
    model = fcidump_model(path, name=path.stem)
    pool = determinant_excitations(model.n, occupied_spin_orbitals(model), max_rank=2)
    generators = [identity_generator(model.n), *pool[:size - 1]]
    bank = MatrixElementBank(ExactMVBackend().state(model.reference, ()),
                             model.hamiltonian, generators, **kwargs)
    bank.matrices()
    return bank


# --- The store itself

def test_layouts_and_backends_are_declared():
    assert LAYOUTS == ("interleaved", "soa")
    assert STORAGE_BACKENDS == ("object", "packed")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_packed_row_replays_its_emission_order(layout):
    """The order the geometric product emitted, not ascending word order.

    ``MV.trace_pairing`` iterates the smaller operand's dict in insertion order,
    so this is the difference between reproducing a committed matrix element and
    merely landing near it.
    """
    operator = MV(2, {0b1101: 1.5 + 0.25j, 0b0010: -2.0, 0b1000: 0.75j})
    emitted = list(operator.terms.items())
    assert [code for code, _ in emitted] != sorted(code for code, _ in emitted), (
        "this fixture only tests anything if its emission order is not ascending")
    store = PackedRowStore(2, GlobalWordTable(), layout=layout)
    store.add((0, 0), operator)
    assert store.terms((0, 0)) == emitted
    assert list(store.materialize((0, 0)).terms.items()) == emitted


@pytest.mark.parametrize("layout", LAYOUTS)
def test_packed_indices_are_ascending_even_though_replay_is_not(layout):
    """Ascending storage is what makes a probe a binary search."""
    operator = MV(2, {0b1101: 1.0, 0b0010: 2.0, 0b1000: 3.0})
    store = PackedRowStore(2, GlobalWordTable(), layout=layout)
    store.add((0, 0), operator)
    indices = store.word_indices((0, 0))
    assert list(indices) == sorted(indices)


def test_bulk_intern_agrees_with_the_scalar_probe():
    """The fast path must resolve exactly what the slow one would, batch by batch.

    Two tables walk the same code sequence, one through ``intern_many`` and one
    through ``intern``, across batches that straddle several table growths. A
    bulk path that disagreed anywhere would assign a row a word index belonging
    to some other word, which no later check would catch.
    """
    rng = random.Random(11)
    codes = [rng.randrange(4 ** 12) for _ in range(20000)]
    bulk, scalar = GlobalWordTable(), GlobalWordTable()
    position = 0
    while position < len(codes):
        batch = codes[position:position + rng.randrange(1, 400)]
        position += len(batch)
        assert bulk.intern_many(batch) == [scalar.intern(code) for code in batch]
    assert len(bulk) == len(scalar)
    assert ([bulk.code(i) for i in range(len(bulk))]
            == [scalar.code(i) for i in range(len(scalar))])


def test_bulk_intern_handles_duplicates_inside_one_batch():
    """Including a code that is new *and* repeated, which only the fallback sees."""
    table = GlobalWordTable()
    fresh, known = 12345, 7
    got = table.intern_many([fresh, known, fresh, known, fresh])
    assert got == [got[0], got[1], got[0], got[1], got[0]]
    assert table.code(got[0]) == fresh
    assert table.code(got[1]) == known
    assert all(type(value) is int for value in got), "must return Python ints"


def test_bulk_intern_survives_a_growth_inside_the_batch():
    """The first probe is read against the pre-batch slots; indices never move."""
    table = GlobalWordTable(capacity=16)
    codes = list(range(100_000, 105_000))
    assert table.intern_many(codes) == list(range(len(codes)))
    assert all(table.lookup(code) == index for index, code in enumerate(codes))


def test_bulk_intern_accepts_an_empty_batch():
    assert GlobalWordTable().intern_many([]) == []


def test_bulk_and_scalar_intern_interleave():
    table = GlobalWordTable()
    rng = random.Random(5)
    seen: dict[int, int] = {}
    for step in range(2000):
        code = rng.randrange(4 ** 10)
        index = table.intern(code) if step % 3 else table.intern_many([code])[0]
        assert seen.setdefault(code, index) == index
    assert all(table.lookup(code) == index for code, index in seen.items())


def test_a_row_is_immutable_once_packed():
    store = PackedRowStore(2, GlobalWordTable())
    store.add((0, 0), MV(2, {1: 1.0}))
    with pytest.raises(KeyError, match="immutable"):
        store.add((0, 0), MV(2, {2: 1.0}))


def test_the_word_table_assigns_stable_indices():
    """Indices are handed out first-seen and never move: packed rows hold them."""
    table = GlobalWordTable()
    first = table.intern(9)
    assert table.intern(4) != first
    assert table.intern(9) == first
    assert table.code(first) == 9
    assert table.lookup(9) == first
    assert table.lookup(1234) is None


def test_packed_rows_cost_exactly_the_modelled_bytes():
    """Four bytes of index, four of permutation, sixteen of coefficient."""
    bank = _bank(H4, storage="packed")
    store = bank._packed["element"]
    assert store.nbytes() == PACKED_BYTES_PER_COEFFICIENT * store.total_coefficients()
    assert store.bytes_per_coefficient() == PACKED_BYTES_PER_COEFFICIENT


def test_reserved_bytes_never_understate_live_bytes():
    """Geometric growth leaves slack, and the slack is real resident memory."""
    bank = _bank(H4, storage="packed")
    for store in bank._packed.values():
        assert store.reserved_bytes() >= store.nbytes()


# --- Backend equivalence, bitwise

@pytest.mark.parametrize("path", [H4, BEH2])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_backends_agree_bitwise_on_a_fixed_label_bank(path, layout):
    reference = _bank(path)
    packed = _bank(path, storage="packed", layout=layout)
    S0, H0 = reference.matrices()
    S1, H1 = packed.matrices()
    assert np.array_equal(S0, S1), "overlap matrix drifted under packing"
    assert np.array_equal(H0, H1), "Hamiltonian matrix drifted under packing"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_backends_agree_bitwise_through_adaptive_selection(layout):
    """The frontier path too: rejected candidates are packed and read back."""
    model = fcidump_model(H4, name="h4")
    rho = ExactMVBackend().state(model.reference, ())
    pool = determinant_excitations(model.n, occupied_spin_orbitals(model), max_rank=2)

    def grow(**kwargs):
        bank = MatrixElementBank(rho, model.hamiltonian, (), **kwargs)
        return run_acase(rho, model.hamiltonian, pool, bank=bank,
                         initial=[identity_generator(model.n)], max_size=4)

    reference = grow()
    packed = grow(storage="packed", layout=layout)
    assert packed.labels == reference.labels
    assert packed.energy == reference.energy, "selected energy drifted under packing"
    S0, H0 = reference.bank.matrices(reference.indices)
    S1, H1 = packed.bank.matrices(packed.indices)
    assert np.array_equal(S0, S1)
    assert np.array_equal(H0, H1)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_every_entry_agrees_bitwise_including_the_transpose(layout):
    reference = _bank(H4)
    packed = _bank(H4, storage="packed", layout=layout)
    size = len(reference.labels)
    for i in range(size):
        for j in range(size):
            assert packed.entry(i, j) == reference.entry(i, j), f"entry ({i},{j})"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_materialized_operators_match_term_for_term(layout):
    reference = _bank(H4)
    packed = _bank(H4, storage="packed", layout=layout)
    size = len(reference.labels)
    for i in range(size):
        for j in range(size):
            for read in ("overlap_operator", "element_operator"):
                expected = getattr(reference, read)(i, j)
                actual = getattr(packed, read)(i, j)
                assert list(actual.terms.items()) == list(expected.terms.items()), (
                    f"{read}({i},{j}) lost its coefficients or their order")


# --- The resource ledger stays backend-neutral

@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_structural_ledger_is_identical_under_both_backends(layout):
    """Packing is a storage change: every count it reports must be unmoved."""
    reference = _bank(H4).resources()
    packed = _bank(H4, storage="packed", layout=layout).resources()
    for field in ("basis_size", "word_universe", "resident_word_universe",
                  "coefficient_occurrences", "coefficient_reuse",
                  "cached_operator_bytes", "pairs_built", "retained_block_pairs",
                  "selection_pairs", "retained_pair_fraction",
                  "resident_operator_rows", "peak_resident_operator_rows",
                  "max_overlap_element_support", "max_hamiltonian_element_support",
                  "hamiltonian_support", "new_words_per_generator"):
        assert packed[field] == reference[field], field


def test_the_ledger_names_the_backend_that_produced_it():
    assert _bank(H4).resources()["storage_backend"] == "object"
    packed = _bank(H4, storage="packed", layout="soa")
    assert packed.resources()["storage_backend"] == "packed"
    assert packed.measured_storage_bytes()["coefficient_layout"] == "soa"
    assert _bank(H4).measured_storage_bytes()["coefficient_layout"] == "python_dict"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_word_set_is_unchanged_by_packing(layout):
    reference = _bank(H4)
    packed = _bank(H4, storage="packed", layout=layout)
    assert packed.word_set() == reference.word_set()
    assert packed.word_set([0, 1]) == reference.word_set([0, 1])


def test_an_unknown_backend_is_refused():
    model = fcidump_model(H4, name="h4")
    with pytest.raises(ValueError, match="storage must be one of"):
        MatrixElementBank(ExactMVBackend().state(model.reference, ()),
                          model.hamiltonian, (), storage="mmap")


def test_an_unknown_layout_is_refused():
    with pytest.raises(ValueError, match="layout must be one of"):
        PackedRowStore(2, GlobalWordTable(), layout="columnar")


# --- What the packing actually buys

def test_packed_rows_beat_the_object_rows_per_coefficient():
    """The per-coefficient reduction 2M-B's go/no-go is stated against.

    Rows only. The shared word table is priced per distinct *word* rather than
    per occurrence, so the total reduction depends on a bank's coefficient
    reuse; that dependence is what the 2M-B record measures, and it is not a
    property of the row representation being tested here.
    """
    reference = _bank(H4).measured_storage_bytes()
    packed = _bank(H4, storage="packed")
    rows = sum(store.nbytes() for store in packed._packed.values())
    per_coefficient = rows / packed.resources()["coefficient_occurrences"]
    assert per_coefficient == PACKED_BYTES_PER_COEFFICIENT
    assert reference["measured_bytes_per_coefficient"] / per_coefficient > 3.0


@pytest.mark.parametrize("layout", LAYOUTS)
def test_both_layouts_store_the_same_number_of_bytes(layout):
    """Interleaved and structure-of-arrays differ in access, not in size."""
    interleaved = _bank(H4, storage="packed", layout="interleaved")
    other = _bank(H4, storage="packed", layout=layout)
    assert (sum(s.nbytes() for s in other._packed.values())
            == sum(s.nbytes() for s in interleaved._packed.values()))
