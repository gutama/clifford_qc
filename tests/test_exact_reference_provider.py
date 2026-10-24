"""Exact provider compatibility, row lifetime, ownership and capability tests."""
from dataclasses import dataclass

import numpy as np
import pytest

from clifford_qc.multivector import MV
from clifford_qc.reference_states import (
    DeterminantReference, MVReference, ReferenceMetadata, UnsupportedReferenceOperation,
)
from clifford_qc.states import ket_density
from clifford_qc.subspace.generator_core import Generator
from clifford_qc.subspace.generators import identity_generator
from clifford_qc.subspace.packed import PackedRowStore
from clifford_qc.subspace.projection import MatrixElementBank
from clifford_qc.subspace.providers import ExactProjectedProvider, ProjectedProblemProvider


def fixture():
    n = 3
    rng = np.random.default_rng(1024)
    h = MV(n, {code: rng.normal() for code in range(4 ** n)})
    generators = [identity_generator(n)]
    for index in range(3):
        codes = rng.permutation(4 ** n)
        generators.append(Generator(f"direction-{index}", MV(n, {
            int(code): complex(rng.normal(), rng.normal()) for code in codes})))
    return h, generators


def assert_bits(left, right):
    left, right = np.asarray(left), np.asarray(right)
    assert left.shape == right.shape
    assert left.tobytes() == right.tobytes()


@pytest.mark.parametrize("adapter", ["determinant", "mv"])
@pytest.mark.parametrize("storage,layout", [("object", "interleaved"), ("packed", "interleaved"),
                                           ("packed", "soa")])
@pytest.mark.parametrize("policy", ["retain_all", "stream_recompute"])
def test_exact_blocks_and_ritz_match_legacy_bytes(adapter, storage, layout, policy):
    h, generators = fixture()
    rho = ket_density(3, "101")
    legacy = MatrixElementBank(rho, h, generators)
    expected = legacy.matrices()
    reference = (DeterminantReference("101", arithmetic="legacy_trace")
                 if adapter == "determinant" else MVReference(rho))
    provider = ExactProjectedProvider(reference, h, generators, storage=storage,
                                       policy=policy, layout=layout, frontier_pairs=1)
    assert isinstance(provider, ProjectedProblemProvider)
    for actual, target in zip(provider.matrices(), expected):
        assert_bits(actual, target)
    assert_bits(provider.solve().energies, legacy.solve().energies)
    problem = provider.projected_problem()
    assert problem.labels == legacy.labels
    assert_bits(problem.solve().energies, legacy.solve().energies)
    # Numerical snapshots do not alias cache buffers.
    problem.overlap[:] = 99
    assert_bits(provider.matrices()[0], expected[0])
    for actual, target in zip(provider.matrices([3, 0, 1]), legacy.matrices([3, 0, 1])):
        assert_bits(actual, target)
    metadata = provider.execution_metadata()
    assert metadata["reference_identity"] == reference.metadata.identity
    assert metadata["reference_evaluation"]["calls"] == 2 * (4 * 5 // 2)
    assert metadata["reference_evaluation"]["terms_scanned"] > 0
    assert metadata["input_snapshot_coefficient_payload_bytes"] == 24 * (
        h.nnz() + sum(g.mv.nnz() for g in generators))


@pytest.mark.parametrize("adapter", ["determinant", "mv"])
@pytest.mark.parametrize("policy", ["retain_all", "stream_recompute"])
def test_packed_generation_order_evaluates_without_materializing_rows(adapter, policy, monkeypatch):
    h, generators = fixture()
    rho = ket_density(3, "110")
    expected = MatrixElementBank(rho, h, generators).matrices()
    reference = (DeterminantReference("110", arithmetic="legacy_trace")
                 if adapter == "determinant" else MVReference(rho))
    provider = ExactProjectedProvider(reference, h, generators, storage="packed",
                                       policy=policy, frontier_pairs=2)
    # First build structural rows without evaluating them; exercise packed reads
    # as well as fresh-product evaluation, for both density-size branches.
    provider.bank.pair_support(0, 0)
    provider.bank.pair_support(1, 1)
    def forbidden(*args, **kwargs):
        raise AssertionError("packed row materialized or paired against a density")
    monkeypatch.setattr(PackedRowStore, "materialize", forbidden)
    monkeypatch.setattr(PackedRowStore, "trace_pairing", forbidden)
    for actual, target in zip(provider.matrices(), expected):
        assert_bits(actual, target)


def test_provider_owns_reference_hamiltonian_and_generator_inputs():
    h, generators = fixture()
    h_before = h.copy()
    gens_before = [Generator(g.label, g.mv.copy()) for g in generators]
    rho = ket_density(3, "101")
    reference = MVReference(rho)
    provider = ExactProjectedProvider(reference, h, generators[:2])
    provider.bank.entry(0, 1)
    before = provider.cache_identity
    h.terms.clear()
    generators[1].mv.terms.clear()
    rho.terms.clear()
    assert provider.cache_identity == before
    provider.bank.add(generators[2])
    expected = MatrixElementBank(ket_density(3, "101"), h_before, gens_before[:3]).matrices()
    for actual, target in zip(provider.matrices(), expected):
        assert_bits(actual, target)
    with pytest.raises(TypeError): provider.bank.hamiltonian.terms[0] = 10
    with pytest.raises(TypeError): provider.bank.generator(1).mv.terms[0] = 10
    with pytest.raises(AttributeError): provider.bank.hamiltonian.n = 4
    with pytest.raises(AttributeError): del provider.bank.hamiltonian.terms
    assert provider.cache_identity != before  # different registered generator set


def test_cache_identity_includes_conventions_arithmetic_and_ordered_coefficients():
    h = MV(2, {0: 1, 3: 0.2})
    reference = DeterminantReference("01")
    base = ExactProjectedProvider(reference, h, [identity_generator(2)])
    variants = [
        ExactProjectedProvider(DeterminantReference("10"), h, [identity_generator(2)]),
        ExactProjectedProvider(DeterminantReference("01", spin_ordering="blocked"), h, [identity_generator(2)]),
        ExactProjectedProvider(DeterminantReference("01", arithmetic="legacy_trace"), h, [identity_generator(2)]),
        ExactProjectedProvider(reference, MV(2, {0: 1, 3: 0.3}), [identity_generator(2)]),
        ExactProjectedProvider(reference, MV(2, {3: 0.2, 0: 1}), [identity_generator(2)]),
        ExactProjectedProvider(reference, h, [identity_generator(2)], storage="packed"),
    ]
    assert all(p.cache_identity != base.cache_identity for p in variants)
    assert len({p.cache_identity for p in variants}) == len(variants)


def test_sampled_only_provider_fails_before_compiling_any_operator():
    @dataclass(frozen=True)
    class SampledOnly:
        n: int = 2
        metadata: ReferenceMetadata = ReferenceMetadata(2, "sampled-id", frozenset({"sampling"}))
    with pytest.raises(UnsupportedReferenceOperation, match="exact expectation"):
        ExactProjectedProvider(SampledOnly(), object())


def test_changed_custom_reference_identity_does_not_reuse_scalar_cache():
    class MutableAdapter:
        def __init__(self):
            self.delegate = DeterminantReference("0")
        @property
        def n(self): return self.delegate.n
        @property
        def metadata(self): return self.delegate.metadata
        @property
        def purity(self): return self.delegate.purity
        def evaluate_terms(self, *args, **kwargs): return self.delegate.evaluate_terms(*args, **kwargs)
        def storage(self): return self.delegate.storage()
    reference = MutableAdapter()
    provider = ExactProjectedProvider(reference, MV(1, {3: 1}), [identity_generator(1)])
    assert provider.bank.entry(0, 0) == (1, 1)
    reference.delegate = DeterminantReference("1")
    with pytest.raises(ValueError, match="identity changed"):
        provider.bank.entry(0, 0)
