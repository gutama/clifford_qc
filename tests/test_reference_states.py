"""Independent determinant semantics, legacy arithmetic, and capability safety."""
from dataclasses import FrozenInstanceError
import itertools
import struct

import numpy as np
import pytest

from clifford_qc.backends import ExactMVBackend
from clifford_qc.ir import Program
from clifford_qc.multivector import MV
from clifford_qc.reference_states import (
    DeterminantReference, MVReference, ReferenceMetadata, ReferenceState,
    UnsupportedReferenceOperation, require_exact_reference,
)
from clifford_qc.states import ket_density


def bits_of(value):
    value = complex(value)
    return struct.pack("!dd", value.real, value.imag)


def raw_operator(n, terms):
    # Exercise small/subnormal coefficients and signed zero without asking the
    # MV constructor's separate pruning policy to discard these test inputs.
    result = MV(n)
    result.terms = dict(terms)
    return result


def independent(bits, terms):
    result = 0j
    for code, value in terms:
        sign = 1
        for q, bit in enumerate(bits):
            letter = (code >> (2 * q)) & 3
            if letter in (1, 2):
                break
            if letter == 3 and bit == "1":
                sign = -sign
        else:
            result += value * sign
    return result


@pytest.mark.parametrize("n", range(5))
def test_every_small_bit_pattern_matches_independent_pauli_calculation(n):
    terms = [(code, complex((code % 7) - 3, (code % 5) - 2)) for code in range(4 ** n)]
    for bits in map("".join, itertools.product("01", repeat=n)):
        reference = DeterminantReference(bits)
        observed = reference.evaluate_terms(iter(terms), row_nnz=len(terms))
        assert observed.value == independent(bits, terms)
        assert observed.terms_scanned == len(terms)
        assert observed.diagonal_terms == 2 ** n
        assert observed.compatibility_sort_terms == 0


@pytest.mark.parametrize("bits", ["", "0", "1", "01", "10", "101", "1100"])
@pytest.mark.parametrize("dense_row", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_legacy_trace_matches_both_operand_branches_and_emission_orders(bits, dense_row, reverse):
    n = len(bits)
    codes = list(range(4 ** n)) if dense_row else [0, 3] if n else [0]
    if reverse:
        codes.reverse()
    terms = [(code, complex((1e8, 1e-8, -1e8, 0.125)[code % 4],
                            (-0.0, 0.5, -0.5)[code % 3])) for code in codes]
    operator = raw_operator(n, terms)
    rho = ket_density(n, bits)
    expected = float(2 ** n) * operator.trace_pairing(rho)
    for reference in (DeterminantReference(bits, arithmetic="legacy_trace"), MVReference(rho)):
        observed = reference.evaluate_terms(iter(terms), row_nnz=len(terms))
        assert bits_of(observed.value) == bits_of(expected)
        assert bits_of(reference.expectation(operator)) == bits_of(expected)
    determinant = DeterminantReference(bits, arithmetic="legacy_trace")
    result = determinant.evaluate_terms(terms, row_nnz=len(terms))
    assert result.compatibility_sort_terms == (2 ** n if len(terms) > 2 ** n else 0)
    assert (result.peak_index_bytes > 0) == (len(terms) > 2 ** n)


@pytest.mark.parametrize("coefficient", [complex(1e-300, -1e-300), complex(1e300, -1e300),
                                        complex(-0.0, 2.0), complex(2.0, -0.0)])
def test_supported_extreme_coefficients_and_signed_zeros(coefficient):
    rho = ket_density(3, "101")
    operator = raw_operator(3, [(3, coefficient), (0, coefficient * 0.125)])
    expected = float(2 ** 3) * operator.trace_pairing(rho)
    assert bits_of(DeterminantReference("101", arithmetic="legacy_trace").expectation(operator)) == bits_of(expected)
    np.testing.assert_allclose(DeterminantReference("101").expectation(operator),
                               independent("101", list(operator.terms.items())), rtol=1e-14, atol=0)


def test_analytic_and_legacy_scaling_are_distinct_contracts():
    # Analytic evaluation preserves a subnormal that trace scaling underflows.
    value = float.fromhex("0x0.0000000000001p-1022")
    terms = [(0, complex(value))]
    analytic = DeterminantReference("000").evaluate_terms(terms, row_nnz=1)
    legacy = DeterminantReference("000", arithmetic="legacy_trace").evaluate_terms(terms, row_nnz=1)
    assert analytic.value.real == value and legacy.value.real == 0
    with pytest.raises(ValueError, match="density cutoff"):
        DeterminantReference("0" * 40, arithmetic="legacy_trace")


def test_large_register_bounded_row_forbids_density_and_hilbert_allocations(monkeypatch):
    import clifford_qc.states as states
    import clifford_qc.dense_reference as dense
    def forbidden(*args, **kwargs):
        raise AssertionError("density or dense Hilbert-space allocation")
    monkeypatch.setattr(states, "ket_density", forbidden)
    monkeypatch.setattr(ExactMVBackend, "__init__", forbidden)
    monkeypatch.setattr(ExactMVBackend, "state", forbidden)
    monkeypatch.setattr(dense, "to_matrix", forbidden)
    for name in ("zeros", "empty", "ones", "eye"):
        monkeypatch.setattr(np, name, forbidden)
    bits = "1" + "0" * 4094 + "1"
    reference = DeterminantReference(bits)
    terms = [(0, 0.5), (3, 0.125j), (3 << (2 * 4095), -2), (1 << (2 * 2000), 12)]
    assert reference.evaluate_terms(terms, row_nnz=4).value == independent(bits, terms)
    assert reference.storage()["density_coefficients"] == 0
    assert reference.storage()["occupation_payload_bytes"] == 4096


def test_preparation_reads_nonprefix_and_repeated_x_and_rejects_other_gates():
    program = Program(4)
    for q in (0, 3, 0, 2):
        program.clifford("X", q)
    reference = DeterminantReference.from_program(program, arithmetic="legacy_trace")
    assert reference.bits == "0011" and reference.occupied == (2, 3)
    rho = ExactMVBackend().state(program, ())
    operator = raw_operator(4, [(code, complex(code + 1)) for code in range(256)])
    assert bits_of(reference.expectation(operator)) == bits_of(16.0 * operator.trace_pairing(rho))
    program.clifford("H", 0)
    with pytest.raises(UnsupportedReferenceOperation, match="X-only"):
        DeterminantReference.from_program(program)


def test_identity_is_immutable_order_and_bit_sensitive_and_snapshots_are_owned():
    reference = DeterminantReference("101")
    with pytest.raises(FrozenInstanceError): reference.bits = "100"
    assert reference.metadata.identity != DeterminantReference("100").metadata.identity
    assert reference.metadata.identity != DeterminantReference("101", spin_ordering="blocked").metadata.identity
    assert reference.metadata.identity != DeterminantReference("101", arithmetic="legacy_trace").metadata.identity
    rho = ket_density(2, "01")
    snapshot = MVReference(rho)
    identity = snapshot.metadata.identity
    original = snapshot.expectation(MV.scalar(2, 1))
    rho.terms.clear()
    snapshot.density().terms.clear()
    assert snapshot.expectation(MV.scalar(2, 1)) == original == 1
    assert snapshot.metadata.identity == identity
    with pytest.raises(TypeError): snapshot._density.terms[0] = 12
    with pytest.raises(AttributeError): snapshot._density.n = 99
    reverse = ket_density(2, "01")
    reverse.terms = dict(reversed(list(reverse.terms.items())))
    assert MVReference(reverse).metadata.identity != identity


def test_sampled_only_reference_does_not_need_an_exact_method():
    class SampledOnly:
        metadata = ReferenceMetadata(4, "preparation-and-device-id", frozenset({"sampling"}))
        n = 4
    reference = SampledOnly()
    assert isinstance(reference, ReferenceState)
    with pytest.raises(UnsupportedReferenceOperation, match="exact expectation"):
        require_exact_reference(reference)


@pytest.mark.parametrize("bits", ["0102", None, 1001])
def test_invalid_determinants(bits):
    with pytest.raises(ValueError): DeterminantReference(bits)


@pytest.mark.parametrize("terms,count", [([(0, 1)], 2), ([(0, 1)], -1),
                                         ([(0, 1)], True), ([(4, 1)], 1),
                                         ([(True, 1)], 1), ([(0, float("nan"))], 1)])
def test_bad_rows_fail_visibly(terms, count):
    with pytest.raises(ValueError):
        DeterminantReference("0").evaluate_terms(terms, row_nnz=count)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("codes", [[2, 1], [3, 2, 1, 0]])
def test_mv_reference_retains_off_diagonal_complex_state_pairings(reverse, codes):
    # A pure Bloch state with a nonzero Y component is not a determinant.
    coefficients = [(0, 0.5), (1, 0.3), (2, 0.4)]
    if reverse:
        coefficients.reverse()
    rho = MV(1, dict(coefficients))
    operator = raw_operator(1, [(code, complex(code - 0.125, 0.25 - code)) for code in codes])
    reference = MVReference(rho, spin_ordering="qubits")
    assert bits_of(reference.expectation(operator)) == bits_of(2.0 * operator.trace_pairing(rho))
    assert reference.evaluate_terms(operator.terms.items(), row_nnz=len(codes)).diagonal_terms is None
    assert reference.purity == 1.0
