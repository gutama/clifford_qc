"""Product-free storage admission for a scoped second-moment block.

This bounds sparse operator *coefficient occurrences*, not bytes or peak RSS.
It neither builds ``H²`` nor asks the bank to build, restore, or materialize a
row.  Bounds use support cardinalities and affine spans of Pauli-word codes:
the code of a product is the XOR of its factors' codes, irrespective of phase.
Cancellation can only reduce these bounds.

Packing and S/H streaming do not discount K: projected-observable rows and
their action caches are ordinary retained multivectors in either backend.
Every operator/reference/generator combination uses the same analytic bound;
names and labels cannot confer admission.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterable, Sequence

from ..multivector import MV
from .projection import MatrixElementBank, _identity_key


# Q16's fixed coefficient-occurrence anchor. This is a finite structural
# budget, not a memory limit or an assertion about available process memory.
DEFAULT_MAX_COEFFICIENT_OCCURRENCES = 96_107_619


def operator_fingerprint(operator: MV) -> str:
    """Canonical exact-value fingerprint, without a product or a dense copy."""
    digest = hashlib.sha256()
    digest.update(f"[{operator.n},[".encode())
    # Hash one term at a time: do not duplicate a potentially large reference
    # into nested JSON arrays and hex strings just to fingerprint its values.
    for position, code in enumerate(sorted(operator.terms)):
        if position:
            digest.update(b",")
        value = operator.terms[code]
        digest.update(json.dumps(
            [code, float(value.real).hex(), float(value.imag).hex()],
            separators=(",", ":"),
        ).encode())
    digest.update(b"]]")
    return digest.hexdigest()


def reference_coefficient_binding(reference) -> tuple[str, int]:
    """Bind a reference without inventing a density for capability-only inputs.

    The coefficient budget counts explicit MV coefficients. A determinant,
    sector or sampled preparation has no reference-density coefficients here;
    its metadata identity binds the external reference capability. Its owned
    amplitudes, callback closure and device memory are outside this metric.
    Historical MV fingerprints and counts are unchanged.
    """
    if isinstance(reference, MV):
        return operator_fingerprint(reference), reference.nnz()
    from ..reference_states import ReferenceMetadata
    metadata = getattr(reference, "metadata", None)
    if not isinstance(metadata, ReferenceMetadata) or reference.n != metadata.n:
        raise TypeError("second moments need a bound reference identity")
    payload = json.dumps([metadata.n, metadata.identity, metadata.bit_order,
                          metadata.spin_ordering], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest(), 0


@dataclass(frozen=True)
class SecondMomentStoragePolicy:
    """Finite coefficient budget for one bank plus its requested K block."""

    max_coefficient_occurrences: int = DEFAULT_MAX_COEFFICIENT_OCCURRENCES

    def __post_init__(self) -> None:
        value = self.max_coefficient_occurrences
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError("max_coefficient_occurrences must be a positive integer")
        object.__setattr__(self, "max_coefficient_occurrences", int(value))


@dataclass(frozen=True)
class SecondMomentStoragePreflight:
    """Immutable, scoped admission evidence, obtained without any products.

    ``coefficient_occurrence_bound`` includes all requested S/H/K rows, H²,
    H² action caches, existing input/action multivectors and existing rows
    outside the requested S/H block. Initial admission counts existing
    observables in full, including second moments already resident. Refresh
    covers only the exact square's scoped cached rows/actions under its K
    bounds. A single output-row temporary is counted separately as well. Python containers,
    support histories, scalar matrices, intern tables and multiplication
    kernel caches are outside this coefficient metric, so this is not an RSS
    guarantee.

    Fingerprints bind the reference, H and the generators in ``indices``.
    New generators do not extend this scope; a new preflight is required.
    """

    indices: tuple[int, ...]
    n_qubits: int
    hamiltonian_terms: int
    generator_supports: tuple[int, ...]
    hamiltonian_square_terms_bound: int
    overlap_coefficient_occurrences_bound: int
    hamiltonian_coefficient_occurrences_bound: int
    second_moment_coefficient_occurrences_bound: int
    second_moment_action_coefficient_occurrences_bound: int
    existing_input_coefficient_occurrences: int
    existing_non_target_coefficient_occurrences: int
    temporary_coefficient_occurrences_bound: int
    coefficient_occurrence_bound: int
    policy: SecondMomentStoragePolicy
    reference_fingerprint: str
    hamiltonian_fingerprint: str
    generator_fingerprints: tuple[tuple[int, str], ...]
    input_fingerprint: str

    @property
    def basis_size(self) -> int:
        return len(self.indices)

    @property
    def block_pairs(self) -> int:
        return self.basis_size * (self.basis_size + 1) // 2

    @property
    def allowed(self) -> bool:
        return self.coefficient_occurrence_bound <= self.policy.max_coefficient_occurrences

    @property
    def coefficient_occurrences_bound(self) -> int:
        """Plural spelling, matching the bank's existing resource vocabulary."""
        return self.coefficient_occurrence_bound

    @property
    def max_coefficient_occurrences(self) -> int:
        return self.policy.max_coefficient_occurrences

    @property
    def reason(self) -> str:
        relation = "is within" if self.allowed else "exceeds"
        return (f"coefficient-occurrence bound {self.coefficient_occurrence_bound:,} "
                f"{relation} the finite budget {self.max_coefficient_occurrences:,}; "
                "this is a sparse-operator coefficient bound, not peak RSS")

    def matches_inputs(self, bank: MatrixElementBank) -> bool:
        """Whether the original scoped inputs still have exactly these values."""
        if not isinstance(bank, MatrixElementBank) or bank.n != self.n_qubits:
            return False
        if any(index >= len(bank) for index in self.indices):
            return False
        return (
            reference_coefficient_binding(bank.reference)[0] == self.reference_fingerprint
            and operator_fingerprint(bank.hamiltonian) == self.hamiltonian_fingerprint
            and all(operator_fingerprint(bank.generator(index).mv) == fingerprint
                    for index, fingerprint in self.generator_fingerprints)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "indices": list(self.indices),
            "basis_size": self.basis_size,
            "block_pairs": self.block_pairs,
            "n_qubits": self.n_qubits,
            "hamiltonian_terms": self.hamiltonian_terms,
            "generator_supports": list(self.generator_supports),
            "hamiltonian_square_terms_bound": self.hamiltonian_square_terms_bound,
            "overlap_coefficient_occurrences_bound": self.overlap_coefficient_occurrences_bound,
            "hamiltonian_coefficient_occurrences_bound": self.hamiltonian_coefficient_occurrences_bound,
            "second_moment_coefficient_occurrences_bound": self.second_moment_coefficient_occurrences_bound,
            "second_moment_action_coefficient_occurrences_bound": self.second_moment_action_coefficient_occurrences_bound,
            "existing_input_coefficient_occurrences": self.existing_input_coefficient_occurrences,
            "existing_non_target_coefficient_occurrences": self.existing_non_target_coefficient_occurrences,
            "temporary_coefficient_occurrences_bound": self.temporary_coefficient_occurrences_bound,
            "coefficient_occurrence_bound": self.coefficient_occurrence_bound,
            "max_coefficient_occurrences": self.max_coefficient_occurrences,
            "input_fingerprint": self.input_fingerprint,
            "metric": "sparse_operator_coefficient_occurrences",
            "bounds_peak_rss": False,
            "second_moment_storage_backend": "object",
        }


class SecondMomentStorageError(ValueError):
    """The requested block exceeds its finite coefficient budget."""

    def __init__(self, report: SecondMomentStoragePreflight):
        self.report = report
        super().__init__(f"second-moment storage gate refused: {report.reason}")


def _linear_basis(values: Iterable[int]) -> tuple[int, ...]:
    """An XOR basis using at most 2n integers, without enumerating its span."""
    pivots: dict[int, int] = {}
    for value in values:
        while value:
            bit = value.bit_length()
            if bit not in pivots:
                pivots[bit] = value
                break
            value ^= pivots[bit]
    return tuple(pivots.values())


def _affine_basis(operator: MV) -> tuple[int, ...]:
    codes = iter(operator.terms)
    origin = next(codes, 0)
    return _linear_basis(code ^ origin for code in codes)


def _support_bound(product_count: int, *bases: tuple[int, ...]) -> int:
    if product_count == 0:
        return 0
    rank = len(_linear_basis(value for basis in bases for value in basis))
    # Avoid even constructing an enormous power of two when the support
    # cardinality has already supplied the smaller bound.
    return product_count if rank >= product_count.bit_length() else min(product_count, 1 << rank)


def _existing_non_target_coefficients(bank: MatrixElementBank,
                                      wanted: frozenset[int],
                                      covered_observable_key: tuple | None = None) -> int:
    total = 0
    # Read native lengths only. In particular _resident_operators() would
    # materialize every packed row and is forbidden in a storage preflight.
    if bank._storage == "packed":
        for store in bank._packed.values():
            total += sum(store.row_length(pair) for pair in store.keys()
                         if not (pair[0] in wanted and pair[1] in wanted))
    else:
        for rows in (bank._overlap_ops, bank._element_ops):
            total += sum(operator.nnz() for pair, operator in rows.items()
                         if not (pair[0] in wanted and pair[1] in wanted))
    for key, record in bank._observables.items():
        covered = covered_observable_key is not None and key == covered_observable_key
        total += sum(operator.nnz() for pair, operator in record["operators"].items()
                     if not (covered and pair[0] in wanted and pair[1] in wanted))
        total += sum(operator.nnz() for index, operator in record["acted"].items()
                     if not (covered and index in wanted))
    return total


def second_moment_storage_preflight(
    bank: MatrixElementBank,
    indices: Sequence[int] | None = None,
    *,
    policy: SecondMomentStoragePolicy | None = None,
) -> SecondMomentStoragePreflight:
    """Bound a requested block without products, row reads, or bank mutation.

    The approved indices are the actual resolved block, including an empty
    block. Passing no indices means all generators currently registered; it
    does not approve future registrations.
    """
    return _second_moment_storage_preflight(bank, indices, policy=policy)


def _second_moment_storage_preflight(
    bank: MatrixElementBank,
    indices: Sequence[int] | None = None,
    *,
    policy: SecondMomentStoragePolicy | None = None,
    covered_observable_key: tuple | None = None,
) -> SecondMomentStoragePreflight:
    if not isinstance(bank, MatrixElementBank):
        raise TypeError("second-moment storage preflight needs a MatrixElementBank")
    if policy is None:
        policy = SecondMomentStoragePolicy()
    if not isinstance(policy, SecondMomentStoragePolicy):
        raise TypeError("policy must be a SecondMomentStoragePolicy")
    order = bank.resolve(indices)
    generators = tuple(bank.generator(index).mv for index in order)
    supports = tuple(operator.nnz() for operator in generators)
    bases = tuple(_affine_basis(operator) for operator in generators)
    h_terms = bank.hamiltonian.nnz()
    h_basis = _affine_basis(bank.hamiltonian)
    square_bound = _support_bound(h_terms * h_terms, h_basis)
    overlap_bound = hamiltonian_bound = second_bound = largest_row = 0
    action_bound = 0
    for j, (right_support, right_basis) in enumerate(zip(supports, bases)):
        action_bound += _support_bound(square_bound * right_support, h_basis, right_basis)
        for left_support, left_basis in zip(supports[:j + 1], bases[:j + 1]):
            pair_count = left_support * right_support
            overlap = _support_bound(pair_count, left_basis, right_basis)
            hamiltonian = _support_bound(pair_count * h_terms, left_basis, h_basis, right_basis)
            second = _support_bound(pair_count * square_bound, left_basis, h_basis, right_basis)
            overlap_bound += overlap
            hamiltonian_bound += hamiltonian
            second_bound += second
            largest_row = max(largest_row, overlap, hamiltonian, second)

    reference_fingerprint, reference_coefficients = reference_coefficient_binding(bank.reference)
    existing_inputs = (reference_coefficients + h_terms
                       + sum(generator.mv.nnz() for generator in bank._generators)
                       + sum(operator.nnz() for operator in bank._adjoints)
                       + sum(operator.nnz() for operator in bank._h_acted))
    existing_other = _existing_non_target_coefficients(
        bank, frozenset(order), covered_observable_key=covered_observable_key)
    # Products may expose one output multivector beside retained rows. This
    # counts its sparse coefficients; Python multiplication accumulators and
    # allocator overhead still require an independent peak-memory analysis.
    temporary = max(largest_row, square_bound,
                    max((_support_bound(square_bound * support, h_basis, basis)
                         for support, basis in zip(supports, bases)), default=0))
    total = (overlap_bound + hamiltonian_bound + second_bound + square_bound
             + action_bound + existing_inputs + existing_other + temporary)
    hamiltonian_fingerprint = operator_fingerprint(bank.hamiltonian)
    generator_fingerprints = tuple((index, operator_fingerprint(operator))
                                   for index, operator in zip(order, generators))
    input_fingerprint = hashlib.sha256(json.dumps(
        [reference_fingerprint, hamiltonian_fingerprint, generator_fingerprints],
        separators=(",", ":"),
    ).encode()).hexdigest()
    return SecondMomentStoragePreflight(
        indices=order, n_qubits=bank.n, hamiltonian_terms=h_terms,
        generator_supports=supports, hamiltonian_square_terms_bound=square_bound,
        overlap_coefficient_occurrences_bound=overlap_bound,
        hamiltonian_coefficient_occurrences_bound=hamiltonian_bound,
        second_moment_coefficient_occurrences_bound=second_bound,
        second_moment_action_coefficient_occurrences_bound=action_bound,
        existing_input_coefficient_occurrences=existing_inputs,
        existing_non_target_coefficient_occurrences=existing_other,
        temporary_coefficient_occurrences_bound=temporary,
        coefficient_occurrence_bound=total, policy=policy,
        reference_fingerprint=reference_fingerprint,
        hamiltonian_fingerprint=hamiltonian_fingerprint,
        generator_fingerprints=generator_fingerprints, input_fingerprint=input_fingerprint,
    )


def require_second_moment_storage(
    bank: MatrixElementBank,
    indices: Sequence[int] | None = None,
    *,
    policy: SecondMomentStoragePolicy | None = None,
) -> SecondMomentStoragePreflight:
    """Return the product-free admission report, raising before work if refused."""
    report = second_moment_storage_preflight(bank, indices, policy=policy)
    if not report.allowed:
        raise SecondMomentStorageError(report)
    return report


def refresh_second_moment_storage(
    bank: MatrixElementBank,
    report: SecondMomentStoragePreflight,
    square: MV,
) -> SecondMomentStoragePreflight:
    """Re-admit a fixed block against current caches, before another row.

    Only this exact square's rows/actions within the approved block are
    covered by the existing analytic K bounds. The same square's rows outside
    the block and all other observables remain additional retained state.
    Callers also guard the square's original fingerprint before using it.
    """
    if not report.matches_inputs(bank):
        raise ValueError("second-moment inputs changed since the storage preflight")
    refreshed = _second_moment_storage_preflight(
        bank, report.indices, policy=report.policy,
        covered_observable_key=_identity_key(square),
    )
    if not refreshed.allowed:
        raise SecondMomentStorageError(refreshed)
    return refreshed
