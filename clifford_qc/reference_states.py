"""Reference identity and exact row evaluation, independent of density storage.

Qubit zero is the leftmost occupation bit and the lowest two-bit Pauli lane.
Pauli words are Hermitian; phases stay in coefficients. The pairing is
bilinear (no coefficient conjugation or Clifford reversion).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import sys
from types import MappingProxyType
from typing import Iterable, Protocol, runtime_checkable

from .multivector import MV, TOL
from .pauli_kernel import pauli_lane_mask, validate_word_code


def immutable_array(value, dtype=None):
    """Own a NumPy snapshot whose bytes cannot be made writeable again."""
    import numpy as np
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class UnsupportedReferenceOperation(TypeError):
    """The reference does not provide the requested execution capability."""


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def operator_identity(operator: MV) -> str:
    """Ordered coefficient bits, including signed zeros, enter cache identity."""
    return _digest([operator.n, [(code, complex(c).real.hex(), complex(c).imag.hex())
                                for code, c in operator.terms.items()]])


class _FrozenMV(MV):
    """Owned immutable input snapshot; ordinary MV operations return fresh MVs."""
    __slots__ = ("_term_container_bytes",)

    def __init__(self, operator: MV):
        terms = {}
        for code, value in operator.terms.items():
            validate_word_code(operator.n, code)
            value = complex(value)
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                raise ValueError("reference/provider coefficients must be finite")
            terms[code] = value
        object.__setattr__(self, "n", operator.n)
        object.__setattr__(self, "terms", MappingProxyType(terms))
        object.__setattr__(self, "_term_container_bytes", sys.getsizeof(terms))

    def __setattr__(self, name, value):
        raise AttributeError("reference/provider input snapshots are immutable")

    def __delattr__(self, name):
        raise AttributeError("reference/provider input snapshots are immutable")

    def boxed_storage(self):
        containers, boxes = super().boxed_storage()
        return containers + self._term_container_bytes, boxes


def snapshot_operator(operator: MV) -> MV:
    """Own an immutable, ordered snapshot without coefficient repruning."""
    return _FrozenMV(operator)


@dataclass(frozen=True)
class ReferenceMetadata:
    n: int
    identity: str
    capabilities: frozenset[str]
    bit_order: str = "q0-leftmost"
    spin_ordering: str = "interleaved"

    def __post_init__(self):
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 0:
            raise ValueError("reference n must be a nonnegative integer")
        if not isinstance(self.identity, str) or not self.identity:
            raise ValueError("reference identity must be a nonempty string")
        if self.bit_order != "q0-leftmost":
            raise ValueError("only q0-leftmost reference bit ordering is supported")
        if self.spin_ordering not in ("interleaved", "blocked", "qubits"):
            raise ValueError("unsupported reference spin ordering")
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))


@runtime_checkable
class ReferenceState(Protocol):
    """Immutable identity/conventions; exact expectation is not required."""
    @property
    def metadata(self) -> ReferenceMetadata: ...

    @property
    def n(self) -> int: ...


@dataclass(frozen=True)
class RowEvaluation:
    value: complex
    terms_scanned: int
    diagonal_terms: int | None
    compatibility_sort_terms: int = 0
    peak_index_bytes: int = 0


@runtime_checkable
class ExactReference(ReferenceState, Protocol):
    @property
    def purity(self) -> float: ...

    def evaluate_terms(self, terms: Iterable[tuple[int, complex]], *,
                       row_nnz: int) -> RowEvaluation: ...

    def storage(self) -> dict: ...


@runtime_checkable
class SectorActionReference(ReferenceState, Protocol):
    """Step 3 boundary: declared sector plus numerical operator actions."""
    @property
    def sector(self) -> tuple[int, float]: ...

    def apply_operator(self, operator): ...


@runtime_checkable
class SampledReference(ReferenceState, Protocol):
    """Step 5 boundary; no exact amplitudes/expectations are implied.

    Implementations bind settings, shots and actual returned observations to
    metadata.identity. Diagnostic exact oracles are separate objects.
    """
    def sample(self, settings, shots: int, *, seed: int): ...


def require_exact_reference(reference) -> ExactReference:
    if (not isinstance(reference, ReferenceState)
            or not isinstance(reference.metadata, ReferenceMetadata)
            or "exact_expectation" not in reference.metadata.capabilities
            or not isinstance(reference, ExactReference)):
        raise UnsupportedReferenceOperation("reference lacks exact expectation capability")
    return reference


def _index_footprint(container, objects):
    """Retained footprint, including referenced boxes (possibly shared with a row)."""
    seen = {id(container)}
    size = sys.getsizeof(container)
    for value in objects:
        if id(value) not in seen:
            seen.add(id(value))
            size += sys.getsizeof(value)
    return size


def _row_terms(n, terms, row_nnz):
    """Rows are canonical distinct-code emissions, as supplied by bank stores."""
    if isinstance(row_nnz, bool) or not isinstance(row_nnz, int) or row_nnz < 0:
        raise ValueError("row_nnz must be a nonnegative integer")
    count = 0
    for code, value in terms:
        if isinstance(code, bool):
            raise ValueError("Pauli word codes must be integers, not booleans")
        validate_word_code(n, code)
        value = complex(value)
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("row coefficients must be finite")
        count += 1
        yield code, value
    if count != row_nnz:
        raise ValueError("row_nnz differs from the supplied term count")


@dataclass(frozen=True)
class DeterminantReference:
    bits: str
    spin_ordering: str = "interleaved"
    arithmetic: str = "analytic"
    metadata: ReferenceMetadata = field(init=False)
    _occupied_lanes: int = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.bits, str) or set(self.bits) - {"0", "1"}:
            raise ValueError("determinant bits must be a binary string")
        if self.arithmetic not in ("analytic", "legacy_trace"):
            raise ValueError("reference arithmetic must be analytic or legacy_trace")
        if self.arithmetic == "legacy_trace" and math.ldexp(1.0, -self.n) <= TOL:
            raise ValueError("legacy_trace requires n < 40; use analytic beyond the MV density cutoff")
        lanes = sum(1 << (2 * q) for q, bit in enumerate(self.bits) if bit == "1")
        object.__setattr__(self, "_occupied_lanes", lanes)
        identity = _digest(["determinant.v1", self.bits, self.spin_ordering, self.arithmetic])
        object.__setattr__(self, "metadata", ReferenceMetadata(
            self.n, identity, frozenset({"exact_expectation", "determinant"}),
            spin_ordering=self.spin_ordering))

    @property
    def n(self): return len(self.bits)

    @property
    def purity(self): return 1.0

    @property
    def occupied(self): return tuple(q for q, bit in enumerate(self.bits) if bit == "1")

    @classmethod
    def from_program(cls, program, **kwargs):
        """Read computational X preparation only; reject unsupported preparations."""
        bits = ["0"] * program.n
        for operation in program.ops:
            if (getattr(operation, "name", None) != "X"
                    or len(operation.qubits) != 1):
                raise UnsupportedReferenceOperation("determinant reference requires an X-only preparation")
            q = operation.qubits[0]
            if not 0 <= q < program.n:
                raise ValueError("reference preparation qubit out of range")
            bits[q] = "0" if bits[q] == "1" else "1"
        return cls("".join(bits), **kwargs)

    def evaluate_terms(self, terms, *, row_nnz):
        mask = pauli_lane_mask(self.n)
        compatible = self.arithmetic == "legacy_trace"
        density_smaller = compatible and row_nnz > (1 << self.n)
        diagonal = [] if density_smaller else None
        total, scanned, hits = 0j, 0, 0
        density_scale = math.ldexp(1.0, -self.n) if compatible else None
        for code, coefficient in _row_terms(self.n, terms, row_nnz):
            scanned += 1
            z = (code >> 1) & mask
            if ((code & mask) ^ z) != 0:
                continue
            hits += 1
            sign = -1 if (z & self._occupied_lanes).bit_count() % 2 else 1
            if density_smaller:
                ordinal, remaining = 0, z
                while remaining:
                    lane = remaining & -remaining
                    q = (lane.bit_length() - 1) // 2
                    ordinal |= 1 << (self.n - 1 - q)
                    remaining ^= lane
                diagonal.append((ordinal, coefficient, sign))
            else:
                partner = complex(sign * density_scale, 0.0) if compatible else sign
                total += coefficient * partner
        index_bytes = 0
        if diagonal is not None:
            # ket_density emits increasing subset ordinals, not packed codes.
            # Only present diagonal terms are indexed; no 2**n reference is made.
            diagonal.sort(key=lambda item: item[0])
            index_bytes = _index_footprint(diagonal, (
                value for item in diagonal for value in (item, *item)))
            for _, coefficient, sign in diagonal:
                total += complex(sign * density_scale, 0.0) * coefficient
        value = math.ldexp(1.0, self.n) * total if compatible else total
        return RowEvaluation(value, scanned, hits, hits if density_smaller else 0, index_bytes)

    def expectation(self, operator: MV) -> complex:
        if operator.n != self.n:
            raise ValueError("reference and operator disagree on n")
        return self.evaluate_terms(operator.terms.items(), row_nnz=operator.nnz()).value

    def storage(self):
        return {"representation": "determinant", "occupation_bits": self.n,
                "occupation_payload_bytes": len(self.bits.encode("ascii")),
                "cached_mask_payload_bytes": (2 * self.n + 7) // 8,
                "density_coefficients": 0,
                "scope": "occupation ASCII and cached mask payload; excludes Python/metadata overhead and RSS"}


@dataclass(frozen=True, init=False)
class MVReference:
    """Immutable compatibility snapshot with existing density-MV semantics."""
    metadata: ReferenceMetadata
    _density: MV = field(repr=False)
    purity: float

    def __init__(self, density: MV, *, spin_ordering="interleaved"):
        if not isinstance(density, MV):
            raise TypeError("MVReference requires an MV density")
        snapshot = _FrozenMV(density)
        if not snapshot.is_hermitian() or abs(snapshot.trace() - 1.0) > 1e-9:
            raise ValueError("reference density must be Hermitian with unit trace")
        identity = _digest(["density_mv.v1", operator_identity(snapshot), spin_ordering])
        object.__setattr__(self, "metadata", ReferenceMetadata(
            snapshot.n, identity, frozenset({"exact_expectation", "density_mv"}),
            spin_ordering=spin_ordering))
        object.__setattr__(self, "_density", snapshot)
        object.__setattr__(self, "purity", float((2 ** snapshot.n) * snapshot.trace_pairing(snapshot).real))

    @property
    def n(self): return self.metadata.n

    def density(self) -> MV:
        """Explicit defensive legacy conversion; never used by determinant evaluation."""
        return self._density.copy()

    def evaluate_terms(self, terms, *, row_nnz):
        total = 0j
        if row_nnz <= self._density.nnz():
            for code, coefficient in _row_terms(self.n, terms, row_nnz):
                partner = self._density.terms.get(code)
                if partner is not None:
                    total += coefficient * partner
            index_bytes = 0
            scanned = row_nnz
        else:
            lookup = dict(_row_terms(self.n, terms, row_nnz))
            index_bytes = _index_footprint(lookup, (
                value for item in lookup.items() for value in item))
            for code, coefficient in self._density.terms.items():
                partner = lookup.get(code)
                if partner is not None:
                    total += coefficient * partner
            scanned = row_nnz + self._density.nnz()
        return RowEvaluation(float(2 ** self.n) * total, scanned, None,
                             peak_index_bytes=index_bytes)

    def expectation(self, operator: MV) -> complex:
        if operator.n != self.n:
            raise ValueError("reference and operator disagree on n")
        return self.evaluate_terms(operator.terms.items(), row_nnz=operator.nnz()).value

    def storage(self):
        containers, boxes = self._density.boxed_storage()
        distinct = {id(box): box for box in boxes}
        return {"representation": "density_mv_snapshot", "density_coefficients": self._density.nnz(),
                "coefficient_payload_bytes": self._density.memory_estimate(),
                "measured_snapshot_bytes": containers + sum(sys.getsizeof(box) for box in distinct.values()),
                "scope": "owned density snapshot containers and boxes; excludes metadata and process RSS"}
