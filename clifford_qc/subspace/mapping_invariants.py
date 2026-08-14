"""Invariant-first gate for the R2 fermion-mapping axis.

Every mapping arm transports one physical problem through a
:class:`Restriction`.  This module compares the transported projected problem
with the JW source before any grouping or device-cost number is allowed to be
reported.  Pure encoding changes require a Pauli-word bijection.  Two-qubit
reductions instead compare the reduced Hamiltonian with an independently
extracted fixed-parity block; word counts are then allowed to change.

The dense spectrum check is a small-system oracle.  Above
``dense_spectrum_max_qubits`` the algebraic CNOT/tableau construction and the
complete projected-matrix equality remain enforced, while the report labels
the full-spectrum oracle as not materialized.  A later benchmark may replace
that field with a sector-sparse oracle without changing the contract here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from ..dense_reference import to_matrix
from ..multivector import MV
from .contracts import as_multivector
from .generator_core import Generator, as_generators
from .linalg import DEFAULT_MAX_CONDITION, DEFAULT_TAU_S
from .projection import MatrixElementBank
from .restriction import Restriction


__all__ = ["MappingInvariantReport", "assert_mapping_invariants"]


@dataclass(frozen=True)
class MappingInvariantReport:
    """Machine-readable evidence emitted only after every applicable gate passes."""

    mapping_name: str
    source_qubits: int
    mapped_qubits: int
    qubits_removed: int
    basis_size: int
    word_universe_before: int
    word_universe_after: int
    effective_rank: int
    condition_number: float
    max_overlap_matrix_error: float
    max_hamiltonian_matrix_error: float
    max_operator_transport_error: float
    max_ritz_error: float
    reference_energy_error: float
    max_generator_leakage: float
    word_bijection_checked: bool
    dense_spectrum_checked: bool
    max_spectrum_error: float | None
    relative_tolerance: float
    absolute_tolerance: float
    leakage_tolerance: float
    zero_tolerance: float

    def to_dict(self) -> dict:
        return asdict(self)


def _max_abs(values) -> float:
    array = np.asarray(values)
    return float(np.max(np.abs(array), initial=0.0))


def _fixed_sector_block(
    rotated: MV, fixed_qubits: Sequence[int], signs: Sequence[int]
) -> np.ndarray:
    """Dense block selected without using ``Restriction.operator``."""
    matrix = to_matrix(rotated)
    keep = []
    for index in range(1 << rotated.n):
        accepted = True
        for qubit, sign in zip(fixed_qubits, signs):
            bit = rotated.n - 1 - qubit
            eigenvalue = 1 if ((index >> bit) & 1) == 0 else -1
            if eigenvalue != sign:
                accepted = False
                break
        if accepted:
            keep.append(index)
    return matrix[np.ix_(keep, keep)]


def _assert_close(
    error: float,
    scale: float,
    relative_tolerance: float,
    absolute_tolerance: float,
    label: str,
) -> None:
    allowed = absolute_tolerance + relative_tolerance * scale
    if error > allowed:
        raise AssertionError(f"mapping invariant failed for {label}: {error:.3e} > {allowed:.3e}")


def _mapped_word_set(restriction: Restriction, words: Sequence[int]) -> frozenset[int]:
    mapped = set()
    for code in words:
        image = restriction.rotate(MV(restriction.n, {int(code): 1.0}))
        if image.nnz() != 1:
            raise AssertionError("a Clifford encoding did not map one Pauli word to one word")
        mapped.add(next(iter(image.terms)))
    return frozenset(mapped)


def _generator_collisions(generators: Sequence[Generator]) -> list[tuple[str, str]]:
    seen: dict[tuple, str] = {}
    collisions = []
    for generator in generators:
        key = tuple(sorted(generator.mv.terms.items()))
        previous = seen.get(key)
        if previous is not None:
            collisions.append((previous, generator.label))
        else:
            seen[key] = generator.label
    return collisions


def assert_mapping_invariants(
    reference: MV,
    hamiltonian,
    generators,
    restriction: Restriction,
    *,
    tau_s: float = DEFAULT_TAU_S,
    max_condition: float = DEFAULT_MAX_CONDITION,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
    leakage_tolerance: float = 1e-12,
    zero_tolerance: float = 1e-12,
    dense_spectrum_max_qubits: int = 10,
) -> MappingInvariantReport:
    """Transport and gate one mapping arm before resource accounting.

    The generators must preserve the fixed parity sector.  A leaking
    generator is rejected rather than silently projected, because allowing the
    mapping arm to change the basis domain would confound encoding locality
    with candidate selection.
    """
    H = as_multivector(hamiltonian)
    source = as_generators(generators)
    if reference.n != H.n or restriction.n != H.n:
        raise ValueError("reference, Hamiltonian, and restriction disagree on n")
    if any(generator.n != H.n for generator in source):
        raise ValueError("generator and Hamiltonian qubit counts disagree")
    tolerances = {
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "leakage_tolerance": leakage_tolerance,
        "zero_tolerance": zero_tolerance,
    }
    if any(value < 0.0 for value in tolerances.values()):
        raise ValueError("mapping tolerances must be non-negative")
    if relative_tolerance == 0.0 and absolute_tolerance == 0.0:
        raise ValueError("relative_tolerance and absolute_tolerance cannot both be zero")
    if dense_spectrum_max_qubits < 0:
        raise ValueError("dense_spectrum_max_qubits must be non-negative")
    source_collisions = _generator_collisions(source)
    if source_collisions:
        pairs = ", ".join(f"{left!r}/{right!r}" for left, right in source_collisions)
        raise ValueError(f"source generator domain contains duplicates: {pairs}")

    transported = restriction.transport(
        hamiltonian=H,
        reference=reference,
        generators=(generator.mv for generator in source),
    )
    max_leakage = max(transported.generator_leakage, default=0.0)
    if max_leakage > leakage_tolerance:
        raise AssertionError(
            "mapping arm changed the physical generator domain: maximum fixed-sector "
            f"leakage is {max_leakage:.3e}"
        )
    annihilated = transported.annihilated_indices(tol=zero_tolerance)
    if annihilated:
        labels = ", ".join(source[index].label for index in annihilated)
        raise AssertionError(f"mapping arm annihilated declared generator(s): {labels}")
    mapped_generators = [
        Generator(generator.label, image)
        for generator, image in zip(source, transported.generators)
    ]
    mapped_collisions = _generator_collisions(mapped_generators)
    if mapped_collisions:
        pairs = ", ".join(f"{left!r}/{right!r}" for left, right in mapped_collisions)
        raise AssertionError(
            f"mapping arm collapsed distinct generators after restriction: {pairs}"
        )

    before = MatrixElementBank(reference, H, source)
    after = MatrixElementBank(
        transported.reference, transported.hamiltonian, mapped_generators
    )
    S_before, H_before = before.matrices()
    S_after, H_after = after.matrices()
    operator_error = 0.0
    operator_scale = 0.0
    for j in range(len(source)):
        for i in range(j + 1):
            for original, mapped in (
                (before.overlap_operator(i, j), after.overlap_operator(i, j)),
                (before.element_operator(i, j), after.element_operator(i, j)),
            ):
                expected = restriction.operator(original)
                operator_error = max(operator_error, (expected - mapped).norm_hs())
                operator_scale = max(operator_scale, expected.norm_hs())
    _assert_close(
        operator_error,
        operator_scale,
        relative_tolerance,
        absolute_tolerance,
        "transported element operators",
    )
    overlap_error = _max_abs(S_after - S_before)
    hamiltonian_error = _max_abs(H_after - H_before)
    _assert_close(
        overlap_error,
        _max_abs(S_before),
        relative_tolerance,
        absolute_tolerance,
        "projected overlap",
    )
    _assert_close(
        hamiltonian_error,
        _max_abs(H_before),
        relative_tolerance,
        absolute_tolerance,
        "projected Hamiltonian",
    )

    solved_before = before.solve(tau_s=tau_s, max_condition=max_condition)
    solved_after = after.solve(tau_s=tau_s, max_condition=max_condition)
    if solved_before.effective_rank != solved_after.effective_rank:
        raise AssertionError("mapping changed the retained projected rank")
    energies_before = np.asarray(solved_before.energies)
    energies_after = np.asarray(solved_after.energies)
    if energies_before.shape != energies_after.shape:
        raise AssertionError("mapping changed the number of retained Ritz values")
    ritz_error = _max_abs(energies_after - energies_before)
    _assert_close(
        ritz_error,
        _max_abs(energies_before),
        relative_tolerance,
        absolute_tolerance,
        "Ritz values",
    )
    condition_error = abs(
        solved_after.condition_number - solved_before.condition_number
    )
    _assert_close(
        condition_error,
        solved_before.condition_number,
        relative_tolerance,
        absolute_tolerance,
        "overlap condition number",
    )

    energy_before = complex((2 ** H.n) * H.trace_pairing(reference))
    energy_after = complex(
        (2 ** transported.n)
        * transported.hamiltonian.trace_pairing(transported.reference)
    )
    reference_error = abs(energy_after - energy_before)
    _assert_close(
        reference_error,
        abs(energy_before),
        relative_tolerance,
        absolute_tolerance,
        "reference energy",
    )

    resources_before = before.resources()
    resources_after = after.resources()
    pure_encoding = restriction.qubits_removed == 0
    if resources_before["basis_size"] != resources_after["basis_size"]:
        raise AssertionError(
            "mapping changed the declared basis size: "
            f"{resources_before['basis_size']} != {resources_after['basis_size']}"
        )
    if pure_encoding:
        for field in (
            "word_universe",
            "max_generator_support",
            "max_overlap_element_support",
            "max_hamiltonian_element_support",
            "hamiltonian_support",
        ):
            if resources_before[field] != resources_after[field]:
                raise AssertionError(
                    f"pure encoding changed invariant resource field {field}: "
                    f"{resources_before[field]} != {resources_after[field]}"
                )
        expected_words = _mapped_word_set(restriction, tuple(before.word_set()))
        if expected_words != after.word_set():
            raise AssertionError("pure encoding broke the Pauli-word universe bijection")

    dense_checked = H.n <= dense_spectrum_max_qubits
    spectrum_error = None
    if dense_checked:
        if pure_encoding:
            expected_spectrum = np.linalg.eigvalsh(to_matrix(H))
        else:
            expected_spectrum = np.linalg.eigvalsh(
                _fixed_sector_block(
                    restriction.rotate(H),
                    restriction.fixed_qubits,
                    restriction.signs,
                )
            )
        mapped_spectrum = np.linalg.eigvalsh(to_matrix(transported.hamiltonian))
        if expected_spectrum.shape != mapped_spectrum.shape:
            raise AssertionError("mapping changed the fixed-sector spectrum dimension")
        spectrum_error = _max_abs(mapped_spectrum - expected_spectrum)
        _assert_close(
            spectrum_error,
            _max_abs(expected_spectrum),
            relative_tolerance,
            absolute_tolerance,
            "fixed-sector spectrum",
        )

    return MappingInvariantReport(
        mapping_name=restriction.label or "unlabelled",
        source_qubits=H.n,
        mapped_qubits=transported.n,
        qubits_removed=restriction.qubits_removed,
        basis_size=len(source),
        word_universe_before=int(resources_before["word_universe"]),
        word_universe_after=int(resources_after["word_universe"]),
        effective_rank=int(solved_before.effective_rank),
        condition_number=float(solved_before.condition_number),
        max_overlap_matrix_error=overlap_error,
        max_hamiltonian_matrix_error=hamiltonian_error,
        max_operator_transport_error=float(operator_error),
        max_ritz_error=ritz_error,
        reference_energy_error=float(reference_error),
        max_generator_leakage=float(max_leakage),
        word_bijection_checked=pure_encoding,
        dense_spectrum_checked=dense_checked,
        max_spectrum_error=spectrum_error,
        relative_tolerance=float(relative_tolerance),
        absolute_tolerance=float(absolute_tolerance),
        leakage_tolerance=float(leakage_tolerance),
        zero_tolerance=float(zero_tolerance),
    )
