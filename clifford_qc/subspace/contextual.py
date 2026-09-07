"""Deterministic contextual-stabilizer restrictions for R4a.

The shared restriction primitive applies a Clifford rotation and then fixes
commuting single-qubit stabilizers. It does not choose those stabilizers, and
its ordinary transport method correctly refuses to project a Hamiltonian that
breaks an exact symmetry.

Contextual-subspace projection has a different contract. Its stabilizers are
artificial symmetries of a declared noncontextual sub-Hamiltonian, so terms of
the full Hamiltonian that anticommute with them are deliberately projected
away. This module supplies that missing boundary without weakening the exact
restriction API:

* candidates are Hamiltonian Pauli words visited in the canonical order
  (-abs(coefficient), packed_word_code);
* only independent commuting words for which the supplied reference is an
  eigenstate with the coefficient-minimising sign are admitted;
* a stabilizer tableau maps those signed words to +Z on the first fixed
  qubits; and
* contextual transport records the Hilbert--Schmidt fraction removed from the
  Hamiltonian instead of silently pretending the projection is exact.

This is the narrow, reference-conditioned construction frozen for R4a. It is
not a general implementation of every noncontextual-Hamiltonian heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from ..ir import PauliWord
from ..multivector import MV
from .contracts import as_multivector, check_reference
from .generator_core import as_generators
from .restriction import RestrictedProblem, Restriction


__all__ = [
    "ContextualProblem",
    "ContextualRestrictionPlan",
    "ContextualStabilizer",
    "ContextualStabilizerSelection",
    "compile_contextual_restriction",
    "project_contextual_problem",
    "select_contextual_stabilizers",
]


@dataclass(frozen=True)
class ContextualStabilizer:
    """One signed Hamiltonian word selected for contextual projection."""

    word: PauliWord
    eigenvalue: int
    coefficient: float
    visit_index: int

    def __post_init__(self) -> None:
        if self.eigenvalue not in (-1, 1):
            raise ValueError("a stabilizer eigenvalue must be +1 or -1")
        if not math.isfinite(self.coefficient) or self.coefficient == 0.0:
            raise ValueError("a stabilizer coefficient must be finite and nonzero")
        if not isinstance(self.visit_index, int) or self.visit_index < 0:
            raise ValueError("visit_index must be a non-negative integer")

    def as_dict(self) -> dict[str, object]:
        return {
            "word": self.word.label,
            "packed_word_code": self.word.code,
            "eigenvalue": self.eigenvalue,
            "coefficient": self.coefficient,
            "visit_index": self.visit_index,
        }


@dataclass(frozen=True)
class ContextualStabilizerSelection:
    """Auditable output of the frozen canonical stabilizer selection rule."""

    n: int
    requested_count: int
    stabilizers: tuple[ContextualStabilizer, ...]
    visited_terms: int
    skipped_reference_not_eigenstate: int
    skipped_reference_sign_mismatch: int
    skipped_anticommuting: int
    skipped_dependent: int

    def __post_init__(self) -> None:
        if len(self.stabilizers) != self.requested_count:
            raise ValueError("selection does not contain its requested stabilizer count")

    @property
    def selected_abs_coefficient(self) -> float:
        return float(sum(abs(row.coefficient) for row in self.stabilizers))

    def as_dict(self) -> dict[str, object]:
        return {
            "n_qubits": self.n,
            "requested_count": self.requested_count,
            "stabilizers": [row.as_dict() for row in self.stabilizers],
            "visited_terms": self.visited_terms,
            "skipped": {
                "reference_not_eigenstate": self.skipped_reference_not_eigenstate,
                "reference_sign_mismatch": self.skipped_reference_sign_mismatch,
                "anticommuting": self.skipped_anticommuting,
                "dependent": self.skipped_dependent,
            },
            "selected_abs_coefficient": self.selected_abs_coefficient,
        }


@dataclass(frozen=True)
class ContextualRestrictionPlan:
    """A selected stabilizer set and its compiled rotate-then-fix map."""

    selection: ContextualStabilizerSelection
    restriction: Restriction

    def __post_init__(self) -> None:
        if self.selection.n != self.restriction.n:
            raise ValueError("selection and restriction act on different qubit counts")
        if self.restriction.qubits_removed != self.selection.requested_count:
            raise ValueError("restriction fixes a different number of qubits")

    def as_dict(self) -> dict[str, object]:
        return {
            "selection": self.selection.as_dict(),
            "fixed_qubits": list(self.restriction.fixed_qubits),
            "fixed_signs_after_rotation": list(self.restriction.signs),
            "active_qubits": self.restriction.n_restricted,
        }


@dataclass(frozen=True)
class ContextualProblem:
    """Projected problem with the approximation cost kept explicit."""

    plan: ContextualRestrictionPlan
    problem: RestrictedProblem
    hamiltonian_removed_hs_fraction: float

    def __post_init__(self) -> None:
        fraction = self.hamiltonian_removed_hs_fraction
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0 + 1e-12:
            raise ValueError("Hamiltonian removed fraction must lie in [0, 1]")

    @property
    def annihilated_indices(self) -> tuple[int, ...]:
        return self.problem.annihilated_indices()


def _symplectic_vector(word: PauliWord) -> int:
    """Packed x | z << n vector used only for GF(2) independence."""
    x = z = 0
    for qubit in range(word.n):
        letter = word.letter(qubit)
        if letter in ("X", "Y"):
            x |= 1 << qubit
        if letter in ("Y", "Z"):
            z |= 1 << qubit
    return x | (z << word.n)


def _gf2_rank(rows: Iterable[int]) -> int:
    basis: dict[int, int] = {}
    for value in rows:
        row = int(value)
        while row:
            pivot = row.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = row
                break
            row ^= basis[pivot]
    return len(basis)


def _commutes(left: PauliWord, right: PauliWord) -> bool:
    if left.n != right.n:
        raise ValueError("commutation operands act on different qubit counts")
    anticommutes = 0
    for qubit in range(left.n):
        a, b = left.letter(qubit), right.letter(qubit)
        if a != "I" and b != "I" and a != b:
            anticommutes ^= 1
    return anticommutes == 0


def _word_expectation(reference: MV, word: PauliWord) -> complex:
    return (word.to_mv() * reference).trace()


def select_contextual_stabilizers(
    hamiltonian,
    reference: MV,
    count: int,
    *,
    tol: float = 1e-9,
) -> ContextualStabilizerSelection:
    """Select count reference-compatible Hamiltonian stabilizers.

    The reference condition is deliberately strong: the expectation must be
    within tol of +1 or -1, and that sign must minimize the candidate word's
    Hamiltonian coefficient. Consequently the same frozen reference can feed
    all four R4a arms. A term with a favourable non-extremal expectation is not
    promoted into a stabilizer.
    """
    H = as_multivector(hamiltonian)
    if not H.is_hermitian(tol):
        raise ValueError("contextual stabilizer selection requires a Hermitian Hamiltonian")
    if not isinstance(reference, MV):
        raise TypeError("reference must be an MV density operator")
    if reference.n != H.n:
        raise ValueError("reference and Hamiltonian disagree on n")
    check_reference(reference)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count < H.n:
        raise ValueError("count must be an integer in [1, n - 1]")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")

    ranked = sorted(
        (
            (code, float(complex(coefficient).real))
            for code, coefficient in H.terms.items()
            if code != 0 and abs(complex(coefficient)) > tol
        ),
        key=lambda row: (-abs(row[1]), row[0]),
    )
    selected: list[ContextualStabilizer] = []
    vectors: list[int] = []
    skipped = {
        "reference_not_eigenstate": 0,
        "reference_sign_mismatch": 0,
        "anticommuting": 0,
        "dependent": 0,
    }
    visited = 0
    for visit_index, (code, coefficient) in enumerate(ranked):
        visited += 1
        word = PauliWord(H.n, code)
        expectation = _word_expectation(reference, word)
        if abs(expectation.imag) > tol or not math.isclose(
            abs(expectation.real), 1.0, rel_tol=0.0, abs_tol=tol
        ):
            skipped["reference_not_eigenstate"] += 1
            continue
        reference_sign = 1 if expectation.real > 0.0 else -1
        target_sign = -1 if coefficient > 0.0 else 1
        if reference_sign != target_sign:
            skipped["reference_sign_mismatch"] += 1
            continue
        if any(not _commutes(word, row.word) for row in selected):
            skipped["anticommuting"] += 1
            continue
        vector = _symplectic_vector(word)
        if _gf2_rank([*vectors, vector]) == len(vectors):
            skipped["dependent"] += 1
            continue
        selected.append(
            ContextualStabilizer(
                word=word,
                eigenvalue=reference_sign,
                coefficient=coefficient,
                visit_index=visit_index,
            )
        )
        vectors.append(vector)
        if len(selected) == count:
            break

    if len(selected) != count:
        raise ValueError(
            f"requested {count} contextual stabilizers, found {len(selected)} "
            "under the frozen reference-compatible rule"
        )
    return ContextualStabilizerSelection(
        n=H.n,
        requested_count=count,
        stabilizers=tuple(selected),
        visited_terms=visited,
        skipped_reference_not_eigenstate=skipped["reference_not_eigenstate"],
        skipped_reference_sign_mismatch=skipped["reference_sign_mismatch"],
        skipped_anticommuting=skipped["anticommuting"],
        skipped_dependent=skipped["dependent"],
    )


def compile_contextual_restriction(
    hamiltonian,
    reference: MV,
    count: int,
    *,
    tol: float = 1e-9,
    label: str | None = None,
) -> ContextualRestrictionPlan:
    """Compile the selected signed stabilizers to +Z fixed qubits."""
    selection = select_contextual_stabilizers(
        hamiltonian, reference, count, tol=tol
    )
    try:
        import stim
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "compile_contextual_restriction requires the 'stim' optional dependency"
        ) from exc

    from ..bridges.stim_bridge import CliffordMap

    stabilizers = [
        stim.PauliString(
            ("+" if row.eigenvalue > 0 else "-") + row.word.label
        )
        for row in selection.stabilizers
    ]
    preparation = stim.Tableau.from_stabilizers(
        stabilizers,
        allow_underconstrained=True,
        allow_redundant=False,
    )
    rotation = CliffordMap(selection.n, preparation.inverse())
    restriction = Restriction(
        n=selection.n,
        clifford=rotation,
        fixed_qubits=tuple(range(count)),
        signs=(1,) * count,
        label=label or f"contextual-{count}",
    )

    for fixed, row in enumerate(selection.stabilizers):
        phase, image = rotation.conjugate(row.word)
        expected = PauliWord.from_label(
            "".join("Z" if qubit == fixed else "I"
                    for qubit in range(selection.n))
        )
        if image != expected or abs(phase - row.eigenvalue) > tol:
            raise AssertionError(
                f"compiled stabilizer {row.word.label} did not map to "
                f"{row.eigenvalue:+d}Z_{fixed}"
            )
    restriction.state(reference, tol=tol)
    return ContextualRestrictionPlan(selection=selection, restriction=restriction)


def project_contextual_problem(
    plan: ContextualRestrictionPlan,
    *,
    hamiltonian,
    reference: MV,
    generators,
    observables=(),
    tol: float = 1e-9,
) -> ContextualProblem:
    """Project all problem objects together and expose the approximation.

    Unlike Restriction.transport, this function intentionally permits
    Hamiltonian terms that anticommute with the artificial stabilizers. Their
    norm fraction is returned as a required field. Exact symmetry tapering
    continues to use Restriction.transport and retains its strict refusal.
    """
    H = as_multivector(hamiltonian)
    if H.n != plan.selection.n:
        raise ValueError("plan and Hamiltonian act on different qubit counts")
    if not H.is_hermitian(tol):
        raise ValueError("contextual projection requires a Hermitian Hamiltonian")
    if not isinstance(reference, MV):
        raise TypeError("reference must be an MV density operator")
    if reference.n != H.n:
        raise ValueError("reference and Hamiltonian disagree on n")

    generator_rows = tuple(as_generators(generators))
    observable_rows = tuple(as_multivector(item) for item in observables)
    restriction = plan.restriction
    problem = RestrictedProblem(
        restriction=restriction,
        hamiltonian=restriction.operator(H, require_commuting=False),
        reference=restriction.state(reference, tol=tol),
        generators=tuple(restriction.operator(row.mv) for row in generator_rows),
        observables=tuple(restriction.operator(row) for row in observable_rows),
        generator_leakage=tuple(restriction.leakage(row.mv) for row in generator_rows),
    )
    return ContextualProblem(
        plan=plan,
        problem=problem,
        hamiltonian_removed_hs_fraction=restriction.leakage(H),
    )
