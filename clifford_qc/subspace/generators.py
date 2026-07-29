"""Generator families for the A-CASE basis hierarchy (research plan §4.2).

A generator ``A_i`` is an ``MV``; the basis state it names,
``|phi_i> = A_i|psi>``, is *never prepared*. Every projected matrix element is
an expectation on the single reference state,

    S_ij = <psi|A_i' A_j|psi>,      H_ij = <psi|A_i' H A_j|psi>,

so a generator is a label attached to a multivector, not a circuit. The levels
below are the ones the plan names; which of them a run uses is an explicit
input to :func:`~clifford_qc.subspace.solver.solve_subspace`, since Phase 1 is
fixed-basis (adaptive growth is Phase 3).

Grade is deliberately *not* a selection criterion here: it is not a good
quantum number for Jordan-Wigner-dressed Hamiltonians, so the families are
organized by operator response (Pauli orbit, commutator, Krylov), not by
Clifford grade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..ir import PauliSum, PauliWord
from ..measurement.bank import CommutatorBank
from ..multivector import MV


@dataclass(frozen=True)
class Generator:
    """One labelled basis direction ``A_i``, whose state ``A_i|psi>`` stays virtual."""

    label: str
    mv: MV

    @property
    def n(self) -> int:
        return self.mv.n

    def support(self) -> int:
        """Number of Pauli words in ``A_i`` -- the ``S_A`` term of the §6 accounting."""
        return self.mv.nnz()


def _to_mv(obj, n: int | None = None) -> MV:
    if isinstance(obj, MV):
        return obj
    if isinstance(obj, (PauliWord, PauliSum)):
        return obj.to_mv()
    if hasattr(obj, "word"):  # PoolOperator and friends
        return obj.word.to_mv()
    raise TypeError(f"cannot read a generator multivector from {type(obj).__name__}")


def _label_of(obj, index: int) -> str:
    for attr in ("label", "name"):
        value = getattr(obj, attr, None)
        if isinstance(value, str):
            return value
    return f"A{index}"


def as_generators(items: Iterable) -> list[Generator]:
    """Coerce a heterogeneous sequence into ``Generator``s.

    Accepts ``Generator``, ``MV``, ``PauliWord``, ``PauliSum``, ``PoolOperator``,
    and ``(label, one-of-those)`` pairs, so a caller can mix a hand-built
    multivector with a pool word without wrapping either by hand. Labels are
    what the result reports, so they must survive to the record.
    """
    out: list[Generator] = []
    for index, item in enumerate(items):
        if isinstance(item, Generator):
            out.append(item)
            continue
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            label, payload = item
            out.append(Generator(label, _to_mv(payload)))
            continue
        out.append(Generator(_label_of(item, index), _to_mv(item)))
    if not out:
        raise ValueError("no generators given")
    n = out[0].n
    if any(g.n != n for g in out):
        raise ValueError("generators live in different algebras")
    return out


def identity_generator(n: int) -> Generator:
    """Level 0: the reference state itself, ``A = 1``."""
    return Generator("I", MV.scalar(n, 1.0))


def pauli_orbit(words: Sequence) -> list[Generator]:
    """Level 1: ``P_j|psi>``, the tangent directions of the Pauli-rotor ansatz at theta=0.

    This is the *word-level benchmark mode* of §4.2 -- one generator per Pauli
    word, directly comparable with the qubit-ADAPT pools. For chemistry the
    symmetry-preserving mode (whole JW images of particle-number- and
    S_z-conserving excitations, kept as one multivector) is the physically
    meaningful default; see ``models.chemistry.excitation_multivectors``.
    """
    out = []
    for index, item in enumerate(words):
        word = item.word if hasattr(item, "word") else item
        label = getattr(item, "label", None) or getattr(word, "label", f"P{index}")
        out.append(Generator(label, _to_mv(word)))
    return out


def commutator_response(hamiltonian: PauliSum, words: Sequence,
                        *, drop_zero: bool = True) -> list[Generator]:
    """Level 2: ``G_j|psi>`` with ``G_j = -i/2 [H, P_j]``.

    The rows are exactly the ones ``CommutatorBank`` already materializes for
    ADAPT selection, reused here as basis directions rather than as scores --
    the same sparse operator, a different question asked of it. Its rows are
    ``-i sum_{k: {W_k,P}=0} h_k W_k P``, which is ``-i/2 [H, P_j]`` term by
    term; the overall scale is irrelevant to the subspace, which normalizes
    every generator before thresholding.

    ``drop_zero`` discards words that commute with ``H``: their response
    direction is the zero operator, which contributes an empty row rather than
    a direction.
    """
    bank = CommutatorBank(hamiltonian, [w.word if hasattr(w, "word") else w
                                        for w in words])
    out = []
    for label, row in zip(bank.labels, bank.coeffs):
        if drop_zero and not row:
            continue
        out.append(Generator(f"G[{label}]", MV(bank.n, dict(row))))
    return out


def krylov_response(hamiltonian, order: int) -> list[Generator]:
    """Level 3: ``H^k|psi>`` for ``k = 1..order``.

    One candidate family among several, not the organizing principle: the
    powers are kept because they are the reference construction every quantum
    Krylov method uses, so a run can price A-CASE against them under the same
    §6 accounting.
    """
    if order < 1:
        raise ValueError("Krylov order must be at least 1")
    H = hamiltonian if isinstance(hamiltonian, MV) else hamiltonian.to_mv()
    out = []
    power = MV.scalar(H.n, 1.0)
    for k in range(1, order + 1):
        power = power * H
        out.append(Generator(f"H^{k}", power.copy()))
    return out


def response_hierarchy(hamiltonian: PauliSum, words: Sequence, *,
                       krylov_order: int = 0, include_identity: bool = True,
                       include_pauli: bool = True,
                       include_commutator: bool = True) -> list[Generator]:
    """Levels 0-3 stacked in order, deduplicated by nothing -- the caller's choice.

    Convenience for the fixed-basis studies: the nested prefixes of the
    returned list are the nested bases whose Ritz values must be monotone
    non-increasing.
    """
    out: list[Generator] = []
    if include_identity:
        out.append(identity_generator(hamiltonian.n))
    if include_pauli:
        out.extend(pauli_orbit(words))
    if include_commutator:
        out.extend(commutator_response(hamiltonian, words))
    if krylov_order:
        out.extend(krylov_response(hamiltonian, krylov_order))
    return out
