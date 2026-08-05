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

from typing import Sequence

from ..ir import PauliSum
from ..measurement.bank import CommutatorBank
from ..multivector import MV
from .generator_core import Generator, _scalar_free_key, _to_mv, as_generators


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


def compound_response(left: Sequence, right: Sequence | None = None, *,
                      max_generators: int | None = None,
                      max_support: int | None = None,
                      drop_scalar: bool = True) -> list[Generator]:
    """Level 4a: the compound directions ``A_i A_j |psi>``.

    ``P_i P_j`` and ``P_i G_j`` in the plan's notation -- pass the Pauli orbit
    as ``left`` and either itself or the commutator response as ``right``. The
    product of two generators is another generator, so nothing new is needed
    downstream; what is new is that the family is *quadratic*, which is why it
    is filtered rather than enumerated.

    Three filters, all of them resource statements rather than physics:

    - ``drop_scalar`` discards products proportional to the identity. ``P P =
      I`` for any Pauli word, so the diagonal of a Pauli-orbit square is pure
      identity and contributes the level-0 direction the basis already has.
    - ``max_support`` caps ``S_A``. A compound generator is as wide as the
      product of its factors, and §6 counts that width; a cap makes the ceiling
      explicit instead of discovering it as an unaffordable bank.
    - ``max_generators`` truncates the surviving list in its deterministic
      order (``left`` major, ``right`` minor), so a run's candidate pool has a
      declared size rather than one that depends on how many products happened
      to survive.

    Products are deduplicated up to a scalar factor: ``P_i P_j`` and
    ``P_j P_i`` differ by a sign for anticommuting words, and the subspace
    normalizes every generator, so keeping both would buy a duplicate column
    and a singular overlap matrix.
    """
    left_gens = as_generators(left)
    right_gens = left_gens if right is None else as_generators(right)
    if not left_gens or not right_gens:
        return []
    n = left_gens[0].n
    for generator in (*left_gens, *right_gens):
        if generator.n != n:
            raise ValueError("compound factors act on different qubit counts")

    out: list[Generator] = []
    seen: set[tuple] = set()
    for a in left_gens:
        for b in right_gens:
            product = a.mv * b.mv
            if product.is_zero():
                continue
            if drop_scalar and product.nnz() == 1 and 0 in product.terms:
                continue
            if max_support is not None and product.nnz() > max_support:
                continue
            key = _scalar_free_key(product)
            if key is None or key in seen:
                continue
            seen.add(key)
            out.append(Generator(f"{a.label}*{b.label}", product))
            if max_generators is not None and len(out) >= max_generators:
                return out
    return out


def response_hierarchy(hamiltonian: PauliSum, words: Sequence, *,
                       krylov_order: int = 0, include_identity: bool = True,
                       include_pauli: bool = True,
                       include_commutator: bool = True,
                       compound: bool = False,
                       max_compound: int | None = None,
                       configurations=None, model=None) -> list[Generator]:
    """Levels 0-4 stacked in order, deduplicated by nothing -- the caller's choice.

    Convenience for the fixed-basis studies: the nested prefixes of the
    returned list are the nested bases whose Ritz values must be monotone
    non-increasing. Levels 0-3 are on by default; level 4 is opt-in, because it
    is the only quadratic family here and its width is a §6 cost.
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
    if compound:
        orbit = pauli_orbit(words)
        out.extend(compound_response(orbit, orbit, max_generators=max_compound))
    if configurations:
        if model is None:
            raise ValueError("configurations need the model whose reference they "
                             "are measured against")
        out.extend(configuration_generators(model, configurations))
    return out


# Backward-compatible imports for callers that historically treated this
# module as the single generator namespace. New code should import each domain
# from ``configuration`` or ``fermionic_generators`` directly.
from .configuration import (  # noqa: E402
    configuration_generator,
    configuration_generators,
    configuration_haar_packets,
    determinant_program,
    state_sector,
)
from .fermionic_generators import (  # noqa: E402
    determinant_excitations,
    fermionic_excitation_generators,
    occupied_spin_orbitals,
)


__all__ = [
    "Generator", "as_generators", "identity_generator", "pauli_orbit",
    "commutator_response", "krylov_response", "compound_response",
    "response_hierarchy", "determinant_program", "configuration_generator",
    "configuration_generators", "configuration_haar_packets", "state_sector",
    "occupied_spin_orbitals", "determinant_excitations",
    "fermionic_excitation_generators",
]
