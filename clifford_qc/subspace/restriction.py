"""One primitive for every Clifford-rotate-then-fix-qubits restriction.

Encoding changes (JW to BK or parity), the two-qubit reductions, ``Z2``
symmetry tapering, and contextual-subspace restriction are the same object: a
Clifford rotation, then a set of commuting stabilizer qubits fixed to ``+-1``.
They differ only in whether the fixing set is empty and in how the rotation is
built.

The reason this is one class rather than four call sites is congruence. A
restriction has to move the Hamiltonian, the reference, the generator pool, and
the observables *together*; projecting some of them and not the others produces
symmetry contamination and spurious subspace directions that still return
numbers. ``RestrictedProblem`` exists so the four objects cannot drift apart.

Two transports, deliberately named differently:

``operator``
    The tapering homomorphism. It preserves products, which is what makes the
    restricted operator's spectrum equal the original's on the fixed sector.
    It is *not* term-by-term: a word ``W`` and its partner ``W*Z_q`` land on
    the same restricted word and their coefficients merge, at the fixed sign.
    Used for the Hamiltonian, generators, and observables.
``state``
    The same homomorphism, plus the check that makes it meaningful for a
    density multivector: an in-sector state's terms pair up (``W`` and
    ``W*Z_q`` carry equal coefficients) and merge under the fix -- the same
    merge described above -- so the trace comes out at 1 by itself. It is
    *not* renormalized: a trace away from 1 means the reference was not in the
    sector being fixed, which is a finding about the caller's declared signs
    and not something to divide away. The check is on the full complex trace,
    so a phase error cannot pass by having the right real part.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..ir import PauliWord
from ..multivector import MV


__all__ = [
    "Restriction",
    "RestrictedProblem",
    "restricted_sector_operators",
]


def _letter(code: int, qubit: int) -> int:
    """0=I, 1=X, 2=Y, 3=Z on ``qubit`` of a packed word code."""
    return (code >> (2 * qubit)) & 3


def _delete_qubits(code: int, n: int, dropped: frozenset[int]) -> int:
    """Re-pack a word code with the ``dropped`` lanes removed, order kept."""
    out = 0
    position = 0
    for qubit in range(n):
        if qubit in dropped:
            continue
        out |= _letter(code, qubit) << (2 * position)
        position += 1
    return out


@dataclass(frozen=True)
class Restriction:
    """A Clifford rotation followed by fixing stabilizer qubits to ``+-1``.

    ``clifford`` is a ``bridges.stim_bridge.CliffordMap`` or ``None`` for the
    identity rotation. It is typed loosely on purpose: the Stim bridge is an
    optional extra, and a restriction with no rotation (plain tapering against
    already-single-qubit stabilizers) must not require it.
    """

    n: int
    clifford: object | None
    fixed_qubits: tuple[int, ...]
    signs: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or self.n < 1:
            raise ValueError("n must be a positive integer")
        if len(self.fixed_qubits) != len(self.signs):
            raise ValueError("fixed_qubits and signs must have equal length")
        if len(set(self.fixed_qubits)) != len(self.fixed_qubits):
            raise ValueError("fixed_qubits must be distinct")
        for qubit in self.fixed_qubits:
            if not isinstance(qubit, int) or not 0 <= qubit < self.n:
                raise ValueError(f"fixed qubit {qubit!r} out of range for n={self.n}")
        for sign in self.signs:
            if sign not in (1, -1):
                raise ValueError("every fixed sign must be +1 or -1")
        if self.clifford is not None and getattr(self.clifford, "n", self.n) != self.n:
            raise ValueError("clifford map acts on a different qubit count")
        if len(self.fixed_qubits) >= self.n:
            raise ValueError("cannot fix every qubit; nothing would remain")

    @classmethod
    def identity(cls, n: int) -> "Restriction":
        """The no-op restriction, useful as a control arm."""
        return cls(n=n, clifford=None, fixed_qubits=(), signs=())

    @classmethod
    def encoding(cls, clifford) -> "Restriction":
        """A pure change of encoding: rotate, fix nothing."""
        return cls(n=clifford.n, clifford=clifford, fixed_qubits=(), signs=())

    @property
    def n_restricted(self) -> int:
        return self.n - len(self.fixed_qubits)

    @property
    def _dropped(self) -> frozenset[int]:
        return frozenset(self.fixed_qubits)

    @property
    def qubits_removed(self) -> int:
        return len(self.fixed_qubits)

    def rotate(self, operator: MV) -> MV:
        """``U A U†`` on all ``n`` qubits, with no fixing."""
        operator = _as_mv(operator, self.n)
        if self.clifford is None:
            return operator.copy()
        terms: dict[int, complex] = {}
        for code, value in operator.terms.items():
            phase, image = self.clifford.conjugate(PauliWord(self.n, code))
            terms[image.code] = terms.get(image.code, 0.0) + phase * value
        return MV(self.n, {c: v for c, v in terms.items() if v != 0})

    def operator(self, operator: MV, *, require_commuting: bool = False) -> MV:
        """Transport an operator through the rotation and the fixing.

        Terms that anticommute with a fixed stabilizer are annihilated by the
        projection -- that is the structural rule an operator moving the state
        to an orthogonal sector vanishes under. Pass ``require_commuting=True``
        for objects that are supposed to commute with every stabilizer (the
        Hamiltonian, the symmetry generators themselves); a dropped term there
        means the declared symmetry is wrong, not that the operator leaks.
        """
        rotated = self.rotate(operator)
        restricted, dropped_norm = self._fix(rotated)
        if require_commuting and dropped_norm > 0.0:
            raise ValueError(
                "operator does not commute with every fixed stabilizer: "
                f"{dropped_norm:.3e} of its Hilbert-Schmidt norm anticommutes. "
                "Either the declared signs/stabilizers are wrong, or this "
                "object should be transported with require_commuting=False."
            )
        return restricted

    def state(self, reference: MV, *, tol: float = 1e-9) -> MV:
        """Transport a density multivector, checking it lies in the sector.

        No renormalization: for an in-sector state the merge of ``W`` with
        ``W*Z_q`` restores the trace on its own, so a shortfall is evidence
        about the declared signs rather than a scale to divide out.

        The comparison is on the full complex trace. Testing only the real part
        would let a trace of ``1 + 0.25i`` through, and a phase error in the
        conjugation is precisely one of the failures this primitive exists to
        catch, so it must not be the one that slips past the sector check.
        """
        restricted = self.operator(reference)
        trace = restricted.trace()
        if abs(trace - 1.0) > tol:
            raise ValueError(
                f"reference is not in the fixed sector: restricted trace {trace:.6f} "
                "!= 1. The declared signs disagree with the reference state, or a "
                "phase was lost in the rotation; infer the signs from the "
                "reference instead of assuming them."
            )
        return restricted

    def leakage(self, operator: MV) -> float:
        """Fraction of an operator's HS norm killed by the sector projection.

        Zero means the operator commutes with every fixed stabilizer; one means
        it is entirely off-sector and the restriction deletes it.
        """
        operator = _as_mv(operator, self.n)
        norm = operator.norm_hs()
        if norm <= 0.0:
            return 0.0
        _, dropped = self._fix(self.rotate(operator))
        return dropped / norm

    def _fix(self, rotated: MV) -> tuple[MV, float]:
        """Project onto the fixed signs and delete the fixed lanes.

        Returns the restricted operator and the Hilbert-Schmidt norm of the
        part the projection removed, so callers can tell "commutes" from
        "leaked" without a second pass.
        """
        dropped = self._dropped
        if not dropped:
            return rotated.copy(), 0.0
        kept: dict[int, complex] = {}
        dropped_weight = 0.0
        for code, value in rotated.terms.items():
            sign = 1
            anticommutes = False
            for qubit, qubit_sign in zip(self.fixed_qubits, self.signs):
                letter = _letter(code, qubit)
                if letter in (1, 2):          # X or Y anticommutes with Z_q
                    anticommutes = True
                    break
                if letter == 3:               # Z picks up the fixed eigenvalue
                    sign *= qubit_sign
            if anticommutes:
                dropped_weight += abs(value) ** 2
                continue
            image = _delete_qubits(code, self.n, dropped)
            kept[image] = kept.get(image, 0.0) + sign * value
        restricted = MV(
            self.n_restricted, {c: v for c, v in kept.items() if v != 0}
        )
        # The norm convention carries a 2**n factor, so the removed weight is
        # measured on the original register to stay comparable with norm_hs().
        return restricted, math.sqrt(2 ** self.n * dropped_weight)

    def transport(self, *, hamiltonian: MV, reference: MV,
                  generators=(), observables=()) -> "RestrictedProblem":
        """Move all four objects together, which is the point of the class."""
        return RestrictedProblem(
            restriction=self,
            hamiltonian=self.operator(hamiltonian, require_commuting=True),
            reference=self.state(reference),
            generators=tuple(self.operator(g) for g in generators),
            observables=tuple(self.operator(o) for o in observables),
            generator_leakage=tuple(self.leakage(g) for g in generators),
        )


@dataclass(frozen=True)
class RestrictedProblem:
    """The four transported objects, plus what the transport cost each one.

    ``generator_leakage[i]`` is the fraction of generator ``i`` the sector
    projection removed. A generator that comes back zero (leakage 1.0) is not
    an error -- it is the restriction correctly reporting that the candidate
    only acted outside the sector -- but it must not silently enter a basis, so
    ``surviving_generators`` is the accessor a selector should use.
    """

    restriction: Restriction
    hamiltonian: MV
    reference: MV
    generators: tuple[MV, ...]
    observables: tuple[MV, ...]
    generator_leakage: tuple[float, ...]

    @property
    def n(self) -> int:
        return self.restriction.n_restricted

    def surviving_generators(self, *, tol: float = 1e-12) -> tuple[MV, ...]:
        return tuple(g for g in self.generators if not g.is_zero(tol))

    def annihilated_indices(self, *, tol: float = 1e-12) -> tuple[int, ...]:
        return tuple(i for i, g in enumerate(self.generators) if g.is_zero(tol))


def restricted_sector_operators(restriction: Restriction, *,
                                spin_ordering="interleaved") -> tuple[MV, MV]:
    """``(N, S_z)`` transported through a restriction.

    ``fermion.total_number_op`` and ``total_sz_op`` build ``n_j = (I - Z_j)/2``
    directly, which identifies the computational basis with the occupation
    basis -- true under Jordan-Wigner and false under BK or parity. Every
    consumer of ``subspace.symmetry`` inherits that assumption, and under a
    non-JW encoding they do not raise, they return plausible wrong numbers.

    Transporting the two operators through the same restriction that moved the
    Hamiltonian is the fix: the number operator's image stays diagonal, so the
    sector diagnostics remain meaningful in the restricted register instead of
    being silently applied to operators they no longer describe.
    """
    from ..fermion import total_number_op, total_sz_op

    number = total_number_op(restriction.n)
    sz = total_sz_op(restriction.n, spin_ordering=spin_ordering)
    return (
        restriction.operator(number, require_commuting=True),
        restriction.operator(sz, require_commuting=True),
    )


def _as_mv(operator, n: int) -> MV:
    operator = getattr(operator, "mv", operator)
    if not isinstance(operator, MV):
        raise TypeError(f"expected a multivector, got {type(operator).__name__}")
    if operator.n != n:
        raise ValueError(
            f"operator acts on {operator.n} qubits, restriction expects {n}"
        )
    return operator
