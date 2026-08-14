"""Linear fermion encodings as explicit Clifford changes of qubit basis.

The package's native fermionic operators use Jordan--Wigner (JW) occupation
bits.  Parity and Bravyi--Kitaev (BK) are therefore implemented here as
invertible binary changes of those bits,

``y = A x  (mod 2)``,

realized by a CNOT-only :class:`~clifford_qc.ir.Program`.  Conjugating the JW
Hamiltonian, reference, generator pool, and observables by that one program is
an encoding change, not a new physical problem.  It must preserve spectra,
projected matrices, Ritz values, conditioning, and the number of Pauli words.

The ``+2q`` arms are deliberately different: their final two encoded bits are
the spin-up and total occupation parities, which are fixed to a declared
``(N, S_z)`` sector and deleted through the shared
:class:`clifford_qc.subspace.restriction.Restriction` primitive.  Only that
fix-and-delete step may reduce the word universe or qubit count.

Qubit and mode index ``0`` is the leftmost tensor factor used by the rest of
the package.  A matrix row is stored as an integer bit mask: bit ``j`` says
that encoded bit ``y_i`` contains occupation bit ``x_j``.  The BK rows use the
standard Fenwick-tree convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .ir import Program


__all__ = [
    "FermionEncoding",
    "fermion_encoding",
]


_BASE_ENCODINGS = {"jw", "parity", "bk"}
_REDUCED_ENCODINGS = {"parity+2q", "bk+2q"}


def _gf2_rank(rows: Sequence[int], n: int) -> int:
    """Rank of packed binary rows, using an independent integer elimination."""
    work = [int(row) for row in rows if row]
    rank = 0
    for column in range(n):
        pivot = next(
            (index for index in range(rank, len(work))
             if (work[index] >> column) & 1),
            None,
        )
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        for index in range(len(work)):
            if index != rank and ((work[index] >> column) & 1):
                work[index] ^= work[rank]
        rank += 1
        if rank == len(work):
            break
    return rank


def _base_rows(name: str, n: int) -> tuple[int, ...]:
    if name == "jw":
        return tuple(1 << index for index in range(n))
    if name == "parity":
        return tuple((1 << (index + 1)) - 1 for index in range(n))
    if name == "bk":
        rows = []
        for index in range(n):
            one_based = index + 1
            start = one_based - (one_based & -one_based)
            rows.append((1 << one_based) - (1 << start))
        return tuple(rows)
    raise ValueError(f"unknown base fermion encoding {name!r}")


def _spin_parity_rows(n: int, spin_ordering: str) -> tuple[int, int]:
    if n % 2:
        raise ValueError("two-qubit reduction requires an even spin-orbital count")
    if spin_ordering == "interleaved":
        spin_up = sum(1 << index for index in range(0, n, 2))
    elif spin_ordering == "blocked":
        spin_up = (1 << (n // 2)) - 1
    else:
        raise ValueError("spin_ordering must be 'interleaved' or 'blocked'")
    total = (1 << n) - 1
    return spin_up, total


def _sector_signs(n: int, n_electrons: int, sz: float) -> tuple[int, int]:
    if not isinstance(n_electrons, int) or n_electrons < 0:
        raise ValueError("n_electrons must be a non-negative integer")
    if n_electrons > n:
        raise ValueError("n_electrons cannot exceed the spin-orbital count")
    twice_sz = 2.0 * float(sz)
    n_alpha = 0.5 * (n_electrons + twice_sz)
    n_beta = n_electrons - n_alpha
    if not math.isclose(n_alpha, round(n_alpha), abs_tol=1e-12):
        raise ValueError("n_electrons and sz do not define an integral spin sector")
    if n_alpha < 0 or n_beta < 0:
        raise ValueError("n_electrons and sz define a negative spin population")
    if n_alpha > n // 2 or n_beta > n // 2:
        raise ValueError(
            "n_electrons and sz define a spin population larger than its orbital count"
        )
    return (-1 if int(round(n_alpha)) % 2 else 1,
            -1 if n_electrons % 2 else 1)


def _rows_with_fixed_parities(
    base: Sequence[int], n: int, spin_ordering: str
) -> tuple[int, ...]:
    """Complete the two parity rows with rows inherited from ``base``.

    Starting from the physical symmetry rows and greedily adding independent
    base-encoding rows makes the reduced parity and BK arms share the same
    fixed symmetries while retaining their distinct non-fixed coordinates.
    """
    spin_up, total = _spin_parity_rows(n, spin_ordering)
    independent = [spin_up, total]
    free: list[int] = []
    rank = _gf2_rank(independent, n)
    for row in base:
        candidate_rank = _gf2_rank([*independent, *free, row], n)
        if candidate_rank > rank:
            free.append(row)
            rank = candidate_rank
        if len(free) == n - 2:
            break
    rows = tuple([*free, spin_up, total])
    if len(rows) != n or _gf2_rank(rows, n) != n:
        raise AssertionError("failed to complete the parity symmetries to an encoding basis")
    return rows


def _elimination_operations(rows: Sequence[int], n: int) -> list[tuple[str, int, int]]:
    """Row operations taking ``rows`` to identity over GF(2)."""
    work = list(rows)
    operations: list[tuple[str, int, int]] = []
    for column in range(n):
        pivot = next(
            (row for row in range(column, n) if (work[row] >> column) & 1),
            None,
        )
        if pivot is None:
            raise ValueError("encoding matrix is singular over GF(2)")
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            operations.append(("swap", column, pivot))
        for row in range(n):
            if row != column and ((work[row] >> column) & 1):
                work[row] ^= work[column]
                operations.append(("add", column, row))
    if work != [1 << index for index in range(n)]:
        raise AssertionError("GF(2) elimination did not produce the identity")
    return operations


def _program_from_rows(rows: Sequence[int], n: int) -> Program:
    """Synthesize ``y=A x`` as CNOTs by reversing binary elimination."""
    program = Program(n)
    for kind, source, target in reversed(_elimination_operations(rows, n)):
        if kind == "add":
            program.clifford("CX", source, target)
        else:
            # A row swap is three CNOTs.  Keeping the network CNOT-only makes
            # the mapping cost explicit and avoids hiding a SWAP convention.
            program.clifford("CX", source, target)
            program.clifford("CX", target, source)
            program.clifford("CX", source, target)
    return program


def _base_program(name: str, n: int) -> Program:
    """Minimal standard networks for the three unreduced encodings."""
    program = Program(n)
    if name == "jw":
        return program
    if name == "parity":
        # Prefix parity: after step j, target j stores x_0 xor ... xor x_j.
        for index in range(1, n):
            program.clifford("CX", index - 1, index)
        return program
    if name == "bk":
        # In-place Fenwick-tree construction.  Parent nodes accumulate the
        # intervals encoded by their children.
        for index in range(n):
            parent = index | (index + 1)
            if parent < n:
                program.clifford("CX", index, parent)
        return program
    raise ValueError(f"unknown base fermion encoding {name!r}")


@dataclass(frozen=True)
class FermionEncoding:
    """One declared linear encoding and its optional two-qubit sector fix."""

    name: str
    n: int
    rows: tuple[int, ...]
    fixed_qubits: tuple[int, ...] = ()
    signs: tuple[int, ...] = ()
    spin_ordering: str = "interleaved"

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or self.n < 1:
            raise ValueError("n must be a positive integer")
        if len(self.rows) != self.n:
            raise ValueError("encoding matrix must have exactly n rows")
        limit = 1 << self.n
        if any(not isinstance(row, int) or not 0 <= row < limit for row in self.rows):
            raise ValueError("encoding rows must be n-bit non-negative integers")
        if _gf2_rank(self.rows, self.n) != self.n:
            raise ValueError("encoding matrix must be invertible over GF(2)")
        if len(self.fixed_qubits) != len(self.signs):
            raise ValueError("fixed_qubits and signs must have equal length")
        if len(set(self.fixed_qubits)) != len(self.fixed_qubits):
            raise ValueError("fixed_qubits must be distinct")
        if any(
            not isinstance(qubit, int) or not 0 <= qubit < self.n
            for qubit in self.fixed_qubits
        ):
            raise ValueError("fixed qubit lies outside the encoding register")
        if len(self.fixed_qubits) >= self.n:
            raise ValueError("an encoding cannot fix every qubit")
        if any(sign not in (-1, 1) for sign in self.signs):
            raise ValueError("fixed signs must be +1 or -1")

    @property
    def qubits_removed(self) -> int:
        return len(self.fixed_qubits)

    @property
    def n_restricted(self) -> int:
        return self.n - self.qubits_removed

    def program(self) -> Program:
        """Return a fresh CNOT-only program implementing the encoding matrix."""
        if not self.fixed_qubits:
            return _base_program(self.name, self.n)
        return _program_from_rows(self.rows, self.n)

    def encode_bits(self, occupation: Sequence[int] | str) -> tuple[int, ...]:
        """Classically evaluate ``y=A x`` for an occupation bit string."""
        raw = tuple(occupation)
        if len(raw) != self.n or any(bit not in (0, 1, "0", "1") for bit in raw):
            raise ValueError("occupation must contain exactly n binary entries")
        values = tuple(int(bit) for bit in raw)
        packed = sum(bit << index for index, bit in enumerate(values))
        return tuple((row & packed).bit_count() & 1 for row in self.rows)

    def restriction(self):
        """Build the shared Clifford-rotate-then-fix restriction.

        Stim remains an optional dependency, so the bridge import is local and
        the binary encoding matrices can be constructed and tested without it.
        """
        from .bridges.stim_bridge import CliffordMap
        from .subspace.restriction import Restriction

        if self.name == "jw" and not self.fixed_qubits:
            return Restriction.identity(self.n)
        clifford = CliffordMap.from_program(self.program())
        return Restriction(
            n=self.n,
            clifford=clifford,
            fixed_qubits=self.fixed_qubits,
            signs=self.signs,
        )


def fermion_encoding(
    name: str,
    n: int,
    *,
    n_electrons: int | None = None,
    sz: float | None = None,
    spin_ordering: str = "interleaved",
) -> FermionEncoding:
    """Construct one of the five predeclared R2 mapping arms.

    Accepted names are ``jw``, ``parity``, ``bk``, ``parity+2q``, and
    ``bk+2q``.  The reduced arms require a declared ``(n_electrons, sz)``
    sector; the signs are derived from that sector rather than guessed from a
    particular determinant.
    """
    normalized = str(name).lower()
    if normalized in _BASE_ENCODINGS:
        if n_electrons is not None or sz is not None:
            raise ValueError("sector arguments are only valid for +2q encodings")
        return FermionEncoding(
            name=normalized,
            n=n,
            rows=_base_rows(normalized, n),
            spin_ordering=spin_ordering,
        )
    if normalized not in _REDUCED_ENCODINGS:
        choices = sorted(_BASE_ENCODINGS | _REDUCED_ENCODINGS)
        raise ValueError(f"unknown fermion encoding {name!r}; choose from {choices}")
    if n < 4:
        raise ValueError("two-qubit reduction requires at least four spin orbitals")
    if n_electrons is None or sz is None:
        raise ValueError("+2q encodings require n_electrons and sz")
    signs = _sector_signs(n, n_electrons, sz)
    base_name = normalized.split("+", 1)[0]
    rows = _rows_with_fixed_parities(_base_rows(base_name, n), n, spin_ordering)
    return FermionEncoding(
        name=normalized,
        n=n,
        rows=rows,
        fixed_qubits=(n - 2, n - 1),
        signs=signs,
        spin_ordering=spin_ordering,
    )
