"""Spin-free excitation pools and real Abelian point-group selection.

Operators use the native complex Cl(2n, C)/Jordan--Wigner representation.
Spatial orbital p owns one alpha and one beta spin orbital: qubits (2p, 2p+1)
under the default ``"interleaved"`` ordering, or (p, n_spatial + p) under
``"blocked"``. Every API here takes that ordering explicitly rather than
inferring it from the operators, which carry no record of it. Each
spatial-orbital irrep is an explicit nonnegative bit character of (Z2)^k:
tensor products are XOR and zero is the identity. Arbitrary program-specific
irrep numbers are not that convention.

Spin-free transfers E_ai = sum_sigma c†_(a,sigma) c_(i,sigma) commute with N,
Sz and both spin ladders. Their products and anti-Hermitian parts therefore
preserve total spin on any reference; they do not purify a spin-mixed state.
"""
from __future__ import annotations

import math
from itertools import combinations_with_replacement
from numbers import Integral, Real
from typing import Iterable, Mapping, Sequence

from ..backends.sector_statevector import _spin_sites
from ..fermion import c_op, cdag_op
from ..ir import PauliSum
from ..multivector import MV
from .generator_core import Generator, as_generators, to_mv

SPIN_ORDERINGS = ("interleaved", "blocked")


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _tolerance(value) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError("hamiltonian_atol must be a finite real number >= 0")
    return float(value)


def _irreps(orbital_irreps: Sequence[int]) -> tuple[int, ...]:
    labels = tuple(_integer(value, "orbital irrep") for value in orbital_irreps)
    if not labels:
        raise ValueError("one irrep character per spatial orbital is required")
    return labels


def _spin_pairs(n_spatial: int, spin_ordering) -> tuple[tuple[int, int], ...]:
    """(alpha, beta) qubits of each spatial orbital under a named ordering."""
    if not isinstance(spin_ordering, str) or spin_ordering not in SPIN_ORDERINGS:
        raise ValueError("spin_ordering must be 'interleaved' or 'blocked'")
    up, down = _spin_sites(2 * n_spatial, spin_ordering)
    return tuple(zip(up, down))


def _character_masks(irreps: tuple[int, ...],
                     pairs: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    """One word-code mask per character bit.

    Mask b holds the low letter bit of both spin orbitals of every spatial
    orbital whose label carries bit b. All-identity labels give no masks.
    """
    masks = []
    for bit in range(max(irreps).bit_length()):
        mask = 0
        for label, qubits in zip(irreps, pairs):
            if label >> bit & 1:
                for qubit in qubits:
                    mask |= 1 << (2 * qubit)
        masks.append(mask)
    return tuple(masks)


def _word_character(code: int, masks: tuple[int, ...]) -> int:
    # Letters take two bits per qubit (I=00, X=01, Y=10, Z=11). X and Y change
    # occupation and are exactly the letters whose two bits differ. Two flips in
    # one spatial orbital contribute its label twice and cancel.
    flips = code ^ (code >> 1)
    character = 0
    for bit, mask in enumerate(masks):
        if (flips & mask).bit_count() & 1:
            character |= 1 << bit
    return character


def _largest_violation(terms: Mapping[int, complex], masks: tuple[int, ...],
                       atol: float) -> float:
    """Largest |coefficient| above ``atol`` on a word with nonzero character."""
    if not masks:
        return 0.0
    return max((abs(value) for code, value in terms.items()
                if abs(value) > atol and _word_character(code, masks)), default=0.0)


def _require_register(n: int, n_spatial: int) -> None:
    if n != 2 * n_spatial:
        raise ValueError("operators require two spin orbitals (alpha, beta) per spatial irrep")


def _check_hamiltonian(hamiltonian, n_spatial: int, masks: tuple[int, ...],
                       atol: float) -> None:
    if hamiltonian is None:
        return
    # MV and PauliSum share the word-code keys, so neither is converted.
    operator = (hamiltonian if isinstance(hamiltonian, (MV, PauliSum))
                else to_mv(hamiltonian))
    _require_register(operator.n, n_spatial)
    worst = _largest_violation(operator.terms, masks, atol)
    if worst:
        raise ValueError(
            "Hamiltonian does not preserve the declared point-group characters: "
            f"largest forbidden |coefficient| is {worst:.3g} (hamiltonian_atol={atol:g})")


def closed_shell_occupied_spatial(occupied_spin_orbitals: Iterable[int], n_spatial: int, *,
                                  spin_ordering: str = "interleaved") -> tuple[int, ...]:
    """Doubly occupied spatial orbitals of a closed-shell determinant.

    Read from the determinant's actual occupied spin orbitals, so a non-prefix
    (non-aufbau) reference is honoured rather than assumed away. A determinant
    with any singly occupied spatial orbital is rejected: the spin-free pool
    spans singlet CISD only on a closed-shell reference.
    """
    count = _integer(n_spatial, "n_spatial", minimum=1)
    pairs = _spin_pairs(count, spin_ordering)
    occupied = [_integer(q, "occupied spin orbital") for q in occupied_spin_orbitals]
    if len(set(occupied)) != len(occupied) or any(q >= 2 * count for q in occupied):
        raise ValueError("occupied spin orbitals must be distinct and in range")
    filled = set(occupied)
    doubly = tuple(p for p, (up, down) in enumerate(pairs) if up in filled and down in filled)
    if 2 * len(doubly) != len(filled):
        raise ValueError("reference is not closed shell: every occupied spatial "
                         "orbital must hold both spins")
    return doubly


def point_group_filter(generators: Iterable, orbital_irreps: Sequence[int], *,
                       hamiltonian=None, hamiltonian_atol: float = 0.0,
                       spin_ordering: str = "interleaved") -> list[Generator]:
    """Keep whole operators that commute with all declared irrep parities.

    The symmetry for character bit b is the product of Z over both spin
    orbitals of each spatial orbital whose label has bit b; ``spin_ordering``
    names which qubits those are and must match how the operators were built.
    This tests global operator commutation, rather than whether a direction
    happens to vanish on one determinant. Input order and Generator objects
    are preserved; mixed-character operators are rejected whole, never
    projected or split into individual Pauli words.

    Supply ``hamiltonian`` to check its symmetry before removing directions.
    Its words with nonzero character must have |coefficient| <=
    ``hamiltonian_atol`` (default 0: none at all); raise it only to accept
    integral noise from an SCF run that did not enforce symmetry. Without a
    Hamiltonian, invariance under these supplied characters is a caller
    assumption. No spin conservation is added by this filter alone.
    """
    irreps = _irreps(orbital_irreps)
    masks = _character_masks(irreps, _spin_pairs(len(irreps), spin_ordering))
    _check_hamiltonian(hamiltonian, len(irreps), masks, _tolerance(hamiltonian_atol))
    items = list(generators)
    if not items:
        return []
    out = []
    for generator in as_generators(items):
        _require_register(generator.n, len(irreps))
        if not _largest_violation(generator.mv.terms, masks, 0.0):
            out.append(generator)
    return out


def spin_adapted_excitations(n_spatial: int, occupied_spatial: Sequence[int], *,
                            max_rank: int = 2, orbital_irreps: Sequence[int] | None = None,
                            hamiltonian=None, hamiltonian_atol: float = 0.0,
                            spin_ordering: str = "interleaved") -> list[Generator]:
    """Spin-scalar singles/doubles over a closed-shell orbital partition.

    ``occupied_spatial`` lists doubly occupied spatial orbitals; every other
    spatial orbital is virtual. :func:`closed_shell_occupied_spatial` reads it
    from a determinant's occupied spin orbitals. The output acts on
    ``2*n_spatial`` spin orbitals in ``spin_ordering`` and contains
    E_ai - E_ai† plus E_ai E_bj - (E_ai E_bj)† for unordered transfer pairs,
    including repeated indices. Both matchings of four distinct spatial
    indices are retained. These span the singlet CISD directions of the
    closed-shell determinant; they need not be orthogonal. Open-shell-specific
    spin couplings and generalized occupied/virtual rotations are outside
    this pool.

    Optional ``orbital_irreps`` are real Abelian XOR characters, one per
    spatial orbital. Singles require irrep(a) XOR irrep(i) = 0, while doubles
    require the XOR of all four labels to vanish. A forbidden single transfer
    can participate in an allowed double. All spin terms stay in one MV.
    ``hamiltonian``, if supplied, must preserve the declared point group to
    within ``hamiltonian_atol`` (see :func:`point_group_filter`).
    No chemistry extra is needed and no measurement/solver default is changed.
    """
    count = _integer(n_spatial, "n_spatial", minimum=1)
    rank = _integer(max_rank, "max_rank", minimum=1)
    if rank not in (1, 2):
        raise ValueError("max_rank must be 1 or 2")
    pairs = _spin_pairs(count, spin_ordering)
    occupied = tuple(_integer(p, "occupied spatial orbital") for p in occupied_spatial)
    if len(set(occupied)) != len(occupied) or any(p >= count for p in occupied):
        raise ValueError("occupied spatial orbitals must be distinct and in range")
    occupied = tuple(sorted(occupied))
    virtual = tuple(p for p in range(count) if p not in occupied)
    irreps = (0,) * count if orbital_irreps is None else _irreps(orbital_irreps)
    if len(irreps) != count:
        raise ValueError("one irrep character per spatial orbital is required")
    _check_hamiltonian(hamiltonian, count, _character_masks(irreps, pairs),
                       _tolerance(hamiltonian_atol))

    n = 2 * count
    transfers = [(a, i) for i in occupied for a in virtual]
    operators = {}
    out = []
    for a, i in transfers:
        # Build even forbidden singles: two of them may form an allowed double.
        term = sum((cdag_op(n, pairs[a][s]) * c_op(n, pairs[i][s]) for s in (0, 1)), MV(n))
        operators[a, i] = term
        if irreps[a] ^ irreps[i] == 0:
            out.append(Generator(f"SF({a}<-{i})", term - term.dagger()))
    if rank == 2:
        for (a, i), (b, j) in combinations_with_replacement(transfers, 2):
            if irreps[a] ^ irreps[i] ^ irreps[b] ^ irreps[j]:
                continue
            term = operators[a, i] * operators[b, j]
            out.append(Generator(f"SF({a}<-{i};{b}<-{j})", term - term.dagger()))
    return out


__all__ = ["SPIN_ORDERINGS", "closed_shell_occupied_spatial", "point_group_filter",
           "spin_adapted_excitations"]
