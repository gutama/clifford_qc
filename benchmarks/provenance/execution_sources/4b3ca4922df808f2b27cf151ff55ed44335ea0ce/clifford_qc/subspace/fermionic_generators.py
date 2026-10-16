"""Fermionic determinant and excitation generator families."""

from __future__ import annotations

from typing import Sequence

from ..multivector import MV
from .generator_core import Generator


def occupied_spin_orbitals(model) -> tuple[int, ...]:
    """Spin orbitals the model's reference determinant fills.

    Read from the reference ``Program``'s X gates rather than assumed to be the
    first ``n_electrons`` indices. A molecular Hartree-Fock determinant does
    fill a prefix; a lattice Neel reference fills every other spin orbital, and
    building excitations against the wrong occupied set silently produces
    generators that annihilate the reference.
    """
    occupied = []
    for operation in model.reference.ops:
        if getattr(operation, "name", None) == "X":
            occupied.extend(operation.qubits)
    return tuple(sorted(occupied))


def determinant_excitations(n_qubits: int, occupied: Sequence[int], *,
                           max_rank: int = 2, conserve_sz: bool = True
                           ) -> list[Generator]:
    """Symmetry-preserving excitations of a determinant, built natively.

    Anti-Hermitian singles ``c†_a c_i - h.c.`` and doubles
    ``c†_a c†_b c_j c_i - h.c.`` between the given occupied set and its
    complement, assembled from the package's own Jordan-Wigner operators -- so
    the §4.2 symmetry-preserving mode is available for lattice models without
    the chemistry extra, and for any reference determinant rather than only a
    prefix-filled one.

    ``conserve_sz`` keeps only excitations that preserve total ``S_z`` under the
    interleaved convention (even indices up, odd down): a single needs matching
    spins, and a double needs the same number of down spins on each side. The
    weaker parity test ``(i+j) % 2 == (a+b) % 2`` admits ``Delta S_z = +-2``
    doubles and is not sufficient -- the same trap ``models.chemistry``
    documents.
    """
    from ..fermion import c_op, cdag_op

    occupied = tuple(sorted(occupied))
    virtual = tuple(p for p in range(n_qubits) if p not in set(occupied))
    if max_rank not in (1, 2):
        raise ValueError("max_rank must be 1 or 2")

    def anti_hermitian(creations: Sequence[int], annihilations: Sequence[int]) -> MV:
        term = MV.scalar(n_qubits, 1.0)
        for p in creations:
            term = term * cdag_op(n_qubits, p)
        for p in reversed(annihilations):
            term = term * c_op(n_qubits, p)
        return term - term.dagger()

    out: list[Generator] = []
    for i in occupied:
        for a in virtual:
            if conserve_sz and (a - i) % 2 != 0:
                continue
            out.append(Generator(f"E({a}<-{i})", anti_hermitian([a], [i])))
    if max_rank < 2:
        return [g for g in out if not g.mv.is_zero(1e-12)]
    for index, i in enumerate(occupied):
        for j in occupied[index + 1:]:
            for a_index, a in enumerate(virtual):
                for b in virtual[a_index + 1:]:
                    if conserve_sz and (i % 2 + j % 2) != (a % 2 + b % 2):
                        continue
                    out.append(Generator(f"E({a},{b}<-{i},{j})",
                                         anti_hermitian([a, b], [j, i])))
    return [g for g in out if not g.mv.is_zero(1e-12)]


def fermionic_excitation_generators(n_qubits: int, n_electrons: int) -> list[Generator]:
    """The chemistry default: whole JW images of conserving excitations (§4.2).

    One multivector per particle-number- and S_z-conserving fermionic
    excitation, *not* split into words. A split word generally leaves the
    physical sector, and a subspace built from split words can lower its Ritz
    value by leaking into states with the wrong electron count -- which is why
    this, rather than :func:`pauli_orbit` over the qubit-ADAPT pool, is what
    chemistry runs should grow from.

    Needs the ``chemistry`` extra (OpenFermion + PySCF), imported here rather
    than at module scope so the A-CASE layer stays importable without it.
    """
    from ..models.chemistry import excitation_multivectors
    return [Generator(label, image.to_mv())
            for label, image in excitation_multivectors(n_qubits, n_electrons)]
