"""Particle-number and spin-sector diagnostics for subspace generators.

Two notions are deliberately kept separate.  ``sector_leakage`` is an
operator-global commutator test: passing it guarantees that a generator
preserves every ``(N, S_z)`` sector.  ``reference_sector_leakage`` asks the
weaker question A-CASE actually needs for a fixed reference: whether
``A|psi>`` remains in the target sector.  The latter is what permits a single
Pauli word to act as a symmetry-safe direction on a determinant even though
the abstract Pauli operator does not commute with particle number.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..multivector import MV
from .contracts import as_multivector
from .generator_core import as_generators


@lru_cache(maxsize=32)
def sector_operators(n: int) -> tuple[MV, MV]:
    from ..fermion import total_number_op, total_sz_op
    return total_number_op(n), total_sz_op(n)


def sector_leakage(generator) -> dict[str, float]:
    """Relative operator-level leakage out of the (N, Sz) sectors."""
    operator = getattr(generator, "mv", generator)
    A = as_multivector(operator)
    norm = A.norm_hs()
    if norm <= 0.0:
        return {"particle_number": 0.0, "sz": 0.0}
    number, sz = sector_operators(A.n)
    return {
        "particle_number": (A * number - number * A).norm_hs() / norm,
        "sz": (A * sz - sz * A).norm_hs() / norm,
    }


def infer_reference_sector(reference_state: MV, *, tol: float = 1e-10
                           ) -> tuple[int, float]:
    """Infer the sharp ``(N, S_z)`` eigenvalues carried by ``reference_state``.

    Inference is intentionally refused for a sector-mixed warm start.  Silently
    rounding an average particle number would turn a state-conditioned check
    into an undocumented projection convention.  Such callers must provide an
    explicit target sector instead.
    """
    trace = float(reference_state.trace().real)
    if not trace > 0.0:
        raise ValueError("reference state has non-positive trace")
    targets: list[float] = []
    for name, operator, spacing in zip(
            ("particle number", "S_z"), sector_operators(reference_state.n),
            (1.0, 0.5)):
        mean = float((operator * reference_state).trace().real / trace)
        second = float((operator * operator * reference_state).trace().real / trace)
        variance = max(0.0, second - mean * mean)
        target = spacing * round(mean / spacing)
        if variance > tol or abs(mean - target) > tol:
            raise ValueError(
                f"cannot infer a sharp {name} sector from the reference "
                f"(mean={mean:.12g}, variance={variance:.3g}); pass sector_target")
        targets.append(target)
    return int(round(targets[0])), float(targets[1])


@lru_cache(maxsize=64)
def _target_projector(n: int, n_electrons: int, sz: float) -> MV:
    from ..backends.sector_statevector import sector_projector
    return sector_projector(n, n_electrons, sz, spin_ordering="interleaved")


def reference_sector_leakage(generator, reference_state: MV, *,
                             sector_target: tuple[int, float] | None = None,
                             infer_tol: float = 1e-10) -> dict[str, float]:
    """State-conditioned weight outside the target ``(N, S_z)`` sector.

    The returned scalar is

    ``1 - <psi|A^dagger P_target A|psi>/<psi|A^dagger A|psi>``.

    It is zero exactly when the virtual basis state ``A|psi>`` is in-sector.
    The projector is used only as a small-instance validation/certification
    object; the A-CASE state remains virtual and no additional state
    preparation is introduced.
    """
    A = as_multivector(getattr(generator, "mv", generator))
    if reference_state.n != A.n:
        raise ValueError("generator and reference act on different qubit counts")
    target = (infer_reference_sector(reference_state, tol=infer_tol)
              if sector_target is None else sector_target)
    n_electrons, sz = int(target[0]), float(target[1])
    if not 0 <= n_electrons <= A.n:
        raise ValueError("target particle number must be in [0, n]")
    if abs(2.0 * sz - round(2.0 * sz)) > 1e-12:
        raise ValueError("target S_z must be an integer or half-integer")
    norm = float((A.dagger() * A * reference_state).trace().real)
    if norm <= 1e-15:
        raise ValueError("generator annihilates the reference state")
    projector = _target_projector(A.n, n_electrons, sz)
    weight = float((A.dagger() * projector * A * reference_state).trace().real / norm)
    # Exact Clifford/determinant paths often land a few ulps outside [0, 1].
    weight = min(1.0, max(0.0, weight))
    leakage = 0.0 if abs(1.0 - weight) <= 1e-13 else 1.0 - weight
    return {"target_sector": leakage}


def subspace_sector_certificate(reference_state: MV, generators, *,
                                sector_target: tuple[int, float] | None = None,
                                tau_s: float = 1e-12) -> dict[str, float | int]:
    """Worst target-sector leakage over every vector in a generator span.

    Let ``S_ij=<psi|A_i^dagger A_j|psi>`` and
    ``L_ij=<psi|A_i^dagger(1-P)A_j|psi>``.  The largest generalized eigenvalue
    of ``(L,S)`` is the maximum leakage attainable by a normalized linear
    combination.  This closes the gap left by checking candidate vectors one
    at a time when a nonzero leakage tolerance is allowed.
    """
    gens = as_generators(generators)
    if not gens:
        raise ValueError("sector certification needs at least one generator")
    target = (infer_reference_sector(reference_state)
              if sector_target is None else sector_target)
    projector = _target_projector(reference_state.n, int(target[0]), float(target[1]))
    complement = MV.scalar(reference_state.n, 1.0) - projector
    size = len(gens)
    overlap = np.zeros((size, size), dtype=complex)
    leakage = np.zeros((size, size), dtype=complex)
    for i, left in enumerate(gens):
        for j, right in enumerate(gens[i:], start=i):
            s_ij = (left.mv.dagger() * right.mv * reference_state).trace()
            l_ij = (left.mv.dagger() * complement * right.mv * reference_state).trace()
            overlap[i, j] = s_ij
            leakage[i, j] = l_ij
            if i != j:
                overlap[j, i] = np.conjugate(s_ij)
                leakage[j, i] = np.conjugate(l_ij)
    values, vectors = np.linalg.eigh(0.5 * (overlap + overlap.conj().T))
    largest = float(values[-1])
    keep = values > max(tau_s, tau_s * max(largest, 1.0))
    if not keep.any():
        raise ValueError("generator span has zero norm on the reference state")
    whitening = vectors[:, keep] / np.sqrt(values[keep])
    reduced = whitening.conj().T @ (0.5 * (leakage + leakage.conj().T)) @ whitening
    worst = float(np.linalg.eigvalsh(0.5 * (reduced + reduced.conj().T))[-1])
    worst = 0.0 if abs(worst) <= 1e-13 else min(1.0, max(0.0, worst))
    return {
        "target_particle_number": int(target[0]),
        "target_sz": float(target[1]),
        "basis_size": size,
        "overlap_rank": int(keep.sum()),
        "max_sector_leakage": worst,
        "min_sector_weight": 1.0 - worst,
    }


__all__ = [
    "sector_operators", "sector_leakage", "infer_reference_sector",
    "reference_sector_leakage", "subspace_sector_certificate",
]
