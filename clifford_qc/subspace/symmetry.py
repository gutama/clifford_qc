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


def _ordering_key(spin_ordering):
    """Hashable, canonical form of a spin-ordering argument for caching."""
    if isinstance(spin_ordering, str):
        return spin_ordering
    return tuple(spin_ordering)


@lru_cache(maxsize=32)
def _sector_operators_cached(n: int, ordering_key) -> tuple[MV, MV]:
    from ..fermion import total_number_op, total_sz_op
    return total_number_op(n), total_sz_op(n, spin_ordering=ordering_key)


def sector_operators(n: int, *, spin_ordering="interleaved") -> tuple[MV, MV]:
    """``(N, S_z)`` for one spin-orbital ordering convention.

    ``N`` is ordering-invariant; ``S_z`` is not, so the convention travels with
    every call rather than being fixed once at import.
    """
    return _sector_operators_cached(n, _ordering_key(spin_ordering))


def sector_leakage(generator, *, spin_ordering="interleaved") -> dict[str, float]:
    """Relative operator-level leakage out of the (N, Sz) sectors."""
    operator = getattr(generator, "mv", generator)
    A = as_multivector(operator)
    norm = A.norm_hs()
    if norm <= 0.0:
        return {"particle_number": 0.0, "sz": 0.0}
    number, sz = sector_operators(A.n, spin_ordering=spin_ordering)
    return {
        "particle_number": (A * number - number * A).norm_hs() / norm,
        "sz": (A * sz - sz * A).norm_hs() / norm,
    }


def infer_reference_sector(reference_state: MV, *, tol: float = 1e-10,
                           spin_ordering="interleaved") -> tuple[int, float]:
    """Infer the sharp ``(N, S_z)`` eigenvalues carried by ``reference_state``.

    Inference is intentionally refused for a sector-mixed warm start.  Silently
    rounding an average particle number would turn a state-conditioned check
    into an undocumented projection convention.  Such callers must provide an
    explicit target sector instead.

    Sharpness alone does not make the answer convention-free.  Every
    computational-basis determinant is a sharp eigenstate of ``S_z`` under
    *every* spin ordering, with a different eigenvalue for each, so this
    function cannot detect a mismatched convention and does not try: pass the
    ``spin_ordering`` the Hamiltonian was mapped with.
    """
    trace = float(reference_state.trace().real)
    if not trace > 0.0:
        raise ValueError("reference state has non-positive trace")
    targets: list[float] = []
    for name, operator, spacing in zip(
            ("particle number", "S_z"),
            sector_operators(reference_state.n, spin_ordering=spin_ordering),
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
def _target_projector_cached(n: int, n_electrons: int, sz: float,
                             ordering_key) -> MV:
    from ..backends.sector_statevector import sector_projector
    return sector_projector(n, n_electrons, sz, spin_ordering=ordering_key)


def _target_projector(n: int, n_electrons: int, sz: float,
                      spin_ordering="interleaved") -> MV:
    projector = _target_projector_cached(n, n_electrons, sz,
                                         _ordering_key(spin_ordering))
    if projector.nnz() == 0:
        raise ValueError(
            f"the (N={n_electrons}, S_z={sz}) sector is empty on {n} qubits "
            f"under {spin_ordering!r} spin ordering")
    return projector


def reference_sector_leakage(generator, reference_state: MV, *,
                             sector_target: tuple[int, float] | None = None,
                             spin_ordering="interleaved",
                             infer_tol: float = 1e-10) -> dict[str, float]:
    """State-conditioned weight outside the target ``(N, S_z)`` sector.

    The returned scalar is

    ``1 - <psi|A^dagger P_target A|psi>/<psi|A^dagger A|psi>``.

    It is zero exactly when the virtual basis state ``A|psi>`` is in-sector.
    The projector is used only as a small-instance validation/certification
    object; the A-CASE state remains virtual and no additional state
    preparation is introduced.

    ``spin_ordering`` fixes which spin-orbital convention ``S_z`` is read in,
    for both the inferred and the explicitly declared target.
    """
    A = as_multivector(getattr(generator, "mv", generator))
    if reference_state.n != A.n:
        raise ValueError("generator and reference act on different qubit counts")
    target = (infer_reference_sector(reference_state, tol=infer_tol,
                                     spin_ordering=spin_ordering)
              if sector_target is None else sector_target)
    n_electrons, sz = int(target[0]), float(target[1])
    if not 0 <= n_electrons <= A.n:
        raise ValueError("target particle number must be in [0, n]")
    if abs(2.0 * sz - round(2.0 * sz)) > 1e-12:
        raise ValueError("target S_z must be an integer or half-integer")
    norm = float((A.dagger() * A * reference_state).trace().real)
    if norm <= 1e-15:
        raise ValueError("generator annihilates the reference state")
    projector = _target_projector(A.n, n_electrons, sz, spin_ordering)
    weight = float((A.dagger() * projector * A * reference_state).trace().real / norm)
    # Exact Clifford/determinant paths often land a few ulps outside [0, 1].
    weight = min(1.0, max(0.0, weight))
    leakage = 0.0 if abs(1.0 - weight) <= 1e-13 else 1.0 - weight
    return {"target_sector": leakage}


def subspace_sector_certificate(reference_state: MV, generators, *,
                                sector_target: tuple[int, float] | None = None,
                                spin_ordering="interleaved",
                                tau_s: float = 1e-12) -> dict[str, float | int]:
    """Worst target-sector leakage over every vector in a generator span.

    Let ``S_ij=<psi|A_i^dagger A_j|psi>`` and
    ``L_ij=<psi|A_i^dagger(1-P)A_j|psi>``.  The largest generalized eigenvalue
    of ``(L,S)`` is the maximum leakage attainable by a normalized linear
    combination.  This closes the gap left by checking candidate vectors one
    at a time when a nonzero leakage tolerance is allowed.

    For a computational determinant, orthonormalize the action amplitudes
    directly and measure their outside-sector rows.  Forming ``L`` by
    subtracting two independently evaluated Gram matrices can amplify sparse
    product pruning into spurious leakage on nearly dependent directions.
    """
    gens = as_generators(generators)
    if not gens:
        raise ValueError("sector certification needs at least one generator")
    if any(generator.mv.n != reference_state.n for generator in gens):
        raise ValueError("generator and reference act on different qubit counts")
    target = (infer_reference_sector(reference_state, spin_ordering=spin_ordering)
              if sector_target is None else sector_target)
    size = len(gens)
    action = _determinant_action_basis(
        reference_state, gens, target, spin_ordering)
    if action is not None:
        basis, outside = action
        vectors, singular_values, _ = np.linalg.svd(basis, full_matrices=False)
        values = singular_values**2
        largest = float(values[0]) if values.size else 0.0
        keep = values > max(tau_s, tau_s * max(largest, 1.0))
        outside_basis = vectors[outside][:, keep]
        reduced = outside_basis.conj().T @ outside_basis
    else:
        overlap = np.zeros((size, size), dtype=complex)
        leakage = np.zeros((size, size), dtype=complex)
        projector = _target_projector(reference_state.n, int(target[0]),
                                      float(target[1]), spin_ordering)
        complement = MV.scalar(reference_state.n, 1.0) - projector
        for i, left in enumerate(gens):
            for j, right in enumerate(gens[i:], start=i):
                s_ij = (left.mv.dagger() * right.mv * reference_state).trace()
                l_ij = (left.mv.dagger() * complement * right.mv
                        * reference_state).trace()
                overlap[i, j] = s_ij
                leakage[i, j] = l_ij
                if i != j:
                    overlap[j, i] = np.conjugate(s_ij)
                    leakage[j, i] = np.conjugate(l_ij)
        values, vectors = np.linalg.eigh(0.5 * (overlap + overlap.conj().T))
        largest = float(values[-1])
        keep = values > max(tau_s, tau_s * max(largest, 1.0))
        whitening = vectors[:, keep] / np.sqrt(values[keep])
        reduced = whitening.conj().T @ (0.5 * (leakage + leakage.conj().T)) @ whitening
    if not keep.any():
        raise ValueError("generator span has zero norm on the reference state")
    worst = float(np.linalg.eigvalsh(0.5 * (reduced + reduced.conj().T))[-1])
    worst = 0.0 if abs(worst) <= 1e-13 else min(1.0, max(0.0, worst))
    return {
        "target_particle_number": int(target[0]),
        "target_sz": float(target[1]),
        "spin_ordering": (spin_ordering if isinstance(spin_ordering, str)
                          else list(spin_ordering)),
        "basis_size": size,
        "overlap_rank": int(keep.sum()),
        "max_sector_leakage": worst,
        "min_sector_weight": 1.0 - worst,
    }


def _determinant_action_basis(reference_state: MV, generators, target,
                              spin_ordering):
    """Action columns and outside-sector rows for an exact determinant.

    Only reached occupation words become rows.  Accumulate Pauli amplitudes
    before inspecting sector membership, preserving interference and avoiding
    MV product pruning.  Other references use the explicit projector above.
    """
    from ..backends import SectorStatevectorBackend
    from ..pauli_action import word_masks
    from ..states import ket_density

    trace = reference_state.trace()
    if trace.imag != 0.0 or trace.real <= 0.0:
        return None
    n = reference_state.n
    bits = "".join("1" if reference_state.terms.get(3 << (2 * j), 0).real < 0
                   else "0" for j in range(n))
    # A nearly pure determinant is not an exact one: its small remaining
    # components must still be included in the projector certificate.
    determinant = ket_density(n, bits)
    if reference_state.terms != {
            code: trace.real * coefficient
            for code, coefficient in determinant.terms.items()}:
        return None
    backend = SectorStatevectorBackend(
        n, int(target[0]), float(target[1]),
        spin_ordering=spin_ordering)
    source = int(bits or "0", 2)
    rows = {}
    for column, generator in enumerate(generators):
        for code, coefficient in generator.mv.terms.items():
            x_mask, z_mask, y_count = word_masks(n, code)
            destination = source ^ x_mask
            if destination not in rows:
                rows[destination] = np.zeros(len(generators), dtype=complex)
            phase = (1j)**y_count * (-1)**((source & z_mask).bit_count())
            rows[destination][column] += coefficient * phase
    basis = np.asarray(list(rows.values()), dtype=complex).reshape(-1, len(generators))
    basis *= np.sqrt(trace.real)
    sector = set(backend.basis.tolist())
    outside = np.asarray([word not in sector for word in rows], dtype=bool)
    return basis, outside


def project_reference_to_sector(reference_state: MV, *,
                                sector_target: tuple[int, float],
                                spin_ordering="interleaved",
                                min_weight: float = 0.0
                                ) -> tuple[MV, dict[str, float]]:
    """Post-select a reference onto the declared ``(N, S_z)`` sector.

    Returns ``(P rho P / tr(P rho), diagnostics)``.  This is exactly the state
    a sector measurement on the prepared reference leaves behind when the
    outcome is kept, which is why the retained weight is reported rather than
    discarded: it *is* the acceptance probability, so ``1/weight`` attempts are
    needed per accepted shot.  A reference that is 99.99% in sector costs
    1.0001x; one that is 60% in sector costs 1.67x, and only the caller can
    decide whether that is worth paying.

    ``shot_overhead`` is a retry factor on *state preparation executions
    downstream of the projection*, and nothing else.  Work already performed on
    the unprojected reference is untouched -- it happened before the sector
    measurement exists, so no rejected attempt can make it more expensive --
    and the factor must not be applied to counts of distinct measurement
    settings, which are structural and not shots.  Multiplying a setting count
    by a retry factor mixes units.

    Two costs are outside this function.  The non-demolition ``(N, S_z)``
    measurement that realizes the projection without collapsing the state onto
    a single determinant is not priced here, and neither is the re-preparation
    circuit itself; only the acceptance probability is returned.

    The target must be declared explicitly.  A warm start assembled from
    symmetry-breaking rotors is by construction sector-mixed, so there is no
    sharp value to infer and guessing one would silently choose which physics
    to keep.

    Projecting is what lets such a reference receive the reference-conditioned
    certificate at all: on the projected state the sector weight is one by
    construction, so :func:`subspace_sector_certificate` becomes meaningful
    rather than vacuously failing.
    """
    declared_n = sector_target[0]
    if int(declared_n) != declared_n:
        raise ValueError("target particle number must be an integer")
    n_electrons, sz = int(declared_n), float(sector_target[1])
    if not 0 <= n_electrons <= reference_state.n:
        raise ValueError("target particle number must be in [0, n]")
    if abs(2.0 * sz - round(2.0 * sz)) > 1e-12:
        raise ValueError("target S_z must be an integer or half-integer")
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError("min_weight must be in [0, 1]")
    trace = float(reference_state.trace().real)
    if not trace > 0.0:
        raise ValueError("reference state has non-positive trace")
    projector = _target_projector(reference_state.n, n_electrons, sz, spin_ordering)
    projected = projector * reference_state * projector
    weight = float(projected.trace().real) / trace
    # An exactly in-sector state can land a few ulps above one, which would
    # report negative leakage and a retry overhead below unity.  Clamp on the
    # same convention reference_sector_leakage uses.
    weight = min(1.0, max(0.0, weight))
    if weight <= 0.0:
        raise ValueError(
            f"reference has no weight in the (N={n_electrons}, S_z={sz}) sector")
    if weight < min_weight:
        raise ValueError(
            f"reference sector weight {weight:.6g} is below the declared "
            f"minimum {min_weight:.6g}")
    return projected * (1.0 / float(projected.trace().real)), {
        "target_particle_number": n_electrons,
        "target_sz": sz,
        "spin_ordering": (spin_ordering if isinstance(spin_ordering, str)
                          else list(spin_ordering)),
        "sector_weight": weight,
        "leakage_removed": 0.0 if abs(1.0 - weight) <= 1e-13 else 1.0 - weight,
        "shot_overhead": 1.0 if abs(1.0 - weight) <= 1e-13 else 1.0 / weight,
    }


__all__ = [
    "sector_operators", "sector_leakage", "infer_reference_sector",
    "reference_sector_leakage", "subspace_sector_certificate",
    "project_reference_to_sector",
]
