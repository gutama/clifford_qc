from __future__ import annotations

import math

from .multivector import MV
from .dense_reference import to_matrix
from .pauli_action import apply_pauli_sum
from .states import partial_transpose


def negativity(rho: MV, transposed: set[int]) -> float:
    import numpy as np
    ev = np.linalg.eigvalsh(to_matrix(partial_transpose(rho, transposed)))
    return float((sum(abs(x) for x in ev) - 1.0) / 2.0)


def vn_entropy(rho: MV, *, base: float = 2.0, tol: float = 1e-12) -> float:
    import numpy as np
    ev = np.linalg.eigvalsh(to_matrix(rho))
    vals = [max(float(x.real), 0.0) for x in ev if x > tol]
    if not vals:
        return 0.0
    log_base = math.log(base)
    return -sum(p * math.log(p) / log_base for p in vals)


def fidelity_pure(rho: MV, psi) -> float:
    import numpy as np
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    return float(np.vdot(psi, apply_pauli_sum(rho, psi)).real)


def trace_cyclicity_error(A: MV, B: MV, C: MV) -> float:
    return abs((A * B * C).trace() - (B * C * A).trace())


def fermionic_sector_diagnostics(rho: MV, n_electrons: int, *,
                                 target_sz: float = 0.0) -> dict[str, float]:
    """Particle-number and spin-sector diagnostics for interleaved JW modes.

    Qubit ``j`` represents an occupied spin orbital when its computational
    bit is one; even indices are spin-up and odd indices spin-down.  The
    returned moments are exact for ``rho`` and ``target_sector_weight`` is
    the probability of simultaneously observing ``N=n_electrons`` and
    ``S_z=target_sz``.

    This diagnostic is intentionally representation-level: a Pauli-word
    rotor derived from a conserving fermionic excitation need not itself
    commute with ``N`` or ``S_z`` once the generator is split into words.
    """
    if not (0 <= n_electrons <= rho.n):
        raise ValueError("n_electrons must be in [0, n]")
    if abs(2.0 * target_sz - round(2.0 * target_sz)) > 1e-12:
        raise ValueError("target_sz must be an integer or half-integer")
    if rho.n > 20:
        raise ValueError(
            "fermionic_sector_diagnostics enumerates 2**n basis states; "
            "intended for small n (<=20)"
        )
    # Only I/Z Pauli terms contribute to computational-basis probabilities.
    # Evaluate the diagonal directly instead of constructing 2**n projector
    # multivectors; the latter is needlessly expensive for the 8-qubit H4
    # diagnostic.
    diagonal_terms = []
    for code, coeff in rho.terms.items():
        z_sites = []
        diagonal = True
        for j in range(rho.n):
            letter = (code >> (2 * j)) & 3
            if letter in (1, 2):  # X/Y has zero computational diagonal
                diagonal = False
                break
            if letter == 3:
                z_sites.append(j)
        if diagonal:
            diagonal_terms.append((tuple(z_sites), float(coeff.real)))

    probs = {}
    for k in range(2 ** rho.n):
        bits = format(k, f"0{rho.n}b")
        probs[bits] = sum(
            coeff * (-1.0 if sum(bits[j] == "1" for j in sites) % 2 else 1.0)
            for sites, coeff in diagonal_terms)
    n_mean = n_second = sz_mean = sz_second = sector = 0.0
    for bits, prob in probs.items():
        # Numerical exact backends may leave tiny signed roundoff in a
        # projector probability.  Preserve normalization while clipping only
        # values at the floating-point noise scale.
        p = 0.0 if abs(prob) < 1e-14 else float(prob)
        number = bits.count("1")
        n_up = sum(bits[j] == "1" for j in range(0, rho.n, 2))
        n_down = sum(bits[j] == "1" for j in range(1, rho.n, 2))
        spin_z = 0.5 * (n_up - n_down)
        n_mean += p * number
        n_second += p * number * number
        sz_mean += p * spin_z
        sz_second += p * spin_z * spin_z
        if number == n_electrons and abs(spin_z - target_sz) < 1e-12:
            sector += p
    return {
        "particle_number_mean": n_mean,
        "particle_number_variance": max(0.0, n_second - n_mean * n_mean),
        "spin_z_mean": sz_mean,
        "spin_z_variance": max(0.0, sz_second - sz_mean * sz_mean),
        "target_sector_weight": min(1.0, max(0.0, sector)),
    }
