"""Bounded singlet projection in disjoint external spatial occupations."""
from __future__ import annotations

import numpy as np

from clifford_qc.fermion import c_op, cdag_op
from clifford_qc.ir import PauliSum
from clifford_qc.multivector import MV
from clifford_qc.subspace.orthogonal import OccupationPacket, canonical_phase


def spin_squared_operator(n):
    """Interleaved JW S², including both ladder orders and Sz²."""
    plus, z = MV(n), MV(n)
    for p in range(n // 2):
        a, b = 2 * p, 2 * p + 1
        plus = plus + cdag_op(n, a) * c_op(n, b)
        z = z + .5 * (cdag_op(n, a) * c_op(n, a) - cdag_op(n, b) * c_op(n, b))
    minus = plus.dagger()
    squared = z * z + .5 * (plus * minus + minus * plus)
    return PauliSum(n, squared.terms)


def spatial_pattern(word, external_mask, n):
    return sum((((word & external_mask) >> (n - 2 - 2 * p)) & 3).bit_count() * 3**p
               for p in range(n // 2))


def singlet_packets(words, amplitudes, excluded, external_mask, n, s2, *,
                    tolerance=1e-9, relative_tolerance=1e-8, spin_tolerance=1e-9, diagnostics=None):
    """Project each external spatial class onto S²=0 before normalization.

    The dense S² blocks and full vectors make this a small-sector classical
    reference. The cutoff is absolute plus relative to the complete external
    preconditioned residual, before any individual group is normalized.
    """
    words = np.asarray(words)
    raw = np.asarray(amplitudes, complex)
    s2 = np.asarray(s2, complex)
    if (words.ndim != 1 or raw.shape != words.shape or s2.shape != (len(words), len(words))
            or not np.isfinite(raw).all() or not np.isfinite(s2).all()
            or not np.isfinite([tolerance, relative_tolerance, spin_tolerance]).all()
            or tolerance <= 0 or relative_tolerance < 0 or spin_tolerance <= 0
            or not np.allclose(s2, s2.conj().T, atol=spin_tolerance, rtol=0)):
        raise ValueError("finite matching vectors, Hermitian S² and positive tolerances required")
    live = np.ones(len(words), bool)
    live[np.asarray(excluded, int)] = False
    if np.any(abs(raw[~live]) > tolerance):
        raise ValueError("corrections have support on the classical span")
    patterns = np.array([spatial_pattern(int(w), external_mask, n) for w in words])
    cutoff = max(tolerance, relative_tolerance * float(np.linalg.norm(raw[live])))
    if diagnostics is not None:
        diagnostics.update(norm_cutoff=cutoff, block_eigensolves=0)
    packets = []
    for pattern in sorted(set(patterns[live])):
        indices = np.flatnonzero(live & (patterns == pattern))
        if np.linalg.norm(raw[indices]) <= cutoff:
            continue
        outside = np.flatnonzero(~(live & (patterns == pattern)))
        if outside.size and np.max(abs(s2[np.ix_(outside, indices)])) > spin_tolerance:
            raise ValueError("spatial class must be invariant under S²")
        if diagnostics is not None:
            diagnostics["block_eigensolves"] += 1
        values, vectors = np.linalg.eigh(s2[np.ix_(indices, indices)])
        singlets = vectors[:, abs(values) <= spin_tolerance]
        projected = singlets @ (singlets.conj().T @ raw[indices])
        norm = float(np.linalg.norm(projected))
        if norm <= cutoff:
            continue
        vector = np.zeros(len(words), complex)
        vector[indices] = projected / norm
        if np.linalg.norm(s2 @ vector) > spin_tolerance:
            raise ValueError("packet is not a singlet")
        packets.append(OccupationPacket(int(pattern), canonical_phase(vector)))
    return tuple(packets)
