"""Independent occupation, spectral and letter-product MORE breadth oracles.

FCIDUMP text parsing and the earlier independently tested occupation/letter
Hamiltonian oracle are shared. No MORE driver, A-CASE selector, matrix-element
bank, native Pauli product, or native generalized eigensolver is called.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from benchmarks.phase17_ceo_oracle import hamiltonian, qe_coefficients


def matrices(spec):
    n = spec['n']
    basis = [word for word in range(2**n)
             if word.bit_count() == spec['n_electrons']
             and sum((1 if j % 2 == 0 else -1) * ((word >> (n - 1 - j)) & 1)
                     for j in range(n)) == round(2 * spec['sz'])]
    index = {word: i for i, word in enumerate(basis)}
    out = {}
    for op in spec['qe']:
        t = np.zeros((len(basis), len(basis)), complex)
        for word, column in index.items():
            if (any((word >> (n - 1 - j)) & 1 for j in op['creation'])
                    or any(not ((word >> (n - 1 - j)) & 1) for j in op['annihilation'])):
                continue
            target = word
            for j in op['creation'] + op['annihilation']:
                target ^= 1 << (n - 1 - j)
            row = index[target]
            t[row, column], t[column, row] = 1, -1
        out[op['label']] = t
    return basis, out


def decode(rows):
    return np.array([[complex(float.fromhex(real), float.fromhex(imag)) for real, imag in row]
                     for row in rows], complex)


def replay(h, references, spectra, labels, parameters):
    """Spectral exponentials and their adjoints for an observed parameter tuple."""
    states, unitaries = [references], []
    for label, angle in zip(labels, parameters):
        values, vectors = spectra[label]
        unitary = (vectors * np.exp(-1j * angle * values)) @ vectors.conj().T
        unitaries.append(unitary)
        states.append(unitary @ states[-1])
    adjoint = h @ states[-1]
    roots = references.shape[1]
    value = float(np.vdot(states[-1], adjoint).real / roots)
    gradient = []
    for j in reversed(range(len(labels))):
        values, vectors = spectra[labels[j]]
        derivative = vectors @ ((-1j * values)[:, None] * (vectors.conj().T @ states[j + 1]))
        gradient.append(float(2 * np.vdot(adjoint, derivative).real / roots))
        adjoint = unitaries[j].conj().T @ adjoint
    return value, list(reversed(gradient)), states[-1]


def root_readout(h, states):
    projected = states.conj().T @ h @ states
    projected = (projected + projected.conj().T) / 2
    return np.linalg.eigvalsh(projected), projected


@lru_cache(maxsize=65536)
def phase(a, b, n):
    value = 1 + 0j
    for j in range(n):
        x, y = (a >> (2 * j)) & 3, (b >> (2 * j)) & 3
        if x and y and x != y:
            value *= 1j if (x, y) in ((1, 2), (2, 3), (3, 1)) else -1j
    return value


def product(a, b, n):
    result = {}
    for left, ca in a.items():
        for right, cb in b.items():
            code = left ^ right
            result[code] = result.get(code, 0) + ca * cb * phase(left, right, n)
    return {code: value for code, value in result.items() if abs(value) > 1e-12}


def coefficients(spec, bits):
    n = spec['n']
    hf = sum(1 << (n - 1 - j) for j in spec['reference_occupied'])
    seeds = [{sum(1 << (2 * j) for j in range(n)
                  if (word ^ hf) & (1 << (n - 1 - j))): 1 + 0j} for word in bits]
    qe = {op['label']: qe_coefficients(op, n) for op in spec['qe']}
    rows = [(f'ref:{j}', seed) for j, seed in enumerate(seeds)]
    rows += [(f'{op["label"]}@ref:{j}', product(qe[op['label']], seed, n))
             for op in spec['qe'] for j, seed in enumerate(seeds)]
    # MatrixElementBank coalesces byte-identical operators, including labels.
    unique = []
    seen = set()
    for label, terms in rows:
        key = tuple(sorted(terms.items()))
        if key not in seen:
            seen.add(key)
            unique.append((label, terms))
    return qe, unique


def candidate_vectors(spec, bits, basis, ts):
    refs = np.eye(len(basis), dtype=complex)[:, [basis.index(word) for word in bits]]
    rows = [(f'ref:{j}', refs[:, j]) for j in range(len(bits))]
    rows += [(f'{op["label"]}@ref:{j}', ts[op['label']] @ refs[:, j])
             for op in spec['qe'] for j in range(len(bits))]
    _, operators = coefficients(spec, bits)
    by_label = dict(rows)
    return np.array([by_label[label] for label, _ in operators]).T


def generalized(s, h, options):
    norms = np.sqrt(np.diag(s).real)
    normalized = s / norms[:, None] / norms[None, :]
    values, vectors = np.linalg.eigh(normalized)
    keep = values > max(options['tau_s'], options['rel_tau'] * float(values.max()))
    whitening = vectors[:, keep] / np.sqrt(values[keep])[None, :] / norms[:, None]
    reduced = whitening.conj().T @ h @ whitening
    energies = np.linalg.eigvalsh((reduced + reduced.conj().T) / 2)
    condition = float(values[keep].max() / values[keep].min())
    return energies, whitening, condition


def candidate_score(h, retained, root_states, energies, candidate, options):
    norm = float(np.vdot(candidate, candidate).real)
    if norm <= 1e-24:
        return None
    fraction = max(0.0, 1 - float(np.sum(np.abs(retained.conj().T @ candidate)**2)) / norm)
    if fraction < options['min_orthogonality']:
        return None
    lowerings = []
    for energy, state in zip(energies, root_states.T):
        overlap = np.vdot(state, candidate)
        orthogonal = candidate - overlap * state
        gap = float(np.vdot(orthogonal, orthogonal).real)
        if gap / norm <= 1e-12:
            lowerings.append(0.0)
            continue
        orthogonal /= math.sqrt(gap)
        coupling = np.vdot(state, h @ orthogonal)
        second = float(np.vdot(orthogonal, h @ orthogonal).real)
        pencil = np.array([[energy, coupling], [coupling.conjugate(), second]], complex)
        lowerings.append(max(0.0, float(energy - np.linalg.eigvalsh(pencil)[0])))
    return float(np.mean(lowerings)), lowerings, fraction
