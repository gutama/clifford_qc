"""Independent exchange coefficients and fermionic occupation Hamiltonian.

Only FCIDUMP text parsing is shared with the package. No native Pauli product,
Jordan--Wigner construction, exchange pool or optimizer is called here.
"""
from itertools import product

import numpy as np

from clifford_qc.models.fcidump import read_fcidump


def multiply(left, right, n):
    out = {}
    for a, ca in left.items():
        for b, cb in right.items():
            phase = 1+0j
            for j in range(n):
                x, y = (a >> (2*j)) & 3, (b >> (2*j)) & 3
                if x and y and x != y:
                    phase *= 1j if (x, y) in ((1, 2), (2, 3), (3, 1)) else -1j
            out[a ^ b] = out.get(a ^ b, 0) + ca*cb*phase
    return {code: c for code, c in out.items() if abs(c) > 1e-12}


def ladder(n, j, creation, fermionic=False):
    parity = sum(3 << (2*k) for k in range(j)) if fermionic else 0
    return {parity | (1 << (2*j)): .5,
            parity | (2 << (2*j)): -.5j if creation else .5j}


def qe_coefficients(op, n):
    terms = {0: 1.}
    for j in op['creation']:
        terms = multiply(terms, ladder(n, j, True), n)
    for j in op['annihilation']:
        terms = multiply(terms, ladder(n, j, False), n)
    return {code: c-c.conjugate() for code, c in terms.items()
            if abs(c-c.conjugate()) > 1e-12}


def pool_coefficients(structural):
    n = structural['n']
    terms = {op['label']: qe_coefficients(op, n) for op in structural['qe']}
    for op in structural['ovp_ceo']:
        if op['parents']:
            a, b = (terms[label] for label in op['parents'])
            terms[op['label']] = {code: a.get(code, 0)+op['sign']*b.get(code, 0)
                                 for code in a.keys() | b.keys()
                                 if abs(a.get(code, 0)+op['sign']*b.get(code, 0)) > 1e-12}
    return terms


def _fermionic_action(word, sequence, n):
    sign = 1
    for j, creation in reversed(sequence):
        mask = 1 << (n-1-j)
        occupied = bool(word & mask)
        if occupied == creation:
            return None
        sign *= (-1)**(word >> (n-j)).bit_count()
        word ^= mask
    return word, sign


def hamiltonian(path, basis):
    """Sector H from occupation signs; H words from separate letter products."""
    data = read_fcidump(path)
    n = 2*data.n_orbitals
    index = {word: i for i, word in enumerate(basis)}
    h = np.eye(len(basis), dtype=complex)*data.core_energy
    words = {0: complex(data.core_energy)}
    monomials = []
    for p, q in product(range(data.n_orbitals), repeat=2):
        c = float(data.one_body[p, q])
        if abs(c) > 1e-12:
            for spin in (0, 1):
                monomials.append((c, [(2*p+spin, True), (2*q+spin, False)]))
    for p, q, r, s in product(range(data.n_orbitals), repeat=4):
        c = .5*float(data.two_body[p, q, r, s])
        if abs(c) > 1e-12:
            for spin, other in product((0, 1), repeat=2):
                monomials.append((c, [(2*p+spin, True), (2*r+other, True),
                                       (2*s+other, False), (2*q+spin, False)]))
    for coefficient, sequence in monomials:
        for word, column in index.items():
            acted = _fermionic_action(word, sequence, n)
            if acted is not None:
                target, sign = acted
                h[index[target], column] += coefficient*sign
        terms = {0: 1.}
        for j, creation in sequence:
            terms = multiply(terms, ladder(n, j, creation, fermionic=True), n)
        scaled = {code: coefficient*c for code, c in terms.items()
                  if abs(coefficient*c) > 1e-12}
        for code, c in scaled.items():
            words[code] = words.get(code, 0)+c
        words = {code: c for code, c in words.items() if abs(c) > 1e-12}
    return h, words
