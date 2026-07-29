"""Sparse reference diagonalization, for ``n`` past dense ``eigh``.

``matrix.py`` builds the full ``2^n x 2^n`` array, which is the right oracle for
the algebra kernel and hopeless past ten or twelve qubits: a 12-qubit
Hamiltonian with a hundred terms would materialize a hundred 134 MB matrices to
add them together. A Pauli word needs none of that. It is a signed permutation
matrix -- exactly one nonzero per row -- so it can be written down directly:

    W = i^{n_Y} X^x Z^z,   (X^x Z^z)|i> = (-1)^{z.i} |i xor x>,

with ``x`` the bit mask of the word's X and Y positions and ``z`` the mask of
its Z and Y positions. Summing those gives a sparse Hamiltonian with at most
``(#terms) * 2^n`` nonzeros, and ``scipy.sparse.linalg.eigsh`` takes it from
there.

Reference tier only. This is the baseline the A-CASE subspace is priced
against, not a scaling path: it still stores a ``2^n``-dimensional vector, and
the sector-restricted spinor backend of Phase 6 is what removes that.
"""

from __future__ import annotations

import numpy as np

from .ir import PauliSum
from .multivector import MV, validate_word_code

_PHASE4 = (1 + 0j, 1j, -1 + 0j, -1j)


def word_masks(n: int, code: int) -> tuple[int, int, int]:
    """``(x_mask, z_mask, y_count)`` of a Pauli word, as basis-index bit masks.

    Qubit ``j`` is the leftmost label character and therefore the *most*
    significant bit of a computational basis index, matching
    ``matrix.code_to_matrix``'s ``kron`` order. Getting that backwards silently
    transposes the lattice, which is why the conversion is a named function
    with a test rather than an inline expression.
    """
    validate_word_code(n, code)
    x_mask = z_mask = y_count = 0
    for j in range(n):
        letter = (code >> (2 * j)) & 3
        bit = 1 << (n - 1 - j)
        if letter in (1, 2):  # X or Y
            x_mask |= bit
        if letter in (2, 3):  # Y or Z
            z_mask |= bit
        if letter == 2:
            y_count += 1
    return x_mask, z_mask, y_count


def _parity(values: np.ndarray, mask: int) -> np.ndarray:
    """Parity of ``popcount(values & mask)``, vectorized and numpy-version-agnostic.

    A fold rather than ``np.bitwise_count``, which needs numpy 2; the parity is
    all the sign needs, and the fold is exact for the int64 indices used here.
    """
    folded = values & mask
    for shift in (32, 16, 8, 4, 2, 1):
        folded = folded ^ (folded >> shift)
    return folded & 1


def to_sparse(operator):
    """CSR matrix of an ``MV`` or ``PauliSum`` without ever forming a dense one."""
    from scipy.sparse import coo_matrix

    mv = operator.to_mv() if isinstance(operator, PauliSum) else operator
    if not isinstance(mv, MV):
        raise TypeError(f"expected MV or PauliSum, got {type(operator).__name__}")
    dim = 2 ** mv.n
    index = np.arange(dim, dtype=np.int64)
    rows, cols, data = [], [], []
    for code, coeff in mv.terms.items():
        x_mask, z_mask, y_count = word_masks(mv.n, code)
        signs = np.where(_parity(index, z_mask), -1.0, 1.0)
        rows.append(index ^ x_mask)
        cols.append(index)
        data.append(coeff * _PHASE4[y_count & 3] * signs)
    if not rows:
        return coo_matrix((dim, dim), dtype=complex).tocsr()
    return coo_matrix((np.concatenate(data), (np.concatenate(rows),
                                              np.concatenate(cols))),
                      shape=(dim, dim), dtype=complex).tocsr()


def sector_indices(n: int, n_electrons: int | None = None,
                   sz: float | None = None) -> np.ndarray:
    """Computational basis indices with a given particle number and ``S_z``.

    Interleaved Jordan-Wigner convention (even spin orbitals up, odd down), and
    qubit ``j`` is bit ``n-1-j`` of the index, matching :func:`word_masks`.

    Why this exists: a Hubbard cluster written grand-canonically has its
    *global* ground state away from half filling unless a chemical potential is
    tuned, while A-CASE stays in whatever sector its reference occupies. A
    comparison against the global minimum would then be true but uninformative
    -- the sector energy is the number the method should be judged against.

    Still an ``O(2^n)`` enumeration of a dense index set; the ideal-basis
    backend of Phase 6 is what makes the sector the *storage* unit rather than
    a mask over the full space.
    """
    index = np.arange(2 ** n, dtype=np.int64)
    keep = np.ones(index.shape, dtype=bool)
    occupied = np.zeros(index.shape, dtype=np.int64)
    spin_sum = np.zeros(index.shape, dtype=np.float64)
    for j in range(n):
        bit = (index >> (n - 1 - j)) & 1
        occupied += bit
        spin_sum += bit * (0.5 if j % 2 == 0 else -0.5)
    if n_electrons is not None:
        keep &= occupied == int(n_electrons)
    if sz is not None:
        keep &= np.abs(spin_sum - float(sz)) < 1e-12
    return index[keep]


def sparse_ground_in_sector(operator, n_electrons: int | None = None,
                            sz: float | None = None, k: int = 1, **kwargs
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Lowest ``k`` eigenpairs restricted to a ``(N, S_z)`` sector.

    The restriction is a submatrix, valid only because the Hamiltonian commutes
    with both symmetries -- checked here, since a silent restriction of a
    non-conserving operator would return a meaningless number.

    Eigenvectors come back in the *full* ``2^n`` space (zero outside the
    sector), so they can be fed to the same observable machinery as any other
    state.
    """
    from scipy.sparse import identity
    from scipy.sparse.linalg import eigsh

    from .fermion import total_number_op, total_sz_op

    mv = operator.to_mv() if isinstance(operator, PauliSum) else operator
    for name, symmetry, target in (("particle number", total_number_op(mv.n), n_electrons),
                                   ("S_z", total_sz_op(mv.n), sz)):
        if target is not None and not (mv * symmetry - symmetry * mv).is_zero(1e-10):
            raise ValueError(f"operator does not conserve {name}; a sector "
                             "restriction would be meaningless")
    indices = sector_indices(mv.n, n_electrons, sz)
    if indices.size == 0:
        raise ValueError("the requested sector is empty")
    block = to_sparse(mv)[indices][:, indices]
    dimension = block.shape[0]
    if k >= dimension - 1:
        values, vectors = np.linalg.eigh(block.toarray())
        values, vectors = values[:k], vectors[:, :k]
    else:
        shift = spectral_bound(mv)
        shifted = block - shift * identity(dimension, dtype=complex, format="csr")
        values, vectors = eigsh(shifted, k=k, which="LM", **kwargs)
        order = np.argsort(values)
        values, vectors = values[order].real + shift, vectors[:, order]
    embedded = np.zeros((2 ** mv.n, vectors.shape[1]), dtype=complex)
    embedded[indices, :] = vectors
    return np.asarray(values).real, embedded


def spectral_bound(operator) -> float:
    """``sum_w |h_w| >= ||H||``: every Pauli word is a unitary of norm one."""
    mv = operator.to_mv() if isinstance(operator, PauliSum) else operator
    return float(sum(abs(coeff) for coeff in mv.terms.values()))


def sparse_ground(operator, k: int = 1, *, tol: float = 0.0,
                  maxiter: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Lowest ``k`` eigenpairs by Lanczos on the sparse form, values ascending.

    Solved as the *largest-magnitude* eigenpairs of ``H - sigma I`` with
    ``sigma = sum_w |h_w| >= ||H||``, which puts the whole spectrum in
    ``[-2 sigma, 0]`` and makes the ground state the extremal one ARPACK is
    most reliable about.

    The shift is not cosmetic. Asking ARPACK for ``which='SA'`` directly returns
    the *wrong* eigenvalue on a spectrum with few distinct values and a large
    ground-space degeneracy: the ``t = 0`` Hubbard cluster is diagonal with
    eigenvalues in ``{0, U, 2U, ...}`` and a 256-fold zero eigenspace, and
    ``which='SA'`` reports ``U``, converged and residual-free, on a matrix whose
    minimum is zero. A residual check cannot catch that -- ``U`` really is an
    eigenvalue -- so the fix has to be in how the problem is posed. Shifting is
    also cheap: no factorization, one extra diagonal.

    ``eigsh`` needs ``k < dim - 1``; smaller systems fall back to a dense solve
    of the same sparse matrix, so the function is total on toy inputs.
    """
    from scipy.sparse import identity
    from scipy.sparse.linalg import eigsh

    matrix = to_sparse(operator)
    dim = matrix.shape[0]
    if k >= dim - 1:
        values, vectors = np.linalg.eigh(matrix.toarray())
        return values[:k], vectors[:, :k]
    shift = spectral_bound(operator)
    shifted = matrix - shift * identity(dim, dtype=complex, format="csr")
    values, vectors = eigsh(shifted, k=k, which="LM", tol=tol, maxiter=maxiter)
    order = np.argsort(values)
    return values[order].real + shift, vectors[:, order]
