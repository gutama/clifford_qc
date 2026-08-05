from __future__ import annotations

import math

from .multivector import MV, validate_n, validate_word_code


def single_pauli_mats():
    import numpy as np
    return {
        0: np.eye(2, dtype=complex),
        1: np.array([[0, 1], [1, 0]], complex),
        2: np.array([[0, -1j], [1j, 0]], complex),
        3: np.array([[1, 0], [0, -1]], complex),
    }


def code_to_matrix(n: int, code: int):
    import numpy as np
    validate_word_code(n, code)
    P = single_pauli_mats()
    if n == 0:
        return np.array([[1]], dtype=complex)
    M = P[code & 3]
    for j in range(1, n):
        M = np.kron(M, P[(code >> (2 * j)) & 3])
    return M


def to_matrix(A: MV):
    import numpy as np
    out = np.zeros((2 ** A.n, 2 ** A.n), complex)
    for code, coeff in A.terms.items():
        out += coeff * code_to_matrix(A.n, code)
    return out


def from_matrix(M, n: int | None = None, tol: float = 1e-12) -> MV:
    import numpy as np
    M = np.asarray(M, dtype=complex)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix")
    dim = M.shape[0]
    if n is None:
        if dim < 1 or dim & (dim - 1):
            raise ValueError("dimension must be a power of two")
        n = int(math.log2(dim))
    validate_n(n)
    if dim != 2 ** n:
        raise ValueError(f"matrix shape {M.shape} incompatible with n={n}")
    coeffs: dict[int, complex] = {}
    denom = 2 ** n
    for code in range(4 ** n):
        P = code_to_matrix(n, code)
        # The Pauli coefficient is the Hilbert--Schmidt pairing.  ``vdot``
        # computes it directly without allocating the dense matrix product.
        c = np.vdot(P, M) / denom
        if abs(c) > tol:
            coeffs[code] = complex(c)
    return MV(n, coeffs)


def density_from_statevector(psi, tol: float = 1e-12) -> MV:
    import numpy as np
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    dim = psi.shape[0]
    if dim < 1 or dim & (dim - 1):
        raise ValueError("statevector length must be a power of two")
    norm = np.linalg.norm(psi)
    if norm <= tol:
        raise ValueError("zero statevector")
    psi = psi / norm
    return from_matrix(np.outer(psi, psi.conj()), tol=tol)


def expm_matrix(A: MV, tol: float = 1e-12) -> MV:
    """Dense matrix exponential, including defective/non-normal matrices.

    Diagonalising a general matrix is not a valid exponential algorithm: a
    Jordan block need not have an invertible eigenvector matrix.  SciPy's
    scaling-and-squaring Pade implementation is used when the research extra
    is installed; the package's scaling-and-squaring Taylor implementation is
    the NumPy-only fallback.
    """
    M = to_matrix(A)
    try:
        from scipy.linalg import expm
    except ImportError:  # pragma: no cover - exercised in a NumPy-only install
        from .gates import expm_taylor
        return expm_taylor(A)
    return from_matrix(expm(M), A.n, tol=tol)


def exact_ground(H: MV):
    import numpy as np
    w, V = np.linalg.eigh(to_matrix(H))
    return float(w[0]), V[:, 0]
