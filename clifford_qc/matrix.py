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
        c = np.trace(P.conj().T @ M) / denom
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
    import numpy as np
    M = to_matrix(A)
    w, V = np.linalg.eig(M)
    E = V @ np.diag(np.exp(w)) @ np.linalg.inv(V)
    return from_matrix(E, A.n, tol=tol)


def exact_ground(H: MV):
    import numpy as np
    w, V = np.linalg.eigh(to_matrix(H))
    return float(w[0]), V[:, 0]
