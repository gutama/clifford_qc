from clifford_qc import MV, X, Z, exact_ground, to_matrix


def tfim(n, J=1.0, h=1.0):
    H = MV(n)
    for i in range(n - 1):
        H += -J * Z(n, i) * Z(n, i + 1)
    for i in range(n):
        H += -h * X(n, i)
    return H

H = tfim(4, 1.0, 1.0)
E0, psi0 = exact_ground(H)
print("E0 =", E0)
print("nnz(H) =", H.nnz())
