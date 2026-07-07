import math
import numpy as np

from clifford_qc import *


def test_pauli_product_and_trace():
    assert (X(2, 0) * Y(2, 0)).is_close(1j * Z(2, 0))
    A = P("XI") + 0.3 * P("ZZ")
    B = 0.2 * P("IX") - 0.1j * P("YY")
    assert np.allclose(to_matrix(A * B), to_matrix(A) @ to_matrix(B))
    assert abs(np.trace(to_matrix(A)) - A.trace()) < 1e-10


def test_gamma_and_fermions():
    n = 3
    gs = [gamma(n, i) for i in range(2 * n)]
    assert all((g * g).is_close(I(n)) for g in gs)
    assert all(anticomm(gs[i], gs[j]).norm_hs() < 1e-12 for i in range(2 * n) for j in range(i + 1, 2 * n))
    assert number_op(n, 1).is_close(number_op_pauli(n, 1))


def test_bell_chsh():
    bell = bell_density()
    A0, A1 = Z(2, 0), X(2, 0)
    B0 = (Z(2, 1) + X(2, 1)) / math.sqrt(2)
    B1 = (Z(2, 1) - X(2, 1)) / math.sqrt(2)
    chsh = sum(s * expectation(bell, Aa * Bb).real for s, Aa, Bb in [(1, A0, B0), (1, A0, B1), (1, A1, B0), (-1, A1, B1)])
    assert abs(chsh - 2 * math.sqrt(2)) < 1e-9
    assert abs(negativity(bell, {1}) - 0.5) < 1e-9


def test_channel_and_density():
    rho = bell_density()
    assert check_kraus(depolarizing(2, 0, 0.2))
    out = apply_channel(rho, depolarizing(2, 0, 0.2), check=True)
    assert out.is_density()
