import math
from clifford_qc import bell_density, X, Z, expectation, negativity, partial_trace, vn_entropy

rho = bell_density()
A0, A1 = Z(2, 0), X(2, 0)
B0 = (Z(2, 1) + X(2, 1)) / math.sqrt(2)
B1 = (Z(2, 1) - X(2, 1)) / math.sqrt(2)
S = sum(s * expectation(rho, Aa * Bb).real for s, Aa, Bb in [(1, A0, B0), (1, A0, B1), (1, A1, B0), (-1, A1, B1)])
print(f"CHSH = {S:.12f}; 2√2 = {2*math.sqrt(2):.12f}")
print(f"negativity = {negativity(rho, {1}):.6f}")
print(f"single-qubit entropy = {vn_entropy(partial_trace(rho, {1})):.6f} bit")
