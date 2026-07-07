from clifford_qc import H, CZ, I, ket_density, evolve, probability, computational_probabilities

n = 2
Hs = H(n, 0) * H(n, 1)
diffusion = Hs * (2 * ket_density(n, "00") - I(n)) * Hs
oracle_11 = CZ(n, 0, 1)
G = diffusion * oracle_11
rho = evolve(evolve(ket_density(n, "00"), Hs), G)
print(computational_probabilities(rho))
print("p(11)=", probability(rho, ket_density(n, "11")))
