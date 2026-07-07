from clifford_qc import bell_density, depolarizing, apply_channel, purity, check_kraus

rho = bell_density()
ks0 = depolarizing(2, 0, 0.25)
ks1 = depolarizing(2, 1, 0.25)
print("Kraus ok:", check_kraus(ks0), check_kraus(ks1))
for step in range(8):
    print(step, purity(rho))
    rho = apply_channel(apply_channel(rho, ks0), ks1)
