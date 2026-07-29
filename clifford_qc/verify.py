from __future__ import annotations

import math


def _check(name: str, ok: bool) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        raise SystemExit(f"verification failed: {name}")


def run_verification() -> None:
    import numpy as np

    from . import (
        MV, I, X, Y, Z, P, comm, anticomm, tensor, gamma, blade_from_mask,
        c_op, cdag_op, number_op, number_op_pauli,
        H, S, T, RX, RY, RZ, CNOT, CZ, SWAP, TOFFOLI, rotor,
        ket_density, evolve, expectation, probability, purity,
        partial_trace, partial_transpose, z_projectors, measure,
        computational_probabilities, bell_density, ghz_density,
        depolarizing, dephasing, amplitude_damping, check_kraus, apply_channel,
        to_matrix, from_matrix, negativity, vn_entropy,
        trotter_unitary, trotter2_unitary, expm_taylor, expm_matrix,
        code_to_label,
    )

    from . import __version__

    print(f"== clifford_qc v{__version__} verification ==\n")

    n = 3
    rng = np.random.default_rng(7)

    def rand_mv():
        return MV(n, {int(rng.integers(0, 4 ** n)): complex(rng.normal(), rng.normal()) for _ in range(8)})

    print("-- algebra kernel --")
    A, B, C = rand_mv(), rand_mv(), rand_mv()
    _check("associativity (AB)C = A(BC)", ((A * B) * C).is_close(A * (B * C), 1e-8))
    _check("(AB)† = B†A†", (A * B).dagger().is_close(B.dagger() * A.dagger(), 1e-8))
    _check("trace cyclicity Tr(ABC)=Tr(BCA)", abs((A * B * C).trace() - (B * C * A).trace()) < 1e-8)
    _check("local Pauli table XY=iZ", (X(n, 1) * Y(n, 1)).is_close(1j * Z(n, 1)))
    _check("different-qubit Paulis commute", comm(X(n, 0), X(n, 1)).norm_hs() < 1e-12)
    _check("label interface roundtrip", P("XIZ").to_labels() == {"XIZ": 1 + 0j})

    print("-- Clifford/Jordan-Wigner layer --")
    gs = [gamma(n, i) for i in range(2 * n)]
    _check("gamma_i^2 = 1", all((g * g).is_close(I(n)) for g in gs))
    _check("gamma_i gamma_j + gamma_j gamma_i = 0 for i≠j",
           all(anticomm(gs[i], gs[j]).norm_hs() < 1e-12 for i in range(2 * n) for j in range(i + 1, 2 * n)))
    _check("gamma_i has grade 1", all(g.grades() == {1} for g in gs))
    _check("Z_j = -i gamma_2j gamma_2j+1", Z(n, 1).is_close(-1j * gs[2] * gs[3]) and Z(n, 1).grades() == {2})
    _check("inverse JW blade-mask decode for every gamma subset",
           all(blade_from_mask(n, mask).blade_mask(next(iter(blade_from_mask(n, mask).terms))) == mask
               for mask in range(1 << (2 * n))))

    print("-- fermionic Witt layer --")
    cs = [c_op(n, j) for j in range(n)]
    cds = [cdag_op(n, j) for j in range(n)]
    car = True
    for i in range(n):
        for j in range(n):
            car &= (cs[i] * cds[j] + cds[j] * cs[i]).is_close(I(n) if i == j else MV(n), 1e-12)
            car &= (cs[i] * cs[j] + cs[j] * cs[i]).norm_hs() < 1e-12
    _check("CAR and nilpotency", car and all((c * c).norm_hs() < 1e-12 for c in cs))
    _check("number operator c†c = 1/2(1-Z)", all(number_op(n, j).is_close(number_op_pauli(n, j), 1e-12) for j in range(n)))

    print("-- matrix representation --")
    MA, MB = to_matrix(A), to_matrix(B)
    _check("matrix(A*B)=matrix(A)@matrix(B)", np.allclose(to_matrix(A * B), MA @ MB, atol=1e-8))
    _check("Tr = 2^n <A>_0", abs(np.trace(MA) - A.trace()) < 1e-8)
    _check("dagger = conjugate transpose", np.allclose(to_matrix(A.dagger()), MA.conj().T, atol=1e-8))
    _check("from_matrix(to_matrix(A)) roundtrip", from_matrix(MA, n).is_close(A, 1e-8))
    _check("tensor matches numpy.kron", np.allclose(to_matrix(tensor(X(1, 0), Z(1, 0))), np.kron(to_matrix(X(1, 0)), to_matrix(Z(1, 0)))))

    print("-- gates and rotors --")
    gates = [H(n, 0), S(n, 1), T(n, 2), RX(n, 0, 0.7), RY(n, 1, 1.1), RZ(n, 2, 2.3), CNOT(n, 0, 1), CZ(n, 1, 2), SWAP(n, 0, 2), TOFFOLI(n, 0, 1, 2)]
    _check("standard gates are unitary", all(U.is_unitary(1e-9) for U in gates))
    _check("rotor identity", RZ(n, 0, 1.3).is_close(math.cos(0.65) * I(n) - 1j * math.sin(0.65) * Z(n, 0)))
    _check("H^2=1, T^2=S, S^2=Z", (H(n, 0) * H(n, 0)).is_close(I(n)) and (T(n, 2) * T(n, 2)).is_close(S(n, 2)) and (S(n, 2) * S(n, 2)).is_close(Z(n, 2)))
    U = CNOT(n, 0, 1)
    _check("CNOT maps Pauli words to single Pauli words", all((U * P(s) * U.dagger()).nnz() == 1 for s in ("XII", "YII", "ZII", "IXI", "IYI", "IZI")))

    print("-- states, measurements, entanglement --")
    rho0 = ket_density(2, "00")
    bell = bell_density()
    target = 0.25 * (I(2) + X(2, 0) * X(2, 1) - Y(2, 0) * Y(2, 1) + Z(2, 0) * Z(2, 1))
    _check("density validity", rho0.is_density() and bell.is_density() and abs(purity(bell) - 1) < 1e-9)
    _check("Bell = 1/4(1+XX-YY+ZZ)", bell.is_close(target, 1e-9))
    red = partial_trace(bell, {1})
    _check("Bell reduced state is maximally mixed", abs(red.trace() - 1) < 1e-9 and abs(purity(red) - 0.5) < 1e-9)
    ev = np.linalg.eigvalsh(to_matrix(partial_transpose(bell, {1})))
    _check("PPT witness min eigenvalue = -1/2", abs(min(ev) + 0.5) < 1e-9)
    _check("negativity and entropy", abs(negativity(bell, {1}) - 0.5) < 1e-9 and abs(vn_entropy(red) - 1.0) < 1e-9)
    meas = measure(bell, z_projectors(2, 0), check_projectors=True)
    _check("Bell Z-measurement probabilities", abs(meas[0][0] - 0.5) < 1e-9 and abs(meas[1][0] - 0.5) < 1e-9)
    ghz = ghz_density(3)
    _check("GHZ pure and one-qubit trace purity 1/2", abs(purity(ghz) - 1) < 1e-9 and abs(purity(partial_trace(ghz, {2})) - 0.5) < 1e-9)

    print("-- CHSH and Grover --")
    A0, A1 = Z(2, 0), X(2, 0)
    B0 = (1 / math.sqrt(2)) * (Z(2, 1) + X(2, 1))
    B1 = (1 / math.sqrt(2)) * (Z(2, 1) - X(2, 1))
    chsh = sum(s * expectation(bell, Aa * Bb).real for s, Aa, Bb in [(1, A0, B0), (1, A0, B1), (1, A1, B0), (-1, A1, B1)])
    _check(f"CHSH = 2√2 (got {chsh:.6f})", abs(chsh - 2 * math.sqrt(2)) < 1e-9)
    Hs = H(2, 0) * H(2, 1)
    Gop = (Hs * (2 * ket_density(2, "00") - I(2)) * Hs) * CZ(2, 0, 1)
    grover = evolve(evolve(ket_density(2, "00"), Hs), Gop)
    _check("Grover-2 finds |11>", abs(probability(grover, ket_density(2, "11")) - 1) < 1e-9)
    _check("computational probabilities sum to one", abs(sum(computational_probabilities(grover).values()) - 1.0) < 1e-9)

    print("-- channels and dynamics --")
    _check("Kraus completeness", all(check_kraus(ks) for ks in [depolarizing(2, 0, 0.3), dephasing(2, 0, 0.2), amplitude_damping(2, 0, 0.4)]))
    noisy = bell
    for _ in range(60):
        noisy = apply_channel(noisy, depolarizing(2, 0, 0.5))
        noisy = apply_channel(noisy, depolarizing(2, 1, 0.5))
    _check("two-qubit depolarizing fixed point purity -> 1/4", abs(purity(noisy) - 0.25) < 1e-6)
    H_terms = [(1.0, X(2, 0) * X(2, 1)), (0.5, Z(2, 0)), (0.5, Z(2, 1))]
    Hmat = sum(c * to_matrix(Pw) for c, Pw in H_terms)
    t = 0.9
    w, V = np.linalg.eigh(Hmat)
    Uex = V @ np.diag(np.exp(-1j * w * t)) @ V.conj().T
    Hmv = sum((c * Pw for c, Pw in H_terms), MV(2))
    _check("Trotter(400) error < 2e-2", np.linalg.norm(to_matrix(trotter_unitary(H_terms, t, 400)) - Uex) < 2e-2)
    _check("Trotter2(40) error < 2e-3", np.linalg.norm(to_matrix(trotter2_unitary(H_terms, t, 40)) - Uex) < 2e-3)
    _check("Taylor expm and matrix expm agree with exact", np.linalg.norm(to_matrix(expm_taylor((-1j * t) * Hmv)) - Uex) < 1e-9 and np.linalg.norm(to_matrix(expm_matrix((-1j * t) * Hmv)) - Uex) < 1e-9)

    print("-- A-CASE operator-response subspace --")
    from .matrix import exact_ground
    from .models.spin import tfim
    from .subspace import (MatrixElementBank, SharedMeasurement, dense_subspace,
                           identity_generator, krylov_response, ritz_uncertainty,
                           run_acase, solve_subspace)

    A_nh = P("XYI").dagger() * P("IZY")  # non-Hermitian: the pairing's real case
    _check("trace pairing = Tr(AB)/2^n on a non-Hermitian product",
           abs((2 ** 3) * A_nh.trace_pairing(A) - np.trace(to_matrix(A_nh) @ MA)) < 1e-8)
    spin = tfim(3, J=1.0, h=1.0)
    ref = ket_density(3, "000")
    Egs, _ = exact_ground(spin.hamiltonian.to_mv())
    gens = [identity_generator(3)] + krylov_response(spin.hamiltonian, 4)
    ritz = [solve_subspace(ref, spin.hamiltonian, gens[:k]).ground_energy
            for k in range(1, len(gens) + 1)]
    _check(f"variational bound E_sub >= E_0 (got {ritz[-1]:.6f} vs {Egs:.6f})",
           all(e >= Egs - 1e-9 for e in ritz))
    _check("nested growth is monotone non-increasing",
           all(b <= a + 1e-9 for a, b in zip(ritz, ritz[1:])))
    _check("operator route matches the dense basis-state route",
           abs(ritz[-1] - dense_subspace(ref, spin.hamiltonian, gens).ground_energy) < 1e-9)
    bank = MatrixElementBank(ref, spin.hamiltonian, gens)
    banked = bank.solve()
    _check("matrix-element bank reproduces the direct assembly bit for bit",
           banked.energies == solve_subspace(ref, spin.hamiltonian, gens).energies)
    _check("projected observable <H> returns the Ritz energy (state never formed)",
           abs(banked.expectation(spin.hamiltonian) - banked.ground_energy) < 1e-9)
    grown = run_acase(ref, spin.hamiltonian, gens[1:] + krylov_response(spin.hamiltonian, 6),
                      max_size=4, exact_ground_energy=Egs)
    _check(f"adaptive growth stays above E_0 and improves ({grown.energy:.6f})",
           grown.energy >= Egs - 1e-9 and grown.energy <= ritz[0] + 1e-9)
    _check("predicted lowering is a lower bound on the actual lowering",
           all(r.actual_lowering >= r.predicted_lowering - 1e-12 for r in grown.records))

    shared = SharedMeasurement(bank)
    S_limit, H_limit = shared.exact_matrices()
    S_exact, H_exact = bank.matrices()
    _check("finite-shot reconstruction reproduces (S,H) in the infinite-shot limit",
           np.abs(S_limit - S_exact).max() < 1e-9
           and np.abs(H_limit - H_exact).max() < 1e-9)
    from .backends.finite_shot import FiniteShotBackend
    cache = shared.measure(FiniteShotBackend(seed=5), 4000)
    measured = shared.solve(cache)
    interval = ritz_uncertainty(shared, cache, measured)
    _check(f"measured Ritz value carries an asymptotic interval "
           f"(E = {measured.ground_energy:.4f} +- {interval.upper - interval.estimate:.1e})",
           interval.evidence == "asymptotic" and not interval.certified
           and interval.lower <= interval.estimate <= interval.upper)

    print("-- materials layer --")
    from .fermion import total_number_op, total_sz_op
    from .models.lattice import hubbard, kitaev_honeycomb
    from .models.observables import double_occupancy, occupation, spin_correlation
    from .subspace import determinant_excitations, occupied_spin_orbitals

    cluster = hubbard(2, t=1.0, U=4.0)
    H_cluster = cluster.hamiltonian.to_mv()
    _check("Hubbard cluster is Hermitian and conserves N and S_z",
           H_cluster.is_hermitian(1e-12)
           and comm(H_cluster, total_number_op(cluster.n)).is_zero(1e-10)
           and comm(H_cluster, total_sz_op(cluster.n)).is_zero(1e-10))
    free = hubbard(2, t=1.0, U=0.0)
    _check("U=0 two-site chain reproduces the free-fermion energy (-2t)",
           abs(exact_ground(free.hamiltonian.to_mv())[0] - (-2.0)) < 1e-9)
    honeycomb = kitaev_honeycomb(2, 2)
    _check("Kitaev cluster is a one-qubit-per-site spin model (8 sites, 8 links)",
           honeycomb.n == 8 and len(honeycomb.hamiltonian.terms) == 8
           and len(kitaev_honeycomb(2, 2, periodic=True).hamiltonian.terms) == 12)

    from .backends.exact_mv import ExactMVBackend
    cluster_rho = ExactMVBackend().state(cluster.reference, ())
    occupied = occupied_spin_orbitals(cluster)
    excitations = determinant_excitations(cluster.n, occupied)
    cluster_bank = MatrixElementBank(cluster_rho, cluster.hamiltonian,
                                     [identity_generator(cluster.n)] + excitations)
    cluster_result = cluster_bank.solve()
    E_cluster, _ = exact_ground(H_cluster)
    _check(f"A-CASE reaches the two-site cluster ground state "
           f"({cluster_result.ground_energy:.6f})",
           abs(cluster_result.ground_energy - E_cluster) < 1e-9)
    _check("projected observables agree with the operators they came from",
           abs(cluster_result.expectation(occupation(cluster, 0)) - 1.0) < 1e-9
           and cluster_result.expectation(double_occupancy(cluster, 0)) >= 0.0
           and cluster_result.expectation(spin_correlation(cluster, 0, 1)) < 0.0)

    print("-- sector-restricted exact tier --")
    from .backends.sector_statevector import (SectorStatevectorBackend,
                                              lanczos_ground, sector_basis,
                                              sector_projector)

    sector = SectorStatevectorBackend(cluster.n, cluster.metadata["n_electrons"],
                                      cluster.metadata["sz"])
    _check(f"sector holds C(n,k) amplitudes, not 2^n "
           f"({sector.dimension} vs {2 ** cluster.n})",
           sector.dimension == len(sector_basis(cluster.n, 2, 0.0))
           and sector.dimension < 2 ** cluster.n)
    projector = sector_projector(cluster.n, cluster.metadata["n_electrons"],
                                 cluster.metadata["sz"])
    _check("sector projector is idempotent and commutes with H",
           (projector * projector).is_close(projector, 1e-10)
           and comm(H_cluster, projector).is_zero(1e-10))
    sector_energy = sector.ground_state(cluster.hamiltonian, k=1,
                                        method="lanczos")[0][0]
    _check(f"matrix-free Lanczos matches the dense ground energy "
           f"({sector_energy:.6f})",
           abs(sector_energy - E_cluster) < 1e-8)
    reference_state = sector.state_from_program(cluster.reference)
    _check("determinant reference maps to one sector amplitude with its energy",
           int(np.count_nonzero(reference_state)) == 1
           and abs(sector.expectation(cluster.hamiltonian, reference_state).real
                   - ExactMVBackend().expectation(cluster.reference,
                                                  cluster.hamiltonian, ())) < 1e-9)
    bigger = SectorStatevectorBackend(16, 8, 0.0)
    _check(f"a 16-qubit sector is 4900 amplitudes, not 65536 "
           f"({bigger.memory_estimate()['sector_bytes'] // 1024} KiB)",
           bigger.dimension == 4900)

    print("\n-- structural report --")
    print(f"  Bell: {bell.nnz()}/16 words | GHZ: {ghz.nnz()}/64 words | Toffoli: {TOFFOLI(3,0,1,2).nnz()}/64 words")
    print(f"  RZ grades: {sorted(RZ(2,0,0.8).grades())}; CNOT grades: {sorted(CNOT(2,0,1).grades())}")
    print("\nAll checks passed.")


if __name__ == "__main__":
    run_verification()
