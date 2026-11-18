# Finite-shot commutator subspaces against Krylov at matched shot budgets

**Status: exploratory.** No declaration preceded this study, and nothing here
is a hardware result or a circuit simulation. The subspaces, matrix elements
and target energies are exact. Shot noise enters only through the declared
element-level model below. The record is
[`matched_shot_commutator_krylov.json`](matched_shot_commutator_krylov.json),
produced by
[`run_matched_shot_commutator_krylov.py`](run_matched_shot_commutator_krylov.py).
A second record,
[`matched_shot_commutator_krylov_floor20.json`](matched_shot_commutator_krylov_floor20.json),
repeats the study with a ten times stricter overlap floor. That variant was
chosen after the first run had been seen, so it is a post hoc sensitivity
check.

## Question

With the same total number of shots, does an A-CASE subspace grown from
excitation and commutator directions on Hartree-Fock reach the ground energy
as well as a Krylov subspace does? And how does the answer change from 6 to
16 qubits, on molecules where Hartree-Fock is far from exact?

## Short answer

Not in this setting. Krylov subspaces get much closer to the target with the
same number of basis vectors, and they keep that lead at matched shot
budgets: the better Krylov arm beats every A-CASE arm in 240 of 261 matched
comparisons. Commutator directions improve the A-CASE pool on N2 and on the
12-qubit water, but nowhere near enough to close the gap.

A-CASE's real advantage is conditioning. Its pencils settle cleanly under shot
noise, but on energies 33 to 359 mHa above the target. No arm reaches chemical
accuracy on a 16-qubit system within 10^12 shots and 12 basis vectors. For
Krylov the limit there is noise amplified by its ill-conditioned overlaps,
not the subspace.

## Systems

Eight active spaces from three families, each mapped by Jordan-Wigner:

- **Stretched water.** Both O-H bonds are 1.916 Å, twice the equilibrium
  length, at the equilibrium angle. Three active spaces: CAS(4e,3o) and
  CAS(8e,6o) in STO-3G (6 and 12 qubits), and CAS(8e,8o) in 6-31G (16 qubits).
- **Linear hydrogen chains.** H4, H6 and H8 at 2.0 Å spacing in STO-3G, all
  orbitals active (8, 12 and 16 qubits).
- **Stretched N2.** Bond length 2.0 Å in STO-3G, with CAS(6e,6o) (12 qubits)
  and CAS(10e,8o) (16 qubits).

Hartree-Fock is 104 to 635 mHa above the exact energy, and the exact state's
weight on it, |⟨HF|target⟩|², is between 0.21 and 0.81 (first generated
table). Everything is
computed in the exact (N, S_z = 0) sector, at most 4,900 determinants, with
PySCF's FCI machinery. The target is the lowest singlet in that sector. In all
eight systems this is also the lowest state of the sector.

## Arms

Every basis starts with the Hartree-Fock determinant and grows to
M = 4, 6, 8 and 12 vectors.

| Arm | Basis vectors |
| --- | --- |
| Real-time Krylov | e^(-iHkΔt)\|HF⟩ for k = 0, ..., M-1, with Δt = π / (E_max - E_min) of the exact sector spectrum. The pencil is Toeplitz. |
| Power Krylov | H'^k \|HF⟩, each normalized, where H' is the qubit Hamiltonian without its identity term. |
| A-CASE, excitations + commutators | Greedy choice from the normalized vectors τ\|HF⟩, [H, τ]\|HF⟩ and [H, [H, τ]]\|HF⟩, where τ runs over every S_z-conserving single and double excitation generator. |
| A-CASE, commutators only | The same greedy choice from the [H, τ] and [H, [H, τ]] vectors alone. |
| A-CASE, excitations only | The same greedy choice from the τ\|HF⟩ vectors alone. This is the standard A-CASE pool, and a test checks that it reproduces the library's `run_acase` energies. |

The greedy rule is A-CASE's. Each step adds the candidate with the largest
exact two-by-two energy lowering against the current basis, with ties broken
canonically, and stops when no candidate lowers the energy by more than
10^-10 Ha. On the 6-qubit water every pool reaches the exact energy and stops
at five vectors, so the A-CASE arms have no M = 6, 8 or 12 there.

## Shot model

The exact overlap S and Hamiltonian H of each normalized basis are computed
first. Each real quantity x is then estimated from n shots of a two-outcome
measurement with outcomes ±b, as x̂ = b(2k/n - 1) with
k ~ Binomial(n, (1 + x/b)/2). For overlaps b = 1. For Hamiltonian elements
b = λ, the Pauli 1-norm of the qubit Hamiltonian without its identity term.
This is the variance of a Hadamard test in which each shot samples one Pauli
term with probability proportional to its coefficient's magnitude. The unit
diagonal of a normalized overlap is known and not measured.

The arms need different numbers of real quantities:

- **Real-time Krylov** needs 4M - 3: the first rows of its Toeplitz S and H,
  which are complex except for the real ⟨HF|H'|HF⟩.
- **The other four arms** need M². Their vectors are real (a real Hamiltonian
  in real orbitals, real excitation generators), so the imaginary parts of
  their pencils vanish and are not measured. That leaves the M(M-1)/2 overlaps
  above the diagonal and the M(M+1)/2 Hamiltonian elements on and above it.

Each arm divides the same total budget N equally among its quantities, so it
gets n = ⌊N/Q⌋ shots per quantity. Budgets run from 10^4 to 10^12 total shots
in powers of ten, with 200 replicas per (system, arm, M, budget) cell. Seeds
come from `SeedSequence([20261118, system, arm, M, budget index])`. A run
reproduces bit for bit, because the script pins BLAS and OpenMP to one thread.

The noisy pencil is solved with the library's `solve_projected`. Directions of
the normalized overlap whose eigenvalue is below the floor
η = 2 sqrt(vM/n) are dropped. Here v = 2 for complex overlap elements and
v = 1 for real ones, and η is the expected spectral norm of the overlap's noise
matrix. The sensitivity record uses 20 sqrt(vM/n) instead.

## What the model credits and leaves out

- **Ideal state preparation.** Every arm is credited with preparing its
  normalized basis states. Power Krylov's H'^k|HF⟩ and the commutator states
  are not unitary images of |HF⟩. On hardware they would be reached through
  moments or products of operators, at costs this model does not see.
  Real-time Krylov needs controlled time evolution. Circuit depth, gate noise
  and Trotter error are not modelled.
- **Shots are a cost model.** Every quantity here is classically computable
  from a determinant reference. The study asks how many shots the declared
  estimator would need, not what a device would return.
- **Free selection for A-CASE.** The A-CASE arms rank candidates with exact
  classical lowering scores that cost no shots. A selection paid for in shots
  would cost those arms more, so this favours A-CASE.
- **An exact time step for real-time Krylov.** Its Δt uses the exact spectral
  range. A device would use a bound such as λ, which gives a smaller step and a
  worse-conditioned basis, so this favours real-time Krylov.
- **Equal allocation.** Every quantity gets the same shots. Variance-weighted
  allocation would help each arm by a different amount.
- **No Pauli-word count for the pencils.** The first table lists the
  qubit-wise-commuting settings of H itself. Elements between commutator
  states involve products of up to five Hamiltonian factors. A survey run
  while designing the study (not part of the record) found their Pauli
  expansions too large for a word-level cost at 12 and 16 qubits.

## Results

All numbers below are in the generated tables at the end.

### Without shot noise, Krylov is far better per basis vector

At M = 12 both Krylov arms are within 2.2 mHa of the target on every system.
The best A-CASE arm at M = 12 is still well above it on the 8- to 16-qubit
systems: 33 mHa on H4, 36 on the 12-qubit water, 156 on H6, 118 on the
12-qubit N2, 40 on the 16-qubit water, 359 on H8 and 137 on the 16-qubit N2.

Krylov pays for this in conditioning. Its overlap condition numbers reach
about 10^10 to 10^17 at M = 12, and some bases are numerically singular. No
A-CASE basis has a condition number above 80.

### Commutators help the A-CASE pool on N2 and on the 12-qubit water

Adding [H, τ] and [H, [H, τ]] to the excitation pool lowers the noise-free
error at M = 12:

- N2: from 175.8 to 117.9 mHa (12 qubits) and from 193.9 to 137.4 mHa
  (16 qubits);
- the 12-qubit water: from 57.8 to 36.3 mHa;
- H4: from 46.1 to 39.0 mHa.

It changes nothing on H6, where the greedy rule picks no commutator, or on the
16-qubit water (40.4 against 40.0 mHa). On H8 the larger pool makes the greedy
choice worse: 384.4 mHa against 359.2 with excitations alone.

Commutators alone beat excitations alone on H4 at M = 12 (32.7 against
46.1 mHa) and on both N2 active spaces, and lose on the other systems. Even
the best commutator pool leaves A-CASE tens to hundreds of millihartree from
the target at M = 12, where Krylov is within 2.2.

### At matched shots, Krylov still wins almost every comparison

Take each matched cell: one system, basis size and total budget. The better
Krylov arm beats every A-CASE arm in 240 of the 261 cells where both families
return energies. Power Krylov has the smallest median error in 214 cells,
real-time Krylov in 53, and the three A-CASE arms in 21 between them.

Choosing the best basis size for each family in hindsight gives the same
picture. At 10^8, 10^10 and 10^12 total shots the better Krylov arm is ahead
on every system above six qubits. The closest case is the 16-qubit water at 10^8 shots: 28.5 mHa
against 29.4.

Most of the 21 A-CASE wins come at 10^7 shots or fewer. There, shot noise
drags the estimate from a poor subspace towards the target, or past it. For
example, on the 12-qubit N2 at M = 8 and 10^6 shots, the excitations-only arm
has a median error of 74.1 mHa although its exact subspace is 192.5 mHa from
the target, and 65% of its replicas lie more than 1.6 mHa below the target.

The rest are on the stretched water at 10^8 shots and M ≤ 8. There A-CASE's
well-conditioned pencil returns close to its exact energy while Krylov at the
same M is still noise-limited. On the 12-qubit water at M = 8, for instance,
A-CASE has 38.1 mHa against Krylov's 43.2. The last two, on the 6-qubit
water at 10^11 and 10^12 shots, are margins below 0.1 mHa.

### Chemical accuracy at 6 to 12 qubits, never at 16

The chemical-accuracy table asks when at least half the replicas land within
1.6 mHa of the target:

- **6 qubits:** every arm, by 10^8 total shots (10^7 for power Krylov).
- **8 qubits (H4):** only the Krylov arms, at 10^8 shots (power) and 10^9
  (real-time).
- **12 qubits:** only power Krylov on H6, at 10^9 shots with M = 12. The
  12-qubit water and N2 never get there.
- **16 qubits:** no arm on any system, up to 10^12 shots and M = 12.

The 16-qubit failure is not a subspace failure. Krylov's exact errors there
are at most 2.2 mHa at M = 12. With 10^12 shots, the best Krylov medians are
still 2.6 mHa on H8, 10.9 on the water and 19.3 on N2. That gap is shot noise
amplified by the ill-conditioned overlaps, together with the directions the
floor discards.

### A-CASE is robust to shots but converges to the wrong energy

The A-CASE pencils are well conditioned, so their shot noise falls cleanly as
λ/√n. Across all A-CASE cells, the median deviation from the arm's own exact
energy is at most 11 mHa at 10^10 shots and 1.2 mHa at 10^12. But that exact
energy is 33 to 359 mHa above the target on the 8- to 16-qubit systems.

On small budgets the noisy A-CASE pencils are strongly non-variational. At
10^6 shots and M = 8, 96% to 100% of the mixed arm's replicas are more than
1.6 mHa below the target on both waters and both N2 active spaces. Their
median errors run from -418 to -3,908 mHa. The lowest eigenvalue of a noisy
pencil sits below the exact one, and the Hamiltonian elements carry λ times
the overlap noise. The Krylov arms show the same bias where their floor
admits noisy directions. Real-time Krylov on the 12-qubit water at M = 8 and
10^8 shots has a median error of -1,165 mHa.

### The overlap floor moves individual cells, not the comparison

A ten times stricter floor removes the worst real-time Krylov failures. On
the 12-qubit water at M = 8 and 10^8 shots, the median error drops from 1,165
to 12.8 mHa. On the 16-qubit N2 at M = 12 and 10^6 shots it drops from 1,419
to 122 mHa.

It costs accuracy elsewhere. For power Krylov on the 16-qubit N2 at M = 12 and
10^12 shots, the error rises from 27.7 to 43.3 mHa. On the 12-qubit N2 at
M = 8 and 10^8 shots it rises from 74.5 to 133 mHa. At 10^4 and 10^5 shots
the stricter floor removes the whole space in 108 cells, which then return no
energy.

The comparison stands: the better Krylov arm beats every A-CASE arm in 213 of
the 236 cells compared, and power Krylov wins 197 of them.

## History

The first full run of this study (from commit `8f7960b`, never committed as
a record) charged power Krylov and the A-CASE arms for measuring the imaginary
parts of their real pencils, and added noise there. That cost them about a
factor of four in effective shots against real-time Krylov. The model was
corrected (commit `0d634c7`) before the records reported here were produced.
Later commits change only the code that prints the report.

The correction moved numbers but no conclusion:

- the better Krylov arm beat every A-CASE arm in 239 of 261 cells before and
  in 240 after;
- power Krylov's cell wins rose from 198 to 214;
- no arm reached chemical accuracy on a 16-qubit system in either run.

The stricter-floor variant was chosen after that first run had shown
real-time Krylov's largest failures.

## A defect noticed in the FCIDUMP importer

`clifford_qc/models/fcidump.py` checks that an imported Hamiltonian is
Hermitian with `total.is_hermitian(1e-10)`. That compares a Hilbert-Schmidt
norm, which grows as 2^(n/2) with the qubit count, against an absolute
tolerance. On the 16-qubit N2 CAS(10e,8o) the residual from coefficient
rounding is 1.6 × 10^-9, while the largest asymmetry of any single
coefficient is 2.4 × 10^-12. The importer therefore rejects a Hamiltonian that
is Hermitian to rounding. This study assembles the same Jordan-Wigner
Hamiltonian itself and checks it coefficient by coefficient. A test confirms
that, where the importer works, the two agree term for term. The importer is
left unchanged here because committed records bind its bytes by digest. Fixing
it means migrating that evidence.

## Reproduce

Running the study needs PySCF (the `molecular` extra). Printing the report
does not.

```bash
python benchmarks/run_matched_shot_commutator_krylov.py --out benchmarks/matched_shot_commutator_krylov.json
python benchmarks/run_matched_shot_commutator_krylov.py --floor-multiplier 20 --out benchmarks/matched_shot_commutator_krylov_floor20.json
python benchmarks/run_matched_shot_commutator_krylov.py --report --out benchmarks/matched_shot_commutator_krylov.json --sensitivity benchmarks/matched_shot_commutator_krylov_floor20.json
```

Each full run took about five and a half minutes on one core. A test checks that the
tables below are exactly what `--report` prints from the two committed
records.

## Generated tables

<!-- generated by run_matched_shot_commutator_krylov.py --report; do not edit -->
| System | Qubits | Sector dim. | HF error (mHa) | \|⟨HF\|target⟩\|² | Pauli terms | λ (Ha) | QWC settings of H |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `h2o_2x_cas4e3o` | 6 | 9 | 104.5 | 0.809 | 61 | 4.2 | 17 |
| `h4_r2.0` | 8 | 36 | 322.2 | 0.482 | 184 | 5.0 | 68 |
| `h2o_2x_cas8e6o` | 12 | 225 | 326.6 | 0.524 | 550 | 26.0 | 166 |
| `h6_r2.0` | 12 | 400 | 478.8 | 0.319 | 918 | 12.8 | 284 |
| `n2_r2.0_cas6e6o` | 12 | 400 | 565.5 | 0.253 | 498 | 12.5 | 176 |
| `h2o_2x_631g_cas8e8o` | 16 | 4,900 | 200.2 | 0.637 | 1,904 | 36.1 | 535 |
| `h8_r2.0` | 16 | 4,900 | 635.3 | 0.209 | 2,912 | 25.1 | 849 |
| `n2_r2.0_cas10e8o` | 16 | 3,136 | 583.6 | 0.246 | 1,468 | 34.0 | 441 |

Exact subspace error (mHa) with no shot noise, and the overlap condition number, by basis size M:

| System | Arm | M = 4 | M = 6 | M = 8 | M = 12 |
| --- | --- | ---: | ---: | ---: | ---: |
| `h2o_2x_cas4e3o` | Real-time Krylov | 0.0 (κ 2e+04) | 0.0 (singular) | 0.0 (singular) | 0.0 (singular) |
| `h2o_2x_cas4e3o` | Power Krylov | 0.0 (κ 1e+08) | 0.0 (singular) | 0.0 (singular) | 0.0 (singular) |
| `h2o_2x_cas4e3o` | A-CASE, excitations + commutators | 0.0 (κ 2e+00) | exhausted at 5 | exhausted at 5 | exhausted at 5 |
| `h2o_2x_cas4e3o` | A-CASE, commutators only | 0.0 (κ 3e+00) | exhausted at 5 | exhausted at 5 | exhausted at 5 |
| `h2o_2x_cas4e3o` | A-CASE, excitations only | 0.0 (κ 1e+00) | exhausted at 5 | exhausted at 5 | exhausted at 5 |
| `h4_r2.0` | Real-time Krylov | 1.3 (κ 2e+03) | 0.8 (κ 5e+04) | 0.0 (κ 3e+06) | 0.0 (κ 6e+15) |
| `h4_r2.0` | Power Krylov | 1.1 (κ 4e+03) | 0.6 (κ 1e+05) | 0.0 (κ 3e+07) | 0.0 (κ 5e+16) |
| `h4_r2.0` | A-CASE, excitations + commutators | 109.1 (κ 1e+00) | 51.5 (κ 1e+00) | 50.1 (κ 2e+00) | 39.0 (κ 3e+00) |
| `h4_r2.0` | A-CASE, commutators only | 241.2 (κ 8e+00) | 82.8 (κ 3e+01) | 55.7 (κ 3e+01) | 32.7 (κ 8e+01) |
| `h4_r2.0` | A-CASE, excitations only | 109.1 (κ 1e+00) | 51.5 (κ 1e+00) | 50.6 (κ 1e+00) | 46.1 (κ 1e+00) |
| `h2o_2x_cas8e6o` | Real-time Krylov | 11.1 (κ 1e+03) | 4.1 (κ 4e+04) | 1.1 (κ 1e+06) | 0.0 (κ 3e+10) |
| `h2o_2x_cas8e6o` | Power Krylov | 8.2 (κ 6e+05) | 3.0 (κ 3e+09) | 1.2 (κ 5e+12) | 0.1 (singular) |
| `h2o_2x_cas8e6o` | A-CASE, excitations + commutators | 108.1 (κ 1e+00) | 48.1 (κ 3e+00) | 42.3 (κ 3e+00) | 36.3 (κ 3e+00) |
| `h2o_2x_cas8e6o` | A-CASE, commutators only | 195.6 (κ 6e+00) | 155.2 (κ 6e+00) | 128.8 (κ 8e+00) | 94.6 (κ 1e+01) |
| `h2o_2x_cas8e6o` | A-CASE, excitations only | 108.1 (κ 1e+00) | 61.5 (κ 1e+00) | 59.7 (κ 1e+00) | 57.8 (κ 1e+00) |
| `h6_r2.0` | Real-time Krylov | 4.4 (κ 2e+03) | 2.0 (κ 7e+05) | 1.3 (κ 3e+07) | 0.2 (κ 1e+11) |
| `h6_r2.0` | Power Krylov | 3.7 (κ 1e+03) | 1.8 (κ 1e+05) | 1.1 (κ 5e+06) | 0.1 (κ 1e+10) |
| `h6_r2.0` | A-CASE, excitations + commutators | 336.1 (κ 1e+00) | 264.9 (κ 1e+00) | 208.4 (κ 1e+00) | 156.2 (κ 1e+00) |
| `h6_r2.0` | A-CASE, commutators only | 445.8 (κ 2e+00) | 414.1 (κ 4e+00) | 397.2 (κ 4e+00) | 361.4 (κ 6e+00) |
| `h6_r2.0` | A-CASE, excitations only | 336.1 (κ 1e+00) | 264.9 (κ 1e+00) | 208.4 (κ 1e+00) | 156.2 (κ 1e+00) |
| `n2_r2.0_cas6e6o` | Real-time Krylov | 93.1 (κ 2e+02) | 15.2 (κ 1e+05) | 0.5 (κ 1e+08) | 0.0 (κ 3e+17) |
| `n2_r2.0_cas6e6o` | Power Krylov | 80.9 (κ 7e+01) | 10.5 (κ 8e+04) | 0.3 (κ 1e+08) | 0.0 (κ 2e+16) |
| `n2_r2.0_cas6e6o` | A-CASE, excitations + commutators | 176.9 (κ 3e+01) | 166.9 (κ 3e+01) | 152.4 (κ 4e+01) | 117.9 (κ 5e+01) |
| `n2_r2.0_cas6e6o` | A-CASE, commutators only | 218.4 (κ 2e+00) | 203.5 (κ 3e+00) | 190.3 (κ 9e+00) | 171.1 (κ 2e+01) |
| `n2_r2.0_cas6e6o` | A-CASE, excitations only | 220.6 (κ 1e+00) | 206.9 (κ 1e+00) | 192.5 (κ 1e+00) | 175.8 (κ 1e+00) |
| `h2o_2x_631g_cas8e8o` | Real-time Krylov | 22.7 (κ 8e+06) | 7.4 (κ 6e+09) | 4.6 (κ 2e+12) | 2.2 (κ 2e+16) |
| `h2o_2x_631g_cas8e8o` | Power Krylov | 22.7 (κ 5e+06) | 7.1 (κ 6e+08) | 3.0 (κ 1e+10) | 1.3 (κ 2e+12) |
| `h2o_2x_631g_cas8e8o` | A-CASE, excitations + commutators | 76.5 (κ 1e+00) | 49.8 (κ 1e+00) | 45.8 (κ 1e+00) | 40.0 (κ 1e+00) |
| `h2o_2x_631g_cas8e8o` | A-CASE, commutators only | 136.4 (κ 7e+00) | 120.5 (κ 1e+01) | 113.1 (κ 1e+01) | 104.8 (κ 1e+01) |
| `h2o_2x_631g_cas8e8o` | A-CASE, excitations only | 76.5 (κ 1e+00) | 49.8 (κ 1e+00) | 45.8 (κ 1e+00) | 40.4 (κ 1e+00) |
| `h8_r2.0` | Real-time Krylov | 25.4 (κ 8e+03) | 3.4 (κ 5e+06) | 2.6 (κ 5e+08) | 1.1 (κ 3e+12) |
| `h8_r2.0` | Power Krylov | 21.6 (κ 2e+03) | 3.2 (κ 3e+05) | 2.4 (κ 6e+06) | 0.4 (κ 7e+09) |
| `h8_r2.0` | A-CASE, excitations + commutators | 469.9 (κ 3e+00) | 449.2 (κ 4e+00) | 428.8 (κ 4e+00) | 384.4 (κ 4e+00) |
| `h8_r2.0` | A-CASE, commutators only | 595.9 (κ 2e+00) | 560.0 (κ 5e+00) | 514.7 (κ 1e+01) | 502.9 (κ 1e+01) |
| `h8_r2.0` | A-CASE, excitations only | 543.8 (κ 1e+00) | 480.0 (κ 1e+00) | 427.6 (κ 1e+00) | 359.2 (κ 1e+00) |
| `n2_r2.0_cas10e8o` | Real-time Krylov | 127.7 (κ 8e+02) | 71.5 (κ 2e+04) | 16.5 (κ 2e+06) | 0.7 (κ 8e+09) |
| `n2_r2.0_cas10e8o` | Power Krylov | 117.1 (κ 1e+04) | 53.9 (κ 1e+06) | 8.6 (κ 6e+08) | 0.9 (κ 5e+13) |
| `n2_r2.0_cas10e8o` | A-CASE, excitations + commutators | 195.5 (κ 3e+01) | 185.6 (κ 3e+01) | 172.0 (κ 4e+01) | 137.4 (κ 4e+01) |
| `n2_r2.0_cas10e8o` | A-CASE, commutators only | 248.2 (κ 2e+00) | 204.1 (κ 1e+01) | 176.1 (κ 2e+01) | 141.9 (κ 3e+01) |
| `n2_r2.0_cas10e8o` | A-CASE, excitations only | 238.6 (κ 1e+00) | 225.0 (κ 1e+00) | 210.6 (κ 1e+00) | 193.9 (κ 1e+00) |

Directions the A-CASE arms selected after the reference, among their first 12 basis vectors, out of the candidate pool:

| System | Pool: excitations τ / commutators [H, τ] and [H, [H, τ]] | Excitations + commutators arm: τ / [H, τ] / [H, [H, τ]] | Commutators-only arm: τ / [H, τ] / [H, [H, τ]] |
| --- | ---: | ---: | ---: |
| `h2o_2x_cas4e3o` | 8 / 16 | 2 / 0 / 2 | 0 / 0 / 4 |
| `h4_r2.0` | 26 / 52 | 10 / 0 / 1 | 0 / 3 / 8 |
| `h2o_2x_cas8e6o` | 92 / 184 | 6 / 2 / 3 | 0 / 4 / 7 |
| `h6_r2.0` | 117 / 234 | 11 / 0 / 0 | 0 / 7 / 4 |
| `n2_r2.0_cas6e6o` | 117 / 234 | 5 / 3 / 3 | 0 / 6 / 5 |
| `h2o_2x_631g_cas8e8o` | 360 / 720 | 10 / 1 / 0 | 0 / 11 / 0 |
| `h8_r2.0` | 360 / 720 | 9 / 2 / 0 | 0 / 7 / 4 |
| `n2_r2.0_cas10e8o` | 315 / 630 | 5 / 2 / 4 | 0 / 5 / 6 |

Smallest total shot budget at which at least half of the 200 replicas are within 1.6 mHa of the target, over all basis sizes (basis size in brackets):

| System | Qubits | Real-time Krylov | Power Krylov | A-CASE, excitations + commutators | A-CASE, commutators only | A-CASE, excitations only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `h2o_2x_cas4e3o` | 6 | 1e+08 (M = 4) | 1e+07 (M = 8) | 1e+08 (M = 4) | 1e+08 (M = 4) | 1e+08 (M = 4) |
| `h4_r2.0` | 8 | 1e+09 (M = 8) | 1e+08 (M = 4) | never | never | never |
| `h2o_2x_cas8e6o` | 12 | never | never | never | never | never |
| `h6_r2.0` | 12 | never | 1e+09 (M = 12) | never | never | never |
| `n2_r2.0_cas6e6o` | 12 | never | never | never | never | never |
| `h2o_2x_631g_cas8e8o` | 16 | never | never | never | never | never |
| `h8_r2.0` | 16 | never | never | never | never | never |
| `n2_r2.0_cas10e8o` | 16 | never | never | never | never | never |

Matched cells won: for every basis size and budget at which at least two arms return an energy, the arm with the smallest median absolute error wins the cell. The last column counts the cells where the better Krylov arm beats every A-CASE arm present:

| System | Qubits | Real-time Krylov | Power Krylov | A-CASE, excitations + commutators | A-CASE, commutators only | A-CASE, excitations only | Krylov beats A-CASE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `h2o_2x_cas4e3o` | 6 | 9 | 25 | 0 | 1 | 1 | 7/9 |
| `h4_r2.0` | 8 | 3 | 33 | 0 | 0 | 0 | 36/36 |
| `h2o_2x_cas8e6o` | 12 | 17 | 16 | 2 | 0 | 1 | 33/36 |
| `h6_r2.0` | 12 | 0 | 36 | 0 | 0 | 0 | 36/36 |
| `n2_r2.0_cas6e6o` | 12 | 15 | 17 | 0 | 1 | 3 | 32/36 |
| `h2o_2x_631g_cas8e8o` | 16 | 1 | 29 | 3 | 0 | 3 | 30/36 |
| `h8_r2.0` | 16 | 2 | 31 | 1 | 0 | 2 | 33/36 |
| `n2_r2.0_cas10e8o` | 16 | 6 | 27 | 0 | 2 | 1 | 33/36 |
| All | | 53 | 214 | 6 | 4 | 11 | 240/261 |

Best median absolute error (mHa) over the basis sizes, for the better Krylov arm and the better A-CASE arm, by total shots. The basis size is chosen with knowledge of the target, for both families alike, so this is an optimistic summary, not a procedure:

| System | Qubits | 1e+06: Krylov / A-CASE | 1e+08: Krylov / A-CASE | 1e+10: Krylov / A-CASE | 1e+12: Krylov / A-CASE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `h2o_2x_cas4e3o` | 6 | 5.1 / 11.4 | 0.4 / 1.2 | 0.1 / 0.1 | 0.0 / 0.0 |
| `h4_r2.0` | 8 | 5.2 / 40.2 | 0.8 / 37.3 | 0.7 / 32.0 | 0.3 / 32.8 |
| `h2o_2x_cas8e6o` | 12 | 73.2 / 89.3 | 19.3 / 38.1 | 4.9 / 36.2 | 2.4 / 36.3 |
| `h6_r2.0` | 12 | 17.1 / 106.5 | 2.3 / 151.1 | 1.9 / 156.0 | 1.4 / 156.2 |
| `n2_r2.0_cas6e6o` | 12 | 110.9 / 74.1 | 65.4 / 88.0 | 14.2 / 117.9 | 11.5 / 117.9 |
| `h2o_2x_631g_cas8e8o` | 16 | 147.4 / 96.7 | 28.5 / 29.4 | 19.0 / 39.5 | 10.9 / 40.0 |
| `h8_r2.0` | 16 | 135.6 / 141.2 | 9.5 / 340.7 | 3.2 / 359.4 | 2.6 / 359.2 |
| `n2_r2.0_cas10e8o` | 16 | 112.8 / 93.4 | 73.7 / 113.5 | 44.3 / 136.4 | 19.3 / 137.5 |

Bias at M = 8: median signed error against the target (mHa) and, in brackets, the fraction of replicas more than 1.6 mHa below it, by total shots. An energy below the target is not variational: it is shot noise, not a better subspace:

| System | Arm | 1e+06 shots | 1e+08 shots | 1e+10 shots | 1e+12 shots |
| --- | --- | ---: | ---: | ---: | ---: |
| `h2o_2x_cas4e3o` | Real-time Krylov | -0.5 (0.46) | -0.9 (0.41) | 0.0 (0.00) | 0.0 (0.01) |
| `h2o_2x_cas4e3o` | Power Krylov | -1.6 (0.50) | 0.0 (0.01) | 0.0 (0.00) | 0.0 (0.17) |
| `h4_r2.0` | Real-time Krylov | -2.3 (0.51) | 0.7 (0.17) | 1.1 (0.02) | 0.4 (0.34) |
| `h4_r2.0` | Power Krylov | -7.7 (0.66) | 0.6 (0.02) | 0.2 (0.45) | 0.2 (0.38) |
| `h4_r2.0` | A-CASE, excitations + commutators | 28.8 (0.34) | 50.4 (0.00) | 50.1 (0.00) | 50.1 (0.00) |
| `h4_r2.0` | A-CASE, commutators only | -74.1 (0.68) | 55.2 (0.01) | 55.6 (0.00) | 55.7 (0.00) |
| `h4_r2.0` | A-CASE, excitations only | 31.7 (0.26) | 51.8 (0.00) | 50.6 (0.00) | 50.6 (0.00) |
| `h2o_2x_cas8e6o` | Real-time Krylov | -496.2 (0.83) | -1,164.8 (0.75) | 5.6 (0.17) | 3.1 (0.37) |
| `h2o_2x_cas8e6o` | Power Krylov | 22.9 (0.43) | -16.5 (0.58) | 6.5 (0.29) | 3.6 (0.35) |
| `h2o_2x_cas8e6o` | A-CASE, excitations + commutators | -418.0 (0.96) | 37.1 (0.08) | 41.9 (0.00) | 42.3 (0.00) |
| `h2o_2x_cas8e6o` | A-CASE, commutators only | -687.0 (0.96) | 120.6 (0.00) | 128.7 (0.00) | 128.8 (0.00) |
| `h2o_2x_cas8e6o` | A-CASE, excitations only | -381.8 (0.99) | 55.0 (0.01) | 59.9 (0.00) | 59.7 (0.00) |
| `h6_r2.0` | Real-time Krylov | -29.4 (0.61) | -2.4 (0.52) | 2.5 (0.14) | 2.2 (0.01) |
| `h6_r2.0` | Power Krylov | -71.1 (0.79) | -6.6 (0.67) | 1.9 (0.00) | 1.5 (0.28) |
| `h6_r2.0` | A-CASE, excitations + commutators | 89.1 (0.20) | 206.3 (0.00) | 208.5 (0.00) | 208.4 (0.00) |
| `h6_r2.0` | A-CASE, commutators only | 212.1 (0.07) | 395.2 (0.00) | 397.1 (0.00) | 397.2 (0.00) |
| `h6_r2.0` | A-CASE, excitations only | 112.4 (0.22) | 207.1 (0.00) | 208.6 (0.00) | 208.4 (0.00) |
| `n2_r2.0_cas6e6o` | Real-time Krylov | -92.3 (0.54) | 54.8 (0.17) | 18.9 (0.32) | 22.7 (0.10) |
| `n2_r2.0_cas6e6o` | Power Krylov | 83.2 (0.34) | 71.9 (0.09) | 9.6 (0.47) | -6.9 (0.55) |
| `n2_r2.0_cas6e6o` | A-CASE, excitations + commutators | -890.1 (0.97) | 148.2 (0.00) | 152.5 (0.00) | 152.4 (0.00) |
| `n2_r2.0_cas6e6o` | A-CASE, commutators only | -248.9 (0.95) | 184.7 (0.00) | 190.4 (0.00) | 190.3 (0.00) |
| `n2_r2.0_cas6e6o` | A-CASE, excitations only | -30.2 (0.65) | 191.2 (0.00) | 192.6 (0.00) | 192.5 (0.00) |
| `h2o_2x_631g_cas8e8o` | Real-time Krylov | 49.3 (0.43) | -26.2 (0.52) | 42.9 (0.17) | 18.2 (0.39) |
| `h2o_2x_631g_cas8e8o` | Power Krylov | -94.0 (0.58) | 30.9 (0.30) | 7.9 (0.46) | 16.7 (0.03) |
| `h2o_2x_631g_cas8e8o` | A-CASE, excitations + commutators | -892.0 (1.00) | 30.2 (0.17) | 45.6 (0.00) | 45.8 (0.00) |
| `h2o_2x_631g_cas8e8o` | A-CASE, commutators only | -1,042.5 (0.99) | 97.6 (0.10) | 113.7 (0.00) | 113.1 (0.00) |
| `h2o_2x_631g_cas8e8o` | A-CASE, excitations only | -922.4 (1.00) | 24.4 (0.23) | 46.0 (0.00) | 45.8 (0.00) |
| `h8_r2.0` | Real-time Krylov | -398.5 (0.73) | -30.0 (0.59) | 6.5 (0.39) | 1.6 (0.45) |
| `h8_r2.0` | Power Krylov | -287.8 (0.80) | -66.0 (0.77) | -2.6 (0.52) | 1.6 (0.45) |
| `h8_r2.0` | A-CASE, excitations + commutators | -130.7 (0.73) | 419.4 (0.00) | 428.7 (0.00) | 428.8 (0.00) |
| `h8_r2.0` | A-CASE, commutators only | -463.3 (0.86) | 504.1 (0.00) | 515.0 (0.00) | 514.7 (0.00) |
| `h8_r2.0` | A-CASE, excitations only | -27.9 (0.55) | 425.1 (0.00) | 427.5 (0.00) | 427.7 (0.00) |
| `n2_r2.0_cas10e8o` | Real-time Krylov | -185.3 (0.61) | 54.7 (0.42) | -152.1 (0.55) | 40.7 (0.28) |
| `n2_r2.0_cas10e8o` | Power Krylov | 100.1 (0.32) | 77.8 (0.27) | 64.9 (0.24) | 57.9 (0.15) |
| `n2_r2.0_cas10e8o` | A-CASE, excitations + commutators | -3,907.6 (1.00) | 102.3 (0.23) | 171.2 (0.00) | 172.0 (0.00) |
| `n2_r2.0_cas10e8o` | A-CASE, commutators only | -1,635.4 (1.00) | 139.5 (0.06) | 176.2 (0.00) | 176.2 (0.00) |
| `n2_r2.0_cas10e8o` | A-CASE, excitations only | -653.4 (1.00) | 194.1 (0.00) | 210.6 (0.00) | 210.6 (0.00) |

Median absolute error against the target (mHa) at M = 4, by total shots. Brackets give the median deviation from the arm's own exact energy:

| System | Arm | 1e+06 shots | 1e+08 shots | 1e+10 shots | 1e+12 shots |
| --- | --- | ---: | ---: | ---: | ---: |
| `h2o_2x_cas4e3o` | Real-time Krylov | 7.0 [7.0] | 1.6 [1.6] | 0.1 [0.1] | 0.0 [0.0] |
| `h2o_2x_cas4e3o` | Power Krylov | 6.2 [6.2] | 0.8 [0.8] | 0.1 [0.1] | 0.1 [0.1] |
| `h2o_2x_cas4e3o` | A-CASE, excitations + commutators | 12.6 [12.6] | 1.5 [1.5] | 0.1 [0.1] | 0.0 [0.0] |
| `h2o_2x_cas4e3o` | A-CASE, commutators only | 18.8 [18.8] | 1.2 [1.2] | 0.1 [0.1] | 0.0 [0.0] |
| `h2o_2x_cas4e3o` | A-CASE, excitations only | 11.4 [11.4] | 1.4 [1.4] | 0.1 [0.1] | 0.0 [0.0] |
| `h4_r2.0` | Real-time Krylov | 18.4 [18.1] | 4.2 [3.2] | 1.4 [0.5] | 1.3 [0.1] |
| `h4_r2.0` | Power Krylov | 12.3 [12.3] | 1.5 [1.1] | 1.0 [0.1] | 1.1 [0.0] |
| `h4_r2.0` | A-CASE, excitations + commutators | 109.1 [18.8] | 109.2 [1.9] | 109.1 [0.2] | 109.1 [0.0] |
| `h4_r2.0` | A-CASE, commutators only | 238.1 [24.2] | 241.4 [2.4] | 241.1 [0.3] | 241.2 [0.0] |
| `h4_r2.0` | A-CASE, excitations only | 105.2 [16.3] | 108.8 [1.8] | 109.1 [0.2] | 109.1 [0.0] |
| `h2o_2x_cas8e6o` | Real-time Krylov | 210.3 [209.1] | 56.9 [51.3] | 14.1 [7.3] | 11.1 [0.7] |
| `h2o_2x_cas8e6o` | Power Krylov | 187.0 [178.7] | 80.9 [72.8] | 60.5 [53.1] | 25.2 [17.0] |
| `h2o_2x_cas8e6o` | A-CASE, excitations + commutators | 96.7 [100.8] | 109.1 [8.5] | 108.1 [0.9] | 108.1 [0.1] |
| `h2o_2x_cas8e6o` | A-CASE, commutators only | 163.7 [146.3] | 194.9 [15.5] | 195.3 [1.5] | 195.5 [0.1] |
| `h2o_2x_cas8e6o` | A-CASE, excitations only | 89.3 [92.5] | 107.8 [8.6] | 108.1 [0.8] | 108.1 [0.1] |
| `h6_r2.0` | Real-time Krylov | 178.1 [173.7] | 55.4 [51.3] | 17.4 [15.0] | 4.3 [1.5] |
| `h6_r2.0` | Power Krylov | 79.7 [79.1] | 16.6 [15.5] | 3.9 [1.3] | 3.7 [0.1] |
| `h6_r2.0` | A-CASE, excitations + commutators | 322.7 [46.9] | 335.7 [3.6] | 336.1 [0.4] | 336.1 [0.0] |
| `h6_r2.0` | A-CASE, commutators only | 434.5 [39.4] | 446.5 [4.1] | 445.9 [0.4] | 445.8 [0.0] |
| `h6_r2.0` | A-CASE, excitations only | 324.2 [44.0] | 337.1 [4.0] | 336.0 [0.4] | 336.1 [0.0] |
| `n2_r2.0_cas6e6o` | Real-time Krylov | 216.0 [124.5] | 104.9 [38.4] | 92.2 [4.2] | 93.0 [0.5] |
| `n2_r2.0_cas6e6o` | Power Krylov | 149.1 [127.8] | 80.4 [12.1] | 80.7 [1.3] | 80.9 [0.1] |
| `n2_r2.0_cas6e6o` | A-CASE, excitations + commutators | 155.3 [88.0] | 175.9 [5.5] | 176.9 [0.5] | 176.9 [0.1] |
| `n2_r2.0_cas6e6o` | A-CASE, commutators only | 171.8 [48.8] | 217.6 [3.4] | 218.5 [0.4] | 218.4 [0.0] |
| `n2_r2.0_cas6e6o` | A-CASE, excitations only | 186.4 [42.5] | 220.0 [3.6] | 220.6 [0.4] | 220.6 [0.0] |
| `h2o_2x_631g_cas8e8o` | Real-time Krylov | 233.1 [222.0] | 115.9 [95.4] | 83.6 [60.9] | 56.7 [37.9] |
| `h2o_2x_631g_cas8e8o` | Power Krylov | 148.4 [131.7] | 99.8 [79.8] | 61.3 [38.7] | 45.0 [24.7] |
| `h2o_2x_631g_cas8e8o` | A-CASE, excitations + commutators | 104.4 [137.8] | 73.6 [13.2] | 76.3 [1.1] | 76.5 [0.1] |
| `h2o_2x_631g_cas8e8o` | A-CASE, commutators only | 164.3 [163.4] | 138.4 [15.8] | 136.3 [1.4] | 136.4 [0.2] |
| `h2o_2x_631g_cas8e8o` | A-CASE, excitations only | 96.7 [128.5] | 75.5 [11.7] | 76.5 [1.2] | 76.5 [0.1] |
| `h8_r2.0` | Real-time Krylov | 432.9 [410.7] | 132.5 [113.1] | 115.5 [94.8] | 32.0 [19.5] |
| `h8_r2.0` | Power Krylov | 377.4 [398.9] | 662.6 [684.1] | 22.7 [8.2] | 21.7 [0.9] |
| `h8_r2.0` | A-CASE, excitations + commutators | 378.3 [115.4] | 469.2 [12.3] | 470.0 [1.1] | 469.9 [0.1] |
| `h8_r2.0` | A-CASE, commutators only | 525.6 [85.0] | 594.9 [8.3] | 596.2 [0.9] | 595.9 [0.1] |
| `h8_r2.0` | A-CASE, excitations only | 476.0 [83.7] | 542.5 [8.2] | 543.8 [0.8] | 543.8 [0.1] |
| `n2_r2.0_cas10e8o` | Real-time Krylov | 212.2 [129.8] | 193.1 [93.9] | 125.5 [14.0] | 127.8 [1.6] |
| `n2_r2.0_cas10e8o` | Power Krylov | 204.3 [111.8] | 141.3 [39.1] | 132.3 [17.5] | 117.5 [3.4] |
| `n2_r2.0_cas10e8o` | A-CASE, excitations + commutators | 341.9 [537.4] | 191.5 [15.0] | 195.6 [1.5] | 195.5 [0.1] |
| `n2_r2.0_cas10e8o` | A-CASE, commutators only | 93.4 [205.4] | 244.0 [11.9] | 248.2 [1.0] | 248.2 [0.1] |
| `n2_r2.0_cas10e8o` | A-CASE, excitations only | 95.4 [165.4] | 236.4 [10.3] | 238.7 [0.9] | 238.6 [0.1] |

Median absolute error against the target (mHa) at M = 6, by total shots. Brackets give the median deviation from the arm's own exact energy:

| System | Arm | 1e+06 shots | 1e+08 shots | 1e+10 shots | 1e+12 shots |
| --- | --- | ---: | ---: | ---: | ---: |
| `h2o_2x_cas4e3o` | Real-time Krylov | 7.0 [7.0] | 1.5 [1.5] | 0.1 [0.1] | 0.0 [0.0] |
| `h2o_2x_cas4e3o` | Power Krylov | 5.1 [5.1] | 0.5 [0.5] | 0.1 [0.1] | 0.0 [0.0] |
| `h4_r2.0` | Real-time Krylov | 19.7 [20.0] | 1.8 [1.5] | 1.1 [0.4] | 1.0 [0.3] |
| `h4_r2.0` | Power Krylov | 10.1 [9.9] | 1.0 [0.8] | 0.9 [0.2] | 0.8 [0.2] |
| `h4_r2.0` | A-CASE, excitations + commutators | 43.0 [26.4] | 51.4 [2.7] | 51.4 [0.3] | 51.5 [0.0] |
| `h4_r2.0` | A-CASE, commutators only | 116.4 [81.0] | 83.1 [8.1] | 82.7 [0.9] | 82.8 [0.1] |
| `h4_r2.0` | A-CASE, excitations only | 49.3 [25.6] | 51.1 [2.8] | 51.4 [0.2] | 51.5 [0.0] |
| `h2o_2x_cas8e6o` | Real-time Krylov | 112.7 [112.5] | 36.7 [34.3] | 8.3 [4.4] | 7.4 [3.4] |
| `h2o_2x_cas8e6o` | Power Krylov | 124.8 [122.8] | 49.8 [47.2] | 22.4 [19.8] | 11.6 [8.8] |
| `h2o_2x_cas8e6o` | A-CASE, excitations + commutators | 172.8 [205.2] | 48.1 [15.6] | 48.3 [1.7] | 48.1 [0.2] |
| `h2o_2x_cas8e6o` | A-CASE, commutators only | 248.9 [363.5] | 150.1 [22.1] | 155.2 [2.3] | 155.2 [0.2] |
| `h2o_2x_cas8e6o` | A-CASE, excitations only | 138.8 [187.6] | 57.1 [12.8] | 61.3 [1.4] | 61.6 [0.1] |
| `h6_r2.0` | Real-time Krylov | 127.8 [125.8] | 14.6 [14.1] | 6.7 [5.4] | 2.6 [0.9] |
| `h6_r2.0` | Power Krylov | 59.8 [61.6] | 8.6 [7.6] | 2.3 [0.6] | 2.0 [0.3] |
| `h6_r2.0` | A-CASE, excitations + commutators | 217.8 [60.9] | 264.5 [5.9] | 264.9 [0.8] | 264.9 [0.1] |
| `h6_r2.0` | A-CASE, commutators only | 332.8 [90.5] | 412.9 [8.4] | 414.1 [0.7] | 414.0 [0.1] |
| `h6_r2.0` | A-CASE, excitations only | 205.3 [65.6] | 264.7 [6.8] | 264.9 [0.8] | 264.9 [0.1] |
| `n2_r2.0_cas6e6o` | Real-time Krylov | 110.9 [97.2] | 113.4 [99.4] | 50.9 [36.6] | 48.7 [33.6] |
| `n2_r2.0_cas6e6o` | Power Krylov | 173.6 [171.9] | 74.8 [64.8] | 58.3 [50.0] | 25.1 [16.8] |
| `n2_r2.0_cas6e6o` | A-CASE, excitations + commutators | 141.2 [272.2] | 163.9 [9.1] | 166.8 [1.0] | 166.9 [0.1] |
| `n2_r2.0_cas6e6o` | A-CASE, commutators only | 74.6 [136.1] | 202.3 [6.0] | 203.3 [0.6] | 203.5 [0.1] |
| `n2_r2.0_cas6e6o` | A-CASE, excitations only | 98.4 [108.4] | 207.1 [5.8] | 206.9 [0.6] | 206.9 [0.1] |
| `h2o_2x_631g_cas8e8o` | Real-time Krylov | 213.7 [209.5] | 81.6 [74.2] | 68.2 [60.8] | 41.3 [34.0] |
| `h2o_2x_631g_cas8e8o` | Power Krylov | 165.2 [158.1] | 66.6 [61.2] | 48.9 [42.4] | 25.0 [18.4] |
| `h2o_2x_631g_cas8e8o` | A-CASE, excitations + commutators | 431.8 [481.7] | 44.1 [18.0] | 50.2 [1.9] | 49.9 [0.2] |
| `h2o_2x_631g_cas8e8o` | A-CASE, commutators only | 531.9 [652.4] | 118.8 [27.5] | 120.4 [2.8] | 120.5 [0.3] |
| `h2o_2x_631g_cas8e8o` | A-CASE, excitations only | 439.1 [488.9] | 42.4 [20.9] | 50.0 [1.9] | 49.8 [0.2] |
| `h8_r2.0` | Real-time Krylov | 244.5 [241.1] | 130.2 [127.2] | 34.5 [31.3] | 12.4 [10.7] |
| `h8_r2.0` | Power Krylov | 306.5 [309.7] | 201.0 [204.2] | 13.9 [17.1] | 3.3 [0.8] |
| `h8_r2.0` | A-CASE, excitations + commutators | 210.4 [280.2] | 445.8 [17.9] | 449.0 [1.7] | 449.2 [0.2] |
| `h8_r2.0` | A-CASE, commutators only | 230.1 [421.4] | 554.4 [18.7] | 560.5 [1.7] | 560.0 [0.2] |
| `h8_r2.0` | A-CASE, excitations only | 258.3 [221.6] | 478.8 [13.5] | 480.3 [1.5] | 480.0 [0.1] |
| `n2_r2.0_cas10e8o` | Real-time Krylov | 301.0 [332.6] | 145.0 [84.3] | 103.6 [32.4] | 79.0 [16.7] |
| `n2_r2.0_cas10e8o` | Power Krylov | 164.0 [119.0] | 120.6 [72.0] | 95.0 [45.6] | 76.9 [24.2] |
| `n2_r2.0_cas10e8o` | A-CASE, excitations + commutators | 1,490.1 [1,675.7] | 173.3 [27.5] | 185.4 [2.3] | 185.6 [0.2] |
| `n2_r2.0_cas10e8o` | A-CASE, commutators only | 564.9 [768.9] | 186.0 [29.3] | 203.6 [3.0] | 204.1 [0.3] |
| `n2_r2.0_cas10e8o` | A-CASE, excitations only | 262.6 [487.6] | 216.6 [14.0] | 224.5 [1.9] | 224.9 [0.2] |

Median absolute error against the target (mHa) at M = 8, by total shots. Brackets give the median deviation from the arm's own exact energy:

| System | Arm | 1e+06 shots | 1e+08 shots | 1e+10 shots | 1e+12 shots |
| --- | --- | ---: | ---: | ---: | ---: |
| `h2o_2x_cas4e3o` | Real-time Krylov | 7.2 [7.2] | 1.4 [1.4] | 0.1 [0.1] | 0.0 [0.0] |
| `h2o_2x_cas4e3o` | Power Krylov | 5.4 [5.4] | 0.6 [0.6] | 0.1 [0.1] | 0.0 [0.0] |
| `h4_r2.0` | Real-time Krylov | 15.3 [15.3] | 1.7 [1.7] | 1.1 [1.1] | 0.9 [0.9] |
| `h4_r2.0` | Power Krylov | 10.3 [10.3] | 0.8 [0.8] | 0.8 [0.8] | 0.6 [0.6] |
| `h4_r2.0` | A-CASE, excitations + commutators | 40.7 [41.9] | 50.4 [3.9] | 50.1 [0.3] | 50.1 [0.0] |
| `h4_r2.0` | A-CASE, commutators only | 138.1 [147.4] | 55.2 [14.2] | 55.6 [1.6] | 55.7 [0.2] |
| `h4_r2.0` | A-CASE, excitations only | 40.2 [38.0] | 51.8 [3.8] | 50.6 [0.3] | 50.6 [0.0] |
| `h2o_2x_cas8e6o` | Real-time Krylov | 496.2 [497.3] | 1,164.8 [1,165.9] | 6.7 [5.7] | 5.2 [4.1] |
| `h2o_2x_cas8e6o` | Power Krylov | 93.9 [93.3] | 43.2 [42.0] | 10.0 [9.8] | 6.7 [5.5] |
| `h2o_2x_cas8e6o` | A-CASE, excitations + commutators | 418.0 [460.3] | 38.1 [21.0] | 41.9 [2.0] | 42.3 [0.2] |
| `h2o_2x_cas8e6o` | A-CASE, commutators only | 687.0 [815.8] | 120.6 [35.0] | 128.7 [2.6] | 128.8 [0.3] |
| `h2o_2x_cas8e6o` | A-CASE, excitations only | 381.8 [441.5] | 55.0 [19.4] | 59.9 [1.6] | 59.7 [0.2] |
| `h6_r2.0` | Real-time Krylov | 75.5 [76.5] | 11.4 [10.8] | 3.3 [2.1] | 2.2 [0.9] |
| `h6_r2.0` | Power Krylov | 71.1 [72.3] | 8.4 [8.5] | 1.9 [0.8] | 1.7 [0.6] |
| `h6_r2.0` | A-CASE, excitations + commutators | 106.5 [125.3] | 206.3 [10.0] | 208.5 [0.9] | 208.4 [0.1] |
| `h6_r2.0` | A-CASE, commutators only | 219.2 [185.2] | 395.2 [8.9] | 397.1 [1.0] | 397.2 [0.1] |
| `h6_r2.0` | A-CASE, excitations only | 125.1 [113.8] | 207.1 [9.6] | 208.6 [1.0] | 208.4 [0.1] |
| `n2_r2.0_cas6e6o` | Real-time Krylov | 183.3 [182.8] | 65.9 [65.5] | 34.3 [33.9] | 23.3 [22.9] |
| `n2_r2.0_cas6e6o` | Power Krylov | 146.0 [145.7] | 74.5 [74.2] | 61.7 [61.4] | 27.8 [27.5] |
| `n2_r2.0_cas6e6o` | A-CASE, excitations + commutators | 890.1 [1,042.5] | 148.2 [13.9] | 152.5 [1.5] | 152.4 [0.1] |
| `n2_r2.0_cas6e6o` | A-CASE, commutators only | 248.9 [439.2] | 184.7 [8.6] | 190.4 [1.0] | 190.3 [0.1] |
| `n2_r2.0_cas6e6o` | A-CASE, excitations only | 74.1 [222.7] | 191.2 [8.5] | 192.6 [1.0] | 192.5 [0.1] |
| `h2o_2x_631g_cas8e8o` | Real-time Krylov | 162.4 [157.8] | 123.6 [119.0] | 49.0 [44.6] | 37.6 [33.6] |
| `h2o_2x_631g_cas8e8o` | Power Krylov | 147.4 [147.4] | 46.0 [43.6] | 35.2 [32.4] | 16.7 [13.8] |
| `h2o_2x_631g_cas8e8o` | A-CASE, excitations + commutators | 892.0 [937.9] | 33.0 [23.3] | 45.6 [2.2] | 45.8 [0.3] |
| `h2o_2x_631g_cas8e8o` | A-CASE, commutators only | 1,042.5 [1,155.6] | 98.5 [44.6] | 113.7 [4.2] | 113.1 [0.4] |
| `h2o_2x_631g_cas8e8o` | A-CASE, excitations only | 922.4 [968.2] | 29.4 [30.6] | 46.0 [2.6] | 45.8 [0.2] |
| `h8_r2.0` | Real-time Krylov | 427.6 [428.7] | 74.5 [73.9] | 13.7 [11.8] | 7.3 [5.4] |
| `h8_r2.0` | Power Krylov | 287.8 [290.2] | 66.0 [68.5] | 4.9 [5.0] | 3.1 [1.0] |
| `h8_r2.0` | A-CASE, excitations + commutators | 173.7 [559.5] | 419.4 [23.5] | 428.7 [2.9] | 428.8 [0.3] |
| `h8_r2.0` | A-CASE, commutators only | 463.3 [977.9] | 504.1 [31.1] | 515.0 [2.6] | 514.7 [0.3] |
| `h8_r2.0` | A-CASE, excitations only | 141.2 [455.6] | 425.1 [18.3] | 427.5 [1.8] | 427.7 [0.2] |
| `n2_r2.0_cas10e8o` | Real-time Krylov | 271.1 [270.3] | 129.5 [112.9] | 152.1 [168.6] | 56.1 [42.5] |
| `n2_r2.0_cas10e8o` | Power Krylov | 143.5 [137.0] | 99.8 [91.8] | 75.0 [66.9] | 60.2 [51.7] |
| `n2_r2.0_cas10e8o` | A-CASE, excitations + commutators | 3,907.6 [4,079.7] | 124.6 [72.9] | 171.2 [4.7] | 172.0 [0.5] |
| `n2_r2.0_cas10e8o` | A-CASE, commutators only | 1,635.4 [1,811.5] | 140.1 [42.3] | 176.2 [4.0] | 176.2 [0.5] |
| `n2_r2.0_cas10e8o` | A-CASE, excitations only | 653.4 [864.0] | 194.1 [22.8] | 210.6 [2.7] | 210.6 [0.2] |

Median absolute error against the target (mHa) at M = 12, by total shots. Brackets give the median deviation from the arm's own exact energy:

| System | Arm | 1e+06 shots | 1e+08 shots | 1e+10 shots | 1e+12 shots |
| --- | --- | ---: | ---: | ---: | ---: |
| `h2o_2x_cas4e3o` | Real-time Krylov | 7.6 [7.6] | 36.2 [36.2] | 0.1 [0.1] | 0.0 [0.0] |
| `h2o_2x_cas4e3o` | Power Krylov | 5.3 [5.3] | 0.4 [0.4] | 0.1 [0.1] | 0.0 [0.0] |
| `h4_r2.0` | Real-time Krylov | 19.5 [19.5] | 190.1 [190.1] | 1.3 [1.3] | 0.3 [0.3] |
| `h4_r2.0` | Power Krylov | 5.2 [5.2] | 0.9 [0.9] | 0.7 [0.7] | 0.3 [0.3] |
| `h4_r2.0` | A-CASE, excitations + commutators | 96.8 [135.8] | 37.3 [6.1] | 39.0 [0.6] | 39.0 [0.1] |
| `h4_r2.0` | A-CASE, commutators only | 322.0 [354.7] | 38.9 [27.0] | 32.0 [2.7] | 32.8 [0.3] |
| `h4_r2.0` | A-CASE, excitations only | 64.3 [107.6] | 44.3 [5.8] | 46.0 [0.4] | 46.1 [0.1] |
| `h2o_2x_cas8e6o` | Real-time Krylov | 1,713.8 [1,713.8] | 787.8 [787.8] | 8.8 [8.8] | 2.4 [2.4] |
| `h2o_2x_cas8e6o` | Power Krylov | 73.2 [73.1] | 19.3 [19.4] | 4.9 [4.8] | 4.1 [4.0] |
| `h2o_2x_cas8e6o` | A-CASE, excitations + commutators | 1,240.6 [1,276.9] | 40.3 [36.9] | 36.2 [3.7] | 36.3 [0.4] |
| `h2o_2x_cas8e6o` | A-CASE, commutators only | 2,197.7 [2,292.3] | 96.4 [58.1] | 94.6 [6.0] | 94.7 [0.6] |
| `h2o_2x_cas8e6o` | A-CASE, excitations only | 1,132.8 [1,190.6] | 41.8 [29.0] | 57.9 [2.8] | 57.9 [0.3] |
| `h6_r2.0` | Real-time Krylov | 102.1 [102.1] | 7.3 [7.2] | 2.7 [2.4] | 2.0 [1.8] |
| `h6_r2.0` | Power Krylov | 17.1 [17.1] | 2.3 [2.2] | 1.9 [1.8] | 1.4 [1.3] |
| `h6_r2.0` | A-CASE, excitations + commutators | 249.9 [406.1] | 152.4 [14.2] | 156.0 [1.3] | 156.2 [0.1] |
| `h6_r2.0` | A-CASE, commutators only | 350.2 [711.5] | 355.4 [22.8] | 361.3 [2.0] | 361.4 [0.2] |
| `h6_r2.0` | A-CASE, excitations only | 224.7 [380.9] | 151.1 [14.1] | 156.0 [1.3] | 156.2 [0.2] |
| `n2_r2.0_cas6e6o` | Real-time Krylov | 146.1 [146.1] | 65.4 [65.4] | 14.2 [14.2] | 11.5 [11.5] |
| `n2_r2.0_cas6e6o` | Power Krylov | 120.3 [120.3] | 88.2 [88.1] | 67.0 [67.0] | 25.3 [25.3] |
| `n2_r2.0_cas6e6o` | A-CASE, excitations + commutators | 1,089.8 [1,207.7] | 88.0 [32.3] | 117.9 [2.2] | 117.9 [0.2] |
| `n2_r2.0_cas6e6o` | A-CASE, commutators only | 1,580.8 [1,751.9] | 146.6 [26.2] | 170.9 [2.0] | 171.0 [0.2] |
| `n2_r2.0_cas6e6o` | A-CASE, excitations only | 422.1 [597.9] | 164.3 [15.7] | 175.9 [1.5] | 175.8 [0.2] |
| `h2o_2x_631g_cas8e8o` | Real-time Krylov | 167.7 [166.5] | 79.3 [77.3] | 44.2 [42.1] | 24.8 [22.6] |
| `h2o_2x_631g_cas8e8o` | Power Krylov | 179.3 [179.3] | 28.5 [28.3] | 19.0 [17.7] | 10.9 [9.9] |
| `h2o_2x_631g_cas8e8o` | A-CASE, excitations + commutators | 2,029.8 [2,069.8] | 30.6 [53.6] | 39.5 [3.4] | 40.0 [0.4] |
| `h2o_2x_631g_cas8e8o` | A-CASE, commutators only | 2,865.9 [2,970.7] | 75.7 [73.4] | 104.3 [7.2] | 104.9 [0.7] |
| `h2o_2x_631g_cas8e8o` | A-CASE, excitations only | 1,974.4 [2,014.8] | 29.9 [59.7] | 39.5 [3.4] | 40.4 [0.3] |
| `h8_r2.0` | Real-time Krylov | 195.3 [195.5] | 120.5 [121.6] | 5.8 [5.5] | 3.5 [2.4] |
| `h8_r2.0` | Power Krylov | 135.6 [136.0] | 9.5 [9.2] | 3.2 [2.9] | 2.6 [2.2] |
| `h8_r2.0` | A-CASE, excitations + commutators | 940.9 [1,325.3] | 362.2 [32.7] | 384.1 [3.7] | 384.4 [0.3] |
| `h8_r2.0` | A-CASE, commutators only | 1,923.5 [2,426.4] | 454.9 [53.6] | 502.3 [3.9] | 502.9 [0.4] |
| `h8_r2.0` | A-CASE, excitations only | 817.3 [1,176.5] | 340.7 [31.2] | 359.4 [2.4] | 359.2 [0.3] |
| `n2_r2.0_cas10e8o` | Real-time Krylov | 1,418.6 [1,419.3] | 971.6 [972.3] | 46.6 [46.4] | 19.3 [19.3] |
| `n2_r2.0_cas10e8o` | Power Krylov | 112.8 [112.0] | 73.7 [72.9] | 44.3 [43.5] | 27.7 [26.9] |
| `n2_r2.0_cas10e8o` | A-CASE, excitations + commutators | 4,741.6 [4,879.1] | 140.9 [276.2] | 136.4 [7.1] | 137.5 [0.6] |
| `n2_r2.0_cas10e8o` | A-CASE, commutators only | 1,828.2 [1,970.1] | 132.2 [174.5] | 144.5 [11.2] | 141.9 [1.2] |
| `n2_r2.0_cas10e8o` | A-CASE, excitations only | 1,843.0 [2,036.9] | 113.5 [80.4] | 193.3 [4.1] | 193.9 [0.3] |

Overlap-floor sensitivity: median absolute error (mHa) of the Krylov arms with the floor at 2 and at 20 times sqrt(vM/n), shown as 2× / 20×, by total shots:

| System | Arm | M | 1e+06 shots | 1e+08 shots | 1e+10 shots | 1e+12 shots |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `h2o_2x_cas4e3o` | Real-time Krylov | 8 | 7.2 / 7.3 | 1.4 / 0.7 | 0.1 / 0.1 | 0.0 / 0.0 |
| `h2o_2x_cas4e3o` | Real-time Krylov | 12 | 7.6 / 7.4 | 36.2 / 0.9 | 0.1 / 0.1 | 0.0 / 0.0 |
| `h2o_2x_cas4e3o` | Power Krylov | 8 | 5.4 / 4.8 | 0.6 / 0.6 | 0.1 / 0.1 | 0.0 / 0.0 |
| `h2o_2x_cas4e3o` | Power Krylov | 12 | 5.3 / 4.1 | 0.4 / 0.4 | 0.1 / 0.1 | 0.0 / 0.0 |
| `h4_r2.0` | Real-time Krylov | 8 | 15.3 / 13.3 | 1.7 / 1.8 | 1.1 / 1.3 | 0.9 / 1.1 |
| `h4_r2.0` | Real-time Krylov | 12 | 19.5 / 15.3 | 190.1 / 1.6 | 1.3 / 1.0 | 0.3 / 0.2 |
| `h4_r2.0` | Power Krylov | 8 | 10.3 / 6.9 | 0.8 / 0.8 | 0.8 / 0.8 | 0.6 / 0.6 |
| `h4_r2.0` | Power Krylov | 12 | 5.2 / 4.9 | 0.9 / 1.1 | 0.7 / 0.7 | 0.3 / 0.3 |
| `h2o_2x_cas8e6o` | Real-time Krylov | 8 | 496.2 / 68.5 | 1,164.8 / 12.8 | 6.7 / 6.6 | 5.2 / 5.8 |
| `h2o_2x_cas8e6o` | Real-time Krylov | 12 | 1,713.8 / 66.9 | 787.8 / 9.9 | 8.8 / 5.5 | 2.4 / 3.7 |
| `h2o_2x_cas8e6o` | Power Krylov | 8 | 93.9 / 115.5 | 43.2 / 30.6 | 10.0 / 10.0 | 6.7 / 7.3 |
| `h2o_2x_cas8e6o` | Power Krylov | 12 | 73.2 / 73.6 | 19.3 / 14.4 | 4.9 / 4.9 | 4.1 / 4.9 |
| `h6_r2.0` | Real-time Krylov | 8 | 75.5 / 55.3 | 11.4 / 8.1 | 3.3 / 3.2 | 2.2 / 3.1 |
| `h6_r2.0` | Real-time Krylov | 12 | 102.1 / 66.4 | 7.3 / 7.0 | 2.7 / 2.7 | 2.0 / 2.2 |
| `h6_r2.0` | Power Krylov | 8 | 71.1 / 20.4 | 8.4 / 3.5 | 1.9 / 1.9 | 1.7 / 1.9 |
| `h6_r2.0` | Power Krylov | 12 | 17.1 / 18.7 | 2.3 / 3.1 | 1.9 / 1.8 | 1.4 / 1.3 |
| `n2_r2.0_cas6e6o` | Real-time Krylov | 8 | 183.3 / 130.7 | 65.9 / 65.9 | 34.3 / 56.5 | 23.3 / 23.5 |
| `n2_r2.0_cas6e6o` | Real-time Krylov | 12 | 146.1 / 145.1 | 65.4 / 66.8 | 14.2 / 14.2 | 11.5 / 11.1 |
| `n2_r2.0_cas6e6o` | Power Krylov | 8 | 146.0 / 164.4 | 74.5 / 132.8 | 61.7 / 75.2 | 27.8 / 23.1 |
| `n2_r2.0_cas6e6o` | Power Krylov | 12 | 120.3 / 166.7 | 88.2 / 133.5 | 67.0 / 76.1 | 25.3 / 25.3 |
| `h2o_2x_631g_cas8e8o` | Real-time Krylov | 8 | 162.4 / 169.9 | 123.6 / 79.7 | 49.0 / 49.0 | 37.6 / 41.2 |
| `h2o_2x_631g_cas8e8o` | Real-time Krylov | 12 | 167.7 / 159.2 | 79.3 / 76.7 | 44.2 / 40.0 | 24.8 / 20.7 |
| `h2o_2x_631g_cas8e8o` | Power Krylov | 8 | 147.4 / 96.4 | 46.0 / 46.0 | 35.2 / 36.3 | 16.7 / 16.7 |
| `h2o_2x_631g_cas8e8o` | Power Krylov | 12 | 179.3 / 70.3 | 28.5 / 28.5 | 19.0 / 22.8 | 10.9 / 9.2 |
| `h8_r2.0` | Real-time Krylov | 8 | 427.6 / 159.4 | 74.5 / 44.1 | 13.7 / 15.0 | 7.3 / 6.2 |
| `h8_r2.0` | Real-time Krylov | 12 | 195.3 / 144.2 | 120.5 / 24.4 | 5.8 / 5.7 | 3.5 / 4.5 |
| `h8_r2.0` | Power Krylov | 8 | 287.8 / 53.7 | 66.0 / 10.4 | 4.9 / 3.4 | 3.1 / 2.9 |
| `h8_r2.0` | Power Krylov | 12 | 135.6 / 36.9 | 9.5 / 6.7 | 3.2 / 3.3 | 2.6 / 2.6 |
| `n2_r2.0_cas10e8o` | Real-time Krylov | 8 | 271.1 / 124.7 | 129.5 / 132.9 | 152.1 / 92.6 | 56.1 / 56.1 |
| `n2_r2.0_cas10e8o` | Real-time Krylov | 12 | 1,418.6 / 122.2 | 971.6 / 92.1 | 46.6 / 41.8 | 19.3 / 38.0 |
| `n2_r2.0_cas10e8o` | Power Krylov | 8 | 143.5 / 177.9 | 99.8 / 113.5 | 75.0 / 89.7 | 60.2 / 67.2 |
| `n2_r2.0_cas10e8o` | Power Krylov | 12 | 112.8 / 128.4 | 73.7 / 85.5 | 44.3 / 61.1 | 27.7 / 43.3 |

Matched cells won with the floor at 20 times sqrt(vM/n):

| System | Qubits | Real-time Krylov | Power Krylov | A-CASE, excitations + commutators | A-CASE, commutators only | A-CASE, excitations only | Krylov beats A-CASE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `h2o_2x_cas4e3o` | 6 | 12 | 23 | 0 | 0 | 1 | 8/9 |
| `h4_r2.0` | 8 | 3 | 33 | 0 | 0 | 0 | 34/34 |
| `h2o_2x_cas8e6o` | 12 | 26 | 9 | 0 | 0 | 1 | 31/32 |
| `h6_r2.0` | 12 | 0 | 35 | 0 | 0 | 0 | 31/31 |
| `n2_r2.0_cas6e6o` | 12 | 16 | 14 | 2 | 3 | 1 | 27/33 |
| `h2o_2x_631g_cas8e8o` | 16 | 0 | 29 | 3 | 0 | 4 | 25/32 |
| `h8_r2.0` | 16 | 0 | 34 | 0 | 0 | 1 | 31/32 |
| `n2_r2.0_cas10e8o` | 16 | 9 | 20 | 4 | 2 | 1 | 26/33 |
| All | | 66 | 197 | 9 | 5 | 9 | 213/236 |
<!-- end of generated tables -->
