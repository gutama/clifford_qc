# Symmetry-shift control variates

This adds the symmetry-shift step of the
[pencil improvement sequence](PENCIL_ESTIMATORS.md#validation-and-remaining-sequence)
to measured banks, and builds it into
[target-aware acquisition](TARGET_ACQUISITION.md). Everything is opt-in;
no default and no frozen record changes. The evidence below is exploratory:
the systems, seeds, schedule and targets were chosen while the method was
being developed, not declared beforehand.

## The identity

A measured bank estimates `S_ij = <A_i† A_j>` and `H_ij = <A_i† H A_j>` from
Pauli-word means. Let `Q` be a conserved charge (particle number `N`, or
`S_z`), let the reference lie in the sector `Q = q`, and let every generator
commute with `Q`. Every bank state `A_j|psi>` then lies in that sector, and
any `K = (Q - q) O` with `[Q, O] = 0` annihilates it:

    <psi| A_i† (H - K) A_j |psi> = <psi| A_i† H A_j |psi>   for every i, j.

The matrices, the solve and every exact quantity are unchanged. What changes
is the Pauli expansion that is measured, and so the shot variance of every
estimate built from it. This is the symmetry shift of Lee et al.
(arXiv:2409.02504) and the BLISS family of Loaiza and Izmaylov
(arXiv:2304.13772), applied to a bank's Ritz functional rather than to `H`
alone.

Measured as one observable, `f = B†(H - E)B` and `f - B†KB` have the same
variance. A shift helps only through how its words split across measurement
settings, so any objective has to include within-setting covariance; a
per-word objective cannot see the benefit.

## Interfaces

All in [`clifford_qc/subspace/symmetry_shifts.py`](../../clifford_qc/subspace/symmetry_shifts.py),
except the acquisition option.

- `sector_shift_family(n_spatial, n_electrons, one_body=True, sz=None,
  spin_ordering="interleaved")` returns `ShiftTerm`s: `N - N_e`,
  `(N - N_e) N`, `(N - N_e)(E_pq + E_qp)` for `p <= q` with `one_body`, and
  `S_z - sz`, `(S_z - sz) N` with `sz`.
- `certify_bank_family(bank, family)` refuses a family unless every sector
  factor vanishes on the bank's reference (`<D^2> = 0`) and commutes with its
  multiplier and with every generator, within `1e-9`. Every entry point below
  certifies before it uses a family, except `corrected_ritz_functional`, the
  per-look step for a caller that has already certified.
- `fit_sector_shift(H, family, objective=...)` returns the shifted
  Hamiltonian. `"hamiltonian"` minimizes the squared non-identity coefficients
  (the state-free BLISS-style proxy). `"ritz"` minimizes the grouped Neyman
  variance `(sum_g sigma_g)^2` of the Ritz functional on a fixed
  qubit-wise-commuting design, by iteratively reweighted least squares.
  `grouped_shot_variance` evaluates any word functional after regrouping.
- `ritz_control_variate(shared, cache, point, family)` applies the shift
  after the fact, with no new shot and no rebuilt bank. Each
  `g_k = B† K_k B` is a fixed combination of word means already in the cache,
  with exactly zero expectation at any fixed `B`. The corrected value is
  `E - sum_k xi_k g_k`, with `xi` fitted from the cache's grouped covariances
  (`xi = Cov(g)^-1 Cov(g, f)`) or supplied. Terms with a word no setting reads
  are left out and named in `unreadable`.
- `acquire_ritz(..., control_variates=family)` refits `xi` at every look. The
  corrected value is what each look reports in `energy` and `std_error`, what
  the stopping rule tests and, under `"ritz_targeted"`, what the next round's
  Neyman allocation targets: the corrected functional `f - sum_k xi_k g_k` is
  itself a word functional. The raw values stay in the ledger as `raw_energy`
  and `raw_std_error`; `result` is the uncorrected solve; both intervals are
  centred on the corrected value with the same Bonferroni schedule as before.
  Under uniform allocation the correction changes no shot.
- `shift_pair_table(bank, indices, family)` expands every `A_a† K_k A_b` once,
  after which each `B† K_k B` is a sparse product with the Ritz vector instead
  of three operator products. `acquire_ritz` builds it once per run and uses it
  at every look and in `bootstrap_replicates`, which with control variates
  re-solves every resample and refits its correction. Both paths keep a word
  when its real coefficient exceeds the operator tolerance, and agree to
  rounding.

## Limits

- **The generators must conserve the charge.** None of the declared spin banks
  qualifies. Their references are parity or magnetization eigenstates, but
  their pool words (odd-`Y` Pauli words) break both, so the bank states leave
  the sector and no shift term annihilates them. Certification refuses such
  a family rather than returning a biased estimate. The molecular banks here
  use spin-adapted excitations ([SYMMETRY_POOLS.md](SYMMETRY_POOLS.md)),
  which conserve `N` and `S_z`.
- **A fitted `xi` is not free.** Fitting it from the same data adds an
  `O(1/N)` bias and makes the standard error slightly optimistic. Below, the
  corrected targeted runs on the six-qubit bank stop with an actual error
  8–16% above the target; the schedule-valid interval still covers.
- **It reduces variance, not truncation error.** A direction the solve drops
  stays dropped. The stopping-rule failure the acquisition study found on its
  held-out six-qubit bank comes from truncation, and this does not address
  it.
- **Classical cost.** Each look forms `B† K_k B` for every term and one joint
  covariance pass: about 20% more time per run on the six-qubit bank below,
  and about 7 s per look on H4 (1,689 settings), against about 1 s for the
  solve and 1 s for the allocation.

## Exploratory evidence

Systems, all with banks of the identity plus spin-adapted singles and doubles:

| Key | Orbitals / electrons | Qubits | Reference | Bank words / settings |
| --- | --- | ---: | --- | --- |
| `tri` | 3 / 2 (synthetic integrals) | 6 | correlated singlet-sector state | 997 / 169 |
| `h4` | 4 / 4: linear H4, STO-3G, 0.9 Å spacing | 8 | CISD state | 15,846 / 1,689 |
| `lih` | 4 / 4: LiH, STO-3G, 1.5949 Å, lowest four RHF orbitals | 8 | CISD state | 15,846 / 1,689 |

### Exact grouped variance (no shots)

Ratio of the Ritz functional's grouped Neyman shot variance before and after
the shift, with the functional's own words regrouped into qubit-wise-commuting
settings. Above one means fewer shots for the same error. Every shift leaves
the bank matrices unchanged to round-off (at most `1.4e-13`).

| System | Ritz words / settings | Unshifted Neyman | `N`, Ritz fit | `N` + one-body, Ritz fit | `N` + one-body + `S_z`, Ritz fit | `N` + one-body, state-free fit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `tri` | 525 / 85 | 15.2 | 2.74x | 6.95x | 7.01x | 2.22x |
| `h4` | 4,190 / 441 | 9.751 | 1.37x | 1.69x | 1.68x | 1.67x |
| `lih` | 1,454 / 357 | 0.4335 | 1.17x | 1.38x | 1.38x | 0.88x |

The state-free fit can lose (LiH), so the targeted fit is the one to use. The
`S_z` terms add little once the number terms are in.

### Correction of existing caches

`ritz_control_variate` on caches of fixed size, against the exact subspace
energy. `prefit` uses coefficients fitted beforehand from exact quantities
(`fit_sector_shift(objective="ritz")`). They target the functional's own
regrouped settings rather than the bank's, which is why the cache fit can beat
them.

| System | Shots per setting | Replicas | Arm | RMSE | Mean error | RMS std. error | 95% coverage [Wilson] | MSE ratio vs raw |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: |
| `tri` | 1,000 | 200 | raw | 0.01754 | -0.00625 | 0.01708 | 0.960 [0.923, 0.980] | 1 |
| `tri` | 1,000 | 200 | fitted | 0.00706 | +0.00162 | 0.00717 | 0.960 [0.923, 0.980] | 6.17 |
| `tri` | 1,000 | 200 | prefit | 0.00818 | +0.00252 | 0.00768 | 0.950 [0.910, 0.973] | 4.60 |
| `tri` | 10,000 | 200 | raw | 0.00558 | -0.00115 | 0.00513 | 0.935 [0.892, 0.962] | 1 |
| `tri` | 10,000 | 200 | fitted | 0.00213 | +0.00032 | 0.00201 | 0.925 [0.880, 0.954] | 6.86 |
| `tri` | 10,000 | 200 | prefit | 0.00225 | +0.00046 | 0.00212 | 0.935 [0.892, 0.962] | 6.13 |
| `h4` | 1,000 | 100 | raw | 0.01399 | -0.00913 | 0.01286 | 0.920 [0.850, 0.959] | 1 |
| `h4` | 1,000 | 100 | fitted | 0.01186 | -0.00655 | 0.01049 | 0.920 [0.850, 0.959] | 1.39 |
| `h4` | 1,000 | 100 | prefit | 0.01177 | -0.00544 | 0.01106 | 0.940 [0.875, 0.972] | 1.41 |
| `lih` | 1,000 | 100 | raw | 0.03092 | -0.02686 | 0.01998 | 0.920 [0.850, 0.959] | 1 |
| `lih` | 1,000 | 100 | fitted | 0.01172 | -0.00460 | 0.01435 | 1.000 [0.963, 1.000] | 6.96 |
| `lih` | 1,000 | 100 | prefit | 0.02715 | -0.02334 | 0.01894 | 0.970 [0.915, 0.990] | 1.30 |

On the six-qubit bank the cache fit cuts mean-squared error six to seven times
at both sizes and keeps its standard error calibrated. On H4 it gives 1.4x,
about what the prefit gives. The full bank reads every word of `S` and `H`
from its own 1,689 settings, so its reduction need not match the regrouped
exact figures above.

LiH shows a second effect. Its raw Ritz value at 1,000 shots per setting is
biased low by 0.027, about 1.3 standard errors: the noise-driven lowering of a
variational estimate whose coefficients are chosen on the same data. The cache
fit removes most of that bias (to -0.005), which is most of its 7x. The
sector terms have exactly zero mean only at fixed coefficients; at the solved
coefficients they pick up part of the noise the minimization exploited, and
subtracting them removes it. This is a second-order effect seen on three
banks (the raw bias shrinks on each), not a property the method guarantees.
Its standard error is conservative there (coverage 100/100), and the prefit,
which fixes the correction before seeing the data, gains only 1.3x.

### Inside acquisition

`acquire_ritz` over a fifteen-look schedule (cumulative shots per setting
100, 141, 200, ..., 12,800), uniform or Ritz-targeted, with and without
control variates. Each run spends the whole schedule; a stopping target is
applied to the ledger afterwards, which is exact because a stopping rule only
truncates the run it would otherwise have continued. Coverage is of the
schedule-valid interval `acquire_ritz` reports at the stop (`z = 2.935` for
fifteen looks), with Wilson 95% intervals.

**`tri`**, 200 replicas.

Without a stopping rule: MSE ratio against uniform without control variates (above one is better), and single-look 95% coverage.

| Cumulative shots per setting | uniform | uniform + CV | targeted | targeted + CV |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 1.00x (0.93) | 7.80x (0.96) | 1.00x (0.93) | 7.80x (0.96) |
| 800 | 1.00x (0.94) | 6.33x (0.94) | 2.33x (0.93) | 12.81x (0.92) |
| 3,200 | 1.00x (0.93) | 6.42x (0.95) | 3.13x (0.95) | 14.36x (0.92) |
| 12,800 | 1.00x (0.95) | 7.21x (0.99) | 2.46x (0.97) | 13.07x (0.94) |

With a stopping rule: mean shots to stop, the saving against uniform without control variates, and coverage of the schedule-valid interval at the stop.

| Target std. error | Arm | Met | Mean shots | Saving | RMSE at stop | Coverage [Wilson] |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0.01 | uniform | 200 | 540,664 | 1.00x | 0.00977 | 0.995 [0.972, 0.999] |
| 0.01 | uniform + CV | 200 | 109,815 | 4.92x | 0.01001 | 0.995 [0.972, 0.999] |
| 0.01 | targeted | 200 | 242,919 | 2.23x | 0.00962 | 0.990 [0.964, 0.997] |
| 0.01 | targeted + CV | 200 | 59,169 | 9.14x | 0.01079 | 0.955 [0.917, 0.976] |
| 0.005 | uniform | 200 | 2,163,200 | 1.00x | 0.00422 | 1.000 [0.981, 1.000] |
| 0.005 | uniform + CV | 200 | 381,090 | 5.68x | 0.00474 | 1.000 [0.981, 1.000] |
| 0.005 | targeted | 200 | 838,071 | 2.58x | 0.00438 | 0.995 [0.972, 0.999] |
| 0.005 | targeted + CV | 200 | 164,827 | 13.12x | 0.00582 | 0.975 [0.943, 0.989] |

Target 0.0025 (left out: uniform without control variates did not reach it within the schedule): uniform met 0, coverage 1.000; uniform + CV met 200, coverage 1.000; targeted met 0, coverage 0.995; targeted + CV met 200, coverage 0.985.

On the six-qubit bank the two gains multiply. Without a stopping rule,
control variates alone cut mean-squared error six to eight times at every
look, targeting alone two to three times once the pilot is spent, and both
together about thirteen to fourteen times. With a stopping rule, the corrected targeted runs reach a
standard error of 0.005 with 13 times fewer shots than uniform allocation, and
the schedule-valid interval still covers. The bank keeps its full rank (6) at
every look of every run, so these runs exercise no truncation.

**`h4`**, 40 replicas.

Without a stopping rule: MSE ratio against uniform without control variates (above one is better), and single-look 95% coverage.

| Cumulative shots per setting | uniform | uniform + CV | targeted | targeted + CV |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 1.00x (0.80) | 2.57x (0.93) | 1.00x (0.80) | 2.57x (0.93) |
| 800 | 1.00x (0.85) | 1.59x (0.82) | 16.18x (0.90) | 23.62x (0.95) |
| 3,200 | 1.00x (0.88) | 1.78x (0.90) | 17.62x (0.85) | 51.51x (0.97) |
| 12,800 | 1.00x (0.93) | 1.64x (0.90) | 25.76x (0.93) | 51.86x (0.97) |

With a stopping rule: mean shots to stop, the saving against uniform without control variates, and coverage of the schedule-valid interval at the stop.

| Target std. error | Arm | Met | Mean shots | Saving | RMSE at stop | Coverage [Wilson] |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0.01 | uniform | 40 | 2,842,376 | 1.00x | 0.01112 | 1.000 [0.912, 1.000] |
| 0.01 | uniform + CV | 40 | 2,128,098 | 1.34x | 0.01014 | 1.000 [0.912, 1.000] |
| 0.01 | targeted | 40 | 477,987 | 5.95x | 0.01079 | 1.000 [0.912, 1.000] |
| 0.01 | targeted + CV | 40 | 397,379 | 7.15x | 0.00942 | 0.975 [0.871, 0.996] |
| 0.005 | uniform | 40 | 10,809,600 | 1.00x | 0.00605 | 1.000 [0.912, 1.000] |
| 0.005 | uniform + CV | 40 | 7,642,725 | 1.41x | 0.00566 | 0.975 [0.871, 0.996] |
| 0.005 | targeted | 40 | 1,025,139 | 10.54x | 0.00490 | 1.000 [0.912, 1.000] |
| 0.005 | targeted + CV | 40 | 857,843 | 12.60x | 0.00513 | 0.975 [0.871, 0.996] |

On H4 targeting is the larger lever. The bank reads 1,689 settings and few of
them matter for the Ritz value, so targeting alone reaches the tighter target
with 10.5 times fewer shots, and control variates add a further 1.2 times. At
fixed shots the corrected values' actual mean-squared error falls more than
their standard error does (1.6 to 2.9 times under targeting, against about 1.4
in variance), because the correction also removes part of the raw estimate's
downward bias: at 3,200 shots per setting under targeting, the mean error goes
from -0.0011 to -0.0005. A stopping rule tests the standard error and does not
see that part. Every H4 run keeps full rank (15). Forty replicas cannot
resolve an under-coverage of a few points; the uncorrected uniform interval at
the 100-shot pilot covers only 80%, because the raw value is biased low by
more than its standard error there.

## Reproducing

```bash
python -m pytest tests/test_symmetry_shifts.py tests/test_acquisition.py -q
```

Every table above comes from one of these runs, made from a clean tree at
commit `0a3c8ef`, with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` set:

```bash
E=benchmarks/explore_control_variates.py; W=work/control-variates
python $E --system tri --mode exact --output $W/tri-exact.json
python $E --system h4 --mode exact --output $W/h4-exact.json
python $E --system lih --mode exact --output $W/lih-exact.json
python $E --system tri --mode caches --seed-root 424243 --replicas 200 --shots 1000 --workers 4 --output $W/tri-caches-1000.json
python $E --system tri --mode caches --seed-root 424244 --replicas 200 --shots 10000 --workers 4 --output $W/tri-caches-10000.json
python $E --system h4 --mode caches --seed-root 424245 --replicas 100 --shots 1000 --workers 4 --output $W/h4-caches-1000.json
python $E --system lih --mode caches --seed-root 424246 --replicas 100 --shots 1000 --workers 4 --output $W/lih-caches-1000.json
python $E --system tri --mode acquisition --seed-root 20261119 --replicas 200 --targets 0.01 0.005 0.0025 --workers 4 --output $W/tri-acquisition.json
python $E --system h4 --mode acquisition --seed-root 20261120 --replicas 40 --targets 0.01 0.005 --workers 4 --output $W/h4-acquisition.json
```

Each output records its named source digests (hashed before the package is
imported), the Git head and whether the tree was clean when the run started,
and refuses to overwrite. `work/` is gitignored. Replica `k` draws from
`SeedSequence([seed_root, k])`. An eight-qubit system takes about ten minutes
to compile its per-setting outcome distributions; that happens once per run,
and each replica only reseeds the sampler, which then draws the same stream as
a fresh backend with that seed. The six-qubit runs take minutes; the H4
acquisition run took about an hour on four cores.
