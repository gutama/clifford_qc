# Reproducing the benchmark artifacts

Every file under `benchmarks/reference_results/` regenerates from a clean
environment with the commands below. All experiments are seeded; JSONL
rows should match up to floating-point noise in wall-clock fields, and the
summary tables should match exactly.

## Access and review policy

This document is the executable reproducibility contract for a private
development repository. Private status changes distribution, not the evidence
standard: frozen inputs, raw records, generators, tests, environment
requirements, and semantic drift gates remain versioned together.

During framework development, editors and referees can receive an
access-controlled frozen review snapshot from the authors on request. The
snapshot must identify its exact source revision and include everything needed
to run the commands in this document; a moving development branch is not
itself treated as the review artifact. A tagged archival release with a
persistent identifier is intended after the operator, measurement, subspace,
and backend interfaces stabilize. Until that release exists, neither this
document nor the manuscript claims anonymous public download access.

**Records and code move together.** A record produced before a change to the
estimator, the selector, or the confidence construction is not comparable with
one produced after, and mixing the two is how stale numbers reach a manuscript.
When any module under `clifford_qc/measurement/` or `clifford_qc/algorithms/`
changes, regenerate every record that depends on it. The manuscript's figures
and tables are then emitted mechanically from those records:

Every runner stamps each JSON object with a
`clifford_qc.execution_provenance.v1` block containing the git SHA and dirty
state, package/Python/dependency versions, platform and BLAS/LAPACK details,
UTC time, and a digest of the complete installed distribution set. Numerical
record comparisons deliberately ignore only this metadata block; all physics,
selection, and resource fields remain value-gated. The scheduled record
regeneration workflow re-executes every committed runner weekly, compares those
scientific fields (excluding only provenance and keys ending in `_seconds`),
and uploads each fresh record whether the comparison passes or fails.

```bash
python paper/make_figures.py       # -> paper/paper_assets/*.pdf
python paper/make_tables.py        # -> paper/tables/*.tex  (\input by the .tex)
python paper/check_manuscript.py   # balance, refs, bib keys, column counts,
                                   # and figures older than their source record
python benchmarks/check_summaries.py  # *_summary.{csv,md} vs their JSONL
python benchmarks/check_docs.py       # this file vs the code it describes
```

That last one exists because this document drifted three times while the
numbers themselves stayed correct: the per-matrix `summarize.py` commands
regenerated only the CSV (which is *how* the Markdown summaries went stale),
the predeclared-parameter section quoted one global `delta` while five
certification experiments used their own, and the environment check
understated the test count. Artifact checkers cannot see prose, so
`check_docs.py` verifies that every documented command names a real script,
that every flag it passes is one the script accepts, that the delta/eps table
matches the constants in each script, and that no benchmark or committed
record is left undocumented.

Building the manuscript itself needs revtex4-2 and the packages the preamble
loads; on Debian/Ubuntu:

```bash
sudo apt-get install -y --no-install-recommends \
    texlive-latex-base texlive-publishers texlive-latex-recommended \
    texlive-fonts-recommended texlive-science

cd paper
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

The build must finish with **zero** overfull boxes and zero undefined
references or citations:

```bash
grep -cE 'Overfull \\hbox|LaTeX Warning: (Reference|Citation)' paper/manuscript.log   # -> 0
```

The `manuscript` CI job runs exactly this sequence plus the three checkers
above and fails on any drift, so a regenerated benchmark that leaves a stale
figure or table behind, or an edit that pushes text into the margin, is caught
before merge rather than at submission.

`check_summaries.py` exists because a regenerated JSONL leaves its
`summarize.py`-derived CSV and Markdown behind unless they are rebuilt too:

```bash
python benchmarks/summarize.py RECORD.jsonl \
    --csv RECORD_summary.csv > RECORD_summary.md
```

No figure or table value in the manuscript is transcribed by hand, and
`check_manuscript.py` exits nonzero if a figure predates the record it plots.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[test,research,chemistry]   # numpy + scipy + openfermion/pyscf
pytest                                      # 848 passed, 6 skipped
```

That install is the reference environment for the quoted pair, and it is
what CI's `manuscript` job builds. The count depends on it: a missing
optional module makes pytest drop the whole test file at collection, so
each absent extra moves one file from the passed count to the skipped
count. The six skips here are the bridge files — `stim` (three of them),
`pennylane`, `pytket`, and `pyzx`.

Adding the remaining extras therefore *changes both numbers*, which is
expected rather than a failure:

```bash
pip install -e .[stim]               # stabilizer backend / Phase 4
pip install -e .[bridges]            # stim + pytket + pennylane + pyzx + openfermion
```

The standalone DA-CASE paper's dyadic Clifford measurement hierarchy uses the
``stim`` extra to synthesize and verify exact logical diagonalizers for the
retained H4 and BeH2 banks:

DA-CASE is the paper-level name for the complete architecture: *Dyadic
Adaptive Clifford-Algebra Subspace Eigensolver*.  The adaptive subspace engine
is followed by a configurable dyadic Clifford measurement stage; its ``k=1``
endpoint is exactly QWC and remains the matched-ledger choice for comparability.
For provenance and backward compatibility, source identifiers such as
``acase_*`` and stored JSON arm labels containing ``A-CASE`` are not
renamed in-place.

```bash
python benchmarks/run_clifford_hierarchy.py --system h4
python benchmarks/run_clifford_hierarchy.py --system beh2
python benchmarks/check_clifford_hierarchy.py
```

This writes `reference_results/clifford_hierarchy_h4.json` and
`reference_results/clifford_hierarchy_beh2.json`. The H4 bank is reconstructed
from the retained labels in `matched_h4.json`, so the hierarchy and the matched
ledger share one bank by construction; the BeH2 bank is an independent DA-CASE
run on `data/beh2_sto3g_r1.3264.FCIDUMP`, which
`benchmarks/make_beh2_fcidump.py` regenerates from PySCF under the `chemistry`
extra. Both records report logical all-to-all CX counts and depth only;
topology routing, device noise, and mitigation are deliberately outside that
experiment. The tableau elimination is a constructive synthesis rather than a
CX-minimizing compiler, and the paper's two-currency break-even proxy omits
single-qubit Clifford costs (the JSON rows still retain their H/S counts).

`check_docs.py` verifies the documented pair by collection. It reports a
skip when the installed extras do not match the environment above; pass
`--require-test-count` to turn that mismatch into a failure, which is how
CI enforces it in the job that owns the contract.

The core package imports with numpy alone; without SciPy the optimizer
falls back to pure-Python Adam (numerically equivalent results at looser
tolerance — the committed artifacts were produced with SciPy's L-BFGS-B).

R8 keeps dense matrices as a small-system oracle rather than an execution
representation. The diagnostic below compares the packed Pauli matvec against
dense BLAS and dense `einsum`; both dense variants intentionally share the same
`O(4^n)` operator allocation so the timing cannot hide that memory cost:

```bash
python benchmarks/compare_pauli_action.py --n 6 8 --terms 64 --repeats 7
```

R7 performance work starts from a deterministic diagnostic rather than an
optimization guess. This profiles cold packed-word multiplication and element
construction, cold QWC grouping, repeated projected solves, sector and R8
full-space matrix-free matvecs, bootstrap re-solves, and the storage implied by
100-vector full-reorthogonalization Krylov bases:

```bash
python benchmarks/profile_hotpaths.py --repeats 5 --bootstrap-replicates 30 --sites 6
```

The script prints JSON to stdout and deliberately writes no reference artifact;
wall-clock profiles are machine-dependent diagnostics, not scientific records.

## Spin-model matrices (Phase 3)

```bash
# 30-seed development matrix (TFIM n=4 + random-field Ising n=4, 7 methods)
python benchmarks/run_benchmark.py --config benchmarks/configs/spin_small.json \
    --out benchmarks/reference_results/spin_small.jsonl
python benchmarks/summarize.py benchmarks/reference_results/spin_small.jsonl \
    --csv benchmarks/reference_results/spin_small_summary.csv \
    > benchmarks/reference_results/spin_small_summary.md

# 100-seed headline matrix (TFIM h-sweep, periodic TFIM, random Ising; n=4)
python benchmarks/run_benchmark.py --config benchmarks/configs/spin_headline_n4.json \
    --out benchmarks/reference_results/spin_headline_n4.jsonl
python benchmarks/summarize.py benchmarks/reference_results/spin_headline_n4.jsonl \
    --csv benchmarks/reference_results/spin_headline_n4_summary.csv \
    > benchmarks/reference_results/spin_headline_n4_summary.md

# 20-seed exploratory matrix at n=6 (TFIM, random Ising, XXZ)
python benchmarks/run_benchmark.py --config benchmarks/configs/spin_n6.json \
    --out benchmarks/reference_results/spin_n6.jsonl
python benchmarks/summarize.py benchmarks/reference_results/spin_n6.jsonl \
    --csv benchmarks/reference_results/spin_n6_summary.csv \
    > benchmarks/reference_results/spin_n6_summary.md
```

Long sweeps shard across k workers: run k processes with
`--shard 0/k … --shard k-1/k`, each with its own `--out`, and concatenate
the JSONL files (the committed 100-seed matrix used 4 shards; row order
differs between shardings but content does not). Approximate costs: one
noisy run is ~3.5 s at n=4 and ~5 min at n=6 on one laptop-class core;
n ≥ 8 sweeps need dedicated hardware.

## Stabilizer seeding go/no-go (Phase 4, `stim` extra)

```bash
python benchmarks/run_stabilizer_seeding.py \
    --out benchmarks/reference_results/stabilizer_seeding.jsonl
```

## Chemistry (Phase 5, `chemistry` extra)

```bash
python benchmarks/run_chemistry.py --seeds 3 \
    --out benchmarks/reference_results/chemistry.jsonl
```

PySCF computes SCF/FCI on the fly (no cached integrals), so the last
decimals of its energies differ across platforms. Those differences do not
move the reported energies, but they used to move the *selection*: the H4
pool is degenerate by symmetry, and a perturbation of ~1e-11 was enough to
hand an exactly tied step to a different operator. Two runs at four BLAS
threads on one machine exchanged nine of twelve labels that way.

The selector now resolves the argmax over a relative tolerance and takes the
lowest-indexed member of the tied class — the choice exact arithmetic would
have made — so tied selections no longer depend on reduction order, thread
count, or BLAS vendor. On well-separated candidates it is plain `max`.

The expensive exact-gradient H4 row can be reproduced independently without
rerunning the complete chemistry matrix:

```bash
python benchmarks/reproduce_exact_h4.py \
    --out reproductions/h4_exact.jsonl \
    --trajectory-out reproductions/h4_exact_trajectory.json \
    --with-reference
```

`--threads` pins the BLAS thread count before NumPy loads (default 1, which
is no slower here since the trajectory is dominated by the multivector
kernel rather than by BLAS).

`--with-reference` runs the trajectory a second time in the same process and
embeds the agreement between the two runs as `reference_comparison`, which is
what backs the reproducibility statement in the paper. It doubles the wall
time, and is how the committed
`benchmarks/reference_results/h4_exact_trajectory.json` was produced.

To run that row alongside the 200-seed certification calibration, use the
watcher. It keeps independent logs and outputs, records the environment in a
manifest, and atomically refreshes `status.json` until it writes a `DONE` or
`FAILED` marker:

```bash
python benchmarks/watch_reproduction.py \
    --run-dir reproductions/h4-certification
```

Write these long-running reproductions outside `benchmarks/reference_results/`
unless intentionally regenerating committed artifacts.

## Effective-Hamiltonian end-to-end showcase (numpy only)

The smallest complete materials-facing path uses the synthetic, canonical
two-site record in `examples/data/wannier_hubbard_dimer.json`:

```bash
python examples/acase_effective_model.py
```

This is an integration benchmark, not a DFT result. It validates the boundary
an upstream Wannier/embedding workflow would use:

```text
Hermitian one-body matrix + onsite U + explicit reference sector
    -> Jordan-Wigner Hamiltonian -> DA-CASE
    -> energy + state coefficients + correlations + Lehmann response
```

Expected invariants (minor last-digit formatting may vary):

- sector exact and adaptive DA-CASE energies both `-0.828427125 eV`;
- absolute energy mismatch below `1e-12 eV`;
- complete response basis `M=4`, rank `4`, `kappa(S)=1`;
- singlet diagnostic `<S^2>` below `1e-12`;
- staggered-spin line at `0.828427125 eV` with weight `0.853553391`.

The program accepts another record as its sole argument. The schema is
`clifford_qc.effective_hamiltonian.v1`; spin orbitals are interleaved
(`2*site=up`, `2*site+1=down`), and complex one-body entries are `[real, imag]`.

## Orbital-basis cost sweep (numpy only)

Every fermionic cost this repository quotes is a site-basis number, and the
site basis is a choice. The single-particle rotation `b_p = sum_i W_pi a_i`
with `W W^T = 1` leaves the spectrum exactly invariant and changes all three
reported currencies:

```bash
python benchmarks/run_orbital_basis.py \
    --out benchmarks/reference_results/orbital_basis.json
```

Roughly four minutes at the default 8-site ring, `U = 4t`, half filling,
periodic. Invariance is asserted rather than reported: every basis must
reproduce the site-basis ground energy to `1e-8` or the script raises and no
record is written, so a subtly wrong rotation cannot produce an attractive cost
table for the wrong Hamiltonian.

Expected invariants of the committed record:

- ground energy `-20.603526300` in all seven bases (site, momentum, db1–db4,
  non-interacting natural orbitals);
- Pauli-word count spanning `41` (site) to `3833` (db4) — a factor of 93 at
  identical physics;
- determinants for `1.6e-3` accuracy spanning `906` (momentum) to `4310` (db1)
  out of a `4900`-determinant sector;
- adjacent-mode Givens counts `0` (site) to `28`.

The disorder arm is the reason no basis can be standardised on. Averaged over
three seeds of uniform diagonal disorder, the most compact basis changes with
disorder strength: momentum at `W = 0` (906 determinants against the site
basis's 3658), the site basis by `W = 6`, and natural orbitals at `W = 12`
(956 against momentum's 3692). Wavelets (db2) are never competitive on either
axis, which is the negative result the sweep exists to record.

Parameters are predeclared at the top of the script (`CHEMICAL_ACCURACY`,
`INVARIANCE_TOL`, `DEFAULT_SITES`, `DEFAULT_U`, `DEFAULT_DISORDER`,
`DEFAULT_DISORDER_SEEDS`); `--sites`, `--interaction`, `--disorder`,
`--disorder-seeds`, and `--accuracy` override them for exploration, and a
`--sites 4` run finishes in a second.

## Configuration-space Haar packet tier (research benchmark)

The negative orbital-basis result above does not test a change of basis among
virtual DA-CASE configurations.  This separate benchmark orders every
non-reference determinant in the fixed `(N, S_z)` sector by excitation rank,
doublon count, charge pattern, and spin pattern; constructs an orthogonal
finite tree-Haar basis over that order; keeps only details with `S_A <= 16`;
and offers those details only during the first `M=20` directions before
handing the retained basis to the ordinary level-4 pool:

```bash
python benchmarks/run_configuration_packets.py \
    --out benchmarks/reference_results/configuration_packets.json
```

The declared selector cost exponent is `gamma=0.5`; both arms use it.  On the
open-boundary `2x2`, `U=4t`, half-filled Hubbard cluster, the committed record
has 35 non-reference determinant leaves and an actual packet-state overlap
residual `max|S-I| = 4.44e-16`.  The comparison is:

| Arm | `M` | energy error (Ha) | `W` | `S_A` | `S_H` |
| --- | ---: | ---: | ---: | ---: | ---: |
| level 4, coarse | 20 | 0.288203 | 9919 | 8 | 768 |
| level 4 + packets, coarse | 20 | 0.253170 | 6245 | 9 | 1224 |
| level 4, converged | 28 | 0 | 14762 | 8 | 1152 |
| packets then level 4, converged | 27 | 0 | 9869 | 9 | 1224 |

The staged arm reaches the same exact sector energy with 33.1% fewer projected
Pauli words and one fewer direction.  The tradeoff is explicit: its largest
projected element is wider and `kappa(S)=12.47` instead of 1.  This is one
exact-arithmetic finite-instance result, not a convergence theorem or a
default basis policy; the packet tier remains opt-in and small-sector only.

## FCIDUMP H4 CAS(4e,4o) benchmark (numpy only)

The larger active-space rung starts from the immutable interchange artifact
`benchmarks/data/h4_sto3g_r0.9.FCIDUMP`, not from an SCF calculation rerun
during the benchmark:

```bash
python benchmarks/run_fcidump_h4.py \
    --out reproductions/fcidump_h4.json
```

The source digest and generation metadata are in the adjacent provenance JSON.
The committed reference record is
`benchmarks/reference_results/fcidump_h4.json`. Expected results:

- 4 electrons in 4 spatial orbitals, 8 qubits, sector dimension 36;
- 185 mapped Pauli terms;
- sector exact `E0 = -2.180316614324 Ha`;
- difference from the external PySCF determinant-space FCI result below
  `1e-10 Ha`;
- adaptive DA-CASE at 8 additions: `M=9`, `W=7371`,
  error `3.018781 mHa`;
- complete singles/doubles coordinate space: `M=27`,
  error `0.765862 mHa` (chemical accuracy).

The adapter itself needs no chemistry extra. Full CI additionally compares its
185 Pauli coefficients against the independent OpenFermion/PySCF construction.
FCIDUMP orbital signs are a gauge; the committed digest fixes one gauge rather
than weakening coefficient tolerances.

### Varying the reference state

The adaptive arm above misses chemical accuracy at its declared DA-CASE budget.
This experiment asks whether changing the reference is sufficient to cross the
accuracy threshold without increasing that nine-vector subspace:

```bash
python benchmarks/run_warm_start.py \
    --out reproductions/warm_start_h4.json
python benchmarks/check_warm_start.py
```

Same FCIDUMP, same DA-CASE candidate pool, and same eight additions; `rho`
changes from the Hartree-Fock determinant to an ADAPT-VQE state. The ADAPT
stage is additional work, and the record includes its pool-gradient
evaluations, optimizer evaluations, and state-preparation rotor count.
Expected results
(`benchmarks/reference_results/warm_start_h4.json`):

- Hartree-Fock reference: `M=9`, error `3.019 mHa`, `kappa(S)=1`, `W=7371`;
- 2-operator ADAPT reference: `M=9`, error `0.342 mHa` (chemical accuracy),
  `kappa(S)=1.02`, `W=7510`, 319 active-pool gradient evaluations,
  15 optimizer evaluations, and 2 state-preparation rotors;
- the ADAPT state alone is `27.091 mHa`, an order of magnitude worse than the
  cold DA-CASE result it improves;
- deeper warm starts are better states and give worse subspaces: `0.612 mHa`
  at `k=4` and `0.768 mHa` at `k=6`.

This benchmark stays NumPy-only: the ADAPT pool is built from the odd-Y words
of the determinant excitations rather than through the OpenFermion-backed
`models.chemistry.excitation_pool`.

Selection and optimization are exact in this record. Therefore its zero
selection-shot count is an exact-simulation label, not an end-to-end hardware
resource estimate. The hybrid result holds the DA-CASE budget fixed; it does
not claim that the total ADAPT+DA-CASE cost equals the cold DA-CASE cost.

### Matched-contract cost comparison, including exact ADAPT-GCIM

Every arm on one H4 contract, with the sector, Hartree-Fock reference, operator
pool, eight-addition budget, and stopping rule held fixed:

```bash
python benchmarks/run_matched_h4.py \
    --out reproductions/matched_h4.json
python benchmarks/check_matched_h4.py
```

The checker uses the same exact-discrete/tolerant-float comparison as
`check_warm_start.py`, and for the same reason: every count reproduces exactly,
while the ADAPT and warm-started arms carry a parameter optimization whose last
bits depend on the runner's BLAS. A byte comparison of this record fails CI on
a `sector_weight` of `1.0` serialized as `0.9999999999999999`.

The record does not assign an exchange rate between a state preparation and a
Pauli word. A physical measurement shot prepares one state in one measurement
setting, while a QWC setting can return several compatible words. Expected
results
(`benchmarks/reference_results/matched_h4.json`):

- the exact workflow has 1 state-evaluation context for fixed-reference arms,
  8 or 16 distinct generating-function contexts for ADAPT-GCIM, and 90
  ADAPT-VQE selection/optimizer contexts. These are **not** physical
  preparation executions; physical preparations occur once per setting-shot;
- ADAPT-VQE's 2424-word selection union has 452 QWC settings and is paid on
  eight changing selection states, giving 3616 selection setting-evaluations.
  DA-CASE's fixed-reference selection banks contain 15783/14401 words but only
  1681/1672 QWC settings at determinant/word resolution, paid once. Under
  uniform `R` shots per setting the selection circuit/preparation counts are
  therefore `3616*R` versus `1681*R`/`1672*R`;
- weighting each ADAPT-VQE selection setting by adaptive-rotor depth gives
  12656 rotor-setting evaluations. Its 82 optimizer evaluations add a lower
  bound of `82*68 = 5576` Hamiltonian setting-evaluations for energy alone;
  physical gradient cost is deliberately unpriced because this exact
  benchmark specifies no hardware gradient protocol;
- ADAPT-GCIM uses the published cumulative-surrogate gradient, fixed
  `theta=pi/4`, no parameter optimization, and `M=2k`. The near-size arm
  (`k=4`, `M=8`) has `13.364 mHa` error, effective rank 6, and
  `kappa(S)=32.9`; the iteration-matched arm (`k=8`, `M=16`) has
  `10.674 mHa` error, rank 12, and `kappa(S)=1.09e3`. Both use the published
  exact overlap cutoff `1e-13` without DA-CASE's additional condition cap;
- those ADAPT-GCIM pencils require 36 Hamiltonian/28 off-diagonal overlap
  pairs and 136/120 pairs, respectively. They are transition measurements
  between separately prepared states, so the record deliberately does not
  mislabel them as DA-CASE's single-reference final `W`;
- ADAPT's final Hamiltonian has 185 words/68 QWC settings; DA-CASE's retained
  word-resolution bank has 2240 words/465 settings and its determinant bank
  has 7371 words/913 settings;
- DA-CASE at determinant resolution and at word resolution return the same
  energy to `4e-16 Ha`, the same `M=9` and `kappa(S)=1`, and the same subspace
  (all nine principal angles zero) at `W=7371` and `W=2240` respectively;
- the word pool is rejected outright under the declared leakage tolerance
  (every odd-Y word has operator leakage `sqrt(2)`), yet with the rule disabled
  the Ritz vector has sector weight 1 to machine precision.

The selection width is the whole element cache, including rows for candidates
that were then rejected; it is strictly larger than the retained `W` the
manuscript's ledger reports.

The warm-started arm appears twice. The first keeps the sector-mixed ADAPT
reference and the operator-global generator rule, and carries no
reference-conditioned certificate: with sector weight `0.999882` there is no
sharp `(N, S_z)` to condition on, and inference refuses rather than rounding an
average occupation into an undeclared convention. The second post-selects that
reference onto the declared sector, `rho -> P rho P / tr(P rho)`. That makes
the sector weight one by construction, so the cascade and the whole-span
certificate apply; it improves the arm from `0.342` to `0.199 mHa`, and the
`1/weight = 1.00012` retry factor is reported in the row's
`sector_post_selection` block rather than folded into any count. It applies to
the A-CASE stage's preparation executions only — every accepted shot there
needs `1/weight` attempts on average — and to no setting count, since
`selection_qwc_group_evaluations` and its rotor-weighted companion are
structural counts of distinct measurement settings rather than shots. The
prelude's ledger is unchanged, because its measurements preceded the
projection. The non-demolition `(N, S_z)` measurement circuit that would
realize `P rho P` without resolving the determinant is explicitly unpriced.

### Finite-shot allocation and overlap regularization

One fixed four-qubit TFIM projected bank, one physical shot budget, and a
`2 x 3` cross of acquisition policy against overlap-truncation rule:

```bash
python benchmarks/run_finite_shot_optimization.py \
    --out reproductions/finite_shot_optimization.json
python benchmarks/check_finite_shot_optimization.py
```

Expected results
(`benchmarks/reference_results/finite_shot_optimization.json`), at 200 replicas
of 104,000 physical setting-shots:

- covariance-aware allocation reduces the median summed projected-matrix
  variance from `272.46` to `84.75` (68.9%) and the exact-Ritz first-order
  variance by 20.0%. Against the fixed overlap cutoff this moves the median
  absolute error only from `4.48` to `4.44 mHa`, but cuts `>0.1 Ha` failures
  from `4/200` to `1/200`;
- the calibrated cutoff must be read **per mode**. Thresholding every mode at
  the worst mode's noise radius retains six or seven modes and never the exact
  rank eight, leaving median errors of `21.4`/`81.4 mHa`. Comparing each mode
  against its own radius recovers rank eight in `71/200` and `67/200` replicas
  and gives median errors of `10.6`/`8.7 mHa`;
- against the fixed cutoff at matched allocation, per-mode calibration trades a
  factor of two in median error for a `22x` RMSE reduction (`12.85` versus
  `285.41 mHa`), a maximum error of `39.4 mHa` rather than `4.03 Ha`, and no
  catastrophic replica. The calibrated arms carry an `8-10 mHa` upward bias
  that a truncated rank cannot avoid.

This is a fixed-bank Monte Carlo diagnostic with a data-derived rank rule. It
is not a coverage certificate, a hardware result, or an end-to-end advantage
claim.

This is an exact implementation of the published ADAPT-GCIM algorithm on the
matched local 26-excitation pool. It is not a reproduction of the original
paper's molecular curves, generalized pool, transpilation, or hardware shot
model.

### Krylov measurement width

The ladder leaves the Krylov arm's `W` blank because the tracked element route
is quadratic in the basis and the powers are dense. For `A_k = H^k` with
Hermitian `H` the union collapses to `H^0 ... H^(2m+1)`, which is linear:

```bash
python benchmarks/run_krylov_width.py \
    --out reproductions/krylov_width.json
```

Expected results (`benchmarks/reference_results/krylov_width.json`), at the
`1e-8` coefficient threshold the record reports:

- `h2` `W=24` and `lih` `W=64`, reproducing the ladder's own tracked counts;
- `h4_chain(r=0.9)` `W=4224` against DA-CASE's `7371`;
- `h2o_4e4o(scale=2.0)` `W=8192` against DA-CASE's `7783`, the one rung where
  the operator-generated basis is the narrower of the two;
- `kitaev` `W=140`, equal to DA-CASE, which there is a pruned Krylov basis.

The threshold matters: repeated multiplication accumulates round-off, so the
raw Krylov count on `h4_chain(r=0.9)` fluctuates near `8184` rather than
`4224`. Every threshold is applied independently to the same unpruned powers;
thresholded powers are never multiplied recursively, so the support sweep is
nested. The script then rebuilds the full overlap and Hamiltonian pencils at
each cutoff. A count is reportable only when effective rank matches and the
normalized pencil entries, Ritz energy, and condition number reproduce the
unpruned construction within the tolerances stored in the v2 record. At
`1e-8` all quoted rungs pass and the stored numerical differences are zero.
`tests/test_krylov_width.py` checks the collapse identity, nested sweep, and
certificate gate.

## Finite-shot nonlinear response uncertainty

```bash
python examples/acase_finite_shot_response.py
python paper_acase/check_response_records.py
```

The example uses 25 shared QWC groups and 8,000 shots per group. A grouped
bootstrap resamples their joint histograms and reruns the entire response
pipeline. With the committed seeds it reports one staggered-spin line near
`0.829 eV`, weight near `0.854`, and `chi(0)` near `2.060 1/eV`.

The intervals are percentile diagnostics labelled `heuristic`:

- they are not finite-sample certificates;
- root-resolved output requires isolated ordered Ritz roots;
- thresholded-rank changes and root collisions are counted as failed replicas;
- at least 80% of replicas must remain usable by default;
- broadened-response bands are pointwise, not simultaneous.

`check_response_records.py` compares a fresh run against the committed records
in four tiers -- `contract` (seeds, budget, basis), `point` (exact spectrum,
conditioning, every point estimate), `census` (how many replicas survived), and
`intervals` (the percentile bands) -- and names the earliest failing tier. The
tiers exist because the bands are all conditional on the census: two fewer
surviving replicas move eight hundred band values, and a flat diff cannot
distinguish that from a changed pipeline.

**Known reproduction gap.** On the committed records the `contract` and `point`
tiers reproduce exactly, including the overlap condition number to the last
stored digit. The ill-conditioned record's `census` does not: it was committed
at `139/200` accepted replicas and reproduces here at `137/200`, which shifts
every percentile band by up to `5.2e-4` relative. The well-conditioned record
keeps its `200/200` census and moves two band values by `2.9e-5`. This is not
last-bit float noise in the acceptance test: acceptance turns on the sign of
the resampled overlap's smallest eigenvalue, that eigenvalue agrees between
`float64` and `float128` to `1.9e-13` across all 200 replicas, and the replica
nearest the sign boundary sits `4.5e-4` away from it -- nine orders of
magnitude clear. The gap predates the tiering and reproduces at the records'
own authoring commit, so it is environmental rather than code drift. The
records store no per-replica fingerprint, so which two replicas moved cannot
currently be recovered from the record alone; adding one would make the next
such disagreement diagnosable in a single step.

## DA-CASE validation ladder (Phase 7, `chemistry` extra)

```bash
python benchmarks/run_acase_ladder.py \
    --config benchmarks/configs/acase_ladder.json \
    --out benchmarks/reference_results/acase_ladder.jsonl
python benchmarks/summarize_ladder.py \
    benchmarks/reference_results/acase_ladder.jsonl \
    --csv benchmarks/reference_results/acase_ladder_summary.csv \
    > benchmarks/reference_results/acase_ladder_summary.md
```

One row per (rung, method) — not per seed — so this record has its own
summarizer. `summarize.py` understands the per-seed ADAPT schema and would
reject the ladder's rows; `check_summaries.py` maps the `acase_ladder` stem
to `summarize_ladder.py` explicitly rather than sniffing the schema.

`--rungs h2,lih_2e2o` restricts the run to named rungs, which is how to
regenerate one row of the record without rebuilding all of it. Rerunning a
subset writes a *partial* record, so the committed file must come from a full
run.

Four conventions in the record are choices, not defaults, and each is
recorded in the row that made it:

- **The error column is measured against the reference's own symmetry
  sector**, not against the global ground state. DA-CASE never leaves the
  sector its reference lives in, and a grand-canonical Hubbard cluster's
  global minimum sits at a different filling. Each row carries the `sector`
  it was scored in.
- **Both fixed arms take an even stride through their family**
  (`selection: stride`, shared by `qse` and `generator_coordinate` through
  `fixed_slice`), because the natural prefix is all singles and every single is
  Brillouin-dead on a Hartree–Fock reference: a prefix of eight reproduces the
  reference energy to machine precision and would report a non-adaptive
  subspace as worthless for a reason that is about the ordering.
  `selection: prefix` reproduces that degenerate arm. `krylov` records
  `selection: null` and keeps its consecutive powers — striding `H^k` builds a
  different, worse-conditioned space rather than a fairer sample of the same
  one.
- **The fermionic sector filter is dropped on models with no fermionic
  sector.** `sector_leakage` measures `||[A,N]||/||A||` and the same for
  `S_z`, and it sees a generator alone, so on the Kitaev cluster — one qubit
  per site, no Jordan–Wigner transformation, no particle number — it rejects
  every candidate and DA-CASE reports a basis of size one at the reference
  energy. That reads as a method failure and is a configuration error. Rows
  carry `leakage_filter` saying whether the tolerance was applied or dropped.
  On the fermionic rungs the excitation candidates conserve both symmetries
  exactly (leakage `0.0`), so the filter is a no-op there.
- **Wide generators fall back to the cyclic contraction.** The
  element-operator route is what makes `W` and `S_H` observable and it costs
  `O(|A_i| |H| |A_j|)` per pair; deep Krylov generators on eight qubits carry
  thousands of words each. Above `max_tracked_support` (512) the row is solved
  through the cheap route and sets `support_tracked: false` rather than
  reporting guessed support columns.

Certified DA-CASE runs on the eight-qubit rungs as `acase_certified_n8`, at
32 000 shots per group against the four-qubit rungs' 4 000. It needs the larger
budget: at 4 000 it abstains immediately on H₄, which is a correct certified
outcome rather than a failure, and the extra shots are nearly free because the
per-group sampling plan is built once and reused.

Getting there needed a fix in the measurement layer, and the first diagnosis of
it was wrong. The QWC partition *is* greedy and quadratic, but it is memoized on
the word set, and the certified loop's universe is the same index set at every
growth step — so it was computed once, not per step, and it was never the
binding cost. The binding cost was `computational_probabilities` at 11.4 s per
eight-qubit readout, called once per group per batch, which made a single batch
over H₄'s 1 689 groups take hours. A Z-basis readout only sees the diagonal
Pauli content of the state, and the outcome vector is the Walsh–Hadamard
transform of those coefficients, so it is `O(2^n n)` rather than `O(4^n)`. With
that plus a cached per-group sampling plan, a warm batch is 0.19 s.

`adapt_shot` still appears only on the four-qubit rungs; its cost is the ADAPT
pool sweep, not the sampler.

The Hubbard rungs carry a §4.2 **level-4** arm (`acase_level4`) beside a
matched levels-0-3 arm at the same budget (`acase_exact_m25`), so the
comparison is like-for-like. Level 4 adds the cluster's competing-order
configurations and their products with the excitation family; it is opt-in per
method (`"level4": true`) because it is the only quadratic family in the
hierarchy. Both arms set `leakage_tol` to null: the operator-level sector
filter rejects every configuration generator by construction (an `X`-string
does not commute with `N`), and the right test for a configuration is the
sector of the state it names, which `subspace.state_sector` reports.

Costs on one laptop-class core: the four-qubit rungs are seconds each, the
`h4_*` rungs a few minutes apiece, and `h2o_cas8e6o` (12 qubits) dominates the
total. The certified eight-qubit rows add roughly three minutes each.

## Phase 10--12 paper-level drivers

Four orchestration/replication runners sit above the primitive benchmarks
documented in the sections above:

- `benchmarks/run_phase10_hybrid.py` runs the Phase 10 QSCI x DA-CASE
  primary-system capability comparison.
- `benchmarks/run_packet_seed_ensemble.py` runs the Phase 11 coarse-to-fine
  packet seed ensemble; inference resamples whole seeds.
- `benchmarks/run_m7_seed_replication.py` replicates the sampling-dependent
  Phase 12 matched-budget `M=7` arms across seeds and packet orderings.
- `benchmarks/run_phase12_paper_b.py` assembles the integrated Paper B
  resource ledger and computes Pareto frontiers only within one evidence
  category.

Their QSCI defaults use the exact sector ground state as a validation oracle,
not an implementable state-preparation claim.  The Phase 10 and 12 H$_4$
systems built through PySCF require the `chemistry` extra; dependency-light
FCIDUMP rungs remain available for the matching checks.

## Demos (not committed as artifacts)

```bash
python benchmarks/demo_shared_word_selection.py --seeds 5
```

## Predeclared experiment parameters

Seeds are `range(n_seeds)` per (model, method); model disorder seeds equal
the run seed. Strict certification calibration uses a fixed candidate family,
Bonferroni allocation across its predeclared decision schedule, and an
additional group-wise split for empirical-Bernstein bounds. Published
normal-bound trajectory sweeps retain explicit Sidak intervals as an
asymptotic heuristic and are never labelled finite-sample certified.
The **trajectory sweeps** (`run_benchmark.py`, `run_baselines.py`) use
delta = 0.05, near-optimality tolerance 0.05, ADAPT gradient threshold 1e-6
(1e-5/1e-6 chemistry), shot escalation base 256 doubling to 64x,
variance-proportional round budget 4096 (growth 2, 7 rounds), operator budgets
8 (n=4/6 spin) and 12 (chemistry and seeding), optimizer L-BFGS-B (gtol 1e-8;
maxiter 150 for the n=6 matrix and 200 for chemistry, unlimited-default
elsewhere). Chemical accuracy is 1.6e-3 Ha against the active-space FCI energy.
Resource metrics follow RESEARCH_PLAN.md section 7 (shots, circuits, unique
words, operators, optimizer evaluations, peak Pauli support).

The **certification experiments do not share those values** — each predeclares
its own, and the constants live at the top of its script:

| experiment | delta | eps | base x ceiling |
|---|---|---|---|
| `run_calibration.py` | grid 0.01 / 0.05 / 0.10 / 0.20 | — (exact-best) | 256 x 64 |
| `run_calibration_eps_best.py` | grid 0.05 / 0.10 / 0.20 | grid 0.15 / 0.30 | 512 x 64 |
| `run_certified_trajectories.py` | 0.10 trajectory-wide, split delta/K | 0.30 | per system |
| `run_ceiling_sweep.py` | 0.10 | 0.30 | 256 x {64 … 16384} |
| `run_hard_instances.py` | 0.10 | 0.30 | 256 x 64, and 4096/word fixed |

Quoting the trajectory-sweep delta for a certification result, or vice versa,
is a reporting error: the manuscript states the applicable value with each.

Empirical-Bernstein certification also requires fixed cumulative sample
endpoints: use `UniformFixed` or `UniformDoubling`. Variance-proportional
allocation chooses future endpoints from observed variances and is therefore
paired only with the asymptotic normal bound unless a confidence sequence is
added.

The configured `delta` is a per-selection-call budget. For a strict trajectory
with at most `K` selections and desired trajectory-wide budget `delta_total`,
use `delta=delta_total/K` (or a sharper predeclared stepwise allocation).

## Certification revision experiments (PRA revision)

```bash
# Calibration of the certification guarantee (headline): wrong-selection vs
# delta, coverage, regret, abstention, cost.
python benchmarks/run_calibration.py --seeds 200 \
    --out benchmarks/reference_results/calibration.jsonl

# Strengthened baseline ladder (exact / random / fixed-shot / shared-only /
# shared-grouped / variance-reuse / strict / fallback) + reuse accounting.
python benchmarks/run_baselines.py --seeds 15 \
    --out benchmarks/reference_results/baselines.jsonl

# Repaired chemistry: infinite-shot (N->inf) proxy ranking and H4 geometry
# sweep. Requires the chemistry extra.
python benchmarks/run_chemistry_repair.py \
    --out benchmarks/reference_results/chemistry_repair.jsonl

# Complete certified eps-best trajectories (spin + molecules): every appended
# operator is finite-sample certified (empirical-Bernstein), with per-step
# status, resolution kind, gradient, confidence radius, shots, and energy.
# The eps-best rule resolves the exact symmetry ties that stall a strict
# exact-best trajectory. Requires the chemistry extra for H2/LiH.
python benchmarks/run_certified_trajectories.py \
    --out benchmarks/reference_results/certified_trajectories.jsonl

# Recalibration of wrong-selection under the eps-best definition
# (|g_selected| >= gmax - eps): broad instance set (symmetric/tie references,
# displaced generic, and exact-ADAPT trajectory states), pooled and stratified
# by the eps-boundary gap, with a one-sided 95% Clopper-Pearson upper bound on
# the conditional error. Also reports the exact-argmax wrong rate for contrast.
python benchmarks/run_calibration_eps_best.py --seeds 60 \
    --out benchmarks/reference_results/calibration_eps_best.jsonl

# Shot-ceiling sweep: separates structural abstention (exact ties, which the
# exact-best rule can never resolve) from budget-limited abstention (small but
# nonzero gaps) and from bound conservatism. Sweeps the cumulative ceiling over
# 64x..16384x on three explicit strata under both resolution rules, and reports
# resolution rate, wrong rate, median shots at resolution, terminal radius, and
# the multiplicative tolerance eta_required.
python benchmarks/run_ceiling_sweep.py --seeds 60 \
    --out benchmarks/reference_results/ceiling_sweep.jsonl

# Gap-stratified hard instances: where an uncertified fixed-shot selector
# silently fails. Bins instances by normalized top-two gap and compares
# fixed-shot / fallback / strict exact-best / strict eps-best on commit rate,
# wrong rate among committed decisions, eps-violation rate, relative regret,
# and cost; plus shot-matched ADAPT trajectories.
python benchmarks/run_hard_instances.py --seeds 60 --traj-seeds 15 \
    --out benchmarks/reference_results/hard_instances.jsonl

# Optional strict finite-sample H4: empirical-Bernstein, fixed endpoints,
# trajectory-wide delta split, and abstention on every unresolved outcome.
python benchmarks/run_chemistry_repair.py --strict-h4 \
    --trajectory-delta 0.05 --h4-operators 10 \
    --out reproductions/chemistry_repair_with_strict_h4.jsonl
```

The strict finite-shot H4 arm (n=8, 160 candidates) is opt-in because it is
beyond the practical single-core simulation budget of this reference
implementation. It may legitimately stop at the first ambiguous selection;
that abstention is the certified result. The default command emits only the
infinite-shot ranking and geometry sweep and never labels fallback output as
certified.
