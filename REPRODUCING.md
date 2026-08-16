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
selection, and resource fields remain value-gated. The versions in it are not
inert, though: `benchmarks/check_record_environment.py` reads them back out,
requires the committed records to agree on one environment, and emits the
interpreter and the pip constraints that CI installs against, so every gate
runs under the versions its own record was produced under.

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

This sequence is a local step, not a CI job: the manuscript build needs a full
REVTeX installation, which is far more expensive to provision than the checks
that gate the code. Run it before a submission, so a regenerated benchmark that
leaves a stale figure or table behind, or an edit that pushes text into the
margin, is caught then rather than at submission.

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
pytest                                      # 1265 passed, 27 skipped
```

That install is the reference environment for the quoted pair. The count
depends on it: a missing
optional module makes pytest drop the whole test file at collection, so
each absent extra moves one file from the passed count to the skipped
count. Thirteen files are dropped that way here — `stim` (ten of them:
`test_block_synthesis`, `test_protocol_axis`, `test_protocol_cost`, `test_bridge_stim`,
`test_clifford_hierarchy_cost`, `test_compiled_measurement`, `test_exact_shot_search`,
`test_phase4`, `test_restriction`, `test_stim_clifford_rotors`), plus `pennylane`,
`pytket`, and `pyzx`. The
remaining fourteen skips are per-test rather than per-file: `test_fermion_mapping`
and `test_mapping_axis` guard only the individual tests that reach the stim
bridge, so those files still run.

`check_docs.py` enforces the pair through the identity relating them. Each of
the thirteen dropped files contributes exactly one skip and no collected tests, so
the remaining `27 - 13 = 14` skips are per-test and *are* collected:

```text
collected == passed + (skipped - files dropped at collection)
1279      == 1265   + (27      -  13)
```

Both sides are computed from the tree, so a drift in either quoted number
breaks the identity. The check runs collection only and never executes a test.
Because the identity is specific to one environment, it is enforced only when
the installed extras are the ones the `pip install` line above names — decided
by importability of the guarded modules, not by counting files. Under any other
extras it reports "unverified" and says which modules differ.

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

This writes schema-v3 `reference_results/clifford_hierarchy_h4.json` and
`reference_results/clifford_hierarchy_beh2.json`; the frozen schema-v2 controls
`clifford_hierarchy_h4_v2.json` and `clifford_hierarchy_beh2_v2.json` remain
beside them, and are not regenerated. The checker projects every v3
record back onto the v2 contract and requires every legacy field and JSON type to
match before checking the new columns. The H4 bank is reconstructed
from the retained labels in `matched_h4.json`, so the hierarchy and the matched
ledger share one bank by construction; the BeH2 bank is an independent DA-CASE
run on `data/beh2_sto3g_r1.3264.FCIDUMP`, which
`benchmarks/make_beh2_fcidump.py` regenerates from PySCF under the `chemistry`
extra.

R1 adds three versioned device cards from `configs/device_cards/`: the zero-error
`logical-alltoall` regression card and explicitly illustrative
`superconducting-like` and `ion-like` sensitivity scenarios. The latter two are
project-defined parameter sets, not vendor calibration or current-hardware
claims. Sparse-connectivity routing is a declared count/depth multiplier rather
than a routing compiler. Every cost row carries the full card and its canonical
SHA-256; scalar times without a named card are rejected by the schema.

Each protocol rung reports per-setting `N_1q`, `N_2q`, `D_1q`, and `D_2q`, the
independent-error fidelity surrogate and admissibility, fixed-shot and
equal-effective-shot times, and both estimators on exactly the same synthesized
settings. Because the time model charges `t_1q·D_1q + t_2q·D_2q` serially, that
pair comes from one as-soon-as-possible schedule over type-homogeneous layers,
so `D_1q + D_2q` is a genuine critical path; the separate `logical_cx_depth`
column stays the two-qubit critical path on its own, which is what the frozen
v2 ledger reports. `single_assignment` reads each word only from its partition group;
`pooled` reads it from every compatible setting, with weights that sum to one.
The record retains coverage and raw/effective word observations separately, so
free post-processing reads are not confused with physical state preparations.

The schema-v3 hierarchy's 1.6 mHa `C(epsilon)` column is evidence-tiered and remains
`asymptotic`: exact frozen-subspace bias plus first-order covariance-aware propagation
of the Ritz functional. It is not silently relabelled after the nonlinear benchmark
below lands. H4's 3.019 mHa subspace bias exceeds the target, so every arm correctly
reports the target as unattainable rather than printing a runtime. BeH2 is priced, and
every scalar names the card that attains it: a rung some declared card cannot run at
the fidelity floor is `partially_priced`, with the inadmissible cards listed rather
than dropped from a minimum. The committed ordering records whether pooling changes
the ranking of `k` rungs under each
card, comparing only the rungs both estimators admit and abstaining with `null`
where neither admits any. The break-even surface likewise names a winner only
where the two schedules are already accuracy-matched; otherwise it reports
`not_accuracy_matched` rather than ranking protocols by a fixed-shot time that
their differing setting counts dominate. These values are not a nonlinear noisy-GEP
study, a hardware prediction, or a replacement for the negative PRD finite-shot
result.

The tableau elimination remains a constructive synthesis rather than a
CX-minimizing compiler. Mitigation and calibrated topology routing remain outside
this experiment.

### R1 exact-oracle nonlinear shot search

The separate exact-tier search reuses the frozen H4 and BeH2 banks and the same
dyadic settings, but replaces the hierarchy record's first-order variance propagation
with actual finite-shot joint sampling and the complete nonlinear selected-rank solve:

```bash
python benchmarks/run_exact_shot_search.py --workers 4
python benchmarks/check_exact_shot_search.py
```

The producer writes schema `clifford_qc.exact_shot_search.v1` to
`benchmarks/reference_results/exact_shot_search.json`. It uses 30 paired exploratory
replicas on the fixed `64…65536` geometric endpoint grid, then an independent block of
100 paired replicas at every grid point through the pilot crossing. This downward
extension makes the priced endpoint the smallest confirmed pass and requires an
actual confirmed failure below it; a nonmonotone confirmation is not priced. Every
phase, rung, replica, estimator, and bootstrap stream has a disjoint NumPy
`SeedSequence` namespace. An endpoint passes only with no solver failure and a
one-sided 95% nonparametric-bootstrap upper bound on replica RMSE at or below 1.6 mHa.
Endpoints are nested; assigned and pooled estimators consume the same joint
histograms. The exact tier names the unavailable-on-hardware oracle comparator; the
bootstrap uncertainty is explicitly heuristic, not a finite-sample energy certificate
or deployable stopping rule.

H4 exits without sampling because its 3.019 mHa exact bank bias already exceeds the
target. On BeH2 the confirmed assigned/pooled passing endpoints are `4096/1024` at
`k=1`, `16384/4096` at `k=2`, `4096/16384` at `k=4`, and `16384/4096` at `k=8`.
All 3,500 confirmatory solves succeed. Device-card costs are computed only from the
smallest confirmed passing endpoints. The fidelity layer remains the same
illustrative `F^-2` surrogate, not a device-noise simulation.

**Three of the eight crossings are not resolved by this experiment**, and the record
says so rather than reporting them as settled counts. Each arm stores the distance
from the target for both deciding endpoints — the smallest confirmed pass, which sets
the reported count, and the largest confirmed failure, which sets that the count is
not smaller — and flags the crossing when either sits within ±10% of 1.6 mHa:

| arm | reported | passing margin | failing margin | marginal side |
|---|---:|---:|---:|---|
| `k=1` assigned | 4096 | −4.6% | +124.8% | passing |
| `k=4` assigned | 4096 | −1.8% | +280.0% | passing |
| `k=4` pooled | 16384 | −72.3% | +6.3% | failing |

The band is measured, not chosen: rebuilding this record under numpy 2.5.2 rather
than the 2.4.6 above moved the `k=4` assigned upper bound at 4096 shots from 1.5705
to 1.6054 mHa — 2.2% of the target, and across it, changing that arm's reported
count from 4096 to 16384. Nothing was wrong with either run. The endpoint deciding
that arm sits on the target, so which side it lands on is a property of the build
environment rather than of the protocol, and `±10%` is roughly four times the
observed sensitivity. `k=4` pooled shows the failing side matters equally: its
passing side is a comfortable −72.3% while its failing side sits at +6.3%, one
environment away from moving 16384 down to 4096 as well.

Applied to both environments the flag names the same three arms even where the
reported count differs, so it is stable exactly where the count is not. Read a
flagged arm as a region rather than an integer; §6.7's `k*` regions are the
downstream consumer of that distinction.

`check_docs.py` verifies the documented pair by collection. It reports a
skip when the installed extras do not match the environment above; pass
`--require-test-count` to turn that mismatch into a failure, which is what a
run that has installed those extras should do. CI installs `test`, `research`,
and `stim` only, so it takes the skip rather than gating on a count it cannot
have produced.

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

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on every push to
`main`. It builds its environment out of the records rather than out of a
hand-written pin:

```bash
python benchmarks/check_record_environment.py --python       # -> 3.11
python benchmarks/check_record_environment.py --constraints  # -> numpy==2.4.6, ...
python benchmarks/check_record_environment.py                # check this machine
```

The first two read the provenance block of every stamped record, require the
records to agree on one version per package, and emit the interpreter and a pip
constraints file; CI installs `test`, `research`, and `stim` against them. The
third compares the running environment against the same agreement, and is worth
a second before a long rebuild locally.

Disagreement is a failure, not something to resolve by majority: if one record
was rebuilt under a newer NumPy than its siblings, no single environment
reproduces all of them, and the odd one out has to be rebuilt rather than
pinned to. That case is not hypothetical — it is how `exact_shot_search.json`
came to be committed under `numpy 2.5.2` while the other eight records were
built under `2.4.6`, which read as scientific drift for as long as the versions
were treated as metadata.

Two tiers run:

| job | when | contents |
| --- | --- | --- |
| `test` | pull request, push to `main` | `ruff`, `pytest --hypothesis-profile=ci`, and the record gates that finish in about a minute: `check_docs`, `check_molecular`, `check_krylov_width`, `check_clifford_hierarchy`, `check_finite_shot_optimization`, `check_warm_start` |
| `records` | push to `main`, manual dispatch | the four gates that rebuild their records from scratch: `check_finite_shot_rethink`, `check_mapping_axis`, `check_matched_h4`, `check_exact_shot_search` |

The split is by cost, not by importance. The four long gates take roughly three,
three, six, and twelve minutes here, so running them on every push to a pull
request would repeat the same computation for the same answer on every rebase.
They run in a matrix with `fail-fast` disabled, because the set of failures is
the diagnosis; the short gates in `test` run past each other's failures for the
same reason. To gate a branch on them before merging rather than after, dispatch
the workflow against that branch from the Actions tab.

`check_summaries.py` is deliberately not a gate yet: five `*_summary` pairs
declared by configs have no committed JSONL, so it fails on `main` today for
reasons that predate the workflow. `paper/check_manuscript.py` and
`paper_a_case_subspaces/check_manuscript.py` stay out for the reason given
above — they need a REVTeX installation.

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

The adapter itself needs no chemistry extra. With that extra installed,
`tests/test_fcidump.py` additionally compares its 185 Pauli coefficients
against the independent OpenFermion/PySCF construction; CI does not install it,
so that comparison is a local step.
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

### Finite-shot reconstruction and rank rule

The same bank, seed, and budget, with acquisition held at uniform and the two
*later* stages crossed instead: how a word mean is reconstructed from the
recorded histograms, and how the retained rank is chosen. See
`PLAN.md` §5 Phase 4R for the argument.

```bash
python benchmarks/run_finite_shot_rethink.py \
    --out reproductions/finite_shot_rethink.json
python benchmarks/check_finite_shot_rethink.py
```

Expected results (`benchmarks/reference_results/finite_shot_rethink.json`), at
200 replicas of 13,000 and 104,000 physical setting-shots. The `assigned`
`fixed` and `calibrated` arms at 104,000 shots reproduce the previous section's
`uniform` rows exactly, which is what makes the rest comparable to them:

- **pooling** reads each word from every QWC setting whose basis records it
  rather than from the one the partition assigned it to — the same circuits,
  shots, and histograms. On this bank a word is recorded by `3.72` settings on
  average. It moves the fixed-cutoff arm from `4.48`/`1786.21 mHa`
  median/RMSE to `2.56`/`11.70`, and the per-mode calibrated arm from
  `10.64`/`13.31` to `2.56`/`4.86`, with exact-rank recovery rising from
  `71/200` to `195/200`. The published trade of median accuracy against tail
  control does not survive it;
- **`solve_selected_rank`** picks the rank minimizing `E_hat + 2*sigma_hat`
  instead of thresholding overlap modes against their own noise. Without
  pooling it gives `4.56 mHa` median and `7.82 mHa` RMSE with no catastrophic
  replica — the fixed rule's median with better than the calibrated rule's
  tail. At the eight-times smaller budget it is the best arm on both
  reconstructions (`10.11 mHa` median, `14.33 mHa` RMSE, `0/200` catastrophes
  pooled);
- **`overlap_ridge`** — smooth per-mode damping in place of truncation — is a
  recorded **negative** result: `50.02 mHa` median and `6166 mHa` RMSE against
  truncation's `2.56` and `4.86`, and no ridge scale between `1x` and `300x`
  the noise radius recovers. The damage is in the numerator, not the metric;
  damping a direction is not truncating it.

Same caveats as above, plus one more: `pooling='shots'` and
`solve_selected_rank` are both off by default, because every committed record
in this repository was produced under the single-assignment estimator and the
fixed cutoff.

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

Both records carry a per-replica fingerprint under
`bootstrap.replica_census`: one row per replica in draw order, holding its
outcome (`accepted`, `rank`, `root_collision`, `solver`), its effective rank,
the smallest eigenvalue of the normalized overlap, and — the field that
actually decides the gate — `rank_decision_margin` with the
`overlap_threshold` and `controlling_mode` it belongs to.

The margin is recorded rather than inferred from the eigenvalue because the
solver retains a mode when it clears *its own* cutoff,
`value > max(tau_s, rel_tau * lambda_max, lambda_max / max_condition,
noise_floor)`. With a calibrated per-mode floor that cutoff varies by mode, so
a positive smallest eigenvalue can still be dropped and no sign test on it is
correct in general. `rank_decision_margin` is the signed distance from its
cutoff of the mode sitting nearest one — the mode that flips the retained
count first — and `controlling_mode` indexes it in descending-eigenvalue
order. On these two fixed records the cutoff is the `tau_s` default `1e-10`,
so the margin and the eigenvalue nearly coincide; the record states the
cutoff so the reader does not have to assume that.

When two environments disagree on the census, the checker diffs every
fingerprint field — not just the outcome label — and names the replicas that
moved, including any index present on only one side:

```
census    FAIL  136/200 replicas accepted here against 137/200 committed (-1)
   replica 0: outcome accepted -> rank, rank_decision_margin +5.679515e-04 -> -5.679516e-06
```

A marginal flip and a pipeline that moved the margin wholesale are then
distinguishable at a glance, rather than appearing as a pair of totals.

The records were regenerated to add the fingerprint, which also closed a
standing environment gap: the ill-conditioned record had been committed at
`139/200` accepted replicas and reproduced at `137/200`, shifting every
percentile band by up to `5.2e-4` relative. That gap was environmental rather
than code drift — it reproduced identically at the records' own authoring
commit — and it was not last-bit numerical noise, since on the normalized
pencil the record thresholds, `float64` and `float128` agree to `2.0e-16`
across all 200 replicas while the replica nearest the boundary sits
`-9.03e-6` from it, some ten orders of magnitude clear. The committed records
now carry the `137/200` census, which is what the manuscript quotes, and each
record is stamped with the revision, dependency set, and BLAS/LAPACK build
that produced it, so a future recurrence names its own cause.

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

## Preconditioned residual expansion

```bash
python benchmarks/run_preconditioned_expansion.py \
    --systems hubbard_2x2,hubbard_2x3,h4_equilibrium,h4_stretched \
    --total-directions 7 --seed 0 \
    --output benchmarks/results/preconditioned_expansion.json
```

`benchmarks/run_preconditioned_expansion.py` measures the corrected accuracy
hierarchy on the Phase 12 primary systems at a matched direction budget. It
writes `benchmarks/results/preconditioned_expansion.json` and refuses to write
at all if its invariants fail.

The arms answer different questions and must not be read as a single ranking:

- `orthogonal_residual` is a **regression arm, not a result**. Normalised Ritz
  residual expansion spans the Krylov space by construction — with
  `|Psi_m> = p_{m-1}(H)|psi>`, the residual `(H - E_m)|Psi_m>` lies in
  `K_{m+1}` and Galerkin orthogonality puts it perpendicular to `K_m` — so this
  arm exists only to confirm the span while holding `S = I`.
- `orthonormalized_power_krylov` is what that regression is gated against: the
  same monomial basis, SVD-orthonormalized before the solve. The gate is a
  fixed absolute energy tolerance (`REGRESSION_ENERGY_TOLERANCE = 1e-9`) plus a
  principal-angle span comparison (`REGRESSION_SPAN_TOLERANCE = 1e-7`), and it
  fires only when the two retained ranks match; the residual arm spanning
  *fewer* directions is a defect, spanning more just means the monomial basis
  went numerically rank-deficient first. A tolerance sized by the raw arm's
  `kappa(S)` would not be a test — on hubbard_2x3 that arm reaches
  `kappa(S) ~ 7e15`, which would admit an energy error of tens of hartree — so
  raw `power_krylov` is kept only as an ill-conditioned contrast and gates
  nothing.
- `davidson` is the principal method. The shift `mu` is swept over a declared
  grid, the whole curve is retained, and the selected value is the minimiser of
  the **projected** Ritz energy. That criterion is variational, so it never
  consults the exact ground energy, and the selection can be re-derived from
  the stored curve without rerunning the benchmark. The sweep's cost is
  reported as `selection_work` and folded into the arm's matvec count.
- `packet_davidson` is gated behind the word-cost preflight below and is only
  run for the `K` values that survive it.
- `matched_selected_ci` is a mandatory classical control at the same budget.
  Every accuracy statement is reported against it, because its repeated ties
  with A-CASE in the Phase 12 ledger mean the operator construction has not yet
  demonstrated greater energy compactness than classical determinant selection.

The reference-policy audit runs `model.reference`, the lowest-diagonal
determinant, a fixed physics-informed determinant where the lattice admits one,
and a fixed-seed control. Budget-matched `L`-determinant reference blocks come
from the sample-independent classical selected-CI ranking and hold
`reference_block_size + M - 1` fixed across `L`. Best/worst-determinant and
ground-state-distilled references are reported **only** under
`oracle_diagnostics` with `evidence_category = "oracle_diagnostic"`.

The packet program prices **the basis each run actually retains**. The
expansion appends a different top-`K` correction at every iteration, so each
`K` is run first (matvecs only, which is the cheap part), and then every
realized packet direction is compiled as `A_new = sum_k c_k A_k` and priced
together, cross elements included. Pricing one probe packet and reusing its
number for a seven-step trajectory would describe a one-step benchmark that
this driver does not run.

Word counts live in explicitly scoped fields, never in a bare `W`:

- `determinant_baseline_W` — the matched determinant bank at the same `M`.
- `W_total` — the full universe of the retained packet bank. This is the only
  count that is like-for-like with an A-CASE row's `W`, and it is what the gate
  compares, against `benchmarks/results/phase12_paper_b_five_system.json`.
- `W_incremental` — what the packet directions add on top of the matched
  determinant bank. Never comparable to an A-CASE total.
- `pair_support_bound` — the no-cancellation union over the products, reported
  beside `cancellation_factor`.

`grouping_contexts` carries its own `scope` and covers the same whole packet
bank, so a group count and a word count in adjacent fields describe one
experiment rather than two. Pricing runs under an abort budget, so a `K`
headed for rejection is abandoned rather than completed, and QWC grouping is
paid for survivors only.

Two costs are gated rather than paid unconditionally, and both report the
reason instead of the number when they are skipped. The packet's sector
certificate is *structural* — every part carries the reference onto one sector
basis determinant, so the packet cannot leave the sector, which is a proof
costing `O(K)` — and that is what the invariant gate reads;
`subspace_sector_certificate` runs as a numeric cross-check only up to
`numeric_certificate_max_qubits = 8`, because it builds the explicit sector
projector and that is a small-`n` object by construction. QWC grouping is
skipped above `DEFAULT_GROUPING_WORD_LIMIT = 20000` words, since the greedy
partition is quadratic in the universe size.

Two claim boundaries travel with every record this driver writes. The Davidson
preconditioner is cheap **in the classical sector backend only**: it is not yet
a bounded-support measurable packet, so its rows are stamped `exact_simulation`
with `preconditioner_category = "classical_preconditioner"` and
`implementable = false`. And the committed H$_4$ warm-start rows establish
reference *sensitivity*, not a general reference-optimisation advantage — the
2/4/6-operator ordering there is a single system and remains unexplained, and
that record's sector-projected row leaves its QND projection circuit unpriced.

Costs on one laptop-class core: the four-qubit and eight-qubit sectors are
seconds to a few minutes; the 12-qubit `hubbard_2x3` word pricing dominates
the total.

## Warm-start replication

```bash
python benchmarks/run_warm_start_replication.py \
    --systems hubbard_2x2,hubbard_2x3,h4_equilibrium,h4_stretched \
    --additions 8 --operator-ladder 1,2,3,4,5,6,8 \
    --output benchmarks/results/warm_start_replication.json
```

`benchmarks/run_warm_start_replication.py` asks whether the warm-start anomaly
in `reference_results/warm_start_h4.json` — downstream A-CASE error getting
*worse* as the ADAPT reference gets *better* — is a property of A-CASE or a
property of one system. That committed trend is a single system, a single
budget, and three points, and it is the whole evidential basis for treating
reference optimisation as an accuracy lever, so it is worth replicating before
it is built on.

The driver runs cold and warm A-CASE across the five Phase 12 primary systems
over a denser operator ladder that contains the committed `(2, 4, 6)` rungs as
a subset. Three things make it a replication rather than another run:

- **A positive control.** The `fcidump_h4_equilibrium` rows must reproduce
  `reference_results/warm_start_h4.json` to 1e-6 mHa, cold row included. A
  failure blocks the record, because a replication whose control has drifted is
  measuring a different experiment. The committed rungs are also scored
  separately from the full ladder, so a denser sweep cannot dilute what the
  original three points said.
- **A rank statistic, not three numbers.** The claim is ordinal, so it is
  tested as one: Kendall's tau-b between the ADAPT reference error and the
  downstream A-CASE error. `tau < 0` is the anomaly. The tie correction
  matters — a flat downstream error must report *no* trend rather than a
  spurious ±1.
- **A mechanism diagnostic.** `subspace_capture` is
  `||P_span |psi_exact>||^2` over the retained span, which is what actually
  bounds the achievable Ritz error; `reference_capture` is the same quantity
  for the reference alone. If capture falls as the reference improves, the
  anomaly is a span problem rather than a conditioning or optimiser artifact —
  and conditioning is ruled out independently, since the committed rows all
  carry full effective rank at `kappa(S) ~ 1.02–1.06`.

Both capture diagnostics are computed matrix-free through `apply_pauli_sum`,
with purity *verified* rather than assumed; the dense
`subspace.reference.pure_statevector` route is a small-`n` oracle that costs
more than the benchmark it diagnoses by 12 qubits.

Warm references here are sector-mixed and unprojected. The committed record's
sector-projected row leaves its QND projection circuit unpriced, and nothing in
this driver removes that caveat; the ADAPT prelude is priced in rotors,
gradient evaluations, and optimizer evaluations, never in shots.

## PRD-CASE paper suite

The suite is preregistered: a frozen manifest fixes the systems, M grid, and
per-stage scope *before* execution, and each producer stamps the resolved
config hash into every record it writes. Dry-run is the default throughout;
nothing expensive runs without `--execute`.

```bash
python benchmarks/run_prd_case_paper_suite.py --execute --results-dir RESULTS
python benchmarks/run_prd_case_matched_acase.py --execute --results-dir RESULTS
python benchmarks/run_prd_case_large_sector_packets.py --execute --results-dir RESULTS
python benchmarks/run_prd_case_finite_shot.py --execute --results-dir RESULTS
```

Records are written outside the source tree by default and are **not**
committed here; the manifests and producers are.

The manifests form a chain, each overlaying the last:

- `configs/prd_case_paper_suite.json` — the frozen v1 grid.
- `configs/prd_case_paper_suite_v2.json` — the post-pilot validity revision.
  It narrows only budgets that crossed exact effective-rank ceilings, using the
  producer invariant and never an energy or method ranking, and it carries the
  v1 config and result-archive digests so the revision is auditable against
  what it revised. v1 records stay labelled pilot/descriptive rather than being
  relabelled confirmatory. Its `system_m_budgets` contains 52 listed pairs:
  the declared 48 primary and one validation pair plus three opt-in
  `hubbard_2x4_u4` exploratory-scaling pairs. The exploratory family is not
  included in the default 49-task analysis.
- `configs/prd_case_paper_suite_v3.json` — the finite-shot extension, with
  `configs/prd_case_finite_shot_bases.json` holding the frozen packet supports
  and coefficients.

Stage scopes are deliberately unequal, and each producer enforces its own:

- `run_prd_case_paper_suite.py` orchestrates the exact tier only. It does not
  dispatch finite-shot work, and the absence of the matched A-CASE baseline is
  explicit in every record rather than passing silently as a favourable
  comparison.
- `run_prd_case_matched_acase.py` runs matched-M A-CASE at determinant and
  odd-Y word resolution, snapshotting every preregistered prefix from one
  nested trajectory. It prices the retained measurement bank *and* the larger
  selection cache, because a rejected candidate still cost a measured row.
- `run_prd_case_large_sector_packets.py` is accuracy-only. The manifest
  authorizes K=64/128 as exact accuracy probes and not as priced packets, so
  every row carries `W_total = null` and a `grouping_contexts` scope of
  `forbidden_for_accuracy_only_extension`; the producer refuses to emit a row
  that carries a word count.
- `run_prd_case_finite_shot.py` samples frozen packet Davidson under the v3
  protocol. `--prepare-only` compiles the QWC group plans without drawing
  shots, and `--summary-only` reduces completed seed records.

Verifying the freeze is a two-line check — resolve the committed manifest and
compare its digest against the one quoted in the record:

```bash
python -c "from benchmarks import run_prd_case_paper_suite as s, run_prd_case_finite_shot as f; print(s.config_sha256(s.load_manifest(f.DEFAULT_CONFIG)))"
```

Claim boundary: exact rows are exact statevector simulation with exact
algorithmic counts; finite-shot rows report sampled estimator behaviour under
one declared allocation rule. Neither tier carries a hardware-runtime,
implementability, or quantum-advantage claim.

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
Resource metrics follow PLAN.md section 9.5 (shots, circuits, unique
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

## Published A-CASE source snapshot (arXiv:2608.00560)

The first A-CASE preprint and the later DA-CASE preprint originally occupied
`paper_acase/` in sequence. The P1 source state at commit `67ea0dd` is restored
under `paper_a_case_subspaces/`; `paper_acase/` remains the P2/DA-CASE tree.
The historical README and checker remain byte-identical and therefore retain
their original `paper_acase/` prose and error-message paths. Use the restored
paths below; changing those historical files would break the source boundary.

```bash
python paper_a_case_subspaces/check_snapshot.py
python paper_a_case_subspaces/check_manuscript.py

cd paper_a_case_subspaces
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

`SOURCE_SNAPSHOT.json` pins the source commit and the Git blob id of every
historical file. The sixteen text/data/table sources must remain byte-identical.
The four PDF figures are generated outputs: `make_figures.py` may regenerate
different PDF metadata under a newer Matplotlib, while `check_manuscript.py`
still checks their existence, scientific inputs, and staleness.

## R3 protocol axis (`mapping x k`)

```bash
python benchmarks/run_protocol_axis.py       # writes reference_results/protocol_axis.json
python benchmarks/check_protocol_axis.py     # rebuilds and compares, then re-derives the contracts
```

Sweeps the dyadic block-commuting rung `k in {1, 2, 4, 8}` inside each of R2b's
five mapping arms on H4 and BeH2, discharging the two quantities the R2b record
listed under `deferred_to_r3`: `G(k)` and coverage across protocol rungs. `k` is
clamped to the arm's measured register, so a `+2q` arm records `k = 6`.

The `k = 1` column reproduces the setting count R2b froze on every arm --
H4 `913/533/351/615/403`, BeH2 `353/41/27/41/27` -- and the producer raises
rather than writing a record if it does not. That agreement is what makes the
grid an extension of the frozen record instead of a separate experiment: one
grouping rule spans it, `block_commuting_partition` at `k = 1` being exactly
qubit-wise commutation.

Labelled `structural`. It reports group counts, coverage, and the synthesis
resources a declared card prices at uniform shots; it does **not** price an
accuracy-matched `C(epsilon)`, which needs R1's exact-tier shot search and which
only BeH2 clears on its bias floor. That layer is the separate producer below.

**Known, unfixed: this record's exact reference is not bit-reproducible.** Its
`error_millihartree` column carries a loosened `(1e-10, 1e-8)` tolerance whose
stated reason is cancellation. Cancellation is real, but it is not the whole
cause. `SectorStatevectorBackend.ground_state` defaults to `method='auto'`,
which selects ARPACK for `k = 1` below the sector dimension, and ARPACK returns
a different last bit in every process — three calls on the same BeH2
Hamiltonian give `-15.566211795095189`, `...217` and `...239`, against
`-15.566211795095168` from `method='dense'` every time. Against energies of
order `1e4` mHa that `5e-14` Ha spread is a `1e-10` mHa shift on the residue,
which is the scale the tolerance was widened to. `run_protocol_cost.py` takes
the dense path for exactly this reason and compares at `1e-12` with no
per-field tolerance at all. Nothing here is wrong — the setting counts this
record exists to report are integers and unaffected — but the same one-line
change would let this record tighten too, and `run_mapping_axis.py` shares the
pattern. Both are left alone deliberately: changing them rewrites frozen floats
in records this branch was not asked to touch.

## R3 accuracy-matched `C(epsilon)` and `k*` regions

```bash
python benchmarks/run_protocol_cost.py --workers 4   # writes reference_results/protocol_cost.json
python benchmarks/check_protocol_cost.py             # rebuilds and compares, then re-derives the contracts
```

The cost half of the same `mapping x k` grid. It runs R1's exact-oracle
nonlinear shot search -- joint bitstrings from each synthesized Clifford
setting, the full measured `(S, H)` reconstruction, the `E + 2 sigma`
selected-rank sweep, replica RMSE against the exact sector ground energy -- once
per `(mapping arm, k)` cell, then turns each confirmed crossing into
`C_time(epsilon)` and the `k*` of §6.7. Same 1.6 mHa target, same
`64…65536` endpoint grid, same 30 exploratory and 100 confirmatory paired
replicas, same pass rule. Each arm draws a disjoint `SeedSequence` namespace
prefixed by its arm index, and the roots are disjoint from R1's, so the `jw`
column is an independent stream over the same bank rather than a rerun of it.

Twenty cells is the expensive gate in this repository: about two and a half
core-hours, and the BeH2 `jw k=1` cell alone -- 353 settings times 130 replicas
times the nested grid -- is half an hour of it. The `records` CI job's timeout
is sized for it.

**H4 is not priced, and the record says why.** Its bank's 3.019 mHa exact
subspace bias exceeds the target on all five arms, so no shot count reaches
1.6 mHa and no arm may be given a runtime. It is recorded with status
`bias_floor_exceeds_target` and empty cost ledgers rather than omitted:
unattainable at this target is the measurement.

**A crossing is a bracket, so a cost is an interval.** The search resolves a
shot count only to the geometric grid -- the true count lies in
`(confirmed_fail, confirmed_pass]` -- so the rung's cost lies in
`(C(confirmed_fail), C(confirmed_pass)]`. R1 additionally flags a crossing
environment-marginal when either deciding endpoint sits within +-10% of the
target, and §6.7 named this layer as that flag's consumer, so a flagged side
widens the interval by one grid step on the side that could move: a marginal
pass could fail elsewhere and push the count up, a marginal failure could pass
elsewhere and pull it down. A crossing with no confirmed failure below it is
recorded as unbounded below rather than pinned to the smallest endpoint tested.
`k*` is then every rung whose interval reaches the smallest upper bound.

**Every `k*` in this record is a region.** Across three cards, two estimators,
and five arms -- thirty determinations -- not one separates to a single rung.
Under `logical-alltoall` the region is the whole ladder on the full-width arms;
under `ion-like` it narrows to `k in {1, 2}` on every arm but three; under
`superconducting-like` it is `{1, 2, 4}` on nine of the ten arm/estimator pairs
and `{1, 2}` on pooled `jw`, with `k = 8` inadmissible on the three full-width
arms -- their deepest settings reach minimum fidelity `0.299`-`0.357` against
that card's `0.5` floor, while the two `+2q` arms clear it at `k = 6` with
`0.513`. The point argmins move
around inside those regions -- `k = 4` under `logical-alltoall` single
assignment, `k = 1` or `k = 2` elsewhere -- which is exactly the reading §6.7
forbids publishing as an integer.

**QR1 is answered, and its falsifier does not fire.** Accuracy-matched cost
reorders the protocol rungs relative to setting count in 28 of the 30
arm/card/estimator combinations. The two exceptions are
`superconducting-like` single assignment on `parity` and `bk`, where the
inadmissible `k = 8` rung leaves only three rungs to order. So setting count is
*not* an adequate proxy for cost on this instance, and the accounting layer is
not overhead. The mechanism is visible in the ordering itself: `jw` at `k = 1`
buys 353 cheap settings against 41 at `k = 2`, and the shot-to-target crossing
moves the other way.

**QR4's falsifier does fire.** Pooling never moves `k*`: on all fifteen
arm/card pairs the single-assignment and pooled regions overlap. It does move
the *point* argmin on five of them, which is precisely why the question is
answered on regions -- a relocated point inside a shared region is not evidence
that coverage relocated `k*`.

**QR3 still abstains at this tier, deliberately.** The mapping spread in
`C(epsilon)` is recorded -- among the three full-width arms it reaches `3.87x`
(`logical-alltoall`, single assignment, `k = 4`, `bk` against `jw`), while the
two `+2q` arms are priced identically at every rung under single assignment and
differ only at `k = 6` under pooling, where one grid step of shot count separates
them by `4.00x`, the largest equal-width spread in the record -- but the
question asks whether the mapping effect exceeds the *instance* spread, and an
instance spread needs two priced instances. Only BeH2 clears its bias floor, so
the record carries `qr3_accuracy_matched: abstains` and the checker fails any
record that upgrades it.

**The `jw` column reprices R1's BeH2 search under an independent stream**, and
the result is the sharpest corroboration in this record of R1's own marginal
flag. Six of the eight crossings agree exactly. The two that disagree -- `k = 4`
under both estimators, where this record confirms 16384 against R1's 4096 -- are
two of the three R1 itself flagged as environment-marginal. So the two records
agree on every crossing R1 called resolved and differ on the ones it called
unresolved, which is what "the flag is stable exactly where the count is not"
predicts. The gate is written to match: a disagreement is tolerated where either
record flags the crossing and is a failure where neither does.

Labelled `exact` on the oracle comparator with `heuristic` Monte Carlo search
uncertainty, as R1 is. These are logical runtimes under declared illustrative
cards -- not a finite-sample energy certificate, a device-noise simulation, a
hardware result, or an instance-independent preference for any mapping or block
size.

## R2b raw-pool fermion-mapping axis

The five predeclared mapping arms are constructed in
`clifford_qc/fermion_mapping.py`:

- `jw`, `parity`, and `bk` are invertible CNOT-only changes of occupation-bit
  basis; parity uses prefix parities and BK uses Fenwick-tree intervals;
- `parity+2q` and `bk+2q` expose spin-up and total occupation parity on two
  encoded qubits, derive their signs from declared `(N,S_z)`, and use the
  shared `Restriction` to rotate, fix, and delete them.

The committed experiment holds the physical generator labels fixed across
representations and evaluates H₄, BeH₂ CAS(4e,4o), equilibrium H₂O CAS(8e,6o),
and the open 2×2 Hubbard model. Run the implementation tests, rebuild the stamped
record, and run its independent checker with:

```bash
python -m pytest tests/test_fermion_mapping.py -q
python -m pytest tests/test_mapping_axis.py tests/test_grouping_packed.py -q
python benchmarks/run_mapping_axis.py
python benchmarks/check_mapping_axis.py
```

The gate compares the transported and JW projected `(S,H)` matrices, basis
size, retained rank, condition number, Ritz values, and reference energy. Pure
encoding arms additionally require a Pauli-word-universe bijection. At small
qubit count, every arm is checked against an independently materialized full or
fixed-parity dense spectrum. H₂O exceeds the declared ten-qubit dense-oracle limit;
the record states that exclusion explicitly while retaining the projected-matrix,
operator-transport, Ritz, reference-energy, leakage, and encoding gates. Every
numerical error is stored with its comparison scale and tolerance, and the checker
recomputes each gate. A sector-changing generator or incorrectly declared reference
sector aborts.

Two record fields are compared on an absolute rather than a relative scale.
`error_millihartree` and `exact_subspace_bias_millihartree` are both
`subspace energy - exact energy`: a residue of order `1e-3` mHa left by energies of
order `1e4` mHa, so seven significant digits are lost to cancellation before any
comparison happens, and the residue's own magnitude is not the scale its arithmetic
can reproduce. Measured across `OMP_NUM_THREADS=1` and `=8` on one machine, the
BeH₂ values move by `1.5e-10` mHa — enough to fail a `1e-10` gate and leave the
checker's verdict depending on the thread count. They are therefore gated at
`1e-8` mHa absolute, which clears the observed spread by ~70× while staying five
orders below the smallest bias the record reports and eight below the 1.6 mHa
target. This is the same discipline as the tie-break above: where reduction order
can be made irrelevant it is, and where it cannot — a cancellation residue is not
bit-reproducible across BLAS reduction orders — the tolerance is set by the
physical scale and stated rather than tuned until the gate passes.

`benchmarks/configs/mapping_axis.json` pins the five-arm order, selected raw-pool
labels and source-row hashes, the grouping protocol for each system, 8000 raw shots
per setting, the single-assignment estimator, the 1.6 mHa target, and all four
inputs. H₄, BeH₂, and Hubbard use the established largest-degree greedy. H₂O alone
uses the scalable full-basis-seeded first-fit cover; its setting counts are
constructive upper bounds and are excluded from the cross-instance QWC verdict.
Neither protocol claims a minimum coloring.

The committed `JW/parity/parity+2q/BK/BK+2q` setting counts are
`913/533/351/615/403` for H₄, `353/41/27/41/27` for BeH₂,
`24334/17118/9908/18108/8759` for H₂O, and `1406/798/457/907/478` for Hubbard.
QR2 passes for every arm. The corrected ratio-versus-ratio QR3 comparison is
negative: mapping spread is not smaller than instance spread for matched-greedy
QWC settings (`13.074×` versus `3.983×`) or mean word weight (`1.571×` versus
`1.489×`). Fixed-shot card rows are derived projections, not independent evidence,
because QWC uses no two-qubit measurement gates. Accuracy-matched QR3 abstains
because only BeH₂ clears the exact subspace-bias floor in all five arms. Its prices
remain `asymptotic`; this producer does not replace R1's nonlinear exact-oracle
shot search.

The original plan also named `G(k)` and coverage across protocol rungs. They are
explicitly recorded as a post-registration deferral to R3; this fixed-QWC record
does not claim to test that protocol interaction.

Ordinary reproduction consumes the committed H₂O FCIDUMP and does not require a
live chemistry build. To regenerate that immutable input with the chemistry extra:

```bash
python benchmarks/make_h2o_fcidump.py
```

The emitted provenance pins the geometry, active space, PySCF version, independent
CASCI energy, and FCIDUMP SHA-256. A changed orbital gauge changes the digest and
must be reviewed as a new benchmark input, not accepted as harmless record drift.
