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
runs under the versions its own record was produced under. The same versions
are checked before the block is written, too: `stamp_record` refuses to stamp
under an interpreter or library version no committed record declares, so a
producer cannot quietly start the split that check exists to catch.

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
pytest                                      # 1436 passed, 31 skipped
```

That install is the reference environment for the quoted pair. The count
depends on it: a missing
optional module makes pytest drop the whole test file at collection, so
each absent extra moves one file from the passed count to the skipped
count. Fourteen files are dropped that way here — `stim` (eleven of them:
`test_block_synthesis`, `test_protocol_axis`, `test_protocol_cost`, `test_bridge_stim`,
`test_clifford_hierarchy_cost`, `test_compiled_measurement`, `test_exact_shot_search`,
`test_phase4`, `test_r3d_qr3_refinement`, `test_restriction`,
`test_stim_clifford_rotors`), plus `pennylane`,
`pytket`, and `pyzx`. The
remaining `31 - 14 = 17` skips are per-test rather than per-file: `test_fermion_mapping`
and `test_mapping_axis` guard only the individual tests that reach the stim
bridge, and `test_contextual_restriction` guards the three tableau-compilation
tests, so those files still run.

`check_docs.py` enforces the pair through the identity relating them. Each of
the fourteen dropped files contributes exactly one skip and no collected tests, so
the remaining `31 - 14 = 17` skips are per-test and *are* collected:

```text
collected == passed + (skipped - files dropped at collection)
1453      == 1436   + (31      -  14)
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

### Phase 14b QWC-versus-fully-commuting sampled record

Phase 14b was split across commits so the comparison could not be redesigned
after seeing its draw. The result-free declaration remains independently gated:

```bash
python benchmarks/check_phase14b_preregistration.py
```

The checker validates `configs/phase14b_qwc_vs_fc.json` against the exact Phase 14a
merge, the committed BeH2 hierarchy and protocol-axis records, the FCIDUMP and its
provenance, and all three device cards. It freezes 1,814 measured words with the
identity analytic, QWC (`k=1`, 353 settings) against fully commuting (`k=8`, 14
settings), the same deterministic coefficient-range allocator, familywise
`delta=0.05` over both protocols and 17 fixed total-shot endpoints, the 1.6 mHa
criterion, fresh streams, and card-specific cost reporting.

That declaration merged as `bb7a76a` before the sampled producer existed. Rebuild
the one registered execution and compare its complete deterministic record with:

```bash
python benchmarks/run_phase14b_qwc_vs_fc.py
python benchmarks/check_phase14b_qwc_vs_fc.py
```

The producer records every endpoint and audit seed, integer shot vector and digest,
empirical-Bernstein radius, estimate error, compiled resource ledger, exact
reconstruction error, covariance cell, and card-specific cost. The checker
independently validates stored fields and first-passing endpoints, re-applies the
shared frozen decision and seed rules, and then replays the exact seeded streams.
The run passes all blocking gates: QWC certifies at `2^29` total shots and fully
commuting at `2^24`, so the `1/32` ratio passes the frozen `<= 1/2` rule. At
matched shots, fully commuting has a 2.22-fold covariance-aware variance advantage;
most of the 32-fold certified-endpoint gap comes from the empirical-Bernstein
certificate's per-group union bound over 179 touched QWC groups versus 7 fully
commuting groups. The four assigned/pooled empirical-to-predicted variance ratios
are `1.0042`, `1.0114`, `1.0471`, and `1.0320`; they validate the covariance
variance model, not empirical coverage of the radius that sets the endpoint. Both
matrices reconstruct with zero observed maximum error. Ion-like and
logical-all-to-all card projections favor fully commuting, while that endpoint is
inadmissible under the superconducting-like card's fidelity floor. The sampled
state is the Hartree-Fock computational-basis determinant `|11110000>`, whose
stabilizer variance structure is a special case. These are oracle measurement
results on one frozen BeH2/JW bank, not calibrated-device or instance-independent
evidence.

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
`main`. Each value gate builds its environment out of the record it verifies
rather than out of a hand-written pin:

```bash
python benchmarks/check_record_environment.py \
    --record protocol_cost.json --python       # -> 3.12
python benchmarks/check_record_environment.py \
    --record protocol_cost.json --constraints  # -> numpy==2.5.2, ...
python benchmarks/check_record_environment.py  # global consistency check
```

The first two read one named provenance block and emit its interpreter and a pip
constraints file; the composite action installs `test`, `research`, and
`stim` against that exact stamp. Every matrix row names its own reference
record. Omitting `--record` retains the stronger all-record consistency check,
which is also a manual `environment-consistency` job. Selecting a record is not
a majority vote and does not certify the other records; it lets independent
value gates run under the environments their own evidence declares while a
cross-record migration is pending.

Disagreement is a failure, not something to resolve by majority: if one record
was rebuilt under a newer NumPy than its siblings, no single environment
reproduces all of them, and the odd one out has to be rebuilt rather than
pinned to. That case is not hypothetical — it is how `exact_shot_search.json`
came to be committed under `numpy 2.5.2` while the other eight records were
built under `2.4.6`, which read as scientific drift for as long as the versions
were treated as metadata.

A gate can only report a split after it is committed, so the same contract is
enforced where a split starts. `stamp_record` refuses to stamp a record under
an interpreter or library version that no committed record declares, and every
producer stamps:

```text
UndeclaredEnvironment: this environment produces records nothing else in the
repository can reproduce:
  numpy 2.4.6 is declared by no committed record; they were built under
  2.5.2 (clifford_hierarchy_beh2.json, clifford_hierarchy_h4.json, ...)
```

Membership, not agreement, is what it tests, so a repository already split can
still be repaired: a rebuild runs under one of the versions in the split, and
demanding agreement would refuse the only run that ends it. What it refuses is
a *third* environment, which is how a split starts. The committed set is
surveyed once per process, so a producer writing ten thousand JSONL rows pays
for one directory read. Outside a checkout — an installed wheel — there are no
committed records to disagree with and the guard stands down.

Migrating the whole set is the one legitimate way to introduce a version no
record declares yet, and it is spelled out rather than inferred:

```bash
CLIFFORD_QC_ALLOW_ENVIRONMENT_MIGRATION=1 \
    python benchmarks/run_protocol_cost.py --workers 4
```

That authorizes one run. It does not migrate the set — the records left behind
still declare the old stack, and `check_record_environment.py` keeps failing
until every one of them is rebuilt.

### What the survey covers, and what it is still owed

Both record shapes are surveyed. Reading only `*.json` was a silent
under-count rather than a policy: a JSONL record stamps its rows exactly as a
JSON record stamps its document, so one that had drifted off the agreed
environment was invisible to the gate *and* uncounted by the producer guard.
Every stamped row is read, because a bench that writes rows as it runs can
straddle a version change; a record contributes its name once however many rows
it has.

Closing that gap exposed one record, and it is named rather than excused.
`benchmarks/migrations/pending_environment_migration.json` lists
`acase_ladder.jsonl`, whose twenty `qsci` rows were produced on 2026-08-06
under Python 3.11 / NumPy 2.4.6 / SciPy 1.17.1 — before the migration in #69 —
while its other seventy-two rows predate execution stamping and claim nothing.
It is not rebuilt here because it is manuscript evidence rather than a
value-gated record: no `check_*.py` rebuilds it, `paper_acase` and
`paper_a_case_subspaces` read it for figures and tables, and regenerating it
moves published inputs across all five families rather than only the stamped
rows. That is an authorship decision.

Each entry is anchored to its record's SHA-256, because the survey never opens
a file it is skipping: exempting a *name* would let the record be deleted,
corrupted, or re-stamped to any environment at all while the gate kept passing.
Absence, unreadability and drift are failures, and an entry carrying no digest
exempts nothing. Naming an outstanding record with `--record` is a failure too,
in every mode: it would otherwise survey to nothing and emit a constraints file
with no pins at all, which the composite action installs against and then
reports as a match.

A listed record is passed over by the default survey, so its declaration does
not set the pin and does not widen what the producer guard accepts — which is
the point of listing it. It stays readable, and the gate names it on stderr on
every run, reading the versions back out of the record itself: the manifest
says why a record is outstanding and never what it declares, since a version
written down twice is a version that can disagree with itself. Were the entry simply dropped
without a rebuild, `3.11` would become a version the committed set declares,
and a producer run under it would be authorized again by exactly the guard that
exists to refuse it. Deleting an entry is the last step of rebuilding its
record, never a way to quiet the report; losing the manifest exempts nothing,
since a manifest that cannot be read lists no records.

Three cost-aware tiers run:

| job | when | contents |
| --- | --- | --- |
| `test` | pull request, push to `main`, manual dispatch | `ruff`, `pytest --hypothesis-profile=ci`, and the short record gates: `check_docs`, `check_molecular`, `check_krylov_width`, `check_clifford_hierarchy`, `check_finite_shot_optimization`, `check_warm_start` |
| `structural-records` | pull request, push to `main`, manual dispatch | deterministic rebuild and lineage gates: `check_mapping_axis`, `check_protocol_axis`, `check_priceability_screen`, `check_r3_environment_migration`, `check_r3b_preregistration`, `check_r3c_preregistration`, `check_r3d_preregistration`, `check_phase14b_preregistration`, `check_g1_structural_preconditioner`, `check_r4a_preregistration`, `check_r4a_contextual_screen` |
| `sampled-records` | manual dispatch only | replica-drawing rebuild gates, each under its own record stamp: `check_r3b_margin_stop_probe`, `check_finite_shot_rethink`, `check_matched_h4`, `check_qr3b_instance_preflight`, `check_exact_shot_search`, `check_protocol_cost`, `check_r3c_lih_full_cost`, `check_r3d_qr3_refinement`, `check_phase14b_qwc_vs_fc` |

The same dispatch also runs `environment-consistency`, which requires every
stamped record to name one common environment. It is deliberately separate
from the record-local rows: one stale record is reported without preventing the
other gates from saying whether their own evidence reproduces.

The split is by cost, not by importance. The exact-tier nonlinear searches
dominate and can take from minutes to hours, so running them on every pull-request
update would repeat the same computation for the same answer on every rebase.
Both matrices have `fail-fast` disabled, because the set of failures is the
diagnosis; the short gates in `test` run past each other's failures for the same
reason. A pull request that changes a sampled producer, config, or record must
name a manual dispatch against its exact branch head before merge. A
structural-only preregistration is covered by the automatic jobs and does not
spend the sampled matrix.

### The python 3.12 / numpy 2.5.2 / scipy 1.18.0 migration

The earlier migration rebuilt every record that existed at the time on this stack.
`priceability_screen.json` (PR #72) and `r3b_margin_stop_probe.json` (PR #73)
were then produced under Python 3.11 / NumPy 2.4.6 / SciPy 1.17.1 — the stack
this repository used *before* the migration, and the default of the containers
the runs happened in — so the repository became split again and the default
all-record check correctly failed. Nothing chose the older libraries: NumPy
2.5.2 and SciPy 1.18.0 both require Python >= 3.12, so an interpreter one minor
version back resolves the newest releases that still support it. Those two
records have now been genuinely rebuilt rather than relabelled, so the global
check is green again. SciPy 1.18.0 remains the accepted target; moving to 1.18.1
is out of scope. No pin is chosen by majority: record-local jobs read one named
stamp, while the global check refuses any split.

The repair is itself gated:

```bash
python benchmarks/check_r3_environment_migration.py
python benchmarks/check_record_environment.py
```

`benchmarks/migrations/r3_environment_3_12.json` binds the old and new SHA-256
and Git-blob identities, the frozen configs, the exact source `main` commit and
the target environment. The R3c config remains byte-for-byte unchanged. Its
historical rationale therefore still names the 3.11 draw (30 resolved, 10
confirmation failures), while the live target-environment R3b record reports
29 resolved and 11 confirmation failures. Both reject the bank, both have zero
grid-fit failures and one ceiling cell, and neither authorizes a full run.

A repeat is what the producer guard above now prevents: run under an
undeclared interpreter today and the run refuses to stamp, naming the versions
the committed records were built under, instead of writing a twelfth record
that reproduces beside none of them.

What moved in the earlier whole-set migration is worth stating precisely,
because the two halves behave
differently and a reader comparing figures across this boundary needs to know
which half they are looking at.

**Nothing this project publishes moved.** All thirty `k*` regions in
`protocol_cost.json` are identical and none moved. All sixteen confirmed
shot-to-target crossings in `exact_shot_search.json` are identical — every
`confirmed_passing_effective_shots_per_setting`, every `confirmed_failing`,
every environment-marginal flag. Every resource count is identical: `W`,
setting counts on all twenty-five mapping-axis arms, group counts, basis sizes,
retained selections. Both negative verdicts stand — QR3 still abstains at the
exact tier, and the LiH candidate is still
`rejected_unresolved_at_frozen_grid`. The two `clifford_hierarchy_*.json`
records preserve their scientific payload but carry regenerated provenance;
`protocol_axis.json` preserves its discrete decisions and resource counts while
small floating diagnostics move in the last bits. None of those three files is
byte-identical.

**The replica statistics underneath them did move.**
`numpy.random.Generator` carries no cross-version bit-stream guarantee (NEP 19
froze `RandomState` for exactly this purpose), so the draws differ, a few
replicas out of two hundred land on a different selected rank, and the
aggregates follow. In `exact_shot_search.json` that is 489 floats and nine
`below_exact_count` integers, moving the millihartree statistics by roughly one
to three percent. In `finite_shot_rethink.json` it is eleven integers and 103
floats, including a 423 mHa move in the `pooled_ridge` arm — an arm that record
exists to characterise as a catastrophic failure mode, so the size of that move
is a property of the arm and not of the migration. The QR3b preflight moved one
cell from resolved to unresolved, 19 to 18, which reinforces its rejection
rather than threatening it.

So: **a bias, an RMSE or a percentile read from a pre-migration record is not
comparable with the same field read after it. A crossing, a `k*`, a setting
count or a verdict is.** The distributional statistics describe a different
draw; the decisions taken on them are the same decisions.

The exact-arithmetic records confirm the split from the other side. No integer
moved in any of them, and the largest float movements are last-bit: 8.0e-15 Ha
(8.0e-12 mHa) on `matched_h4`'s `error_hartree` and 7.0e-6 microseconds on a
`fixed_shot_time_us` of 1.0e9. Those two are the reason
`CROSS_MACHINE_ATOL`/`_RTOL` needs both legs — the first clears only on the
absolute one, the second only on the relative one, and they are seventeen
orders of magnitude apart.

One dependency the migration exposed is now fixed. For BeH2,
`configs/mapping_axis.json` pins `selection_payload_sha256`, the digest of a
canonical selection payload containing the record schema, system, selected
labels, selected ground energy, and bank provenance. Regenerated top-level
provenance therefore does not invalidate an unchanged bank, while a changed
selection does. The `h4` row continues to hash its selected ladder row.

### What a record gate can and cannot reproduce

The gates compare a committed record against one rebuilt on the spot, and those
two numbers come from different machines. That distinction is the whole of the
tolerance story, and it was worth measuring rather than assuming.

*Within one machine these records are bit-exact.* Rebuilding
`finite_shot_rethink.json` and diffing every compared field against the
committed one gives a worst drift of exactly `0.0` — no differing field at all
— and it stays `0.0` at `OMP_NUM_THREADS=1` and `=4` alike. So the position
these checkers have always taken, that widening is the wrong response because
nothing here is noisy in the run-to-run sense, is correct and is not being
retreated from.

*Across machines they are not.* On `main`, the same gates drift by `3.4e-12` to
`1.0e-11` absolute and up to `9.8e-11` relative, on millihartree quantities of
order `0.07` to `2.6`. OpenBLAS selects its kernels by instruction set as well
as by thread count, so a runner with a different CPU sums a reduction in a
different order. Part of that is closable and now is: `ci.yml` pins
`OMP_NUM_THREADS=1`, because the same `eigvalsh` call returns four distinct
last bits across thread counts 1–4 — deterministically within each, spread
`4.5e-15`, and only on the 256-dimension arms, since the 64-dimension `+2q`
arms sit below the threading threshold. The instruction-set half is not
closable from a workflow.

So a committed-versus-rebuilt comparison is a cross-machine comparison, and
`clifford_qc.reproducibility.CROSS_MACHINE_ATOL` / `_RTOL` is the floor it has
to clear: `1e-9`, an order above the worst observed drift. On the measured
energy-difference fields of order `0.07` to `2.6` mHa, a change of `1e-8` mHa
still fails the gate and the floor remains nine orders below the 1.6 mHa target.
Fixed numerical contracts do not inherit that blanket allowance: their checkers
use exact path or key overrides, including the finite-shot overlap threshold and
the QR3b invariant tolerances. Before this floor existed the cross-machine
comparisons ran at `1e-12` — inside their own noise — which is why three gates
were red on `main` across five merges and why `check_exact_shot_search` once
went red and then green on inputs that had provably not changed.

Re-derivations keep the tight comparison, because they cross no machine
boundary: `check_protocol_cost.py` recomputes its verdicts from the record's
own contents at `atol=0.0`, and `check_mapping_axis.py` recomputes the QR3
summary at `1e-12`. Both run in the process that holds the record.

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
five mapping arms on H4, H4-converged and BeH2, discharging the two quantities
the R2b record listed under `deferred_to_r3`: `G(k)` and coverage across
protocol rungs. `k` is clamped to the arm's measured register, so a `+2q` arm
records `k = 6`.

The `k = 1` column reproduces the setting count R2b froze on every arm --
H4 `913/533/351/615/403`, H4-converged `913/533/351/615/405`,
BeH2 `353/41/27/41/27` -- and the producer raises rather than writing a record
if it does not. That agreement is what makes the grid an extension of the
frozen record instead of a separate experiment: one grouping rule spans it,
`block_commuting_partition` at `k = 1` being exactly qubit-wise commutation.

Labelled `structural`. It reports group counts, coverage, and the synthesis
resources a declared card prices at uniform shots; it does **not** price an
accuracy-matched `C(epsilon)`, which needs R1's exact-tier shot search. That
layer is the separate producer below.

**Fixed here: this record's exact reference is now bit-reproducible.**
`SectorStatevectorBackend.ground_state` defaults to `method='auto'`, which
selects ARPACK for `k = 1` below the sector dimension, and ARPACK returns a
different last bit in every process -- three calls on the same BeH2 Hamiltonian
gave `-15.566211795095189`, `...217` and `...239`, against
`-15.566211795095168` from `method='dense'` every time. Against energies of
order `1e4` mHa that `5e-14` Ha spread lands as a `1e-10` mHa shift on the
`error_millihartree` residue, which is part of what the loosened `(1e-10, 1e-8)`
tolerance was absorbing under the heading of cancellation.

`run_protocol_axis.py` and `run_mapping_axis.py` now both take the dense path,
which `run_protocol_cost.py` already did. Regenerating both records moved
exactly the two energy-difference fields and nothing else -- setting counts,
coverage, weights, synthesis resources, device costs and every digest are bit
identical to the previous record -- and the two records' `exact_sector_energy`
now agree with each other exactly, where before each carried its own ARPACK
draw.

The `(1e-10, 1e-8)` tolerance stays. ARPACK was one cause of the drift, not the
only one: `check_mapping_axis.py` documents a measured `1.5e-10` mHa spread
between `OMP_NUM_THREADS=1` and `=8` on the same code path, which the default
`atol=1e-10` sits directly on top of and which this change does not touch.
Tightening the gate on the strength of the ARPACK fix alone would trade a
reproducible record for a flaky one.

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

**Two gates, and the structural grid's three banks fail them in two different
places.** `cost_layer_scope` partitions that grid into what this record prices
and what it defers, with a reason on each deferral, and
`check_protocol_cost.py` fails a record where a system is merely absent from
both.

*`h4` fails the accuracy gate.* Its budget-8 bank's 3.019 mHa exact subspace
bias exceeds the target on all five arms, so no shot count reaches 1.6 mHa and
no arm may be given a runtime. It is recorded with status
`bias_floor_exceeds_target` and empty cost ledgers rather than omitted:
unattainable at this target is the measurement.

*`h4_converged` passes that gate and is right-censored by the resolution one.* At 0.766 mHa it
clears the floor with a factor of two to spare, and that is what makes QR3
eligible at the asymptotic tier in `mapping_axis.json`. But a crossing is
resolvable only inside the `64…65536` endpoint grid, and this bank's crossings
are not. Its word universe is 7926 against BeH2's 1814 on the full-width arms
and 2047 against 511 on the `+2q` arms; four times the words reconstructed from
the same shots is four times the pencil variance, which moves the confirmed
crossings from BeH2's 4096-16384 up to 16384-65536, against a grid whose last
point is 65536. A reduced-replica scoping probe -- 2 exploratory and 2
confirmatory, recorded in the config and labelled `is_a_record: false` because
no number in it may be quoted as a cost -- left 12 of 20 single-assignment
cells unresolved and put three of the eight that did confirm on the final grid
point. It also took 49 minutes against the BeH2-only probe's 2, which
extrapolates to roughly a day of four-core time at the headline replica counts,
for a record that would still be mostly unpriced.

So the evidence status is `right_censored` at a ceiling of `65536`, while
`further_search: deferred` records the separate decision not to extend the
frozen grid. The censoring is on resolution, not accuracy, and the distinction
is the finding: **clearing the bias floor is necessary for a price and is not
sufficient.** What would lift it is endpoints above 65536, which
`SEARCH_ENDPOINTS` pins and `exact_shot_search.json` shares -- a change to R1
and R3 together, not a scope change to R3.

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
instance spread needs two priced instances. A second bank that clears the bias
floor now exists -- `h4_converged` -- and it is deferred here on resolution
rather than accuracy, per `cost_layer_scope` above. So the record still carries
`qr3_accuracy_matched: abstains`; the checker re-derives that verdict from the
record's own priced cells rather than reading it, and fails any record that
upgrades it or that drops a structural system without deferring it.

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

## QR3b independent-instance preflight (LiH)

PR #67's H4-converged result remains right-censored at the frozen
`64…65536` endpoint grid. The separately labelled QR3b extension therefore
tests a chemically independent candidate instead of extending that grid or
silently replacing the original H4-versus-BeH2 estimand:

```bash
# Optional chemistry-extra regeneration of the immutable input.
python benchmarks/make_lih_fcidump.py

# The only authorized stochastic run: 2 exploratory + 2 confirmatory replicas.
python benchmarks/run_qr3b_instance_preflight.py --workers 4
python benchmarks/check_qr3b_instance_preflight.py --workers 4
```

The committed input is equilibrium LiH at 1.5949 Å in STO-3G, reduced to the
lowest four RHF canonical spatial orbitals with four active electrons and no
frozen electron pair. Its PySCF CASCI/FCI energy is `-7.863222550891 Ha`.
Ordinary reproduction consumes the FCIDUMP and does not require PySCF; the
builder emits a provenance record binding the geometry, orbital choice,
independent energy and FCIDUMP digest.

Selection is fixed before mapping costs are inspected. ACASE starts from the
Hartree–Fock determinant, receives the complete symmetry-preserving rank-at-most-two
pool under a nonbinding implementation budget, and stops only on its intrinsic
predicted-lowering threshold. It reproducibly selects 13 basis vectors with
energy `-7.863222354968 Ha`, an exact-sector bias of `0.000196 mHa`, so all
five mapping arms clear the `1.6 mHa` bias gate.

The producer then reuses the exact R1/R3 nonlinear search on the five mappings,
`k in {1,2,4,8}`, and both estimators, but freezes the probe at `2+2`
replicas. Its record is labelled `scope_decision_only`; inherited fixed-shot
prices and every accuracy-matched cost bracket or `k*` derivative are stripped,
so it cannot answer QR3b and always records `full_run_authorized: false`. A
later `30+100` run would require a separate preregistration even if every
probe cell resolves strictly before `65536`.

**Result: rejected on resolution, not accuracy.** The bias gate passes on all
five arms, but only 19 of 40 mapping/rung/estimator cells confirm a crossing
inside the frozen grid. Twenty-one remain unresolved; of the 19 that resolve,
9 land exactly on `65536`, 9 on `16384`, and 1 on `4096`. Thus only 10 of
40 cells have the endpoint headroom the preregistration requires. The record
therefore reports `rejected_unresolved_at_frozen_grid`,
`eligible_for_full_run: false`, and `full_run_authorized: false`. This is a
negative candidate-screening result, not an accuracy-matched cost or QR3b
verdict, and it does not alter H4-converged's independently frozen
right-censoring.

## R3S priceability screen

The QR3b preflight spent forty sampled cells to reach a verdict its own
acceptance gates could have reached in seconds: its record carries
`screen_would_have_rejected_before_probe: true`, meaning the word-universe
ceiling was evaluated *after* the probe rather than before it. This phase runs
that gate first, across every declared candidate:

```bash
python benchmarks/run_priceability_screen.py
python benchmarks/check_priceability_screen.py
```

Nothing here is sampled and no chemistry extra is needed. Every quantity is a
bias or a word count in deterministic double-precision arithmetic — dense
eigensolves, reproducible run to run on fixed BLAS threading, but floating-point
and compared to tolerance by the checker rather than exact. Each candidate is
walked along the greedy ordering its own preregistration already froze — the
four R2b banks keep their `mapping_axis` labels, LiH keeps the QR3b ones, and no
ordering is re-derived — and two structural quantities are recorded at every
prefix: the exact subspace bias against the `1.6 mHa` target, and the word
universe. A candidate is admissible when some prefix clears the target with the
declared margin at a binding word universe within the ceiling.

**The screen counts words as `protocol_axis` and the QR3b probe do, excluding
the identity.** Two conventions are in the tree and they differ by one:
`mapping_axis` includes the identity word (BeH₂ full-width `1815`), the other
two records do not (`1814`). The `2048` ceiling was calibrated in the second, and
the identity is the one word a shot budget never buys — its expectation is fixed
by normalization — so it contributes nothing to the reconstruction variance the
ceiling stands in for. `check_priceability_screen.py` calibrates against
`protocol_axis.json` for the same reason; comparing across the two conventions
would fail by exactly one on every arm and say nothing about the gate.

**The stopping rule is the phase's one methodological change, and it is declared
rather than fitted.** The frozen rule runs A-CASE to its own predicted-lowering
threshold. That is accuracy-maximizing, while the exact-tier price is
resolution-limited, so the two pull apart: on LiH the greedy spends four orders
of magnitude of bias headroom to buy `5.4×` the word universe. The screen stops
instead at the smallest prefix whose bias is at or below
`accuracy_target / margin_factor`, a function of the accuracy target alone. The
margin is `3` because the shot search's pass rule bounds replica RMSE and RMSE
combines bank bias with sampling scatter in quadrature: at margin `3` the bias
takes `0.533 mHa`, which is `(1/3)² = 11.1%` of the *MSE* budget, and the
statistical *RMSE* allowance falls from `1.600` to `1.5085 mHa`, a `5.7%`
reduction. Those are two different fractions, and the smaller is not a share of a
shot budget consumed. A one-generator prefix is excluded by declaration — it is
the identity alone, so its Ritz value is the Hartree–Fock energy and its span is
not a subspace.

**The margin factor is `declared_here_not_preregistered`.** The quadrature
argument motivates having a margin; it does not pick `3` out of `2` or `5`, and
this config arrives in the same commit as the first result it produces, so the
LiH admission is exploratory evidence for the rule rather than a test of it. The
record says so rather than claiming a preregistration the history does not
support, and every candidate carries a `margin_sensitivity` range re-derived from
its own walked rows so a reader can see whether a verdict turns on the number:
LiH is admitted for every margin from `1` to about `4.33`, and the three rejected
candidates stay rejected at every margin at or above `1`. None does. The rule is
frozen from this commit; the preregistered use is the next candidate screened
under it.

Selecting a prefix on measurement cost is exactly what QR3b's
`selection_may_not_use_mapping_cost_direction` gate forbids, so the rule may not
see a mapping, an arm, or a word count. It is evaluated on the source-side bias,
which the linear encoding family leaves invariant, and the record carries the
per-arm bias at the chosen prefix so the checker can confirm the arms agree to
rounding rather than take the declaration on trust.

**Result.**

| candidate | margin stop | binding `W` | intrinsic stop | intrinsic `W` | verdict |
|---|---|---|---|---|---|
| `lih_cas4e4o` | `M = 2`, `0.370 mHa` | `1439` | `M = 13`, `0.0002 mHa` | `7740` | **admissible** |
| `beh2` | `M = 3`, `0.0695 mHa` | `1223` | `M = 5`, `0.0033 mHa` | `1814` | **admissible** |
| `h4_converged` | none | — | `M = 15`, `0.766 mHa` | `7926` | rejected on the ceiling |
| `hubbard_2x2` | none | — | `M = 9`, `863 mHa` | `5536` | rejected on the ceiling |
| `h2o_cas8e6o` | none | — | `M = 9`, `14.2 mHa` | `143116` | rejected on the ceiling |

LiH is admissible at a binding word universe *below* BeH₂'s `1814` — the one
bank this repository has ever priced inside the frozen `64…65536` grid — while
the same instance at the greedy's own stopping point sits four times above the
ceiling. So the QR3b rejection was of a stopping rule, not of an instance, and a
second priceable candidate is reachable with `SEARCH_ENDPOINTS` untouched.
`h4_converged` is rejected under both rules and its deferral stands: its bias
never reaches the margin anywhere in the frozen ordering. The binding-arm rule is
what carries it — its two `+2q` arms sit at `2047`, one word under the gate, so a
screen reading any single reduced arm would have admitted a bank the frozen grid
has already failed to resolve.

**Why the walk may stop early.** Two monotonicities, both re-derived per
candidate by the checker rather than assumed: bias is non-increasing along the
greedy prefix, since a Ritz value cannot rise as the span grows, so the first
clearing prefix is the unique smallest one; and the word universe is
non-decreasing, since a longer prefix adds matrix-element pairs and removes none,
so once `W` passes the ceiling with the bias gate unmet, no longer prefix can
pass either. `h4` is not screened separately for the same reason: its frozen
labels are the first nine of `h4_converged`'s, so over a shared prefix the two
banks are the same object, and the shared walk leaves the ceiling at `M = 3`,
inside `h4`'s own ordering and without having cleared the margin. The checker
re-derives that from the two frozen label lists and the walked rows.

Labelled `structural`. The screen may authorize or withhold a sampling probe; it
may not price `C(epsilon)`, answer QR3 or QR3b, or reclassify any frozen
censoring decision. Candidates are admitted or rejected **under this declared
screen**: the `2048` ceiling is an operational admission threshold calibrated on
one priced bank, not a demonstrated necessary condition for priceability —
coefficient magnitudes, grouping, estimator variance and pencil conditioning all
bear on whether a bank resolves, and a rank-2 pencil may condition differently
from BeH₂'s rank-5. So an admission is not a demonstration that a bank will
resolve, and a rejection is not a demonstration that it cannot. Deciding that is
what a probe is for, and `check_priceability_screen.py` fails any record that
grows a cost field.

## R3b margin-stop probe — preregistration

R3S admitted the LiH margin-stop bank; this is the preregistration for the probe
that tests it, and it is deliberately a separate commit from the run:

```bash
python benchmarks/check_r3b_preregistration.py
```

No sampling happens here and none has happened under this config. The commit
carries `benchmarks/configs/r3b_margin_stop_probe.json` and its checker, nothing
else, because **preregistration is a property of commit order** (`PLAN.md` §13):
a rule is preregistered with respect to a result only if the commit declaring it
precedes the commit reporting that result. R3S could not meet that for its margin
factor and says so in its own record. This config is that rule's first
preregistered *use* — the rule is frozen upstream, and here the bank, gates and
seeds are fixed before any shot is spent.

**What is frozen.** LiH CAS(4e,4o) at `M = 2`, labels `I` and `E(6,7<-2,3)`,
exact subspace bias `0.3698 mHa`, binding `W = 1439` against BeH₂'s `1814` — the
one bank this repository has priced inside the grid. QR3b's protocol is inherited
endpoint for endpoint: the `64…65536` grid, `k ∈ {1,2,4,8}`, both estimators,
`2+2` replicas, and the scope-decision-not-cost-record contract. Seed roots are
fresh (`130813000` / `130913000` / `131013000`, the `13` naming execution step
13b) so the two probes' replica and bootstrap streams cannot alias.

**The one substantive difference from QR3b**, declared rather than silent:
`selection_must_stop_intrinsically` is `false`. QR3b required the A-CASE greedy's
own predicted-lowering threshold; this bank comes from the R3S margin rule
instead, because the intrinsic rule is accuracy-maximizing while the exact-tier
price is resolution-limited, and on this instance the two conflict — the greedy
spends four orders of magnitude of bias headroom to buy `5.4×` the word universe.
The gate carries its reason in `selection_must_stop_intrinsically_basis`.

**What the checker proves, in deterministic double-precision with nothing
sampled.** The committed FCIDUMP and provenance still hash to the declared
digests and the provenance still binds the FCIDUMP; the lineage digests still
bind the R3S record and config and the QR3b config; the declared labels really
are a prefix of QR3b's frozen ordering, so this is a shorter draw from a decided
ordering rather than a fresh selection; the protocol is QR3b's field by field
with non-aliasing seeds; and no result, verdict, or cost field is present — a
`provenance` *object* would make this a record, though the `provenance` *path*
it legitimately carries would not.

The load-bearing one is the last: the checker re-runs the R3S margin rule over
the frozen ordering and requires that it select exactly the declared prefix, then
rebuilds the bank through all five arms and compares bias, per-arm `W`, per-arm
settings, and the binding `W` against both the declared values and the R3S
record. A prefix chosen because it is cheap to measure rather than because the
accuracy target selects it fails there. `tests/test_r3b_preregistration.py` pins
that with a hand-picked `M = 3`, which is *more* accurate and still rejected.

**What it does not do.** It authorizes no run — `full_30_plus_100_run_authorized_by_this_config`
is `false`, as in QR3b. When the probe runs it will be scope-decision evidence
only: it may accept or reject this bank for a later preregistered `30+100` run,
and it may not price `C(epsilon)`, answer QR3 or QR3b, or alter any frozen
censoring decision. A pass buys eligibility; a failure is a result about the
`2048` ceiling, which would then be recalibrated on two points rather than one.

## R3b margin-stop probe — the run

The preregistration's other half. The bank it froze has now been probed:

```bash
python benchmarks/run_r3b_margin_stop_probe.py --workers 4
python benchmarks/check_r3b_margin_stop_probe.py --workers 4
```

Everything the probe could vary is QR3b's, unchanged — the `64…65536` grid,
`k ∈ {1,2,4,8}`, both estimators, `2+2` replicas, the same
scope-decision-not-cost-record contract — so the two records differ in the bank
and nothing else, and comparing them is like for like. About three minutes on
four workers, against QR3b's 150 CI-minutes.

**Result: the bank is rejected, and the gate is not relaxed.**
`rejected_unresolved_at_frozen_grid`, `eligible_for_full_run: false`,
`full_run_authorized: false`. In the target-environment redraw, 29 of
40 cells resolved and 11 did not,
and the preregistration requires every cell to resolve strictly before `65536`.
So no `30+100` run is authorized on this bank, and the earlier expectation that
the margin-stop bank would deliver the exact tier's second priced instance is
**not** borne out.

**But the failure mode changed completely, and that is the finding.**

| | QR3b — intrinsic stop, `W = 7740` | R3b — margin stop, `W = 1439` |
|---|---|---|
| resolved | 18 / 40 | **29 / 40** (30 / 40 historical) |
| `not_bracketed_within_search_grid` | **15** | **0** |
| confirmation failures | 7 | 11 (10 historical) |
| cells at/over the `65536` ceiling | 9 | 1 |
| modal passing endpoint | `65536` | `16384`, then `4096` |

`not_bracketed_within_search_grid` means no crossing was found anywhere inside
the grid — the failure the word-universe ceiling is a proxy for, since `W` sets
the reconstruction variance the search has to overcome. It accounted for 15 of
QR3b's 22 unresolved cells and **none** of R3b's. Every crossing this bank has
lies inside the grid, and the passing endpoints fell by roughly two grid steps.

What remains is a different mechanism. In the target-environment redraw,
eleven cells found an exploratory crossing the two confirmatory replicas did
not reproduce, and one cell bracketed only at the last grid point. The
historical 3.11 draw had nine unconfirmed exploratory crossings and one
nonmonotone confirmation, with the same one ceiling cell. These are the `2+2`
probe's replica count and its headroom, not statements about `W`.

**So the record's verdict is
`corroborated_on_grid_fit_headroom_marginal`**, and the split behind it is
labelled `post_hoc_diagnostic_not_preregistered`: the preregistration declared a
corroborate/falsify binary, and the drawn cells showed that binary conflates two
mechanisms with different remedies. Refining it after the draw is legitimate
only because it is labelled and because it leaves the preregistered decision
untouched — the gate, its inputs and its rejection are exactly as frozen, and
the new field only says which mechanism produced the rejection.

**What this does not license.** Not a relaxed gate. A `2+2` probe was sized
against grid-fit failures, and whether it is the right instrument for
confirmation failures is a real question — but answering it by widening the
probe *after* seeing which cells failed is precisely the move the preregistration
discipline exists to prevent. That question needs its own preregistration. Nor
does it license widening `SEARCH_ENDPOINTS`: the grid was never the binding
constraint here.

`check_r3b_margin_stop_probe.py` rebuilds the record, re-derives the decision
from the probe's own cells, re-derives `screen_prediction` from the resolution
gate and the failure-mode counts, and separately asserts that QR3b's record
still carries its own rejection unchanged — this phase extends that record and
may not reopen it.

It also refuses a record that inherits the preregistration's claim boundary. The
config's boundary opens *"No sampling has been performed under this config"* —
true of the commit that landed it, false of a record reporting forty sampled
cells — so the record states its own boundary and quotes the config's under
`preregistration.config_claim_boundary_at_landing`, where it remains a true
statement about what it describes. Editing the preregistration instead would
undo the thing landing it first exists to establish.

## R3c LiH full-cost run — preregistration

R3c was preregistered against the historical rejected 2+2 scope probe: 30 of
40 cells resolved, zero failed to bracket anywhere in the grid, ten failed
confirmation, and one resolved only at the `65536` ceiling. The later genuine
target-environment redraw reports 29 resolved and eleven confirmation failures,
while preserving zero grid-fit failures, one ceiling cell, and rejection. In
both records `eligible_for_full_run: false` and `full_run_authorized: false` are
immutable. R3c does not rewrite either finding. It is a new declaration,
motivated by the probe's explicitly labelled post-hoc failure-mode split, for
one execution of the target instrument:

```bash
python benchmarks/check_r3c_preregistration.py
```

That command is deterministic and samples nothing. The result-free
`benchmarks/configs/r3c_lih_full_cost.json` freezes the exact bank R3b piloted
(`M = 2`, binding `W = 1439`), all five mapping arms,
`k ∈ {1,2,4,8}`, both estimators, the unchanged `64…65536` endpoint grid,
30 exploratory and 100 confirmatory replicas, nested endpoints, 10,000
bootstrap replicates, one-sided `delta = 0.05`, the exact-tier execution
environment (Python 3.12 / NumPy 2.5.2 / SciPy 1.18.0 / Stim 1.16.0), and fresh roots
`132813000 / 132913000 / 133013000`. Those roots are disjoint from QR3b,
R3b, the R1 exact search, and the existing R3 cost record.

The crossing rule is unchanged: every confirmatory solve must succeed and the
one-sided 95% bootstrap upper bound on replica RMSE must be at most `1.6 mHa`.
A finite interval requires a confirmed failing predecessor and confirmed
passing endpoint. A cell without one at `65536` is right-censored; it is not
assigned infinite cost and does not license a wider grid. Logical time may be
reported only under the three existing device cards, pinned by name and SHA-256.

The checker binds the current R3b config by canonical SHA-256 and its historical
sampled record by Git blob SHA-1, then follows the declared migration manifest
to the live rebuilt successor. It preserves both 2+2 rejections and their
respective diagnostic counts, reuses R3b's deterministic bank reconstruction,
rejects any candidate or protocol drift, checks every random root against the
prior streams, and refuses any result, verdict, execution stamp, or cost field.
Its claim boundary is deliberately tenseless: this config carries no sampled
result; a later record may report only the frozen run.

No producer or sampled record lands with this preregistration. The future
`run_r3c_lih_full_cost.py`, its record, and
`check_r3c_lih_full_cost.py` must land in a separate commit. Only that record
can show whether LiH becomes the second priced instance; otherwise QR3 remains
undetermined.

## R3c LiH full-cost run — the producer

`benchmarks/run_r3c_lih_full_cost.py` draws the authorized run. It landed
without a record, ahead of the run; the sampled `r3c_lih_full_cost.json`, its
checker and the `sampled-records` matrix entry that gates them followed, and
*the result* below reports what the run found. This section is about the
producer and the gates it applies before it spends a shot.

```bash
python benchmarks/run_r3c_lih_full_cost.py --skip-run   # structural half, no replicas
python benchmarks/run_r3c_lih_full_cost.py --workers 4
```

Replica counts are not command-line arguments. They come from the config, as do
the arms, the block sizes, the estimators, the endpoint grid, the bootstrap
replicates, the pass rule and the three seed roots — a producer that accepted
`--confirmatory-replicas` would accept a record drawn at a count the
preregistration did not freeze. `--skip-run` builds the structural half and
writes a record whose `full_cost_run.executed` is `false`; it is a preview, and
the committed evidence is the run.

**Every gate that can fail runs before a shot is drawn**, in this order, because
the run costs hours and a lineage failure found at the stamp costs all of them:

1. the whole result-free checker, re-run on the config the producer is about to
   consume, plus the config file's SHA-256 against the digest the migration
   manifest froze when it landed — `static_problems` validates whatever config
   it is handed, so only the digest says the preregistration was not edited
   after the fact;
2. the execution stack against the frozen Python 3.12 / NumPy 2.5.2 /
   SciPy 1.18.0 / Stim 1.16.0. This is stricter than `stamp_record`'s guard and
   asks a different question: not whether the run belongs to the repository's
   record set, but whether it is the stack these seed roots name a stream under.
   Under another NumPy the same roots draw different shots;
3. the three device cards by name and SHA-256;
4. R3b's record, which must still carry `full_run_authorized: false`. R3c's
   authorization is its own config's `full_30_plus_100_run_authorized_by_this_config`,
   and the two only stay distinguishable while the pilot keeps saying what it
   said;
5. `protocol_cost.json` as the other half of a QR3 comparison — same replica
   counts, endpoint grid, estimators, cards and stack, disjoint seed roots, and
   BeH2 still `searched`. Two prices are comparable only if one instrument
   measured both;
6. the bank: the R3S margin rule re-derived over the frozen QR3b ordering
   through R3b's own implementation, then each arm's word universe and qwc
   setting count against the preregistered per-arm values, and every arm's bias
   against the 1.6 mHa floor.

**The readout is the config's own.** Each of the forty `(arm, k, estimator)`
cells is classified by the frozen crossing rule: a finite interval needs a
confirmed failing predecessor and a confirmed passing endpoint. Three distinct
things fall short and the record names them separately rather than pooling
them — no confirmed crossing anywhere in the grid, which is censoring by the
grid; a passing endpoint R1 flagged environment-marginal at `65536`, which has
no finite upper bound; and a crossing at the first endpoint tested, which has no
confirmed failing predecessor. `pricing.status` is then
`priced_second_instance` when finite intervals survive on a declared card and
`right_censored_at_frozen_grid` when none does. Censoring is not infinite cost
and licenses no wider grid, and the record says so in the field
`wider_grid_licensed_by_this_record`.

The censored cells are also tabulated by their search status. That table is
data and no gate reads it. R3b's split of grid fit from confirmation power was
labelled `post_hoc_diagnostic_not_preregistered`, and reporting the same counts
here does not promote it into a test.

**QR3 is re-derived only on a second priced instance**, which is the
preregistration's gate. When the LiH bank prices, the producer puts its cells
beside the frozen BeH2 cells under one instrument and reports the three-valued
mapping-versus-instance verdict; when it does not, the record says
`not_re_derived` with the reason and QR3 stays undetermined. Either way
`protocol_cost.json` is read and not rewritten: its own
`qr3_accuracy_matched: abstains` remains a true statement about the record that
carries it, and the combination lives in the new record.

**The claim boundary is inherited from the config**, which is what R3c's
tenseless wording was for. R3b's record could not inherit its config's — that
one opens *"No sampling has been performed under this config"* and would have a
forty-cell record deny its own contents — so it had to state its own and quote
the config's beside it. R3c's config says what the *config* carries and what a
record may report, and both stay true once the run exists.

The structural half takes about a minute. The run has one measured anchor:
`jw` `k = 1` — 349 settings, the widest of the twenty rungs — took **977 s on
one core** at the full 30+100 replicas, drawn as a timing measurement.

Extrapolating from it needs the two phases separately. Exploration tests all
six endpoints on every cell whatever its crossing, so it costs a fixed
`30 x 87364` shot-units per setting and scales with the setting count alone;
the twenty cells total 2506 settings against that one cell's 349. Confirmation
adds 100 replicas at two endpoints and scales with *where* the crossing lands,
which is what the run is measuring, so it is bounded rather than known: at most
`100 x (16384 + 65536)` per setting, roughly two and a half times what this cell
paid. That puts the whole run near two core-hours and under four — well inside
one session on four workers, and not a `--workers 1` job.

The setting counts are worth stating on their own, because they are the reason
the two banks do not scale together: this bank's binding `W = 1439` is *below*
BeH2's `1814`, but shots are spent per setting and LiH needs 2506 of them
against BeH2's 864, concentrated where BeH2 collapses to a few dozen and LiH
does not (`bk` k=2: 202 against 41). `check_protocol_cost`'s 210-minute
timeout is a CI ceiling for that sweep, not a measurement of it, so it is not
the number to scale from in either direction.

## R3c LiH full-cost run — the result

The run has been executed once, under the frozen stack, and its record and
checker land with it:

```bash
python benchmarks/run_r3c_lih_full_cost.py --workers 4
python benchmarks/check_r3c_lih_full_cost.py --workers 4
```

About 26 minutes on four workers — close to the two core-hours the single-cell
anchor projected.

**Result: the bank prices, and LiH is the exact tier's second priced instance.**
`pricing.status: priced_second_instance`. Thirty-eight of the forty
`(arm, k, estimator)` cells supply a confirmed finite interval; two are
right-censored. All 19,300 confirmatory solves succeeded, so the zero-failure
gate never bit. No cell is open above at the `65536` ceiling and none is open
below the first endpoint, so every one of the thirty-eight intervals has both a
confirmed failing predecessor and a confirmed passing endpoint, which is what
the preregistration requires of a finite one.

| | R3b — the `2+2` probe | R3c — the `30+100` protocol |
|---|---|---|
| cells with a confirmed crossing | 29 / 40 | **38 / 40** |
| `not_bracketed_within_search_grid` | 0 | **0** |
| `exploratory_crossing_not_confirmed` | 11 | **2** |
| modal passing endpoint | `16384`, then `4096` | `16384` (19), then `4096` (16) |

The two records draw the same forty cells on the same bank over the same grid
and differ in the replica counts and the seed roots. **This is an observation,
not a preregistered test.** R3c's gate was the pricing, and its estimand named
the 30+100 protocol as the target instrument rather than as an experiment on
R3b's failure-mode split. What the table shows is that the split R3b labelled
`post_hoc_diagnostic_not_preregistered` pointed the right way: the failures that
survived a fourfold-plus increase in confirmation power are two, not eleven, and
grid fit was never the binding constraint on this bank in either record.

The two survivors are `parity` k=4 and `bk` k=2, both single-assignment, both
`exploratory_crossing_not_confirmed`. They stay right-censored. Censoring is not
infinite cost, and `wider_grid_licensed_by_this_record` is `false`: three cells
crossed at `65536` and confirmed there against a failing predecessor at `16384`,
so the grid held for them too.

Pricing is per card, because admissibility is: `ion-like` 38, `logical-alltoall`
38, `superconducting-like` 23. The fifteen cells the superconducting card drops
carry no runtime on it and are reported as inadmissible rather than folded into
a minimum they cannot attain.

**QR3, re-derived and indeterminate.** The preregistration ties the
re-derivation to a second priced instance, and it now has one, so the record
puts LiH's cells beside the frozen BeH2 cells under one instrument — 88 mapping
cells and 99 instance cells that both banks have admissible and priced. The
result is three-valued and lands in the middle:

- widest mapping spread — `ion-like`, pooled, `k = 4`, the three full-width
  arms — point `17.52×`, bracket `[1.36, 70.08]`;
- narrowest instance spread — `superconducting-like`, single-assignment,
  `k = 2`, `parity` — point `1.02×`, bracket `[1.00, 4.08]`;
- verdict `indeterminate_at_this_shot_grid`.

The point estimates order emphatically — a seventeen-fold mapping effect against
a two-percent instance effect — and the record declines to report that as the
answer, because the brackets overlap and a cost here is never a number but the
interval `(C(confirmed_fail), C(confirmed_pass)]` the geometric grid licenses.
PLAN §6.7's rule is that an overlap is a finding about resolution rather than a
mapping result. So QR3's reason for not answering has moved once more: from
accuracy, to grid resolution, to the `2+2` probe's confirmation power, and now
to the width of the brackets the two effects are compared across. What has
changed is that it is no longer *abstaining* — it is compared, on two priced
instances, and unresolved.

`protocol_cost.json` is read and not rewritten. Its own
`qr3_accuracy_matched: abstains` over `[beh2]` stays a true statement about the
record that carries it; the comparison that needed two instances lives here, and
`check_r3c_lih_full_cost.py` fails a run that reaches back and relabels either
that record or R3b's rejection.

**What the checker gates.** Most of the pricing contract is imported from
`check_protocol_cost` rather than restated, so these cells face the rules BeH2's
faced: smallest confirmed pass with a confirmed failure below it, monotone
confirmation, one retained rank across both deciding endpoints, margins agreeing
with the marginal flag, an interval widened exactly on the sides R1 flagged, and
a `k*` region accounting for every rung once. The rank-stability rule was the
one genuinely at risk — `M = 2` makes the pencil `2x2`, so a deciding panel
could split its retained rank where BeH2's never did — and it holds on all
thirty-eight priced cells.

On top of that the checker requires what is specific to this phase: the record
inherits the preregistration's claim boundary verbatim, which is the one place
in this tree where inheritance is required rather than refused; the config is
still the file that landed, by frozen digest, and still passes the whole
result-free checker; the pricing table and the QR3 statement re-derive from the
drawn cells; the stamped stack is the frozen one; and no probe-only field —
`full_run_authorized`, `screen_prediction`, `eligible_for_full_run` — appears in
a record that prices the run those probes were deciding about.

## R3d QR3 within-grid refinement — preregistration and result

R3d declares the resolution experiment motivated by R3c's overlapping QR3
brackets. Run its structural checker with:

```bash
python benchmarks/check_r3d_preregistration.py
```

The command reads committed files and samples nothing. The result-free config
cryptographically binds both parent cost records, the R3c config and producer,
the shared shot-search and cost implementations, all three device-card files,
and the post-R4a `main` head from which the declaration descends. It freezes the
Python 3.12 / NumPy 2.5.2 / SciPy 1.18.0 / Stim 1.16.0 execution stack, target
30 exploratory plus 100 confirmatory replica blocks, 10,000 bootstrap
replicates, one-sided 95% RMSE upper bound, zero-failure gate, and three fresh
seed roots.

The target coordinates come directly from R3c's two point-estimate supports.
They are a post-R3c choice, which the config labels; every coordinate and
endpoint is nevertheless fixed before any R3d draw:

| support | parent cell | inherited endpoint interval | new endpoints |
| --- | --- | --- | --- |
| mapping | LiH / `jw` / `k=4` / pooled / ion-like | `(16384, 65536]` | `32768` |
| mapping | LiH / `parity` / `k=4` / pooled / ion-like | `(1024, 16384]` | `2048, 8192` |
| mapping | LiH / `bk` / `k=4` / pooled / ion-like | `(1024, 16384]` | `2048, 8192` |
| instance | BeH2 / `parity` / `k=2` / single-assignment / superconducting-like | `(4096, 16384]` | `8192` |
| instance | LiH / `parity` / `k=2` / single-assignment / superconducting-like | `(1024, 4096]` | `2048` |

Those are seven endpoint cells. They are geometric midpoints inside the
already-conservative parent intervals; no endpoint above the existing `65536`
ceiling is declared. Every endpoint is sampled rather than selected by its
exploratory block. Only the confirmatory block can tighten a parent interval:
a non-marginal failure raises its lower endpoint and a non-marginal pass lowers
its upper endpoint. An endpoint whose RMSE upper bound lies within 10% of the
1.6 mHa target is environment-marginal and therefore non-informative for
tightening. The parent rows are never redrawn, pooled with R3d, or reinterpreted.
Any missing endpoint, non-finite statistic, nonmonotone combined sequence, or
invalid interval makes the refinement indeterminate.

The future readout is deliberately one-directional. From the three mapping
intervals, form the conservative mapping-spread lower bound
`max(lower costs) / min(upper costs)`, clamped at one. From the two instance
intervals, form the conservative instance-spread upper bound
`max(upper costs) / min(lower costs)`. Only a strict first-greater-than-second
inequality permits `qr3_mapping_spread_larger_on_preregistered_support`.
Equality, overlap, or a failed refinement reports
`indeterminate_after_refinement`. The targeted point-support design cannot
establish a negative QR3 direction, cannot reselect global extrema after the
draw, and cannot support a mapping-wide claim.

The declaration landed result-free in merge commit `c9fe494`. The producer was
then frozen separately before the one authorized execution. Reproduce the
sampled record and its independent contract audit with:

```bash
python benchmarks/run_r3d_qr3_refinement.py --workers 4
python benchmarks/check_r3d_qr3_refinement.py --workers 4
```

The run reports all seven cells under the frozen stack and leaves both parent
records immutable:

| parent cell | confirmatory midpoint result | refined interval |
| --- | --- | --- |
| LiH / `jw` / `k=4` / pooled | `32768`: UCB `0.819173` mHa, pass | `(16384, 32768]` |
| LiH / `parity` / `k=4` / pooled | `2048`: `2.194747`, fail; `8192`: `0.876929`, pass | `(2048, 8192]` |
| LiH / `bk` / `k=4` / pooled | `2048`: `2.044669`, fail; `8192`: `1.044007`, pass | `(2048, 8192]` |
| BeH2 / `parity` / `k=2` / single-assignment | `8192`: `1.702140`, environment-marginal | unchanged `(4096, 16384]` |
| LiH / `parity` / `k=2` / single-assignment | `2048`: `2.936853`, fail | `(2048, 4096]` |

The mapping-support minimum is `2.1906417x`; the instance-support maximum is
`3.9997993x`. Because the former is not strictly greater than the latter, the
preregistered result is `indeterminate_after_refinement`. That overlap does not
license the negative direction: R3d did not refine all 187 global-extremum
cells and cannot reselect supports after observing the draw.

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
inputs. H₄ (both banks), BeH₂, and Hubbard use the established largest-degree
greedy. H₂O alone uses the scalable full-basis-seeded first-fit cover; its setting counts are
constructive upper bounds and are excluded from the cross-instance QWC verdict.
Neither protocol claims a minimum coloring.

The committed `JW/parity/parity+2q/BK/BK+2q` setting counts are
`913/533/351/615/403` for H₄, `913/533/351/615/405` for H₄-converged,
`353/41/27/41/27` for BeH₂,
`24334/17118/9908/18108/8759` for H₂O, and `1406/798/457/907/478` for Hubbard.
QR2 passes for every arm. Structural QR3 excludes `h4_converged` from the
cross-instance denominator because it is the same physical H4 instance as `h4`
at a second subspace budget; it remains in the record as subspace-robustness
evidence for P5. The corrected ratio-versus-ratio QR3 comparison is negative: mapping spread is not smaller than instance spread for matched-greedy
QWC settings (`13.074×` versus `3.983×`) or mean word weight (`1.571×` versus
`1.489×`). Fixed-shot card rows are derived projections, not independent evidence,
because QWC uses no two-qubit measurement gates. **Accuracy-matched QR3 no longer
abstains at this tier.** `h4_converged` clears the exact subspace-bias floor at
`0.766 mHa` on all five arms, so `qr3.accuracy_matched` reads
`eligible_for_cross_instance_comparison` over `[h4_converged, beh2]` where it read
`insufficient_eligible_instances`. Its prices remain `asymptotic`; this producer
does not replace R1's nonlinear exact-oracle shot search, and at *that* tier the
second instance is deferred for a different reason (see the cost-layer section
above).

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

## R4a contextual-interaction preregistration (result-free)

R4a begins with a declaration, not an experiment:

```bash
python benchmarks/check_r4a_preregistration.py
```

The checker binds the reviewed BeH₂/JW input and R3S margin-stop bank to four
ordered arms (`C0`, `C_CS`, `C_ACASE`, `C_CS+ACASE`). It also freezes the
reference-conditioned contextual-stabilizer selection rule, canonical
coefficient/code tie-break, signs, independence and commutation gates,
fixed-qubit ladder, structural admission thresholds, measurement grid, future
seed namespaces, accuracy/censoring rules, and three device cards.

This command is static. It hashes already-reviewed inputs and validates the API
and declaration; it does not select stabilizers for BeH₂, solve a projected
bank, draw samples, report a cost ratio, or classify QR5. The declaration
authorizes exactly one later structural screen, whose record must land in a
separate commit. It authorizes no sampled execution: every arm must first pass
the frozen `0.533333…` mHa bias-margin and `2048` nonidentity-word gates.

The selection API itself is covered independently:

```bash
python -m pytest tests/test_contextual_restriction.py \
    tests/test_r4a_preregistration.py -q
```

`select_contextual_stabilizers` is NumPy-only. The compilation and projection
tests require the `stim` extra because `compile_contextual_restriction` builds
and verifies the stabilizer tableau. Exact symmetry use continues through the
strict `Restriction.transport`; only the explicitly named contextual projection
may remove Hamiltonian terms, and it reports their fraction of the non-identity
Hamiltonian Hilbert–Schmidt norm so a scalar energy shift cannot dilute the diagnostic.

## R4a contextual structural screen (completed; no sampling)

The one structural execution authorized above is reproduced and gated by:

```bash
python benchmarks/run_r4a_contextual_screen.py
python benchmarks/check_r4a_contextual_screen.py
python -m pytest tests/test_r4a_contextual_screen.py -q
```

The producer walks all seven frozen contextual rungs and reports each of the
four arms in the preregistered order. Every row includes the basis size before
and after scalar-free projected deduplication, annihilated and duplicate source
indices, active qubits, absolute bias against the common dense sector energy,
the non-identity-normalized Hamiltonian removal fraction, non-identity word
universe, and constructive block-commuting setting counts for `k = 1, 2, 4, 8`.
The checker regenerates every value and derives every gate flag and the
largest-passing-rung decision rather than trusting the stored summary.

The screen is negative: no contextual rung admits all four arms. The complete
27-vector full-QSE control has `0.0033015` mHa bias but `14,350` non-identity
words, above the frozen `2,048` operational ceiling. A-CASE alone passes both
gates (`0.0695037` mHa, `1,223` words). The contextual QSE arm compresses to
`997` words at rung 2 and fewer thereafter, but its best bias is `5.3452775`
mHa; the contextual A-CASE arm annihilates both nonidentity bank directions
already at rung 1 and has `5.8994523` mHa bias. Both contextual biases exceed
the preregistered `0.5333333` mHa margin.

Accordingly `selected_contextual_rung` is null and R4a stops without sampling.
The full-QSE ceiling failure is an operational screen rejection, not infinite
cost or proof that the bank cannot be priced. The contextual bias floors are
above the `1.6` mHa accuracy target as well as the stricter margin, but this
structural record still reports no `C(epsilon)`, ratio, `Delta`, or QR5
classification.

## G1 pre-encoding structural preconditioner (numpy only)

`PLAN.md` §3.5 declares five filters that restrict a candidate pool *before* a
fermion-to-qubit encoding is chosen, and §14 records that the section has "no
quantitative content whatsoever" until this record exists. QG1 is the question
that decides whether Track G exists: does pre-encoding algebraic restriction
remove candidates the package's existing post-encoding filters keep?

```bash
python benchmarks/run_g1_structural_preconditioner.py
python benchmarks/check_g1_structural_preconditioner.py
```

Nothing is sampled, no energy is computed, and no chemistry extra is needed. The
whole record is candidate counts and exact actions on a computational
determinant: a Pauli word maps a basis state to one other basis state times a
unit-modulus phase, so the action layer is exact up to the candidates' own
coefficients, with no eigensolve, no projector, and no `2^n` intermediate.
Roughly 70 seconds on one core.

**The deliverable is the marginal in the order applied, per pool — never an
aggregate.** Filters A–D have post-encoding analogues in
`clifford_qc.subspace.symmetry` that the package has applied since Phase 4, so
counting their removals as new content double-counts shipped machinery. The
chain runs `A, B → D → C → E` and each stage records what entered, what
survived, and which labels it removed.

**Two pools, because a marginal is a property of a pool.** The Majorana monomial
pool (Hermitian monomials of degree `0..4` over the `2n` generators) is §3.5's
own, and it is wide enough that A and B have something to remove. The
determinant-excitation pool is the one the mapping and cost records are built
on. Reporting a marginal from the wide pool as though it applied to the narrow
one is the central error available here, so both are measured and every
statement names its pool. The config refuses an excitation pool built with
`conserve_sz` already applied: that would report a vacuous filter as a measured
zero marginal.

**Gate 1 — filters A–D reproduce the existing post-encoding filters.** On every
pool and instance, the survivors of A, B, D and C are exactly the set
`reference_sector_leakage` accepts, with the annihilated-reference exception
counted as a rejection. The checker requires an *empty symmetric difference* in
both directions, not equal counts. On the excitation pool a second, independent
form of the same gate holds: A–D land on exactly the set
`determinant_excitations(conserve_sz=True)` keeps — the pool every mapping and
cost record uses. Disagreement is a bug in one of the two paths and blocks the
phase; it is not a finding.

**Gate 2 — filter E's marginal is reported separately**, because it is the only
filter with no post-encoding analogue. The record carries the number of distinct
Pauli words entering E under the package's own `scalar_free_key` identity, so a
removal cannot be explained as deduplication.

**Filter C runs through the shipped `Restriction`**, not a second implementation
of `P_s A P_s`. `Restriction.transport` moves the Hamiltonian, the reference and
the candidates together, so §3.5's congruence rule holds by construction, and R4's
contextual-subspace comparator shares the one implementation. The declared arm is
`parity+2q`, which fixes two stabilizer qubits; an identity restriction would make
C vacuous by construction rather than by measurement.

**Filter B is reference-conditioned, and the record measures why that matters.**
Global commutation `[A,Q] = 0` is sufficient for sector preservation and not
necessary; the condition A-CASE needs is `(Q − q_target) A ψ_ref = 0`. On the
Majorana pool the global commutant admits `37` candidates against the
reference-conditioned `549`, so a filter B written as the global test would
reject `512` candidates the shipped reference-aware path accepts — failing gate 1
outright — and would foreclose §7.4's excited-state track on the way past. The
`(γ_2p γ_2p+1)² = −1` witnesses are computed per register for the same reason:
§3.5's argument that the real `Cl(2n,0)` branch is vacuous for these stabilizers
rests on that square, so the record carries it as a number rather than a claim.

**The result, on BeH₂, H₄ and Hubbard 2×2 at 8 qubits.** Filters A–D reproduce
the existing accept set exactly on both pools and all three instances — that is
the required agreement, not a finding. Filter E's marginal splits by pool: it
removes `522` of the `549` candidates reaching it on the Majorana pool, over
entrants that are all distinct Pauli words, and it removes **nothing** on the
excitation pool. At the matched degree cap the `27` surviving Majorana classes
reach exactly the determinants `{identity} ∪` the `26` rank-≤2 excitations reach —
same set, no candidate on either side the other misses. So the chain
*reconstructs* the pool the package already builds rather than producing a
different or smaller one.

QG1's falsifier is the conjunction "A–D agree **and** E removes nothing", and it
does not fire: E has content on the pool §3.5 specifies. But the content is
pool-dependent, and the checker re-derives the verdict from the measured fields
so it cannot be written by hand.

**What the record does not license.** No resource claim of any kind, and no
statement that either pool is cheaper to measure. E's marginal on the Majorana
pool removes redundancy the excitation builder never creates, so it may not be
reported as a reduction of the pool the mapping records use. The filters are
pre-encoding in the sense that none of them consults the encoding — each is a
function of the Majorana index data, the declared conserved quantities and the
reference. They are *not* computed in an encoding-free representation: this
package's only arithmetic substrate is the Jordan–Wigner Pauli image, as
`CONVENTIONS.md` states.

**The degree sweep bounds the correspondence.** On BeH₂ the Majorana pool's
action-equivalence class count runs `9 → 27 → 35 → 36` at degree caps
`2, 4, 6, 8`, against the `36`-dimensional `(N=4, S_z=0)` sector. The excitation
reach is matched at degree `4` and nowhere else, and the checker requires exactly
that: a match above the cap would mean the cap is not what makes the comparison
valid, and a mismatch at it would invalidate the comparison. A wider pool reaching
more determinants is the pool being wider, not a filter finding more.

**The character probe records an interaction worth knowing before §7.4.** §3.5B
requires the target character to be a parameter from the first commit. Run under
a declared `(N=4, S_z=1)` character, filter B admits `240` candidates instead of
`549` — the parameter is live at the filter that owns it. But the declared
restriction arm fixes its stabilizer signs from the *reference* sector, so filter
C then annihilates all `240` and the chain returns an empty admissible pool
rather than an error; without a restriction, `12` survive. The character and the
restriction arm are therefore not independent declarations: §7.4's track needs
both moved together, and a G1 that let them drift would close that track at
filter C while §3.5B's requirement at filter B still looked satisfied.
