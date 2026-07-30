# Reproducing the benchmark artifacts

Every file under `benchmarks/reference_results/` regenerates from a clean
environment with the commands below. All experiments are seeded; JSONL
rows should match up to floating-point noise in wall-clock fields, and the
summary tables should match exactly.

**Records and code move together.** A record produced before a change to the
estimator, the selector, or the confidence construction is not comparable with
one produced after, and mixing the two is how stale numbers reach a manuscript.
When any module under `clifford_qc/measurement/` or `clifford_qc/algorithms/`
changes, regenerate every record that depends on it. The manuscript's figures
and tables are then emitted mechanically from those records:

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
pytest                                      # 740 passed, 6 skipped
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

`check_docs.py` verifies the documented pair by collection. It reports a
skip when the installed extras do not match the environment above; pass
`--require-test-count` to turn that mismatch into a failure, which is how
CI enforces it in the job that owns the contract.

The core package imports with numpy alone; without SciPy the optimizer
falls back to pure-Python Adam (numerically equivalent results at looser
tolerance — the committed artifacts were produced with SciPy's L-BFGS-B).

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
    -> Jordan-Wigner Hamiltonian -> A-CASE
    -> energy + state coefficients + correlations + Lehmann response
```

Expected invariants (minor last-digit formatting may vary):

- sector exact and adaptive A-CASE energies both `-0.828427125 eV`;
- absolute energy mismatch below `1e-12 eV`;
- complete response basis `M=4`, rank `4`, `kappa(S)=1`;
- singlet diagnostic `<S^2>` below `1e-12`;
- staggered-spin line at `0.828427125 eV` with weight `0.853553391`.

The program accepts another record as its sole argument. The schema is
`clifford_qc.effective_hamiltonian.v1`; spin orbitals are interleaved
(`2*site=up`, `2*site+1=down`), and complex one-body entries are `[real, imag]`.

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
- adaptive A-CASE at 8 additions: `M=9`, `W=7371`,
  error `3.018781 mHa`;
- complete singles/doubles coordinate space: `M=27`,
  error `0.765862 mHa` (chemical accuracy).

The adapter itself needs no chemistry extra. Full CI additionally compares its
185 Pauli coefficients against the independent OpenFermion/PySCF construction.
FCIDUMP orbital signs are a gauge; the committed digest fixes one gauge rather
than weakening coefficient tolerances.

## Finite-shot nonlinear response uncertainty

```bash
python examples/acase_finite_shot_response.py
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

## A-CASE validation ladder (Phase 7, `chemistry` extra)

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
  sector**, not against the global ground state. A-CASE never leaves the
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
  every candidate and A-CASE reports a basis of size one at the reference
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

Certified A-CASE runs on the eight-qubit rungs as `acase_certified_n8`, at
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
