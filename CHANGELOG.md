# Changelog

Notable changes to the `clifford_qc` package, newest first.

**Identifying a version.** The package is alpha research software installed
from source; it is not published on a package index. `__version__` names the
interface generation and is stamped into benchmark records, so it does not
change between commits. The revision that reproduces a result is the Git
commit: record `git rev-parse HEAD` with any experiment, as the README asks.
Benchmark campaigns, their declarations and their outcomes are recorded in
[PLAN.md](PLAN.md) and [REPRODUCING.md](REPRODUCING.md), not here.

## Unreleased

### Added

- **Structure-of-arrays product kernel in the exact bank (Rank 5b).**
  `MatrixElementBank(..., kernel="soa")`, `StreamingMatrixElementBank` (which
  inherits it), `projected_matrices(..., kernel="soa")` and
  `solve_subspace(..., kernel="soa")` compute `H A_j`, `A_i' A_j` and
  `A_i' H A_j` with `clifford_qc.soa_kernel.soa_mul` instead of the
  dictionary product. The kernel batches XOR/popcount over packed code arrays,
  merges by sorting, and never reads or fills the process-wide word memo.
  It produces the same words and coefficients (tested to `1e-12`; equal to
  the last bit on the bundled molecular pools), the same word ledger, and the
  same A-CASE selections. Rows are inserted in ascending code order, so a
  matrix entry may differ from the `mv` route in its last few ulps.
  `kernel="mv"` stays the default and the only route committed records
  replay through; `kernel="soa"` is a speed option for new runs and reports
  `product_kernel` in `projected_matrices` resources.
  `PauliSoA.to_mv()` now enforces `MV`'s invariants on whole arrays and adopts
  the dictionary directly, about 5x faster than re-validating each term.
  `benchmarks/run_soa_bank_throughput.py` measures pair throughput on the
  frozen Phase 2M pools and one full A-CASE solve per kernel in fresh
  interpreters, and checks the selected labels and energy against
  `bank_policy_matrix.json`; its result is
  `benchmarks/timing_results/soa_bank_throughput.json`, kept outside
  `reference_results/` because it is a host-specific timing record, not a
  replayable one.

- **Bootstrap interval at the acquisition stop**
  ([docs/restart/BOOTSTRAP_STOP.md](docs/restart/BOOTSTRAP_STOP.md)).
  `acquire_ritz(bootstrap_replicates=..., bootstrap_seed=...)` resamples the
  data at the stopping look by setting, re-solves, and reports
  `bootstrap_interval`: the estimate plus or minus the delta interval's `z`
  times the bootstrap spread `(q84 - q16) / 2`. The stop, the shots and every
  other reported value are unchanged; a failed replica is recorded, and the
  option now also supports certified control variates: each replica refits the
  correction at its own Ritz vector and the interval is centred on the
  corrected estimate. The combined policy has not been tested for empirical
  coverage. `bootstrap_spread` and
  `bootstrap_spread_interval` build it from existing draws. On fresh seeds and
  under criteria fixed before the run it covered in 99% to 100% of runs on the
  held-out six-qubit bank, where the delta interval covered 84.5% to 93.5%.
  `benchmarks/run_bootstrap_stop.py` and `check_bootstrap_stop.py` reproduce
  and verify the records.

- **Symmetry-shift control variates**
  ([docs/restart/SYMMETRY_SHIFTS.md](docs/restart/SYMMETRY_SHIFTS.md)).
  `clifford_qc.subspace.symmetry_shifts` builds BLISS-style families
  `K = (Q - q)O` that annihilate every state of a bank whose generators
  conserve particle number and `S_z`, and certifies them before use. A shift
  is fitted to the grouped Neyman shot variance of the Ritz functional or to a
  state-free coefficient proxy, or applied after the fact as a control variate
  read from an existing cache (`ritz_control_variate`), with no new shot.
  `acquire_ritz(control_variates=...)` reports, stops on and allocates to the
  corrected Ritz value at every look. A family that fails certification is
  refused before any shot. Defaults and frozen records are unchanged; the
  recorded bytes of `acquisition.py` are packaged so the target-acquisition
  replay stays on. `benchmarks/explore_control_variates.py` reproduces the
  exploratory runs.

- **Symmetry-adapted excitation pools**
  ([docs/restart/SYMMETRY_POOLS.md](docs/restart/SYMMETRY_POOLS.md)).
  `clifford_qc.subspace.symmetry_pools` adds native spin-free closed-shell
  singles/doubles and whole-operator filtering by explicit real-Abelian
  point-group characters, with an optional Hamiltonian-invariance check and
  an explicit `hamiltonian_atol` for integral noise. Both APIs take an
  explicit interleaved or blocked `spin_ordering`, and
  `closed_shell_occupied_spatial` reads the doubly occupied orbitals from a
  determinant rather than assuming an aufbau prefix.
  A bounded exact molecular-bank audit counts initial screening and complete
  candidate-bank resources separately, with per-stage timings. Fewer
  generators do not by themselves imply fewer measurement words or shots.

- **Hierarchical commutator subspaces and resource-aware information acquisition**
  ([benchmarks/GEOMETRIC_INFORMATION_STUDY.md](benchmarks/GEOMETRIC_INFORMATION_STUDY.md)).
  `clifford_qc.subspace` adds two pure-NumPy research modules (re-exported in
  `clifford_qc.subspace.__all__`):
  - `clifford_qc.subspace.commutators`: relative-tolerance commutator builders
    (`commutator_mv`, `commutator_apply`), hierarchical Lie-algebraic closure
    and span diagnostics (`nested_commutator_hierarchy`,
    `hierarchy_span_diagnostics`), and demand-driven expansion
    (`CommutatorFrontier`, `run_dynamic_commutator_acase`);
  - `clifford_qc.subspace.information`: incremental QWC, `DeviceCard`, and
    classical cost accounting (`IncrementalCostModel`,
    `estimate_incremental_cost`), 7D finite-shot delta-method lowering
    propagation and cost-ratio selection (`expected_lowering_estimate`,
    `score_information_candidate`, `select_information_candidate`,
    `run_resource_aware_acase`), and oracle-free closed-loop acquisition to a
    target precision (`acquire_to_accuracy`).
  Defaults for existing solvers and all frozen baselines remain unchanged.

- **Opt-in estimators for measured projected pencils**
  ([docs/restart/PENCIL_ESTIMATORS.md](docs/restart/PENCIL_ESTIMATORS.md)).
  `SharedMeasurement.solve` gains three opt-in alternatives to plain
  thresholding:
  - nearest-correlation overlap repair (`overlap_repair`);
  - fixed-angle pencil rotation (`pencil_angle`);
  - a covariance-aware constrained ratio (`ratio_error_budget`).

  The constrained ratio minimizes `c* H c / c* S c` under a propagated
  standard-error budget and needs SciPy. It is built on two new pieces:
  `SharedMeasurement.entry_covariance`, the joint covariance of every measured
  `S` and `H` coordinate, and `measurement.functionals.covariance_matrix`,
  which computes a covariance matrix in one histogram pass.

  `subspace.coverage` measures empirical interval coverage over repeated
  measurements (`interval_replica`, `summarize_coverage`, `wilson_interval`),
  and `subspace.uncertainty.bootstrap_ritz_samples` exposes the bootstrap
  draws behind `bootstrap_ritz`, whose results are unchanged.

  `subspace.acquisition.acquire_ritz` measures a uniform pilot and then spends
  predeclared rounds uniformly or by Neyman allocation on the current Ritz
  functional (`neyman_group_plan`). A stopping target is evaluated only at the
  predeclared looks, and the interval reported there splits its error budget
  over every look; the single-look interval is returned beside it. A look that
  cannot be solved raises `AcquisitionError`, which keeps the shots spent.

  Each estimator records its policy, so `bootstrap_ritz(..., result=point)`
  reruns it on resampled histograms. Defaults are unchanged, and no estimator
  claims interval coverage or improved accuracy.

### Fixed

- Information acquisition now includes the measured Ritz-vector response in
  candidate uncertainties, refreshes the solve after screening measurements,
  preserves unresolved remaining bias at basis limits, and checks shot and
  execution-time budgets before every batch. Commutator hierarchies no longer
  report Lie closure when caps or filters leave branches uncomputed.
  Review follow-ups: a commutator erased only by `MV`'s absolute cutoff is
  reported as `"underflow"` and never counts toward closure; truncation caps
  seed support too; seeds already in the basis stay expandable; refinement
  allocates to every word a setting reads; the lowering threshold applies
  before the cost-ratio argmax; both resource-aware runners end with the
  retained-subspace sector certificate (`retained_sector_certificate`); a
  run stopped by hardware time says so; and a word an existing setting
  already reads adds no setting cost.
- The constrained-ratio single-live-generator shortcut now rejects an
  unattainable error budget or denominator-CV gate. Deterministic phase-mixed
  starts also prevent the reviewed stationary two-direction maximum from
  being returned as convergence; the bounded multistart policy is recorded
  in the result and rerun by the grouped bootstrap.
- **Python 3.14 stored different coefficient bytes.** Python 3.14 changed mixed
  real/complex arithmetic so that a `-0.0` component survives `0.0 + z`. The
  kernel accumulated from a real zero, so `(1j*Y)*X` stored `1-0j` on 3.14 and
  `1+0j` on every earlier interpreter: equal as numbers, different as bytes,
  and enough to fail ten record-replay tests that compare `float.hex`. `MV`
  now accumulates from a complex zero and never stores a negative zero, on any
  interpreter; results on Python 3.13 and earlier are bit-for-bit unchanged.
  The independent MORE oracle in `benchmarks/` has the same fix.
- `run_acase(rho, hamiltonian, ..., bank=bank)` ignored `rho` and `hamiltonian`
  when a bank was supplied, so a bank built for one Hamiltonian returned that
  Hamiltonian's energy whichever one was passed with it. A bank that does not
  match the reference and Hamiltonian passed beside it is now a `ValueError`.
- Install hints named a distribution that does not exist. A missing extra used
  to suggest `pip install 'clifford-qc[extra]'`; the hint is now
  `pip install -e '.[extra]'` in a source checkout and an install from the
  repository URL elsewhere.
- `Model` defined `__post_init__` twice. The copies were identical, so
  behaviour is unchanged.
- The documentation gate crashed without SciPy: probing for a guarded
  submodule such as `scipy.sparse` raised instead of reporting it absent.

### Changed

- `clifford_qc.workflows.__all__` now lists the unified workflow interface
  (`WorkflowResult`, `run_workflow`, `export_workflow_evidence`,
  `verify_workflow_evidence`) beside `adapt_warm_start`, so
  `from clifford_qc.workflows import *` binds those names too.
- Only `clifford_qc` is installed. The repository's `molecular/` comparison
  suite was also packaged, as a top-level package named `molecular`; it is run
  from a checkout (`python -m molecular.run`) and is no longer installed.
- The source distribution no longer ships the test suite, which cannot run
  without the repository's benchmark records and Git history.
- The test suite no longer depends on the machine it runs on:
  - it collects and passes with NumPy alone, with SciPy but not Stim, and with
    every extra; a test whose optional dependency is absent skips and names it;
  - tests that walk commit history skip, with the reason, in a shallow clone or
    a source archive instead of failing inside `git`;
  - the one test that asserts the exact stack a frozen record named skips
    elsewhere, and the comparison it guards is now tested on every interpreter;
  - a process-table race in one test is gone.

  The record-environment CI job sets `CLIFFORD_QC_STRICT_TEST_ENVIRONMENT=1`,
  under which those environment skips are failures.
- CI tests what the package advertises: every Python version in the
  classifiers (3.10–3.14), a NumPy-only install, and an install with every
  bridge and chemistry extra, in addition to the pinned record environment.
- Lint runs all of Pyflakes instead of four rule groups, and the package is
  type-checked with mypy on every pull request. Two modules are exempt, by
  name and with the reason, in `pyproject.toml`: frozen records hash them
  byte-for-byte, and their findings are cosmetic.
- Two checks that hash source files against a frozen record
  (`check_noise_aware_selection.py` and `run_observable_noise.py --check`)
  accept a file that has changed since the experiment only if its
  execution-time bytes are packaged under
  `benchmarks/provenance/execution_sources/`, and report it as `EVOLVED`.

### Added

- `clifford_qc.kernel_cache_info()` and `clear_kernel_caches()` report and
  empty the process-wide Pauli-product and blade-mask memos, which no bank's
  resource ledger includes. `CLIFFORD_QC_KERNEL_CACHE_SIZE` resizes or disables
  them before import. The default is unchanged.
- [CONVENTIONS.md](CONVENTIONS.md#coefficient-storage-and-equality) documents
  the coefficient store: absolute pruning at `1e-12`, no negative zeros, and
  `==` as identity.
