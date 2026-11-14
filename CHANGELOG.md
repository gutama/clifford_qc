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

### Fixed

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
