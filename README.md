# clifford_qc

**Quantum simulation with sparse Pauli operators and adaptive eigensolvers.**
`clifford_qc` is a Python research toolkit for building spin, lattice, and
molecular models; comparing VQE, ADAPT-VQE, and subspace methods; and accounting
for their measurement and classical storage costs.

Its main subspace method, **A-CASE** (Adaptive Clifford-Algebra Subspace
Eigensolver), grows virtual directions around a reference state and solves a
small generalized eigenproblem. Shared operator calculations and measurement
caches let researchers compare methods using the same Hamiltonian and data.

**Python 3.10+ · NumPy core · Apache-2.0 · Alpha research software.**
Install from source; interfaces may change. The committed studies support
small-system comparisons, and a public archival release is intended.

[Install](#install-from-source) · [Quick start](#quick-start) ·
[Choose a solver](#choosing-a-solver) · [Molecular pipeline](molecular/PIPELINE.md) ·
[Architecture](ARCHITECTURE.md) · [Reproduce results](REPRODUCING.md)

## What you can do

| Task | Implemented features | Start here |
|---|---|---|
| Work with operators | Sparse Pauli and fermionic operators, gates, density operators, channels, traces, and entanglement diagnostics | [Operator example](#operator-calculations), [conventions](CONVENTIONS.md) |
| Build models | Spin chains, Hubbard and other lattice models, restricted real FCIDUMP import, effective-Hamiltonian JSON, orbital rotations — every model declares its encoding and sector under a validated metadata contract | [Model examples](#examples), [conventions](CONVENTIONS.md#model-metadata), `clifford_qc.models` |
| Run circuits | Parameterized Pauli rotors and Clifford gates, JSON serialization, gradients, OpenQASM 3 export | [Circuit example](#circuit-programs), `clifford_qc.ir` |
| Compare eigensolvers | VQE, ADAPT-VQE, fixed and adaptive operator-response subspaces, QSCI/SQD (with reference, ADAPT-VQE, and exact or Trotter time-evolved sampling inputs), selected-CI controls, hybrid bases | [Solver guide](#choosing-a-solver), [subspace example](#adaptive-subspaces) |
| Check a Ritz state against an eigenstate | True Ritz residual norms and energy variances from the second-moment block `⟨A_i†H²A_j⟩`, with automatic coefficient-storage admission for each requested block; exact pairings over a matrix-element bank, validated on the five banks the Phase 15 preflight licensed. No finite-shot estimator is built: measuring the block reads `NONE` under both QWC and fully commuting grouping | `clifford_qc.subspace.SecondMomentBank`, [example](#residuals-and-folded-roots), [conventions](CONVENTIONS.md#second-moments-residuals-and-folded-roots), [reproducing](REPRODUCING.md#phase-15-secondmomentbank-and-its-validation) |
| Find and bound interior roots | Exact folded-spectrum roots of `K − 2ωH + ω²S`, refined Ritz vectors, target and Weinstein intervals that each contain an eigenvalue, and monotone variance minimization; validated on the same five banks | `clifford_qc.subspace.FoldedSpectrum`, [example](#residuals-and-folded-roots), [reproducing](REPRODUCING.md#phase-15-folded-spectrum-roots-and-their-validation) |
| Count eigenvalues near Ritz roots | Disjoint cluster hulls from the exact block residual `Q'KQ − Θ²`, each holding at least as many eigenvalues as roots, so a degenerate level is counted with its multiplicity; held to dense spectra in the test suite | `clifford_qc.subspace.ritz_clusters`, [example](#residuals-and-folded-roots), [conventions](CONVENTIONS.md#second-moments-residuals-and-folded-roots) |
| Assess why adaptive growth stopped | Opt-in exact convergence reports over the final retained block, with numerical-resolution and ground-state evidence reported separately; Q17 extrapolation is a diagnostic only | `ConvergenceConfig`, `convergence_report`, [example](examples/acase_convergence.py), [contract](REPRODUCING.md#phase-15-exact-convergence-reporting) |
| Solve embedding fragments | One callback returning energy plus one- and two-particle density matrices, implemented by sector FCI, QSCI, selected CI, and the QSCI × A-CASE hybrid; the embedding loop itself stays outside the package | `clifford_qc.subspace.fragment`, [conventions](CONVENTIONS.md#density-matrices) |
| Reuse measurements | QWC and block-commuting grouping, compiled Clifford readouts, shared caches, covariance, allocation, uncertainty estimates | [Finite-shot examples](#examples), [architecture](ARCHITECTURE.md#measurement-and-evidence) |
| Control classical costs | Cached preparation, optional reference validation, object or packed coefficients, streaming with recomputation | [Pipeline guide](molecular/PIPELINE.md) |
| Use other quantum software | Optional Stim, OpenFermion, pytket, PennyLane, and PyZX bridges | [Optional dependencies](#optional-dependencies) |

The core representation is a sparse operator `MV` in the Pauli-word basis of

\[
\mathrm{Cl}(2n,\mathbb C) \cong M(2^n,\mathbb C).
\]

Models and circuit programs expose `PauliSum`; conversion to `MV` preserves
word codes and operator semantics. Sparse support can grow rapidly, so this
representation is not a general speedup over matrix methods.

## Install from source

Clone the repository, then install the package into your Python environment:

```bash
git clone https://github.com/gutama/clifford_qc.git
cd clifford_qc
python -m pip install -e .
python -m clifford_qc.verify
```

NumPy is the only core runtime dependency. The smoke check validates the
installation; it does not reproduce the full benchmark suite. If you already
have a checkout, run the last two commands from its root. For reproducible
experiments, record `git rev-parse HEAD` and the dependency environment.

### Optional dependencies

Install only the extras needed for your workflow:

| Extra | Purpose |
|---|---|
| `research` | SciPy reference solvers and research workflows |
| `chemistry` | PySCF/OpenFermion molecular input generation; reading supported FCIDUMP files needs only the core |
| `stim` | Stabilizer execution and compiled Clifford measurement/restriction paths |
| `openfermion`, `pytket`, `pennylane`, `pyzx` | Individual ecosystem bridges |
| `bridges` | All bridge dependencies, including Stim |
| `test` | Pytest, property-based checks, coverage tools, and lint |
| `release` | Distribution build and metadata-check tools |

For example:

```bash
python -m pip install -e '.[research,stim]'
```

Bridge modules are imported explicitly from `clifford_qc.bridges`. Each has a
supported operation subset; installing a bridge does not make arbitrary
circuits portable between frameworks.

To see which capabilities an installation has — without importing any of them,
so the check is fast and side-effect free:

```bash
python -m clifford_qc.capabilities
```

```
  [yes] sparse_linalg                  extra=research
  [no ] pyzx_bridge                    extra=pyzx  (pip install -e '.[pyzx]')
```

`--require NAME` exits non-zero when a named capability is absent, so a CI job
or a script can state the environment it expects instead of discovering the
answer through a failure further in. `from clifford_qc import capabilities;
capabilities.capabilities()` returns the same report as a dict, and `python -m clifford_qc.verify` prints it
alongside the smoke check. Every entry names the extra that supplies it, and so
does the `ImportError` raised by a feature whose extra is missing.

## Quick start

### Operator calculations

Prepare a Bell state and evaluate its correlations:

```python
from clifford_qc import CNOT, H, X, Z, evolve, expectation, ket_density

rho = evolve(ket_density(2, "00"), CNOT(2, 0, 1) * H(2, 0))
print(expectation(rho, X(2, 0) * X(2, 1)).real)  # 1.0
print(expectation(rho, Z(2, 0) * Z(2, 1)).real)  # 1.0
```

Qubit zero is the leftmost character in a Pauli label. Multiplication follows
matrix order: the rightmost gate acts first. See [CONVENTIONS.md](CONVENTIONS.md)
for word codes, scalar pairings, trace normalization, and rotor signs.

### Circuit programs

Build a parameterized program, evaluate an observable, and export its gates:

```python
from clifford_qc import Parameter, PauliSum, Program, adjoint_gradient, to_qasm3

theta = Parameter("theta")
program = (
    Program(2, parameters=[theta])
    .clifford("H", 0)
    .clifford("CX", 0, 1)
    .rotor("ZI", theta)
    .measure_expectation({"XX": 1.0})
    .measure_z(0, 1)
)
observable = PauliSum.from_labels({"XX": 1.0})
print(program.run([0.4]))
print(adjoint_gradient(program, observable, [0.4]))
print(to_qasm3(program, [0.4]))
```

OpenQASM 3 is an export format. Expectation tasks are evaluated by the package;
computational-basis sampling tasks become QASM measurement statements.

### Adaptive subspaces

Grow a small operator-response basis for a transverse-field Ising model:

```python
from clifford_qc import PauliWord
from clifford_qc.backends import ExactMVBackend
from clifford_qc.models import tfim
from clifford_qc.subspace import pauli_orbit, run_acase

model = tfim(4, J=1.0, h=1.0)
rho = ExactMVBackend().state(model.reference, ())
words = [PauliWord.from_label(label) for label in ("ZZII", "IZZI", "IIZZ")]
result = run_acase(rho, model.hamiltonian, pauli_orbit(words), max_size=3)
print(result.energy)
print(result.stopped_reason)
print(result.resources)
```

A-CASE forms virtual directions `A_i |psi>` and solves `Hc = ESc`.
Every projected matrix element is an expectation on the reference; the basis
states need not be prepared separately.
The example uses a deliberately small candidate pool and no ground-state oracle.
`max_size=3` allows up to three additions beyond the initial identity direction.
This example uses exact classical expectations; finite-shot workflows have
separate examples below. An exact projected solve does not certify convergence
to the full ground state.

For a bounded example of full residual reporting, run
`python examples/acase_convergence.py`. Pass
`convergence=ConvergenceConfig(residual_tolerance=...)` to `run_acase`, or
call `convergence_report` on an existing bank solve. The returned
`convergence_reports` keep the growth stopping reason beside each tracked
root's true residual, numerical-resolution status and optional ground-state
assessment. Reporting is off by default and automatically preflights only the
retained `H²` block before construction, with a default budget of 96,107,619
sparse coefficient occurrences. `ConvergenceConfig(storage_policy=...)` accepts
an explicit finite budget. This bounds coefficients rather than peak memory;
packed or streaming `(S, H)` storage does not pack or evict the `H²` rows.
See the [storage-gate contract](REPRODUCING.md#phase-15-runtime-storage-gate).
A supplied `GroundStateReference`
must describe an independently established spectrum in the state's space.
Neither a small residual nor a Q17 extrapolated energy alone establishes
ground-state convergence. See the [reporting contract](REPRODUCING.md#phase-15-exact-convergence-reporting).

For repeated molecular experiments, [molecular/PIPELINE.md](molecular/PIPELINE.md) separates
FCIDUMP preparation, independent A-CASE solves, and optional sector validation.
Object storage with `retain_all` is the default. Packed storage and
`stream_recompute` are explicit alternatives with different memory/time costs.

The complete [molecular simulation suite](molecular/README.md) lives in
`molecular/`, with its catalog, cached PySCF preparation, isolated per-molecule
workers, resumable runs, historical results, and reports. Install
`python -m pip install -e '.[molecular]'` and start with
`python -m molecular.run --molecules lih --max-additions 2 --complete-sd off`.
New runs write to `molecular/runs/`; committed evidence remains in `molecular/results/`.

### Residuals and folded roots

A projected solve has zero projected residual by construction. The true
residual of a Ritz state needs the second-moment block `K = ⟨A_i†H²A_j⟩`,
which `SecondMomentBank` pairs exactly beside `S` and `H`. Continuing the
transverse-field Ising example:

```python
from clifford_qc.subspace import (
    FoldedSpectrum, MatrixElementBank, SecondMomentBank, identity_generator,
    ritz_clusters)

bank = MatrixElementBank(rho, model.hamiltonian,
                         [identity_generator(model.n), *pauli_orbit(words)])
moments = SecondMomentBank(bank)
ritz = bank.solve()
print(moments.residual(ritz).residual_norm)          # true residual of the Ritz root
root = FoldedSpectrum(moments).refined_ritz(ritz.energies[0])
print(root.residual_norm, root.weinstein_interval())  # an eigenvalue lies inside
for cluster in ritz_clusters(ritz, moments=moments).clusters:
    print(cluster.roots, cluster.interval())         # >= len(roots) eigenvalues inside
```

`FoldedSpectrum.roots(w)` returns the states of the span closest, by residual,
to an eigenvalue near a target `w`.
`refined_ritz` sets `w` to a Ritz value, and its residual never exceeds the
Ritz vector's. `minimize_variance` iterates `w ← E` to a variance-stationary
state. Each root carries a target interval and a Weinstein interval, and an
eigenvalue of `H` lies in each. Neither says which eigenvalue, and overlapping
intervals may share one. `ritz_clusters(ritz)` counts instead: it merges roots
until their hulls are disjoint, and each hull holds at least as many
eigenvalues as roots, so a degenerate level is counted with its multiplicity.
Variances are
reported signed and count as resolved only above a rounding floor, as
[CONVENTIONS.md](CONVENTIONS.md#second-moments-residuals-and-folded-roots)
defines. These are exact pairings. Q18-S1 and Q18-S4 found that, under uniform
allocation, measuring the residual to the energy's standard error costs more
than ten times the energy's shots on every bank tested, so no finite-shot
estimator is built.

For a bounded example of convergence reporting, run
`python examples/acase_convergence.py`. Pass
`convergence=ConvergenceConfig(residual_tolerance=...)` to `run_acase`, or
call `convergence_report` on an existing bank solve. The returned
`convergence_reports` keep the growth stopping reason beside each tracked
root's true residual, numerical-resolution status and optional ground-state
assessment. Reporting is off by default and adds the retained `H²` block;
larger banks need a storage preflight. A supplied `GroundStateReference`
must describe an independently established spectrum in the state's space.
Neither a small residual nor a Q17 extrapolated energy alone establishes
ground-state convergence. See the [reporting contract](REPRODUCING.md#phase-15-exact-convergence-reporting).

## Choosing a solver

For a small exact comparison of generalized qubit-excitation and CEO pools,
`clifford_qc.algorithms.exchange` provides `exchange_pools` and
`run_exchange_adapt` with QE, OVP-only and mixed-CEO arms. It stores dense
matrices in a declared particle-number/spin sector and records every optimizer
call. This is an exact comparison path with unmeasured quantum-shot costs;
the [Phase 17 declaration](benchmarks/PHASE17_CEO_DECLARATION.md) states its
scope and the separate generic Pauli-ladder resource upper bounds.

For an exact multi-root calculation, `clifford_qc.algorithms.run_more_adapt`
applies one QE unitary to several orthogonal occupation references. It minimizes
their mean energy and returns sorted Rayleigh--Ritz roots, with optimizer traces
and separate resource ledgers. Install `.[research]`. For example:

```python
from clifford_qc.algorithms import run_more_adapt
from clifford_qc.models import hubbard

result = run_more_adapt(hubbard(2), [3, 6], max_iterations=4, max_parameters=4)
print(result["root_energies"])
```

Occupation integers encode computational-basis states in the declared N/Sz
sector. The [MORE breadth declaration](benchmarks/PHASE17_MORE_DECLARATION.md)
defines the separate comparison with native mean-root A-CASE; reproduction and
execution provenance are documented in [REPRODUCING.md](REPRODUCING.md).

| Workflow | What changes during the calculation | Main tradeoff |
|---|---|---|
| VQE | Parameters of a fixed circuit | Repeated objective and gradient evaluations |
| ADAPT-VQE | Circuit generators and their optimized parameters | Pool screening, optimization, and growing circuit depth |
| A-CASE | A basis of virtual operator-response directions | Projected matrix construction, overlap conditioning, and subspace bias |
| QSCI/SQD | Determinants retained from computational-basis samples | Sampling coverage and classical projected diagonalization |
| Selected CI | Determinants selected classically | A classical control for separating selection effects from quantum sampling |

Use A-CASE when you want to expand around an existing reference without
preparing a separate circuit for every basis direction. An ADAPT-VQE warm
start can supply that reference through `workflows.adapt_warm_start`.
Compare total selection, measurement, optimization, and classical costs at a
matched accuracy target; no method is uniformly preferable.

The [H2O singlet comparison](benchmarks/results/h2o_residual_comparison.md)
tests refreshed residual packets, spin-adapted selected CI and active-space
enlargement with cached H actions and isolated-process cost records. Packets
reach the fixed 1.6 mHa target with 12 retained columns versus selected CI's
14, but use 213 H actions versus 105. This is small-sector classical evidence,
not a quantum or total-cost advantage.

The [packet work comparison](benchmarks/results/packet_work_reuse.md) adds exact
fixed-frame action reuse and heuristic residual shortlisting as experimental
classical paths. On the same H2O fixture, reuse preserves the complete original
trajectory and reduces target H actions from 213 to 57. Shortlisting reduces
exact scores from 210 to 63; the combined arm also uses 57 H actions and 12
columns. Caching adds owned frame/action arrays, and tiny-sector time differences
do not establish scaling or quantum advantage. See
`clifford_qc.subspace.packet_selection` and the [reproduction instructions](REPRODUCING.md#residual-packet-action-reuse-and-shortlisting).

The [prepared-input CLI](molecular/PIPELINE.md) currently exposes exact A-CASE with
determinant-excitation candidates. Other methods use Python APIs and examples.
Its `prepare`, `solve`, and optional `validate` commands have separate outputs,
so repeated solves can reuse one prepared input without repeating a reference
eigensolve.

## Examples

After installation, run these scripts from the repository root:

| Example | Demonstrates |
|---|---|
| [bell_chsh.py](examples/bell_chsh.py) | Bell-state correlations and diagnostics |
| [grover_2q.py](examples/grover_2q.py) | A small circuit expressed as operators |
| [fermion_car.py](examples/fermion_car.py) | Fermionic anticommutation relations |
| [noisy_channel.py](examples/noisy_channel.py) | Density operators and Kraus channels |
| [tfim_exact.py](examples/tfim_exact.py) | A small exact spin-model reference |
| [acase_adaptive.py](examples/acase_adaptive.py) | Adaptive growth and fixed-basis comparisons |
| [acase_effective_model.py](examples/acase_effective_model.py) | Effective-Hamiltonian import, correlations, and response |
| [acase_finite_shot.py](examples/acase_finite_shot.py) | Shared measurements and finite-shot subspaces |
| [acase_finite_shot_response.py](examples/acase_finite_shot_response.py) | Response bootstrap and its heuristic intervals |
| [acase_materials.py](examples/acase_materials.py) | Lattice models, projected observables, and excited states |
| [acase_sector_backend.py](examples/acase_sector_backend.py) | Sector statevectors and matrix-free reference calculations |
| [acase_premise_check.py](examples/acase_premise_check.py) | Subspace identities and resource accounting |

```bash
python examples/acase_effective_model.py
```

The bundled Wannier-Hubbard dimer input is synthetic. It demonstrates an
upstream effective-model interface, not a completed DFT/Wannier calculation.
The loader labels energy units but does not convert them. Some examples need
optional extras; benchmark-specific environments are in [REPRODUCING.md](REPRODUCING.md).

## Interpreting results

- **Evidence:** exact, asymptotic, finite-sample, and heuristic results have
  different meanings. A fixed-functional certificate does not automatically
  certify a nonlinear Ritz eigenvalue or total ground-state error. A new
  benchmark record must declare a tier and a scope for each quantity under the
  versioned `clifford_qc.evidence.v1` contract. Older records predate it and
  are frozen by digest rather than relabelled, so labels are not yet uniform
  across all records.
- **Residuals:** a small true residual places some eigenvalue nearby; it does
  not say which one. Ground-state claims need a separately established
  spectrum. A variance at the rounding floor is reported as unresolved, not as
  converged. Every true Ritz residual the package reports is an exact pairing,
  not a finite-shot estimate.
- **Memory:** streaming bounds persistent overlap/Hamiltonian coefficient rows,
  not total process memory. Histories, references, generators, and observables
  have separate lifetimes; recomputation can increase runtime.
- **Cost:** fewer Pauli words or settings need not mean fewer shots or less
  device time. Device cards are illustrative accounting models, not hardware
  measurements. QSCI's classically built projected matrix still incurs sampling
  and classical computation costs.
- **Scale:** the committed studies support small-system method comparisons.
  They establish neither general scaling nor an advantage over other packages.

## Validation

For development tests:

```bash
python -m pip install -e '.[test,research,stim]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest
```

Optional bridge tests require their corresponding extras. Replaying frozen
scientific records additionally requires the dependency versions documented in
[REPRODUCING.md](REPRODUCING.md); an arbitrary current installation is not that
environment.

Tests cover algebra, dense-reference equivalence, circuit serialization and
lowering, gradients, solvers, storage policies, and installed bridges. Benchmark
checks have different roles: some rebuild numerical records, while others check
configuration, provenance, or documentation. Not every record has a numerical
rebuild gate, and expensive sampled gates require manual CI dispatch.

## Documentation and papers

| Document | Purpose |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layers, interfaces, execution paths, and reduction semantics |
| [molecular/PIPELINE.md](molecular/PIPELINE.md) | Reusable molecular preparation, solve/validate commands, and storage choices |
| [CONVENTIONS.md](CONVENTIONS.md) | Mathematical and representation conventions |
| [REPRODUCING.md](REPRODUCING.md) | Frozen environments, benchmark commands, and checks |
| [Architecture paper](paper_architecture/README.md) | *clifford_qc: A Python Toolkit for Quantum Simulation* ([arXiv:2609.23059](https://arxiv.org/abs/2609.23059)); features, architecture, and evidence |
| [CITATION.cff](CITATION.cff) | Software metadata and research citations |

The repository also contains the [measurement/ADAPT manuscript](paper/README.md),
the [historical A-CASE source snapshot](paper_a_case_subspaces/README.md), and
the [DA-CASE manuscript](paper_acase/README.md). Each has its own provenance and
validation boundary. Cite the method used and identify the software revision;
a paper citation does not identify a particular code snapshot. Use
[CITATION.cff](CITATION.cff) for the software citation and include the commit
used for your experiment.

Licensed under [Apache-2.0](LICENSE).

## Research status

Implementation progress is tracked in [PHASE_STATUS.json](PHASE_STATUS.json)
and [PLAN.md](PLAN.md). The generated score below measures the numbered research
programme; it is not a package-readiness or scientific-advantage score.

<!-- PHASE-STATUS-SUMMARY:START -->
| numbered phase scope | lifecycle | implementation |
|---|---|---:|
| Phases 0--14 | complete | 15 / 15 |
| Phase 15: Second-moment bank | partial | 50% |
| Phase 16: Time-evolved inputs | partial | 50% |
| Phase 17: Mapping validation and breadth | complete | 100% |
| Phase 18: Embedding boundary | complete | 100% |
| Phase 19: Anticommuting-clique partitioning | complete | 100% |

Strict complete-phase score: **18 / 20 = 90.00%**.
Progress-weighted score: **19.00 / 20 = 95.00%**.
Retired and conditional adjunct phases are tracked separately and do not change this denominator. Source: `PHASE_STATUS.json`; validate with `python benchmarks/check_phase_status.py`.
<!-- PHASE-STATUS-SUMMARY:END -->
