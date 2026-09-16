# clifford_qc

**A Python toolkit for quantum simulation.** Build spin, lattice, and molecular
models; compare variational and subspace eigensolvers; and study measurement
and classical storage costs using a shared sparse Pauli representation.

The package is intended for researchers developing algorithms and reproducing
small-system comparisons. Its strengths are reusable operator calculations,
virtual subspace bases, shared measurement data, and explicit resource reports.

`clifford_qc` is an **alpha research package**. Interfaces can still change.
A public archival release is intended; these instructions install a source
checkout and do not assume an existing package-index release. For reproducible
work, record the source revision and dependency environment.

## What you can do

| Task | Implemented features | Start here |
|---|---|---|
| Work with operators | Sparse Pauli and fermionic operators, gates, density operators, channels, traces, and entanglement diagnostics | [Operator example](#operator-calculations), [conventions](CONVENTIONS.md) |
| Build models | Spin chains, Hubbard and other lattice models, restricted real FCIDUMP import, effective-Hamiltonian JSON, orbital rotations | [Model examples](#examples), `clifford_qc.models` |
| Run circuits | Parameterized Pauli rotors and Clifford gates, JSON serialization, gradients, OpenQASM 3 export | [Circuit example](#circuit-programs), `clifford_qc.ir` |
| Compare eigensolvers | VQE, ADAPT-VQE, fixed and adaptive operator-response subspaces, QSCI/SQD, selected-CI controls, hybrid bases | [Subspace example](#adaptive-subspaces), `algorithms/`, `subspace/` |
| Reuse measurements | QWC and block-commuting grouping, compiled Clifford readouts, shared caches, covariance, allocation, uncertainty estimates | [Finite-shot examples](#examples), [architecture](ARCHITECTURE.md#measurement-and-evidence) |
| Control classical costs | Cached preparation, optional reference validation, object or packed coefficients, streaming with recomputation | [Pipeline guide](PIPELINE.md) |
| Use other quantum software | Optional Stim, OpenFermion, pytket, PennyLane, and PyZX bridges | [Optional dependencies](#optional-dependencies) |

The core representation is a sparse operator `MV` in the Pauli-word basis of

\[
\mathrm{Cl}(2n,\mathbb C) \cong M(2^n,\mathbb C).
\]

Models and circuit programs expose `PauliSum`; conversion to `MV` preserves
word codes and operator semantics. Sparse support can grow rapidly, so this
representation is not a general speedup over matrix methods.

## Install from source

Use Python 3.10 or newer. From a checked-out repository:

```bash
python -m pip install -e .
python -m clifford_qc.verify
```

NumPy is the only core runtime dependency. The smoke check validates the
installation; it does not reproduce the full benchmark suite.

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

A-CASE (Adaptive Clifford-Algebra Subspace Eigensolver) forms virtual directions
`A_i |psi>` and solves `Hc = ESc`. Every projected matrix element is an
expectation on the reference; the basis states need not be prepared separately.
The example uses a deliberately small candidate pool and no ground-state oracle.
An exact projected solve does not certify convergence to the full ground state.

For repeated molecular experiments, [PIPELINE.md](PIPELINE.md) separates
FCIDUMP preparation, independent A-CASE solves, and optional sector validation.
Object storage with `retain_all` is the default. Packed storage and
`stream_recompute` are explicit alternatives with different memory/time costs.

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
  certify a nonlinear Ritz eigenvalue or total ground-state error. Evidence
  labels are not yet schema-validated uniformly across all records.
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
| [PIPELINE.md](PIPELINE.md) | Reusable molecular preparation, solve/validate commands, and storage choices |
| [CONVENTIONS.md](CONVENTIONS.md) | Mathematical and representation conventions |
| [MIGRATION.md](MIGRATION.md) | Legacy API mapping and adopting the preparation pipeline |
| [REPRODUCING.md](REPRODUCING.md) | Frozen environments, benchmark commands, and checks |
| [Architecture paper](paper_architecture/README.md) | *clifford_qc: A Python Toolkit for Quantum Simulation*; features, architecture, and evidence |
| [CITATION.cff](CITATION.cff) | Software metadata and research citations |

The repository also contains the [operator/ADAPT manuscript](paper/README.md),
the [historical A-CASE source snapshot](paper_a_case_subspaces/README.md), and
the [DA-CASE manuscript](paper_acase/README.md). Each has its own provenance and
validation boundary. Cite the method used and identify the software revision;
a paper citation does not identify a particular code snapshot.

Licensed under [Apache-2.0](LICENSE).

## Research status

Implementation progress is tracked in [PHASE_STATUS.json](PHASE_STATUS.json)
and [PLAN.md](PLAN.md). The generated score below measures the numbered research
programme; it is not a package-readiness or scientific-advantage score.

<!-- PHASE-STATUS-SUMMARY:START -->
| numbered phase scope | lifecycle | implementation |
|---|---|---:|
| Phases 0--14 | complete | 15 / 15 |
| Phase 15: Second-moment bank | open | 0% |
| Phase 16: Time-evolved inputs | open | 0% |
| Phase 17: Mapping validation and breadth | partial | 75% |
| Phase 18: Embedding boundary | partial | 50% |
| Phase 19: Anticommuting-clique partitioning | proposed | 0% |

Strict complete-phase score: **15 / 20 = 75.00%**.
Progress-weighted score: **16.25 / 20 = 81.25%**.
Retired and conditional adjunct phases are tracked separately and do not change this denominator. Source: `PHASE_STATUS.json`; validate with `python benchmarks/check_phase_status.py`.
<!-- PHASE-STATUS-SUMMARY:END -->
