# `clifford_qc` architecture

The current shape of the package, derived from the source on this branch
rather than from intent: 92 modules and ~22.5k lines under `clifford_qc/`,
97 test modules and ~21.9k lines under `tests/`, 41 record producers and 27
record gates under `benchmarks/`.

Two things explain most of the layout.

1. **One representation all the way down.** States, gates, observables,
   channels, Jordan-Wigner Clifford generators, and CAR operators are all the
   same sparse Pauli-word object `MV`. There is no conversion boundary between
   "circuit land" and "operator land", so the layers above are free to move
   between them.
2. **Every number carries what kind of number it is.** `EvidenceLevel`
   (`exact` / `asymptotic` / `finite_sample` / `heuristic` / `none`) threads
   from the measurement layer through the solvers into the committed records,
   and the benchmark gates compare against it. Much of the layering exists to
   keep an oracle-fed quantity from being reported beside a measured one
   without a label.

---

## 1. Layer stack

Arrows are runtime `import` dependencies, and they only point downward.

```mermaid
flowchart TD
    subgraph L6["Evidence & reproduction"]
        BENCH["benchmarks/<br/>41 run_*.py producers<br/>27 check_*.py gates"]
        REPRO["reproducibility.py<br/>record_environment.py<br/>verify.py"]
    end

    subgraph L5["Solvers"]
        SUBSPACE["subspace/ &mdash; 27 modules, 8.9k lines<br/>A-CASE, QSCI/SQD, selected-CI controls,<br/>hybrid, response, certification"]
        ALGOS["algorithms/ &mdash; 8 modules, 1.4k lines<br/>VQE, ADAPT-VQE, pools, initialization"]
        WORKFLOWS["workflows.py<br/>cross-layer orchestration"]
    end

    subgraph L4["Problem definition"]
        MODELS["models/ &mdash; 8 modules, 2.3k lines<br/>spin, lattice, FCIDUMP,<br/>effective Hamiltonian, orbital bases"]
    end

    subgraph L3["Execution & measurement"]
        BACKENDS["backends/ &mdash; 7 modules, 1.1k lines<br/>exact MV, dense statevector,<br/>sector statevector, finite shot, stabilizer"]
        MEAS["measurement/ &mdash; 13 modules, 3.9k lines<br/>grouping, caches, allocation,<br/>confidence, block-commuting plans, cost"]
    end

    subgraph L2["Program IR"]
        IR["ir.py &mdash; PauliWord, PauliSum, Program,<br/>Rotor, NamedClifford, gradients"]
        QASM["qasm3.py &mdash; export pass"]
        FMAP["fermion_mapping.py"]
    end

    subgraph L1["Algebra kernel"]
        MV["multivector.py &mdash; MV"]
        KERNEL["pauli_kernel.py &mdash; packed word arithmetic"]
        PAULI["pauli.py, clifford.py, gates.py,<br/>states.py, fermion.py, channels.py"]
        DENSE["dense_reference.py, sparse.py,<br/>pauli_action.py, matrix.py"]
    end

    BRIDGES["bridges/ &mdash; 6 modules<br/>stim, OpenFermion, pytket,<br/>PennyLane, PyZX<br/>(optional extras)"]

    BENCH --> SUBSPACE
    BENCH --> ALGOS
    BENCH --> MODELS
    BENCH --> MEAS
    REPRO --> BENCH
    SUBSPACE --> MEAS
    SUBSPACE --> BACKENDS
    SUBSPACE --> MODELS
    ALGOS --> MEAS
    ALGOS --> BACKENDS
    WORKFLOWS --> ALGOS
    WORKFLOWS --> BACKENDS
    MODELS --> IR
    MODELS --> PAULI
    BACKENDS --> IR
    MEAS --> IR
    IR --> MV
    QASM --> IR
    FMAP --> IR
    PAULI --> MV
    DENSE --> MV
    MV --> KERNEL
    IR -.-> BRIDGES
    MEAS -.-> BRIDGES
```

Dotted edges are optional: `stim` is imported only when
`compile_block_measurement_plan` or the stabilizer backend is called, never at
package import.

### Layer inventory

| Layer | Location | Lines | Owns |
|---|---|---:|---|
| Algebra kernel | `clifford_qc/*.py` | 4,214 | `MV`, packed word codes, the three pairings, JW generators, CAR operators, dense/sparse references |
| Program IR | `ir.py`, `qasm3.py`, `fermion_mapping.py` | (in the above) | Pauli-rotor programs, parameters, gradients, QASM3 export, encoding choice |
| Execution | `backends/` | 1,058 | The `Backend` / `SamplingBackend` protocol and its five implementations |
| Measurement | `measurement/` | 3,924 | QWC and block-commuting grouping, shot allocation, cumulative caches, confidence bounds, device cost |
| Models | `models/` | 2,338 | Hamiltonians and observables from lattices, FCIDUMP records, Wannier/embedding JSON, orbital bases |
| Algorithms | `algorithms/` | 1,428 | Fixed-depth VQE, ADAPT-VQE with certified selection, pools, Clifford-point seeding |
| Subspace solvers | `subspace/` | 8,866 | A-CASE, QSCI/SQD, classical selected-CI controls, hybrid arms, response spectra, growth certificates |
| Evidence | `benchmarks/`, `reproducibility.py` | — | Record producers, drift gates, environment provenance |

---

## 2. Solver stack

What the layers above `models/` actually compute, and how the arms relate.
This is the diagram that matters for reading results: the three solver arms
are deliberately comparable, and the bottom row is what they are scored
against.

```mermaid
flowchart TD
    SRC["DFT / Wannier downfolding / FCIDUMP records<br/>lattice and spin model builders"]
    MODELS["models/ &mdash; one Model:<br/>PauliSum Hamiltonian, Clifford reference<br/>program, named HVA layers"]
    ORB["models/orbital.py<br/>single-particle basis is a named argument,<br/>not an unstated default"]

    ACASE["<b>A-CASE</b> &mdash; subspace/adaptive.py<br/>Rayleigh-Ritz in an operator-response<br/>subspace; the basis states A_i&middot;psi are never<br/>prepared, every element is an expectation<br/>on one reference state"]
    QSCI["<b>QSCI / SQD</b> &mdash; subspace/qsci.py<br/>sampled determinant subspace;<br/>the projected matrix is built classically,<br/>so W = 0 and cost is reported elsewhere"]
    ADAPT["<b>ADAPT-VQE</b> &mdash; algorithms/adapt.py<br/>confidence-certified selection on the<br/>odd-Y pool from shared word caches"]

    CTRL["subspace/selected_ci.py<br/>classical controls: excitation closure,<br/>budget-matched selected CI,<br/>span/principal-angle containment"]
    HYBRID["subspace/hybrid.py<br/>sampled determinants dressed by<br/>operator-response directions"]

    ORACLE["Oracles &amp; validation<br/>backends/sector_statevector.py, sparse.py,<br/>subspace/reference.py, dense_reference.py"]
    LADDER["benchmarks/run_acase_ladder.py<br/>benchmarks/run_phase12_paper_b.py<br/>one schema, Pareto fronts computed only<br/>inside a single evidence category"]

    SRC --> MODELS
    ORB --> MODELS
    MODELS --> ACASE
    MODELS --> QSCI
    MODELS --> ADAPT
    ACASE --> HYBRID
    QSCI --> HYBRID
    QSCI --> CTRL
    ADAPT -->|"warm start via workflows.adapt_warm_start"| ACASE
    ACASE --> LADDER
    QSCI --> LADDER
    ADAPT --> LADDER
    HYBRID --> LADDER
    CTRL --> LADDER
    ORACLE --> LADDER
```

---

## 3. Finite-shot data path

The measurement layer is the part with the most structure, because it is where
a number acquires its evidence label. One pass, from a Hamiltonian to a
labelled interval:

```mermaid
flowchart LR
    WORDS["word universe<br/>PauliSum / bank rows"]
    GROUP["grouping.py &mdash; QWC cover<br/>block_commuting.py &mdash; block size k<br/>k=1 QWC &hellip; k=n fully commuting"]
    PLAN["planning.py<br/>CompiledMeasurementPlan:<br/>frozen groups, stim-synthesized<br/>Clifford settings, signed Z-parity<br/>readouts, resource ledger"]
    ALLOC["allocation.py<br/>UniformFixed / UniformDoubling /<br/>VarianceProportional / GroupVarianceOptimal"]
    BACK["backends/finite_shot.py<br/>SamplingBackend"]
    SAMP["GroupSample<br/>joint outcome histogram<br/>per setting"]
    CACHE["cache.py &mdash; GroupedWordCache<br/>cumulative across rounds;<br/>a word shared by many observables<br/>is paid for once"]
    FUNC["functionals.py &mdash; WordFunctional<br/>linear read of cached words"]
    CONF["confidence.py<br/>Sidak / Bonferroni, Jeffreys,<br/>empirical Bernstein"]
    OUT["labelled interval<br/>EvidenceLevel: exact, asymptotic,<br/>finite_sample, heuristic"]

    WORDS --> GROUP --> PLAN --> BACK
    ALLOC --> BACK
    BACK --> SAMP --> CACHE --> FUNC --> CONF --> OUT
    CACHE -.->|"covariance-aware variance"| ALLOC
    COST["cost.py &mdash; DeviceCard<br/>duration, fidelity, break-even surface"]
    PLAN --> COST
```

The `GroupSample` contract carries both forms in one type: a QWC setting
identified by its per-qubit basis, or a Clifford-diagonalized setting with an
explicit `setting_key` and signed `readouts` map. That is what lets the
block-commuting hierarchy feed the same estimators without pretending it was
measured qubit-wise.

---

## 4. The seams

These are the interfaces that make the layers replaceable. Changing one is a
cross-cutting change; changing anything else is local.

| Seam | Defined in | Contract |
|---|---|---|
| `MV` | `multivector.py` | Sparse Pauli-word operator. Two bits per qubit (`0=I, 1=X, 2=Y, 3=Z`); multiplication is XOR/AND/popcount mod 4. Three distinct pairings — `scalar_product`, `hs_product`, `trace_pairing` — that are **not** interchangeable |
| `Program` / `PauliSum` | `ir.py` | The interchange layer. Named Cliffords, Pauli rotors `exp(-i·theta·P/2)`, ordered parameters, measurement tasks. Everything lowers exactly to `MV` |
| `Backend`, `SamplingBackend` | `backends/protocol.py` | The only way the algorithm layer talks to execution. Exact evaluation, plus finite-shot sampling returning `MeasurementBatch` |
| `GroupSample` | `backends/protocol.py` | One commuting setting's joint histogram — QWC basis or compiled Clifford readout |
| `Model` | `models/spin.py` | The `PauliSum` Hamiltonian, a Clifford `Program` preparing the reference product state, and named HVA layers. Every builder in `models/` emits this same frozen type |
| `Generator` | `subspace/generator_core.py` | One labelled basis direction `A_i` as an `MV`, whose state stays virtual; `support()` is the resource term `S_A` |
| `MatrixElementBank` | `subspace/projection.py` | Cached projected matrix elements and their word universe |
| `EvidenceLevel`, `SelectionStatus` | `selection.py` | The evidence contract, plus `canonical_argmax` — deterministic tie resolution shared by every selector |
| `CompiledMeasurementPlan` | `measurement/planning.py` | The public block-commuting boundary; freezes a word universe into groups, executable settings, and a matched resource ledger |
| `Restriction` | `subspace/restriction.py` | Symmetry-sector restriction of operators and states |
| `as_multivector`, `check_reference` | `subspace/contracts.py` | Representation coercion and reference-state validation |

---

## 5. Dependency rules and their recorded exceptions

The runtime import graph is acyclic across layers. The upward references that
exist are deliberate and are all deferred:

- **`measurement/` never imports `subspace/` at runtime.** `functionals.py` and
  `session.py` reference `MatrixElementBank` and `SubspaceResult` under
  `TYPE_CHECKING` only, and pull `subspace.linalg` inside the function bodies
  that need it. `session.py` goes further and *restates* the canonical
  `linalg` tolerances with a comment saying why, rather than importing the
  higher layer during its own initialization.
- **`subspace/projection.py` pulls `measurement.grouping` inside a method**, so
  costing a word universe does not make the solver depend on the measurement
  package at import.
- **`workflows.py` exists precisely to keep `algorithms/` and `subspace/`
  apart.** It imports `algorithms.adapt` at call time so importing either
  numerical package does not eagerly initialize the other.
- **`fermion.py` and `pauli_action.py` reach into
  `backends/sector_statevector.py` from function scope** for sector helpers.
- **`multivector.py` pulls `dense_reference.to_matrix` inside a method**, so the
  kernel does not depend on the dense bridge at import.
- **Optional dependencies are kept out of every package `__init__`**, by one of
  two mechanisms. Either the import is function-scope — `stim` in
  `measurement/planning.py`, `subspace/contextual.py`, and
  `fermion_mapping.py` — or the module imports it at module level and is
  deliberately *not* re-exported, so reaching it requires naming it:
  `measurement/block_synthesis.py`, `backends/stabilizer.py`, and everything
  in `bridges/`. `models/chemistry.py` (PySCF + OpenFermion) uses the same
  non-re-export rule, so a bare install can still do
  `from clifford_qc.models import hubbard`.

The core package depends on `numpy` alone. `scipy` is the `research` extra;
everything else is per-bridge.

---

## 6. Evidence and reproduction architecture

`benchmarks/` is not a scratch directory — it is a two-part contract, and the
pairing is the architecture:

```mermaid
flowchart LR
    CFG["benchmarks/configs/*.json<br/>predeclared parameters,<br/>device cards"]
    RUN["benchmarks/run_*.py<br/>41 producers"]
    REC["benchmarks/reference_results/<br/>45 committed records<br/>+ execution provenance"]
    CHK["benchmarks/check_*.py<br/>27 gates"]
    SUM["benchmarks/summarize*.py<br/>CSV + Markdown"]
    PAPER["paper/, paper_acase/,<br/>paper_a_case_subspaces/<br/>make_tables.py, make_figures.py"]
    CI["CI &mdash; ruff, pytest,<br/>named per-gate steps,<br/>OMP_NUM_THREADS=1"]

    CFG --> RUN --> REC
    REC --> CHK
    REC --> SUM --> PAPER
    CHK --> CI
    DOCS["check_docs.py<br/>REPRODUCING.md must still describe<br/>the code it claims to reproduce"] --> CI
    ENV["record_environment.py<br/>reproducibility.execution_provenance"] --> REC
```

Three properties of this layer are load-bearing:

- **Producer/gate pairing.** A `run_X.py` writes a record; a `check_X.py`
  re-derives it and fails on drift. CI runs each gate as its own named step
  with `if: !cancelled()`, so a drift suite reports every symptom rather than
  stopping at the first.
- **Thread pinning.** CI sets `OMP_NUM_THREADS=1` because threaded BLAS
  reductions sum in a thread-count-dependent order — `eigvalsh` returns four
  different last bits across `OMP_NUM_THREADS` 1..4. This closes the one source
  of nondeterminism a workflow can close; it does not make records portable
  across CPU microarchitectures, and `REPRODUCING.md` records the floor that
  leaves.
- **Documentation is gated too.** `check_docs.py` verifies that every command
  in `REPRODUCING.md` names a script that exists, that every long flag is one
  the script accepts, that predeclared constants match the source, that every
  producer is documented, that every committed record is named, and that the
  quoted `pytest` count matches what the suite collects.

---

## 7. Reading order

For someone new to the codebase:

1. `CONVENTIONS.md` — word codes, qubit ordering, the three pairings, rotor
   convention. Nothing below makes sense without it.
2. `multivector.py` then `pauli_kernel.py` — the whole engine is here.
3. `ir.py` — the interchange layer everything above speaks.
4. `backends/protocol.py` — the execution seam, and the `GroupSample` docstring
   for why grouped measurement is shaped the way it is.
5. `subspace/adaptive.py` and `subspace/projection.py` — A-CASE proper.
6. `selection.py` — short, and it explains the evidence vocabulary that the
   records, the ladder, and the manuscripts all use.
