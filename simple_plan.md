# `clifford_qc` roadmap

Goal: make the operator-centric Cl(2n,C) engine credible and interoperable.
The core stays dependency-light (numpy only). Everything ecosystem-facing
goes through one native interchange layer — the **Pauli-rotor IR** — and
every bridge is an optional extra that must prove itself against the exact
MV semantics in CI.

Guiding rules:

- The IR, not QASM or any vendor format, is the native representation.
  Bridges translate *from* the IR and are validated *against* the MV backend.
- A bridge ships only with conformance tests (exact unitary/expectation
  agreement, or documented tolerance); no test, no bridge.
- Optional dependencies never leak into the core: `clifford_qc` must import
  with numpy alone; bridge modules import their dependency at module scope
  and tests `importorskip` them.

## Phase 1 — Core hardening ✅

Make the invariants regression-checkable and the package publishable.

- [x] Move all rich `verify.py` invariants into pytest
      (`tests/test_invariants.py`; `python -m clifford_qc.verify` is kept as
      a dependency-light smoke entry point, pytest is the source of truth).
- [x] Property-based tests with hypothesis (`tests/test_properties.py`):
      associativity/distributivity, dagger anti-homomorphism, trace
      cyclicity, matrix-bridge homomorphism and roundtrip, label/word-code
      and blade-mask roundtrips, rotor group law, channel trace/positivity
      preservation, partial trace/transpose identities.
- [x] CI (`.github/workflows/ci.yml`): core matrix on Python 3.10–3.13
      plus a bridges job installing all extras.
- [x] pyproject metadata: real author, Apache-2.0 license, classifiers,
      URLs, version single-sourced from `clifford_qc.__version__`, extras
      per bridge (`test`, `stim`, `openfermion`, `pytket`, `pennylane`,
      `bridges`).
- [x] Keep docs focused on conventions, algebra, and invariants
      (README/CONVENTIONS unchanged in scope; new layers documented tersely).

Acceptance: `pip install -e .[test] && pytest` green with no extras;
verify entry point still passes.

## Phase 2 — Native interchange layer ✅

One IR that everything else lowers from (`clifford_qc/ir.py`):

| IR object        | Role                                                        |
|------------------|-------------------------------------------------------------|
| `PauliWord`      | single Pauli word, label/code/support/weight                |
| `PauliSum`       | complex Pauli-word sum; the observable type; `from_mv/to_mv`|
| `Rotor`          | `exp(-i θ P/2)`, θ a float or named `Parameter`             |
| `NamedClifford`  | H, S, SDG, X, Y, Z, CX, CZ, SWAP on explicit qubits         |
| `MeasurementTask`| `expectation` (PauliSum) or `sample_z` (qubit list)         |
| `ParameterGroup` | ordered named parameters; binding + gradient ordering       |
| `Program`        | n, ops, measurements, parameters; exact `unitary()`/`run()` |

- [x] IR as primary: exact lowering of every op to `MV`; JSON
      serialization (`Program.to_dict/from_dict`) as the interchange format.
- [x] Golden conformance vectors (`tests/golden/programs.json`): named
      programs with expected expectations/probabilities, re-derived through
      two independent paths (sparse MV and dense matrix) plus byte-exact
      QASM3 emission (`tests/test_golden.py`).
- [x] OpenFermion converters (`bridges/openfermion_bridge.py`):
      `QubitOperator ↔ MV/PauliSum` (lossless, cross-checked against
      `get_sparse_operator`), `FermionOperator → MV` via the package's own
      Witt operators, verified identical to OpenFermion's `jordan_wigner`.
- [x] QASM3 lowering as an export pass, not the native format
      (`clifford_qc/qasm3.py`): rotor → basis changes + CX parity ladder +
      `rz` + uncompute; shared abstract lowering is evaluated back to MV in
      tests so text and semantics cannot drift.
- [x] Exact gradients on the IR: `parameter_shift_gradient` and
      `adjoint_gradient` (forward-state/backward-observable sweep), needed
      by Phase 5.

Acceptance: golden vectors pass through both evaluation paths; QASM3
decomposition reproduces `rotor()` exactly for every letter pattern.

## Phase 3 — Stim validation harness ✅

Validate Clifford claims at scales the dense backend cannot reach
(`bridges/stim_bridge.py`):

- [x] Clifford-only circuit extraction: `Program → stim.Circuit`
      (rejects non-Clifford rotors), `stim.Tableau` construction.
- [x] `CliffordMap`: conjugation action `P → U P U†` with phase, via stim;
      cross-validated against exact MV conjugation
      (`conjugate_pauli_word`) on random circuits at n ≤ 4 and on X/Z
      generator images.
- [x] Gottesman–Knill / blade-preservation at large n: random depth-400
      circuits at n = 100 — every Pauli word maps to exactly one Pauli word
      with phase in {±1, ±i}; composition (group property) checked at n = 60.

Acceptance: stim and MV agree word-for-word and phase-for-phase wherever
both can compute; stim-only checks pass at n ≥ 100.

## Phase 4 — pytket bridge ✅

Compilation/routing ecosystem as a credibility layer
(`bridges/pytket_bridge.py`):

- [x] `Rotor → PauliExpBox` (tket half-turns: `t = θ/π`),
      `NamedClifford →` native `OpType`s.
- [x] Round-trip where possible: `tket_to_program` inverts circuits built
      from supported gates and PauliExpBoxes; unsupported ops raise.
- [x] Credibility check: `DecomposeBoxes` + `FullPeepholeOptimise`
      compiled circuits still match the MV unitary (up to global phase).

Acceptance: `circuit.get_unitary()` equals `to_matrix(program.unitary())`
up to global phase for rotors, Cliffords, and compiled circuits.

## Phase 5 — PennyLane frontend mode ✅

Gradient cross-validation, **no device plugin yet**
(`bridges/pennylane_bridge.py`):

- [x] IR → PennyLane: `Rotor → qml.PauliRot` (same `exp(-iθP/2)`
      convention), Cliffords → native ops, `PauliSum → qml.Hamiltonian`;
      `make_qnode(program, observable)`.
- [x] `ParameterGroup` drives the comparison: PennyLane parameter-shift
      gradients match `clifford_qc`'s exact adjoint and parameter-shift
      gradients (atol 1e-8), including rotors sharing one parameter.
- [ ] Device plugin: deliberately out of scope until there is a use case.

Acceptance: expectations and gradients agree between PennyLane and the MV
engine on parameterized ansatz circuits.

## Later (unchanged, still deliberately deferred)

- Cirq as low-cost export, mostly through existing ecosystem conversion
  (QASM3 output already covers most of it).
- QuTiP only when open-system/Lindblad work becomes a paper.
- CUDA-Q / Qulacs / Qibo only for cross-validation or an HPC narrative.
- `FermionOperator ← MV` (inverse JW extraction) if a workflow needs it.
- Symbolic (unbound) parameters in the pytket bridge.

## Status summary

Phases 1–5 are implemented and tested. Run everything with:

```bash
pip install -e .[test,bridges]
pytest
```

Without the bridge extras, bridge tests skip and the core suite still
covers Phases 1–2 (IR, golden vectors, QASM3 export).
