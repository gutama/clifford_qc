Phase 1 — Core hardening
  - Move all rich verify.py invariants into pytest
  - Add hypothesis/property-based tests
  - Add CI
  - Fix pyproject metadata, author/package info, versioning
  - Keep docs focused on conventions, algebra, and invariants

Phase 2 — Native interchange layer
  - Pauli-rotor IR as primary
  - Golden conformance vectors
  - OpenFermion converters: QubitOperator/FermionOperator ↔ MV
  - QASM3 lowering as an export pass, not the native format

Phase 3 — Stim validation harness
  - Clifford-only circuit extraction
  - Validate Gottesman–Knill/blade-preservation claims at large n
  - Benchmark support preservation against Stim tableau behavior

Phase 4 — pytket bridge
  - Pauli-rotor IR → PauliExpBox
  - Round-trip checks where possible
  - Compilation/routing as credibility layer

Phase 5 — PennyLane frontend mode
  - IR → PennyLane circuit
  - Compare PennyLane parameter-shift gradients with clifford_qc adjoint gradients
  - Do not build a PennyLane device plugin yet

Later
  - Cirq as low-cost export, mostly through existing ecosystem conversion
  - QuTiP only when open-system/Lindblad work becomes a paper
  - CUDA-Q/Qulacs/Qibo only for cross-validation or HPC narrative
  
  
IR contains:
  PauliWord
  PauliSum
  Rotor
  NamedClifford
  MeasurementTask
  ParameterGroup
  Program

Stim bridge computes:
  CliffordMap

pytket bridge lowers:
  Rotor -> PauliExpBox
  NamedClifford -> native gates

PennyLane bridge lowers:
  Rotor -> PauliRot / qml.exp
  ParameterGroup -> gradient-comparison tests

QASM3 bridge lowers:
  Rotor -> basis changes + parity ladder + RZ + uncompute