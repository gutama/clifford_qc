# Migration from `ga_qc_operator_improved.py`

The old single-file API maps directly onto `clifford_qc`:

| Old | New |
|---|---|
| `MV` | `from clifford_qc import MV` |
| `I, X, Y, Z` | same |
| `pauli_string(n, s)` | same, or `P(s)` if `n=len(s)` |
| `gamma` | same |
| `c_op, cdag_op` | same |
| `H_gate, S_gate, T_gate` | same aliases; preferred names `H, S, T` |
| `rotor, RX, RY, RZ` | same |
| `CNOT, CZ, SWAP, TOFFOLI` | same |
| `ket_density, evolve, expectation, purity` | same |
| `partial_trace, partial_transpose` | same |
| `depolarizing, dephasing, amplitude_damping` | same |
| `to_matrix, from_matrix` | same |
| `negativity, vn_entropy` | same |
| `run_verification()` | `python -m clifford_qc.verify` |

## Example

Old:

```python
from ga_qc_operator_improved import *
bell = evolve(ket_density(2, "00"), CNOT(2,0,1) * H_gate(2,0))
```

New:

```python
from clifford_qc import CNOT, H, evolve, ket_density

bell = evolve(ket_density(2, "00"), CNOT(2, 0, 1) * H(2, 0))
```

or simply:

```python
from clifford_qc import bell_density

bell = bell_density()
```

## Moving from operators to programs

Existing `MV` calculations remain supported. Model Hamiltonians and program
observables use `PauliSum`; call `to_mv()` when passing one to an operator-level
API. This is a container conversion in the same Pauli basis. `Program` adds
parameter binding, measurement tasks, serialization, and export without
requiring a rewrite of operator-only code. See the [README](README.md#circuit-programs).

## Reusing molecular preparation

For new repeated FCIDUMP experiments, the [pipeline guide](PIPELINE.md) separates
model preparation, A-CASE solving, and optional sector validation. Historical
benchmark scripts remain the entry points for reproducing their own records.
The new CLI is not a replacement for every benchmark or Python solver API.

Object storage and `retain_all` remain the defaults. Select `packed` storage
and/or `stream_recompute` explicitly when studying storage tradeoffs. Streaming
can rebuild evicted rows, and its persistent coefficient-row bound does not
bound process memory. Do not compare its live counts with cumulative counts
from an older record as if they measured the same population.

Preparation performs no ground-state reference solve. The solve record reports
exact projected arithmetic, not certified full-ground-state convergence.
Request reference validation separately when needed and account for its cost.
No automatic migration of historical benchmark records is performed.
