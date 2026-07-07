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
from clifford_qc import *
bell = evolve(ket_density(2, "00"), CNOT(2,0,1) * H(2,0))
```

or simply:

```python
bell = bell_density()
```
