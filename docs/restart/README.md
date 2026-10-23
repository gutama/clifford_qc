# Current restart roadmap

Start with the [unchanged 24 October blueprint](blueprint_2026-10-24.md),
[frozen v1 evidence bindings](v1_evidence_snapshot.json),
[first implementation contract](CONTRACT.md), and
[checked compatibility scope](COMPATIBILITY.md).

| Step | Status | Deliverable |
| --- | --- | --- |
| 1 | Implemented | Frozen evidence identities and one determinant molecular problem/claim contract |
| 2 | Implemented | Reference capabilities, determinant/MV adapters, exact provider, density-free determinant pipeline |
| 3 | Planned | Sector consumer, bounded action buffers, explicit owned/RSS budgets, root recovery, spin and action-residual evidence |
| 4 | Planned | Wider old-suite reproduction through the supported provider path; first-PR H4/LiH tests are the bounded starting checks |
| 5 | Planned | Sampled consumer and a preregistered independent pilot/evaluation ratio experiment |
| 6 | Conditional | Promote only benefits that meet the new frozen comparison, or close its hypothesis |

## Current capabilities

| Capability | Implemented? | Validation scope | Needs an exact oracle to operate? | Demonstrated advantage |
| --- | --- | --- | --- | --- |
| Determinant expectations | Yes | Bit patterns, complex rows, compatibility and large-register bounded rows | No density or spectral oracle | Removes explicit reference-density expansion by construction; no timing/RSS campaign |
| Exact projected provider | Yes | Complex 3-qubit controls; object/packed and retain/stream policies | Exact reference-evaluation capability; no spectrum oracle | Numerical interface and immutable cache inputs; no algorithmic advantage claim |
| Prepared determinant workflow | Yes | H4/LiH S/H, energies and selection/stopping compatibility | No ground-state oracle under default config | Reference representation changed; variational span unchanged |
| MV compatibility adapter | Yes | Owned immutable snapshot and legacy trace comparisons | Exact density supplied explicitly | Compatibility only |
| Sector/action and sampled boundaries | Protocols only | Exact capability rejects sampled-only references | Consumer-specific | Concrete adapters and experiments deferred |
| Ground-root recovery and measured ratio estimation | No new consumer yet | Existing historical evidence retained | Must be assessed under subsequent contracts | None claimed |

The supported first workflow is documented in the [molecular pipeline guide](../../molecular/PIPELINE.md).
Historical phase status remains in [PHASE_STATUS.json](../../PHASE_STATUS.json)
and [PLAN.md](../../PLAN.md). In particular, this interface work does not reopen
the packet line or change the failed frame-coordinate shortlist criterion.

Use the numerical API directly when no adaptive selector is needed:

```python
from clifford_qc.multivector import MV
from clifford_qc.reference_states import DeterminantReference
from clifford_qc.subspace.generators import identity_generator
from clifford_qc.subspace.providers import ExactProjectedProvider

reference = DeterminantReference("01", arithmetic="legacy_trace")
hamiltonian = MV.from_terms(2, {"ZI": 0.5, "IZ": -0.25, "XX": 0.1})
provider = ExactProjectedProvider(reference, hamiltonian, [identity_generator(2)])
result = provider.solve()
print(result.ground_energy, result.resources["reference_evaluation"])
```

This one-direction example evaluates a projected energy. It does not certify a
ground root or demonstrate a quantum resource benefit.
